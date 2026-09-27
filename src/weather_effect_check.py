"""Is a temperature effect on marks visible in the data at all, before any model?

Weather was built and measured out as model features on 2026-08-25: temp_at_sb,
mean_race_temp and mean_humidity all scored negative over 10 paired seeds, and
each landed within noise of a shuffled permutation of itself. HANDOFF left one
framing untried, and this script tests it: heat helps sprints and hurts the
5000m, so one pooled feature with one sign cannot express the effect, but an
adjustment applied per discipline could.

This is the cheap half of that idea. If the effect is not visible in the marks
themselves it cannot help a model, and the experiment stops here.

Method, and why:

  The naive regression of mark quality on temperature is confounded, because
  the best athletes choose the best meetings and the best meetings are not
  randomly distributed across the weather. So this is a WITHIN-ATHLETE
  estimate: every athlete's score and temperature are centred on that
  athlete's own mean within a discipline, which removes athlete quality
  entirely, and the slope is fitted on what is left.

  Scores are World Athletics' own Results Score, already comparable across
  disciplines and already the right direction (higher is better), so a time
  event and a throw can be read on one scale without inventing one.

  Each discipline gets a shuffled control: the same rows with temperatures
  permuted inside that discipline. A real effect has to beat what noise
  produces on the same data, which is the standard this project has applied
  to every feature since the seed-noise finding.

Usage:
    python src/weather_effect_check.py
"""
import glob
import json
import os

import numpy as np
import pandas as pd

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
WEATHER_PATH = os.path.join(BASE_DIR, "data", "venue_weather.csv")
OUT_PATH = os.path.join(BASE_DIR, "outputs", "weather_effect_by_discipline.json")

# An athlete needs at least this many marks in a discipline to say anything
# about how that athlete varies with temperature.
MIN_MARKS_PER_ATHLETE = 3
# And a discipline needs this many usable rows before a slope means anything.
MIN_ROWS = 200
SHUFFLES = 20
SEED = 42


def load():
    """Every scraped mark that carries a World Athletics score and a real
    temperature for the day it was set."""
    weather = pd.read_csv(WEATHER_PATH).dropna(subset=["temp_mean_c"])
    weather["date"] = weather["date"].astype(str)

    frames = []
    for path in glob.glob(os.path.join(RAW_DIR, "*.csv")):
        try:
            frame = pd.read_csv(path, low_memory=False)
        except Exception:
            continue
        if not {"Venue", "Date", "Results Score", "discipline", "Competitor"} <= set(frame.columns):
            continue
        frames.append(frame)
    marks = pd.concat(frames, ignore_index=True)

    # The cache keys dates as ISO; the scraped files carry "16 MAY 2026".
    marks["date"] = pd.to_datetime(marks["Date"], format="%d %b %Y", errors="coerce")
    marks = marks.dropna(subset=["date"])
    marks["date"] = marks["date"].dt.strftime("%Y-%m-%d")

    joined = marks.merge(weather, left_on=["Venue", "date"], right_on=["venue", "date"])
    joined = joined.dropna(subset=["Results Score", "temp_mean_c", "Competitor", "discipline"])
    joined["score"] = pd.to_numeric(joined["Results Score"], errors="coerce")
    joined["doy"] = pd.to_datetime(joined["date"]).dt.dayofyear
    return joined.dropna(subset=["score"])


def within_athlete_slope(group, control_season=False):
    """Points of World Athletics score per degree, with each athlete centred on
    their own mean. Returns None when nothing is left to fit.

    With control_season, day-of-year and its square are centred the same way and
    fitted alongside, so the temperature term is what survives after an
    athlete's own shape across the season is taken out. This matters more than
    it sounds: the hot days ARE the peak-season meetings, so an uncontrolled
    slope can read the calendar and call it weather."""
    counts = group.groupby("Competitor")["score"].transform("size")
    kept = group[counts >= MIN_MARKS_PER_ATHLETE]
    if len(kept) < MIN_ROWS:
        return None

    def centred(col):
        return (col - col.groupby(kept["Competitor"]).transform("mean")).to_numpy(float)

    y = centred(kept["score"])
    cols = [centred(kept["temp_mean_c"])]
    if control_season:
        doy = kept["doy"].astype(float)
        cols.append(centred(doy))
        cols.append(centred(doy ** 2))
    x = np.column_stack(cols)
    if not np.isfinite(x).all() or float((x[:, 0] ** 2).sum()) <= 0:
        return None

    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    slope = float(coef[0])
    resid = y - x @ coef
    dof = len(kept) - kept["Competitor"].nunique() - x.shape[1]
    if dof <= 0:
        return None
    sigma2 = float((resid ** 2).sum() / dof)
    try:
        cov = sigma2 * np.linalg.inv(x.T @ x)
        se = float(np.sqrt(cov[0, 0]))
    except np.linalg.LinAlgError:
        se = float("nan")
    return {"slope": slope, "se": se, "rows": int(len(kept)),
            "athletes": int(kept["Competitor"].nunique())}


def main():
    data = load()
    rng = np.random.default_rng(SEED)
    out = []
    for disc, group in data.groupby("discipline"):
        real = within_athlete_slope(group)
        if real is None:
            continue
        controlled = within_athlete_slope(group, control_season=True)
        # The same rows, temperatures shuffled inside this discipline.
        null = []
        for _ in range(SHUFFLES):
            shuffled = group.copy()
            shuffled["temp_mean_c"] = rng.permutation(shuffled["temp_mean_c"].values)
            got = within_athlete_slope(shuffled)
            if got:
                null.append(got["slope"])
        null_sd = float(np.std(null)) if null else float("nan")
        out.append({
            "discipline": disc,
            "pointsPerDegree": round(real["slope"], 4),
            "se": round(real["se"], 4),
            "tStat": round(real["slope"] / real["se"], 2) if real["se"] else None,
            "rows": real["rows"],
            "athletes": real["athletes"],
            "pointsPerDegreeSeasonControlled": round(controlled["slope"], 4) if controlled else None,
            "tStatSeasonControlled": (round(controlled["slope"] / controlled["se"], 2)
                                      if controlled and controlled["se"] else None),
            "shuffledSd": round(null_sd, 4),
            # The honest test: does the real slope stand outside what shuffling
            # the same temperatures produces?
            "beatsShuffled": bool(abs(real["slope"]) > 2 * null_sd) if null_sd == null_sd else None,
        })

    out.sort(key=lambda r: -abs(r["pointsPerDegree"]))

    # The decisive check, and the cheapest. Air is air: whatever heat does to a
    # 5000m it must do to the women's 5000m too. If the men's and women's
    # versions of the same event disagree on the SIGN, the slope is reading
    # something other than the weather.
    by = {r["discipline"]: r for r in out}
    pairs = []
    for key, row in by.items():
        if not key.startswith("men_"):
            continue
        other = by.get("women_" + key[4:])
        if not other:
            continue
        a = row["pointsPerDegreeSeasonControlled"]
        b = other["pointsPerDegreeSeasonControlled"]
        if a is None or b is None:
            continue
        pairs.append({"event": key[4:], "men": a, "women": b, "sameSign": (a > 0) == (b > 0)})
    agree = sum(p["sameSign"] for p in pairs)
    report = {
        "what": "Within-athlete effect of temperature on World Athletics score, per "
                "discipline. No model involved. Run before deciding whether a per-discipline "
                "weather adjustment is worth building.",
        "method": {
            "estimate": "athlete-centred least squares, points of WA score per degree C",
            "minMarksPerAthlete": MIN_MARKS_PER_ATHLETE,
            "minRowsPerDiscipline": MIN_ROWS,
            "control": f"{SHUFFLES} permutations of temperature within each discipline",
            "seed": SEED,
        },
        "rows": len(data),
        "disciplines": len(out),
        "pairedSignCheck": {
            "what": "Men's against women's version of the same event, season-controlled. "
                    "A real temperature effect has to share a sign across the pair.",
            "pairs": pairs,
            "agreeing": agree,
            "of": len(pairs),
            "chance": "a coin flip gives half",
        },
        "results": out,
    }
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)

    print(f"  {len(data)} marks with a score and a temperature, {len(out)} disciplines\n")
    print(f"  {'discipline':<22} {'raw':>8} {'t':>6} {'season-ctrl':>12} {'t':>6} {'rows':>7}")
    for r in out:
        c = r["pointsPerDegreeSeasonControlled"]
        print(f"  {r['discipline']:<22} {r['pointsPerDegree']:>8.3f} {r['tStat'] or 0:>6.1f} "
              f"{(c if c is not None else 0):>12.3f} {(r['tStatSeasonControlled'] or 0):>6.1f} {r['rows']:>7}")
    beat = [r for r in out if r["beatsShuffled"]]
    print(f"\n  {len(beat)} of {len(out)} disciplines stand outside their own shuffled control")
    surv = [r for r in out
            if r["tStatSeasonControlled"] and abs(r["tStatSeasonControlled"]) >= 2]
    print(f"  {len(surv)} of {len(out)} still reach |t| >= 2 once season shape is "
          f"controlled; chance alone gives about {len(out) * 0.05:.1f}")
    print(f"  men's and women's versions agree on sign in {agree} of {len(pairs)} "
          f"paired events; a coin flip gives {len(pairs) / 2:.0f}")
    print(f"  -> {OUT_PATH}")


if __name__ == "__main__":
    main()
