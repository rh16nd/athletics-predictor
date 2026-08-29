"""
athlete_analytics.py -- analyst-grade per-athlete statistics computed from
the full race log, not from a season-best row.

WHY THIS IS POSSIBLE NOW
------------------------
Until src/worldwide_scraper.py ran, the site's view of an athlete was the
Diamond League circuit only: the median athlete had 2 races on record for a
whole season. You cannot compute a win rate, a consistency figure or a
season shape from 2 races -- that is the same data-density wall the sixth
session's `final_top3_rate` experiment died on. With the worldwide log the
median is 5 and climbing, and 95% of those rows carry a real finishing
POSITION, which is what unlocks everything below.

WHAT IS DELIBERATELY NOT HERE
-----------------------------
Nothing in this module feeds the model. It reads the quarantined
data/worldwide/ log plus the DL-circuit files, and is imported only by
api.py's profile endpoint. Adding any of these as a training feature is a
separate decision with its own backtest -- see HANDOFF's Failed Attempts
for what happened the last three times a plausible-looking one was added.

ON RANKING COMPETITIONS
-----------------------
World Athletics' own `rankingCategory` is reported, never re-ranked into a
single quality score. The categories were read off real meeting names
rather than assumed: "A" is Continental Tour Gold (Paavo Nurmi Games, FBK
Games, Kip Keino Classic), "GL" is a continental championship (European,
African, Asian, NACAC), "B" covers Silver and the big national
championships (USA, Japanese, Chinese), "C" Bronze, "D" Challenger. A
strict ordering across those is a judgement call this module refuses to
make -- an African Championships and a Continental Tour Gold are not
comparable on one axis. The breakdown is shown per category instead, so a
reader can see for themselves that an athlete wins at Challenger level and
places sixth at Gold.
"""
import os
import re

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
WORLDWIDE_DIR = os.path.join(BASE_DIR, "data", "worldwide")

MEETS_YEAR = 2026

# Verified against real meeting names in the scraped log, not assumed.
TIER_LABELS = {
    "GL": "Continental championship",
    "A":  "Continental Tour Gold",
    "B":  "Silver / national championship",
    "C":  "Continental Tour Bronze",
    "D":  "Challenger",
    "E":  "Regional games",
    "F":  "National (smaller federations)",
    "DL": "Diamond League",
}

# Display order only. NOT a quality ranking -- see the module docstring.
TIER_ORDER = ["DL", "GL", "A", "B", "C", "D", "F", "E"]

# "Top tier" for the single summary figure, stated explicitly so the number
# is checkable: the Diamond League, continental championships, and
# Continental Tour Gold.
TOP_TIERS = {"DL", "GL", "A"}

# "1h1"/"3sf2" are heat and semi-final placings; "2f1" and "4." are finals.
# The worldwide scraper and the DL per-meeting scrapers only ever read races
# WA labels "Final", so their rows are finals by construction -- this only
# matters for the historical toplist file, which mixes rounds in.
_NON_FINAL_ROUND = re.compile(r"\d\s*(h|sf|qf|q|r)\s*\d*$", re.I)


def parse_position(value):
    """(place, is_final) from the three encodings this data uses.

    The same finish is written "1" in the historical toplist file, "1." by
    the worldwide scraper and "1.0" once pandas has read a float column, so
    a naive int() works on none of them reliably."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None, True
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return None, True
    if _NON_FINAL_ROUND.search(text):
        return None, False
    match = re.match(r"^(\d+)", text)
    if not match:
        return None, True
    return int(match.group(1)), True


def parse_mark_value(mark):
    """Seconds or metres. None for the non-marks real results are full of
    (NM/NH/DNS/DNF/DQ) -- those are outcomes, not bad rows."""
    text = str(mark).strip()
    if not text or text.lower() in {"nan", "none"}:
        return None
    try:
        if text.endswith("m"):
            return float(text[:-1])
        if ":" in text:
            parts = text.split(":")
            return float(parts[0]) * 60 + float(parts[1])
        return float(text)
    except (ValueError, TypeError):
        return None


def _normalise(df, source, tier=None):
    """Every source into one shape: place, value, date, tier, meeting."""
    if df.empty:
        return pd.DataFrame()
    out = pd.DataFrame()
    out["Competitor"] = df["Competitor"]
    out["Mark"] = df["Mark"]
    out["value"] = df["Mark"].map(parse_mark_value)
    parsed = df["Pos"].map(parse_position) if "Pos" in df.columns else None
    out["place"] = [p[0] for p in parsed] if parsed is not None else None
    out["isFinal"] = [p[1] for p in parsed] if parsed is not None else True
    out["date"] = pd.to_datetime(df["Date"], format="%d %b %Y", errors="coerce")
    out["year"] = df["year"] if "year" in df.columns else out["date"].dt.year
    out["Venue"] = df.get("Venue")
    out["Meeting"] = df.get("Meeting", df.get("Venue"))
    out["tier"] = df["tier"] if "tier" in df.columns else tier
    out["source"] = source
    return out


_race_log_cache = {}


def load_race_log(disc_key):
    """One athlete-agnostic race log for a discipline, across every source.

    The worldwide scraper deliberately excludes the Diamond League (that is
    season_results_scraper's job and group 627 is not in its GROUPS), so a
    log built from it alone would be missing exactly the races this site is
    about. Both are merged here, and DL rows are tagged tier "DL".

    Cached on the source files' mtimes, not just the discipline key. Reading
    and concatenating three files per profile request measured a 722ms
    median and a 3.5s worst case across all 237 athletes, which is a page
    load; mtimes keep that correct while the worldwide scrape is still
    actively writing, and it stays fresh after any later re-scrape. Same
    pattern, and the same reasoning, as api.load_season_scores."""
    paths = [
        os.path.join(WORLDWIDE_DIR, f"{disc_key}.csv"),
        os.path.join(RAW_DIR, f"{disc_key}_{MEETS_YEAR}_meetings.csv"),
        os.path.join(RAW_DIR, f"{disc_key}.csv"),
    ]
    key = (disc_key, tuple(
        os.path.getmtime(p) if os.path.exists(p) else None for p in paths
    ))
    cached = _race_log_cache.get(key)
    if cached is not None:
        return cached

    frames = []

    wide_path = os.path.join(WORLDWIDE_DIR, f"{disc_key}.csv")
    if os.path.exists(wide_path):
        try:
            frames.append(_normalise(pd.read_csv(wide_path), "worldwide"))
        except Exception:
            pass

    dl_path = os.path.join(RAW_DIR, f"{disc_key}_{MEETS_YEAR}_meetings.csv")
    if os.path.exists(dl_path):
        try:
            frames.append(_normalise(pd.read_csv(dl_path), "dl", tier="DL"))
        except Exception:
            pass

    hist_path = os.path.join(RAW_DIR, f"{disc_key}.csv")
    if os.path.exists(hist_path):
        try:
            hist = pd.read_csv(hist_path)
            # Only the per-meeting rows: the toplist rows in this file are
            # one-per-athlete season bests, which would double-count a race
            # that is already in the log under its meeting name.
            if "source" in hist.columns:
                hist = hist[hist["source"].isin(["dl_meeting", "major_meet"])]
            frames.append(_normalise(hist, "dl", tier="DL"))
        except Exception:
            pass

    frames = [f for f in frames if not f.empty]
    if not frames:
        log = pd.DataFrame(columns=["Competitor", "Mark", "value", "place",
                                    "isFinal", "date", "year", "Venue",
                                    "Meeting", "tier", "source"])
    else:
        log = pd.concat(frames, ignore_index=True)
        # The same race reaches this from more than one scraper under
        # differently-formatted venue names; (athlete, date, mark) is the race.
        log = log.drop_duplicates(subset=["Competitor", "date", "Mark"])
        log = log[log["isFinal"]]
    # One entry per file-state; otherwise this grows once per re-scrape for
    # the lifetime of the process.
    _race_log_cache.clear()
    _race_log_cache[key] = log
    return log


def athlete_rows(log, athlete_name):
    if log.empty:
        return log
    return log[log["Competitor"].str.lower() == str(athlete_name).lower()]


def competition_record(rows):
    """Wins, podiums and finishing positions -- the numbers a season best
    cannot tell you. An athlete with the fourth-best mark in the world who
    wins every race they enter is a different proposition from one who set
    a fast time once and has finished fifth since."""
    placed = rows.dropna(subset=["place"])
    if placed.empty:
        return None
    places = placed["place"].astype(int)
    total = len(places)

    by_tier = []
    for tier, group in placed.groupby("tier", dropna=False):
        tier_places = group["place"].astype(int)
        by_tier.append({
            "tier":       None if pd.isna(tier) else str(tier),
            "label":      TIER_LABELS.get(str(tier), "Other"),
            "races":      int(len(tier_places)),
            "wins":       int((tier_places == 1).sum()),
            "podiums":    int((tier_places <= 3).sum()),
            "avgFinish":  round(float(tier_places.mean()), 2),
        })
    by_tier.sort(key=lambda t: TIER_ORDER.index(t["tier"])
                 if t["tier"] in TIER_ORDER else len(TIER_ORDER))

    by_season = []
    for year, group in placed.groupby("year"):
        season_places = group["place"].astype(int)
        by_season.append({
            "year":    int(year),
            "races":   int(len(season_places)),
            "wins":    int((season_places == 1).sum()),
            "podiums": int((season_places <= 3).sum()),
        })
    by_season.sort(key=lambda s: s["year"])

    top_tier = placed[placed["tier"].isin(TOP_TIERS)]
    return {
        "races":       total,
        "wins":        int((places == 1).sum()),
        "podiums":     int((places <= 3).sum()),
        "winRate":     round(float((places == 1).mean()) * 100, 1),
        "podiumRate":  round(float((places <= 3).mean()) * 100, 1),
        "avgFinish":   round(float(places.mean()), 2),
        "bestFinish":  int(places.min()),
        # Stated as a count and a share rather than a quality score, because
        # TIER_ORDER is a display order and not a ranking.
        "topTierRaces": int(len(top_tier)),
        "topTierShare": round(100 * len(top_tier) / total, 1),
        "byTier":      by_tier,
        "bySeason":    by_season,
        "seasons":     len(by_season),
    }


MIN_MARKS_FOR_CONSISTENCY = 3


def form_by_season(rows, is_field):
    """Per season: the average of the best three, the median, and how much
    the athlete varies -- all computed over the races in this log.

    Top-3 average is here because a season best is one lucky afternoon and
    every athletics analyst knows it. Consistency is the coefficient of
    variation (sd / mean), which is unit-free, so it reads the same for a
    9.8-second sprinter and a 74-metre thrower, and direction-free, so it
    needs no track/field flip.

    DELIBERATELY DOES NOT PUBLISH A SEASON BEST, and that is not an
    oversight. api.load_career_progression already owns that number and
    computes it from the TOPLIST, which carries an athlete's real best
    wherever it was set -- including meetings this log has never scraped.
    The two disagreed on the same athlete: Joe Kovacs's real 2018 best is
    21.02m and the best race in this log is 20.36m. Publishing both would
    have put two different "2018 best" figures on one page. `bestLogged`
    stays for the top-3 ordering and is not for display.

    Consistency is withheld below MIN_MARKS_FOR_CONSISTENCY marks: a
    coefficient of variation from two races is noise wearing two decimal
    places."""
    marked = rows.dropna(subset=["value", "year"])
    if marked.empty:
        return []

    seasons = []
    for year, group in marked.groupby("year"):
        values = group["value"].astype(float)
        ordered = values.sort_values(ascending=not is_field)
        top3 = ordered.head(3)
        mean = float(values.mean())
        enough = len(values) >= MIN_MARKS_FOR_CONSISTENCY
        seasons.append({
            "year":        int(year),
            "marks":       int(len(values)),
            "bestLogged":  round(float(values.max() if is_field else values.min()), 3),
            "top3Average": round(float(top3.mean()), 3),
            "top3Count":   int(len(top3)),
            "median":      round(float(values.median()), 3),
            # As a percentage: 1.2% means this athlete's marks sit within
            # about a percent of each other all season.
            "consistency": round(float(values.std(ddof=0) / mean * 100), 2) if (enough and mean) else None,
            "spread":      round(float(values.max() - values.min()), 3),
        })
    seasons.sort(key=lambda s: s["year"])
    return seasons


MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def season_shape(rows, year, is_field):
    """When in the season an athlete races, and when their best mark lands.

    Directly relevant to a Final in September: an athlete whose best came in
    May and who has raced flat since is a different bet from one peaking in
    August, and neither is visible in a season-best figure."""
    season = rows[(rows["year"] == year)].dropna(subset=["date"])
    if season.empty:
        return None
    by_month = []
    for month in range(1, 13):
        month_rows = season[season["date"].dt.month == month]
        if month_rows.empty:
            continue
        by_month.append({"month": MONTHS[month - 1], "races": int(len(month_rows))})

    marked = season.dropna(subset=["value"])
    best_month = None
    if not marked.empty:
        idx = marked["value"].idxmax() if is_field else marked["value"].idxmin()
        best_row = marked.loc[idx]
        best_month = MONTHS[int(best_row["date"].month) - 1]

    return {
        "byMonth":     by_month,
        "bestMonth":   best_month,
        "firstRace":   season["date"].min().strftime("%d %b"),
        "lastRace":    season["date"].max().strftime("%d %b"),
        "races":       int(len(season)),
    }


def head_to_head(log, athlete_name, opponents=None, min_meetings=1):
    """Real head-to-head, from actually being in the same race.

    Every other h2h source in this project was assembled meeting by meeting
    from a separate scrape with known gaps (HANDOFF 0o: 63.1% coverage after
    the data was corrected, and 64% fabricated sweeps before it). This is
    derived: two athletes appear in one race when they share a discipline, a
    date and a meeting, and whoever has the lower finishing position won.
    Nothing is inferred and nothing is filled in.

    `opponents=None` returns every opponent this athlete has ever met, which
    is the analyst view; the profile passes a shortlist."""
    mine = athlete_rows(log, athlete_name).dropna(subset=["place"])
    if mine.empty:
        return []

    race_keys = set(zip(mine["date"], mine["Meeting"]))
    my_places = {(d, m): int(p) for d, m, p in
                 zip(mine["date"], mine["Meeting"], mine["place"])}

    others = log.dropna(subset=["place"])
    others = others[others["Competitor"].str.lower() != str(athlete_name).lower()]
    if opponents is not None:
        wanted = {str(o).lower() for o in opponents}
        others = others[others["Competitor"].str.lower().isin(wanted)]

    records = {}
    for name, date, meeting, place in zip(others["Competitor"], others["date"],
                                          others["Meeting"], others["place"]):
        if (date, meeting) not in race_keys:
            continue
        rec = records.setdefault(name, {"name": name, "wins": 0, "losses": 0,
                                        "draws": 0, "meetings": 0, "lastMet": None})
        mine_place = my_places[(date, meeting)]
        rec["meetings"] += 1
        if mine_place < int(place):
            rec["wins"] += 1
        elif mine_place > int(place):
            rec["losses"] += 1
        else:
            rec["draws"] += 1
        if rec["lastMet"] is None or date > rec["lastMet"]:
            rec["lastMet"] = date

    out = []
    for rec in records.values():
        if rec["meetings"] < min_meetings:
            continue
        rec["lastMet"] = rec["lastMet"].strftime("%d %b %Y") if rec["lastMet"] is not None else None
        rec["winRate"] = round(100 * rec["wins"] / rec["meetings"], 1)
        out.append(rec)
    out.sort(key=lambda r: (-r["meetings"], -r["winRate"]))
    return out


def build_analytics(disc_key, athlete_name, is_field, opponents=None, year=MEETS_YEAR):
    """The whole analyst block for one athlete. Returns None when the race
    log has nothing for them -- the profile then shows what it always did
    rather than a page of zeroes."""
    log = load_race_log(disc_key)
    if log.empty:
        return None
    rows = athlete_rows(log, athlete_name)
    if rows.empty:
        return None

    record = competition_record(rows)
    return {
        "raceCount":   int(len(rows)),
        "record":      record,
        "form":        form_by_season(rows, is_field),
        "seasonShape": season_shape(rows, year, is_field),
        "headToHead":  head_to_head(log, athlete_name, opponents),
        # What the numbers above are actually computed from, so the page can
        # say so instead of implying a complete career.
        "coverage": {
            "seasons":  sorted({int(y) for y in rows["year"].dropna().unique()}),
            "sources":  sorted(rows["source"].dropna().unique().tolist()),
            "withPlace": int(rows["place"].notna().sum()),
        },
    }
