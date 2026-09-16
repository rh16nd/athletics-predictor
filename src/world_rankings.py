"""
world_rankings.py -- per-discipline "best in the world" lists for the Track and
Field pages, in two orderings the UI toggles between:

  points : World Athletics performance points (the Results Score of each
           athlete's season best) -- an objective ranking by what they've run.
  model  : the championship model's rating (src/field_model.py), from the model
           that calls the championships. The top 20 by points are read as if
           they met in one final, and `ratingPct` is the chance it gives each of
           finishing in the top three, so an event's 20 add up to 300. Both
           lists carry it. The site calls it the model rating and never a
           podium chance, which it names only for a real competition, where the
           athletes are actually entered (the user's rule, 2026-09-17).

Every event is read this way since 2026-09-17. Until then the 32 Diamond League
events were rated by the Diamond League model (src/train_model.py, a random
forest), and only the hammer and the 10,000m, which it has never seen, had the
field model. That rating followed meetings raced more than marks (measured
2026-09-07: 1.2% at no Diamond League meetings against 28.8% at five, on almost
the same World Athletics score), and on the same 373 past finals the field model
named more medallists (67.4% against 60.9%; src/model_head_to_head.py), so by
the rule fixed before that test it replaced the forest everywhere.

Both are built from data we already scrape (the season toplists and the top
20s' seasons race by race), so this runs in the normal refresh with no new
sources. Writes data/world_rankings.json:
  { "<disc_key>": { "isField": bool, "modelAvailable": bool, "modelKind": str,
                    "model": [rows...], "points": [rows...] } }
each row: { rank, name, nat, mark, score, ratingPct, racesOnRecord,
            profileUrl }.

Usage:
    python src/world_rankings.py
    python src/world_rankings.py --only men_HT women_HT   # rewrite just these
    python src/world_rankings.py --refresh-races          # fetch the field model's top 20s' seasons again
"""
import argparse
import glob
import json
import os
import sys
from datetime import date

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import asian_games_scraper as ags  # noqa: E402
import field_data as fd  # noqa: E402
import field_model as fm  # noqa: E402
from feature_builder import FIELD_EVENTS, RAW_DIR  # noqa: E402
from season_activity import races_on_record  # noqa: E402

# The field model's top 20s, their seasons race by race, for a model that reads
# races (field_model.needs_races). Fetched when missing, or again with --refresh-races.
CURRENT_RACES_DIR = os.path.join(fd.FIELD_DIR, "current")

# Overridable so an experimental build can be written beside the served one.
#   PODIUMCALL_RANKINGS_OUT  where to write the rankings JSON
OUT_PATH = os.environ.get(
    "PODIUMCALL_RANKINGS_OUT",
    os.path.join(os.path.dirname(__file__), "..", "data", "world_rankings.json"))
YEAR = 2026
TOP_N = 20


def discipline_keys():
    keys = []
    for path in glob.glob(os.path.join(RAW_DIR, f"*_{YEAR}.csv")):
        base = os.path.basename(path)[: -len(f"_{YEAR}.csv")]
        keys.append(base)
    return sorted(keys)


def toplist_meta(key):
    """name -> {mark, nat, score, url} from the raw toplist, taking each
    athlete's best (first) row. Nationality has a blank header in WA's export,
    so it is read by position (column 5), same as everywhere else."""
    path = os.path.join(RAW_DIR, f"{key}_{YEAR}.csv")
    raw = pd.read_csv(path).dropna(subset=["Competitor"])
    meta = {}
    for _, r in raw.iterrows():
        name = r["Competitor"]
        if name in meta:
            continue
        try:
            score = int(r["Results Score"]) if pd.notna(r["Results Score"]) else None
        except (ValueError, TypeError):
            score = None
        meta[name] = {
            "mark": str(r["Mark"]) if pd.notna(r["Mark"]) else None,
            "nat": str(r.iloc[5]) if pd.notna(r.iloc[5]) else None,
            "score": score,
            "url": str(r["ProfileURL"]) if pd.notna(r["ProfileURL"]) else None,
        }
    return meta


def add_h2h(df, key):
    """h2h_win_rate as a model input, computed over this pool -- the same
    case-insensitive lookup run.py uses (h2h_rates.csv is normal-case, the
    toplist is ALL-CAPS-surname). No longer used here since the rankings left
    the Diamond League model; kept for ultimate_predictions.py, which imports it."""
    h2h_path = os.path.join(os.path.dirname(__file__), "..", "data", "h2h", "h2h_rates.csv")
    if not os.path.exists(h2h_path):
        df["h2h_win_rate"] = 0.5
        return df
    h2h = pd.read_csv(h2h_path)
    disc = h2h[(h2h["discipline"] == key) & (h2h["meetings"] >= 2)].copy()
    lookup = {}
    for _, r in disc.iterrows():
        lookup.setdefault(str(r["athlete_a"]).lower(), {})[str(r["athlete_b"]).lower()] = r["win_rate"]
    names_lower = df["athlete_name"].str.lower().tolist()

    def rate(n):
        rates = [lookup[n][o] for o in names_lower if o != n and n in lookup and o in lookup[n]]
        return sum(rates) / len(rates) if rates else 0.5

    df["h2h_win_rate"] = [rate(n) for n in names_lower]
    return df


def ranked(rows, sort_key, reverse=True):
    """The top TOP_N rows by one key, best first, with missing values last."""
    ordered = sorted(rows, key=lambda x: (x[sort_key] is None, x[sort_key] or 0), reverse=reverse)
    # None-last even when reverse=True: push missing values to the bottom
    ordered = [x for x in ordered if x[sort_key] is not None] + [x for x in ordered if x[sort_key] is None]
    out = []
    for i, x in enumerate(ordered[:TOP_N], 1):
        out.append({
            "rank": i, "name": x["name"], "nat": x["nat"], "mark": x["mark"],
            "score": x["score"], "ratingPct": x.get("ratingPct"),
            "racesOnRecord": x["racesOnRecord"], "profileUrl": x["profileUrl"],
        })
    return out


def current_races(athletes, refresh=False):
    """{World Athletics id: this season race by race} for these athletes,
    fetched into CURRENT_RACES_DIR when missing, or all again with `refresh`."""
    wanted = {(int(a["waId"]), YEAR) for a in athletes if str(a.get("waId") or "").isdigit()}
    fd.fetch_seasons(wanted, fetch=fd.fetch_races, seasons_dir=CURRENT_RACES_DIR, refresh=refresh)
    return fd.load_seasons(CURRENT_RACES_DIR).get(YEAR, {})


def field_model_rows(key, rows, today=None, model=None, races=None, refresh_races=False):
    """The field model's view of one event: its top 20 by points, each with the
    model rating, best first. Empty without a saved field model, or with fewer
    than three athletes it can read.

    `ratingPct` is the chance of a top three as if the 20 met in one final, so
    the 20 add up to 300: one hundred for each podium place. The model reads
    the field the way it was chosen to (field_model.spec_of): since 2026-09-15
    each athlete's season race by race
    as well as their season best, career best, last season and age, with
    `races` fetched by current_races when not given."""
    model = model if model is not None else fm.load_model()
    if not model:
        return []
    field = ranked(rows, "score")
    by_name = {r["name"]: r for r in rows}
    # The end of today, so a mark set today still counts as this season's.
    cutoff = pd.Timestamp(today or date.today()) + pd.Timedelta(days=1)
    athletes = [{"name": r["name"], "nat": r["nat"], "waId": ags.wa_id(r["profileUrl"])} for r in field]
    how = {}
    if fm.needs_races(model):
        how = {"races": current_races(athletes, refresh_races) if races is None else races,
               "race_how": fm.spec_of(model).get("race")}
    served = fm.serving_rows(key, athletes, os.path.join(RAW_DIR, f"{key}_{YEAR}.csv"), cutoff, YEAR, **how)
    chances = fm.field_chances(model, served, cutoff, key)
    scored = [{**by_name[name], "ratingPct": round(podium * 100, 1), "prob": podium, "win": win}
              for name, (podium, win) in chances.items() if name in by_name]
    # Ties go to the likelier winner, then the better score; ranked() keeps
    # that order among equal ratings because Python's sort is stable.
    order = sorted(scored, key=lambda r: (-r["win"], -(r["score"] or 0)))
    return ranked(order, "prob")


def field_model_discipline(key, today=None, model=None, refresh_races=False):
    """One event's two lists: the world's top 20 by points, and the same 20
    ranked by the championship model's rating. Every event since 2026-09-17;
    the hammer and the 10,000m since 2026-09-15.

    The rating travels on the points list too, so the Track and Field table
    shows it whichever order the reader picks. `modelAvailable` is false only
    when the field model has nothing to say, and then both lists carry None."""
    meta = toplist_meta(key)
    if not meta:
        return None
    activity = races_on_record(key, YEAR)
    rows = [{
        "name": name, "nat": m["nat"], "mark": m["mark"], "score": m["score"],
        "ratingPct": None,
        "racesOnRecord": activity.get(str(name).upper().strip()),
        "profileUrl": m["url"],
    } for name, m in meta.items()]
    field = field_model_rows(key, rows, today, model, refresh_races=refresh_races)
    rating = {r["name"]: r["ratingPct"] for r in field}
    points = [{**r, "ratingPct": rating.get(r["name"])} for r in ranked(rows, "score")]
    return {
        "isField": key in FIELD_EVENTS,
        "modelAvailable": bool(field),
        "modelKind": "field" if field else None,
        "model": field,
        "points": points,
    }


def score_discipline(key, refresh_races=False):
    return field_model_discipline(key, refresh_races=refresh_races)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Per-discipline top-20 lists for the Track and Field pages.")
    parser.add_argument("--only", nargs="+", metavar="KEY",
                        help="rebuild just these disciplines and keep the rest of the saved file as it is")
    parser.add_argument("--refresh-races", action="store_true",
                        help="fetch the field model's top 20s' seasons again, for meetings since the last fetch")
    args = parser.parse_args(argv)

    print("=== Building world rankings (points + model) per discipline ===")
    out = {}
    if args.only:
        try:
            with open(OUT_PATH, encoding="utf-8") as f:
                out = json.load(f)
        except (OSError, ValueError):
            out = {}
    for key in args.only or discipline_keys():
        res = score_discipline(key, args.refresh_races)
        if res:
            out[key] = res
            model_top = res["model"][0]["name"] if res["model"] else "(points only)"
            print(f"  {key}: model#1={model_top} ({res.get('modelKind')})  points#1={res['points'][0]['name']}")
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"\n  {len(out)} disciplines -> {os.path.abspath(OUT_PATH)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
