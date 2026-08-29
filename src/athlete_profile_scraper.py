"""
athlete_profile_scraper.py -- per-athlete career data from World Athletics:
personal bests, honours, world ranking, and a complete current-season race
log across every event they contest.

READ THIS IF YOU HAVE SEEN HANDOFF'S "DEAD RESOLVER" NOTE
---------------------------------------------------------
HANDOFF's Failed Attempts records that per-athlete race-log scraping was
tried in the sixth session and abandoned: `getSingleCompetitor.resultsByDate`
returned null for every combination of athlete id, year and ordering, and
the entry says "Don't re-attempt this without new evidence the resolver was
fixed -- it's a dead field, not a query-construction mistake."

That finding is correct and this module does not contradict it. There is a
DIFFERENT field on the same object, `resultsByYear`, and it resolves fine:

    getSingleCompetitor(id: 14536762) {
      resultsByYear {
        activeYears
        resultsByEvent { discipline results { date competition venue place mark wind } }
      }
    }

returns Noah Lyles's real 2026 season across seven events -- including the
60m indoors and a 150m at Ostrava that no meeting-level scrape of the
outdoor competition groups would ever pick up. `resultsByDate` is still
dead. The lesson is narrow: a dead resolver on an object does not mean the
object has nothing.

WHAT IS AND ISN'T AVAILABLE
---------------------------
Available, verified live:
  * personalBests -- every discipline, with mark, venue and date
  * honours       -- grouped and LABELLED by World Athletics itself via
                     `categoryName`: "Olympic Games", "World Championships",
                     "Diamond League Final", "World Indoor Championships",
                     "World Athletics Relays". Without that label the groups
                     are anonymous and a national title is indistinguishable
                     from an Olympic one -- the first fetch omitted it and
                     the honours were unusable for anything but a raw count.
  * worldRankings -- WA's own current ranking place per event group
  * resultsByYear -- the CURRENT season only; there is no year argument on
                     either `resultsByYear` or `getSingleCompetitor`
                     (`year`, `season` and `resultsByYear` were all tried
                     and rejected as unknown arguments), though
                     `activeYears` does list every season they competed in.

Not available, on this endpoint or any other checked: split times, reaction
times, segment times, top or peak speed. The result object is
`CalendarResultsRaceResult` and its fields are id, competitor, mark,
nationality, place, points, qualified, records, wind, remark and details --
introspection of the schema itself is blocked (503 on `__schema`), so this
was established by probing field names and reading the validation errors.
Speed can only be computed as distance over time, which is an AVERAGE and
must never be presented as a top speed.

QUARANTINE
----------
Writes data/athlete_profiles/, alongside data/worldwide/ and outside
data/raw/, for the same reason: nothing in the modelling path may pick this
up by globbing. Feeding any of it to the model is a separate decision with
its own backtest.

Usage:
    python src/athlete_profile_scraper.py            # athletes in predictions_latest.csv
    python src/athlete_profile_scraper.py --all      # every athlete with a toplist profile URL
    python src/athlete_profile_scraper.py --limit 20
    python src/athlete_profile_scraper.py --status
"""
import argparse
import glob
import io
import json
import os
import re
import sys
import time

import pandas as pd

if not (sys.stdout.encoding or "").lower().startswith("utf"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.dirname(__file__))
import dl_final_results_scraper as dlr  # noqa: E402

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
OUT_DIR = os.path.join(BASE_DIR, "data", "athlete_profiles")
INDEX_PATH = os.path.join(OUT_DIR, "_index.json")
PREDICTIONS = os.path.join(BASE_DIR, "outputs", "predictions_latest.csv")

REQUEST_PAUSE = 0.35
ATHLETE_ID = re.compile(r"athlete[=/](\d+)")

PROFILE_QUERY = """query GetSingleCompetitor($id: Int) {
  getSingleCompetitor(id: $id) {
    basicData { birthDate countryCode }
    personalBests { results { discipline mark venue date } }
    honours { categoryName results { competition mark place } }
    worldRankings { current { eventGroup place } }
    resultsByYear {
      activeYears
      resultsByEvent {
        discipline
        results { date competition venue place mark wind }
      }
    }
  }
}"""


def athlete_ids():
    """(athlete name, WA id) from every toplist ProfileURL on disk. The id
    is the only reliable key -- names collide and are formatted
    inconsistently across sources."""
    found = {}
    for path in glob.glob(os.path.join(RAW_DIR, "*_2026.csv")):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if "ProfileURL" not in df.columns:
            continue
        for name, url in zip(df["Competitor"], df["ProfileURL"]):
            match = ATHLETE_ID.search(str(url))
            if match and pd.notna(name):
                found.setdefault(str(name), match.group(1))
    return found


def predicted_athletes():
    """Just the athletes the site actually renders a page for."""
    if not os.path.exists(PREDICTIONS):
        return set()
    try:
        return set(pd.read_csv(PREDICTIONS)["athlete_name"].dropna())
    except Exception:
        return set()


def load_index():
    try:
        with open(INDEX_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_index(index):
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)


def fetch_profile(athlete_id):
    data = dlr.graphql("GetSingleCompetitor", {"id": int(athlete_id)}, PROFILE_QUERY)
    return data.get("getSingleCompetitor")


def summarise(profile):
    """Counts for the log line, so a run that is silently returning empty
    objects is visible while it happens rather than afterwards."""
    if not profile:
        return "no data"
    pbs = len(((profile.get("personalBests") or {}).get("results")) or [])
    honours = sum(len(h.get("results") or []) for h in (profile.get("honours") or []))
    rankings = len(((profile.get("worldRankings") or {}).get("current")) or [])
    by_event = ((profile.get("resultsByYear") or {}).get("resultsByEvent")) or []
    races = sum(len(e.get("results") or []) for e in by_event)
    return f"{pbs} PBs, {honours} honours, {rankings} rankings, {races} races"


def status():
    index = load_index()
    print(f"{len(index)} athlete profiles fetched -> {os.path.abspath(OUT_DIR)}")
    if not index:
        return
    totals = {"pbs": 0, "honours": 0, "races": 0, "ranked": 0}
    for entry in index.values():
        totals["pbs"] += entry.get("pbs", 0)
        totals["honours"] += entry.get("honours", 0)
        totals["races"] += entry.get("races", 0)
        totals["ranked"] += 1 if entry.get("rankings") else 0
    print(f"  {totals['pbs']:,} personal bests, {totals['honours']:,} honours entries, "
          f"{totals['races']:,} logged races, {totals['ranked']:,} with a world ranking")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true",
                        help="every athlete with a toplist profile URL, not just predicted ones")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--refresh", action="store_true",
                        help="re-fetch athletes already on disk")
    args = parser.parse_args()

    if args.status:
        status()
        return

    ids = athlete_ids()
    if not args.all:
        wanted = predicted_athletes()
        ids = {n: i for n, i in ids.items() if n in wanted}
    index = load_index()
    os.makedirs(OUT_DIR, exist_ok=True)

    todo = [(n, i) for n, i in sorted(ids.items())
            if args.refresh or i not in index]
    print(f"{len(ids)} athletes in scope, {len(todo)} to fetch "
          f"({len(index)} already on disk)\n")

    done = 0
    for name, athlete_id in todo:
        if args.limit is not None and done >= args.limit:
            print(f"\nStopping at --limit {args.limit}.")
            break
        try:
            profile = fetch_profile(athlete_id)
        except Exception as exc:
            print(f"  ! {name}: {str(exc)[:90]}")
            time.sleep(REQUEST_PAUSE)
            continue
        if not profile:
            print(f"  - {name}: no profile returned")
            time.sleep(REQUEST_PAUSE)
            continue

        with open(os.path.join(OUT_DIR, f"{athlete_id}.json"), "w", encoding="utf-8") as f:
            json.dump({"id": athlete_id, "name": name, "profile": profile},
                      f, indent=2, ensure_ascii=False)

        by_event = ((profile.get("resultsByYear") or {}).get("resultsByEvent")) or []
        index[athlete_id] = {
            "name":     name,
            "pbs":      len(((profile.get("personalBests") or {}).get("results")) or []),
            "honours":  sum(len(h.get("results") or []) for h in (profile.get("honours") or [])),
            "rankings": len(((profile.get("worldRankings") or {}).get("current")) or []),
            "races":    sum(len(e.get("results") or []) for e in by_event),
            "events":   len(by_event),
        }
        done += 1
        print(f"  {name[:34]:<34} {summarise(profile)}")
        if done % 25 == 0:
            save_index(index)
        time.sleep(REQUEST_PAUSE)

    save_index(index)
    print(f"\nFetched {done} profiles. {len(index)} on disk.")


if __name__ == "__main__":
    main()
