"""
worldwide_scraper.py -- every senior outdoor track & field meeting World
Athletics has results for, not just the Diamond League circuit.

WHY THIS EXISTS, AND WHY IT IS NOT A MODEL CHANGE
-------------------------------------------------
The prediction pipeline deliberately eats a narrow diet: the DL circuit
(season_results_scraper.py) plus four top-tier groups (major_meets_scraper).
Widening it was tried in the sixth session and made the backtest WORSE --
lower-tier marks flow into season_best/consistency/weighted_season_best and
dilute them faster than the extra rows help. See HANDOFF's Failed Attempts.
That verdict stands and this file does not touch it.

This is for the PRODUCT, which has the opposite problem. Measured on disk
2026-08-29: the median athlete has **2 marks across 2018-2025 and 2 races in
2026**, and only 28% have four or more races this season. A profile page
built on that renders two dots and calls it a career. The sixth session's
own prototype showed this data is exactly the fix -- median final-round
races per athlete-season went from 1.0 to 3.0-4.0 -- it just measured that
gain against the wrong target.

QUARANTINE -- read before changing any path here
------------------------------------------------
Output goes to `data/worldwide/{discipline}.csv`, **outside data/raw/**.
That is not cosmetic and not a preference:
  - train_model.py reads `data/raw/{discipline}.csv` by name.
  - venue_geo.py and venue_weather.py glob `data/raw/*.csv`.
Anything written into data/raw/ risks being pulled into training and
re-running the exact regression that was already measured and reverted.
Nothing in the modelling path reads data/worldwide/. Keep it that way.

SCOPE
-----
Senior OUTDOOR track & field only. Every group below is a separate
competitionGroupId, so indoor, road, cross country, race walking, combined
events and U18/U20/U23 are excluded by construction rather than by filtering
names. Measured for 2025: ~413 meetings with results across these groups.

Rows are kept for athletes the site actually knows -- the union of every
discipline's toplist names (~8.5k athletes). An unrecognised name is one the
site can never render a page for, so storing their rows only inflates the
files.

Resumable and polite: every completed competition id is recorded in
data/worldwide/_state.json, files are flushed every FLUSH_EVERY meetings,
and re-running skips what is already done. Seasons run newest-first so the
most useful data lands soonest.

Usage:
    python src/worldwide_scraper.py                  # 2018-2026, newest first
    python src/worldwide_scraper.py --years 2026 2025
    python src/worldwide_scraper.py --limit 25       # stop after 25 meetings
    python src/worldwide_scraper.py --status         # what's done so far
"""
import argparse
import calendar
import io
import json
import os
import re
import sys
import time
from collections import defaultdict

import pandas as pd

# Guarded for the reason every module in src/ guards it -- see HANDOFF 0i.
if not (sys.stdout.encoding or "").lower().startswith("utf"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.dirname(__file__))
import dl_final_results_scraper as dlr  # noqa: E402

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "worldwide")
STATE_PATH = os.path.join(OUT_DIR, "_state.json")

YEARS = [y for y in range(2018, 2027) if y != 2020]  # 2020 was COVID-thin

# Senior outdoor track & field competition groups, with the 2025 meeting
# counts that justified including each one (measured, not assumed).
GROUPS = {
    3773: "Continental Tour",          # 264 -- all tiers, the density fix
    3731: "National Championships",    # 131 -- where many season bests are set
    3771: "Area Regional Champs",      #  13
    3660: "Area Champs",               #   3
    3802: "Traditional International",  #   1
    3804: "FISU",                      #   1
}

REQUEST_PAUSE = 0.25   # seconds between GraphQL calls
FLUSH_EVERY = 20       # meetings between disk writes
CALENDAR_RETRIES = 3   # a failed calendar window silently loses a whole month

# The group split does almost all the work -- measured on 2026, exactly ONE
# of 399 meetings in these groups names itself indoor ("Balkan Indoor
# Championships", misfiled under Area Regional Champs). It still has to go:
# an indoor 200m or long jump maps to the same discipline key as its outdoor
# version but is a different event (banked track, no wind), so those rows
# would quietly corrupt every per-athlete progression built from them.
# \b anchors are load-bearing -- an unanchored "hallen" matches
# "C-hallen-ger" and threw out ten Continental Tour Challenger meetings on
# the first attempt.
INDOOR_NAME = re.compile(r"\b(indoor|indoors|hallen\w*|en salle|banked)\b", re.I)

CALENDAR_QUERY = """query getCalendarEvents($startDate: String, $endDate: String, $competitionGroupId: Int) {
  getCalendarEvents(startDate: $startDate, endDate: $endDate, competitionGroupId: $competitionGroupId, limit: 100) {
    results { id name venue rankingCategory startDate hasApiResults }
  }
}"""

RESULTS_QUERY = """query getCalendarCompetitionResults($competitionId: Int, $day: Int, $eventId: Int) {
  getCalendarCompetitionResults(competitionId: $competitionId, day: $day, eventId: $eventId) {
    options { days { day date } }
    eventTitles {
      rankingCategory
      events {
        event
        gender
        races {
          race
          results { place mark wind nationality competitor { name birthDate } }
        }
      }
    }
  }
}"""


def load_state():
    try:
        with open(STATE_PATH, encoding="utf-8") as f:
            state = json.load(f)
    except (OSError, ValueError):
        state = {}
    state.setdefault("done", [])
    state.setdefault("empty", [])
    state.setdefault("failed_windows", [])
    return state


def save_state(state):
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def recognized_everywhere():
    """Union of every discipline's toplist names. Per-discipline filtering
    (what major_meets_scraper does) is right for training rows but wrong
    here: a 400m runner's 200m race is exactly the kind of row a profile
    page wants, and it would be dropped for not being top-N in the 200m."""
    names = set()
    if not os.path.isdir(RAW_DIR):
        return names
    for fname in os.listdir(RAW_DIR):
        if not fname.endswith(".csv") or "_2026" in fname or fname == "archive.zip":
            continue
        names |= dlr.load_recognized_names(fname[:-4], RAW_DIR)
    return names


def calendar_window(gid, start, end):
    """One calendar query, retried -- a window that fails is a whole month
    of meetings silently missing, which is indistinguishable from a month
    that genuinely had none. Both failure modes were real on the first
    trial run: transient `Lambda:Unhandled` errors that succeed on retry,
    and one deterministic crash from asking for 29 February in a non-leap
    year (WA's resolver throws rather than returning an error)."""
    last_error = None
    for attempt in range(CALENDAR_RETRIES):
        try:
            return dlr.graphql(
                "getCalendarEvents",
                {"startDate": start, "endDate": end, "competitionGroupId": gid},
                CALENDAR_QUERY,
            )["getCalendarEvents"]["results"]
        except Exception as exc:
            last_error = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"{start}..{end} group {gid}: {last_error}")


def find_year_meetings(year, failures=None):
    """Every meeting with results in the target groups, month by month.

    Month-sized windows are not an optimisation -- getCalendarEvents caps at
    100 results and returns no indication that it truncated, so a full-year
    query for the Continental Tour silently returns 100 of ~265. That trap
    cost the sixth session real time; don't widen these windows.

    `calendar.monthrange` rather than a hardcoded month-length table: a
    table with 29 for February crashes WA's resolver in every non-leap
    year, taking that month's meetings with it."""
    meetings = []
    seen = set()
    for gid, label in GROUPS.items():
        for month in range(1, 13):
            start = f"{year}-{month:02d}-01"
            end = f"{year}-{month:02d}-{calendar.monthrange(year, month)[1]}"
            try:
                results = calendar_window(gid, start, end)
            except Exception as exc:
                # Recorded, not just printed -- an unreported gap in the
                # calendar is a gap in every stat computed downstream.
                print(f"    ! GAVE UP on {label} {year}-{month:02d}: {exc}")
                if failures is not None:
                    failures.append(f"{year}-{month:02d} group {gid} ({label})")
                continue
            if len(results) >= 100:
                print(f"    ! {label} {year}-{month:02d} returned {len(results)} -- "
                      "at the cap, this month may be truncated")
            for meeting in results:
                if not meeting.get("hasApiResults") or meeting["id"] in seen:
                    continue
                if INDOOR_NAME.search(meeting.get("name") or ""):
                    print(f"    - skipping indoor: {meeting['name']}")
                    continue
                seen.add(meeting["id"])
                meetings.append((meeting, label))
            time.sleep(REQUEST_PAUSE)
    meetings.sort(key=lambda pair: pair[0].get("startDate") or "")
    return meetings


def scrape_meeting(meeting, year, label, keep_names):
    """Final-round results for every senior T&F event at one meeting.

    Only races WA labels "Final" are read, so `place` is always a real
    final placing rather than a heat position -- the same contract
    major_meets_scraper.py relies on."""
    rows = []
    probe = dlr.graphql(
        "getCalendarCompetitionResults",
        {"competitionId": meeting["id"], "day": None, "eventId": None},
        RESULTS_QUERY,
    )["getCalendarCompetitionResults"]
    time.sleep(REQUEST_PAUSE)
    days = (probe.get("options") or {}).get("days") or [{"day": None, "date": None}]

    seen_events = set()
    for day_info in days:
        if day_info.get("day") is None:
            data = probe
        else:
            data = dlr.graphql(
                "getCalendarCompetitionResults",
                {"competitionId": meeting["id"], "day": day_info["day"], "eventId": None},
                RESULTS_QUERY,
            )["getCalendarCompetitionResults"]
            time.sleep(REQUEST_PAUSE)
        date_str = day_info.get("date") or meeting.get("startDate")

        for group in data.get("eventTitles") or []:
            for event in group.get("events") or []:
                # No mile_as_1500: these rows are a per-meeting time series,
                # where a Mile is genuinely not a 1500m. See HANDOFF 0c.
                key = dlr.resolve_discipline_key(event["gender"], event["event"])
                if key is None:
                    continue
                event_key = (key, event["event"])
                if event_key in seen_events:
                    continue  # multi-day meetings repeat listings across day queries
                finals = [r for r in event["races"] if r["race"] == "Final"]
                if not finals:
                    continue
                seen_events.add(event_key)
                for result in finals[0]["results"]:
                    competitor = result.get("competitor") or {}
                    name = competitor.get("name")
                    if not name or name not in keep_names:
                        continue
                    rows.append({
                        "discipline": key,
                        "Competitor": name,
                        "DOB": competitor.get("birthDate"),
                        "Mark": result.get("mark"),
                        "WIND": result.get("wind"),
                        "Nat": result.get("nationality"),
                        "Venue": meeting.get("venue") or meeting.get("name"),
                        "Meeting": meeting.get("name"),
                        "Date": date_str,
                        "year": year,
                        "Pos": result.get("place"),
                        "tier": meeting.get("rankingCategory"),
                        "group": label,
                        "competitionId": meeting["id"],
                        "source": "worldwide",
                    })
    return rows


def flush(buffered):
    """Append buffered rows per discipline, de-duplicated on the natural key.

    De-duplication matters because a re-run after an interrupted flush can
    re-scrape a meeting whose rows were already written -- the state file is
    only saved at flush points."""
    os.makedirs(OUT_DIR, exist_ok=True)
    written = 0
    for key, rows in buffered.items():
        if not rows:
            continue
        path = os.path.join(OUT_DIR, f"{key}.csv")
        new = pd.DataFrame(rows)
        if os.path.exists(path):
            new = pd.concat([pd.read_csv(path), new], ignore_index=True)
        new = new.drop_duplicates(subset=["Competitor", "Mark", "Date", "competitionId", "discipline"])
        new.to_csv(path, index=False)
        written += len(rows)
    buffered.clear()
    return written


def status():
    state = load_state()
    print(f"Completed meetings: {len(state['done'])} ({len(state['empty'])} with no usable rows)")
    if not os.path.isdir(OUT_DIR):
        print("No output yet.")
        return
    total = 0
    per = []
    for fname in sorted(os.listdir(OUT_DIR)):
        if not fname.endswith(".csv"):
            continue
        n = len(pd.read_csv(os.path.join(OUT_DIR, fname)))
        per.append((fname[:-4], n))
        total += n
    print(f"{len(per)} disciplines, {total:,} rows")
    for key, n in sorted(per, key=lambda kv: -kv[1])[:10]:
        print(f"  {key:<16} {n:>7,}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", type=int, nargs="+", default=None)
    parser.add_argument("--limit", type=int, default=None,
                        help="stop after this many meetings (for a cheap trial run)")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()

    if args.status:
        status()
        return

    years = args.years or sorted(YEARS, reverse=True)  # newest first
    state = load_state()
    done = set(state["done"])
    keep_names = recognized_everywhere()
    print(f"Filtering to {len(keep_names):,} athletes the site recognises.")
    print(f"Output: {os.path.abspath(OUT_DIR)} (quarantined from data/raw)\n")

    buffered = defaultdict(list)
    scraped = kept = 0

    for year in years:
        print(f"[{year}] finding meetings...")
        meetings = find_year_meetings(year, failures=state["failed_windows"])
        todo = [(m, label) for m, label in meetings if m["id"] not in done]
        print(f"[{year}] {len(meetings)} meetings with results, {len(todo)} still to do")

        for meeting, label in todo:
            if args.limit is not None and scraped >= args.limit:
                print(f"\nStopping at --limit {args.limit}.")
                kept += flush(buffered)
                state["done"] = sorted(done)
                save_state(state)
                print(f"Kept {kept:,} rows from {scraped} meetings.")
                return
            try:
                rows = scrape_meeting(meeting, year, label, keep_names)
            except Exception as exc:
                print(f"    ! {meeting['name'][:44]}: {exc}")
                continue
            scraped += 1
            done.add(meeting["id"])
            if rows:
                for row in rows:
                    buffered[row["discipline"]].append(row)
            else:
                state["empty"].append(meeting["id"])
            print(f"  [{year}] {meeting.get('startDate', '')[:10]} {label:<26} "
                  f"{meeting['name'][:40]:<40} {len(rows):>4} rows")

            if scraped % FLUSH_EVERY == 0:
                kept += flush(buffered)
                state["done"] = sorted(done)
                save_state(state)
                print(f"  -- checkpoint: {scraped} meetings, {kept:,} rows kept")

    kept += flush(buffered)
    state["done"] = sorted(done)
    save_state(state)
    print(f"\nDone. {scraped} meetings scraped this run, {kept:,} rows kept.")


if __name__ == "__main__":
    main()
