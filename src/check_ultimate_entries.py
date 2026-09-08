"""
check_ultimate_entries.py -- has World Athletics put the Ultimate
Championship's entry lists into their API yet?

WHY THIS EXISTS
On 2026-09-07 WA's minisite announced "FINAL ENTRY LISTS PUBLISHED -- 381
athletes from 65 federations" while their competition API still returned
nothing: all 40 phases flagged isStartlistPublished false, and asking for a
start list directly gave a row of nulls. The entry list was a press release,
not data. The championship runs 11-13 September, so the gap between "announced"
and "in the API" is a few days at most, and nobody wants to poll it by hand.

WHAT IT DOES
Asks once. If the entries are there it rebuilds data/ultimate/event.json so the
site has them; if not it says so and changes nothing. It never invents a field
and it never parses the press release -- prose is not a start list.

Exit code is 0 either way. "Not published yet" is a correct answer, not a
failure, and a scheduled run should not look broken for reporting it. Read the
first line of output for the state:

    ENTRIES PUBLISHED: ...    entries are in, event.json rebuilt
    NOT YET: ...              nothing to fetch, nothing written

Usage:
    python src/check_ultimate_entries.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import ultimate_scraper as us  # noqa: E402


def main():
    try:
        startlists = us.fetch_startlists()
    except Exception as e:
        # A network or API failure is not "not published", and saying so would
        # be the same kind of quiet wrong answer this script exists to avoid.
        print(f"CHECK FAILED: {type(e).__name__}: {str(e)[:160]}")
        return 0

    if not startlists:
        try:
            phases = us.fetch_timetable()
        except Exception:
            phases = []
        print(f"NOT YET: World Athletics has {len(phases)} timetable phases for the Ultimate "
              f"and no start lists in the API. Nothing written.")
        return 0

    athletes = sum(len(s.get("entries") or []) for s in startlists)
    event = us.build_event()
    os.makedirs(os.path.dirname(us.OUT_PATH), exist_ok=True)
    with open(us.OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(event, f, ensure_ascii=False, indent=1)

    print(f"ENTRIES PUBLISHED: {athletes} athletes across {len(startlists)} events. "
          f"Rebuilt {os.path.relpath(us.OUT_PATH)}.")
    print("  Next: rebuild the static API and commit both repos "
          "(python src/build_static_api.py), then push when ready.")
    for s in startlists[:6]:
        print(f"    {s.get('sex')} {s.get('discipline')} {s.get('phase')}: "
              f"{len(s.get('entries') or [])} entered")
    return 0


if __name__ == "__main__":
    sys.exit(main())
