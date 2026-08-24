"""
major_meets_scraper.py -- adds real per-meeting results from the handful of
non-Diamond-League meetings that genuinely produce big results: the Olympic
Games, the (senior outdoor) World Athletics Championships, World Athletics
Continental Tour Gold meetings, and the European Athletics Championships
(the only one of the six Area Senior Outdoor Championships included --
user-scoped down from all six after seeing quality varies a lot by
continent). This is the user-requested middle ground between
season_results_scraper.py (Diamond League circuit only) and scraping every
meeting worldwide (a much bigger, noisier undertaking) -- these four groups
are exactly the meetings World Athletics' own calendar already tags as its
top competition tiers, not a hand-picked guess.

How the groups were chosen: WA's getCalendarEvents query accepts a
competitionGroupId and exposes the full list of groups via its own
`options.competitionGroups` field (queried via introspection, not guessed).
Four groups were selected as the meetings actually known for producing
season/career bests beyond the Diamond League circuit:
  - Olympic Games (id 5) -- ~1/year in Olympic years only.
  - World Athletics Series (id 3806) -- a broader group that also contains
    U20/Indoor/Cross Country/Road Running/Relay/Combined-Events championships
    in the same season; NAME_EXCLUDE below filters down to just the senior
    outdoor World Athletics Championships.
  - World Athletics Continental Tour (id 3773) -- this group is mostly
    Silver/Bronze/Challenger-level meetings (hundreds/year, not "big result"
    venues); filtered to rankingCategory == "A" (Gold tier) only, confirmed
    via a live query to be the well-known near-DL meets (FBK Games, Paavo
    Nurmi Games, Boris Hanzekovic Memorial, etc.), ~5/year.
  - Area Senior Outdoor Championships (id 3660) -- filtered to the European
    Athletics Championships only (name contains "European"); the other five
    (African/Asian/American/Oceania/NACAC) are skipped by user choice.

No rankingCategory filtering is applied to the *results within* a chosen
meeting (unlike season_results_scraper.py's GW-only DL filter) -- that filter
exists there specifically because DL's "DF" group IS that year's Final (the
prediction target), so including it would leak the label into training
features. None of these four groups is ever the Diamond League Final, so
every event/race within them is fair game.

Rows are filtered to athletes ever present in that discipline's own toplist
(any year -- see dl_final_results_scraper.load_recognized_names) before
being written: a big meeting can still have a weak field in a discipline
that isn't its main draw (e.g. a field-filler making a Continental Tour
Gold meeting's javelin final), and that athlete's marks add noise rather
than real signal about the athletes DL Final prediction actually cares
about.

Usage:
    python src/major_meets_scraper.py
"""
import os
import sys
import time
import io

import pandas as pd

# Some meeting names (e.g. Polish/Croatian Continental Tour Gold meets) use
# characters outside Windows' default cp1252 console encoding -- without this,
# printing them crashes the scrape partway through instead of just showing
# the name correctly.
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

sys.path.insert(0, os.path.dirname(__file__))
import dl_final_results_scraper as dlr  # noqa: E402 -- reuse graphql()/resolve_discipline_key()

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
YEARS = [y for y in range(2018, 2026) if y != 2020]  # matches train_model.LABEL_YEARS

OLYMPIC_GAMES = 5
WORLD_ATHLETICS_SERIES = 3806
CONTINENTAL_TOUR = 3773
AREA_CHAMPIONSHIPS = 3660

# World Athletics Series also contains U20/indoor/cross-country/road-running/
# relay/combined-events championships in the same season -- keep only the
# senior outdoor "World Athletics Championships" / "World Championships in
# Athletics" meeting itself.
SERIES_NAME_EXCLUDE = ["U20", "Indoor", "Cross Country", "Race Walking", "Road Running", "Relays", "Combined Events"]

CALENDAR_QUERY = """query getCalendarEvents($startDate: String, $endDate: String, $competitionGroupId: Int) {
  getCalendarEvents(startDate: $startDate, endDate: $endDate, competitionGroupId: $competitionGroupId, limit: 100) {
    results { id name rankingCategory startDate hasApiResults }
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
          results { place mark nationality competitor { name birthDate } }
        }
      }
    }
  }
}"""


def find_meetings(year, group_id):
    data = dlr.graphql(
        "getCalendarEvents",
        {"startDate": f"{year}-01-01", "endDate": f"{year}-12-31", "competitionGroupId": group_id},
        CALENDAR_QUERY,
    )
    return [m for m in data["getCalendarEvents"]["results"] if m.get("hasApiResults")]


def find_year_meetings(year):
    """Returns every target meeting for a season across all four groups,
    each tagged with which group it came from (for logging only)."""
    meetings = []

    for m in find_meetings(year, OLYMPIC_GAMES):
        meetings.append((m, "Olympics"))

    for m in find_meetings(year, WORLD_ATHLETICS_SERIES):
        if any(kw.lower() in m["name"].lower() for kw in SERIES_NAME_EXCLUDE):
            continue
        meetings.append((m, "World Champs"))

    for m in find_meetings(year, CONTINENTAL_TOUR):
        if m.get("rankingCategory") != "A":
            continue
        meetings.append((m, "Continental Tour Gold"))

    for m in find_meetings(year, AREA_CHAMPIONSHIPS):
        if "european" not in m["name"].lower():
            continue
        meetings.append((m, "European Champs"))

    return meetings


def scrape_meeting(meeting, year):
    rows = []
    probe = dlr.graphql(
        "getCalendarCompetitionResults",
        {"competitionId": meeting["id"], "day": None, "eventId": None},
        RESULTS_QUERY,
    )["getCalendarCompetitionResults"]
    days = probe["options"]["days"] or [{"day": None, "date": None}]

    seen_events = set()
    for day_info in days:
        data = dlr.graphql(
            "getCalendarCompetitionResults",
            {"competitionId": meeting["id"], "day": day_info["day"], "eventId": None},
            RESULTS_QUERY,
        )["getCalendarCompetitionResults"]
        date_str = day_info.get("date") or meeting.get("startDate")

        for group in data["eventTitles"]:
            for event in group["events"]:
                # No mile_as_1500 here, deliberately: these rows become a
                # per-meeting time series, where a Mile is not a 1500m.
                # See dlr.MILE_AS_1500_KEY.
                key = dlr.resolve_discipline_key(event["gender"], event["event"])
                if key is None:
                    continue
                event_key = (key, event["event"])
                if event_key in seen_events:
                    continue  # a multi-day champs can repeat an event listing across day queries
                final_races = [r for r in event["races"] if r["race"] == "Final"]
                if not final_races:
                    continue
                for result in final_races[0]["results"]:
                    competitor = result.get("competitor") or {}
                    name = competitor.get("name")
                    if not name:
                        continue
                    rows.append({
                        "discipline": key,
                        "Competitor": name,
                        "DOB": competitor.get("birthDate"),
                        "Mark": result.get("mark"),
                        "Nat": result.get("nationality"),
                        "Venue": meeting.get("name"),
                        "Date": date_str,
                        "year": year,
                        # WA's own response already carries `place` -- this
                        # scraper only ever reads the "Final" race (see the
                        # final_races filter above), so it's always a real
                        # final-round placement, never a heat/semi. Was
                        # fetched but silently dropped until 2026-08-23; see
                        # train_model.py's parse_pos() for what consumes this.
                        "Pos": result.get("place"),
                    })
                seen_events.add(event_key)
        time.sleep(0.3)
    return rows


def scrape_year(year):
    meetings = find_year_meetings(year)
    print(f"  {year}: {len(meetings)} major meetings found")
    all_rows = []
    for meeting, group_label in meetings:
        try:
            rows = scrape_meeting(meeting, year)
        except Exception as e:
            print(f"    [{group_label}] {meeting['name']}: ERROR ({e})")
            continue
        print(f"    [{group_label}] {meeting['name']}: {len(rows)} rows")
        all_rows.extend(rows)
    return all_rows


if __name__ == "__main__":
    print("=== Scraping major non-DL meetings (Olympics/Worlds/Continental Tour Gold/Area Champs) ===")
    by_discipline = {}
    for year in YEARS:
        for row in scrape_year(year):
            by_discipline.setdefault(row["discipline"], []).append(row)

    for key, rows in sorted(by_discipline.items()):
        new_df = pd.DataFrame(rows).drop(columns=["discipline"])
        new_df["source"] = "major_meet"

        # Only keep rows for athletes ever toplisted in this discipline --
        # drops field-fillers who made a major meeting's final in a discipline
        # they weren't actually competitive in that season (see
        # season_results_scraper.py's docstring / dl_final_results_scraper.load_recognized_names
        # for the full reasoning -- same filter, same rationale).
        recognized = dlr.load_recognized_names(key, RAW_DIR)
        before = len(new_df)
        new_df = new_df[new_df["Competitor"].isin(recognized)]
        dropped = before - len(new_df)

        path = os.path.join(RAW_DIR, f"{key}.csv")
        if os.path.exists(path):
            existing = pd.read_csv(path)
            # Idempotent: drop this scraper's own rows from a previous run
            # before re-adding fresh ones, same pattern as season_results_scraper.py.
            if "source" not in existing.columns:
                existing["source"] = "toplist"
            existing["source"] = existing["source"].fillna("toplist")
            existing = existing[existing["source"] != "major_meet"]
            combined = pd.concat([existing, new_df], ignore_index=True)
        else:
            combined = new_df
        combined.to_csv(path, index=False)
        print(f"  {key}: +{len(new_df)} major-meet rows (dropped {dropped} unrecognized) -> {path} ({len(combined)} total rows)")

    print("\nDone.")
