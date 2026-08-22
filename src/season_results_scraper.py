"""
season_results_scraper.py -- scrapes every Diamond League circuit meeting's
results each season (not just the Final), giving multiple real rows per
athlete per season instead of the single season-best row a toplist provides.

Why this matters: meets_count/consistency/recent_trend in train_model.py's
build_features()/add_new_features() have been structurally constant (zero
signal) for the whole project, because data/raw/{discipline}.csv (built by
historical_scraper.py) is a World Athletics TOPLIST -- one row per athlete
per season (their single best mark), not a meet-by-meet log.
build_features()/add_new_features() already handle multiple rows per
athlete-season correctly (season_best/career_best take the best across
however many rows exist, consistency takes their std dev, recent_trend looks
at their last 3 marks) -- they just never HAD more than one row per athlete
to work with. This scraper adds the missing rows using the same real WA
GraphQL API dl_final_results_scraper.py already uses, reusing that module's
graphql()/resolve_discipline_key()/GRAPHQL_URL/HEADERS rather than
duplicating them.

Scope: real Diamond League circuit meetings only (rankingCategory "GW" or
"DF" -- excludes each meeting's "F National Events"/"F U20 Events" side
competitions, which aren't the senior international circuit), 2021-2025.
This is NOT every race an athlete ran worldwide that season (that would need
per-athlete profile scraping -- a much bigger undertaking) -- but DL
qualifiers overwhelmingly race multiple DL meetings before the Final, so
this still gives real multi-meeting history for the athletes who matter.

Output rows are appended to data/raw/{discipline}.csv alongside the existing
toplist rows (not replacing them) -- the toplist row for a given athlete-year
already reliably has DOB/nationality even for athletes this scraper's
narrower DL-only view might miss a meeting for, so keeping both is safer
than replacing. A little duplication (a meeting mark that happens to match
the toplist's season-best) is a real value counted twice among several,
not a correctness bug, at the scale a handful of per-season race counts.

Usage:
    python src/season_results_scraper.py
"""
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import dl_final_results_scraper as dlr  # noqa: E402 -- reuse graphql()/resolve_discipline_key()

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
YEARS = range(2021, 2026)  # matches train_model.LABEL_YEARS -- see that module for why not earlier

# "DF" (Diamond Discipline / Final) deliberately excluded: "DF" only appears
# at a season's actual DL Final meeting -- including it here would mean a
# label year's season_best/meets_count/consistency/recent_trend features are
# partly computed FROM that same year's Final result, the exact outcome
# train_model.py's dl_top3 label is trying to predict. Regular-season
# meetings ("GW") already give real, legitimately pre-Final season history.
ELIGIBLE_CATEGORIES = {"GW"}

CALENDAR_QUERY = """query getMinisiteCalendarEvents($competitionGroupId: Int, $competitionSubgroupId: Int, $season: String) {
  getMinisiteCalendarEvents(competitionGroupId: $competitionGroupId, competitionSubgroupId: $competitionSubgroupId, season: $season) {
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


def find_season_meetings(year):
    data = dlr.graphql(
        "getMinisiteCalendarEvents",
        {"season": str(year), "competitionGroupId": 627, "competitionSubgroupId": 0},
        CALENDAR_QUERY,
    )
    return [m for m in data["getMinisiteCalendarEvents"]["results"] if m.get("hasApiResults")]


def scrape_meeting(meeting, year):
    rows = []
    probe = dlr.graphql(
        "getCalendarCompetitionResults",
        {"competitionId": meeting["id"], "day": None, "eventId": None},
        RESULTS_QUERY,
    )["getCalendarCompetitionResults"]
    days = probe["options"]["days"] or [{"day": None, "date": None}]

    for day_info in days:
        data = dlr.graphql(
            "getCalendarCompetitionResults",
            {"competitionId": meeting["id"], "day": day_info["day"], "eventId": None},
            RESULTS_QUERY,
        )["getCalendarCompetitionResults"]
        date_str = day_info.get("date") or meeting.get("startDate")

        for group in data["eventTitles"]:
            if group["rankingCategory"] not in ELIGIBLE_CATEGORIES:
                continue
            for event in group["events"]:
                key = dlr.resolve_discipline_key(event["gender"], event["event"])
                if key is None:
                    continue
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
                    })
        time.sleep(0.3)
    return rows


def scrape_year(year):
    meetings = find_season_meetings(year)
    print(f"  {year}: {len(meetings)} meetings with API results")
    all_rows = []
    for meeting in meetings:
        try:
            rows = scrape_meeting(meeting, year)
        except Exception as e:
            print(f"    {meeting['name']}: ERROR ({e})")
            continue
        print(f"    {meeting['name']}: {len(rows)} rows")
        all_rows.extend(rows)
    return all_rows


if __name__ == "__main__":
    print("=== Scraping real per-meeting season results from World Athletics ===")
    by_discipline = {}
    for year in YEARS:
        for row in scrape_year(year):
            by_discipline.setdefault(row["discipline"], []).append(row)

    for key, rows in sorted(by_discipline.items()):
        new_df = pd.DataFrame(rows).drop(columns=["discipline"])
        new_df["source"] = "dl_meeting"
        path = os.path.join(RAW_DIR, f"{key}.csv")
        if os.path.exists(path):
            existing = pd.read_csv(path)
            # Idempotent: drop this scraper's own rows from a previous run
            # before re-adding fresh ones, so running it twice doesn't
            # double up. Rows from historical_scraper.py (the toplist) never
            # have a "source" column at all -- fillna keeps them as "toplist".
            if "source" not in existing.columns:
                existing["source"] = "toplist"
            existing["source"] = existing["source"].fillna("toplist")
            existing = existing[existing["source"] != "dl_meeting"]
            combined = pd.concat([existing, new_df], ignore_index=True)
        else:
            combined = new_df
        combined.to_csv(path, index=False)
        print(f"  {key}: +{len(new_df)} meeting rows -> {path} ({len(combined)} total rows)")

    print("\nDone.")
