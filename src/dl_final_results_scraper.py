"""
dl_final_results_scraper.py -- scrapes REAL Diamond League Final results
directly from World Athletics' own public GraphQL API, replacing the
hand-researched/hand-typed DL_RESULTS list that used to live in
train_model.py. That hardcoded list had real errors (wrong podiums,
wrong athletes) found and fixed by hand across multiple rounds this
project went through -- this script exists so the training labels come
from WA's own systems instead of being retyped by a human or an agent.

How the API was found: WA's site itself calls
https://graphql-prod-4881.edge.aws.worldathletics.org/graphql from the
browser to render its own Calendar/Results pages, authenticated with an
`x-api-key` header. That key is a public AWS AppSync API key shipped in
every visitor's page load (not a secret, not bypassing any login) -- it's
the exact mechanism the site's own frontend uses to fetch this same
public results data. Confirmed via GraphQL introspection (enabled on this
endpoint) which fields/args exist; no fields or types here are guessed.

Two queries, no hardcoded competition IDs or results:
  1. getMinisiteCalendarEvents(season) -- returns every meeting that
     season with a `rankingCategory`. The Final is whichever meeting(s)
     have rankingCategory == "DF" (Diamond Final) -- found by filtering,
     not by a lookup table of "2023 was in Eugene" etc.
  2. getCalendarCompetitionResults(competitionId, day) -- returns every
     event contested at that meeting that day, grouped by rankingCategory.
     Only the "DF" / "Diamond Discipline" group counts toward the Diamond
     Trophy (other events at the same meet are ordinary invitational
     races). Whether a discipline was contested at all that year -- and
     what it was contested as (e.g. "Men's Mile" some years instead of
     "Men's 1500 Metres") -- is read directly from what's present in this
     response. No NOT_CONTESTED list to hand-maintain: absence from this
     scrape *is* the signal.

Scope: only years using the single-meeting Final format are included.
2018 and 2019 split the Final's scoring across two separate meetings
(Zurich + Brussels) -- a top-3-at-one-meeting label doesn't apply
cleanly there. 2020's "Inspiration Games" was a COVID-era exhibition
event, not a real qualifying Final. Both are skipped rather than forced
into a format they don't fit -- see SPLIT_FORMAT_YEARS/SKIP_YEARS below.

Usage:
    python src/dl_final_results_scraper.py
Writes data/dl_final_results.csv: discipline,year,athlete_name,place,mark,nationality
"""
import csv
import os
import sys
import time

import requests

GRAPHQL_URL = "https://graphql-prod-4881.edge.aws.worldathletics.org/graphql"
API_KEY = "da2-wbnmtmvlpbhifh3uc2xaxsue5i"
HEADERS = {"Content-Type": "application/json", "x-api-key": API_KEY}

OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "dl_final_results.csv")

YEARS = range(2021, 2026)  # 2021-2025 inclusive -- see module docstring for why not earlier
SPLIT_FORMAT_YEARS = {2018, 2019}  # two-meeting Final, different scoring model
SKIP_YEARS = {2020}  # COVID-era "Inspiration Games", not a real qualifying Final

CALENDAR_QUERY = """query getMinisiteCalendarEvents($competitionGroupId: Int, $competitionSubgroupId: Int, $season: String) {
  getMinisiteCalendarEvents(competitionGroupId: $competitionGroupId, competitionSubgroupId: $competitionSubgroupId, season: $season) {
    results { id name venue country startDate rankingCategory }
  }
}"""

RESULTS_QUERY = """query getCalendarCompetitionResults($competitionId: Int, $day: Int, $eventId: Int) {
  getCalendarCompetitionResults(competitionId: $competitionId, day: $day, eventId: $eventId) {
    options { days { day } }
    eventTitles {
      rankingCategory
      eventTitle
      events {
        event
        gender
        races {
          race
          results { place mark nationality competitor { name } }
        }
      }
    }
  }
}"""

# WA's own event names -> our internal discipline keys (src/train_model.py's
# TRAIN_DISCIPLINES). This is a name-translation table, not a results table --
# analogous to api.py's DISC_LABELS -- so it needs updating only if WA
# renames an event, never when a result changes.
WA_EVENT_TO_KEY = {
    ("M", "100 Metres"): "men_100m",
    ("M", "200 Metres"): "men_200m",
    ("M", "400 Metres"): "men_400m",
    ("M", "800 Metres"): "men_800m",
    ("M", "1500 Metres"): "men_1500m",
    ("M", "Mile"): "men_1500m",  # some years' Final substitutes the Mile for the 1500m
    ("M", "5000 Metres"): "men_5000m",
    ("M", "110 Metres Hurdles"): "men_110h",
    ("M", "400 Metres Hurdles"): "men_400h",
    ("M", "3000 Metres Steeplechase"): "men_3000sc",
    ("M", "High Jump"): "men_HJ",
    ("M", "Pole Vault"): "men_PV",
    ("M", "Long Jump"): "men_LJ",
    ("M", "Triple Jump"): "men_TJ",
    ("M", "Shot Put"): "men_SP",
    ("M", "Discus Throw"): "men_DT",
    ("M", "Javelin Throw"): "men_JT",
    ("W", "100 Metres"): "women_100m",
    ("W", "200 Metres"): "women_200m",
    ("W", "400 Metres"): "women_400m",
    ("W", "800 Metres"): "women_800m",
    ("W", "1500 Metres"): "women_1500m",
    ("W", "Mile"): "women_1500m",
    ("W", "5000 Metres"): "women_5000m",
    ("W", "100 Metres Hurdles"): "women_100h",
    ("W", "400 Metres Hurdles"): "women_400h",
    ("W", "3000 Metres Steeplechase"): "women_3000sc",
    ("W", "High Jump"): "women_HJ",
    ("W", "Pole Vault"): "women_PV",
    ("W", "Long Jump"): "women_LJ",
    ("W", "Triple Jump"): "women_TJ",
    ("W", "Shot Put"): "women_SP",
    ("W", "Discus Throw"): "women_DT",
    ("W", "Javelin Throw"): "women_JT",
}


def graphql(operation_name, variables, query):
    payload = {"operationName": operation_name, "variables": variables, "query": query}
    r = requests.post(GRAPHQL_URL, json=payload, headers=HEADERS, timeout=20)
    r.raise_for_status()
    body = r.json()
    if "errors" in body:
        raise RuntimeError(f"GraphQL error: {body['errors']}")
    return body["data"]


def find_final_competition_ids(year):
    """Returns the list of competition IDs with rankingCategory == 'DF' for
    a season. Normally exactly one (the single-meeting Final format) --
    callers should treat >1 or 0 as a signal to check SPLIT_FORMAT_YEARS/
    SKIP_YEARS rather than silently guessing which one is "the" Final."""
    data = graphql(
        "getMinisiteCalendarEvents",
        {"season": str(year), "competitionGroupId": 627, "competitionSubgroupId": 0},
        CALENDAR_QUERY,
    )
    results = data["getMinisiteCalendarEvents"]["results"]
    return [r for r in results if r["rankingCategory"] == "DF"]


def scrape_year(year):
    """Returns a list of {discipline, athlete_name, place, mark, nationality}
    dicts for every Diamond Discipline event contested at that year's Final."""
    finals = find_final_competition_ids(year)
    if len(finals) != 1:
        print(f"  {year}: {len(finals)} DF meeting(s) found ({[f['name'] for f in finals]}) -- skipping (not single-meeting format)")
        return []

    comp = finals[0]
    print(f"  {year}: {comp['name']} ({comp['venue']}), competition id {comp['id']}")

    day_data = graphql(
        "getCalendarCompetitionResults",
        {"competitionId": comp["id"], "day": None, "eventId": None},
        RESULTS_QUERY,
    )["getCalendarCompetitionResults"]
    days = [d["day"] for d in day_data["options"]["days"]] or [None]

    rows = []
    seen_events = set()
    for day in days:
        data = graphql(
            "getCalendarCompetitionResults",
            {"competitionId": comp["id"], "day": day, "eventId": None},
            RESULTS_QUERY,
        )["getCalendarCompetitionResults"]

        for group in data["eventTitles"]:
            if group["rankingCategory"] != "DF":
                continue
            for event in group["events"]:
                # WA's event field is the full display name ("Men's 100 Metres"),
                # but the leading "Men's "/"Women's " is redundant with the
                # separate `gender` field -- strip it so WA_EVENT_TO_KEY only
                # needs to spell out the discipline part once.
                event_name = event["event"].split("'s ", 1)[-1]
                key = WA_EVENT_TO_KEY.get((event["gender"], event_name))
                if key is None:
                    continue  # not one of our 32 disciplines (e.g. relays, a non-standard extra race)
                if key in seen_events:
                    continue  # Mile/1500m substitution etc. shouldn't double-count across days
                final_races = [r for r in event["races"] if r["race"] == "Final"]
                if not final_races:
                    continue
                for result in final_races[0]["results"]:
                    name = (result.get("competitor") or {}).get("name")
                    if not name:
                        continue
                    rows.append({
                        "discipline": key,
                        "year": year,
                        "athlete_name": name,
                        "place": result.get("place", "").rstrip("."),
                        "mark": result.get("mark"),
                        "nationality": result.get("nationality"),
                    })
                seen_events.add(key)
        time.sleep(0.5)  # be a reasonable citizen of a public API, even a permissive one

    return rows


if __name__ == "__main__":
    print("=== Scraping real Diamond League Final results from World Athletics ===")
    all_rows = []
    for year in YEARS:
        if year in SPLIT_FORMAT_YEARS or year in SKIP_YEARS:
            print(f"  {year}: skipped (see SPLIT_FORMAT_YEARS/SKIP_YEARS in this file)")
            continue
        try:
            all_rows.extend(scrape_year(year))
        except Exception as e:
            print(f"  {year}: ERROR ({e})")

    disciplines_found = sorted(set(r["discipline"] for r in all_rows))
    print(f"\n  {len(all_rows)} result rows across {len(disciplines_found)} disciplines")

    out_path = os.path.abspath(OUT_PATH)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["discipline", "year", "athlete_name", "place", "mark", "nationality"])
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"  Saved -> {out_path}")
