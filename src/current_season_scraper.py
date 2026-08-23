"""
current_season_scraper.py -- real per-meeting results for the CURRENT,
in-progress Diamond League season, so the athlete profile page's "Real
Season Form" chart can show this year's actual form instead of falling
back to an athlete's last completed season.

Why this is a separate script from season_results_scraper.py rather than
just widening that one's YEARS list: that script deliberately stops at the
last *completed* season (2018-2025, matching train_model.LABEL_YEARS) --
its output feeds the model's training features, and the current season has
no Final result yet to serve as a label. Including it there would be wrong
for that purpose. This script reuses the exact same real scraping mechanism
(find_season_meetings/scrape_meeting from season_results_scraper.py -- same
WA GraphQL API, same "GW"-only regular-season filter, same recognized-
athlete filter) but writes to its own dedicated file,
data/raw/{discipline}_{year}_meetings.csv, so it never touches the
historical training file (data/raw/{discipline}.csv) or the live toplist
snapshot (data/raw/{discipline}_2026.csv, owned/overwritten by
live_fetcher.py) -- no risk of either pipeline clobbering the other.

api.py's load_athlete_history() checks this file first for real in-season
rows before falling back to an athlete's last completed season.

Usage:
    python src/current_season_scraper.py            # current calendar year
    python src/current_season_scraper.py --year 2026 # explicit year
"""
import argparse
import io
import os
import sys
from datetime import date

import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.dirname(__file__))
import dl_final_results_scraper as dlr  # noqa: E402 -- reuse graphql()/load_recognized_names()
from season_results_scraper import find_season_meetings, scrape_meeting  # noqa: E402

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


def scrape_current_season(year):
    meetings = find_season_meetings(year)
    print(f"  {year}: {len(meetings)} meetings with API results")
    by_discipline = {}
    for meeting in meetings:
        try:
            rows = scrape_meeting(meeting, year)
        except Exception as e:
            print(f"    {meeting['name']}: ERROR ({e})")
            continue
        print(f"    {meeting['name']}: {len(rows)} rows")
        for row in rows:
            by_discipline.setdefault(row["discipline"], []).append(row)

    for key, rows in sorted(by_discipline.items()):
        new_df = pd.DataFrame(rows).drop(columns=["discipline"])
        new_df["source"] = "dl_meeting"

        # Same recognized-athlete filter season_results_scraper.py uses --
        # checked against the historical toplist file (data/raw/{key}.csv),
        # which is the right source even for the current season: a real
        # contender almost always has a toplisted history from prior years.
        recognized = dlr.load_recognized_names(key, RAW_DIR)
        before = len(new_df)
        new_df = new_df[new_df["Competitor"].isin(recognized)]
        dropped = before - len(new_df)

        out_path = os.path.join(RAW_DIR, f"{key}_{year}_meetings.csv")
        new_df.to_csv(out_path, index=False)
        print(f"  {key}: {len(new_df)} meeting rows (dropped {dropped} unrecognized) -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=date.today().year)
    args = parser.parse_args()

    print(f"=== Scraping real per-meeting {args.year} season results from World Athletics ===")
    scrape_current_season(args.year)
    print("\nDone.")
