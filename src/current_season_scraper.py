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


def get_recognized_names(key, year, raw_dir):
    """Union of the historical-toplist "recognized" set (dl_final_results_scraper's
    load_recognized_names) with this season's own live toplist snapshot
    (data/raw/{key}_{year}.csv, owned by live_fetcher.py). Split out from
    scrape_current_season() (2026-08-24) so this real bug fix is unit-testable
    without needing a live scrape: the historical-only version silently
    dropped real, currently-ranked athletes with no multi-year history in
    this discipline -- confirmed live for Femke Bol, who switched from 400H
    to the 800m for 2026 and is genuinely ranked #3 in predictions_latest.csv,
    but has zero historical presence in women_800m.csv, so every one of her
    real 2026 races was being filtered out as "unrecognized." Being in this
    season's own live toplist is itself direct proof of current relevance,
    independent of history -- an athlete switching events, returning from a
    long layoff, or breaking out for the first time this year all hit the
    same gap the historical-only check missed."""
    recognized = dlr.load_recognized_names(key, raw_dir)
    current_toplist_path = os.path.join(raw_dir, f"{key}_{year}.csv")
    if os.path.exists(current_toplist_path):
        recognized = recognized | set(pd.read_csv(current_toplist_path)["Competitor"].dropna())
    return recognized


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

        recognized = get_recognized_names(key, year, RAW_DIR)
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
