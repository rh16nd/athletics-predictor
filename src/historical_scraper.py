"""
historical_scraper.py — rebuilds data/raw/{discipline}.csv directly from
World Athletics toplists across multiple past seasons (2018-2025), for the
13 disciplines used by src/train_model.py.

Replaces two broken/stale sources:
  - src/scraper.py (Wikipedia scraper) never actually worked -- confirmed
    live it returns 0 rows for every discipline/year it targets, since
    "2024_in_100_metres"-style Wikipedia pages don't exist.
  - The archive.zip / src/extract_new.py path is a static, one-time import
    that caps out at 2023 and can't be refreshed.

Same World Athletics toplist page used by live_fetcher.py for 2026 works
for any past season (confirmed live) -- just swap the year in the URL.
Reuses live_fetcher.scrape_toplist() as-is (single page, no pagination
needed here since this is building a broad training corpus, not matching
specific qualifiers).

Usage:
    python src/historical_scraper.py
"""
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import live_fetcher as lf  # noqa: E402 -- also sets sys.stdout to a UTF-8 wrapper

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

TRAIN_DISCIPLINES = [
    "men_100m", "women_100m", "men_200m", "men_400h", "women_400h", "men_PV",
    "women_200m", "men_800m", "women_800m", "men_1500m", "women_1500m",
    "women_PV", "men_LJ",
    # Added to extend training beyond the original 13 disciplines (see HANDOFF.md
    # Next Steps #1) -- URLs for these already exist in live_fetcher.DISCIPLINE_URLS
    # since they're used for 2026 live predictions, just never had historical data.
    "men_400m", "women_400m", "men_110h", "women_100h",
    "men_5000m", "women_5000m", "men_3000sc", "women_3000sc",
    "men_HJ", "women_HJ", "men_TJ", "women_TJ",
    "men_SP", "women_SP", "men_DT", "women_DT", "men_JT", "women_JT",
    "women_LJ",
]
YEARS = range(2018, 2026)  # 2018-2025 inclusive


def historical_url(discipline_key, year):
    base = lf.DISCIPLINE_URLS[discipline_key]
    return base.rsplit("/", 1)[0] + f"/{year}"


def scrape_discipline_history(key):
    driver = lf.create_driver(headless=True)
    year_dfs = []
    try:
        for year in YEARS:
            url = historical_url(key, year)
            try:
                df = lf.scrape_toplist(driver, url, key, year=year, wait_seconds=8)
            except Exception as e:
                print(f"    {year}: ERROR ({e})")
                continue
            if df.empty:
                print(f"    {year}: no data")
                continue
            print(f"    {year}: {len(df)} rows")
            year_dfs.append(df)
            time.sleep(1)
    finally:
        driver.quit()

    if not year_dfs:
        return pd.DataFrame()
    combined = pd.concat(year_dfs, ignore_index=True)

    # WA's nationality column has no text header in the source table (just a
    # flag icon), so BeautifulSoup captures it as a blank/"Unnamed" column
    # right after DOB. train_model.py's clean_discipline() expects a literal
    # "Nat" column (matching the older data sources this replaces), so name
    # it explicitly here instead of leaving it positional.
    cols = list(combined.columns)
    if "DOB" in cols:
        dob_idx = cols.index("DOB")
        if dob_idx + 1 < len(cols) and cols[dob_idx + 1] != "Nat":
            combined = combined.rename(columns={cols[dob_idx + 1]: "Nat"})

    return combined


NEWLY_ADDED = [
    "men_400m", "women_400m", "men_110h", "women_100h",
    "men_5000m", "women_5000m", "men_3000sc", "women_3000sc",
    "men_HJ", "women_HJ", "men_TJ", "women_TJ",
    "men_SP", "women_SP", "men_DT", "women_DT", "men_JT", "women_JT",
    "women_LJ",
]

if __name__ == "__main__":
    targets = NEWLY_ADDED if "--new-only" in sys.argv else TRAIN_DISCIPLINES
    print("=== Rebuilding historical data/raw/{discipline}.csv from World Athletics ===")
    for key in targets:
        print(f"\n{key}:")
        combined = scrape_discipline_history(key)
        if combined.empty:
            print(f"  WARNING: no data collected for {key}, leaving existing file untouched")
            continue
        out_path = os.path.join(RAW_DIR, f"{key}.csv")
        combined.to_csv(out_path, index=False)
        print(f"  Saved {out_path}: {len(combined)} rows across {combined['year'].nunique()} years")

    print("\nDone.")
