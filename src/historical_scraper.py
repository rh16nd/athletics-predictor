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

def merge_toplist_years(path, new_df):
    """Replace ONLY this scraper's own rows for the years it just scraped.

    A plain overwrite is what this script has always done, and it is only safe
    while it scrapes every year the file holds. data/raw/<key>.csv is a THREE
    source file -- toplist rows from here, `dl_meeting` rows from
    season_results_scraper, `major_meet` rows from major_meets_scraper -- so
    overwriting it with a partial year range would silently delete two other
    scrapers' work along with the years not being refreshed.

    Same shape as major_meets_scraper's merge: read what is there, drop the
    rows this run is responsible for, concatenate. Idempotent, so re-running a
    year fixes it rather than duplicating it."""
    if not os.path.exists(path):
        return new_df
    try:
        existing = pd.read_csv(path, low_memory=False)
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        # Returning new_df here would look like a merge and be an overwrite:
        # this file also holds two other scrapers' rows and eight seasons this
        # run did not fetch. Refusing costs one discipline; guessing costs the
        # file. Raised for real once, mid-run, on a file that parsed correctly
        # a minute either side of it -- so it is worth surviving rather than
        # taking the other 17 disciplines down with it.
        raise RuntimeError(
            f"{path} could not be read ({type(exc).__name__}), so its existing "
            f"rows cannot be preserved -- refusing to overwrite. Re-run this "
            f"discipline alone with --only.") from exc
    if "source" not in existing.columns:
        existing["source"] = "toplist"
    existing["source"] = existing["source"].fillna("toplist")

    # WA's toplist table has more than one header-less column (the flag cell,
    # and a spacer), so what comes back can carry the SAME column name twice.
    # pandas.concat raises InvalidIndexError on that rather than picking one,
    # which is the right thing to do -- but it means the duplicates have to be
    # resolved here, before the join, not discovered during it.
    if new_df.columns.duplicated().any():
        dupes = sorted(set(new_df.columns[new_df.columns.duplicated()]))
        print(f"    de-duplicating scraped columns: {dupes}")
        seen = {}
        renamed = []
        for col in new_df.columns:
            if col in seen:
                seen[col] += 1
                renamed.append(f"{col}.{seen[col]}")
            else:
                seen[col] = 0
                renamed.append(col)
        new_df = new_df.copy()
        new_df.columns = renamed
    # Anything the scrape invented that the file has never held is dropped
    # rather than widening the schema for two years out of ten.
    new_df = new_df[[c for c in new_df.columns if c in set(existing.columns)]]

    years = set(new_df["year"].unique())
    stale = (existing["source"] == "toplist") & existing["year"].isin(years)
    kept = existing[~stale]
    print(f"    merging: kept {len(kept)} existing rows, replacing {int(stale.sum())} "
          f"for {sorted(years)}")
    return pd.concat([kept, new_df], ignore_index=True)


if __name__ == "__main__":
    targets = NEWLY_ADDED if "--new-only" in sys.argv else TRAIN_DISCIPLINES

    # --years 2008-2017 scrapes a range other than the default and MERGES
    # rather than overwriting. Added 2026-09-08 to extend the training set
    # backwards: World Athletics serves championship results from 2009, but
    # tags every pre-2018 meeting `rankingCategory: "Pre 2018"`, so
    # major_meets_scraper's "OW" filter had been excluding all of them.
    # Labels are useless without features for the same seasons, which is what
    # this fetches.
    span = None
    for i, arg in enumerate(sys.argv):
        if arg == "--years" and i + 1 < len(sys.argv):
            lo, _, hi = sys.argv[i + 1].partition("-")
            span = range(int(lo), int(hi or lo) + 1)
    if "--only" in sys.argv:
        # Comma-separated, so a partial re-run after a skip is one process.
        # Two of these running at once is what corrupted a read mid-write on
        # 2026-09-08 -- they share every output file. Run one.
        targets = [k.strip() for k in sys.argv[sys.argv.index("--only") + 1].split(",")]
    if span is not None:
        YEARS = span

    failed = []
    print("=== Rebuilding historical data/raw/{discipline}.csv from World Athletics ===")
    print(f"    years: {list(YEARS)}   disciplines: {len(targets)}   "
          f"mode: {'merge' if span is not None else 'overwrite'}")
    for key in targets:
        print(f"\n{key}:")
        combined = scrape_discipline_history(key)
        if combined.empty:
            print(f"  WARNING: no data collected for {key}, leaving existing file untouched")
            continue
        out_path = os.path.join(RAW_DIR, f"{key}.csv")
        if span is not None:
            combined["source"] = "toplist"
            try:
                combined = merge_toplist_years(out_path, combined)
            except RuntimeError as exc:
                # One unreadable file must not end a 32-discipline scrape that
                # takes the better part of an hour. Name it, skip it, carry on.
                print(f"  SKIPPED: {exc}")
                failed.append(key)
                continue
        combined.to_csv(out_path, index=False)
        print(f"  Saved {out_path}: {len(combined)} rows across {combined['year'].nunique()} years")

    if failed:
        print(f"\n{len(failed)} discipline(s) skipped, re-run each with --only: "
              f"{', '.join(failed)}")
    print("\nDone.")
