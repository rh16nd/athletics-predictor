"""
refresh_final_results.py -- scrapes the CURRENT year's Diamond League Final
results into data/dl_final_<year>_results.csv, for the site's "result vs
prediction" comparison.

This is deliberately SEPARATE from dl_final_results_scraper.py's own output
(data/dl_final_results.csv), which holds the 2018-2025 training labels. The
two must never share a file: the training file feeds train_model.py, and a
half-finished live meeting (day 1 run, day 2 pending) has no business in the
model's labels. This script reuses that module's scrape_year() -- the exact
same World Athletics GraphQL path, so the live comparison and the training
labels are read the same way -- and only changes where the rows land.

Absence is the signal, same as the training scraper: a discipline whose Final
has not been contested yet simply produces no rows, so the site shows its
projection untouched until the race is actually run.

Usage:
    python src/refresh_final_results.py [year]   # year defaults to 2026
Writes data/dl_final_<year>_results.csv: discipline,year,athlete_name,place,mark,nationality
"""
import csv
import os
import sys

from dl_final_results_scraper import scrape_year

DEFAULT_YEAR = 2026


def out_path(year):
    return os.path.join(
        os.path.dirname(__file__), "..", "data", f"dl_final_{year}_results.csv"
    )


def refresh(year):
    rows = scrape_year(year)
    disciplines = sorted(set(r["discipline"] for r in rows))
    print(f"  {len(rows)} result rows across {len(disciplines)} disciplines")

    path = os.path.abspath(out_path(year))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["discipline", "year", "athlete_name", "place", "mark", "nationality"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved -> {path}")
    return rows


if __name__ == "__main__":
    year = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_YEAR
    print(f"=== Scraping {year} Diamond League Final results (live) ===")
    refresh(year)
