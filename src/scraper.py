import requests
import pandas as pd
from bs4 import BeautifulSoup
from io import StringIO
import time
import os

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

DISCIPLINES = [
    ("men_100m",   "100 metres",         "men"),
    ("women_100m", "100 metres",         "women"),
    ("men_200m",   "200 metres",         "men"),
    ("men_400h",   "400 metres hurdles", "men"),
    ("women_400h", "400 metres hurdles", "women"),
    ("men_PV",     "pole vault",         "men"),
]

YEARS = [2023, 2024, 2025]

def get_page_html(year, discipline):
    """
    Use Wikipedia's official REST API to get page HTML.
    This is allowed and never returns 403.
    """
    title = f"{year} in {discipline}".replace(" ", "_")
    url = f"https://en.wikipedia.org/api/rest_v1/page/html/{title}"
    headers = {
        "User-Agent": "athletics-predictor/1.0 (student project; contact: student@example.com)",
        "Accept": "text/html"
    }
    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()
    return resp.text, title


def scrape_year(year, discipline, gender, key):
    try:
        html, title = get_page_html(year, discipline)
    except Exception as e:
        print(f"    Error fetching page: {e}")
        return pd.DataFrame()

    soup = BeautifulSoup(html, "html.parser")
    tables = soup.find_all("table")

    if not tables:
        print(f"    No tables found on page: {title}")
        return pd.DataFrame()

    all_rows = []
    for table in tables:
        try:
            # Use pandas to parse each table
            df = pd.read_html(StringIO(str(table)))[0]

            # Flatten multi-level columns if any
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [" ".join(str(c) for c in col).strip() for col in df.columns]

            col_lower = [str(c).lower() for c in df.columns]

            # Only keep tables that look like performance lists
            has_perf = any(k in " ".join(col_lower) for k in ["time", "mark", "height", "distance", "perf"])
            has_athlete = any("athlete" in c for c in col_lower)

            if not (has_perf or has_athlete):
                continue

            df["gender"] = gender
            df["discipline"] = discipline
            df["year"] = year
            all_rows.append(df)

        except Exception:
            continue

    if not all_rows:
        print(f"    No performance tables found for {key} {year}")
        return pd.DataFrame()

    combined = pd.concat(all_rows, ignore_index=True)
    return combined


def collect_all():
    for key, discipline, gender in DISCIPLINES:
        all_years = []
        for year in YEARS:
            print(f"  Fetching {key} — {year}...")
            df = scrape_year(year, discipline, gender, key)
            if not df.empty:
                all_years.append(df)
                print(f"    Got {len(df)} rows")
            time.sleep(1.5)

        if all_years:
            combined = pd.concat(all_years, ignore_index=True)
            out_path = os.path.join(RAW_DIR, f"{key}.csv")
            combined.to_csv(out_path, index=False)
            print(f"  Saved → {out_path}  ({len(combined)} rows)\n")
        else:
            print(f"  No data collected for {key}\n")


if __name__ == "__main__":
    print("Starting data collection via Wikipedia API...\n")
    collect_all()
    print("\nDone! Check data/raw/ for your CSV files.")