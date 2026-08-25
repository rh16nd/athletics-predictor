from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import pandas as pd
import time
import os
import re
import json
import sys
import io
from datetime import datetime, timezone
# Guarded: several modules in src/ do this, and each wraps the SAME
# sys.stdout.buffer. With two of them imported into one process the first
# wrapper to be garbage-collected closes the buffer under the second, and
# every later write dies with "I/O operation on closed file" -- which took
# down the whole pytest run on 2026-08-25. After the first wrap the
# encoding is already utf-8, so this becomes a no-op.
if not (sys.stdout.encoding or "").lower().startswith("utf"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

DISCIPLINE_URLS = {
    "men_100m":    "https://worldathletics.org/records/toplists/sprints/100-metres/outdoor/men/senior/2026",
    "women_100m":  "https://worldathletics.org/records/toplists/sprints/100-metres/outdoor/women/senior/2026",
    "men_200m":    "https://worldathletics.org/records/toplists/sprints/200-metres/outdoor/men/senior/2026",
    "women_200m":  "https://worldathletics.org/records/toplists/sprints/200-metres/outdoor/women/senior/2026",
    "men_400m":    "https://worldathletics.org/records/toplists/sprints/400-metres/outdoor/men/senior/2026",
    "women_400m":  "https://worldathletics.org/records/toplists/sprints/400-metres/outdoor/women/senior/2026",
    "men_110h":    "https://worldathletics.org/records/toplists/hurdles/110-metres-hurdles/outdoor/men/senior/2026",
    "women_100h":  "https://worldathletics.org/records/toplists/hurdles/100-metres-hurdles/outdoor/women/senior/2026",
    "men_400h":    "https://worldathletics.org/records/toplists/hurdles/400-metres-hurdles/outdoor/men/senior/2026",
    "women_400h":  "https://worldathletics.org/records/toplists/hurdles/400-metres-hurdles/outdoor/women/senior/2026",
    "men_800m":    "https://worldathletics.org/records/toplists/middlelong/800-metres/outdoor/men/senior/2026",
    "women_800m":  "https://worldathletics.org/records/toplists/middlelong/800-metres/outdoor/women/senior/2026",
    "men_1500m":   "https://worldathletics.org/records/toplists/middlelong/1500-metres/outdoor/men/senior/2026",
    "women_1500m": "https://worldathletics.org/records/toplists/middlelong/1500-metres/outdoor/women/senior/2026",
    "men_5000m":   "https://worldathletics.org/records/toplists/middlelong/5000-metres/outdoor/men/senior/2026",
    "women_5000m": "https://worldathletics.org/records/toplists/middlelong/5000-metres/outdoor/women/senior/2026",
    "men_3000sc":  "https://worldathletics.org/records/toplists/middlelong/3000-metres-steeplechase/outdoor/men/senior/2026",
    "women_3000sc": "https://worldathletics.org/records/toplists/middlelong/3000-metres-steeplechase/outdoor/women/senior/2026",
    "men_HJ":      "https://worldathletics.org/records/toplists/jumps/high-jump/outdoor/men/senior/2026",
    "women_HJ":    "https://worldathletics.org/records/toplists/jumps/high-jump/outdoor/women/senior/2026",
    "men_PV":      "https://worldathletics.org/records/toplists/jumps/pole-vault/outdoor/men/senior/2026",
    "women_PV":    "https://worldathletics.org/records/toplists/jumps/pole-vault/outdoor/women/senior/2026",
    "men_LJ":      "https://worldathletics.org/records/toplists/jumps/long-jump/outdoor/men/senior/2026",
    "women_LJ":    "https://worldathletics.org/records/toplists/jumps/long-jump/outdoor/women/senior/2026",
    "men_TJ":      "https://worldathletics.org/records/toplists/jumps/triple-jump/outdoor/men/senior/2026",
    "women_TJ":    "https://worldathletics.org/records/toplists/jumps/triple-jump/outdoor/women/senior/2026",
    "men_SP":      "https://worldathletics.org/records/toplists/throws/shot-put/outdoor/men/senior/2026",
    "women_SP":    "https://worldathletics.org/records/toplists/throws/shot-put/outdoor/women/senior/2026",
    "men_DT":      "https://worldathletics.org/records/toplists/throws/discus-throw/outdoor/men/senior/2026",
    "women_DT":    "https://worldathletics.org/records/toplists/throws/discus-throw/outdoor/women/senior/2026",
    "men_JT":      "https://worldathletics.org/records/toplists/throws/javelin-throw/outdoor/men/senior/2026",
    "women_JT":    "https://worldathletics.org/records/toplists/throws/javelin-throw/outdoor/women/senior/2026",
}

MEN_TABLE_ORDER = [
    "men_100m", "men_200m", "men_400m", "men_800m",
    "men_1500m", "men_5000m", "men_110h", "men_400h",
    "men_3000sc", "men_HJ", "men_PV", "men_LJ",
    "men_TJ", "men_SP", "men_DT", "men_JT"
]

WOMEN_TABLE_ORDER = [
    "women_100m", "women_200m", "women_400m", "women_800m",
    "women_1500m", "women_5000m", "women_100h", "women_400h",
    "women_3000sc", "women_HJ", "women_PV", "women_LJ",
    "women_TJ", "women_SP", "women_DT", "women_JT"
]

OUR_DISCIPLINES = set(DISCIPLINE_URLS.keys())

FIELD_KEYS = {"men_PV", "women_PV", "men_LJ", "women_LJ", "men_TJ", "women_TJ",
              "men_HJ", "women_HJ", "men_SP", "women_SP", "men_DT", "women_DT", "men_JT", "women_JT"}
LONG_DISTANCE_KEYS = {"men_1500m", "women_1500m", "men_5000m", "women_5000m", "men_3000sc", "women_3000sc"}


def get_qual_limit(discipline_key):
    if discipline_key in FIELD_KEYS:
        return 6
    elif discipline_key in LONG_DISTANCE_KEYS:
        return 10
    return 8


def create_driver(headless=True):
    options = Options()
    if headless:
        options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--log-level=3")
    options.add_argument("--window-size=1920,1080")
    return webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options
    )


def scrape_toplist(driver, url, discipline, year=2026, wait_seconds=8, required_athletes=None, max_pages=5):
    """Scrapes the top-100 page, then keeps paging (?page=2, 3, ...) only if
    required_athletes (DL-qualified names for this discipline) aren't all in
    yet — most disciplines stop after page 1. Caps at max_pages (top ~500)
    since a qualifier who placed on points rather than raw mark may not have
    a top-500 time/mark at all, and isn't worth an unbounded search for."""
    required_athletes = set(required_athletes or [])
    all_rows = []
    all_profile_urls = []
    headers = None
    found = set()

    for page in range(1, max_pages + 1):
        page_url = url if page == 1 else f"{url}?page={page}"
        driver.get(page_url)
        try:
            WebDriverWait(driver, wait_seconds).until(
                EC.presence_of_element_located((By.TAG_NAME, "table"))
            )
        except Exception:
            driver.execute_script("window.scrollTo(0, 500)")
            time.sleep(3)

        soup = BeautifulSoup(driver.page_source, "html.parser")
        table = soup.find("table")
        if not table:
            break

        if headers is None:
            headers = [th.get_text(strip=True) for th in table.find_all("th")]

        trs = [tr for tr in table.find_all("tr")[1:] if tr.find_all("td")]
        rows = [[td.get_text(strip=True) for td in tr.find_all("td")] for tr in trs]
        if not rows:
            break

        # The competitor-name cell links straight to their WA profile
        # (/athletes/athlete={id}), which resolves directly to the real
        # profile page -- no need to construct the "pretty" slug URL.
        for tr in trs:
            link = tr.find("a", href=True)
            all_profile_urls.append(
                "https://worldathletics.org" + link["href"] if link and link["href"].startswith("/")
                else (link["href"] if link else None)
            )

        all_rows.extend(rows)

        if headers and "Competitor" in headers:
            comp_idx = headers.index("Competitor")
            found.update(r[comp_idx] for r in rows if comp_idx < len(r))

        missing = required_athletes - found
        if not missing:
            break
        if page > 1:
            print(f"    page {page}: still missing {sorted(missing)}")

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows, columns=headers[:len(all_rows[0])])
    df["discipline"] = discipline
    df["year"] = year
    df["ProfileURL"] = all_profile_urls
    return df


STANDINGS_HEADER_WORDS = ("METRES", "JUMP", "VAULT", "PUT", "THROW",
                          "HURDLES", "STEEPLECHASE", "MILE")


def _as_number(text):
    """Points/events cells as numbers, or None. Kept tolerant on purpose:
    a blank or a dash in one cell must not drop the whole row."""
    text = (text or "").strip().replace(",", "")
    if not text:
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    return int(value) if value.is_integer() else value


def parse_standings_table(table):
    """Every row of one WA Diamond League standings table, in full.

    The columns are [rank, athlete, country, events, points]. This scraper
    used to keep only the name and truncate at the qualification cut, which
    threw away the two things needed to answer "who can still qualify?" --
    the points, and every row below the line.

    Header rows are dropped by requiring a numeric rank rather than by
    matching the header's text: that text renders uppercase via CSS, so what
    BeautifulSoup sees is a styling detail, not a stable contract."""
    rows = []
    for tr in table.find_all("tr"):
        cells = [c.get_text(strip=True) for c in tr.find_all(["td", "th"])]
        if len(cells) < 5:
            continue
        rank, name, country, events, points = cells[:5]
        if not rank.isdigit() or not name:
            continue
        rows.append({
            "rank": int(rank),
            "name": name,
            "country": country or None,
            "events": _as_number(events),
            "points": _as_number(points),
        })
    return rows


def standings_table_label(table):
    """The discipline heading WA prints above a standings table ("100
    METRES"). Only used to sanity-check that the Nth table really is the Nth
    discipline in our own table order -- reading tables positionally has
    silently mismatched a discipline before, and a warning is cheap."""
    text = table.find_previous(string=re.compile("|".join(STANDINGS_HEADER_WORDS), re.I))
    return " ".join(str(text).split()) if text else None


KEY_LABEL_TOKENS = {
    "100m": "100 METRES", "200m": "200 METRES", "400m": "400 METRES",
    "800m": "800 METRES", "1500m": "1500 METRES", "5000m": "5000 METRES",
    "110h": "110 METRES HURDLES", "100h": "100 METRES HURDLES",
    "400h": "400 METRES HURDLES", "3000sc": "3000 METRES STEEPLECHASE",
    "HJ": "HIGH JUMP", "PV": "POLE VAULT", "LJ": "LONG JUMP",
    "TJ": "TRIPLE JUMP", "SP": "SHOT PUT", "DT": "DISCUS THROW",
    "JT": "JAVELIN THROW",
}


def label_matches_key(label, discipline_key):
    """Is WA's own heading above the Nth table the Nth discipline we expect?

    The standings tables are read positionally against MEN_TABLE_ORDER /
    WOMEN_TABLE_ORDER, and a positional read has silently mismatched a
    discipline in this project before. This only warns -- skipping a table on
    a failed match would lose a whole discipline over a wording change."""
    if not label:
        return True
    expected = KEY_LABEL_TOKENS.get(discipline_key.split("_", 1)[-1])
    if not expected:
        return True
    label = " ".join(label.upper().split())
    if expected not in label:
        return False
    # "400 METRES" is a substring of "400 METRES HURDLES", so containment
    # alone would read the hurdles table as the flat 400. Anything left over
    # must be numeric -- WA lists the 5000m as "3000/5000 METRES".
    leftover = label.replace(expected, " ").replace("/", " ").split()
    return all(word.isdigit() for word in leftover)


def scrape_dl_standings(driver, year=2026, wait_seconds=8):
    STANDINGS_URLS = {
        "men":   (f"https://worldathletics.org/competitions/diamond-league/standings/{year}/men", MEN_TABLE_ORDER),
        "women": (f"https://worldathletics.org/competitions/diamond-league/standings/{year}/women", WOMEN_TABLE_ORDER),
    }

    qualified = {}
    detail = {}

    for gender_key, (url, table_order) in STANDINGS_URLS.items():
        print(f"Loading DL standings ({gender_key})...")
        driver.get(url)
        time.sleep(wait_seconds)

        soup = BeautifulSoup(driver.page_source, "html.parser")
        tables = soup.find_all("table")

        for i, table in enumerate(tables):
            if i >= len(table_order):
                break
            discipline_key = table_order[i]
            if discipline_key not in OUR_DISCIPLINES:
                continue

            # Sorted by WA's own rank column, NOT by the order the rows
            # happen to render in. Those two disagree inside a group tied on
            # points -- WA ranks the tie, but lays the rows out some other
            # way -- and truncating at the qualification cut by row order
            # therefore kept the wrong athletes. Live on 2026-08-25 in the
            # men's 1500m: three athletes tied on 8 points rendered as Kerr
            # (rank 10), Farken (11), Habz (9), so `athletes[:10]` qualified
            # Farken and dropped Habz, who is genuinely 9th.
            rows = sorted(parse_standings_table(table), key=lambda r: r["rank"])
            athletes = [r["name"] for r in rows]
            label = standings_table_label(table)
            if not label_matches_key(label, discipline_key):
                print(f"  WARNING: table {i} reads as {discipline_key} but WA "
                      f"heads it \"{label}\" -- check the table order")

            limit = get_qual_limit(discipline_key)
            qualified[discipline_key] = athletes[:limit]
            detail[discipline_key] = {
                "qualLimit": limit,
                "waLabel": label,
                "standings": rows,
            }
            print(f"  {discipline_key}: {len(athletes[:limit])} qualified athletes"
                  f" ({len(rows)} in the standings)")

    standings_path = os.path.join(RAW_DIR, "..", "standings.json")
    with open(standings_path, "w", encoding="utf-8") as f:
        json.dump(qualified, f, ensure_ascii=False, indent=2)

    # Written alongside standings.json rather than replacing it: run.py and
    # api.py both read that file's list-of-names shape, and the qualification
    # race is a separate question from "who is in the field".
    detail_path = os.path.join(RAW_DIR, "..", "standings_detail.json")
    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump({
            "year": year,
            "scrapedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "disciplines": detail,
        }, f, ensure_ascii=False, indent=2)

    return qualified


MIDDLE_DISTANCE = {
    "men_800m", "women_800m", "men_1500m", "women_1500m",
    "women_200m", "men_5000m", "women_5000m",
    "men_3000sc", "women_3000sc"
}

if __name__ == "__main__":
    driver = create_driver(headless=True)
    print("=== Scraping DL standings ===")
    standings = scrape_dl_standings(driver)
    driver.quit()

    print("\n=== Scraping 2026 top lists ===")
    for key, url in DISCIPLINE_URLS.items():
        print(f"Fetching {key}...")
        use_headless = key not in MIDDLE_DISTANCE
        driver = create_driver(headless=use_headless)
        wait = 15 if key in MIDDLE_DISTANCE else 8
        df = scrape_toplist(driver, url, key, wait_seconds=wait, required_athletes=standings.get(key, []))
        driver.quit()

        if not df.empty:
            out_path = os.path.join(RAW_DIR, f"{key}_2026.csv")
            df.to_csv(out_path, index=False)
            print(f"  Saved {len(df)} rows")
        else:
            print(f"  No data for {key}")
        time.sleep(1)

    print("\n=== DL Qualified athletes ===")
    for key, athletes in standings.items():
        print(f"  {key}: {athletes[:5]}")