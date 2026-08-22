"""
injury_checker.py — scrapes athletics news sources for injury/withdrawal
mentions of the 2026 DL qualified athletes and writes data/injury_flags.json.

Sources: LetsRun.com, Athletics Weekly, World Athletics news.
Run standalone (after live_fetcher.py has produced data/standings.json):
    python src/injury_checker.py
"""
import io
import json
import os
import re
import sys
import time
from datetime import date, datetime, timezone

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
STANDINGS_PATH = os.path.join(DATA_DIR, "standings.json")
OUTPUT_PATH = os.path.join(DATA_DIR, "injury_flags.json")

# LetsRun and Athletics Weekly both sit behind bot-detection that blocks
# headless Chrome — they need a headful browser. World Athletics is fine
# headless (same as live_fetcher.py).
SOURCES = [
    {"name": "letsrun",         "url": "https://www.letsrun.com/news/",      "headless": False, "wait_seconds": 6},
    {"name": "athleticsweekly", "url": "https://athleticsweekly.com/news/",  "headless": False, "wait_seconds": 6},
    {"name": "worldathletics",  "url": "https://worldathletics.org/news",    "headless": True,  "wait_seconds": 5},
]

# Keyword sets are matched as whole words against lowercased headlines.
# REMOVE = athlete is confirmed out of competition -> drop from predictions.
# WATCH  = injury-adjacent mention that isn't a confirmed withdrawal -> keep
#          in predictions but surface a warning (may still get escalated to
#          REMOVE below if the estimated recovery time won't clear in time
#          for the final).
REMOVE_KEYWORDS = [
    "withdraw", "withdrawn", "withdraws", "pulls out", "pulled out",
    "ruled out", "out for the season", "out for the year", "will not compete",
    "will miss", "scratched", "did not start", "forced to withdraw",
    "ends season", "season is over", "retires", "retirement",
]
GENERIC_WATCH_KEYWORDS = [
    "injury", "injured", "surgery", "setback", "sidelined", "recovering",
    "fitness concern", "doubtful", "injury scare",
]

# Body-part keywords double as (a) a watch trigger on their own -- "Jackson's
# calf tightens up" is injury-adjacent even without the word "injury" -- and
# (b) the input to the recovery-time estimate below. (min_weeks, max_weeks)
# are rough, literature-typical recovery windows for a moderate strain/sprain
# of that body part -- NOT a medical diagnosis. Real recovery time depends
# heavily on grade/severity that vague news text usually doesn't specify,
# which is why these ranges are wide and skew conservative (short end) so we
# don't over-remove on a guess.
INJURY_RECOVERY_WEEKS = {
    "hamstring":       (2, 10),
    "achilles":        (6, 20),
    "calf":            (1, 5),
    "groin":           (2, 6),
    "quad":            (2, 6),
    "quadriceps":      (2, 6),
    "knee":            (3, 12),
    "ankle":           (1, 6),
    "foot":            (4, 10),
    "stress fracture": (6, 12),
    "back":            (2, 8),
    "hip":             (2, 8),
}
WATCH_KEYWORDS = GENERIC_WATCH_KEYWORDS + list(INJURY_RECOVERY_WEEKS.keys())

# Words that shift the estimate up or down regardless of body part.
SEVERITY_UP_WORDS = ["surgery", "torn", "tear", "rupture", "ruptured", "operation"]
SEVERITY_DOWN_WORDS = ["minor", "slight", "small", "tweak", "niggle", "precaution", "tightness"]

FINAL_DATE = date(2026, 9, 4)  # Brussels DL Final

_KEYWORD_RE = {
    kw: re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE)
    for kw in REMOVE_KEYWORDS + WATCH_KEYWORDS
}

_SOURCE_HEADLESS = {s["name"]: s["headless"] for s in SOURCES}
_SOURCE_BASE_URL = {
    "letsrun": "https://www.letsrun.com",
    "athleticsweekly": "https://athleticsweekly.com",
    "worldathletics": "https://worldathletics.org",
}

# Full meet-results recaps (as opposed to narrative news coverage) list every
# athlete's result, including DNF/DNS/DQ -- a much more reliable signal than
# hoping a headline happens to name a specific athlete. Identified by a
# headline containing "results" plus a known DL city.
DL_CITIES = [
    "doha", "shanghai", "suzhou", "shaoxing", "rabat", "florence", "paris",
    "oslo", "lausanne", "stockholm", "silesia", "monaco", "london",
    "zurich", "zürich", "brussels", "eugene", "birmingham", "rome", "xiamen",
]


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
        options=options,
    )


def load_qualified_athletes():
    """Returns {normalized athlete name -> set of discipline keys}."""
    if not os.path.exists(STANDINGS_PATH):
        return {}
    with open(STANDINGS_PATH, encoding="utf-8") as f:
        standings = json.load(f)

    athletes = {}
    for discipline_key, names in standings.items():
        for name in names:
            normalized = " ".join(name.split()).title()
            athletes.setdefault(normalized, set()).add(discipline_key)
    return athletes


def fetch_headlines(source):
    driver = create_driver(headless=source["headless"])
    headlines = []
    try:
        driver.get(source["url"])
        try:
            WebDriverWait(driver, source["wait_seconds"]).until(
                EC.presence_of_element_located((By.TAG_NAME, "a"))
            )
        except Exception:
            pass
        time.sleep(source["wait_seconds"])

        soup = BeautifulSoup(driver.page_source, "html.parser")
        for a in soup.find_all("a", href=True):
            text = a.get_text(strip=True)
            if len(text) < 12:
                continue
            headlines.append({"headline": text, "url": a["href"], "source": source["name"]})
    finally:
        driver.quit()
    return headlines


def match_keywords(headline_lower):
    matched = {"remove": [], "watch": []}
    for kw in REMOVE_KEYWORDS:
        if _KEYWORD_RE[kw].search(headline_lower):
            matched["remove"].append(kw)
    for kw in WATCH_KEYWORDS:
        if _KEYWORD_RE[kw].search(headline_lower):
            matched["watch"].append(kw)
    return matched


def estimate_recovery_weeks(text_lower):
    """Rough recovery-time estimate from injury-adjacent text. This is a
    heuristic, not a medical diagnosis -- vague news text ("hamstring issue")
    rarely gives a grade/severity, so treat the returned range as a wide,
    conservative guess. Returns (min_weeks, max_weeks) for the most severe
    body part matched, or None if no recognized body part is named."""
    best = None
    for body_part, weeks in INJURY_RECOVERY_WEEKS.items():
        if re.search(r"\b" + re.escape(body_part) + r"\b", text_lower):
            if best is None or weeks[1] > best[1]:
                best = weeks
    if best is None:
        return None

    min_w, max_w = best
    if any(re.search(r"\b" + re.escape(w) + r"\b", text_lower) for w in SEVERITY_UP_WORDS):
        min_w, max_w = max(min_w, 12), max(max_w, 26)
    elif any(re.search(r"\b" + re.escape(w) + r"\b", text_lower) for w in SEVERITY_DOWN_WORDS):
        min_w, max_w = max(1, min_w // 2), max(1, max_w // 2)
    return (min_w, max_w)


def upgrade_status(entry, new_status):
    """Status only ever moves toward 'remove', never back down from it."""
    if new_status == "remove":
        entry["status"] = "remove"
    elif entry["status"] != "remove":
        entry["status"] = new_status


def find_results_articles(headlines, max_articles=3):
    """Picks out headlines that look like a full meet-results recap rather
    than an ordinary news story."""
    matches = []
    seen_urls = set()
    for item in headlines:
        headline_lower = item["headline"].lower()
        if "results" not in headline_lower:
            continue
        if not any(city in headline_lower for city in DL_CITIES):
            continue
        if item["url"] in seen_urls:
            continue
        seen_urls.add(item["url"])
        matches.append(item)
        if len(matches) >= max_articles:
            break
    return matches


def fetch_article_text(url, source_name):
    if not url.startswith("http"):
        url = _SOURCE_BASE_URL.get(source_name, "") + url
    driver = create_driver(headless=_SOURCE_HEADLESS.get(source_name, False))
    try:
        driver.get(url)
        time.sleep(6)
        soup = BeautifulSoup(driver.page_source, "html.parser")
        return soup.get_text(" ", strip=True)
    finally:
        driver.quit()


def find_dnf_athletes(article_text, athlete_names):
    """Scans results-article body text for an athlete's name immediately
    followed by DNF, returning {name: recovery_estimate_or_None}. DNS is
    deliberately NOT treated as a signal here -- an athlete can skip a race
    for rest, travel, or scheduling reasons that have nothing to do with
    injury. DQ is ignored too -- a disqualification is a rules call by
    officials and has no bearing on health.

    The recovery estimate looks at a wider window around the name (not just
    the DNF itself) in case the same recap article separately describes what
    happened, e.g. "... DNF ... appeared to pull up with a hamstring issue".
    """
    text_lower = article_text.lower()
    found = {}
    for name in athlete_names:
        name_lower = name.lower()
        idx = text_lower.find(name_lower)
        if idx == -1:
            continue
        dnf_window = text_lower[idx: idx + len(name_lower) + 40]
        if not re.search(r"\bdnf\b", dnf_window):
            continue
        context_window = text_lower[max(0, idx - 100): idx + len(name_lower) + 300]
        found[name] = estimate_recovery_weeks(context_window)
    return found


def check_injuries():
    athletes = load_qualified_athletes()
    if not athletes:
        print("  No qualified athletes found (run live_fetcher.py first). Skipping.")
        result = {"checked_at": datetime.now(timezone.utc).isoformat(), "sources_ok": [], "athletes": {}}
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        return result

    all_headlines = []
    sources_ok = []
    for source in SOURCES:
        print(f"  Scraping {source['name']}...")
        try:
            headlines = fetch_headlines(source)
            print(f"    {len(headlines)} headlines fetched")
            all_headlines.extend(headlines)
            sources_ok.append(source["name"])
        except Exception as e:
            print(f"    WARNING: {source['name']} failed ({e})")

    days_to_final = (FINAL_DATE - date.today()).days

    flags = {}
    for name, discipline_keys in athletes.items():
        name_lower = name.lower()
        for item in all_headlines:
            headline_lower = item["headline"].lower()
            if name_lower not in headline_lower:
                continue
            matched = match_keywords(headline_lower)
            if not matched["remove"] and not matched["watch"]:
                continue
            status = "remove" if matched["remove"] else "watch"

            recovery = estimate_recovery_weeks(headline_lower)
            likely_out_for_final = recovery is not None and (recovery[0] * 7 > days_to_final)
            if likely_out_for_final:
                status = "remove"

            entry = flags.setdefault(name, {
                "status": "watch",
                "disciplines": sorted(discipline_keys),
                "matches": [],
            })
            upgrade_status(entry, status)
            match_record = {
                "headline": item["headline"],
                "url": item["url"],
                "source": item["source"],
                "keywords": matched["remove"] + matched["watch"],
            }
            if recovery is not None:
                match_record["estimated_recovery_weeks"] = list(recovery)
                match_record["likely_out_for_final"] = likely_out_for_final
            entry["matches"].append(match_record)

    # Cross-check qualified athletes against recent full meet-results recaps
    # for a DNF -- catches cases like a mid-race injury that never gets its
    # own headline (it's just a bare "DNF" in a results table).
    results_articles = find_results_articles(all_headlines)
    for article in results_articles:
        print(f"  Checking meet results: {article['headline']}")
        try:
            text = fetch_article_text(article["url"], article["source"])
        except Exception as e:
            print(f"    WARNING: failed to fetch results article ({e})")
            continue
        for name, recovery in find_dnf_athletes(text, athletes.keys()).items():
            likely_out_for_final = recovery is not None and (recovery[0] * 7 > days_to_final)
            status = "remove" if likely_out_for_final else "watch"

            entry = flags.setdefault(name, {
                "status": "watch",
                "disciplines": sorted(athletes[name]),
                "matches": [],
            })
            upgrade_status(entry, status)
            match_record = {
                "headline": article["headline"],
                "url": article["url"],
                "source": f"{article['source']}_results",
                "keywords": ["dnf"],
            }
            if recovery is not None:
                match_record["estimated_recovery_weeks"] = list(recovery)
                match_record["likely_out_for_final"] = likely_out_for_final
            entry["matches"].append(match_record)

    result = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "sources_ok": sources_ok,
        "days_to_final": days_to_final,
        "athletes": flags,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result


def load_injury_flags():
    """Returns {athlete_name -> {status, disciplines, matches}}, or {} if unavailable."""
    if not os.path.exists(OUTPUT_PATH):
        return {}
    try:
        with open(OUTPUT_PATH, encoding="utf-8") as f:
            return json.load(f).get("athletes", {})
    except (json.JSONDecodeError, OSError):
        return {}


if __name__ == "__main__":
    print("=== Checking for athlete injuries / withdrawals ===")
    outcome = check_injuries()
    flagged = outcome["athletes"]
    if flagged:
        print(f"\n  {len(flagged)} athlete(s) flagged:")
        for name, info in flagged.items():
            print(f"    [{info['status'].upper()}] {name} - {info['matches'][0]['headline']}")
    else:
        print("\n  No injury/withdrawal mentions found.")
