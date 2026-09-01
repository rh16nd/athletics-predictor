"""
api.py — Flask bridge between run.py predictions and the React dashboard
Run with: python api.py
Serves at: http://localhost:5000
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
import cv2
import requests
import json
import os
import re
import sys
from datetime import date, datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import dl_final_results_scraper as dlr  # noqa: E402 -- reuse the same graphql()/HEADERS every other scraper does
from feature_builder import get_qual_limit  # noqa: E402 -- one definition of the field size, shared with run.py
import athlete_analytics  # noqa: E402 -- race-log statistics; reads data/worldwide, never feeds the model
import athlete_career  # noqa: E402 -- honours/rankings/PBs as World Athletics states them; never feeds the model

app = Flask(__name__)
CORS(app)  # allows React dev server to call this API

OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "outputs")
RAW_DIR     = os.path.join(os.path.dirname(__file__), "data", "raw")
INJURY_FLAGS_PATH = os.path.join(os.path.dirname(__file__), "data", "injury_flags.json")
H2H_PATH    = os.path.join(os.path.dirname(__file__), "data", "h2h", "h2h_rates.csv")
DL_FINAL_RESULTS_PATH = os.path.join(os.path.dirname(__file__), "data", "dl_final_results.csv")
MODELS_DIR  = os.path.join(os.path.dirname(__file__), "data", "models")
FOCUS_CACHE_PATH = os.path.join(os.path.dirname(__file__), "data", "photo_focus_cache.json")

MEETS_YEAR = 2026


def load_injury_flags():
    """Returns {normalized athlete name -> {status, disciplines, matches}}, or {}."""
    if not os.path.exists(INJURY_FLAGS_PATH):
        return {}
    try:
        with open(INJURY_FLAGS_PATH, encoding="utf-8") as f:
            return json.load(f).get("athletes", {})
    except (json.JSONDecodeError, OSError):
        return {}


def normalize_athlete_name(name):
    """Matches run.py's/injury_checker.py's normalization so predictions_latest.csv's
    ALL-CAPS-surname names ('Shericka JACKSON') can be looked up against
    injury_flags.json's Title-case keys ('Shericka Jackson')."""
    return " ".join(str(name).split()).title()


def injury_evidence(entry):
    """Pulls a short, human-readable reason + source link from an injury_flags.json
    entry's most recent match -- without this, the dashboard only ever showed a
    generic 'flagged for review' tooltip even though the real headline/URL was
    sitting right there in the file the whole time."""
    matches = (entry or {}).get("matches") or []
    if not matches:
        return None, None
    m = matches[-1]
    headline = m.get("headline")
    source = (m.get("source") or "").replace("_results", "")
    reason = f'"{headline}" ({source})' if headline else None
    return reason, m.get("url")

# "status" here is just the meet's calendar position, not literal — done/next/upcoming
# are recomputed against today's date on every request (see compute_meet_statuses),
# so this never needs manual updating as the season progresses. Only the last entry's
# "final" label is authoritative (it marks the championship meet, not a point in time).
# Reconciled against World Athletics' own 2026 Diamond League calendar on
# 2026-08-29 (getMinisiteCalendarEvents, competitionGroupId 627 -- the same
# query season_results_scraper.py uses), because a hand-typed version of it
# had drifted: it opened the season with "08 May Doha", a meeting that does
# not exist. Doha actually ran 19 Jun, the opener was 16 May in
# Shaoxing/Keqiao (listed as "Shanghai"), and Paris/Eugene were each a day
# or two out. Those five rows were rendering on the Schedule page as fact.
# `dateEnd` is set for the three genuinely two-day meetings rather than
# picking one of their days arbitrarily -- the old list showed day 2 for
# Lausanne/Silesia/Zürich and day 1 for everything else.
# `tests/test_dl_calendar.py` pins this list to a committed snapshot of WA's
# calendar (tests/fixtures/wa_dl_calendar_2026.json, regenerate with
# `python src/dl_calendar.py --snapshot`) so it cannot drift again silently.
MEETS = [
    {"n": 1,  "date": "16 May", "city": "Shaoxing/Keqiao"},
    {"n": 2,  "date": "23 May", "city": "Xiamen"},
    {"n": 3,  "date": "31 May", "city": "Rabat"},
    {"n": 4,  "date": "04 Jun", "city": "Rome"},
    {"n": 5,  "date": "07 Jun", "city": "Stockholm"},
    {"n": 6,  "date": "10 Jun", "city": "Oslo"},
    {"n": 7,  "date": "19 Jun", "city": "Doha"},
    {"n": 8,  "date": "28 Jun", "city": "Paris"},
    {"n": 9,  "date": "03 Jul", "dateEnd": "04 Jul", "city": "Eugene"},
    {"n": 10, "date": "10 Jul", "city": "Monaco"},
    {"n": 11, "date": "18 Jul", "city": "London"},
    {"n": 12, "date": "20 Aug", "dateEnd": "21 Aug", "city": "Lausanne"},
    {"n": 13, "date": "22 Aug", "dateEnd": "23 Aug", "city": "Silesia"},
    {"n": 14, "date": "26 Aug", "dateEnd": "27 Aug", "city": "Zürich"},
    # `final` is explicit rather than "whichever entry is last". The Diamond
    # League Final is not always one meeting: in 2018 and 2019 it was SPLIT
    # across Zürich and Brussels, two separate meetings both carrying
    # World Athletics' "DF" ranking category (dl_final_results_scraper.py
    # already aggregates them, which is why its find_final_competition_ids
    # returns a list). Under the old last-index rule, the first leg of a
    # split Final would have been scored as a qualifying meeting -- so
    # meetings_remaining() would have claimed points were still winnable at
    # a Final. 2026 is a single Final (verified against WA: Zürich 26-27 Aug
    # is "GW", every one of its event groups included; only Brussels is
    # "DF"), but the flag costs nothing and removes the trap.
    {"n": 15, "date": "04 Sep", "city": "Brussels — Final", "final": True},
]


def _meet_date(value):
    """A "%d %b" string in MEETS_YEAR, or None if it isn't one."""
    try:
        return datetime.strptime(f"{value} {MEETS_YEAR}", "%d %b %Y").date()
    except (ValueError, TypeError):
        return None


def compute_meet_statuses(meets, today=None):
    today = today or date.today()
    result = []
    next_assigned = False
    # Fall back to "the last entry is the Final" only when nothing declares
    # itself -- that is what the shape used to mean, and callers (including
    # tests) still pass bare lists.
    flagged = any(m.get("final") for m in meets)
    last_index = len(meets) - 1

    for i, meet in enumerate(meets):
        if meet.get("final") if flagged else i == last_index:
            result.append({**meet, "status": "final"})
            continue
        # A two-day meeting is not over on its first morning, so "done" is
        # judged on the last day it is actually contested.
        meet_date = _meet_date(meet.get("dateEnd")) or _meet_date(meet.get("date"))
        if meet_date is None:
            result.append({**meet, "status": "upcoming"})
            continue

        if meet_date < today:
            status = "done"
        elif not next_assigned:
            status = "next"
            next_assigned = True
        else:
            status = "upcoming"
        result.append({**meet, "status": status})

    return result

FIELD_EVENTS = {
    "men_PV", "women_PV", "men_LJ", "women_LJ",
    "men_TJ", "women_TJ", "men_HJ", "women_HJ",
    "men_SP", "women_SP", "men_DT", "women_DT",
    "men_JT", "women_JT"
}
MIDDLE_DISTANCE = {
    "men_800m", "women_800m", "men_1500m", "women_1500m",
    "men_5000m", "women_5000m", "men_3000sc", "women_3000sc",
}

DISC_LABELS = {
    "men_100m":    "Men's 100m",
    "women_100m":  "Women's 100m",
    "men_200m":    "Men's 200m",
    "women_200m":  "Women's 200m",
    "men_400m":    "Men's 400m",
    "women_400m":  "Women's 400m",
    "men_110h":    "Men's 110m Hurdles",
    "women_100h":  "Women's 100m Hurdles",
    "men_400h":    "Men's 400m Hurdles",
    "women_400h":  "Women's 400m Hurdles",
    "men_800m":    "Men's 800m",
    "women_800m":  "Women's 800m",
    "men_1500m":   "Men's 1500m",
    "women_1500m": "Women's 1500m",
    "men_5000m":   "Men's 5000m",
    "women_5000m": "Women's 5000m",
    "men_3000sc":  "Men's 3000m Steeplechase",
    "women_3000sc":"Women's 3000m Steeplechase",
    "men_HJ":      "Men's High Jump",
    "women_HJ":    "Women's High Jump",
    "men_PV":      "Men's Pole Vault",
    "women_PV":    "Women's Pole Vault",
    "men_LJ":      "Men's Long Jump",
    "women_LJ":    "Women's Long Jump",
    "men_TJ":      "Men's Triple Jump",
    "women_TJ":    "Women's Triple Jump",
    "men_SP":      "Men's Shot Put",
    "women_SP":    "Women's Shot Put",
    "men_DT":      "Men's Discus Throw",
    "women_DT":    "Women's Discus Throw",
    "men_JT":      "Men's Javelin Throw",
    "women_JT":    "Women's Javelin Throw",
}

def parse_mark(m):
    m = str(m).strip()
    if m.endswith("m"):
        return float(m[:-1])
    if ":" in m:
        parts = m.split(":")
        return float(parts[0]) * 60 + float(parts[1])
    return float(m)

def safe_parse_mark(m):
    """parse_mark, but None instead of an exception for the non-marks that
    real result sets are full of: "NM"/"NH" (no valid attempt), "DNS",
    "DNF", "DQ". Those are outcomes, not bad data -- they just have no
    number to plot."""
    try:
        return parse_mark(m)
    except (ValueError, TypeError):
        return None


def format_mark(val, disc):
    if disc in FIELD_EVENTS:
        return f"{val:.2f}m"
    if disc in MIDDLE_DISTANCE:
        mins = int(val // 60)
        secs = val % 60
        return f"{mins}:{secs:05.2f}"
    return f"{val:.2f}"


def _season_rows_to_history(season, disc_key):
    """Shared row->dict conversion for load_athlete_history()'s two sources
    (the current-season meetings file and the historical toplist file) --
    same real-race dedup logic either way: the same race can appear twice
    (once from a toplist scrape, once from a per-meeting scraper under a
    differently-formatted venue name), which is an accepted duplication for
    aggregate training features but reads as a display bug in a per-athlete
    list. Keeps one row per (date, mark), preferring whichever copy has a
    Results Score."""
    season = season.copy()
    if "Results Score" not in season.columns:
        # current_season_scraper.py's output has no Results Score column at
        # all (that field only ever came from the historical toplist scrape)
        # -- dedup below still works fine on (Date, Mark) alone.
        season["Results Score"] = pd.NA
    season["_date"] = pd.to_datetime(season["Date"], format="%d %b %Y", errors="coerce")
    season = season.sort_values("_date")
    season = season.sort_values("Results Score", na_position="last")
    season = season.drop_duplicates(subset=["Date", "Mark"], keep="first")
    season = season.sort_values("_date")
    history = []
    for _, r in season.iterrows():
        mark_val = None
        try:
            mark_val = parse_mark(str(r.get("Mark", "")))
            mark_display = format_mark(mark_val, disc_key)
        except Exception:
            mark_display = str(r.get("Mark", ""))
        score = r.get("Results Score")
        history.append({
            "date": r.get("Date"),
            "mark": mark_display,
            "markValue": mark_val,
            "venue": r.get("Venue"),
            "resultsScore": None if pd.isna(score) else int(score),
        })
    return history


WORLDWIDE_DIR = os.path.join(os.path.dirname(__file__), "data", "worldwide")


def load_worldwide_rows(disc_key, athlete_name):
    """One athlete's races from src/worldwide_scraper.py, if it has run.

    Optional by design and safe to call when the file is absent, partial, or
    mid-scrape: the scraper writes newest season first and flushes every 20
    meetings, so this returns whatever has landed rather than waiting for a
    complete run. Nothing in the modelling path reads this directory -- see
    that script's quarantine note before moving it."""
    path = os.path.join(WORLDWIDE_DIR, f"{disc_key}.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    if df.empty or "Competitor" not in df.columns:
        return pd.DataFrame()
    return df[df["Competitor"].str.lower() == athlete_name.lower()]


def load_career_progression(disc_key, athlete_name):
    """Season-by-season best for one athlete, across every year on record.

    The profile already charts the CURRENT season race by race; this is the
    other axis, and the site had no view of it at all. Assembled from every
    source that carries a dated mark -- the historical 2018-2025 file, this
    season's toplist row and per-meeting results, and the worldwide race log
    when it exists -- because no single one of them spans a career.

    `best` is the season's best mark in the direction that discipline
    actually means: lowest time for a track event, highest distance for a
    field event. Getting that backwards is the same mistake that inverted
    weighted_season_best (HANDOFF 0i2), so it is written once here.

    Indoor marks are counted and reported per season rather than dropped or
    silently mixed -- for the vertical jumps that is up to half the data,
    and a progression line that hides it is a claim the data cannot make."""
    is_field = disc_key in FIELD_EVENTS
    frames = []

    hist_path = os.path.join(RAW_DIR, f"{disc_key}.csv")
    if os.path.exists(hist_path):
        try:
            hist = pd.read_csv(hist_path)
            frames.append(hist[hist["Competitor"].str.lower() == athlete_name.lower()])
        except Exception:
            pass

    for fname in (f"{disc_key}_{MEETS_YEAR}.csv", f"{disc_key}_{MEETS_YEAR}_meetings.csv"):
        path = os.path.join(RAW_DIR, fname)
        if not os.path.exists(path):
            continue
        try:
            cur = pd.read_csv(path)
            mine = cur[cur["Competitor"].str.lower() == athlete_name.lower()].copy()
            if "year" not in mine.columns:
                mine["year"] = MEETS_YEAR
            frames.append(mine)
        except Exception:
            pass

    wide = load_worldwide_rows(disc_key, athlete_name)
    if not wide.empty:
        frames.append(wide)

    frames = [f for f in frames if not f.empty]
    if not frames:
        return []

    df = pd.concat(frames, ignore_index=True)
    # parse_mark raises on anything that isn't a mark, and real results are
    # full of things that aren't: "NM" (no valid attempt), "NH", "DNS",
    # "DNF", "DQ". Those are genuine outcomes, not corrupt rows, and they
    # simply have no value to plot -- dropped here rather than allowed to
    # take the whole profile down with a 500, which is what they did on the
    # first run (Jessica Schilder, women's shot put).
    df["_value"] = df["Mark"].map(safe_parse_mark)
    df = df.dropna(subset=["_value", "year"])
    if df.empty:
        return []
    df["_indoor"] = df.get("Venue", pd.Series(dtype=object)).map(is_indoor_venue)
    # The same race reaches this frame from more than one scraper under
    # differently-formatted venue names -- fine for aggregates, a double
    # count here.
    df = df.drop_duplicates(subset=["year", "Date", "Mark"])

    seasons = []
    for year, group in df.groupby("year"):
        best = group["_value"].max() if is_field else group["_value"].min()
        seasons.append({
            "year":       int(year),
            "best":       round(float(best), 3),
            "bestMark":   format_mark(float(best), disc_key),
            "marks":      int(len(group)),
            "indoorMarks": int(group["_indoor"].sum()),
        })
    seasons.sort(key=lambda s: s["year"])
    return seasons


def load_athlete_history(disc_key, athlete_name):
    """Real per-meet marks for an athlete -- the current, in-progress
    season if src/current_season_scraper.py has real meeting data for them
    (data/raw/{disc_key}_{MEETS_YEAR}_meetings.csv), falling back to their
    own most recent *completed* season on record otherwise. Both are real
    scraped results, never fabricated interpolation (see the Projections
    page's known-fabricated trajectory chart for the mistake this
    deliberately avoids).

    The current-season file is a separate, small, dedicated scrape
    (per-meeting DL circuit results for MEETS_YEAR only) -- kept apart from
    both data/raw/{disc_key}.csv (the historical 2018-2025 training file,
    which must never gain current-season rows: the model's LABEL_YEARS
    logic assumes only completed seasons with a real Final result to
    label) and data/raw/{disc_key}_2026.csv (the live worldwide toplist
    snapshot live_fetcher.py overwrites on every run.py run, one row per
    athlete). See current_season_scraper.py's own docstring for the full
    reasoning.

    Falling back to a prior season is picked per-athlete, not as one fixed
    year for everyone: this dataset only covers the Diamond League circuit
    + major meets (see Known Limitations), so an athlete who was hurt,
    skipped the circuit, or focused elsewhere in a given year can have zero
    rows that year while genuinely having a real, fuller season on record
    from an earlier one -- e.g. Shaunae Miller-Uibo has no 2025 rows here
    but a real 2022 season. Returns [] only if the athlete truly has no
    real data on record anywhere, current or historical (some newer
    athletes won't)."""
    current_path = os.path.join(RAW_DIR, f"{disc_key}_{MEETS_YEAR}_meetings.csv")
    if os.path.exists(current_path):
        current_df = pd.read_csv(current_path)
        if not current_df.empty:
            mine = current_df[current_df["Competitor"].str.lower() == athlete_name.lower()]
            if not mine.empty:
                return _season_rows_to_history(mine, disc_key), MEETS_YEAR

    path = os.path.join(RAW_DIR, f"{disc_key}.csv")
    if not os.path.exists(path):
        return [], None
    df = pd.read_csv(path)
    if df.empty:
        return [], None
    mine = df[df["Competitor"].str.lower() == athlete_name.lower()]
    if mine.empty:
        return [], None
    last_year = int(mine["year"].max())
    season = mine[mine["year"] == last_year]
    return _season_rows_to_history(season, disc_key), last_year


def load_h2h_vs_rivals(disc_key, athlete_name, rival_names):
    """Head-to-head record vs. this discipline's other top predicted
    contenders.

    Derived from the race log when there is one: two athletes met if they
    appear in the same meeting on the same date, and the lower finishing
    position won. That is exact, and it is a deeper sample than the
    alternative -- Joe Kovacs vs Ryan Crouser reads 5-23 over 28 shared
    races from the log against 4-9 over 13 from h2h_rates.csv.

    **This had to change because the two disagreed on the same page.** Once
    the analytics block started showing derived records, the old numbers sat
    directly beneath them contradicting every pair. h2h_rates.csv was thin
    as well as wrong -- 63.1% coverage after HANDOFF 0o corrected the
    fabricated sweeps in it. Recording the round closed most of that gap
    (0o's open half, now done): coverage is 77.1% and the file no longer
    reports a Kovacs-Crouser record three races deep.

    It is still the fallback, and it still feeds the MODEL unchanged via
    train_model.add_h2h_features -- swapping the model's h2h source is an
    accuracy change with its own backtest, not a display fix."""
    if not rival_names:
        return []

    derived = athlete_analytics.head_to_head(
        athlete_analytics.load_race_log(disc_key), athlete_name,
        opponents=rival_names, min_meetings=2,
    )
    if derived:
        return [{
            "opponent":  d["name"],
            "wins":      d["wins"],
            "losses":    d["losses"],
            "meetings":  d["meetings"],
        } for d in derived]

    if not os.path.exists(H2H_PATH):
        return []
    h2h = pd.read_csv(H2H_PATH)
    sub = h2h[
        (h2h["discipline"] == disc_key)
        & (h2h["athlete_a"].str.lower() == athlete_name.lower())
    ]
    matchups = []
    for rival in rival_names:
        row = sub[sub["athlete_b"].str.lower() == rival.lower()]
        if row.empty:
            continue
        r = row.iloc[0]
        meetings = int(r["meetings"])
        if meetings < 2:
            continue  # matches train_model.py's add_h2h_features threshold
        matchups.append({
            "opponent": rival,
            "wins": int(r["wins"]),
            "losses": meetings - int(r["wins"]),
            "meetings": meetings,
        })
    return matchups


PHOTO_QUERY = """query GetAthletePhoto($ids: [Int]) {
  getAthleteActionPictureByIds(ids: $ids) {
    id
    primaryMediaId
  }
}"""


def load_athlete_photo(profile_url):
    """Real headshot from World Athletics' own asset CDN -- not a stand-in
    or a generic avatar. Found via live GraphQL schema introspection (2026-
    08-23): getAthleteActionPictureByIds(ids: [competitorId]) resolves a
    real primaryMediaId for a real athlete (confirmed: fullName echoed back
    matched), and https://assets.aws.worldathletics.org/<primaryMediaId> is
    the exact same URL WA's own profile pages use as their og:image meta tag
    (verified by fetching a real profile page's HTML) -- a public asset,
    not something hotlinked against their wishes. Returns None (not a fake
    placeholder) if the competitor id can't be parsed from profile_url or
    the athlete has no photo on file -- the frontend shows an initials
    badge in that case, same principle as everywhere else in this project.

    profile_url comes in two real WA formats depending on the athlete:
    '.../athletes/athlete=<id>' (most common) and '.../athletes/<country>/
    <slug>-<id>' (~8% of rows, seen for newer/lower-profile athletes). Both
    are matched here -- an athlete landing in the second format isn't a
    'no photo on file' case, it was just never actually queried."""
    if not isinstance(profile_url, str):
        return None
    m = re.search(r"athlete=(\d+)", profile_url) or re.search(r"-(\d+)/?$", profile_url)
    if not m:
        return None
    competitor_id = int(m.group(1))
    try:
        data = dlr.graphql("GetAthletePhoto", {"ids": [competitor_id]}, PHOTO_QUERY)
    except Exception:
        return None
    results = data.get("getAthleteActionPictureByIds") or []
    if not results or not results[0].get("primaryMediaId"):
        return None
    return f"https://assets.aws.worldathletics.org/{results[0]['primaryMediaId']}"


# OpenCV's classic res10 SSD face detector -- not bundled with the
# opencv-python-headless wheel, so it's fetched once (same on-demand-download
# pattern webdriver-manager already uses elsewhere in this project) from
# OpenCV's own canonical model URLs and cached to disk under data/models/.
# Chosen over a plain Haar cascade after testing both on real WA photos:
# Haar produced a real false-positive (locked onto a shirt/bib pattern, not
# a face) on a shot put photo the SSD model got right; the SSD model was
# also the only one of the two that correctly disambiguated two real faces
# in one photo (Noah Lyles mid-celebration next to a competitor) by area.
FACE_PROTOTXT_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
FACE_CAFFEMODEL_URL = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
FACE_PROTOTXT_PATH = os.path.join(MODELS_DIR, "face_deploy.prototxt")
FACE_CAFFEMODEL_PATH = os.path.join(MODELS_DIR, "face_res10_300x300_ssd.caffemodel")

_face_net = None


def _get_face_net():
    """Lazily download (once) and load the face-detector model. Returns None
    if the download fails (e.g. no network) -- callers fall back to the
    fixed 15%-from-top crop default, same as when no face is found."""
    global _face_net
    if _face_net is not None:
        return _face_net
    try:
        os.makedirs(MODELS_DIR, exist_ok=True)
        if not os.path.exists(FACE_PROTOTXT_PATH):
            r = requests.get(FACE_PROTOTXT_URL, timeout=20)
            r.raise_for_status()
            with open(FACE_PROTOTXT_PATH, "wb") as f:
                f.write(r.content)
        if not os.path.exists(FACE_CAFFEMODEL_PATH):
            r = requests.get(FACE_CAFFEMODEL_URL, timeout=60)
            r.raise_for_status()
            with open(FACE_CAFFEMODEL_PATH, "wb") as f:
                f.write(r.content)
        _face_net = cv2.dnn.readNetFromCaffe(FACE_PROTOTXT_PATH, FACE_CAFFEMODEL_PATH)
    except Exception:
        return None
    return _face_net


def detect_face_focus(image_bytes):
    """Real face detection on the real downloaded photo -- returns the
    detected face's own center as {x, y} percentages of the image, meant to
    be fed straight into CSS background-position so the crop follows
    wherever the face actually is instead of a fixed guess. Picks the
    LARGEST detected face (by box area) when more than one appears (e.g. a
    competitor in the background) -- the photographed athlete is reliably
    the closer/more prominent figure in these action shots, which is a more
    reliable signal than raw detector confidence (a calmer bystander's face
    can score higher confidence than the actual subject mid-celebration).
    Returns None if no face is found or the model can't be loaded."""
    net = _get_face_net()
    if net is None:
        return None
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    best_box, best_area = None, 0.0
    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])
        if confidence < 0.5:
            continue
        x1, y1, x2, y2 = detections[0, 0, i, 3:7] * [w, h, w, h]
        area = max(x2 - x1, 0) * max(y2 - y1, 0)
        if area > best_area:
            best_area, best_box = area, (x1, y1, x2, y2)
    if best_box is None:
        return None
    x1, y1, x2, y2 = best_box
    return {
        "x": round(float((x1 + x2) / 2 / w * 100), 1),
        "y": round(float((y1 + y2) / 2 / h * 100), 1),
    }


def _load_focus_cache():
    if not os.path.exists(FOCUS_CACHE_PATH):
        return {}
    try:
        with open(FOCUS_CACHE_PATH, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_focus_cache(cache):
    os.makedirs(os.path.dirname(FOCUS_CACHE_PATH), exist_ok=True)
    with open(FOCUS_CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2, sort_keys=True)


def get_photo_focus(photo_url):
    """Real per-photo face position, cached to disk by photo URL so the
    same athlete's photo isn't re-downloaded and re-detected on every
    profile-page view. A cached `null` means detection genuinely ran and
    found no face (not 'not computed yet') -- the frontend falls back to a
    fixed top-biased crop in that case, same as when photo_url is None."""
    if not photo_url:
        return None
    cache = _load_focus_cache()
    if photo_url in cache:
        return cache[photo_url]
    try:
        r = requests.get(photo_url, timeout=15)
        r.raise_for_status()
        focus = detect_face_focus(r.content)
    except Exception:
        focus = None
    cache[photo_url] = focus
    _save_focus_cache(cache)
    return focus


def load_predictions():
    """Load predictions_latest.csv and build discipline data."""
    path = os.path.join(OUTPUTS_DIR, "predictions_latest.csv")
    if not os.path.exists(path):
            return None, None
    
    df = pd.read_csv(path)
    injury_flags = load_injury_flags()
    track = []
    field = []

    for disc_key, label in DISC_LABELS.items():
        disc_df = df[df["discipline"] == label].copy()
        if disc_df.empty:
            continue

        athletes = []
        near_miss = []
        for _, row in disc_df.iterrows():
            mark_str = str(row.get("season_best", ""))
            try:
                mark_val = parse_mark(mark_str)
                mark_display = format_mark(mark_val, disc_key)
            except:
                mark_display = mark_str

            prob = row.get("win_probability", "0%")
            if isinstance(prob, str):
                prob = int(prob.replace("%", ""))
            else:
                prob = int(float(prob) * 100) if float(prob) <= 1 else int(prob)

            wa_url = row.get("profile_url")
            if not isinstance(wa_url, str) or not wa_url or wa_url == "nan":
                wa_url = f"https://www.worldathletics.org/search/?q={row['athlete_name'].replace(' ', '+')}"

            injury_watch = bool(row.get("injury_watch", False))
            reason, evidence_url = (
                injury_evidence(injury_flags.get(normalize_athlete_name(row["athlete_name"])))
                if injury_watch else (None, None)
            )

            # True when run.py actually found this athlete in WA's own
            # scraped 2026 Diamond League standings for this discipline;
            # False on the rare worldwide-season-best-ranking fallback (see
            # run.py's WARNING print for when that triggers). Defaults True
            # for CSVs written before this column existed -- verified live
            # (2026-08-23) that every discipline was standings-based then,
            # so that default reflects real historical fact, not a guess.
            dl_qualified = row.get("dl_qualified")
            qualified = True if pd.isna(dl_qualified) else bool(dl_qualified)

            # predicted_rank is NULL for near-miss athletes -- run.py
            # deliberately refuses to number them as if they were finalists.
            raw_rank = row.get("predicted_rank")
            rank = int(raw_rank) if pd.notna(raw_rank) else 0

            (athletes if qualified else near_miss).append({
                "rank":         rank,
                "name":         row["athlete_name"],
                "nat":          str(row.get("nationality", "—")),
                "qualified":    qualified,
                "mark":         mark_display,
                "prob":         prob,
                "waUrl":        wa_url,
                "injuryWatch":  injury_watch,
                "injuryReason": reason,
                "injuryUrl":    evidence_url,
            })

        # `athletes` stays the CONFIRMED Diamond League field and nothing
        # else. Every model-derived aggregate downstream -- top winners,
        # confidence, storylines, discipline_favourite -- reads this list, and
        # a near-miss athlete leaking into it would be presented as a
        # projected finalist. They travel separately.
        disc_obj = {
            "id":       disc_key,
            "label":    label,
            # How many places this discipline actually has at the Final
            # (6 field / 10 long distance / 8 otherwise). The field itself
            # can be SHORTER than this -- an injury removal leaves a gap --
            # so the UI can't infer the number from len(athletes).
            "qualLimit": get_qual_limit(disc_key),
            "athletes": sorted(athletes, key=lambda x: x["rank"]),
            # Ranked within their own group by season best (run.py writes
            # them in that order), purely so the frontend's shared sort can
            # order them the same way it orders the real field. The UI never
            # shows these numbers -- a near-miss athlete has no rank.
            "nearMiss": [{**a, "rank": i + 1} for i, a in enumerate(near_miss)],
        }

        if disc_key in FIELD_EVENTS:
            field.append(disc_obj)
        else:
            track.append(disc_obj)

    return track, field


STANDINGS_PATH = os.path.join(os.path.dirname(__file__), "data", "standings.json")


def load_standings():
    """WA's own Diamond League standings per discipline -- the list run.py
    actually restricts the projected field to."""
    try:
        with open(STANDINGS_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def toplist_entry(disc_key, athlete_name):
    """An athlete's row in this season's worldwide toplist, whether or not
    they are in the projected field. Returns (mark, world_rank, wa_url)."""
    path = os.path.join(RAW_DIR, f"{disc_key}_{MEETS_YEAR}.csv")
    if not os.path.exists(path):
        return None, None, None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None, None, None
    hit = df[df["Competitor"].astype(str).str.lower() == athlete_name.lower()]
    if hit.empty:
        return None, None, None
    r = hit.iloc[0]
    rank = r.get("Rank")
    url = r.get("ProfileURL")
    return (
        str(r.get("Mark")) if pd.notna(r.get("Mark")) else None,
        int(rank) if pd.notna(rank) else None,
        url if isinstance(url, str) and url and url != "nan" else None,
    )


def toplist_bio(disc_key, athlete_name):
    """Nationality and age straight from the season toplist row.

    The 128 near-miss athletes run.py scores carry these already, but anyone
    further down the toplist (Jakob Ingebrigtsen, 14th in the 1500m
    standings, is the live example) has no prediction row at all -- and
    there is no reason their page should be blank on two facts the scrape
    already collected."""
    path = os.path.join(RAW_DIR, f"{disc_key}_{MEETS_YEAR}.csv")
    if not os.path.exists(path):
        return {}
    try:
        df = pd.read_csv(path)
    except Exception:
        return {}
    hit = df[df["Competitor"].astype(str).str.lower() == athlete_name.lower()]
    if hit.empty:
        return {}
    row = hit.iloc[0]

    out = {}
    # WA's toplist export leaves the nationality column unnamed.
    for col in ("Unnamed: 5", "Nat", "Country"):
        val = row.get(col)
        if isinstance(val, str) and val.strip() and val.strip().lower() != "nan":
            out["nat"] = val.strip()
            break

    dob = row.get("DOB")
    if isinstance(dob, str) and dob.strip():
        try:
            born = datetime.strptime(dob.strip(), "%d %b %Y").date()
            out["age"] = round((date.today() - born).days / 365.25, 1)
        except ValueError:
            pass
    return out


def scored_prediction_row(disc_key, athlete_name):
    """This athlete's row in predictions_latest.csv, if run.py scored them.

    True for the near-miss athletes (dl_qualified = False, no rank) and for
    nobody else outside the field -- so it is the difference between a page
    that can show real season stats and a model probability, and one that
    can only show a season best."""
    label = DISC_LABELS.get(disc_key)
    path = os.path.join(OUTPUTS_DIR, "predictions_latest.csv")
    if label is None or not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    hit = df[(df["discipline"] == label)
             & (df["athlete_name"].astype(str).str.lower() == athlete_name.lower())]
    return None if hit.empty else hit.iloc[0]


def projected_field_names(disc_key, limit=6):
    """The confirmed finalists, best-ranked first -- the opponents a
    head-to-head record for a non-qualified athlete is actually about
    ("how does he do against the ones who got in?")."""
    track, field = load_predictions()
    disc = next((d for d in (track or []) + (field or []) if d["id"] == disc_key), None)
    if not disc:
        return []
    return [a["name"] for a in sorted(disc["athletes"], key=lambda a: a["rank"])[:limit]]


def dl_meetings_count(disc_key, athlete_name):
    """How many Diamond League meetings this athlete contested this season.

    Read from the same file refresh_current_season_stats derives the scored
    athletes' meets_count from, so an unscored athlete's tile means exactly
    what a scored one's does. Returns 0, not None, when the file exists and
    simply has no rows for them -- for an athlete outside the field that
    zero is the answer, and frequently the reason."""
    path = os.path.join(RAW_DIR, f"{disc_key}_{MEETS_YEAR}_meetings.csv")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if "Competitor" not in df.columns:
        return None
    return int((df["Competitor"].str.lower() == str(athlete_name).lower()).sum())


def athlete_field_status(disc_key, athlete_name):
    """Why is (or isn't) this athlete in the projected field?

    Mirrors run.py's actual selection order, so the answer is the real
    mechanism rather than a guess:
      1. restrict to WA's official DL standings for the discipline
      2. drop anyone the injury checker marked "remove"
      3. drop anyone missing model features
      4. keep the top N by season best (8 track / 6 field / 10 long distance)

    Returns a dict the frontend can render verbatim. `reason` is None when
    the athlete IS in the field."""
    label = DISC_LABELS.get(disc_key)
    mark, world_rank, wa_url = toplist_entry(disc_key, athlete_name)
    out = {
        "discKey": disc_key,
        "disc": label,
        "name": athlete_name,
        "seasonBest": mark,
        "worldRank": world_rank,
        "waUrl": wa_url,
        "inField": False,
        "reason": None,
        "reasonCode": None,
    }

    track, field = load_predictions()
    disciplines = (track or []) + (field or [])
    disc = next((d for d in disciplines if d["id"] == disc_key), None)
    if disc:
        hit = next((a for a in disc["athletes"]
                    if a["name"].lower() == athlete_name.lower()), None)
        if hit:
            out.update({"inField": True, "name": hit["name"], "seasonBest": hit["mark"]})
            return out

    # Real per-meet season form, exactly the same source the in-field
    # profile chart uses -- being outside the projected eight doesn't make
    # an athlete's actual races any less real, and "here is his season" is
    # most of what someone came to the page for.
    history, history_year = load_athlete_history(disc_key, athlete_name)
    out["history"] = history
    out["historyYear"] = history_year

    # Same real World Athletics headshot the in-field profiles get -- there
    # is no reason an athlete outside the projected eight should get a
    # visibly lesser page. Returns None (never a stock image) if WA has no
    # photo, and the frontend falls back to the initials monogram.
    photo_url = load_athlete_photo(wa_url) if wa_url else None
    out["photoUrl"] = photo_url
    out["photoFocus"] = get_photo_focus(photo_url) if photo_url else None

    # Everything below is the same real data the in-field profile shows,
    # for the same reason the photo is: none of it stops being true because
    # the athlete missed the cut. What is deliberately NOT carried over is
    # the finalist-only material -- a predicted rank, and a podium chance
    # presented as a forecast -- since neither means anything here.
    bio = toplist_bio(disc_key, athlete_name)
    row = scored_prediction_row(disc_key, athlete_name)

    def clean(val):
        if val is None or pd.isna(val):
            return None
        return val.item() if hasattr(val, "item") else val

    if row is not None:
        prob = row.get("win_probability")
        out.update({
            "name":          row["athlete_name"],
            "nat":           str(row.get("nationality")) if pd.notna(row.get("nationality")) else bio.get("nat"),
            "careerBest":    clean(row.get("career_best")),
            "pbGap":         clean(row.get("pb_gap")),
            "age":           clean(row.get("age")),
            "meetsCount":    clean(row.get("meets_count")),
            "daysSinceLast": clean(row.get("days_since_last")),
            # The model DID score this athlete -- run.py scores the near-miss
            # group with the same features and the same forest. It is a real
            # number, but a conditional one, so the frontend has to label it
            # "if they qualified" rather than as a prediction about the Final.
            "hypotheticalProb": int(str(prob).replace("%", "")) if isinstance(prob, str)
                                else (None if pd.isna(prob) else int(float(prob) * 100)),
        })
    else:
        # Not scored by run.py, so there is no predictions_latest.csv row to
        # read career best, PB gap or meeting count off. They were left blank
        # -- but only the SCORE needs the model. The underlying facts are all
        # on disk, and the page was showing dashes over data it already had.
        #
        # Reported by the user for Dina Asher-Smith, who is not in the
        # predictions file at all yet has 41 races on record, 8 seasons of
        # history and a 10.83 career best in the 100m.
        #
        # Career best is the best mark across every season on record, in the
        # direction the discipline actually means -- the same track/field
        # flip that inverted weighted_season_best when it was written once
        # (HANDOFF 0i2). PB gap matches feature_builder's definition exactly,
        # `abs(season_best - career_best)`, so the number means the same
        # thing here as on a scored athlete's page.
        career = load_career_progression(disc_key, athlete_name)
        career_best_val = None
        if career:
            values = [s["best"] for s in career]
            career_best_val = max(values) if disc_key in FIELD_EVENTS else min(values)
        season_val = safe_parse_mark(out.get("seasonBest"))

        out.update({
            "nat":              bio.get("nat"),
            "age":              bio.get("age"),
            "careerBest":       (format_mark(career_best_val, disc_key)
                                 if career_best_val is not None else None),
            "pbGap":            (round(abs(season_val - career_best_val), 3)
                                 if season_val is not None and career_best_val is not None
                                 else None),
            "meetsCount":       dl_meetings_count(disc_key, athlete_name),
            "daysSinceLast":    None,
            "hypotheticalProb": None,
        })

    # Head-to-head against the athletes who DID qualify. Same function and
    # same >=2-meetings threshold the in-field profile uses -- only the
    # opponent list differs, and here it is the whole point of the panel.
    field_names = projected_field_names(disc_key)
    out["h2h"] = load_h2h_vs_rivals(disc_key, athlete_name, field_names)

    # A near-miss athlete very often has ZERO Diamond League meetings -- that
    # is frequently the whole reason they are not qualified -- so the
    # DL-derived meets_count and days_since_last run.py writes come back 0
    # and blank, and the season-stats panel read as broken rather than as
    # "raced, but not here". Measured across the 127 non-qualified athletes:
    # 85 were already complete, 23 had real races the race log knew about
    # and the panel was not using, and 19 genuinely have no individual race
    # result on record anywhere.
    #
    # `daysSinceLast` is filled from the race log, which spans every
    # competition rather than the DL circuit alone. `racesThisSeason` is
    # reported SEPARATELY from meetsCount rather than overwriting it,
    # because the two count different things and the tiles label them that
    # way -- collapsing them would recreate exactly the scope clash that had
    # Lyles reading 2 in one place and 4 in another.
    log_rows = athlete_analytics.athlete_rows(
        athlete_analytics.load_race_log(disc_key), athlete_name,
    )
    season_rows = log_rows[log_rows["year"] == MEETS_YEAR] if not log_rows.empty else log_rows
    out["racesThisSeason"] = int(len(season_rows))
    out["racesOnRecord"] = int(len(log_rows))

    if not season_rows.empty and season_rows["date"].notna().any():
        last_race = season_rows["date"].max()
        days = int((pd.Timestamp(date.today()) - last_race).days)
        # The race log WINS here, it does not merely fill a blank. The tile
        # says "Last competed" with no qualifier, and run.py's figure counts
        # only Diamond League meetings: Noah Lyles read "62d ago" (his last
        # DL race, Paris on 28 Jun) while he had actually raced 36 days
        # earlier than that reading, on 24 Jul, at a non-DL meeting. An
        # unqualified claim has to be the true one.
        existing = out.get("daysSinceLast")
        if existing is None or days < existing:
            out["daysSinceLast"] = days
        out["lastRaceDate"] = last_race.strftime("%d %b %Y")
    else:
        out["lastRaceDate"] = None

    # The same analyst material an in-field profile gets. None of it depends
    # on being selected for the Final: a win rate, a season shape and a
    # head-to-head are facts about races already run. Withholding them here
    # made the near-miss page look like a stub of the real one, when for a
    # reader asking "should this athlete have qualified?" the record is
    # exactly the evidence they came for -- Noah Lyles is the case this
    # page exists for, and he is a world champion sitting 9th on points.
    out["careerSeasons"] = load_career_progression(disc_key, athlete_name)
    out["scoreContext"]  = athlete_score_context(disc_key, athlete_name)
    out["analytics"]     = athlete_analytics.build_analytics(
        disc_key, athlete_name, disc_key in FIELD_EVENTS,
    )
    # Here the "in field" marker means the opponents who DID qualify, which
    # is the more pointed reading of the same badge.
    out["rivalNames"] = field_names
    out["career"] = athlete_career.build_career(athlete_name)

    standings = load_standings().get(disc_key, [])
    in_standings = any(n.lower() == athlete_name.lower() for n in standings)

    # Their real position in the FULL standings table, which runs well below
    # the qualifying places that standings.json keeps. Attached whether or
    # not it is the reason they're out, since "9th on 15 points" is the fact
    # a reader came here for either way.
    dl = standings_position(disc_key, athlete_name)
    out["dl"] = dl

    if standings and not in_standings:
        if dl:
            out["reasonCode"] = "outside_points_cut"
            out["reason"] = points_cut_reason(label, dl)
        else:
            out["reasonCode"] = "not_in_standings"
            out["reason"] = (
                f"Not in World Athletics' official Diamond League standings for "
                f"{label} — no Diamond League points scored in this discipline "
                f"this season. Points are what earns a place at the Final, "
                f"regardless of how fast they have run elsewhere."
            )
        return out

    flags = load_injury_flags()
    entry = flags.get(normalize_athlete_name(athlete_name))
    if entry and entry.get("status") == "remove":
        why, url = injury_evidence(entry)
        out["reasonCode"] = "injury_removed"
        out["reason"] = "Removed from the projected field by the injury check."
        out["injuryReason"] = why
        out["injuryUrl"] = url
        return out

    if mark is not None:
        out["reasonCode"] = "outside_cut"
        out["reason"] = (
            f"In the Diamond League standings but outside the projected "
            f"top {get_qual_limit(disc_key)} on season best for {label}."
        )
        return out

    out["reasonCode"] = "no_data"
    out["reason"] = f"No {MEETS_YEAR} season mark on record for {label}."
    return out


STANDINGS_DETAIL_PATH = os.path.join(os.path.dirname(__file__), "data", "standings_detail.json")

# A Diamond League meeting scores 8-7-6-5-4-3-2-1 for the first eight
# places, so a win is worth 8. Checked against the scraped standings rather
# than taken on trust: across all 32 disciplines the highest total held by
# anyone with a single meeting to their name is exactly 8, and nobody
# anywhere averages more than 8 points per meeting.
MAX_POINTS_PER_MEETING = 8


def load_standings_detail():
    """WA's full Diamond League standings tables -- every athlete with their
    points, rather than just the names above the qualification cut that
    standings.json keeps. Written by scrape_dl_standings(); absent until the
    standings scraper has run at least once since 2026-08-25."""
    try:
        with open(STANDINGS_DETAIL_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def meetings_remaining(meets=None, today=None):
    """Scoring meetings still to come. The Final is excluded on purpose: it
    is the thing being qualified for, so its result cannot change who was
    invited to it."""
    statuses = compute_meet_statuses(meets or MEETS, today=today)
    return sum(1 for m in statuses if m["status"] in ("next", "upcoming"))


def qualification_race(rows, qual_limit, meetings_left,
                       max_points=MAX_POINTS_PER_MEETING):
    """Who is in, who is still alive, and who is arithmetically out.

    Deliberately arithmetic rather than a model. Points never go down and
    the most anyone can still gain is `meetings_left * max_points`, which
    this late in the season settles most of the table outright -- a
    probability here would be a worse answer dressed as a better one.

    Each verdict is only stated when it is certain:
      * "out"  -- at least `qual_limit` athletes ALREADY hold more points
                  than this athlete can reach at absolute maximum.
      * "safe" -- fewer than `qual_limit` athletes can reach this athlete's
                  current total even at maximum, so nobody can displace
                  them even if they never score again.
    A tie counts against the athlete in both tests, since World Athletics'
    tie-break rules are not in this data.

    Everyone else is "in" (above the cut line as it stands) or "chasing"
    (below it and still mathematically alive).

    Note that this assumes the discipline is actually on the programme of
    the remaining meeting(s). If it isn't, its standings are already final
    -- which only ever makes "out" more true, never less."""
    gain = max(meetings_left, 0) * max_points
    scored = [r for r in rows if r.get("points") is not None]
    cut_points = scored[qual_limit - 1]["points"] if len(scored) >= qual_limit else None

    out = []
    for row in rows:
        points = row.get("points")
        if points is None:
            out.append({**row, "gap": None, "maxPoints": None, "status": "unknown"})
            continue
        ceiling = points + gain
        certainly_ahead = sum(1 for o in scored if o is not row and o["points"] > ceiling)
        could_finish_ahead = sum(1 for o in scored if o is not row and o["points"] + gain >= points)

        if certainly_ahead >= qual_limit:
            status = "out"
        elif could_finish_ahead < qual_limit:
            status = "safe"
        elif row.get("rank") is not None and row["rank"] <= qual_limit:
            status = "in"
        else:
            status = "chasing"

        out.append({
            **row,
            # Positive = points behind the cut line; 0 or negative = on or
            # above it. The line itself moves at the last meeting, so this
            # is where the race stands today, not a prediction.
            "gap": None if cut_points is None else round(cut_points - points, 2),
            "maxPoints": ceiling,
            "status": status,
        })
    return {"cutPoints": cut_points, "rows": out}


def build_qualification():
    """The race for the Final, per discipline: real WA points, the real cut
    line, and the gap to it (HANDOFF item 0l). Everything here comes from
    WA's own standings table -- nothing is modelled."""
    detail = load_standings_detail()
    disciplines = detail.get("disciplines") or {}
    left = meetings_remaining()
    statuses = compute_meet_statuses(MEETS)
    next_meet = next((m for m in statuses if m["status"] == "next"), None)

    out = []
    for key, label in DISC_LABELS.items():
        entry = disciplines.get(key)
        if not entry:
            continue
        limit = entry.get("qualLimit") or get_qual_limit(key)
        race = qualification_race(entry.get("standings") or [], limit, left)
        out.append({
            "discKey":    key,
            "disc":       label,
            "isField":    key in FIELD_EVENTS,
            "qualLimit":  limit,
            "cutPoints":  race["cutPoints"],
            "standings":  race["rows"],
        })

    return {
        "scrapedAt":     detail.get("scrapedAt"),
        "meetingsLeft":  left,
        "nextMeet":      next_meet,
        "pointsForAWin": MAX_POINTS_PER_MEETING,
        "disciplines":   out,
    }


def build_news(limit=20):
    """Every real news item the injury checker matched, as a feed.

    The evidence was already being scraped and stored -- it just only ever
    surfaced as a tooltip on whichever athlete it flagged, so a reader had
    to already suspect someone to find it. Showing it as a list also makes
    bad matches visible: the item that removed Cole Hocker is a headline
    about Jakob Ingebrigtsen, which is obvious the moment you read it in a
    feed and invisible when it is buried behind a badge.

    Deduped by URL -- the same article routinely matches several athletes."""
    flags = load_injury_flags()
    seen, items = set(), []
    for name, entry in flags.items():
        status = entry.get("status")
        matches = entry.get("matches") or []

        # A removal must never be invisible. This feed is now the ONLY place
        # withdrawn athletes are listed (the dashboard's separate "Removed
        # from predictions" panel was pure duplication and was deleted), so
        # a "remove" entry whose matches carry no usable headline still gets
        # a row rather than silently disappearing from the site.
        if status == "remove" and not any(m.get("headline") for m in matches):
            items.append({
                "headline":    "Flagged for removal by the automatic injury check.",
                "url":         next((m.get("url") for m in matches if m.get("url")), None),
                "source":      "injury check",
                "athlete":     name,
                "status":      status,
                "disciplines": [DISC_LABELS.get(d, d) for d in entry.get("disciplines", [])],
                "keywords":    sorted({k for m in matches for k in (m.get("keywords") or [])}),
            })
            continue

        for m in matches:
            url = m.get("url")
            headline = m.get("headline")
            if not headline:
                continue
            key = url or headline
            if key in seen:
                continue
            seen.add(key)
            items.append({
                "headline":    headline,
                "url":         url,
                "source":      (m.get("source") or "").replace("_results", ""),
                "athlete":     name,
                "status":      status,
                "disciplines": [DISC_LABELS.get(d, d) for d in entry.get("disciplines", [])],
                "keywords":    m.get("keywords") or [],
            })
    # "remove" outranks "watch": a withdrawal changes the field, a watch
    # only qualifies it.
    items.sort(key=lambda x: (x["status"] != "remove", x["athlete"]))
    return items[:limit]


def search_athletes(query, limit=25):
    """Every athlete in this season's worldwide toplists, not just the ~230
    in the projected field -- so "why isn't Lyles in the 100m?" is an
    answerable question rather than a silent absence."""
    q = (query or "").strip().lower()
    if len(q) < 2:
        return []
    results = []
    for disc_key, label in DISC_LABELS.items():
        path = os.path.join(RAW_DIR, f"{disc_key}_{MEETS_YEAR}.csv")
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path, usecols=["Competitor", "Mark", "Rank"])
        except Exception:
            continue
        names = df["Competitor"].astype(str)
        hits = df[names.str.lower().str.contains(q, regex=False, na=False)]
        for _, r in hits.iterrows():
            rank = r.get("Rank")
            results.append({
                "name": str(r["Competitor"]),
                "disc": label,
                "discKey": disc_key,
                "mark": str(r["Mark"]) if pd.notna(r.get("Mark")) else None,
                "worldRank": int(rank) if pd.notna(rank) else None,
            })
    # Best world rank first: the athlete someone is looking for is usually
    # the highest-ranked match, and a name can appear in several disciplines.
    results.sort(key=lambda x: (x["worldRank"] is None, x["worldRank"] or 9999))
    return results[:limit]


def build_athlete_profile(disc_key, athlete_name):
    """Full detail for one athlete's profile page: their own real stats
    (season/career best, PB gap, age, activity), real per-meet history from
    their last completed season, and a real head-to-head record against
    this discipline's other current top contenders. Returns None if the
    discipline or athlete isn't found (a stale/typo'd URL, not an error
    worth a 500)."""
    label = DISC_LABELS.get(disc_key)
    if label is None:
        return None

    path = os.path.join(OUTPUTS_DIR, "predictions_latest.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    disc_df = df[df["discipline"] == label]
    if disc_df.empty:
        return None

    match = disc_df[disc_df["athlete_name"].str.lower() == athlete_name.lower()]
    if match.empty:
        return None
    row = match.iloc[0]

    # Near-miss athletes now live in predictions_latest.csv too (run.py scores
    # them with dl_qualified = False and predicted_rank = None). They are NOT
    # finalists, so they must not get a finalist profile -- returning None
    # sends the route to /api/athlete-status, which explains why they are out
    # and still gives them a photo, season best and form chart.
    #
    # Returning None rather than defending each field individually is
    # deliberate: this function builds rank, podium chance and a head-to-head
    # against the rest of the field, none of which mean anything for someone
    # outside it. Before this guard it raised
    # "cannot convert float NaN to integer" on predicted_rank and the endpoint
    # 500'd -- and the frontend only falls back on 404, so every near-miss
    # athlete's profile was an error page.
    if not bool(row.get("dl_qualified", True)) or pd.isna(row.get("predicted_rank")):
        return None

    # Rivals are the real field only -- a near-miss athlete is not a rival.
    rivals_df = disc_df[
        (disc_df["athlete_name"].str.lower() != athlete_name.lower())
        & disc_df["predicted_rank"].notna()
    ]
    rivals_df = rivals_df.sort_values("predicted_rank").head(6)
    rival_names = rivals_df["athlete_name"].tolist()

    injury_flags = load_injury_flags()
    injury_watch = bool(row.get("injury_watch", False))
    reason, evidence_url = (
        injury_evidence(injury_flags.get(normalize_athlete_name(row["athlete_name"])))
        if injury_watch else (None, None)
    )

    prob = row.get("win_probability", "0%")
    prob = int(str(prob).replace("%", "")) if isinstance(prob, str) else int(float(prob) * 100)

    wa_url = row.get("profile_url")
    if not isinstance(wa_url, str) or not wa_url or wa_url == "nan":
        wa_url = f"https://www.worldathletics.org/search/?q={athlete_name.replace(' ', '+')}"

    def clean(val):
        # pandas/numpy scalar types (int64, float64, ...) aren't JSON
        # serializable as-is -- .item() converts to the native Python type.
        if pd.isna(val):
            return None
        return val.item() if hasattr(val, "item") else val

    history, history_year = load_athlete_history(disc_key, athlete_name)
    photo_url = load_athlete_photo(wa_url)

    # "Last competed" carries no qualifier, so it has to mean the last time
    # they competed -- anywhere. run.py's days_since_last is derived from the
    # Diamond League meetings file alone, and for 30 of the 237 in-field
    # athletes that overstated the gap: Patrizia Van Der Weken read "71d ago"
    # having actually raced 19 days earlier at a non-DL meeting. The race log
    # spans every competition, so it corrects the claim wherever it knows of
    # a more recent race.
    #
    # DISPLAY ONLY. predictions_latest.csv's column is untouched and
    # days_since_last remains a trained model feature computed the way it
    # always was -- redefining a feature is an accuracy change with its own
    # backtest, not a copy fix.
    days_since_last = clean(row.get("days_since_last"))
    last_race_date = None
    log_rows = athlete_analytics.athlete_rows(
        athlete_analytics.load_race_log(disc_key), athlete_name,
    )
    season_rows = log_rows[log_rows["year"] == MEETS_YEAR] if not log_rows.empty else log_rows
    if not season_rows.empty and season_rows["date"].notna().any():
        last_race = season_rows["date"].max()
        days = int((pd.Timestamp(date.today()) - last_race).days)
        if days_since_last is None or days < days_since_last:
            days_since_last = days
        last_race_date = last_race.strftime("%d %b %Y")

    return {
        "name":            row["athlete_name"],
        "discKey":         disc_key,
        "disc":            label,
        "nat":             str(row.get("nationality", "—")),
        "rank":            int(row["predicted_rank"]),
        "mark":            row["season_best"],
        "careerBest":      clean(row.get("career_best")),
        "pbGap":           clean(row.get("pb_gap")),
        "age":             clean(row.get("age")),
        "meetsCount":      clean(row.get("meets_count")),
        "daysSinceLast":   days_since_last,
        "lastRaceDate":    last_race_date,
        "racesThisSeason": int(len(season_rows)),
        "prob":            prob,
        "waUrl":           wa_url,
        "photoUrl":        photo_url,
        "photoFocus":      get_photo_focus(photo_url),
        "injuryWatch":     injury_watch,
        "injuryReason":    reason,
        "injuryUrl":       evidence_url,
        "history":         history,
        "historyYear":     history_year,
        "h2h":             load_h2h_vs_rivals(disc_key, athlete_name, rival_names),
        # The model's rival shortlist by name. The profile used to render a
        # second head-to-head panel scoped to these; that panel was removed
        # once both drew the same derived numbers and it became a duplicate.
        # The names still matter -- the merged panel marks which opponents
        # an athlete will actually meet at the Final -- so they travel on
        # their own rather than being inferred from `h2h`, which drops any
        # rival they have met fewer than twice.
        "rivalNames":      rival_names,
        # The two additions the "deep stats" pass brought in. `history` is
        # this season race by race; `careerSeasons` is the other axis, which
        # the site had no view of at all. `scoreContext` answers "is that
        # mark actually good" in terms that hold across events.
        "careerSeasons":   load_career_progression(disc_key, athlete_name),
        "scoreContext":    athlete_score_context(disc_key, athlete_name),
        # Race-log statistics: win/podium record, form, season shape, and a
        # head-to-head derived from actually sharing a race rather than
        # from the separate h2h scrape. None until worldwide_scraper.py has
        # run -- the profile falls back to what it always showed.
        #
        # NOTE the scope difference this creates on the page: `meetsCount`
        # above counts DIAMOND LEAGUE meetings (refresh_current_season_stats
        # derives it from {disc}_2026_meetings.csv) while these race totals
        # count every scraped final, so Lyles reads 2 there and 4 here. Both
        # are right; the UI labels each one's scope rather than picking a
        # winner, because meets_count is also a trained model feature and
        # changing what it means is an accuracy decision, not a display fix.
        "analytics":       athlete_analytics.build_analytics(
            disc_key, athlete_name, disc_key in FIELD_EVENTS,
        ),
        # What World Athletics says this athlete has already won, and where
        # it ranks them. Read, not derived -- see athlete_career's docstring
        # for why that is kept in a separate module from the race-log stats.
        "career":          athlete_career.build_career(athlete_name),
    }


def discipline_favourite(disc):
    """The model's actual #1 pick for a discipline: highest WIN PROBABILITY.

    NOT `athletes[0]`. That list is ordered by real season-best mark (see
    load_predictions), and the two orderings genuinely disagree across about
    half the field -- measured 2026-08-24: **15 of 32 disciplines**, e.g. the
    Men's 100m's best mark is Noah Lyles at 16% while the model's pick is
    Oblique Seville at 27%, and the Women's 3000m Steeplechase's best mark is
    Peruth Chemutai at 31% while the model picks Winfred Yavi at 52%.

    Reading `athletes[0]` as "the favourite" is a mistake this project has now
    made in four separate places (the Projections hero, the rivalry storyline,
    and both callers below). Route every "who does the model pick" question
    through this function rather than indexing the list again.

    Ties resolve to the best season mark, because `max` keeps the first
    maximum and the list is already mark-ordered."""
    if not disc["athletes"]:
        return None
    return max(disc["athletes"], key=lambda a: a["prob"])


def build_top_winners(track, field):
    """Returns a list of {rank, name, disc, ...} dicts, sorted by probability,
    capped at 6. `rank` is the list POSITION (1-6) for the frontend to render
    as a podium-style badge -- not the athlete's own predicted_rank within
    their discipline (every entry here is already a #1 pick). Per-athlete
    history/h2h detail lives behind /api/athlete/<disc>/<name> instead of
    embedded here -- computing it for every winner on every dashboard load
    was wasted work once the frontend moved to a dedicated profile page
    fetched on demand."""
    all_discs = (track or []) + (field or [])
    winners = []
    for disc in all_discs:
        a = discipline_favourite(disc)
        if a:
            winners.append({
                "name":         a["name"],
                "disc":         disc["label"],
                "discKey":      disc.get("id"),
                "mark":         a["mark"],
                "prob":         a["prob"],
                "waUrl":        a["waUrl"],
                "injuryWatch":  a["injuryWatch"],
                "injuryReason": a["injuryReason"],
                "injuryUrl":    a["injuryUrl"],
            })
    top6 = sorted(winners, key=lambda x: -x["prob"])[:6]
    for i, w in enumerate(top6):
        w["rank"] = i + 1
    return top6


def build_removed_athletes(injury_flags):
    """Athletes filtered out of predictions entirely by run.py's injury check
    (status == 'remove') never appear anywhere in predictions_latest.csv, so
    without this they'd just silently vanish from the dashboard with zero
    explanation -- indistinguishable from a scraping gap."""
    removed = []
    for name, entry in injury_flags.items():
        if entry.get("status") != "remove":
            continue
        reason, url = injury_evidence(entry)
        removed.append({
            "name":        name,
            "disciplines": [DISC_LABELS.get(k, k) for k in entry.get("disciplines", [])],
            "reason":      reason,
            "url":         url,
        })
    return removed


def build_confidence(track, field):
    """How confident the model is in each discipline, i.e. the probability of
    its own top pick -- so it reads through discipline_favourite() for the
    same reason build_top_winners() does. This previously used
    `athletes[0]["prob"]` (best mark), which under-reported confidence for
    every discipline where mark and probability disagree: the Women's 3000m
    Steeplechase showed 31% instead of its real 52%."""
    all_discs = (track or []) + (field or [])
    scored = []
    for d in all_discs:
        fav = discipline_favourite(d)
        # discKey travels with the label so a caller can link to the
        # discipline without matching on a display string. `.get` rather than
        # `[...]`: a discipline dict without an id should cost a link, not a
        # 500 on the whole predictions payload.
        scored.append({
            "disc":    d["label"],
            "discKey": d.get("id"),
            "value":   fav["prob"] if fav else 0,
        })
    return sorted(scored, key=lambda x: -x["value"])

MODEL_METRICS_PATH = os.path.join(OUTPUTS_DIR, "model_metrics.json")


def load_model_metrics():
    """Both backtest numbers train_model.py records, or {} before it has
    been re-run since 2026-08-25."""
    try:
        with open(MODEL_METRICS_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def get_model_accuracy():
    """The accuracy the SITE should quote about itself.

    Two different numbers come off the same predictions. The historical one
    in model_accuracy.txt picks 3 from every athlete in a discipline-year's
    toplist (~101 names). This site never does that -- run.py ranks the
    ~8-10 athletes in WA's Diamond League standings, and scored on that task
    the same model reads about 12 points higher (measured 2026-08-25:
    59.7% vs 72.3%).

    Quoting the toplist figure understated the product while describing a
    question nobody asked, so the field figure is preferred when present.
    Falls back to the old file, and finally to a conservative constant, so
    an un-retrained checkout still renders."""
    metrics = load_model_metrics()
    field_pct = metrics.get("final_field_pct")
    if isinstance(field_pct, (int, float)):
        return float(field_pct)
    try:
        return float(open(os.path.join(OUTPUTS_DIR, "model_accuracy.txt")).read().strip())
    except Exception:
        return 44.0


def get_model_accuracy_basis():
    """One line saying what the quoted number measures, so the UI caption is
    a fact rather than a vague reassurance."""
    if isinstance(load_model_metrics().get("final_field_pct"), (int, float)):
        return "top-3 hit rate among the real Final field, walk-forward '23-'25"
    return "walk-forward '23-'25"


def build_discipline_trajectories(disc_key, athletes, limit=4):
    """Real per-meet season form for a discipline's top contenders, reusing
    the exact same real data + logic as the athlete profile page's chart
    (load_athlete_history) -- this is what replaces the Projections page's
    old fabricated single-line 'illustrative trend' curve. Each athlete's
    own real points only, current season if they have it, their own most
    recent completed season otherwise (see load_athlete_history's
    docstring) -- never synthesized between real points."""
    trajectories = []
    for a in athletes[:limit]:
        history, history_year = load_athlete_history(disc_key, a["name"])
        if not history:
            continue
        trajectories.append({
            "name":        a["name"],
            "rank":        a["rank"],
            "prob":        a["prob"],
            "historyYear": history_year,
            "history":     history,
        })
    return trajectories


def _dl_final_history(disc_key):
    """Real DL Final results (2018-2025) for one discipline, or an empty
    DataFrame if the ground-truth file isn't present. Cached per-call is
    unnecessary here -- storyline building happens once per lazy page
    view, same cost class as the rest of build_athlete_profile()."""
    if not os.path.exists(DL_FINAL_RESULTS_PATH):
        return pd.DataFrame()
    df = pd.read_csv(DL_FINAL_RESULTS_PATH)
    return df[df["discipline"] == disc_key]


# How strongly a storyline's real number pulls against the model's own pick.
# The Projections page features exactly ONE storyline at large scale, and
# picking it by the generators' fixed type order meant a routine "#2 First
# Final appearance" got the big treatment while the far more striking real
# fact -- "12-0, Lyles leads the head-to-head, yet the model picks Seville"
# -- was buried in the small list underneath. Every level below is a check
# against real data, so which card gets featured stays computed rather than
# editorial.
#
# The top two tiers are deliberately distinct, and the Men's 100m is exactly
# why: Oblique Seville is simultaneously the model's 27% favourite AND a
# first-time finalist AND the man Noah Lyles has beaten in all 12 of their
# real career meetings. Scoring "debutant favourite" and "the head-to-head
# says otherwise" the same made the debut card win on generator order and
# put the weaker fact in the featured slot. A debut is an ABSENCE of prior
# finals -- it defies an expectation but is not evidence against the pick.
# A 12-0 losing record is measured counter-evidence. They are not the same
# claim and don't get the same weight.
SURPRISE_CONTRADICTS  = 3  # measured evidence points at a different athlete
SURPRISE_DEFIES_PRIOR = 2  # cuts against an expectation, but isn't counter-evidence
SURPRISE_NOTABLE      = 1  # real and worth saying, but agrees with the model
SURPRISE_CONTEXT      = 0  # background colour


def storyline_surprise(story, prob_leader, prob_top3):
    """Score one storyline against the model's own call.

    Relies on a contract every generator in build_storylines() honours: a
    storyline's `athletes[0]` is its protagonist -- the head-to-head leader
    for a rivalry, the flagged athlete for an injury watch, the debutant for
    a debut, and so on. (This project has had real bugs from assuming
    `athletes[0]` meant "the favourite"; here it deliberately does not, which
    is exactly what makes the comparison against `prob_leader` meaningful.)
    """
    kind = story["type"]
    hero = (story["athletes"][0] if story["athletes"] else "").lower()
    leader = (prob_leader or "").lower()
    top3 = {n.lower() for n in prob_top3}

    if kind == "rivalry":
        wins, _, losses = story["stat"].partition("-")
        if wins == losses:
            return SURPRISE_NOTABLE  # level record -- nobody for the model to disagree with
        return SURPRISE_CONTRADICTS if hero and leader and hero != leader else SURPRISE_NOTABLE

    if kind == "injury_watch":
        # A flagged athlete the model still makes favourite is a real tension;
        # one flagged further down the field is a caveat, not a headline.
        if hero and hero == leader:
            return SURPRISE_CONTRADICTS
        return SURPRISE_DEFIES_PRIOR if hero in top3 else SURPRISE_NOTABLE

    if kind == "debutant":
        # A first-timer the model picks to WIN cuts against the obvious prior.
        # A first-timer projected third does not -- that's just a fact.
        return SURPRISE_DEFIES_PRIOR if hero and hero == leader else SURPRISE_CONTEXT

    if kind == "photo_finish":
        # Already gated at <=6pt upstream. Inside ~2pt the model is
        # effectively declining to call it, which is a statement in itself.
        m = re.match(r"(\d+)", story["stat"])
        return SURPRISE_DEFIES_PRIOR if m and int(m.group(1)) <= 2 else SURPRISE_NOTABLE

    if kind in ("returning_champion", "hot_streak"):
        return SURPRISE_NOTABLE

    return SURPRISE_CONTEXT


def rank_storylines(stories, prob_leader, prob_top3, limit=4):
    """Order storylines most-surprising-first and keep the top `limit`.

    Python's sort is stable, so equally-surprising storylines keep the
    generators' original order -- nothing gets shuffled arbitrarily, only
    genuinely contradicting facts move up. Sorting before truncating is
    deliberate: a striking storyline that used to fall outside the cut now
    survives it."""
    return sorted(
        stories, key=lambda s: -storyline_surprise(s, prob_leader, prob_top3)
    )[:limit]


def build_storylines(disc_key, disc_label, athletes):
    """Real, computed narrative angles for a discipline -- replaces the
    Projections page's old static, identical-on-every-discipline 'how this
    page works' text with athlete-specific storylines, each backed by a
    real number pulled live, not written by hand. Every generator either
    returns a real storyline or nothing; only non-empty ones are shown, so
    a discipline with a thin storyline crop (e.g. no debutants) just shows
    fewer cards rather than a fabricated one to fill space.

    Each storyline carries a `stat` (a short, real headline value -- e.g.
    '8pt gap', '12-0') separate from `text` (the supporting sentence) and
    `athletes` (real names, for the frontend to link to their profile).
    Added so the UI can put the real number itself forward as the visual
    anchor instead of burying it inside a paragraph -- the first version
    of this UI put icon+heading+paragraph in same-size boxes, which read
    as generic filler; the fix is structural (a real number to design
    around), not just a restyle.

    Every generator sets `athletes[0]` to that storyline's protagonist --
    see storyline_surprise(), which relies on it to decide which card the
    page features. The returned order is by surprise, not by generator."""
    stories = []
    top = athletes[:8]
    is_track = disc_key not in FIELD_EVENTS

    # Photo finish: real gap between the top two predicted WIN PROBABILITIES
    # -- deliberately re-sorted by prob, not by `athletes`' own rank order
    # (predicted_rank is sorted by real season-best mark, not by the
    # model's probability, so rank #1 isn't always the higher-probability
    # pick; comparing top[0]/top[1] by rank alone produced a real negative
    # "gap" once, confirmed live before this fix).
    by_prob = sorted(top, key=lambda a: -a["prob"])
    prob_leader = by_prob[0]["name"] if by_prob else None
    prob_top3 = [a["name"] for a in by_prob[:3]]
    if len(by_prob) >= 2:
        gap = by_prob[0]["prob"] - by_prob[1]["prob"]
        if gap <= 6:
            stories.append({
                "type":     "photo_finish",
                "title":    "Photo finish",
                "stat":     f"{gap}pt gap",
                # "race" is wrong for the 12 field disciplines -- a shot
                # putter does not race. Same fix as the frontend's
                # startNoun(), applied here because this string is
                # server-generated.
                "text":     f"{by_prob[0]['prob']}% to {by_prob[1]['prob']}% — the closest "
                            f"projected {'contest' if disc_key in FIELD_EVENTS else 'race'} "
                            f"in {disc_label}.",
                "athletes": [by_prob[0]["name"], by_prob[1]["name"]],
            })

    # Injury watch: any real flagged contender still in the predicted field.
    watched = [a for a in top if a["injuryWatch"]]
    if watched:
        w = watched[0]
        stories.append({
            "type":     "injury_watch",
            "title":    "One to watch",
            "stat":     f"#{w['rank']}",
            "text":     f"Flagged for a possible injury or recent DNF "
                        f"({w['injuryReason'] or 'see evidence link'}) but remains projected "
                        f"at rank #{w['rank']} — {w['prob']}% win probability if fit.",
            "athletes": [w["name"]],
        })

    # Debutant / returning champion: real prior-Final participation.
    finals = _dl_final_history(disc_key)
    if not finals.empty:
        names_lower = {n.lower() for n in finals["athlete_name"].dropna()}
        champions = {
            r["athlete_name"]: int(r["year"])
            for _, r in finals[finals["place"] == 1].sort_values("year").iterrows()
        }
        champs_lower = {n.lower(): (n, y) for n, y in champions.items()}
        for a in top[:5]:
            name_lower = a["name"].lower()
            if name_lower in champs_lower:
                champ_name, champ_year = champs_lower[name_lower]
                stories.append({
                    "type":     "returning_champion",
                    "title":    "Returning champion",
                    "stat":     str(champ_year),
                    "text":     f"Won the {champ_year} Diamond League Final in "
                                f"{disc_label} and is projected rank #{a['rank']} to defend it.",
                    "athletes": [a["name"]],
                })
                break
        for a in top[:3]:
            if a["name"].lower() not in names_lower:
                stories.append({
                    "type":     "debutant",
                    "title":    "First Final appearance",
                    "stat":     f"#{a['rank']}",
                    "text":     f"Never made a Diamond League Final in "
                                f"{disc_label} before (no appearance 2018-2025).",
                    "athletes": [a["name"]],
                })
                break

    # Rivalry renewed: real head-to-head record between the top two.
    if len(top) >= 2:
        rivals = load_h2h_vs_rivals(disc_key, top[0]["name"], [top[1]["name"]])
        if rivals:
            r = rivals[0]
            # `wins`/`losses` come back from top[0]'s point of view, and top[0]
            # is only the best season MARK -- which does not make them the
            # better head-to-head athlete. The old text asserted "top[0] leads"
            # unconditionally, so a 2-5 record rendered as a lead. Name whoever
            # actually leads, and put them first in `athletes`: the card reads
            # "A vs. B", and storyline_surprise() treats athletes[0] as the
            # protagonist.
            # The card renders as "A vs. B — {text}", so `text` deliberately
            # does NOT re-name the leader: "A vs. B — A leads across..." said
            # it twice in a row, which is loud now that this card can take the
            # featured slot. First-named is the leader, and that's enough.
            if r["wins"] == r["losses"]:
                stat = f"{r['wins']}-{r['losses']}"
                names = [top[0]["name"], top[1]["name"]]
                text = f"Level across {r['meetings']} real career meetings."
            else:
                lead, trail = (top[0], top[1]) if r["wins"] > r["losses"] else (top[1], top[0])
                won = max(r["wins"], r["losses"])
                stat = f"{won}-{min(r['wins'], r['losses'])}"
                names = [lead["name"], trail["name"]]
                text = f"{won} wins from {r['meetings']} real career meetings"
                # The whole point of letting this card be featured: the career
                # record and the model can point at different athletes.
                if prob_leader and prob_leader != lead["name"]:
                    text += (f", yet the model makes {prob_leader} the "
                             f"{by_prob[0]['prob']}% pick for the Final")
                text += "."
            stories.append({
                "type":     "rivalry",
                "title":    "Rivalry renewed",
                "stat":     stat,
                "text":     text,
                "athletes": names,
            })

    # Hot streak: real season-long improvement across an athlete's own
    # actual 2026 meetings (not the internal recent_trend model feature,
    # which isn't exposed -- this recomputes the same idea transparently
    # from the real per-meet marks so the number in the card is directly
    # checkable against the trajectory chart above it).
    best_gain, best_athlete = None, None
    for a in top:
        history, history_year = load_athlete_history(disc_key, a["name"])
        if history_year != MEETS_YEAR or len(history) < 2:
            continue
        values = [h["markValue"] for h in history if h["markValue"] is not None]
        if len(values) < 2:
            continue
        gain = (values[0] - values[-1]) if is_track else (values[-1] - values[0])
        if gain > 0 and (best_gain is None or gain > best_gain):
            best_gain, best_athlete = gain, (a, history)
    if best_athlete:
        a, history = best_athlete
        stat = f"-{best_gain:.2f}s" if is_track else f"+{best_gain:.2f}m"
        stories.append({
            "type":     "hot_streak",
            "title":    "Trending up",
            "stat":     stat,
            "text":     f"{history[0]['mark']} → {history[-1]['mark']} across "
                        f"{len(history)} real meetings this season — the biggest real "
                        f"in-season gain among the top contenders.",
            "athletes": [a["name"]],
        })

    return rank_storylines(stories, prob_leader, prob_top3)


@app.route("/api/predictions")
def predictions():
    track, field = load_predictions()
    if track is None:
        return jsonify({"error": "predictions_latest.csv not found — run python run.py first"}), 404

    return jsonify({
        "lastUpdated":   str(date.today().strftime("%d %b %Y")),
        "daysToFinal":   (date(2026, 9, 4) - date.today()).days,
        "meets":         compute_meet_statuses(MEETS),
        "trackDisciplines": track,
        "fieldDisciplines": field,
        "topWinners":    build_top_winners(track, field),
        "removedAthletes": build_removed_athletes(load_injury_flags()),
        "confidence":    build_confidence(track, field),
        "modelAccuracy": get_model_accuracy(),
        "modelAccuracyBasis": get_model_accuracy_basis(),
        # The historical ~101-athlete-toplist figure, kept visible rather
        # than quietly replaced -- every number in HANDOFF.md is this one.
        "modelAccuracyToplist": load_model_metrics().get("toplist_pool_pct"),
    })


@app.route("/api/athlete/<disc_key>/<athlete_name>")
def athlete_profile(disc_key, athlete_name):
    profile = build_athlete_profile(disc_key, athlete_name)
    if profile is None:
        return jsonify({"error": "athlete not found"}), 404
    return jsonify(profile)


@app.route("/api/projections/<disc_key>")
def projections_detail(disc_key):
    """Real per-discipline detail for the Projections page: the top
    contenders' actual season trajectories (replacing the old fabricated
    illustrative curve) plus real, computed storylines -- both lazy, same
    pattern as /api/athlete/<discKey>/<name>, not embedded in the bulk
    /api/predictions payload."""
    track, field = load_predictions()
    if track is None:
        return jsonify({"error": "predictions_latest.csv not found — run python run.py first"}), 404
    disc = next((d for d in track + field if d["id"] == disc_key), None)
    if disc is None:
        return jsonify({"error": "discipline not found"}), 404
    names = [a["name"] for a in disc["athletes"]]
    return jsonify({
        "trajectories": build_discipline_trajectories(disc_key, disc["athletes"]),
        "storylines":   build_storylines(disc_key, disc["label"], disc["athletes"]),
        # How this field's athletes compare to EACH OTHER, which is the
        # question a ranked list with probabilities beside it cannot answer.
        # Viable because the pairs genuinely exist: measured across all 32
        # 2026 fields, the median discipline has raced 100% of its possible
        # pairings and the worst is 82%.
        "fieldAnalysis": athlete_analytics.build_field_analysis(
            disc_key, names, disc_key in FIELD_EVENTS,
        ),
    })


def ordinal(n):
    return f"{n}{'th' if 11 <= n % 100 <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def standings_position(disc_key, athlete_name):
    """Where an athlete actually sits in WA's FULL Diamond League standings
    for a discipline -- points, rank, and the gap to the cut. None when they
    genuinely have no Diamond League points in it.

    standings.json is truncated to the qualifying places, so on its own it
    cannot tell "has never scored here" apart from "has scored but is 9th".
    The site asserted the former for both, which made it tell readers that
    Noah Lyles had no Diamond League points in the 100m when he was 9th on
    15, two short of the cut."""
    entry = (load_standings_detail().get("disciplines") or {}).get(disc_key)
    if not entry:
        return None
    limit = entry.get("qualLimit") or get_qual_limit(disc_key)
    race = qualification_race(entry.get("standings") or [], limit, meetings_remaining())
    row = next((r for r in race["rows"]
                if str(r.get("name", "")).lower() == athlete_name.lower()), None)
    if row is None:
        return None
    return {**row, "qualLimit": limit, "cutPoints": race["cutPoints"]}


def points_cut_reason(label, dl):
    """Why an athlete with real Diamond League points still isn't in the
    field -- stated as the arithmetic it is, not as a guess."""
    reason = (f"{ordinal(dl['rank'])} in the {label} Diamond League standings on "
              f"{dl['points']} points, outside the top {dl['qualLimit']} who qualify "
              f"for the Final.")
    if dl["status"] == "out":
        return reason + " Too far back to be caught up now."
    gap = dl["gap"]
    if gap is None:
        return reason
    if gap > 0:
        return reason + f" {gap} point{'' if gap == 1 else 's'} short of the cut."
    return reason + " Level on points with the cut, behind on World Athletics' tie-break."


# ---------------------------------------------------------------------------
# Cross-discipline performance stats.
#
# Built on `Results Score`, World Athletics' own scoring-table points, which
# is present on 100% of toplist rows and was read by nothing until now. It is
# the only number in this dataset that compares a shot putter to a 1500m
# runner, so it is what makes "best performance of the season, any event"
# answerable at all. Measured 2026 across a uniform top 100 per discipline
# (see TOPLIST_DEPTH -- an earlier reading of "939 (women's SP)" here was an
# artefact of two events being scraped 500 deep): it ranges 1007-1353 with
# per-discipline medians spanning 1053 (women's JT) to 1206 (men's 1500m,
# level with men's 100m), so it is broadly
# comparable but NOT perfectly flat -- the API reports each discipline's own
# median alongside so a reader can see the baseline rather than assume one.
# ---------------------------------------------------------------------------

# World Athletics writes indoor marks into its outdoor toplists, tagged only
# by a "(i)" suffix on the venue (or an obviously indoor venue name). This is
# not cosmetic: 13.0% of 2026 toplist rows are indoor, and in the vertical
# jumps it is nearly half (men's HJ 47%, women's HJ 44%, women's PV 41%,
# men's PV 39%). Duplantis's 2026 season best, the highest-scoring
# performance in the whole dataset, was set indoors in Uppsala. Those marks
# are legitimate performances and are NOT filtered out -- for a vault or a
# shot put, indoors is arguably the truer measure of ability. They are
# LABELLED instead, so the site never presents one as an outdoor mark
# without saying so.
INDOOR_VENUE = re.compile(r"\(i\)\s*$|\bindoor\b", re.I)

_season_scores_cache = {}


def is_indoor_venue(venue):
    return bool(INDOOR_VENUE.search(str(venue or "")))


# Every cross-discipline number on the Performance Index is a comparison
# between disciplines, so all 32 have to be sampled to the SAME depth or the
# comparison is measuring the scrape, not the sport.
#
# They are not all the same depth on disk. `scrape_toplist()` pages past the
# top 100 when a DL-qualified athlete hasn't appeared yet -- an athlete who
# qualified on Diamond League points rather than raw mark may sit outside the
# world top 100 -- so it walks up to page 5. That is correct for predictions
# and wrong to reuse here unfiltered: it left women's SP and women's 5000m
# with 500 rows against everyone else's 100.
#
# Untruncated, that put women's 5000m 29th of 32 by median score and women's
# SP dead last on a median of 939, which the site rendered as "how deep each
# event is". Both were artefacts of ranks 101-500 being included for those
# two events alone. Capped, women's 5000m is 15th (median 1175, +14 places)
# and women's SP's median is 1067, not 939.
#
# The cap is applied where disciplines are COMPARED, not in the loader. Those
# extra rows are the whole reason the deeper pages exist: the athlete they
# were fetched for is a real finalist who qualified on Diamond League points
# from outside the world top 100, and dropping them at load time would erase
# that athlete's score from their own profile page.
TOPLIST_DEPTH = 100


def to_uniform_depth(df, depth=TOPLIST_DEPTH):
    """The same number of ranked athletes from every discipline, so a
    cross-discipline figure measures the sport rather than the scrape."""
    if df.empty:
        return df
    if "Rank" in df.columns:
        df = df.sort_values("Rank", kind="stable")
    return df.groupby("discKey", group_keys=False, sort=False).head(depth)


def _toplist_paths(year):
    return [(k, os.path.join(RAW_DIR, f"{k}_{year}.csv")) for k in DISC_LABELS]


def load_season_scores(year=None):
    """Every discipline's toplist for a season in one frame, with WA's
    Results Score and an `indoor` flag.

    Cached on the FILES, not just the year: the key includes RAW_DIR and
    every toplist's mtime. A plain per-year cache looked fine and was
    wrong twice over -- it kept serving pre-refresh numbers after a run.py
    rewrote the toplists (this app's stated contract is that both dev
    servers re-read data fresh on every request), and it leaked fixture
    data between tests that repoint RAW_DIR. Stat-ing 32 files costs
    nothing next to re-reading them."""
    year = year or MEETS_YEAR
    paths = _toplist_paths(year)
    key = (RAW_DIR, year, tuple(
        os.path.getmtime(p) if os.path.exists(p) else None for _, p in paths
    ))
    cached = _season_scores_cache.get(key)
    if cached is not None:
        return cached

    frames = []
    for disc_key, path in paths:
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if "Results Score" not in df.columns:
            continue
        cols = ["Competitor", "Mark", "Results Score", "Venue", "Date"]
        # Keep WA's own rank where the scrape preserved it: the depth cap
        # below has to mean "the world top 100", not "the first 100 rows this
        # file happens to hold".
        df = df[cols + (["Rank"] if "Rank" in df.columns else [])].copy()
        df["discKey"] = disc_key
        frames.append(df)

    if not frames:
        out = pd.DataFrame(columns=["Competitor", "Mark", "Results Score",
                                    "Venue", "Date", "discKey", "indoor"])
    else:
        out = pd.concat(frames, ignore_index=True)
        out = out.dropna(subset=["Results Score", "Competitor"])
        out["indoor"] = out["Venue"].map(is_indoor_venue)
    # One entry per distinct file-state; the dict would otherwise grow by one
    # every refresh for the lifetime of the process.
    _season_scores_cache.clear()
    _season_scores_cache[key] = out
    return out


def _performance_row(row):
    disc_key = row["discKey"]
    value = parse_mark(row["Mark"])
    return {
        "athlete":  row["Competitor"],
        "discKey":  disc_key,
        "disc":     DISC_LABELS.get(disc_key, disc_key),
        "isField":  disc_key in FIELD_EVENTS,
        # Fall back to WA's raw string rather than dropping the row: a mark
        # that won't parse is still a real performance with a real score.
        "mark":     format_mark(value, disc_key) if value is not None else str(row["Mark"]),
        "score":    int(row["Results Score"]),
        "venue":    row["Venue"],
        "date":     row["Date"],
        "indoor":   bool(row["indoor"]),
    }


_corpus_cache = {}


def _corpus_paths():
    """The historical training files -- `{disc}.csv`, not the `_{year}` ones,
    which are this season's toplists rather than the corpus."""
    return [os.path.join(RAW_DIR, f"{k}.csv") for k in DISC_LABELS]


def build_training_corpus():
    """What the model was actually trained on, counted rather than claimed.

    This exists because the landing page carried "+ dozens more meetings,
    7 seasons" beside a six-row sample of real meeting names, and both
    halves were wrong: it is thousands of competitions, not dozens, across
    eight seasons, not seven. Neither number was read from anything -- they
    were typed next to real data, which is the shape of mistake this
    project keeps finding in its plumbing rather than its model.

    `competitions` counts distinct (venue, date) pairs. That is the honest
    unit available here: the raw rows carry where and when, not a meeting
    id, so a two-day meeting counts as two and a venue hosting several
    meetings a year counts each. It is a count of competition days, and
    `venues` is reported beside it so neither has to stand alone.

    Cached on the files' mtimes, matching load_season_scores -- the same
    reason applies: this app re-reads data on every request, and a plain
    cache would serve pre-refresh figures after a run.py."""
    paths = _corpus_paths()
    key = (RAW_DIR, tuple(
        os.path.getmtime(p) if os.path.exists(p) else None for p in paths
    ))
    cached = _corpus_cache.get(key)
    if cached is not None:
        return cached

    marks = 0
    seasons, venues, competitions = set(), set(), set()
    for path in paths:
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        marks += len(df)
        if "year" in df.columns:
            seasons |= {int(y) for y in df["year"].dropna()}
        if {"Venue", "Date"} <= set(df.columns):
            pairs = df[["Venue", "Date"]].dropna().astype(str)
            venues |= set(pairs["Venue"])
            competitions |= set(map(tuple, pairs.values))

    result = None if not marks else {
        "marks":        marks,
        "seasons":      len(seasons),
        "firstSeason":  min(seasons) if seasons else None,
        "lastSeason":   max(seasons) if seasons else None,
        "venues":       len(venues),
        "competitions": len(competitions),
    }
    _corpus_cache[key] = result
    return result


def build_stats(top_n=40):
    """The season ranked by WA score rather than by event.

    Deliberately reports each discipline's own median next to the headline
    list: a 1300 in the men's discus and a 1300 in the women's shot put are
    the same number against very different baselines, and a leaderboard that
    hides that is a ranking pretending to be a fact."""
    # Every figure on this page compares disciplines, so all 32 are read to
    # the same depth -- see TOPLIST_DEPTH.
    df = to_uniform_depth(load_season_scores())
    if df.empty:
        return {"topPerformances": [], "disciplineDepth": [], "scoreScale": None,
                "indoor": None, "corpus": build_training_corpus()}

    scores = df["Results Score"]
    depth = []
    for disc_key, group in df.groupby("discKey"):
        depth.append({
            "discKey":      disc_key,
            "disc":         DISC_LABELS.get(disc_key, disc_key),
            "isField":      disc_key in FIELD_EVENTS,
            "athletes":     int(len(group)),
            "medianScore":  int(group["Results Score"].median()),
            "topScore":     int(group["Results Score"].max()),
            # How much of this discipline's toplist was set indoors. Worth
            # showing per discipline rather than as one site-wide number --
            # it is 0% for most track events and ~47% for the high jump.
            "indoorShare":  round(float(100 * group["indoor"].mean()), 1),
        })
    depth.sort(key=lambda d: -d["medianScore"])

    top = df.nlargest(top_n, "Results Score")
    return {
        "topPerformances": [_performance_row(r) for _, r in top.iterrows()],
        "disciplineDepth": depth,
        "scoreScale": {
            "min":    int(scores.min()),
            "max":    int(scores.max()),
            "median": int(scores.median()),
            "rows":   int(len(df)),
        },
        "indoor": {
            "rows":  int(df["indoor"].sum()),
            "total": int(len(df)),
            "share": round(float(100 * df["indoor"].mean()), 1),
        },
        # What the model learned from, counted rather than typed.
        "corpus": build_training_corpus(),
        "season": MEETS_YEAR,
    }


def athlete_score_context(disc_key, athlete_name):
    """One athlete's WA score, and where it sits against everyone the site
    tracks. `percentile` is across ALL disciplines -- that is the whole
    point of using WA's score -- while `discPercentile` keeps the
    within-event reading that a discipline table already implies.

    The athlete is looked up in the FULL toplist so nobody loses their score
    for ranking outside the world top 100, but both percentiles and the
    median are measured against the uniform top 100 -- otherwise the same
    "99th percentile" would mean 99th of 100 on one page and 99th of 500 on
    another, and the profile's discMedian would contradict the Performance
    Index's median for the same event."""
    full = load_season_scores()
    if full.empty:
        return None
    mine = full[(full["discKey"] == disc_key) & (full["Competitor"] == athlete_name)]
    if mine.empty:
        return None
    row = mine.nlargest(1, "Results Score").iloc[0]
    score = float(row["Results Score"])
    df = to_uniform_depth(full)
    same = df[df["discKey"] == disc_key]["Results Score"]
    if same.empty:
        return None
    return {
        "score":          int(score),
        "percentile":     round(float(100 * (df["Results Score"] <= score).mean()), 1),
        "discPercentile": round(float(100 * (same <= score).mean()), 1),
        "discMedian":     int(same.median()),
        "indoor":         bool(row["indoor"]),
        "venue":          row["Venue"],
        # True for an athlete who qualified for the Final on Diamond League
        # points from outside WA's top 100 -- the reason those two toplists
        # were scraped 500 deep. Their percentiles are floored at 0 by
        # construction, so the UI needs to be able to say why.
        "outsideTopList": bool(score < same.min()),
    }


# ---------------------------------------------------------------------------
# Discipline vs discipline.
#
# The third level of the site (discipline -> field -> athlete) and the one
# that did not exist: which events are genuinely deep and which are one
# athlete and a gap.
#
# Measured on World Athletics' Results Score and NOTHING ELSE, for a reason
# worth stating. The obvious measure is the model's own probabilities -- a
# field where nobody clears 40% looks wide open. They cannot be used for a
# comparison BETWEEN disciplines: the target is top-three membership and each
# athlete is scored independently, so a field's probabilities do not sum to
# any fixed total. Measured across the 32 real 2026 fields they sum to
# anywhere from 31 (men's 1500m) to 320 (women's 5000m), median 171. Ranking
# disciplines by "how many athletes clear 25%" would therefore rank the
# model's per-event confidence, not the depth of the field. The probabilities
# order athletes correctly WITHIN a discipline, and that is where the site
# uses them.
#
# WA's score has neither problem: it is scraped, it is on every toplist row,
# and it is the one number in this data that compares a shot putter to a
# 1500m runner. Sampled to a uniform depth (TOPLIST_DEPTH) it is comparable
# across all 32.


def _field_scores(scores_df, disc_key, athletes):
    """Each finalist's best WA score this season, highest first.

    Looked up in the full toplist rather than the uniform top 100: a finalist
    who qualified on Diamond League points can rank outside it, and reporting
    a field's spread having silently dropped its weakest member would make a
    top-heavy field look level."""
    same = scores_df[scores_df["discKey"] == disc_key]
    out = []
    for a in athletes:
        mine = same[same["Competitor"] == a["name"]]["Results Score"]
        if len(mine):
            out.append({"name": a["name"], "score": int(mine.max()), "prob": a["prob"]})
    return sorted(out, key=lambda r: -r["score"])


def build_depth_index(year=None):
    """All 32 finals on one scale, tightest field first.

    `spread` is the WA-score distance from the best finalist to the weakest
    one. A small spread means the field is level on measured ability; a large
    one means the entry list has a top and a tail."""
    track, field = load_predictions()
    if track is None:
        return []
    full = load_season_scores(year)
    if full.empty:
        return []
    uniform = to_uniform_depth(full)

    rows = []
    for disc in track + field:
        athletes = disc["athletes"]
        scored = _field_scores(full, disc["id"], athletes)
        if len(scored) < 2:
            continue
        same = uniform[uniform["discKey"] == disc["id"]]["Results Score"]
        probs = sorted((a["prob"] for a in athletes), reverse=True)
        rows.append({
            "discKey":     disc["id"],
            "disc":        disc["label"],
            "isField":     disc["id"] in FIELD_EVENTS,
            "fieldSize":   len(athletes),
            # How many of the field carried a score at all. Shown rather than
            # assumed: a spread computed over 5 of 8 athletes is a different
            # claim from one computed over all 8.
            "scored":      len(scored),
            "spread":      scored[0]["score"] - scored[-1]["score"],
            "bestScore":   scored[0]["score"],
            "bestAthlete": scored[0]["name"],
            "worstScore":  scored[-1]["score"],
            # The world top-100 median for this event, so the field can be
            # placed against the discipline it is drawn from.
            "toplistMedian": int(same.median()) if len(same) else None,
            # Within-discipline only -- see the note above on why these are
            # never ranked across disciplines.
            "favouriteProb": probs[0],
            "probGap":       probs[0] - probs[1] if len(probs) > 1 else None,
        })

    rows.sort(key=lambda r: r["spread"])
    for i, r in enumerate(rows, start=1):
        r["spreadRank"] = i
    return rows


# A field is called level or top-heavy by where its spread falls among all
# 32, not against a number picked by hand -- the same "state the arithmetic"
# rule the Qualifying page follows. Terciles of the real 2026 spreads fall at
# 55 and 85 WA points.
DEPTH_VERDICTS = {
    "level":     ("LEVEL FIELD",  "one of the tightest thirds of the 32 finals"),
    "mixed":     ("A TOP AND A TAIL", "the middle third of the 32 finals"),
    "topHeavy":  ("ONE AND A GAP", "one of the widest thirds of the 32 finals"),
}


def depth_verdict(rank, total):
    """Tercile of the spread ranking. Stated as the arithmetic it is."""
    if total < 3:
        return None
    if rank <= total / 3:
        key = "level"
    elif rank > 2 * total / 3:
        key = "topHeavy"
    else:
        key = "mixed"
    label, basis = DEPTH_VERDICTS[key]
    return {"key": key, "label": label, "basis": basis}


@app.route("/api/stats")
def stats():
    payload = build_stats()
    if not payload["topPerformances"]:
        return jsonify({"error": f"no {MEETS_YEAR} toplists found — run python run.py first"}), 404
    return jsonify(payload)


@app.route("/api/depth")
def depth_index():
    """All 32 finals ranked by how level they are. The comparison the
    Performance Index's flat 32-row table could not make."""
    rows = build_depth_index()
    if not rows:
        return jsonify({"error": "no predictions or toplists yet — run python run.py first"}), 404
    for r in rows:
        r["verdict"] = depth_verdict(r["spreadRank"], len(rows))
    return jsonify({"disciplines": rows, "total": len(rows), "season": MEETS_YEAR,
                    "toplistDepth": TOPLIST_DEPTH})


@app.route("/api/discipline/<disc_key>")
def discipline_report(disc_key):
    """One discipline read as a field: how level it is against the other 31,
    who is in it, and who has actually raced whom.

    Composed from the existing pieces rather than recomputed -- the matrix
    and the per-athlete comparison are the same ones the Projections page
    uses, so the two pages cannot drift apart."""
    track, field = load_predictions()
    if track is None:
        return jsonify({"error": "predictions_latest.csv not found — run python run.py first"}), 404
    disc = next((d for d in track + field if d["id"] == disc_key), None)
    if disc is None:
        return jsonify({"error": "discipline not found"}), 404

    index = build_depth_index()
    mine = next((r for r in index if r["discKey"] == disc_key), None)
    if mine is not None:
        mine = {**mine, "verdict": depth_verdict(mine["spreadRank"], len(index)),
                "of": len(index)}

    names = [a["name"] for a in disc["athletes"]]
    return jsonify({
        "discKey": disc_key,
        "disc":    disc["label"],
        "isField": disc_key in FIELD_EVENTS,
        "season":  MEETS_YEAR,
        "athletes": disc["athletes"],
        "depth":    mine,
        # Each finalist's own WA score, so the spread above is inspectable
        # rather than asserted.
        "scores":   _field_scores(load_season_scores(), disc_key, disc["athletes"]),
        # Absorbed from /api/projections/<key> when the two pages merged:
        # Projections and this page were both "everything about one event",
        # and rendered the same matrix and the same ranked field. These two
        # blocks were the only things unique to Projections, so they moved
        # here rather than being lost with the route.
        "trajectories": build_discipline_trajectories(disc_key, disc["athletes"]),
        "storylines":   build_storylines(disc_key, disc["label"], disc["athletes"]),
        "fieldAnalysis": athlete_analytics.build_field_analysis(
            disc_key, names, disc_key in FIELD_EVENTS,
        ),
    })


@app.route("/api/qualification")
def qualification():
    """Standings points and the gap to the qualification cut. Returns 404
    with a runnable fix rather than an empty page when the standings
    scraper hasn't produced data/standings_detail.json yet."""
    payload = build_qualification()
    if not payload["disciplines"]:
        return jsonify({"error": "standings_detail.json not found — run python src/live_fetcher.py first"}), 404
    return jsonify(payload)


@app.route("/api/news")
def news():
    flags_meta = {}
    try:
        with open(INJURY_FLAGS_PATH, encoding="utf-8") as f:
            raw = json.load(f)
        flags_meta = {"checkedAt": raw.get("checked_at"), "sources": raw.get("sources_ok", [])}
    except Exception:
        pass
    return jsonify({"items": build_news(), **flags_meta})


@app.route("/api/search")
def search():
    return jsonify({"results": search_athletes(request.args.get("q", ""))})


@app.route("/api/athlete-status/<disc_key>/<path:athlete_name>")
def athlete_status(disc_key, athlete_name):
    if disc_key not in DISC_LABELS:
        return jsonify({"error": "unknown discipline"}), 404
    return jsonify(athlete_field_status(disc_key, athlete_name))


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "date": str(date.today())})

if __name__ == "__main__":
    print("Starting DL Predictor API on http://localhost:5000")
    print("Make sure you've run: python run.py")
    app.run(debug=True, port=5000)