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
MEETS = [
    {"n": 1,  "date": "08 May", "city": "Doha"},
    {"n": 2,  "date": "16 May", "city": "Shanghai"},
    {"n": 3,  "date": "23 May", "city": "Xiamen"},
    {"n": 4,  "date": "31 May", "city": "Rabat"},
    {"n": 5,  "date": "04 Jun", "city": "Rome"},
    {"n": 6,  "date": "07 Jun", "city": "Stockholm"},
    {"n": 7,  "date": "10 Jun", "city": "Oslo"},
    {"n": 8,  "date": "26 Jun", "city": "Paris"},
    {"n": 9,  "date": "04 Jul", "city": "Eugene"},
    {"n": 10, "date": "10 Jul", "city": "Monaco"},
    {"n": 11, "date": "18 Jul", "city": "London"},
    {"n": 12, "date": "21 Aug", "city": "Lausanne"},
    {"n": 13, "date": "23 Aug", "city": "Silesia"},
    {"n": 14, "date": "27 Aug", "city": "Zürich"},
    {"n": 15, "date": "04 Sep", "city": "Brussels — Final"},
]


def compute_meet_statuses(meets, today=None):
    today = today or date.today()
    result = []
    next_assigned = False
    last_index = len(meets) - 1

    for i, meet in enumerate(meets):
        if i == last_index:
            result.append({**meet, "status": "final"})
            continue
        try:
            meet_date = datetime.strptime(f"{meet['date']} {MEETS_YEAR}", "%d %b %Y").date()
        except ValueError:
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
    """Real head-to-head record vs. this discipline's other top predicted
    contenders -- data/h2h/h2h_rates.csv already has 156k+ real matchup rows
    scraped from World Athletics, but until now it was only ever consumed as
    one blended win-rate number fed into the model (see train_model.py's
    add_h2h_features) -- never shown to a user. Both directions of a pair
    are stored as separate rows (verified live), so a direct athlete_a match
    is enough, no need to also check the reverse."""
    if not os.path.exists(H2H_PATH) or not rival_names:
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
    
    def get_model_accuracy():
        acc_path = os.path.join(OUTPUTS_DIR, "model_accuracy.txt")
        try:
            return float(open(acc_path).read().strip())
        except:
            return 44.0
        
    df = pd.read_csv(path)
    injury_flags = load_injury_flags()
    track = []
    field = []

    for disc_key, label in DISC_LABELS.items():
        disc_df = df[df["discipline"] == label].copy()
        if disc_df.empty:
            continue

        athletes = []
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

            athletes.append({
                "rank":         int(row.get("predicted_rank", len(athletes) + 1)),
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

        disc_obj = {
            "id":       disc_key,
            "label":    label,
            "athletes": sorted(athletes, key=lambda x: x["rank"]),
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

    standings = load_standings().get(disc_key, [])
    in_standings = any(n.lower() == athlete_name.lower() for n in standings)
    if standings and not in_standings:
        out["reasonCode"] = "not_in_standings"
        out["reason"] = (
            f"Not in World Athletics' official Diamond League standings for "
            f"{label}. Only athletes who have scored Diamond League points in "
            f"this discipline are eligible for the Final, regardless of how "
            f"fast they have run elsewhere."
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
        for m in entry.get("matches") or []:
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

    rivals_df = disc_df[disc_df["athlete_name"].str.lower() != athlete_name.lower()]
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
        "daysSinceLast":   clean(row.get("days_since_last")),
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
        scored.append({"disc": d["label"], "value": fav["prob"] if fav else 0})
    return sorted(scored, key=lambda x: -x["value"])

def get_model_accuracy():
    acc_path = os.path.join(OUTPUTS_DIR, "model_accuracy.txt")
    try:
        return float(open(acc_path).read().strip())
    except:
        return 44.0


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
                "text":     f"{by_prob[0]['prob']}% to {by_prob[1]['prob']}% — the closest "
                            f"projected race in {disc_label}.",
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
    return jsonify({
        "trajectories": build_discipline_trajectories(disc_key, disc["athletes"]),
        "storylines":   build_storylines(disc_key, disc["label"], disc["athletes"]),
    })


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