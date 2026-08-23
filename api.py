"""
api.py — Flask bridge between run.py predictions and the React dashboard
Run with: python api.py
Serves at: http://localhost:5000
"""

from flask import Flask, jsonify
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

app = Flask(__name__)
CORS(app)  # allows React dev server to call this API

OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "outputs")
RAW_DIR     = os.path.join(os.path.dirname(__file__), "data", "raw")
INJURY_FLAGS_PATH = os.path.join(os.path.dirname(__file__), "data", "injury_flags.json")
H2H_PATH    = os.path.join(os.path.dirname(__file__), "data", "h2h", "h2h_rates.csv")
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


def load_athlete_history(disc_key, athlete_name):
    """Real per-meet marks from the athlete's own most recent *completed*
    season with real data on record -- never 2026 (the live 2026 toplist has
    exactly one row per athlete, a season-best snapshot, not a results log,
    see build_2026_features()'s own comment on this in run.py, so there's no
    real in-season trend to show for the current year yet; data/raw/*.csv
    only ever holds 2018-2025 anyway, 2026 lives in a separate file this
    function doesn't read). Showing last season's actual meet-by-meet marks
    is honest (real dates/venues, clearly a prior season) where synthesizing
    a 2026 trend from a single point would repeat exactly the
    fabricated-interpolation mistake already flagged on the Projections
    page.

    'Most recent' is picked per-athlete, not as one fixed year for
    everyone: this dataset only covers the Diamond League circuit + major
    meets (see Known Limitations), so an athlete who was hurt, skipped the
    circuit, or focused elsewhere in 2025 can have zero rows that year while
    genuinely having a real, fuller season on record from an earlier year --
    e.g. Rai Benjamin, Shaunae Miller-Uibo, and Mutaz Barshim all have no
    2025 rows here but real 2023/2024 (or earlier) seasons. Using the
    dataset's global max year for every athlete showed those as having 'no
    history on record' even though they do -- this uses each athlete's own
    max year instead. Returns [] only if the athlete truly has no historical
    rows at all (some newer athletes won't)."""
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
    season = season.copy()
    season["_date"] = pd.to_datetime(season["Date"], format="%d %b %Y", errors="coerce")
    season = season.sort_values("_date")
    # The same real race often exists twice (once from the toplist scrape,
    # once from season_results_scraper.py/major_meets_scraper.py under a
    # differently-formatted venue name) -- an accepted duplication for the
    # aggregate training features (see season_results_scraper.py's
    # docstring), but showing literally the same race twice in a per-athlete
    # list reads as a display bug, not real data. Keep one row per
    # (date, mark), preferring whichever copy has a Results Score.
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
    return history, last_year


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
        if disc["athletes"]:
            a = disc["athletes"][0]
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
    all_discs = (track or []) + (field or [])
    return [
        {"disc": d["label"], "value": d["athletes"][0]["prob"] if d["athletes"] else 0}
        for d in sorted(all_discs, key=lambda x: -(x["athletes"][0]["prob"] if x["athletes"] else 0))
    ]

def get_model_accuracy():
    acc_path = os.path.join(OUTPUTS_DIR, "model_accuracy.txt")
    try:
        return float(open(acc_path).read().strip())
    except:
        return 44.0
    
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


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "date": str(date.today())})

if __name__ == "__main__":
    print("Starting DL Predictor API on http://localhost:5000")
    print("Make sure you've run: python run.py")
    app.run(debug=True, port=5000)