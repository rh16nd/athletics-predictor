"""
api.py — Flask bridge between run.py predictions and the React dashboard
Run with: python api.py
Serves at: http://localhost:5000
"""

from flask import Flask, jsonify
from flask_cors import CORS
import pandas as pd
import json
import os
from datetime import date

app = Flask(__name__)
CORS(app)  # allows React dev server to call this API

OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "outputs")
RAW_DIR     = os.path.join(os.path.dirname(__file__), "data", "raw")

MEETS = [
    {"n": 1,  "date": "08 May", "city": "Doha",            "status": "done"},
    {"n": 2,  "date": "16 May", "city": "Shanghai",        "status": "done"},
    {"n": 3,  "date": "23 May", "city": "Xiamen",          "status": "done"},
    {"n": 4,  "date": "31 May", "city": "Rabat",           "status": "done"},
    {"n": 5,  "date": "04 Jun", "city": "Rome",            "status": "done"},
    {"n": 6,  "date": "07 Jun", "city": "Stockholm",       "status": "done"},
    {"n": 7,  "date": "10 Jun", "city": "Oslo",            "status": "done"},
    {"n": 8,  "date": "26 Jun", "city": "Paris",           "status": "done"},
    {"n": 9,  "date": "04 Jul", "city": "Eugene",          "status": "done"},
    {"n": 10, "date": "10 Jul", "city": "Monaco",          "status": "done"},
    {"n": 11, "date": "18 Jul", "city": "London",          "status": "done"},
    {"n": 12, "date": "21 Aug", "city": "Lausanne",        "status": "next"},
    {"n": 13, "date": "23 Aug", "city": "Silesia",         "status": "upcoming"},
    {"n": 14, "date": "27 Aug", "city": "Zürich",          "status": "upcoming"},
    {"n": 15, "date": "04 Sep", "city": "Brussels — Final","status": "final"},
]

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


def load_predictions():
    """Load predictions_latest.csv and build discipline data."""
    path = os.path.join(OUTPUTS_DIR, "predictions_latest.csv")
    if not os.path.exists(path):
        return None, None

    df = pd.read_csv(path)
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

            wa_url = f"https://www.worldathletics.org/search/?q={row['athlete_name'].replace(' ', '+')}"

            athletes.append({
                "rank":      int(row.get("predicted_rank", len(athletes) + 1)),
                "name":      row["athlete_name"],
                "nat":       str(row.get("nationality", "—")),
                "qualified": True,
                "mark":      mark_display,
                "prob":      prob,
                "waUrl":     wa_url,
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


def build_top_winners(track, field):
    all_discs = (track or []) + (field or [])
    winners = []
    medals = ["🥇", "🥈", "🥉", "🏅", "🏅", "🏅"]
    for disc in all_discs:
        if disc["athletes"]:
            a = disc["athletes"][0]
            winners.append({
                "medal": medals[len(winners)] if len(winners) < len(medals) else "🏅",
                "name":  a["name"],
                "disc":  disc["label"],
                "mark":  a["mark"],
                "prob":  a["prob"],
                "waUrl": a["waUrl"],
            })
    return sorted(winners, key=lambda x: -x["prob"])[:6]


def build_confidence(track, field):
    all_discs = (track or []) + (field or [])
    return [
        {"disc": d["label"], "value": d["athletes"][0]["prob"] if d["athletes"] else 0}
        for d in sorted(all_discs, key=lambda x: -(x["athletes"][0]["prob"] if x["athletes"] else 0))
    ]


@app.route("/api/predictions")
def predictions():
    track, field = load_predictions()
    if track is None:
        return jsonify({"error": "predictions_latest.csv not found — run python run.py first"}), 404

    return jsonify({
        "lastUpdated":   str(date.today().strftime("%d %b %Y")),
        "daysToFinal":   (date(2026, 9, 4) - date.today()).days,
        "meets":         MEETS,
        "trackDisciplines": track,
        "fieldDisciplines": field,
        "topWinners":    build_top_winners(track, field),
        "confidence":    build_confidence(track, field),
        "modelAccuracy": 46.2,
    })


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "date": str(date.today())})

if __name__ == "__main__":
    print("Starting DL Predictor API on http://localhost:5000")
    print("Make sure you've run: python run.py")
    app.run(debug=True, port=5000)