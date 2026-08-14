import subprocess
import sys
import os
import pickle
import pandas as pd
import numpy as np
from datetime import date
import json
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("=" * 60)
print("   ATHLETICS PREDICTOR — 2026 DL PREDICTIONS")
print(f"   Running: {date.today().strftime('%B %d, %Y')}")
print("=" * 60)

print("\n[1/4] Fetching live data from World Athletics + DL standings...")
result = subprocess.run(
    [sys.executable, "-u", "src/live_fetcher.py"],
    capture_output=True, text=True, encoding='utf-8'
)

if result.returncode != 0:
    print("  ERROR in live fetcher:")
    print(result.stderr[-500:])
    sys.exit(1)

for line in result.stdout.split('\n'):
    if any(k in line for k in ["Saved", "athletes", "ERROR", "Done"]):
        print(f"  {line.strip()}")

standings_path = os.path.join("data", "standings.json")
if os.path.exists(standings_path):
    with open(standings_path, encoding="utf-8") as f:
        standings = json.load(f)
else:
    standings = {}

print("  Done.")

print("\n[2/4] Loading trained model...")
OUTPUTS_DIR = "outputs"
MODEL_PATH    = os.path.join(OUTPUTS_DIR, "model_rf.pkl")
SCALER_PATH   = os.path.join(OUTPUTS_DIR, "scaler.pkl")
FEATURES_PATH = os.path.join(OUTPUTS_DIR, "feature_cols.pkl")

if not os.path.exists(MODEL_PATH):
    print("  ERROR: No trained model found.")
    sys.exit(1)

with open(MODEL_PATH,    "rb") as f: model     = pickle.load(f)
with open(SCALER_PATH,   "rb") as f: scaler    = pickle.load(f)
with open(FEATURES_PATH, "rb") as f: feat_cols = pickle.load(f)
print(f"  Model loaded. Features: {feat_cols}")

print("\n[3/4] Building features from 2026 data...")
RAW_DIR = "data/raw"

DISCIPLINES_2026 = {
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

FIELD_EVENTS = {
    "men_PV", "women_PV", "men_LJ", "women_LJ",
    "men_TJ", "women_TJ", "men_HJ", "women_HJ",
    "men_SP", "women_SP", "men_DT", "women_DT",
    "men_JT", "women_JT"
}

def parse_mark(m):
    try:
        m = str(m).strip()
        if ":" in m:
            parts = m.split(":")
            if len(parts) == 2:
                return float(parts[0]) * 60 + float(parts[1])
        return float(m)
    except:
        return None

def seconds_to_time(seconds, discipline):
    is_field  = discipline in FIELD_EVENTS
    is_middle = discipline in ["men_800m", "women_800m", "men_1500m", "women_1500m",
                               "men_5000m", "women_5000m", "men_3000sc", "women_3000sc"]
    if is_field:
        return f"{seconds:.2f}m"
    elif is_middle:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}:{secs:05.2f}"
    else:
        return f"{seconds:.2f}"

def build_2026_features(key):
    path = os.path.join(RAW_DIR, f"{key}_2026.csv")
    if not os.path.exists(path):
        return pd.DataFrame()

    df = pd.read_csv(path)
    df = df.rename(columns={"Competitor": "athlete_name", "DOB": "dob", "Mark": "mark_str"})
    df["Mark"] = df["mark_str"].apply(parse_mark)
    df = df.dropna(subset=["Mark"])

    df["dob"] = pd.to_datetime(df["dob"], format="%d %b %Y", errors="coerce")
    today = pd.Timestamp(date.today())
    df["age"] = ((today - df["dob"]).dt.days / 365.25).round(1)

    is_track = key not in FIELD_EVENTS

    if is_track:
        sb = df.groupby("athlete_name")["Mark"].min()
    else:
        sb = df.groupby("athlete_name")["Mark"].max()

    meets = df.groupby("athlete_name")["Mark"].count()
    age   = df.groupby("athlete_name")["age"].first()

    feat = pd.DataFrame({"season_best": sb, "meets_count": meets, "age": age}).reset_index()
    feat["discipline"] = key
    feat["year"]       = 2026

    hist_path = os.path.join(RAW_DIR, f"{key}.csv")
    if os.path.exists(hist_path):
        hist = pd.read_csv(hist_path)
        hist["Mark_num"] = hist["Mark"].apply(parse_mark)
        hist = hist.dropna(subset=["Mark_num"])
        if is_track:
            cb = hist.groupby("Competitor")["Mark_num"].min().reset_index()
        else:
            cb = hist.groupby("Competitor")["Mark_num"].max().reset_index()
        cb.columns = ["athlete_name", "career_best"]
        feat = feat.merge(cb, on="athlete_name", how="left")
    else:
        feat["career_best"] = feat["season_best"]

    feat["career_best"]      = feat["career_best"].fillna(feat["season_best"])
    feat["pb_gap"]           = abs(feat["season_best"] - feat["career_best"])
    feat["consistency"]      = 0.05
    feat["yoy_improvement"]  = 0.0

    if is_track:
        feat["season_rank"]       = feat["season_best"].rank(ascending=True)
        feat["season_percentile"] = feat["season_best"].rank(ascending=False) / len(feat)
    else:
        feat["season_rank"]       = feat["season_best"].rank(ascending=False)
        feat["season_percentile"] = feat["season_best"].rank(ascending=True) / len(feat)

    return feat

features_2026 = {}
for key in DISCIPLINES_2026:
    features_2026[key] = build_2026_features(key)
    if not features_2026[key].empty:
        print(f"  {key}: {len(features_2026[key])} athletes")

print("\n[4/4] Generating predictions...\n")
print("=" * 60)
print("   2026 DIAMOND LEAGUE FINAL — PREDICTIONS")
print(f"   {date.today().strftime('%B %d, %Y')}")
print("=" * 60)

all_predictions = []

for key, label in DISCIPLINES_2026.items():
    df = features_2026.get(key, pd.DataFrame())
    if df.empty:
        continue

    qualified = standings.get(key, [])
    if not qualified:
        df_qual = df.copy().head(8)
    else:
        df_qual = df[df["athlete_name"].isin(qualified)].copy()

    if df_qual.empty:
        continue

    df_qual = df_qual.dropna(subset=feat_cols)
    if df_qual.empty:
        continue

    X = scaler.transform(df_qual[feat_cols])
    df_qual["win_probability"] = model.predict_proba(X)[:, 1]

    # Load h2h data and compute win rates
    h2h_path = os.path.join("data", "h2h", "h2h_rates.csv")
    if os.path.exists(h2h_path):
        h2h_df = pd.read_csv(h2h_path)
        disc_h2h = h2h_df[h2h_df["discipline"] == key]
        opponents = df_qual["athlete_name"].tolist()
        
        def get_h2h_rate(athlete):
            rates = []
            for opp in opponents:
                if opp == athlete:
                    continue
                row = disc_h2h[
                    (disc_h2h["athlete_a"] == athlete) & 
                    (disc_h2h["athlete_b"] == opp) &
                    (disc_h2h["meetings"] >= 2)
                ]
                if not row.empty:
                    rates.append(row.iloc[0]["win_rate"])
            return sum(rates) / len(rates) if rates else 0.5
        
        df_qual["h2h_win_rate"] = df_qual["athlete_name"].apply(get_h2h_rate)
        
        # Blend h2h with model probability (60% model, 40% h2h)
        df_qual["win_probability"] = (
            df_qual["win_probability"] * 0.6 + 
            df_qual["h2h_win_rate"] * 0.4
        )
        
    is_field = key in FIELD_EVENTS
    df_qual = df_qual.sort_values("season_best", ascending=not is_field)

    print(f"\n-- {label} --")

    nat_map = {}
    raw_path = os.path.join(RAW_DIR, f"{key}_2026.csv")
    if os.path.exists(raw_path):
        raw_df = pd.read_csv(raw_path)
        cols = list(raw_df.columns)
        dob_idx = cols.index("DOB") if "DOB" in cols else -1
        nat_col = cols[dob_idx + 1] if dob_idx >= 0 and dob_idx + 1 < len(cols) else None
        if nat_col is not None:
            nat_map = dict(zip(raw_df["Competitor"], raw_df[nat_col]))

    for i, (_, row) in enumerate(df_qual.head(8).iterrows()):
        medal = ["1", "2", "3"][i] if i < 3 else "  "
        sb    = seconds_to_time(row["season_best"], key)
        prob  = row["win_probability"]
        nat   = nat_map.get(row["athlete_name"], "--")
        if i < 3:
            print(f"  {medal} {row['athlete_name']}  {sb}  ({prob:.0%})")
        all_predictions.append({
            "discipline":      label,
            "predicted_rank":  i + 1,
            "athlete_name":    row["athlete_name"],
            "nationality":     nat,
            "season_best":     sb,
            "win_probability": f"{prob:.0%}",
            "date":            str(date.today()),
        })

out_path = os.path.join(OUTPUTS_DIR, "predictions_latest.csv")
pd.DataFrame(all_predictions).to_csv(out_path, index=False)

print(f"\n{'=' * 60}")
print(f"  Predictions saved -> {out_path}")
print(f"  Run this file again any time to refresh.")
print(f"{'=' * 60}")
