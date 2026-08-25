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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
# build_2026_features() and its helpers were extracted to src/feature_builder.py
# (2026-08-24) so they can be unit-tested without importing this module --
# run.py has no `if __name__ == "__main__":` guard, so importing it directly
# kicks off a real ~1hr live scrape via the code below. This import is a pure
# move, not a behavior change: same functions, same RAW_DIR default.
from feature_builder import (
    FIELD_EVENTS, LONG_DISTANCE_EVENTS, RAW_DIR,
    get_qual_limit, parse_mark, seconds_to_time, build_2026_features,
)

print("=" * 60)
print("   ATHLETICS PREDICTOR — 2026 DL PREDICTIONS")
print(f"   Running: {date.today().strftime('%B %d, %Y')}")
print("=" * 60)

# --no-scrape re-runs only the modelling half against whatever is already in
# data/raw. The full pipeline is a ~1hr live scrape, which makes any change to
# the prediction logic effectively unverifiable while developing it. Live data
# is untouched either way -- this only skips re-fetching it.
SKIP_SCRAPE = "--no-scrape" in sys.argv

if SKIP_SCRAPE:
    print("\n[1/5] --no-scrape: reusing the existing data/raw snapshot.")
else:
    print("\n[1/5] Fetching live data from World Athletics + DL standings...")
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

if SKIP_SCRAPE:
    print("\n[2/5] --no-scrape: reusing the existing injury flags.")
else:
    print("\n[2/5] Checking for injuries / withdrawals...")
    injury_result = subprocess.run(
        [sys.executable, "-u", "src/injury_checker.py"],
        capture_output=True, text=True, encoding='utf-8'
    )
    for line in injury_result.stdout.split('\n'):
        if any(k in line for k in ["Scraping", "flagged", "WARNING", "REMOVE", "WATCH", "No injury", "meet results"]):
            print(f"  {line.strip()}")
    if injury_result.returncode != 0:
        print("  WARNING: injury checker failed, continuing without injury data.")
        print(injury_result.stderr[-500:])

injury_flags_path = os.path.join("data", "injury_flags.json")
injury_flags = {}
if os.path.exists(injury_flags_path):
    with open(injury_flags_path, encoding="utf-8") as f:
        injury_flags = json.load(f).get("athletes", {})

print("\n[3/5] Loading trained model...")
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

print("\n[4/5] Building features from 2026 data...")

# How many athletes beyond the confirmed Diamond League field to score and
# export per discipline. Small on purpose: these are "who would be a threat
# if they got in", not a second leaderboard.
NEAR_MISS_COUNT = 4

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

features_2026 = {}
for key in DISCIPLINES_2026:
    features_2026[key] = build_2026_features(key)
    if not features_2026[key].empty:
        print(f"  {key}: {len(features_2026[key])} athletes")

print("\n[5/5] Generating predictions...\n")
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
        # Real DL standings weren't available for this discipline (scrape
        # gap, or too early in the season for a standings page to exist
        # yet) -- falls back to ranking by worldwide season-best time/mark
        # so the discipline still gets predictions instead of silently
        # disappearing, but these athletes are NOT verified Diamond League
        # participants, just fast times run anywhere. Flagged per-row below
        # (dl_qualified) so this is never silently presented as equivalent
        # to a real standings-based list.
        print(f"  [{label}] WARNING: no official DL standings found -- falling back to worldwide season-best ranking (not restricted to real Diamond League participants)")
        is_field_fallback = key in FIELD_EVENTS
        df_qual = df.sort_values("season_best", ascending=not is_field_fallback).head(get_qual_limit(key)).copy()
        df_qual["dl_qualified"] = False
    else:
        df_qual = df[df["athlete_name"].isin(qualified)].copy()
        df_qual["dl_qualified"] = True

        # Near-miss athletes: the next fastest who are NOT in the official
        # standings. They are scored by the same model and exported with
        # dl_qualified = False so the frontend can show "who would be a
        # threat if they got in" without ever presenting them as finalists.
        #
        # dl_qualified used to be per-DISCIPLINE (just "did we have standings
        # for this event"), which meant every athlete in every discipline
        # carried the same value and the Q badge conveyed nothing. It is now
        # genuinely per-athlete.
        is_field_disc = key in FIELD_EVENTS
        near = (df[~df["athlete_name"].isin(qualified)]
                .sort_values("season_best", ascending=not is_field_disc)
                .head(NEAR_MISS_COUNT)
                .copy())
        if not near.empty:
            near["dl_qualified"] = False
            df_qual = pd.concat([df_qual, near], ignore_index=True)

    if df_qual.empty:
        continue

    # Injury / withdrawal filtering
    if injury_flags:
        injury_key = df_qual["athlete_name"].apply(lambda n: " ".join(str(n).split()).title())
        df_qual["injury_status"] = injury_key.map(lambda k: injury_flags.get(k, {}).get("status"))
        removed = df_qual[df_qual["injury_status"] == "remove"]
        if not removed.empty:
            print(f"  [{label}] Removed (injury/withdrawal): {', '.join(removed['athlete_name'])}")
        df_qual = df_qual[df_qual["injury_status"] != "remove"].copy()
    else:
        df_qual["injury_status"] = None

    if df_qual.empty:
        continue

    # h2h_win_rate as a model input feature (see src/train_model.py's
    # add_h2h_features for why this must be case-insensitive: h2h_rates.csv
    # uses normal-case names ("Trayvon Bromell") while every other source
    # here uses WA's ALL-CAPS-surname format ("Trayvon BROMELL"). An exact
    # match finds zero rows, which silently made this feature a constant
    # neutral 0.5 in the old manual 60/40 blend for every prediction ever
    # made. Now trained-in instead of blended after the fact.
    h2h_path = os.path.join("data", "h2h", "h2h_rates.csv")
    if os.path.exists(h2h_path):
        h2h_df = pd.read_csv(h2h_path)
        disc_h2h = h2h_df[(h2h_df["discipline"] == key) & (h2h_df["meetings"] >= 2)].copy()
        disc_h2h["a_lower"] = disc_h2h["athlete_a"].str.lower()
        disc_h2h["b_lower"] = disc_h2h["athlete_b"].str.lower()
        lookup = {}
        for _, r in disc_h2h.iterrows():
            lookup.setdefault(r["a_lower"], {})[r["b_lower"]] = r["win_rate"]

        names_lower = df_qual["athlete_name"].str.lower().tolist()

        def get_h2h_rate(name_lower):
            opp_rates = [
                lookup[name_lower][opp] for opp in names_lower
                if opp != name_lower and name_lower in lookup and opp in lookup[name_lower]
            ]
            return sum(opp_rates) / len(opp_rates) if opp_rates else 0.5

        df_qual["h2h_win_rate"] = [get_h2h_rate(n) for n in names_lower]
    else:
        df_qual["h2h_win_rate"] = 0.5

    df_qual = df_qual.dropna(subset=feat_cols)
    if df_qual.empty:
        continue

    X = scaler.transform(df_qual[feat_cols])
    df_qual["win_probability"] = model.predict_proba(X)[:, 1]

    # Display bound only -- no artificial rescaling. predict_proba is
    # already a real, differentiated probability estimate (balanced-class
    # RandomForest); the old "(prob/total)*3, clip to 0.95" scheme assumed
    # probabilities should sum to ~3 (one per medal), but once h2h_win_rate
    # became a real feature, combined raw scores routinely exceed 1.0 before
    # the clip -- e.g. two different favorites both landing on the exact
    # 0.95 ceiling and becoming indistinguishable, sometimes several per
    # discipline. Clipping only to avoid a literal 0%/100% display.
    df_qual["win_probability"] = df_qual["win_probability"].clip(0.01, 0.99)

    is_field = key in FIELD_EVENTS
    df_qual = df_qual.sort_values("season_best", ascending=not is_field)

    print(f"\n-- {label} --")

    nat_map = {}
    profile_map = {}
    raw_path = os.path.join(RAW_DIR, f"{key}_2026.csv")
    if os.path.exists(raw_path):
        raw_df = pd.read_csv(raw_path)
        cols = list(raw_df.columns)
        dob_idx = cols.index("DOB") if "DOB" in cols else -1
        nat_col = cols[dob_idx + 1] if dob_idx >= 0 and dob_idx + 1 < len(cols) else None
        if nat_col is not None:
            nat_map = dict(zip(raw_df["Competitor"], raw_df[nat_col]))
        if "ProfileURL" in raw_df.columns:
            profile_map = dict(zip(raw_df["Competitor"], raw_df["ProfileURL"]))

    # Qualified athletes keep the real ranking and the podium summary.
    # Near-miss athletes are appended after them, scored by the same model
    # but deliberately given predicted_rank = None: they are not finalists,
    # and numbering them alongside the real field is exactly the kind of
    # implied claim this project keeps having to walk back.
    qual_rows = df_qual[df_qual["dl_qualified"]].head(get_qual_limit(key))
    near_rows = df_qual[~df_qual["dl_qualified"]]
    ordered = list(qual_rows.iterrows()) + list(near_rows.iterrows())

    for i, (_, row) in enumerate(ordered):
        row_qualified = bool(row["dl_qualified"])
        medal = ["1", "2", "3"][i] if (row_qualified and i < 3) else "  "
        sb    = seconds_to_time(row["season_best"], key)
        prob  = row["win_probability"]
        nat   = nat_map.get(row["athlete_name"], "--")
        is_watch = row.get("injury_status") == "watch"
        watch_marker = " [INJURY WATCH]" if is_watch else ""
        if row_qualified and i < 3:
            print(f"  {medal} {row['athlete_name']}{watch_marker}  {sb}  ({prob:.0%})")

        # Real WA profile link scraped from the toplist page; only falls
        # back to a search query if we somehow never captured it (older
        # cached data, or an athlete row with no linked cell).
        profile_url = profile_map.get(row["athlete_name"])
        if not isinstance(profile_url, str) or not profile_url:
            profile_url = f"https://www.worldathletics.org/search/?q={row['athlete_name'].replace(' ', '+')}"

        all_predictions.append({
            "discipline":      label,
            "predicted_rank":  (i + 1) if row_qualified else None,
            "athlete_name":    row["athlete_name"],
            "nationality":     nat,
            "season_best":     sb,
            "win_probability": f"{prob:.0%}",
            "injury_watch":    is_watch,
            "profile_url":     profile_url,
            "date":            str(date.today()),
            # Real, already-computed model features that were being thrown
            # away at export time -- api.py's athlete profile page surfaces
            # these directly instead of only using them as invisible model
            # inputs.
            "career_best":      seconds_to_time(row["career_best"], key) if pd.notna(row.get("career_best")) else None,
            "pb_gap":           round(float(row["pb_gap"]), 3) if pd.notna(row.get("pb_gap")) else None,
            "age":              round(float(row["age"]), 1) if pd.notna(row.get("age")) else None,
            "meets_count":      int(row["meets_count"]) if pd.notna(row.get("meets_count")) else None,
            "days_since_last":  int(row["days_since_last"]) if pd.notna(row.get("days_since_last")) and row["days_since_last"] < 999 else None,
            # Real per-discipline flag: True when this athlete's inclusion
            # came from actually being in WA's own scraped 2026 Diamond
            # League standings for this discipline (scrape_dl_standings),
            # False when standings data wasn't available and this row is a
            # worldwide season-best-ranking fallback instead (see the
            # WARNING print above) -- so "real Diamond League participant"
            # is a fact this can actually assert, not an assumption.
            "dl_qualified":     row_qualified,
        })

out_path = os.path.join(OUTPUTS_DIR, "predictions_latest.csv")
pd.DataFrame(all_predictions).to_csv(out_path, index=False)

print(f"\n{'=' * 60}")
print(f"  Predictions saved -> {out_path}")
print(f"  Run this file again any time to refresh.")
print(f"{'=' * 60}")
