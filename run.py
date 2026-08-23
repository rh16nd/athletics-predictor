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

LONG_DISTANCE_EVENTS = {
    "men_1500m", "women_1500m", "men_5000m", "women_5000m",
    "men_3000sc", "women_3000sc",
}

def get_qual_limit(discipline_key):
    if discipline_key in FIELD_EVENTS:
        return 6
    if discipline_key in LONG_DISTANCE_EVENTS:
        return 10
    return 8

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

    # meets_count from df (the toplist snapshot) would always be 1 for
    # every single athlete -- that file structurally has exactly one row
    # per athlete (their season-best mark), not a results log. Prefer the
    # real per-meeting current-season file (current_season_scraper.py's
    # output, one real row per real race) when it exists, same fix as
    # days_since_last/recent_trend below.
    #
    # A recognized athlete with ZERO rows in that file has a real, verified
    # answer -- zero confirmed DL-circuit meetings so far this season, not
    # the toplist's structural "1" every athlete gets regardless of their
    # real meeting count (confirmed live, 2026-08-24: Jessica Hull showed
    # "1 meet this season" while days_since_last/the Season Form chart both
    # honestly showed no 2026 data at all for her -- a real contradiction
    # on the same stat panel). So once the meetings file exists, it fully
    # replaces the toplist count (fill_value=0), not just fills gaps in it.
    # The toplist's always-1 count is used only as a last resort when the
    # meetings file doesn't exist yet at all for this discipline.
    meets = df.groupby("athlete_name")["Mark"].count()
    meetings_path = os.path.join(RAW_DIR, f"{key}_2026_meetings.csv")
    if os.path.exists(meetings_path):
        meetings_df = pd.read_csv(meetings_path)
        real_meets = meetings_df.groupby("Competitor")["Mark"].count()
        real_meets.index.name = "athlete_name"
        meets = real_meets.reindex(meets.index, fill_value=0)

    age   = df.groupby("athlete_name")["age"].first()
    # Real per-athlete variability across this season's marks (was a flat 0.05 for everyone).
    consistency = df.groupby("athlete_name")["Mark"].std().fillna(0.0)

    # Competition level weighting
    DL_VENUES = [
        "doha", "shanghai", "suzhou", "shaoxing", "rabat", "florence", "paris",
        "oslo", "lausanne", "stockholm", "silesia", "monaco", "london",
        "zurich", "brussels", "eugene", "birmingham", "rome", "xiamen"
    ]
    MAJOR_KEYWORDS = ["olympic", "world championship", "world athletics", "european championship"]

    def competition_weight(venue):
        if not isinstance(venue, str):
            return 1.0
        venue_lower = venue.lower()
        if any(k in venue_lower for k in MAJOR_KEYWORDS):
            return 1.3
        if any(dl in venue_lower for dl in DL_VENUES):
            return 1.2
        return 1.0

    weighted_sb = None
    if "Venue" in df.columns:
        df["comp_weight"] = df["Venue"].apply(competition_weight)
        df["weighted_mark"] = df["Mark"] * df["comp_weight"]
        if is_track:
            weighted_sb = df.groupby("athlete_name")["weighted_mark"].min().rename("weighted_season_best")
        else:
            weighted_sb = df.groupby("athlete_name")["weighted_mark"].max().rename("weighted_season_best")

    # Wind adjustment for sprint/hurdle events
    WIND_EVENTS = {"men_100m", "women_100m", "men_200m", "women_200m", "men_110h", "women_100h"}
    wind_sb = None
    if key in WIND_EVENTS and "WIND" in df.columns:
        def wind_adjusted_mark(row):
            try:
                wind = float(str(row["WIND"]).replace("+", "").strip())
                mark = row["Mark"]
                if pd.isna(mark) or pd.isna(wind):
                    return mark
                if wind > 1.0:
                    penalty = (wind - 1.0) * 0.01
                    return mark + penalty
                return mark
            except:
                return row["Mark"]
        df["wind_adj_mark"] = df.apply(wind_adjusted_mark, axis=1)
        wind_sb = df.groupby("athlete_name")["wind_adj_mark"].min().rename("wind_adj_season_best")

    # Build features DataFrame
    feat_dict = {"season_best": sb, "meets_count": meets, "age": age, "consistency": consistency}
    if weighted_sb is not None:
        feat_dict["weighted_season_best"] = weighted_sb
    if wind_sb is not None:
        feat_dict["wind_adj_season_best"] = wind_sb

    feat = pd.DataFrame(feat_dict).reset_index()
    if "weighted_season_best" not in feat.columns:
        feat["weighted_season_best"] = feat["season_best"]
    if "wind_adj_season_best" not in feat.columns:
        feat["wind_adj_season_best"] = feat["season_best"]

    feat["discipline"] = key
    feat["year"]       = 2026

    hist_path = os.path.join(RAW_DIR, f"{key}.csv")
    prev_year = 2025  # season immediately before this file's target year (2026)
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

        # Real year-over-year change (was a flat 0.0 for everyone). Requires the
        # historical top-list file to actually contain prev_year rows — as of
        # 2026-08 data/raw/*.csv caps out at 2023, so this still evaluates to 0.0
        # until that scrape is refreshed with 2024/2025 seasons.
        prev_hist = hist[hist["year"] == prev_year] if "year" in hist.columns else pd.DataFrame()
        if not prev_hist.empty:
            if is_track:
                prev_best = prev_hist.groupby("Competitor")["Mark_num"].min().reset_index()
            else:
                prev_best = prev_hist.groupby("Competitor")["Mark_num"].max().reset_index()
            prev_best.columns = ["athlete_name", "prev_season_best"]
            feat = feat.merge(prev_best, on="athlete_name", how="left")
        else:
            feat["prev_season_best"] = np.nan

        # Real consistency: the live 2026 top-list has exactly one mark per
        # athlete (it's a best-mark snapshot, not a results log), so there's no
        # in-season spread to measure yet. Use the spread across the athlete's
        # most recent historical season instead — same "within one season,
        # multiple meets" idea the model was trained on, just using the latest
        # season we have data for rather than the not-yet-available current one.
        if "year" in hist.columns:
            last_year_per_athlete = hist.groupby("Competitor")["year"].transform("max")
            last_season = hist[hist["year"] == last_year_per_athlete]
            hist_consistency = last_season.groupby("Competitor")["Mark_num"].std().fillna(0.0)
            hist_consistency = hist_consistency.reset_index()
            hist_consistency.columns = ["athlete_name", "hist_consistency"]
            feat = feat.merge(hist_consistency, on="athlete_name", how="left")
            feat["consistency"] = feat["hist_consistency"].fillna(feat["consistency"])
            feat = feat.drop(columns=["hist_consistency"])
    else:
        feat["career_best"] = feat["season_best"]
        feat["prev_season_best"] = np.nan

    feat["career_best"]      = feat["career_best"].fillna(feat["season_best"])
    feat["pb_gap"]           = abs(feat["season_best"] - feat["career_best"])

    if is_track:
        feat["yoy_improvement"] = feat["prev_season_best"] - feat["season_best"]
    else:
        feat["yoy_improvement"] = feat["season_best"] - feat["prev_season_best"]
    feat["yoy_improvement"] = feat["yoy_improvement"].fillna(0.0)
    feat = feat.drop(columns=["prev_season_best"])

    if is_track:
        feat["season_rank"]       = feat["season_best"].rank(ascending=True)
        feat["season_percentile"] = feat["season_best"].rank(ascending=False) / len(feat)
    else:
        feat["season_rank"]       = feat["season_best"].rank(ascending=False)
        feat["season_percentile"] = feat["season_best"].rank(ascending=True) / len(feat)

    # Recent form features -- prefer the real current-season per-meeting file
    # (current_season_scraper.py) when it exists over the live toplist
    # snapshot (df) this function otherwise reads: the toplist has exactly
    # ONE row per athlete (their season's single best mark, per its own
    # comment above), so computing "days since last"/"recent trend" from it
    # actually measured "days since their BEST mark", not their most recent
    # race -- silently wrong whenever an athlete's best mark wasn't also
    # their latest one (e.g. they peaked mid-season, then raced again more
    # recently without beating it -- confirmed live, 2026-08-24: Rai
    # Benjamin's toplist row was dated 18 Jul, but he'd actually raced again
    # on 23 Aug). The meetings file has one real row per real race, so these
    # features reflect an athlete's actual last competition instead.
    try:
        meetings_path = os.path.join(RAW_DIR, f"{key}_2026_meetings.csv")
        if os.path.exists(meetings_path):
            raw_df = pd.read_csv(meetings_path).rename(columns={"Competitor": "athlete_name"})
            raw_df["Mark"] = raw_df["Mark"].apply(parse_mark)
            raw_df = raw_df.dropna(subset=["Mark"])
        else:
            raw_df = df.copy()
        if "Date" in raw_df.columns:
            raw_df["date"] = pd.to_datetime(raw_df["Date"], dayfirst=True, errors="coerce")
            recent_trends = []
            days_since = []
            for athlete in feat["athlete_name"]:
                ath_df = raw_df[raw_df["athlete_name"] == athlete].sort_values("date", ascending=False)
                if ath_df.empty or ath_df["date"].isna().all():
                    recent_trends.append(0.0)
                    days_since.append(999)
                    continue
                last_date = ath_df["date"].dropna().iloc[0]
                days_since.append((today - last_date).days)
                recent = ath_df.head(3)["Mark"].tolist()
                if len(recent) >= 2:
                    trend = recent[-1] - recent[0] if not is_track else recent[0] - recent[-1]
                    recent_trends.append(trend)
                else:
                    recent_trends.append(0.0)
            feat["recent_trend"]    = recent_trends
            feat["days_since_last"] = days_since
        else:
            feat["recent_trend"]    = 0.0
            feat["days_since_last"] = 999
    except:
        feat["recent_trend"]    = 0.0
        feat["days_since_last"] = 999

    return feat

    
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
    dl_qualified_disc = bool(qualified)
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
    else:
        df_qual = df[df["athlete_name"].isin(qualified)].copy()

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

    for i, (_, row) in enumerate(df_qual.head(get_qual_limit(key)).iterrows()):
        medal = ["1", "2", "3"][i] if i < 3 else "  "
        sb    = seconds_to_time(row["season_best"], key)
        prob  = row["win_probability"]
        nat   = nat_map.get(row["athlete_name"], "--")
        is_watch = row.get("injury_status") == "watch"
        watch_marker = " [INJURY WATCH]" if is_watch else ""
        if i < 3:
            print(f"  {medal} {row['athlete_name']}{watch_marker}  {sb}  ({prob:.0%})")

        # Real WA profile link scraped from the toplist page; only falls
        # back to a search query if we somehow never captured it (older
        # cached data, or an athlete row with no linked cell).
        profile_url = profile_map.get(row["athlete_name"])
        if not isinstance(profile_url, str) or not profile_url:
            profile_url = f"https://www.worldathletics.org/search/?q={row['athlete_name'].replace(' ', '+')}"

        all_predictions.append({
            "discipline":      label,
            "predicted_rank":  i + 1,
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
            "dl_qualified":     dl_qualified_disc,
        })

out_path = os.path.join(OUTPUTS_DIR, "predictions_latest.csv")
pd.DataFrame(all_predictions).to_csv(out_path, index=False)

print(f"\n{'=' * 60}")
print(f"  Predictions saved -> {out_path}")
print(f"  Run this file again any time to refresh.")
print(f"{'=' * 60}")
