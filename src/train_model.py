"""
train_model.py — rebuilds the historical training set (2021-2023) and retrains
the RandomForest used by run.py, backtesting on 2023 the same way the original
notebook (notebooks/01_eda.ipynb, cell 28) did.

Fixes a bug found while adding recency features: the notebook's feature builder
looked for data/raw/{discipline}_{year}.csv (never existed for training years),
so weighted_season_best/wind_adj_season_best silently fell back to a copy of
season_best, and recent_trend/days_since_last always fell back to 0.0/999 for
every single training row — i.e. 4 of the model's intended features carried no
real signal. This reads from the actual historical file (data/raw/{discipline}.csv,
which has a year column) instead.

Usage:
    python src/train_model.py                  # fixed weighted/wind features only
    python src/train_model.py --with-recency    # + recent_trend, days_since_last
"""
import argparse
import os
import pickle
import sys
import io

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

TRAIN_DISCIPLINES = {
    "men_100m":    "Men 100m",
    "women_100m":  "Women 100m",
    "men_200m":    "Men 200m",
    "men_400h":    "Men 400m Hurdles",
    "women_400h":  "Women 400m Hurdles",
    "men_PV":      "Men Pole Vault",
    "women_200m":  "Women 200m",
    "men_800m":    "Men 800m",
    "women_800m":  "Women 800m",
    "men_1500m":   "Men 1500m",
    "women_1500m": "Women 1500m",
    "women_PV":    "Women Pole Vault",
    "men_LJ":      "Men Long Jump",
}
FIELD_EVENTS = {"men_PV", "women_PV", "men_LJ"}
WIND_EVENTS = {"men_100m", "women_100m", "men_200m", "women_200m"}
DL_VENUES = [
    "doha", "shanghai", "suzhou", "shaoxing", "rabat", "florence", "paris",
    "oslo", "lausanne", "stockholm", "silesia", "monaco", "london",
    "zurich", "brussels", "eugene", "birmingham", "rome", "xiamen",
]
MAJOR_KEYWORDS = ["olympic", "world championship", "world athletics", "european championship"]

# DL Final winners and top-3 finishers 2021-2023 (unchanged from the notebook)
DL_RESULTS = [
    {"discipline": "men_100m", "year": 2021, "athlete_name": "Lamont Marcell JACOBS", "dl_rank": 1},
    {"discipline": "men_100m", "year": 2021, "athlete_name": "Zharnel HUGHES", "dl_rank": 2},
    {"discipline": "men_100m", "year": 2021, "athlete_name": "Fred KERLEY", "dl_rank": 3},
    {"discipline": "men_100m", "year": 2022, "athlete_name": "Fred KERLEY", "dl_rank": 1},
    {"discipline": "men_100m", "year": 2022, "athlete_name": "Trayvon BROMELL", "dl_rank": 2},
    {"discipline": "men_100m", "year": 2022, "athlete_name": "Oblique SEVILLE", "dl_rank": 3},
    {"discipline": "men_100m", "year": 2023, "athlete_name": "Noah LYLES", "dl_rank": 1},
    {"discipline": "men_100m", "year": 2023, "athlete_name": "Oblique SEVILLE", "dl_rank": 2},
    {"discipline": "men_100m", "year": 2023, "athlete_name": "Zharnel HUGHES", "dl_rank": 3},
    {"discipline": "women_100m", "year": 2021, "athlete_name": "Elaine THOMPSON-HERAH", "dl_rank": 1},
    {"discipline": "women_100m", "year": 2021, "athlete_name": "Shericka JACKSON", "dl_rank": 2},
    {"discipline": "women_100m", "year": 2021, "athlete_name": "Marie-Josee TA LOU", "dl_rank": 3},
    {"discipline": "women_100m", "year": 2022, "athlete_name": "Shericka JACKSON", "dl_rank": 1},
    {"discipline": "women_100m", "year": 2022, "athlete_name": "Elaine THOMPSON-HERAH", "dl_rank": 2},
    {"discipline": "women_100m", "year": 2022, "athlete_name": "Dina ASHER-SMITH", "dl_rank": 3},
    {"discipline": "women_100m", "year": 2023, "athlete_name": "Sha'Carri RICHARDSON", "dl_rank": 1},
    {"discipline": "women_100m", "year": 2023, "athlete_name": "Shericka JACKSON", "dl_rank": 2},
    {"discipline": "women_100m", "year": 2023, "athlete_name": "Elaine THOMPSON-HERAH", "dl_rank": 3},
    {"discipline": "men_200m", "year": 2021, "athlete_name": "Kenneth BEDNAREK", "dl_rank": 1},
    {"discipline": "men_200m", "year": 2021, "athlete_name": "Noah LYLES", "dl_rank": 2},
    {"discipline": "men_200m", "year": 2021, "athlete_name": "Fred KERLEY", "dl_rank": 3},
    {"discipline": "men_200m", "year": 2022, "athlete_name": "Noah LYLES", "dl_rank": 1},
    {"discipline": "men_200m", "year": 2022, "athlete_name": "Kenneth BEDNAREK", "dl_rank": 2},
    {"discipline": "men_200m", "year": 2022, "athlete_name": "Erriyon KNIGHTON", "dl_rank": 3},
    {"discipline": "men_200m", "year": 2023, "athlete_name": "Noah LYLES", "dl_rank": 1},
    {"discipline": "men_200m", "year": 2023, "athlete_name": "Kenneth BEDNAREK", "dl_rank": 2},
    {"discipline": "men_200m", "year": 2023, "athlete_name": "Erriyon KNIGHTON", "dl_rank": 3},
    {"discipline": "men_400h", "year": 2021, "athlete_name": "Karsten WARHOLM", "dl_rank": 1},
    {"discipline": "men_400h", "year": 2021, "athlete_name": "Alison DOS SANTOS", "dl_rank": 2},
    {"discipline": "men_400h", "year": 2021, "athlete_name": "Rai BENJAMIN", "dl_rank": 3},
    {"discipline": "men_400h", "year": 2022, "athlete_name": "Karsten WARHOLM", "dl_rank": 1},
    {"discipline": "men_400h", "year": 2022, "athlete_name": "Alison DOS SANTOS", "dl_rank": 2},
    {"discipline": "men_400h", "year": 2022, "athlete_name": "Rai BENJAMIN", "dl_rank": 3},
    {"discipline": "men_400h", "year": 2023, "athlete_name": "Karsten WARHOLM", "dl_rank": 1},
    {"discipline": "men_400h", "year": 2023, "athlete_name": "Alison DOS SANTOS", "dl_rank": 2},
    {"discipline": "men_400h", "year": 2023, "athlete_name": "Rai BENJAMIN", "dl_rank": 3},
    {"discipline": "women_400h", "year": 2021, "athlete_name": "Sydney MCLAUGHLIN", "dl_rank": 1},
    {"discipline": "women_400h", "year": 2021, "athlete_name": "Femke BOL", "dl_rank": 2},
    {"discipline": "women_400h", "year": 2021, "athlete_name": "Dalilah MUHAMMAD", "dl_rank": 3},
    {"discipline": "women_400h", "year": 2022, "athlete_name": "Sydney MCLAUGHLIN", "dl_rank": 1},
    {"discipline": "women_400h", "year": 2022, "athlete_name": "Femke BOL", "dl_rank": 2},
    {"discipline": "women_400h", "year": 2022, "athlete_name": "Anna COCKRELL", "dl_rank": 3},
    {"discipline": "women_400h", "year": 2023, "athlete_name": "Femke BOL", "dl_rank": 1},
    {"discipline": "women_400h", "year": 2023, "athlete_name": "Sydney MCLAUGHLIN", "dl_rank": 2},
    {"discipline": "women_400h", "year": 2023, "athlete_name": "Anna COCKRELL", "dl_rank": 3},
    {"discipline": "men_PV", "year": 2021, "athlete_name": "Armand DUPLANTIS", "dl_rank": 1},
    {"discipline": "men_PV", "year": 2021, "athlete_name": "Christopher NILSEN", "dl_rank": 2},
    {"discipline": "men_PV", "year": 2021, "athlete_name": "Ernest John OBIENA", "dl_rank": 3},
    {"discipline": "men_PV", "year": 2022, "athlete_name": "Armand DUPLANTIS", "dl_rank": 1},
    {"discipline": "men_PV", "year": 2022, "athlete_name": "Christopher NILSEN", "dl_rank": 2},
    {"discipline": "men_PV", "year": 2022, "athlete_name": "Ernest John OBIENA", "dl_rank": 3},
    {"discipline": "men_PV", "year": 2023, "athlete_name": "Armand DUPLANTIS", "dl_rank": 1},
    {"discipline": "men_PV", "year": 2023, "athlete_name": "Christopher NILSEN", "dl_rank": 2},
    {"discipline": "men_PV", "year": 2023, "athlete_name": "Ernest John OBIENA", "dl_rank": 3},
    {"discipline": "women_200m", "year": 2021, "athlete_name": "Gabrielle THOMAS", "dl_rank": 1},
    {"discipline": "women_200m", "year": 2021, "athlete_name": "Christine MBOMA", "dl_rank": 2},
    {"discipline": "women_200m", "year": 2021, "athlete_name": "Blessing OKAGBARE", "dl_rank": 3},
    {"discipline": "women_200m", "year": 2022, "athlete_name": "Shericka JACKSON", "dl_rank": 1},
    {"discipline": "women_200m", "year": 2022, "athlete_name": "Dafne SCHIPPERS", "dl_rank": 2},
    {"discipline": "women_200m", "year": 2022, "athlete_name": "Gabrielle THOMAS", "dl_rank": 3},
    {"discipline": "women_200m", "year": 2023, "athlete_name": "Sha'Carri RICHARDSON", "dl_rank": 1},
    {"discipline": "women_200m", "year": 2023, "athlete_name": "Gabrielle THOMAS", "dl_rank": 2},
    {"discipline": "women_200m", "year": 2023, "athlete_name": "Shericka JACKSON", "dl_rank": 3},
    {"discipline": "men_800m", "year": 2021, "athlete_name": "Emmanuel Kipkurui KORIR", "dl_rank": 1},
    {"discipline": "men_800m", "year": 2021, "athlete_name": "Peter BOL", "dl_rank": 2},
    {"discipline": "men_800m", "year": 2021, "athlete_name": "Nijel AMOS", "dl_rank": 3},
    {"discipline": "men_800m", "year": 2022, "athlete_name": "Marco AROP", "dl_rank": 1},
    {"discipline": "men_800m", "year": 2022, "athlete_name": "Emmanuel Kipkurui KORIR", "dl_rank": 2},
    {"discipline": "men_800m", "year": 2022, "athlete_name": "Djamel SEDJATI", "dl_rank": 3},
    {"discipline": "men_800m", "year": 2023, "athlete_name": "Marco AROP", "dl_rank": 1},
    {"discipline": "men_800m", "year": 2023, "athlete_name": "Djamel SEDJATI", "dl_rank": 2},
    {"discipline": "men_800m", "year": 2023, "athlete_name": "Emmanuel Kipkurui KORIR", "dl_rank": 3},
    {"discipline": "women_800m", "year": 2021, "athlete_name": "Athing MU", "dl_rank": 1},
    {"discipline": "women_800m", "year": 2021, "athlete_name": "Raevyn ROGERS", "dl_rank": 2},
    {"discipline": "women_800m", "year": 2021, "athlete_name": "Habitam ALEMU", "dl_rank": 3},
    {"discipline": "women_800m", "year": 2022, "athlete_name": "Athing MU", "dl_rank": 1},
    {"discipline": "women_800m", "year": 2022, "athlete_name": "Mary MORAA", "dl_rank": 2},
    {"discipline": "women_800m", "year": 2022, "athlete_name": "Keely HODGKINSON", "dl_rank": 3},
    {"discipline": "women_800m", "year": 2023, "athlete_name": "Mary MORAA", "dl_rank": 1},
    {"discipline": "women_800m", "year": 2023, "athlete_name": "Keely HODGKINSON", "dl_rank": 2},
    {"discipline": "women_800m", "year": 2023, "athlete_name": "Athing MU", "dl_rank": 3},
    {"discipline": "men_1500m", "year": 2021, "athlete_name": "Timothy CHERUIYOT", "dl_rank": 1},
    {"discipline": "men_1500m", "year": 2021, "athlete_name": "Jakob INGEBRIGTSEN", "dl_rank": 2},
    {"discipline": "men_1500m", "year": 2021, "athlete_name": "Josh KERR", "dl_rank": 3},
    {"discipline": "men_1500m", "year": 2022, "athlete_name": "Jakob INGEBRIGTSEN", "dl_rank": 1},
    {"discipline": "men_1500m", "year": 2022, "athlete_name": "Timothy CHERUIYOT", "dl_rank": 2},
    {"discipline": "men_1500m", "year": 2022, "athlete_name": "Josh KERR", "dl_rank": 3},
    {"discipline": "men_1500m", "year": 2023, "athlete_name": "Jakob INGEBRIGTSEN", "dl_rank": 1},
    {"discipline": "men_1500m", "year": 2023, "athlete_name": "Josh KERR", "dl_rank": 2},
    {"discipline": "men_1500m", "year": 2023, "athlete_name": "Yomif KEJELCHA", "dl_rank": 3},
    {"discipline": "women_1500m", "year": 2021, "athlete_name": "Faith Chepngetich KIPYEGON", "dl_rank": 1},
    {"discipline": "women_1500m", "year": 2021, "athlete_name": "Laura MUIR", "dl_rank": 2},
    {"discipline": "women_1500m", "year": 2021, "athlete_name": "Gudaf TSEGAY", "dl_rank": 3},
    {"discipline": "women_1500m", "year": 2022, "athlete_name": "Faith Chepngetich KIPYEGON", "dl_rank": 1},
    {"discipline": "women_1500m", "year": 2022, "athlete_name": "Laura MUIR", "dl_rank": 2},
    {"discipline": "women_1500m", "year": 2022, "athlete_name": "Gudaf TSEGAY", "dl_rank": 3},
    {"discipline": "women_1500m", "year": 2023, "athlete_name": "Faith Chepngetich KIPYEGON", "dl_rank": 1},
    {"discipline": "women_1500m", "year": 2023, "athlete_name": "Laura MUIR", "dl_rank": 2},
    {"discipline": "women_1500m", "year": 2023, "athlete_name": "Diribe WELTEJI", "dl_rank": 3},
    {"discipline": "women_PV", "year": 2021, "athlete_name": "Katie NAGEOTTE", "dl_rank": 1},
    {"discipline": "women_PV", "year": 2021, "athlete_name": "Anzhelika SIDOROVA", "dl_rank": 2},
    {"discipline": "women_PV", "year": 2021, "athlete_name": "Katerina STEFANIDI", "dl_rank": 3},
    {"discipline": "women_PV", "year": 2022, "athlete_name": "Nina KENNEDY", "dl_rank": 1},
    {"discipline": "women_PV", "year": 2022, "athlete_name": "Katie NAGEOTTE", "dl_rank": 2},
    {"discipline": "women_PV", "year": 2022, "athlete_name": "Angelica BENGTSSON", "dl_rank": 3},
    {"discipline": "women_PV", "year": 2023, "athlete_name": "Nina KENNEDY", "dl_rank": 1},
    {"discipline": "women_PV", "year": 2023, "athlete_name": "Katie NAGEOTTE", "dl_rank": 2},
    {"discipline": "women_PV", "year": 2023, "athlete_name": "Alysha NEWMAN", "dl_rank": 3},
    {"discipline": "men_LJ", "year": 2021, "athlete_name": "Miltiadis TENTOGLOU", "dl_rank": 1},
    {"discipline": "men_LJ", "year": 2021, "athlete_name": "Juan Miguel ECHEVARRIA", "dl_rank": 2},
    {"discipline": "men_LJ", "year": 2021, "athlete_name": "Marquise GOODWIN", "dl_rank": 3},
    {"discipline": "men_LJ", "year": 2022, "athlete_name": "Miltiadis TENTOGLOU", "dl_rank": 1},
    {"discipline": "men_LJ", "year": 2022, "athlete_name": "Juan Miguel ECHEVARRIA", "dl_rank": 2},
    {"discipline": "men_LJ", "year": 2022, "athlete_name": "Tajay GAYLE", "dl_rank": 3},
    {"discipline": "men_LJ", "year": 2023, "athlete_name": "Miltiadis TENTOGLOU", "dl_rank": 1},
    {"discipline": "men_LJ", "year": 2023, "athlete_name": "Mattia FURLANI", "dl_rank": 2},
    {"discipline": "men_LJ", "year": 2023, "athlete_name": "Carey McLeod", "dl_rank": 3},
]
NAME_FIXES = {
    "Marcell JACOBS": "Lamont Marcell JACOBS",
    "Kenny BEDNAREK": "Kenneth BEDNAREK",
    "Mondo DUPLANTIS": "Armand DUPLANTIS",
}


def convert_mark_to_seconds(mark_str):
    try:
        mark_str = str(mark_str).strip()
        if ":" in mark_str:
            parts = mark_str.split(":")
            if len(parts) == 2:
                return float(parts[0]) * 60 + float(parts[1])
        return float(mark_str)
    except Exception:
        return None


def clean_discipline(df):
    df = df.drop(columns=["Unnamed: 0", "discipline"], errors="ignore")
    rename_map = {
        "Competitor": "athlete_name", "DOB": "dob", "Nat": "country",
        "Results Score": "results_score", "Pos": "pos", "Venue": "venue", "Date": "date",
    }
    if "WIND" in df.columns:
        rename_map["WIND"] = "wind"
    df = df.rename(columns=rename_map)
    if "wind" not in df.columns:
        df["wind"] = np.nan
    df["date"] = pd.to_datetime(df["date"], format="%d %b %Y", errors="coerce")
    df["dob"] = pd.to_datetime(df["dob"], format="%d %b %Y", errors="coerce")
    df["age"] = ((df["date"] - df["dob"]).dt.days / 365.25).round(1)
    df_recent = df[df["year"].between(2021, 2023)].copy()
    df_recent = df_recent.dropna(subset=["Mark"])
    return df_recent


def build_features(df, discipline_key):
    records = []
    is_track = discipline_key not in FIELD_EVENTS
    for athlete in df["athlete_name"].unique():
        ath = df[df["athlete_name"] == athlete].copy()
        ath["Mark_num"] = ath["Mark"].apply(convert_mark_to_seconds)
        ath = ath.dropna(subset=["Mark_num"])
        if ath.empty:
            continue
        for year in [2021, 2022, 2023]:
            season = ath[ath["year"] == year]
            prev = ath[ath["year"] < year]
            if season.empty:
                continue
            if is_track:
                season_best = season["Mark_num"].min()
                career_best = ath["Mark_num"].min()
            else:
                season_best = season["Mark_num"].max()
                career_best = ath["Mark_num"].max()
            pb_gap = abs(season_best - career_best)
            meets_count = len(season)
            consistency = season["Mark_num"].std() if len(season) > 1 else 0.0
            if not prev.empty:
                prev_best = prev["Mark_num"].min() if is_track else prev["Mark_num"].max()
                yoy = (prev_best - season_best) if is_track else (season_best - prev_best)
            else:
                yoy = 0.0
            age = ath["age"].dropna().median()
            country = ath["country"].iloc[0]
            records.append({
                "athlete_name": athlete, "country": country, "discipline": discipline_key,
                "year": year, "season_best": round(season_best, 4), "career_best": round(career_best, 4),
                "pb_gap": round(pb_gap, 4), "meets_count": meets_count,
                "consistency": round(consistency, 4), "yoy_improvement": round(yoy, 4),
                "age": round(age, 1) if not np.isnan(age) else np.nan,
            })
    return pd.DataFrame(records)


def add_season_rank(df):
    all_groups = []
    for (discipline, year), group in df.groupby(["discipline", "year"]):
        group = group.copy()
        if discipline == "men_PV":
            group["season_rank"] = group["season_best"].rank(ascending=False)
            group["season_percentile"] = group["season_best"].rank(ascending=True) / len(group)
        else:
            group["season_rank"] = group["season_best"].rank(ascending=True)
            group["season_percentile"] = group["season_best"].rank(ascending=False) / len(group)
        all_groups.append(group)
    return pd.concat(all_groups, ignore_index=True)


def competition_weight(venue):
    if not isinstance(venue, str):
        return 1.0
    v = venue.lower()
    if any(k in v for k in MAJOR_KEYWORDS):
        return 1.3
    if any(dl in v for dl in DL_VENUES):
        return 1.2
    return 1.0


def add_new_features(df):
    """Same idea as the notebook's add_new_features, but reads the real
    historical file (data/raw/{discipline}.csv, filtered by year) instead of
    a per-year file that never existed — that bug is why weighted_season_best/
    wind_adj_season_best were silent duplicates of season_best, and why
    recent_trend/days_since_last were always 0.0/999 for every training row."""
    all_groups = []
    for (discipline, year), group in df.groupby(["discipline", "year"]):
        group = group.copy()
        is_field = discipline in FIELD_EVENTS
        weighted_sb_map, wind_adj_map, trend_map, days_map = {}, {}, {}, {}

        raw_path = os.path.join(RAW_DIR, f"{discipline}.csv")
        if os.path.exists(raw_path):
            raw_full = pd.read_csv(raw_path)
            raw = raw_full[raw_full["year"] == year].copy()
            raw = raw.rename(columns={"Competitor": "athlete_name", "Mark": "mark_str"})
            raw["Mark"] = raw["mark_str"].apply(convert_mark_to_seconds)
            raw = raw.dropna(subset=["Mark"])

            if "Venue" in raw.columns:
                raw["comp_weight"] = raw["Venue"].apply(competition_weight)
                raw["weighted_mark"] = raw["Mark"] * raw["comp_weight"]
                wsb = (raw.groupby("athlete_name")["weighted_mark"].max() if is_field
                       else raw.groupby("athlete_name")["weighted_mark"].min())
                weighted_sb_map = wsb.to_dict()

            if discipline in WIND_EVENTS and "WIND" in raw.columns:
                def wind_adj(row):
                    try:
                        wind = float(str(row["WIND"]).replace("+", "").strip())
                        if wind > 1.0:
                            return row["Mark"] + (wind - 1.0) * 0.01
                        return row["Mark"]
                    except Exception:
                        return row["Mark"]
                raw["wind_adj"] = raw.apply(wind_adj, axis=1)
                wind_adj_map = raw.groupby("athlete_name")["wind_adj"].min().to_dict()

            if "Date" in raw.columns:
                raw["date"] = pd.to_datetime(raw["Date"], format="%d %b %Y", errors="coerce")
                ref_date = pd.Timestamp(f"{year}-09-01")
                for athlete in group["athlete_name"]:
                    ath = raw[raw["athlete_name"] == athlete].sort_values("date", ascending=False)
                    if ath.empty or ath["date"].isna().all():
                        trend_map[athlete] = 0.0
                        days_map[athlete] = 999
                        continue
                    last = ath["date"].dropna().iloc[0]
                    days_map[athlete] = (ref_date - last).days
                    recent = ath.head(3)["Mark"].tolist()
                    if len(recent) >= 2:
                        trend_map[athlete] = recent[0] - recent[-1] if not is_field else recent[-1] - recent[0]
                    else:
                        trend_map[athlete] = 0.0

        group["weighted_season_best"] = group["athlete_name"].map(weighted_sb_map).fillna(group["season_best"])
        group["wind_adj_season_best"] = group["athlete_name"].map(wind_adj_map).fillna(group["season_best"])
        group["recent_trend"] = group["athlete_name"].map(trend_map).fillna(0.0)
        group["days_since_last"] = group["athlete_name"].map(days_map).fillna(999)
        all_groups.append(group)
    return pd.concat(all_groups, ignore_index=True)


def build_labeled_dataset():
    dfs = {}
    for key in TRAIN_DISCIPLINES:
        path = os.path.join(RAW_DIR, f"{key}.csv")
        df = pd.read_csv(path)
        dfs[key] = clean_discipline(df)

    all_features = {key: build_features(dfs[key], key) for key in TRAIN_DISCIPLINES}
    master = pd.concat(all_features.values(), ignore_index=True)

    dl_df = pd.DataFrame(DL_RESULTS)
    dl_df["athlete_name"] = dl_df["athlete_name"].replace(NAME_FIXES)
    dl_df["dl_winner"] = (dl_df["dl_rank"] == 1).astype(int)
    dl_df["dl_top3"] = (dl_df["dl_rank"] <= 3).astype(int)

    labeled = master.merge(
        dl_df[["discipline", "year", "athlete_name", "dl_winner", "dl_top3", "dl_rank"]],
        on=["discipline", "year", "athlete_name"], how="left",
    )
    labeled["dl_winner"] = labeled["dl_winner"].fillna(0).astype(int)
    labeled["dl_top3"] = labeled["dl_top3"].fillna(0).astype(int)
    labeled["dl_rank"] = labeled["dl_rank"].fillna(0).astype(int)
    return labeled


def train_and_backtest(feature_cols, label=""):
    labeled = build_labeled_dataset()
    ranked = add_season_rank(labeled)
    full = add_new_features(ranked)

    train = full[full["year"].isin([2021, 2022])].dropna(subset=feature_cols)
    test = full[full["year"] == 2023].dropna(subset=feature_cols)

    X_train, y_train = train[feature_cols], train["dl_top3"]
    X_test, y_test = test[feature_cols], test["dl_top3"]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
    model.fit(X_train_scaled, y_train)

    test = test.copy()
    test["win_probability"] = model.predict_proba(X_test_scaled)[:, 1]

    print(f"\n=== {label} — 2023 Backtest ===")
    total_correct = 0
    n_disciplines = test["discipline"].nunique()
    for discipline in test["discipline"].unique():
        disc_df = test[test["discipline"] == discipline].sort_values("win_probability", ascending=False)
        top3_predicted = disc_df.head(3)["athlete_name"].tolist()
        top3_actual = disc_df[disc_df["dl_top3"] == 1]["athlete_name"].tolist()
        hits = len(set(top3_predicted) & set(top3_actual))
        total_correct += hits
        print(f"  {discipline}: {hits}/3")

    accuracy_pct = round(total_correct / (n_disciplines * 3) * 100, 1)
    print(f"  Total: {total_correct}/{n_disciplines * 3} = {accuracy_pct}%")

    print("\n  Feature importances:")
    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    for feat, imp in importances.items():
        print(f"    {feat:24s} {imp:.4f}")

    return model, scaler, accuracy_pct


def save_artifacts(model, scaler, feature_cols, accuracy_pct):
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    with open(os.path.join(OUTPUTS_DIR, "model_rf.pkl"), "wb") as f:
        pickle.dump(model, f)
    with open(os.path.join(OUTPUTS_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(OUTPUTS_DIR, "feature_cols.pkl"), "wb") as f:
        pickle.dump(feature_cols, f)
    with open(os.path.join(OUTPUTS_DIR, "model_accuracy.txt"), "w") as f:
        f.write(str(accuracy_pct))
    print(f"\nSaved model_rf.pkl, scaler.pkl, feature_cols.pkl, model_accuracy.txt ({accuracy_pct}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-recency", action="store_true",
                        help="Add recent_trend/days_since_last to the trained feature set")
    parser.add_argument("--dry-run", action="store_true",
                        help="Backtest only, don't overwrite outputs/")
    args = parser.parse_args()

    base_cols = [
        "season_best", "career_best", "pb_gap", "meets_count", "consistency",
        "yoy_improvement", "age", "season_rank", "season_percentile",
        "weighted_season_best", "wind_adj_season_best",
    ]
    feature_cols = base_cols + (["recent_trend", "days_since_last"] if args.with_recency else [])
    label = "V4 (recency features)" if args.with_recency else "V3-fixed (real weighted/wind features)"

    model, scaler, accuracy_pct = train_and_backtest(feature_cols, label=label)

    if not args.dry_run:
        save_artifacts(model, scaler, feature_cols, accuracy_pct)
    else:
        print("\n[dry run — outputs/ not modified]")
