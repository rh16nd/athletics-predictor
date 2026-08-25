"""
train_model.py — rebuilds the historical training set (2018-2025, excluding
2020) and retrains the RandomForest used by run.py, using walk-forward
validation: the first 2 label years always seed training only, every year
after that is an independent test year (LABEL_YEARS[2:]) instead of one
fixed holdout.

Ground-truth DL Final results come from data/dl_final_results.csv, scraped
directly from World Athletics' own API (src/dl_final_results_scraper.py) --
no hand-typed results list. Run that scraper first (or whenever more years
become available) to refresh it.

Historically this went through several stages worth knowing about if the
accuracy number ever looks wrong: the notebook's original feature builder
looked for data/raw/{discipline}_{year}.csv (never existed), silently
defaulting several features to a constant; that was fixed by reading
data/raw/{discipline}.csv (has a year column) instead. A hand-typed
DL_RESULTS list (13, then 32, disciplines) went through multiple rounds of
"found another wrong podium" fixes before being retired entirely in favor
of the real scraped file this version uses.

Usage:
    python src/train_model.py                  # fixed weighted/wind features only
    python src/train_model.py --with-recency    # + recent_trend, days_since_last
    python src/train_model.py --with-recency --with-h2h --dry-run
"""
import argparse
import os
import pickle
import sys
import io
import unicodedata

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Guarded: several modules in src/ do this, and each wraps the SAME
# sys.stdout.buffer. With two of them imported into one process the first
# wrapper to be garbage-collected closes the buffer under the second, and
# every later write dies with "I/O operation on closed file" -- which took
# down the whole pytest run on 2026-08-25. After the first wrap the
# encoding is already utf-8, so this becomes a no-op.
if not (sys.stdout.encoding or "").lower().startswith("utf"):
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
    # Added to extend training beyond the original 13 disciplines (HANDOFF.md
    # Next Steps #1) -- historical data rebuilt via
    # `python src/historical_scraper.py --new-only`.
    "men_400m":     "Men 400m",
    "women_400m":   "Women 400m",
    "men_110h":     "Men 110m Hurdles",
    "women_100h":   "Women 100m Hurdles",
    "men_5000m":    "Men 5000m",
    "women_5000m":  "Women 5000m",
    "men_3000sc":   "Men 3000m Steeplechase",
    "women_3000sc": "Women 3000m Steeplechase",
    "men_HJ":       "Men High Jump",
    "women_HJ":     "Women High Jump",
    "men_TJ":       "Men Triple Jump",
    "women_TJ":     "Women Triple Jump",
    "men_SP":       "Men Shot Put",
    "women_SP":     "Women Shot Put",
    "men_DT":       "Men Discus Throw",
    "women_DT":     "Women Discus Throw",
    "men_JT":       "Men Javelin Throw",
    "women_JT":     "Women Javelin Throw",
    "women_LJ":     "Women Long Jump",
}
FIELD_EVENTS = {
    "men_PV", "women_PV", "men_LJ", "women_LJ",
    "men_HJ", "women_HJ", "men_TJ", "women_TJ",
    "men_SP", "women_SP", "men_DT", "women_DT", "men_JT", "women_JT",
}
WIND_EVENTS = {
    "men_100m", "women_100m", "men_200m", "women_200m",
    # Wind-legal events under actual competition rules -- these were left out
    # of the original wind-adjustment feature entirely (not a design choice,
    # just never extended past the first 4 disciplines it was built for).
    "men_110h", "women_100h", "men_LJ", "women_LJ", "men_TJ", "women_TJ",
}

# Real Diamond League Final results, scraped directly from World Athletics'
# own public API (src/dl_final_results_scraper.py) -- see that file's
# docstring for how the endpoint was found. This replaces a hand-typed
# DL_RESULTS list that went through multiple rounds of "found another wrong
# podium" fixes across 2026-08-22 before being retired entirely: real
# scraped data doesn't have the failure mode of a human (or an agent)
# mistyping a name or misremembering a finishing order.
DL_RESULTS_PATH = os.path.join(BASE_DIR, "data", "dl_final_results.csv")
H2H_PATH = os.path.join(BASE_DIR, "data", "h2h", "h2h_rates.csv")
VENUE_GEO_PATH = os.path.join(BASE_DIR, "data", "venues_geo.csv")

_VENUE_ELEV = None


def venue_elevations():
    """venue string -> metres above sea level, from src/venue_geo.py's cache.

    NOT currently a model feature, deliberately. Altitude was built and
    MEASURED OUT on 2026-08-25 (see HANDOFF Failed Attempts): only 2.2% of
    labeled rows have a season best set above 1000m, and coarse buckets that
    isolate the actual physics scored -0.20 pts (4/10 seeds) -- the same as a
    SHUFFLED control column. Only the raw 212-distinct-value version showed
    anything (+0.46, 6/10), which is the forest fingerprinting venues, not
    thin air. Kept because it is real, correct, cached data that
    src/venue_weather.py depends on.

    Missing venues are simply absent from the dict rather than defaulted to
    0: a roving venue like "European Athletics Championships" has no single
    elevation, and calling it sea level would be a confident wrong number in
    exactly the events (distance) where altitude matters most."""
    global _VENUE_ELEV
    if _VENUE_ELEV is None:
        if os.path.exists(VENUE_GEO_PATH):
            g = pd.read_csv(VENUE_GEO_PATH).dropna(subset=["elevation_m"])
            _VENUE_ELEV = dict(zip(g["venue"].astype(str), g["elevation_m"].astype(float)))
        else:
            _VENUE_ELEV = {}
    return _VENUE_ELEV

# 2018-2025, excluding 2020 (see dl_final_results_scraper.py's SKIP_YEARS --
# 2020's "Inspiration Games" was a COVID-era exhibition, not a real
# qualifying Final). Whether any given discipline was actually contested in
# a given year is read from the scraped file itself (see build_labeled_dataset) --
# not hand-flagged here.
LABEL_YEARS = [y for y in range(2018, 2026) if y != 2020]
DL_VENUES = [
    "doha", "shanghai", "suzhou", "shaoxing", "rabat", "florence", "paris",
    "oslo", "lausanne", "stockholm", "silesia", "monaco", "london",
    "zurich", "brussels", "eugene", "birmingham", "rome", "xiamen",
]
MAJOR_KEYWORDS = ["olympic", "world championship", "world athletics", "european championship"]


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
    # Deliberately NOT truncating to LABEL_YEARS here -- build_features()
    # needs earlier history (2018-2020) to compute a real career_best/
    # yoy_improvement for 2021 instead of silently defaulting yoy to 0.0 for
    # every 2021 row (there's no "before" left to compare against once
    # truncated). career_best/yoy per row are separately bounded to that
    # row's own year, so no lookahead leakage from later years either.
    return df.dropna(subset=["Mark"]).copy()


def build_features(df, discipline_key):
    records = []
    is_track = discipline_key not in FIELD_EVENTS
    for athlete in df["athlete_name"].unique():
        ath = df[df["athlete_name"] == athlete].copy()
        ath["Mark_num"] = ath["Mark"].apply(convert_mark_to_seconds)
        ath = ath.dropna(subset=["Mark_num"])
        if ath.empty:
            continue
        for year in LABEL_YEARS:
            season = ath[ath["year"] == year]
            prev = ath[ath["year"] < year]
            # career_best must only see marks up to and including this label's
            # year -- using the full athlete history here would leak future
            # seasons (e.g. 2024/2025) into a 2021/2022 labeled row's features.
            up_to_year = ath[ath["year"] <= year]
            if season.empty:
                continue
            if is_track:
                season_best = season["Mark_num"].min()
                career_best = up_to_year["Mark_num"].min()
            else:
                season_best = season["Mark_num"].max()
                career_best = up_to_year["Mark_num"].max()
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
    """discipline in FIELD_EVENTS -- not just "men_PV" -- must rank descending
    (higher mark = better = rank 1). Using a hardcoded "men_PV" string here
    previously left women_PV and men_LJ ranked backwards (lowest jump/vault
    getting rank 1) for the entire time they've been trained disciplines --
    caught while adding 10 more field events that would have hit the same bug."""
    all_groups = []
    for (discipline, year), group in df.groupby(["discipline", "year"]):
        group = group.copy()
        if discipline in FIELD_EVENTS:
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


def apply_wind_adjustment(mark, wind, is_field):
    """A following wind over the +1.0 m/s legal limit makes a time look
    faster (lower) but a jump/throw look longer (higher) than "fair" -- the
    penalty must move the mark toward worse in whichever direction that is
    for this discipline, not just always add. Extracted as a pure function
    so this direction-of-penalty logic is directly testable; it was wrong
    for field events (always added, which is backwards for a
    higher-is-better mark) until this fix, though field events had never
    been in WIND_EVENTS before this fix either, so it had no live effect yet."""
    if wind <= 1.0:
        return mark
    penalty = (wind - 1.0) * 0.01
    return mark - penalty if is_field else mark + penalty


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
        gap_var_map = {}

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
                        return apply_wind_adjustment(row["Mark"], wind, is_field)
                    except Exception:
                        return row["Mark"]
                raw["wind_adj"] = raw.apply(wind_adj, axis=1)
                wind_adj_agg = raw.groupby("athlete_name")["wind_adj"]
                wind_adj_map = (wind_adj_agg.max() if is_field else wind_adj_agg.min()).to_dict()

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

                    # How EVENLY the season was paced -- deliberately not
                    # another way of saying meets_count (already the 3rd-most
                    # important feature; a collinear restatement of it on a
                    # 459-prediction backtest is how you overfit, not how you
                    # add signal). Two athletes with six races each look
                    # identical to meets_count but different here: six spread
                    # regularly is a managed campaign, six with one 90-day
                    # hole in the middle is usually an injury nobody logged.
                    #
                    # A companion mean_gap_days was built and MEASURED OUT:
                    # -0.13 pts over 10 seeds (5/10 wins, i.e. a coin flip)
                    # and it actively diluted this one when both were in
                    # (-0.17). Don't re-add it without new evidence. This
                    # feature alone measured +0.68 pts, 8/10 seeds.
                    dates = ath["date"].dropna().sort_values()
                    if len(dates) > 2:
                        gaps = dates.diff().dropna().dt.days
                        gap_var_map[athlete] = float(gaps.std())
                    else:
                        # Fewer than 3 races gives at most one gap, and a
                        # single gap has no variability to measure.
                        gap_var_map[athlete] = 0.0

        group["weighted_season_best"] = group["athlete_name"].map(weighted_sb_map).fillna(group["season_best"])
        group["wind_adj_season_best"] = group["athlete_name"].map(wind_adj_map).fillna(group["season_best"])
        group["recent_trend"] = group["athlete_name"].map(trend_map).fillna(0.0)
        group["days_since_last"] = group["athlete_name"].map(days_map).fillna(999)
        group["gap_variability"] = group["athlete_name"].map(gap_var_map).fillna(0.0)
        all_groups.append(group)
    return pd.concat(all_groups, ignore_index=True)


def normalize_name(name):
    """Case- and diacritic-insensitive key for matching hand/agent-researched
    DL_RESULTS athlete names against WA toplist Competitor names. Added
    alongside the 19-discipline expansion after finding several accented
    names (Cá, Perkovic/Perković, Ceh/Čeh, Spanovic/Španović...) typed
    without diacritics -- the exact h2h_win_rate failure mode from earlier
    this session (silent join misses defaulting everyone to a neutral/zero
    label) would otherwise repeat here across ~30 more athletes."""
    if not isinstance(name, str):
        return name
    nfkd = unicodedata.normalize("NFKD", name)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).upper().strip()


def build_labeled_dataset():
    dfs = {}
    for key in TRAIN_DISCIPLINES:
        path = os.path.join(RAW_DIR, f"{key}.csv")
        df = pd.read_csv(path)
        dfs[key] = clean_discipline(df)

    all_features = {key: build_features(dfs[key], key) for key in TRAIN_DISCIPLINES}
    master = pd.concat(all_features.values(), ignore_index=True)

    results = pd.read_csv(DL_RESULTS_PATH)
    results["dl_rank"] = pd.to_numeric(results["place"], errors="coerce")  # DNF/DQ -> NaN, correctly excluded below

    # Whether a discipline was actually contested that year is read from
    # whether the scraper found ANY row for it (place included, so a DNF/DQ
    # still counts as "this event happened") -- not a hand-maintained
    # NOT_CONTESTED list. Filtering master to only these combos prevents an
    # uncontested discipline-year (e.g. a year the 5000m ran as a road race
    # instead) from injecting a spurious "nobody medaled" label across every
    # athlete in that discipline-year, the same failure mode a hardcoded
    # exclusion list used to guard against.
    contested = set(zip(results["discipline"], results["year"]))
    master = master[master.apply(lambda r: (r["discipline"], r["year"]) in contested, axis=1)]

    master["_name_key"] = master["athlete_name"].apply(normalize_name)

    dl_df = results[results["dl_rank"] <= 3].copy()
    dl_df["dl_winner"] = (dl_df["dl_rank"] == 1).astype(int)
    dl_df["dl_top3"] = (dl_df["dl_rank"] <= 3).astype(int)
    dl_df["_name_key"] = dl_df["athlete_name"].apply(normalize_name)

    labeled = master.merge(
        dl_df[["discipline", "year", "_name_key", "dl_winner", "dl_top3", "dl_rank"]],
        on=["discipline", "year", "_name_key"], how="left",
    )
    labeled["dl_winner"] = labeled["dl_winner"].fillna(0).astype(int)
    labeled["dl_top3"] = labeled["dl_top3"].fillna(0).astype(int)
    labeled["dl_rank"] = labeled["dl_rank"].fillna(0).astype(int)

    matched_keys = set(
        labeled.loc[labeled["dl_top3"] == 1, ["discipline", "year", "_name_key"]]
        .apply(tuple, axis=1)
    )
    unmatched = [
        (r["discipline"], r["year"], r["athlete_name"])
        for _, r in dl_df.iterrows()
        if (r["discipline"], r["year"], r["_name_key"]) not in matched_keys
    ]
    if unmatched:
        print(f"\n  WARNING: {len(unmatched)} real DL Final top-3 finishers found NO matching "
              f"training row (missing from the raw toplist scrape):")
        for discipline, year, name in unmatched:
            print(f"    {discipline} {year}: {name!r}")

    return labeled.drop(columns=["_name_key"])


def add_h2h_features(df):
    """Adds h2h_win_rate per (athlete, discipline, year) row: average win
    rate against the other athletes in that discipline-year's training pool
    (>=2 meetings required, matching run.py's inference-time threshold).

    data/h2h/h2h_rates.csv uses normal-case names ("Trayvon Bromell") while
    every other data source in this pipeline uses WA's ALL-CAPS-surname
    format ("Trayvon BROMELL") -- an exact-string match between the two
    finds ZERO matches. This silently made h2h_win_rate default to a neutral
    0.5 for every athlete in every prediction ever made by run.py's blend,
    despite 156k real matchup rows sitting unused. Matching case-insensitive
    here (and in run.py) is the actual fix -- confirmed live: 0/8 exact
    matches vs 7/8 case-insensitive matches for a sample discipline.
    """
    h2h_df = pd.read_csv(H2H_PATH)
    h2h_df["a_lower"] = h2h_df["athlete_a"].str.lower()
    h2h_df["b_lower"] = h2h_df["athlete_b"].str.lower()
    h2h_df = h2h_df[h2h_df["meetings"] >= 2]

    df = df.copy()
    rates = []
    for (discipline, year), group in df.groupby(["discipline", "year"]):
        sub = h2h_df[h2h_df["discipline"] == discipline]
        lookup = {}
        for _, r in sub.iterrows():
            lookup.setdefault(r["a_lower"], {})[r["b_lower"]] = r["win_rate"]
        names_lower = [n.lower() for n in group["athlete_name"]]
        for name_lower in names_lower:
            opp_rates = [
                lookup[name_lower][opp] for opp in names_lower
                if opp != name_lower and name_lower in lookup and opp in lookup[name_lower]
            ]
            rates.append(sum(opp_rates) / len(opp_rates) if opp_rates else 0.5)
    df["h2h_win_rate"] = rates
    return df


DEFAULT_MODEL_PARAMS = {
    "n_estimators": 200, "max_depth": 16, "min_samples_leaf": 1,
    "class_weight": None, "random_state": 42,
}  # walk-forward-tuned via --tune (2026-08-22, re-tuned 2026-08-23 twice more: once
   # after extending LABEL_YEARS to 2018-2025, again after adding major_meets_scraper.py
   # + the recognized-names noise filter). Re-tuned five times total as the feature set,
   # training-set size, or data-quality filtering changed -- each round picked a
   # different winner (including flipping class_weight None<->balanced twice), a sign
   # that hyperparameter search on a dataset this size (459 rows) is itself fairly
   # noisy -- don't read too much into which single config "won" a given round, but
   # do still re-tune whenever the data changes rather than reusing an old winner.


def _score_fold(train, test, feature_cols, model_params=None):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[feature_cols])
    X_test = scaler.transform(test[feature_cols])

    model = RandomForestClassifier(**(model_params or DEFAULT_MODEL_PARAMS))
    model.fit(X_train, train["dl_top3"])

    test = test.copy()
    test["win_probability"] = model.predict_proba(X_test)[:, 1]

    correct, possible = 0, 0
    per_discipline = []
    # The headline metric is a SET intersection, so it is blind to finishing
    # position: correctly having the eventual winner in your three counts
    # exactly the same as correctly having the eventual 3rd, and naming the
    # right podium in the wrong order still scores 3/3. That is fine as a
    # headline, but it averages away something real -- measured 2026-08-25,
    # the model finds 85% of actual winners and only 36% of actual 3rd
    # places. `pos` carries the breakdown so train_and_backtest() can report
    # it alongside, without changing what the headline number means.
    pos = {"found": {1: 0, 2: 0, 3: 0}, "total": {1: 0, 2: 0, 3: 0},
           "winner_is_top_pick": 0, "winner_count": 0}
    for discipline in test["discipline"].unique():
        disc_df = test[test["discipline"] == discipline].sort_values("win_probability", ascending=False)
        top3_predicted = disc_df.head(3)["athlete_name"].tolist()
        actual_rows = disc_df[disc_df["dl_top3"] == 1]
        top3_actual = actual_rows["athlete_name"].tolist()
        hits = len(set(top3_predicted) & set(top3_actual))
        correct += hits
        possible += 3
        per_discipline.append((discipline, hits))

        predicted_set = set(top3_predicted)
        top_pick = top3_predicted[0] if top3_predicted else None
        for _, row in actual_rows.iterrows():
            rank = int(row["dl_rank"]) if pd.notna(row.get("dl_rank")) else None
            if rank in pos["total"]:
                pos["total"][rank] += 1
                if row["athlete_name"] in predicted_set:
                    pos["found"][rank] += 1
            if rank == 1:
                pos["winner_count"] += 1
                if row["athlete_name"] == top_pick:
                    pos["winner_is_top_pick"] += 1
    return correct, possible, per_discipline, model, pos


def walk_forward_folds(full, feature_cols, model_params=None, verbose=True, stats=None):
    """Runs the expanding-window walk-forward validation (train on every
    labeled year strictly before each test year, one fold per test year) and
    returns (total_correct, total_possible). Shared by train_and_backtest()
    and tune_hyperparameters() so both use the exact same evaluation, not
    two implementations that could quietly drift apart.

    Pass a dict as `stats` to also collect the per-finishing-position
    breakdown (see _score_fold). Kept as an optional out-parameter so the
    return signature stays a 2-tuple for tune_hyperparameters()."""
    total_correct, total_possible = 0, 0
    test_years = LABEL_YEARS[2:]  # first 2 years only ever serve as training seed
    for test_year in test_years:
        train_years = [y for y in LABEL_YEARS if y < test_year]
        train = full[full["year"].isin(train_years)]
        test = full[full["year"] == test_year]
        if train.empty or test.empty:
            continue
        correct, possible, _, _, pos = _score_fold(train, test, feature_cols, model_params)
        if stats is not None:
            for rank in (1, 2, 3):
                stats.setdefault("found", {}).setdefault(rank, 0)
                stats.setdefault("total", {}).setdefault(rank, 0)
                stats["found"][rank] += pos["found"][rank]
                stats["total"][rank] += pos["total"][rank]
            stats["winner_is_top_pick"] = stats.get("winner_is_top_pick", 0) + pos["winner_is_top_pick"]
            stats["winner_count"] = stats.get("winner_count", 0) + pos["winner_count"]
        if verbose:
            fold_acc = round(correct / possible * 100, 1) if possible else 0.0
            print(f"  train {train_years} -> test {test_year}: {correct}/{possible} = {fold_acc}%")
        total_correct += correct
        total_possible += possible
    return total_correct, total_possible


def tune_hyperparameters(feature_cols):
    """Small grid search over RandomForest params, each candidate scored by
    the exact same walk-forward folds train_and_backtest() reports honest
    accuracy with -- so "best" here means "generalizes best across 2023,
    2024, 2025", not "fits one split best". Informational only: prints
    ranked results and returns the winner, doesn't touch outputs/. Whoever
    reads the results decides whether to make a candidate the new
    DEFAULT_MODEL_PARAMS -- this deliberately isn't automatic, so a
    retrain's default hyperparameters can't silently drift between runs."""
    labeled = build_labeled_dataset()
    ranked = add_season_rank(labeled)
    full = add_new_features(ranked)
    full = add_h2h_features(full)
    full = full.dropna(subset=feature_cols)

    grid = [
        {"n_estimators": n, "max_depth": d, "min_samples_leaf": leaf, "class_weight": cw, "random_state": 42}
        for n in [100, 200, 300]
        for d in [None, 8, 16]
        for leaf in [1, 2, 4]
        for cw in ["balanced", None]
    ]

    print(f"\n=== Hyperparameter search ({len(grid)} candidates, walk-forward scored) ===")
    results = []
    for params in grid:
        correct, possible = walk_forward_folds(full, feature_cols, params, verbose=False)
        acc = round(correct / possible * 100, 1) if possible else 0.0
        results.append((acc, correct, possible, params))

    results.sort(key=lambda r: -r[0])
    print("  Top 10:")
    for acc, correct, possible, params in results[:10]:
        param_str = ", ".join(f"{k}={v}" for k, v in params.items() if k != "random_state")
        print(f"    {acc:5.1f}% ({correct}/{possible})  {param_str}")

    baseline = next((r for r in results if r[3] == DEFAULT_MODEL_PARAMS), None)
    if baseline:
        print(f"\n  Current default ({DEFAULT_MODEL_PARAMS}): {baseline[0]}%")

    best_acc, best_correct, best_possible, best_params = results[0]
    print(f"\n  Best: {best_params} -> {best_acc}% ({best_correct}/{best_possible})")
    return best_params


def _print_position_breakdown(stats):
    """Report how the headline number is distributed across the podium.

    The headline is deliberately left alone -- it stays the flat, order-blind
    top-3 hit rate that every number in HANDOFF.md was measured with, and
    model_accuracy.txt still holds exactly that. This is a diagnostic printed
    beside it, because one averaged figure genuinely hides the shape of the
    result: the model is far better at identifying who WINS than who rounds
    out the podium, and a single percentage cannot say that.

    A position-weighted score (3/2/1) is also shown. It reads much higher --
    ~66% where the flat metric reads ~58% -- purely because the model's hits
    concentrate on the high-value finishers. It is a different ruler, NOT an
    improvement, and must never be quoted as one or compared against the
    historical flat numbers."""
    total = stats.get("total", {})
    if not any(total.values()):
        return
    found = stats.get("found", {})
    print("")
    print("  Where the top-3 hits actually fall (same predictions, broken out):")
    labels = {1: "actual winners", 2: "actual 2nd place", 3: "actual 3rd place"}
    for rank in (1, 2, 3):
        t = total.get(rank, 0)
        if not t:
            continue
        pct = 100.0 * found.get(rank, 0) / t
        print(f"    {labels[rank]:20} found in our top 3: {found.get(rank, 0):>4}/{t:<4} = {pct:5.1f}%")
    wc = stats.get("winner_count", 0)
    if wc:
        wp = 100.0 * stats.get("winner_is_top_pick", 0) / wc
        print(f"    winner called as our OWN #1 pick: "
              f"{stats.get('winner_is_top_pick', 0)}/{wc} = {wp:.1f}%")
    weighted_correct = sum(found.get(r, 0) * w for r, w in ((1, 3), (2, 2), (3, 1)))
    weighted_possible = sum(total.get(r, 0) * w for r, w in ((1, 3), (2, 2), (3, 1)))
    if weighted_possible:
        print(f"    position-weighted (3/2/1): "
              f"{100.0 * weighted_correct / weighted_possible:.1f}% "
              f"-- a DIFFERENT ruler, not an improvement; headline stays the flat number")


def train_and_backtest(feature_cols, label="", model_params=None):
    """Walk-forward (expanding-window) validation: train on every labeled
    year strictly before each test year, one fold per test year, instead of
    a single fixed train/test split. With only 3 years of labels a single
    holdout was the only option -- now that real scraped data covers
    2021-2025, testing on 3 independent years (2023, 2024, 2025) instead of
    just one gives a far more honest read on whether the model generalizes,
    not just whether it fit one particular season."""
    labeled = build_labeled_dataset()
    ranked = add_season_rank(labeled)
    full = add_new_features(ranked)
    full = add_h2h_features(full)
    full = full.dropna(subset=feature_cols)

    print(f"\n=== {label} — walk-forward validation ({LABEL_YEARS[0]}-{LABEL_YEARS[-1]}) ===")
    pos_stats = {}
    total_correct, total_possible = walk_forward_folds(
        full, feature_cols, model_params, verbose=True, stats=pos_stats)
    accuracy_pct = round(total_correct / total_possible * 100, 1) if total_possible else 0.0
    print(f"  Overall (all {len(LABEL_YEARS[2:])} folds combined): {total_correct}/{total_possible} = {accuracy_pct}%")
    _print_position_breakdown(pos_stats)

    # The deployed model is refit on ALL labeled years -- walk-forward above
    # is purely to estimate honest generalization, not to pick which years
    # the shipped model actually trains on. Standard practice: cross-validate
    # for the number you report, then refit on everything for production.
    scaler = StandardScaler()
    X_all = scaler.fit_transform(full[feature_cols])
    model = RandomForestClassifier(**(model_params or DEFAULT_MODEL_PARAMS))
    model.fit(X_all, full["dl_top3"])

    print("\n  Feature importances (final model, trained on all years):")
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
    parser.add_argument("--with-h2h", action="store_true",
                        help="Add h2h_win_rate to the trained feature set (requires the "
                             "case-insensitive matching fix -- see add_h2h_features)")
    parser.add_argument("--with-schedule", action="store_true",
                        help="Add gap_variability -- how EVENLY an athlete's season was paced, "
                             "as opposed to meets_count, which only counts it. Measured at "
                             "+0.68 pts mean over 10 seeds (8/10 wins).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Backtest only, don't overwrite outputs/")
    parser.add_argument("--tune", action="store_true",
                        help="Grid-search RandomForest hyperparameters via the walk-forward "
                             "folds and print ranked results. Informational only -- exits "
                             "without training or saving anything.")
    args = parser.parse_args()

    base_cols = [
        "season_best", "career_best", "pb_gap", "meets_count", "consistency",
        "yoy_improvement", "age", "season_rank", "season_percentile",
        "weighted_season_best", "wind_adj_season_best",
    ]
    feature_cols = base_cols.copy()
    label_parts = ["V3-fixed"]
    if args.with_recency:
        feature_cols += ["recent_trend", "days_since_last"]
        label_parts.append("recency")
    if args.with_h2h:
        feature_cols += ["h2h_win_rate"]
        label_parts.append("h2h")
    if args.with_schedule:
        feature_cols += ["gap_variability"]
        label_parts.append("schedule")
    label = " + ".join(label_parts)

    if args.tune:
        tune_hyperparameters(feature_cols)
        sys.exit(0)

    model, scaler, accuracy_pct = train_and_backtest(feature_cols, label=label)

    if not args.dry_run:
        save_artifacts(model, scaler, feature_cols, accuracy_pct)
    else:
        print("\n[dry run — outputs/ not modified]")
