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
import json
import pickle
import sys
import io
import unicodedata

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# The three form features are defined ONCE, in the module run.py serves from,
# so training and inference cannot drift apart on what a column means.
import feature_builder

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
# Every final we have a podium for, not just the Diamond League ones
# (src/final_labels.py). Used by --pooled.
FINALS_LABELS_PATH = os.path.join(BASE_DIR, "data", "labels", "finals.csv")
# Who actually contested each of those finals, for the "real field" metric.
FINALS_FIELDS_PATH = os.path.join(BASE_DIR, "data", "labels", "final_fields.csv")
H2H_PATH = os.path.join(BASE_DIR, "data", "h2h", "h2h_rates.csv")
# The raw races behind h2h_rates.csv, plus when each meeting was held
# (src/h2h_scraper.py --dates-only). Together these let a rivalry be read as it
# stood on the day of a given final instead of as it stands today.
MEET_RESULTS_PATH = os.path.join(BASE_DIR, "data", "h2h", "meet_results.csv")
MEET_DATES_PATH = os.path.join(BASE_DIR, "data", "h2h", "meet_dates.csv")
VENUE_GEO_PATH = os.path.join(BASE_DIR, "data", "venues_geo.csv")
VENUE_WEATHER_PATH = os.path.join(BASE_DIR, "data", "venue_weather.csv")

_VENUE_WEATHER = None


def venue_weather():
    """(venue, 'YYYY-MM-DD') -> (temp_mean_c, humidity_pct) from
    src/venue_weather.py's cache. Real ERA5 reanalysis, ~67% of rows.

    NOT currently a model feature, deliberately. Weather was built and
    MEASURED OUT on 2026-08-25 (see HANDOFF Failed Attempts): temp_at_sb,
    mean_race_temp and mean_humidity all scored NEGATIVE across 10 seeds,
    and each real column landed within noise of a SHUFFLED permutation of
    itself. Kept because the data is real, validated and cached, and because
    one framing is still untried -- see that HANDOFF entry."""
    global _VENUE_WEATHER
    if _VENUE_WEATHER is None:
        if os.path.exists(VENUE_WEATHER_PATH):
            w = pd.read_csv(VENUE_WEATHER_PATH).dropna(subset=["temp_mean_c"])
            _VENUE_WEATHER = {
                (str(r.venue), str(r.date)): (float(r.temp_mean_c), float(r.humidity_pct))
                for r in w.itertuples()
            }
        else:
            _VENUE_WEATHER = {}
    return _VENUE_WEATHER

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

# The first year the walk-forward backtest SCORES. Every earlier label year
# is training seed only. Pinned rather than derived so that adding history
# behind the labels cannot move the metric's population -- see
# walk_forward_folds. 2021 is what LABEL_YEARS[2:] meant when the labels
# started at 2018, so every accuracy figure in HANDOFF stays comparable.
FIRST_TEST_YEAR = 2021
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


def build_features(df, discipline_key, finals=None):
    """One feature row per athlete per FINAL.

    `finals` is (year, competition, cutoff date) triples -- every final this
    discipline has a label for. Passing None keeps the original behaviour of
    one row per athlete-year, cut at nothing, which is what the
    Diamond-League-only model was trained on and is preserved so that path can
    still be reproduced exactly.

    THE CUT-OFF IS THE POINT. With one September final a year you can get away
    with letting a row see the athlete's whole season. Pool in a World
    Championships final held in July and that same shortcut is reading the
    answer out of the future: the season best it scores the athlete on may have
    been set in August, weeks after the race being predicted. Every aggregate
    below is therefore taken over marks set strictly BEFORE the final's date."""
    records = []
    is_track = discipline_key not in FIELD_EVENTS
    groups = (
        [(y, None, None) for y in LABEL_YEARS] if finals is None
        else [(int(y), c, pd.Timestamp(d)) for y, c, d in finals]
    )
    for athlete in df["athlete_name"].unique():
        ath = df[df["athlete_name"] == athlete].copy()
        ath["Mark_num"] = ath["Mark"].apply(convert_mark_to_seconds)
        ath = ath.dropna(subset=["Mark_num"])
        if ath.empty:
            continue
        for year, competition, cutoff in groups:
            before = ath if cutoff is None else ath[ath["date"] < cutoff]
            season = before[before["year"] == year]
            prev = before[before["year"] < year]
            # career_best must only see marks up to and including this label's
            # year -- using the full athlete history here would leak future
            # seasons (e.g. 2024/2025) into a 2021/2022 labeled row's features.
            up_to_year = before[before["year"] <= year]
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
            record = {
                "athlete_name": athlete, "country": country, "discipline": discipline_key,
                "year": year, "season_best": round(season_best, 4), "career_best": round(career_best, 4),
                "pb_gap": round(pb_gap, 4), "meets_count": meets_count,
                "consistency": round(consistency, 4), "yoy_improvement": round(yoy, 4),
                "age": round(age, 1) if not np.isnan(age) else np.nan,
            }
            if competition is not None:
                record["competition"] = competition
                record["cutoff"] = cutoff
            records.append(record)
    return pd.DataFrame(records)


def group_keys(df):
    """What identifies one field of athletes being scored together.

    A Diamond-League-only run has one final per discipline-year, so the year is
    enough. Pooled, a discipline-year holds several finals with different
    fields and different cut-off dates, and ranking an athlete against the
    wrong one is how a season rank stops meaning anything."""
    return ["discipline", "year", "competition"] if "competition" in df.columns else ["discipline", "year"]


def add_season_rank(df):
    """discipline in FIELD_EVENTS -- not just "men_PV" -- must rank descending
    (higher mark = better = rank 1). Using a hardcoded "men_PV" string here
    previously left women_PV and men_LJ ranked backwards (lowest jump/vault
    getting rank 1) for the entire time they've been trained disciplines --
    caught while adding 10 more field events that would have hit the same bug."""
    all_groups = []
    for keys, group in df.groupby(group_keys(df)):
        discipline = keys[0]
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
    keys_used = group_keys(df)
    for keys, group in df.groupby(keys_used):
        discipline, year = keys[0], keys[1]
        group = group.copy()
        is_field = discipline in FIELD_EVENTS
        weighted_sb_map, wind_adj_map, trend_map, days_map = {}, {}, {}, {}
        gap_var_map = {}
        # Every dated mark this group can legally see, handed to the shared
        # form-feature definitions below. None when the discipline has no raw
        # file or no usable dates.
        form_log, form_reference = None, None
        # The final's own date when there is one, so nothing after the race
        # can reach a feature that claims to predict it. Without a cut-off
        # (the Diamond-League-only path) this stays the 1 September the model
        # was always built on.
        cutoff = group["cutoff"].iloc[0] if "cutoff" in group.columns else None

        raw_path = os.path.join(RAW_DIR, f"{discipline}.csv")
        if os.path.exists(raw_path):
            raw_full = pd.read_csv(raw_path)
            raw = raw_full[raw_full["year"] == year].copy()
            raw = raw.rename(columns={"Competitor": "athlete_name", "Mark": "mark_str"})
            raw["Mark"] = raw["mark_str"].apply(convert_mark_to_seconds)
            raw = raw.dropna(subset=["Mark"])
            if cutoff is not None and "Date" in raw.columns:
                run_on = pd.to_datetime(raw["Date"], format="%d %b %Y", errors="coerce")
                raw = raw[run_on < pd.Timestamp(cutoff)]

            if "Venue" in raw.columns:
                raw["comp_weight"] = raw["Venue"].apply(competition_weight)
                # Direction matters and used to be wrong for TRACK. The intent
                # is "a mark set at a bigger meet counts for more". For a TIME,
                # better means SMALLER, so a big-meet time has to be scaled
                # DOWN to look better -- multiplying it by 1.2 made it look
                # 20% slower, and the min() below then systematically picked
                # whichever mark came from the WEAKEST meet. Measured live
                # 2026-08-25: Bromell's weighted_season_best read 11.892 for a
                # 9.91 sprinter. Field events were always fine, because there
                # bigger is better and max() agrees with the multiply.
                raw["weighted_mark"] = (raw["Mark"] * raw["comp_weight"] if is_field
                                        else raw["Mark"] / raw["comp_weight"])
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
                ref_date = pd.Timestamp(cutoff) if cutoff is not None else pd.Timestamp(f"{year}-09-01")
                form_log = raw[["athlete_name", "date", "Mark"]].rename(
                    columns={"athlete_name": "name", "Mark": "mark"})
                form_reference = ref_date

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

        # The form group (--with-form). Definitions are IMPORTED from
        # feature_builder rather than written twice: run.py selects its feature
        # columns by name, so a column computed differently here and there is
        # present, correctly named, and quietly means something else.
        group["field_gap"] = feature_builder.field_gap(
            group["season_best"], is_field).to_numpy()
        sb_age, ratio = feature_builder.form_from_log(
            form_log, group["athlete_name"], group["season_best"],
            form_reference if form_reference is not None else f"{year}-09-01",
            is_field)
        group["sb_age_days"] = sb_age
        group["recent_best_ratio"] = ratio
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


def build_pooled_dataset(tiers=None):
    """The same features, but labelled against EVERY final we have a podium
    for -- Diamond League Finals, Olympics, World and European Championships,
    Continental Cup and the Continental Tour meetings -- instead of Diamond
    League Finals alone.

    980 finals against 215, which is the point: a model whose every label is a
    Diamond League Final learns "raced the Diamond League" as a shortcut to
    "podiums", because within that label set it is one. It cannot learn that
    from a pool where more than half the podiums were won at meetings the
    circuit's regulars did not enter.

    Each row is one athlete at one final, with features cut at that final's own
    date (see build_features). `tier` rides along so accuracy can be read per
    kind of competition rather than as a single average that hides which kind
    got worse."""
    labels = pd.read_csv(FINALS_LABELS_PATH, parse_dates=["date"])
    labels = labels[labels["year"].isin(LABEL_YEARS)]
    if tiers:
        labels = labels[labels["tier"].isin(tiers)]

    frames = []
    for key in TRAIN_DISCIPLINES:
        path = os.path.join(RAW_DIR, f"{key}.csv")
        if not os.path.exists(path):
            continue
        disc_labels = labels[labels["discipline"] == key]
        if disc_labels.empty:
            continue
        finals = (
            disc_labels[["year", "competition", "date"]]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )
        frames.append(build_features(clean_discipline(pd.read_csv(path)), key, list(finals)))

    master = pd.concat(frames, ignore_index=True)
    master = add_season_rank(master)
    master = add_new_features(master)
    master["_name_key"] = master["athlete_name"].apply(normalize_name)

    podium = labels.copy()
    podium["dl_rank"] = podium["place"]
    podium["dl_winner"] = (podium["place"] == 1).astype(int)
    podium["dl_top3"] = 1
    podium["_name_key"] = podium["name_key"]

    labeled = master.merge(
        podium[["discipline", "year", "competition", "_name_key",
                "dl_winner", "dl_top3", "dl_rank", "tier"]],
        on=["discipline", "year", "competition", "_name_key"], how="left",
    )
    # A row that matched no podium is a real negative: that athlete had a mark
    # in this discipline before this final and did not medal in it.
    labeled["dl_winner"] = labeled["dl_winner"].fillna(0).astype(int)
    labeled["dl_top3"] = labeled["dl_top3"].fillna(0).astype(int)
    labeled["dl_rank"] = labeled["dl_rank"].fillna(0).astype(int)
    tiers = labels.drop_duplicates(["discipline", "year", "competition"]).set_index(
        ["discipline", "year", "competition"])["tier"]
    labeled["tier"] = [
        tiers.get((d, y, c), "tour")
        for d, y, c in zip(labeled["discipline"], labeled["year"], labeled["competition"])
    ]
    return labeled.drop(columns=["_name_key"])


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


MIN_H2H_MEETINGS = 2  # matches run.py's inference-time threshold


def build_h2h_lookup(rates_df):
    """{a_lower: {b_lower: win_rate}} for one discipline.

    data/h2h/h2h_rates.csv uses normal-case names ("Trayvon Bromell") while
    every other data source in this pipeline uses WA's ALL-CAPS-surname
    format ("Trayvon BROMELL") -- an exact-string match between the two
    finds ZERO matches. This silently made h2h_win_rate default to a neutral
    0.5 for every athlete in every prediction ever made by run.py's blend,
    despite 156k real matchup rows sitting unused. Matching case-insensitive
    here (and in run.py) is the actual fix -- confirmed live: 0/8 exact
    matches vs 7/8 case-insensitive matches for a sample discipline."""
    lookup = {}
    for a, b, rate in zip(rates_df["athlete_a"].str.lower(),
                          rates_df["athlete_b"].str.lower(),
                          rates_df["win_rate"]):
        lookup.setdefault(a, {})[b] = rate
    return lookup


def mean_rate_against(lookup, names_lower):
    """Each athlete's average win rate against the others in the same list.

    0.5 -- an honest coin flip -- when this pairing has never been raced, which
    is most of them. This is the shape run.py computes at serve time against
    the field it is about to score."""
    out = []
    for me in names_lower:
        mine = lookup.get(me)
        if not mine:
            out.append(0.5)
            continue
        seen = [mine[opp] for opp in names_lower if opp != me and opp in mine]
        out.append(sum(seen) / len(seen) if seen else 0.5)
    return out


_DATED_RECORDS = None


def dated_h2h_records():
    """Every pairwise race result with the date it was run on, or an empty
    frame when the meet dates have not been scraped yet.

    data/h2h/h2h_rates.csv has no time dimension at all -- it aggregates every
    meeting on disk -- so reading it whole for a 2022 final scored that final
    on a rivalry as it stood in 2025.

    And it is worse than ordinary lookahead. The scraper's meeting list and the
    label file overlap: **229 of the 531 finals in the pooled training set have
    their own meeting in this data**, because the 2022 World Championships
    men's 100m page and the men_100m 2022 Worlds label are the same race. For
    those 43% of labels the model was reading who won the race it was being
    asked to predict, out of a feature. Measured over 10 paired seeds, that is
    worth 2.5 points of toplist accuracy and 2.0 of field accuracy, on the
    model's second-most-important feature.

    `python src/h2h_scraper.py --dates-only` writes the dates this needs."""
    global _DATED_RECORDS
    if _DATED_RECORDS is not None:
        return _DATED_RECORDS
    if not (os.path.exists(MEET_RESULTS_PATH) and os.path.exists(MEET_DATES_PATH)):
        _DATED_RECORDS = pd.DataFrame()
        return _DATED_RECORDS

    import h2h_calculator  # same directory; imported late so a plain
                           # `import train_model` stays cheap
    results = pd.read_csv(MEET_RESULTS_PATH, low_memory=False)
    records = h2h_calculator.pairwise_records(results)
    dates = pd.read_csv(MEET_DATES_PATH, parse_dates=["date"])[["meet", "date"]]
    records = records.merge(dates, on="meet", how="left")
    # An undated meet is dropped rather than defaulted. Giving it a date would
    # be inventing when a race happened in order to decide whether a later
    # final was allowed to see it, which is the exact mistake being fixed.
    undated = records["date"].isna().sum()
    if undated:
        print(f"  WARNING: {undated} h2h records from undated meets, excluded")
    records = records.dropna(subset=["date"])
    records["a_lower"] = records["athlete_a"].str.lower()
    records["b_lower"] = records["athlete_b"].str.lower()
    _DATED_RECORDS = records.sort_values("date").reset_index(drop=True)
    return _DATED_RECORDS


def add_h2h_features(df):
    """Adds h2h_win_rate: an athlete's average win rate against the rest of
    the pool being scored with them.

    THE CUT-OFF IS THE POINT, exactly as it is in build_features. When every
    label was one September Diamond League Final a year, reading the whole
    rivalry file was survivable. Pooled across 531 championship finals it reads
    the answer out of the future, and for 229 of them out of the race itself --
    see dated_h2h_records. So when the frame carries per-final cut-offs, the
    rates are rebuilt from the races run strictly BEFORE each final.

    A meeting is dated on its LAST day, which for a championship event page is
    the final's own day, so that championship's heats and semis are excluded
    along with the final. Slightly conservative -- a semi-final really is
    information available beforehand -- and it is the right kind of
    conservative: it matches what run.py can see when it projects a race that
    has not happened.

    Without cut-offs (the Diamond-League-only path, which build_labeled_dataset
    produces with neither a cutoff nor a competition column) this reads
    h2h_rates.csv whole and groups by discipline-year, which is byte-identical
    to the behaviour the deployed model was trained on. Verified, not assumed.

    run.py deliberately keeps reading the whole file at serve time. Predicting
    Budapest from every race already run is not lookahead, it is the point."""
    df = df.copy()
    keys = group_keys(df)
    records = dated_h2h_records() if "cutoff" in df.columns else pd.DataFrame()

    if records.empty:
        rates_df = pd.read_csv(H2H_PATH)
        rates_df = rates_df[rates_df["meetings"] >= MIN_H2H_MEETINGS]
        by_discipline = {d: build_h2h_lookup(g) for d, g in rates_df.groupby("discipline")}
        per_cutoff = None
    else:
        by_discipline = None
        # Aggregated once per distinct (discipline, cut-off) rather than once
        # per final: a discipline's finals share cut-off dates across years
        # only rarely, but rebuilding 531 times over 141k records is minutes
        # of work to arrive at the same tables.
        per_cutoff = {}
        for discipline, disc_records in records.groupby("discipline", sort=False):
            dates = disc_records["date"].to_numpy()
            cutoffs = sorted({pd.Timestamp(c) for c in
                              df.loc[df["discipline"] == discipline, "cutoff"].unique()})
            for cutoff in cutoffs:
                before = disc_records.iloc[:int(np.searchsorted(dates, cutoff, side="left"))]
                agg = h2h_aggregate(before)
                per_cutoff[(discipline, cutoff)] = build_h2h_lookup(agg) if len(agg) else {}

    rates = pd.Series(index=df.index, dtype=float)
    for _, group in df.groupby(keys, sort=False):
        discipline = group["discipline"].iloc[0]
        if per_cutoff is None:
            lookup = by_discipline.get(discipline, {})
        else:
            lookup = per_cutoff.get((discipline, pd.Timestamp(group["cutoff"].iloc[0])), {})
        names_lower = [n.lower() for n in group["athlete_name"]]
        # Assigned by INDEX, not position. The old version appended to a list
        # and assigned it positionally, which was only ever correct because
        # every caller happened to hand it a frame already sorted the way
        # groupby iterates. Nothing enforced that, and nothing would have
        # raised if it stopped being true.
        rates.loc[group.index] = mean_rate_against(lookup, names_lower)
    df["h2h_win_rate"] = rates
    return df


def h2h_aggregate(records):
    """Pairwise records collapsed to win rates, thresholded the way run.py
    thresholds them. Kept next to its caller rather than in h2h_calculator
    because the >=2 threshold is a modelling choice, not part of what a
    head-to-head record IS."""
    if records.empty:
        return pd.DataFrame(columns=["athlete_a", "athlete_b", "win_rate"])
    agg = records.groupby(["athlete_a", "athlete_b"], sort=False).agg(
        wins=("a_wins", "sum"), meetings=("total", "sum")).reset_index()
    agg = agg[agg["meetings"] >= MIN_H2H_MEETINGS].copy()
    agg["win_rate"] = agg["wins"] / agg["meetings"]
    return agg


def mark_final_field(df):
    """Flags the rows the model is actually DEPLOYED on.

    run.py only ever scores athletes in WA's Diamond League standings -- the
    ~8-10 who contest a Final -- but this backtest pools ~101 toplist
    athletes per discipline-year, so the headline number answers "find the
    podium among 101 names", which is nobody's question. build_labeled_dataset
    merges only `dl_rank <= 3`, so the frame cannot tell a 7th-place finalist
    from someone who never turned up; this reads the full field back out of
    dl_final_results.csv.

    Adds `in_final_field`, used to report a second metric alongside the
    original one. It changes no training and no features.

    Pooled runs read the real start list of EACH final from
    data/labels/final_fields.csv instead. Reusing the Diamond League field
    there would ask "did the model find the World Championships podium among
    the athletes who contested the Diamond League Final?", which is a question
    about two different meetings and would quietly report a number for it."""
    df = df.copy()
    keys = df["athlete_name"].apply(normalize_name)
    if "competition" in df.columns and os.path.exists(FINALS_FIELDS_PATH):
        fields = pd.read_csv(FINALS_FIELDS_PATH)
        field = set(zip(fields["discipline"], fields["year"],
                        fields["competition"], fields["name_key"]))
        df["in_final_field"] = [
            (d, y, c, k) in field
            for d, y, c, k in zip(df["discipline"], df["year"], df["competition"], keys)
        ]
        return df

    results = pd.read_csv(DL_RESULTS_PATH)
    results["_k"] = results["athlete_name"].apply(normalize_name)
    field = set(zip(results["discipline"], results["year"], results["_k"]))
    df["in_final_field"] = [
        (d, y, k) in field for d, y, k in zip(df["discipline"], df["year"], keys)
    ]
    return df


DEFAULT_MODEL_PARAMS = {
    "n_estimators": 300, "max_depth": 16, "min_samples_leaf": 4,
    "class_weight": None, "random_state": 42,
}
# 2026-09-08: min_samples_leaf 1 -> 4, n_estimators 200 -> 300. This is the
# FIRST re-tune ever run against the model that actually ships -- --tune built
# the Diamond-League-only dataset until today, so every earlier round tuned a
# model nobody runs. The grid put this candidate 0.3 ahead at seed 42, which
# this project's own noise floor says means nothing; measured properly it is
# +0.71 toplist on 10 of 10 paired seeds and +0.64 field on 8 of 10. Adopted on
# that, not on the grid.
#
# Earlier history, kept: walk-forward-tuned via --tune (2026-08-22, re-tuned 2026-08-23 twice more: once
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
    # Second scoring pass over the same predictions, restricted to the
    # athletes who actually contested that Final -- the population run.py
    # applies the model to. Same model, same probabilities, different
    # candidate pool.
    field_correct, field_possible = 0, 0
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
           "winner_is_top_pick": 0, "winner_count": 0,
           "field_found": {1: 0, 2: 0, 3: 0}, "field_total": {1: 0, 2: 0, 3: 0},
           "field_correct": 0, "field_possible": 0,
           # Pooled runs only: hits and chances per KIND of competition. The
           # whole risk of pooling this widely is that "podium in a final"
           # stops meaning one thing, and a single average is exactly what
           # would hide it happening.
           "tier_correct": {}, "tier_possible": {},
           # ...and the same per tier among the athletes who were ACTUALLY
           # THERE, which for pooled data is the only coherent version. The
           # toplist metric ranks the whole world's season list, and a
           # European final can only be won by Europeans: measured on the 2024
           # men's 1500m, the world top three were Ingebrigtsen, Hocker and
           # Kerr, while the European podium was Ingebrigtsen, Vermeulen and
           # Arese -- Hocker was never eligible to enter. Marking the model
           # wrong for naming him scores it against a rule it was never told.
           # The same holds, less starkly, for an invitational meeting most of
           # the toplist simply did not attend.
           "tier_field_correct": {}, "tier_field_possible": {}}
    # One scored contest per final. Grouping by discipline alone was right
    # while a discipline-year held exactly one final; pooled it would mash
    # every final that discipline ran that season into one ranking.
    contest_keys = ["discipline", "competition"] if "competition" in test.columns else ["discipline"]
    for keys, disc_df in test.groupby(contest_keys):
        discipline = keys[0] if isinstance(keys, tuple) else keys
        disc_df = disc_df.sort_values("win_probability", ascending=False)
        top3_predicted = disc_df.head(3)["athlete_name"].tolist()
        actual_rows = disc_df[disc_df["dl_top3"] == 1]
        top3_actual = actual_rows["athlete_name"].tolist()
        hits = len(set(top3_predicted) & set(top3_actual))
        correct += hits
        possible += 3
        per_discipline.append((discipline, hits))
        if "tier" in disc_df.columns:
            tier = disc_df["tier"].iloc[0]
            pos["tier_correct"][tier] = pos["tier_correct"].get(tier, 0) + hits
            pos["tier_possible"][tier] = pos["tier_possible"].get(tier, 0) + 3

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

        if "in_final_field" in disc_df.columns:
            fd = disc_df[disc_df["in_final_field"]]
            if not fd.empty:
                field_picked = set(fd.head(3)["athlete_name"])
                field_actual = fd[fd["dl_top3"] == 1]
                field_correct += len(field_picked & set(field_actual["athlete_name"]))
                field_possible += 3
                for _, row in field_actual.iterrows():
                    rank = int(row["dl_rank"]) if pd.notna(row.get("dl_rank")) else None
                    if rank in pos["field_total"]:
                        pos["field_total"][rank] += 1
                        if row["athlete_name"] in field_picked:
                            pos["field_found"][rank] += 1
                if "tier" in disc_df.columns:
                    tier = disc_df["tier"].iloc[0]
                    hits_here = len(field_picked & set(field_actual["athlete_name"]))
                    pos["tier_field_correct"][tier] = pos["tier_field_correct"].get(tier, 0) + hits_here
                    pos["tier_field_possible"][tier] = pos["tier_field_possible"].get(tier, 0) + 3

    pos["field_correct"] = field_correct
    pos["field_possible"] = field_possible
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
    # Which years get SCORED is deliberately pinned, not derived from how many
    # label years exist. It used to be LABEL_YEARS[2:] -- "everything after the
    # first two" -- which is the same thing while the labels are 2018-2025, and
    # stops being the same thing the moment history is added behind them.
    #
    # Extending the labels back to 2009 (2026-09-08) would otherwise have moved
    # the test years from 2021-2025 to 2011-2025, changing the denominator from
    # 1,119 predictions to something else and quietly making every accuracy
    # figure in HANDOFF incomparable to the next one. Worse, the added folds are
    # the WEAKEST ones -- a 2011 fold trains on 2009-2010 alone -- so more and
    # better training data would have shown up as a lower number.
    #
    # Extra history feeds TRAINING, which is what it is for. The scored
    # population does not move.
    test_years = [y for y in LABEL_YEARS if y >= FIRST_TEST_YEAR]
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
            for rank in (1, 2, 3):
                stats.setdefault("field_found", {}).setdefault(rank, 0)
                stats.setdefault("field_total", {}).setdefault(rank, 0)
                stats["field_found"][rank] += pos["field_found"][rank]
                stats["field_total"][rank] += pos["field_total"][rank]
            stats["field_correct"] = stats.get("field_correct", 0) + pos["field_correct"]
            stats["field_possible"] = stats.get("field_possible", 0) + pos["field_possible"]
            for bucket in ("tier_correct", "tier_possible",
                           "tier_field_correct", "tier_field_possible"):
                target = stats.setdefault(bucket, {})
                for tier, n in pos[bucket].items():
                    target[tier] = target.get(tier, 0) + n
        if verbose:
            fold_acc = round(correct / possible * 100, 1) if possible else 0.0
            print(f"  train {train_years} -> test {test_year}: {correct}/{possible} = {fold_acc}%")
        total_correct += correct
        total_possible += possible
    return total_correct, total_possible


def tune_hyperparameters(feature_cols, pooled=False, tiers=None):
    """Small grid search over RandomForest params, each candidate scored by
    the exact same walk-forward folds train_and_backtest() reports honest
    accuracy with -- so "best" here means "generalizes best across 2023,
    2024, 2025", not "fits one split best". Informational only: prints
    ranked results and returns the winner, doesn't touch outputs/. Whoever
    reads the results decides whether to make a candidate the new
    DEFAULT_MODEL_PARAMS -- this deliberately isn't automatic, so a
    retrain's default hyperparameters can't silently drift between runs.

    `pooled` matters and did not used to be here. The DEPLOYED model is the
    531-final championships pool, and this only ever built the
    Diamond-League-only dataset -- so every re-tune since that model shipped
    has been tuning a model nobody runs, and reporting it under a heading that
    did not say so. Pass the same --pooled --tiers the retrain uses."""
    if pooled:
        full = build_pooled_dataset(tiers)
    else:
        full = add_new_features(add_season_rank(build_labeled_dataset()))
    full = add_h2h_features(full)
    full = full.dropna(subset=feature_cols)

    grid = [
        {"n_estimators": n, "max_depth": d, "min_samples_leaf": leaf, "class_weight": cw, "random_state": 42}
        for n in [100, 200, 300]
        for d in [None, 8, 16]
        for leaf in [1, 2, 4]
        for cw in ["balanced", None]
    ]

    print(f"\n=== Hyperparameter search ({len(grid)} candidates, walk-forward scored"
          f"{', pooled' if pooled else ''}) ===")
    results = []
    for params in grid:
        # n_jobs is a speed knob, not a modelling one -- a forest's trees are
        # seeded from random_state regardless of how many cores build them --
        # so it is added at scoring time and kept OUT of the grid dicts, which
        # are compared against DEFAULT_MODEL_PARAMS by equality below.
        correct, possible = walk_forward_folds(
            full, feature_cols, {**params, "n_jobs": -1}, verbose=False)
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


def _print_both_metrics(stats, toplist_pct):
    """Print the metric the SITE's task actually corresponds to, beside the
    historical one.

    The number above it picks 3 from every athlete in a discipline-year's
    toplist (~101 names). run.py picks 3 from the ~8-10 athletes in WA's
    Diamond League standings. Those are different questions and they get
    different scores -- measured 2026-08-25, 59.7% and 72.3% off the exact
    same predictions.

    Neither is wrong. The toplist figure is the one every number in
    HANDOFF.md was measured with and stays the comparison baseline; the
    field figure is the one the site should quote about itself. Returns the
    field figure so save_artifacts can record both."""
    possible = stats.get("field_possible", 0)
    if not possible:
        return None
    correct = stats.get("field_correct", 0)
    field_pct = round(100.0 * correct / possible, 1)
    print("")
    print("  Two candidate pools, same predictions -- these answer different questions:")
    print(f"    among the full ~101-athlete toplist : {toplist_pct}%   "
          f"(the historical baseline; every HANDOFF number is this one)")
    print(f"    among the real Final field only     : {field_pct}%   "
          f"({correct}/{possible}) -- the task run.py actually performs")

    found, total = stats.get("field_found", {}), stats.get("field_total", {})
    if any(total.values()):
        parts = " ".join(
            f"{label} {100.0 * found.get(r, 0) / total[r]:.0f}%"
            for r, label in ((1, "1st"), (2, "2nd"), (3, "3rd")) if total.get(r)
        )
        print(f"    per position, real field            : {parts}")
    return field_pct


def _print_tier_breakdown(stats):
    """Accuracy per kind of competition, for pooled runs.

    This is the number that decides whether pooling worked. An overall average
    can rise while the Diamond League Finals -- the thing the site currently
    predicts -- get worse, and that trade has to be visible before anyone
    accepts it."""
    correct, possible = stats.get("tier_correct", {}), stats.get("tier_possible", {})
    fc, fp = stats.get("tier_field_correct", {}), stats.get("tier_field_possible", {})
    if not possible:
        return
    print("\n  By kind of competition (the pooling risk, made visible):")
    print("                 among the world toplist |   among who was actually there")
    for tier in sorted(possible, key=lambda t: -possible[t]):
        pct = 100.0 * correct.get(tier, 0) / possible[tier]
        line = f"    {tier:<12} {correct.get(tier, 0):>5}/{possible[tier]:<5} = {pct:>5.1f}%"
        if fp.get(tier):
            line += f"   |   {fc.get(tier, 0):>5}/{fp[tier]:<5} = {100.0 * fc.get(tier, 0) / fp[tier]:>5.1f}%"
        print(line)
    print("    The right-hand column is the one to read: the left ranks the whole world's")
    print("    season list, which includes athletes who were not eligible to enter.")


def train_and_backtest(feature_cols, label="", model_params=None, pooled=False, tiers=None):
    """Walk-forward (expanding-window) validation: train on every labeled
    year strictly before each test year, one fold per test year, instead of
    a single fixed train/test split. With only 3 years of labels a single
    holdout was the only option -- now that real scraped data covers
    2021-2025, testing on 3 independent years (2023, 2024, 2025) instead of
    just one gives a far more honest read on whether the model generalizes,
    not just whether it fit one particular season."""
    if pooled:
        # build_pooled_dataset already ranks and adds the recency features,
        # because both of those have to happen per FINAL rather than per year.
        full = build_pooled_dataset(tiers)
    else:
        labeled = build_labeled_dataset()
        ranked = add_season_rank(labeled)
        full = add_new_features(ranked)
    full = add_h2h_features(full)
    full = full.dropna(subset=feature_cols)
    full = mark_final_field(full)

    print(f"\n=== {label} — walk-forward validation ({LABEL_YEARS[0]}-{LABEL_YEARS[-1]}) ===")
    pos_stats = {}
    total_correct, total_possible = walk_forward_folds(
        full, feature_cols, model_params, verbose=True, stats=pos_stats)
    accuracy_pct = round(total_correct / total_possible * 100, 1) if total_possible else 0.0
    print(f"  Overall (all {len(LABEL_YEARS[2:])} folds combined): {total_correct}/{total_possible} = {accuracy_pct}%")
    field_pct = _print_both_metrics(pos_stats, accuracy_pct)
    _print_position_breakdown(pos_stats)
    _print_tier_breakdown(pos_stats)

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

    return model, scaler, accuracy_pct, field_pct


def save_artifacts(model, scaler, feature_cols, accuracy_pct, field_pct=None, out_dir=None):
    """Writes the trained model where run.py and world_rankings.py look for it.

    `out_dir` exists so an experimental model can be trained and driven through
    the whole site WITHOUT displacing the one that is deployed. A pooled model
    saved over outputs/ would silently become the live model on the next
    refresh, which is not a thing to find out afterwards."""
    out_dir = out_dir or OUTPUTS_DIR
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "model_rf.pkl"), "wb") as f:
        pickle.dump(model, f)
    with open(os.path.join(out_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(out_dir, "feature_cols.pkl"), "wb") as f:
        pickle.dump(feature_cols, f)
    # Unchanged meaning on purpose: this file has always held the
    # ~101-athlete-toplist number and other code reads it.
    with open(os.path.join(out_dir, "model_accuracy.txt"), "w") as f:
        f.write(str(accuracy_pct))
    # Both metrics, labelled, so the site can quote the one that matches
    # what it actually does instead of the historical baseline.
    with open(os.path.join(out_dir, "model_metrics.json"), "w") as f:
        json.dump({
            "toplist_pool_pct": accuracy_pct,
            "final_field_pct": field_pct,
            "note": ("toplist_pool_pct picks 3 from every athlete in a "
                     "discipline-year's toplist (~101); final_field_pct picks 3 "
                     "from the ~8-10 who actually contested the Final, which is "
                     "what run.py does."),
        }, f, indent=2)
    print(f"\nSaved model_rf.pkl, scaler.pkl, feature_cols.pkl, model_accuracy.txt "
          f"({accuracy_pct}%), model_metrics.json (field {field_pct}%)")


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
    parser.add_argument("--with-form", action="store_true",
                        help="Add sb_age_days, recent_best_ratio and field_gap -- how old the "
                             "season best is, how far off it the athlete currently is, and how "
                             "far behind the field's leader they sit in MARKS rather than "
                             "places. Measured together at +0.73 toplist / +0.97 field over 10 "
                             "paired seeds (9/10 and 10/10 wins), and +0.73/+1.19 again on ten "
                             "held-out seeds. Individually two of the three are nothing; the "
                             "gain is the three together, and it survives a same-width shuffled "
                             "control, which a gain from merely widening the feature matrix "
                             "would not.")
    parser.add_argument("--pooled", action="store_true",
                        help="Train on EVERY final we have a podium for (data/labels/finals.csv: "
                             "Diamond League Finals, Olympics, World and European Championships, "
                             "Continental Cup and Continental Tour meetings -- 980 finals against "
                             "215) instead of Diamond League Finals alone. Features are cut at "
                             "each final's own date. Reports accuracy per kind of competition.")
    parser.add_argument("--tiers", default=None,
                        help="Comma-separated tiers to keep when --pooled: dl_final, global, "
                             "continental, tour. Omit for all four. Exists because the widest pool "
                             "is not automatically the best one and that has to be measurable.")
    parser.add_argument("--out-dir", default=None,
                        help="Where to save the trained model. Defaults to outputs/, which is "
                             "what run.py and world_rankings.py read -- point it elsewhere to "
                             "try a model without displacing the deployed one.")
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
    if args.with_form:
        feature_cols += ["sb_age_days", "recent_best_ratio", "field_gap"]
        label_parts.append("form")
    label = " + ".join(label_parts)

    tiers = [t.strip() for t in args.tiers.split(",")] if args.tiers else None

    if args.tune:
        tune_hyperparameters(feature_cols, pooled=args.pooled, tiers=tiers)
        sys.exit(0)

    if args.pooled:
        label += " + pooled finals"
    if tiers:
        label += " [" + ",".join(tiers) + "]"
    model, scaler, accuracy_pct, field_pct = train_and_backtest(
        feature_cols, label=label, pooled=args.pooled, tiers=tiers)

    if not args.dry_run:
        save_artifacts(model, scaler, feature_cols, accuracy_pct, field_pct, args.out_dir)
    else:
        print("\n[dry run — outputs/ not modified]")
