"""Real 2026-season feature building for run.py's live prediction pipeline.

Extracted out of run.py (2026-08-24) so build_2026_features() and its small
helpers can be imported and unit-tested directly. run.py itself is a
straight-through script with no `if __name__ == "__main__":` guard -- the
moment it's imported, its top-level code kicks off a real ~1hr live scrape
(see its own docstring/HANDOFF.md's "near-miss" note about this). Pulling
the pure, side-effect-free feature logic out here means tests can import it
safely; run.py imports these same functions instead of defining them
inline, so this is a pure move, not a behavior change.
"""
import os

import numpy as np
import pandas as pd
from datetime import date

RAW_DIR = "data/raw"

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
