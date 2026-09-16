"""
field_model.py -- podium chances for any field, each athlete rated against the
others in the same final.

WHY
The model in train_model.py answers "who podiums at a Diamond League-calibre
final, given the world top 100". Asked about an Asian Games field it scores most
entrants against the world list and against Diamond League activity they never
had, so their chances come out under 1%, and the Asian Games call sent 28 of 36
events to a plain points ranking (2026-09-14). This model never looks outside the
field it is asked about.

FEATURES (field_features; the same function trains and serves)
From each athlete's season-best Results Score before the competition, measured
against the rest of the field: gap to the best and gap to the third best, which
stay the same however many slower athletes are added. From their own history:
career best before this season, change on last season, how long ago the season
best was set, age. Plus whether they had no mark this season before the
competition, so that last season's stands in, and whether they have any
history at all. No Diamond League meetings, no head-to-head, no world rank.

OLD MARKS (the user's rule, 2026-09-16)
Recent performances outweigh old ones, and a mark counts for less the older it
is. faded_level() reads an athlete's level as this season in full, lifted by at
most a faded share of an older best that still stands above it, and never by
enough to pass whoever is ahead of them on this season's marks: so an athlete
breaking records this season is not beaten by a mark that was good two years
ago, whatever that mark was. The rule is structure rather than a fitted weight,
and FADE_BOUNDS holds the signs it needs. How much last season counts and how
fast older seasons fade (FADE_LAST_SEASON, FADE_PER_YEAR, FADE_CAP_SHARE) are
tuned on past seasons.

MODEL
Plackett-Luce: a softmax over the field on a linear score of those features,
fitted by maximum likelihood on the order of each final's top three. A podium
chance is the exact probability of finishing in the top three under that model,
so a field's chances add up to three podium places.

THE SHIP RULE (fixed before any backtest was run, user decision 2026-09-14)
The model is used for an event group only where it beats simply taking the top
three by season-best points, on finals it was not trained on (walk-forward,
scored 2021-2025). Per group, with d = model hits minus points hits per held-out
final: the one-sided 90% bootstrap lower bound of the mean d must be above 0,
the mean d on the Asian finals alone must be at least 0, and the same model
trained on features shuffled within each field must NOT pass. A group that
fails stays on points.

Usage:
    python src/field_model.py --backtest            # report + ship decisions, fits and saves the model
    python src/field_model.py --compare             # race by race against today, on DEV_YEARS
    python src/field_model.py --experiments [NAME]  # try EXPERIMENTS on DEV_YEARS, every run logged
    python src/field_model.py --holdout NAME        # the chosen experiment on HOLDOUT_YEARS, once
    python src/field_model.py --all-seasons         # three ways old marks count, every season
    python src/field_model.py --rule-test [NAME]    # the rule for old marks, on made-up finals
    python src/field_model.py --old-marks           # trial and error on the old-marks settings, every season
    python src/field_model.py --refit [NAME]        # retrain the served experiment on every final on file
Reads data/field/finals.csv (field_data.py --report). --backtest writes
outputs/field_model.json and outputs/field_model_report.json. --compare,
--experiments and --holdout write field_model_compare.json,
field_model_experiments.json and field_model_holdout.json (the holdout also
saves its candidate to field_model_v2.json), and never the served model.
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import field_data as fd  # noqa: E402

BASE_DIR = fd.BASE_DIR
MODEL_PATH = os.path.join(BASE_DIR, "outputs", "field_model.json")
REPORT_PATH = os.path.join(BASE_DIR, "outputs", "field_model_report.json")
COMPARE_PATH = os.path.join(BASE_DIR, "outputs", "field_model_compare.json")
V2_MODEL_PATH = os.path.join(BASE_DIR, "outputs", "field_model_v2.json")
PREVIOUS_MODEL_PATH = os.path.join(BASE_DIR, "outputs", "field_model_previous.json")
EXPERIMENTS_PATH = os.path.join(BASE_DIR, "outputs", "field_model_experiments.json")
HOLDOUT_PATH = os.path.join(BASE_DIR, "outputs", "field_model_holdout.json")
ALL_SEASONS_PATH = os.path.join(BASE_DIR, "outputs", "field_model_all_seasons.json")
OLD_MARKS_PATH = os.path.join(BASE_DIR, "outputs", "field_model_old_marks.json")

# The same scored seasons as train_model.FIRST_TEST_YEAR onwards, so the two
# models' numbers cover the same years (tests/test_field_model.py pins it).
TEST_YEARS = [2021, 2022, 2023, 2024, 2025]
# Improving a model on the finals it is scored on measures the improving, not
# the model. From 2026-09-15, when the user asked to keep improving the model
# against past results, candidates are tried and compared on the first three
# test years only. The last two stay locked until one candidate has been chosen
# (choose_experiment), which is then scored on them once (run_holdout). Fixed
# before any candidate's number existed.
DEV_YEARS = [2021, 2022, 2023]
HOLDOUT_YEARS = [2024, 2025]
# A final is used only when this many of its finalists have a season score.
# Fewer, and "rank within the field" describes a handful of names, not a field.
MIN_SCORED = 5
L2 = 0.01
BOOTSTRAP = 10_000
LOWER_QUANTILE = 0.10

GROUPS = {
    "sprints": {"100m", "200m", "400m", "110h", "100h", "400h"},
    "distance": {"800m", "1500m", "5000m", "10000m", "3000sc"},
    "jumps": {"HJ", "PV", "LJ", "TJ"},
    "throws": {"SP", "DT", "JT", "HT"},
}

# No feature depends on how many weaker athletes share the field. The model
# learns on finals but is asked about entry lists (the Asian Games) and a world
# top 20 (Track/Field), both longer than a final, and a z-score or a rank divided
# by field size would read the same athlete differently in each. Both were in
# the first version and were taken out on 2026-09-14, before any backtest ran.
FEATURES = ["gap_best", "gap_third", "pb_gap", "no_history", "sb_months", "yoy", "age", "sb_prior_season"]
# Race by race, from field_data.race_summary and h2h_top (2026-09-15). The user
# asked for a call that moves after big meetings, rates a clear leader clearly
# and leans on history when a season is thin. compare() measures these against
# FEATURES, under a rule fixed before it first ran.
NEW_FEATURES = ["form_gap", "recent_delta", "no_recent", "big_podiums", "h2h_top", "pb_gap_thin",
                "breakout_backed"]
FEATURES_V2 = FEATURES + NEW_FEATURES
# A season of this many marks or fewer is thin.
THIN_SEASON = 2
CONTROLS = 5

# The user's rule (2026-09-15): an athlete's new marks are never outweighed by
# their old ones, and the more consistent their season, the more they are
# judged on it alone. strength() blends this season's form with the career best
# before it, the new marks weighing at least half: all of it for an athlete with
# CONSISTENT_RACES marks inside CONSISTENT_SPREAD points, and all of it for an
# athlete who is better now than before. The pb_gap, yoy, pb_gap_thin and
# breakout_backed features, which let history outweigh the season, are left out.
CONSISTENT_RACES = 5
CONSISTENT_SPREAD = 50.0
STRENGTH_GAP_FEATURES = ["strength_gap_best", "strength_gap_third"]
STRENGTH_FEATURES = STRENGTH_GAP_FEATURES + ["consistency"]
RECENCY_FEATURES = STRENGTH_FEATURES + ["no_history", "sb_months", "age", "sb_prior_season",
                                        "recent_delta", "no_recent", "big_podiums", "h2h_top"]
# Signs the rule fixes, held as bounds on the fit: a stronger, more consistent
# or more in-form athlete can only gain, and being read on last season's mark
# for want of one this season can only count against them.
RECENCY_BOUNDS = {"strength_gap_best": (0.0, None), "strength_gap_third": (0.0, None),
                  "consistency": (0.0, None), "recent_delta": (0.0, None), "sb_prior_season": (None, 0.0)}
# The user's rule for old marks (2026-09-16), and why it is structure and not a
# fitted weight: they asked that recent performances outweigh old ones, that a
# mark count for less the older it is, and that an athlete breaking records this
# season never be beaten by a mark that was good two years ago. faded_level()
# reads this season in full, lifted by at most a faded share of an older best
# that still stands above it, and never by enough to pass the athlete leading on
# this season's marks. The features that let history outweigh the season
# (pb_gap, pb_gap_thin, yoy, breakout_backed) are left out, as in the recency
# family. These three numbers are what the grid on past seasons tunes; a spec's
# "fade" replaces them.
# Chosen by run_old_marks on 2026-09-16: every one of 30 settings walked
# forward over every season from 2012 to 2026, and the most accurate of those
# the rule allows (outputs/field_model_old_marks.json).
FADE_LAST_SEASON = 0.35
FADE_PER_YEAR = 0.3
FADE_CAP_SHARE = 0.75
FADE_GAP_FEATURES = ["fade_gap_best", "fade_gap_third"]
FADE_FEATURES = FADE_GAP_FEATURES + ["consistency", "no_history", "sb_months", "age", "sb_prior_season",
                                     "recent_delta", "no_recent", "big_podiums", "h2h_top"]
# The signs the rule fixes, as RECENCY_BOUNDS does for the recency family: a
# higher level, a steadier season or better recent form can only help, and being
# read on last season's mark for want of one this season can only count against.
FADE_BOUNDS = {"fade_gap_best": (0.0, None), "fade_gap_third": (0.0, None), "consistency": (0.0, None),
               "recent_delta": (0.0, None), "sb_prior_season": (None, 0.0)}
ALL_FEATURES = list(dict.fromkeys(FEATURES_V2 + RECENCY_FEATURES + FADE_FEATURES))
# The user's question on 2026-09-16: should old marks count at all? These read
# an athlete's career best or last season's best. SEASON_ONLY_FEATURES is
# FEATURES_V2 without them; sb_prior_season stays, so last season stands in
# only for an athlete with no mark this season.
OLD_MARK_FEATURES = ["pb_gap", "pb_gap_thin", "yoy", "breakout_backed", "no_history"]
SEASON_ONLY_FEATURES = [name for name in FEATURES_V2 if name not in OLD_MARK_FEATURES]


def group_of(disc_key):
    event = str(disc_key).split("_", 1)[-1]
    return next((name for name, events in GROUPS.items() if event in events), None)


# ---- features -----------------------------------------------------------------

def field_features(rows, cutoff, features=FEATURES, fade=None):
    """`features` for every row of one field that has a season score: today's
    FEATURES unless asked for FEATURES_V2.

    `rows` needs sb_score, sb_date, sb_prior_season, career_best,
    prev_season_best and dob, and the race-by-race features read
    field_data.RACE_COLUMNS as well. Scores are divided by 100 so a coefficient
    reads per 100 World Athletics points. An athlete with no age takes the
    field's median, and 25 when nobody in the field has one.

    The race-by-race features (2026-09-15), none of which moves when slower
    athletes join the field:
      form_gap         form, the mean of the best three scores, below the field's
                       best season score; the season best stands in for an
                       athlete with no races on file
      recent_delta     the best score of the last six weeks against form, 0 with none
      no_recent        no score in those six weeks
      big_podiums      podiums in finals at big meetings this season
      h2h_top          net record against the field's three strongest others
      pb_gap_thin      pb_gap in a season of THIN_SEASON marks or fewer, else 0:
                       history counts for more when the season says little
      breakout_backed  how far form sits above the career best: a breakout the
                       races back up, where one big mark barely moves form

    The recency family (RECENCY_FEATURES) reads strength() instead:
      strength_gap_best, strength_gap_third  strength below the field's best
                       and its third best
      consistency      how many marks and how tightly the best of them sit

    The old-marks family (FADE_FEATURES) reads faded_level() instead, whose
    three settings come from `fade` ({last_season, per_year, cap_share}), the
    module's defaults when it is None:
      fade_gap_best, fade_gap_third  the faded level below the field's best
                       level and its third best"""
    df = rows[pd.notna(rows["sb_score"])].copy()
    cutoff = pd.Timestamp(cutoff)
    s = df["sb_score"].astype(float)
    n = len(df)
    ordered = s.sort_values(ascending=False).to_numpy()
    third = ordered[2] if n >= 3 else (ordered[-1] if n else 0.0)

    out = pd.DataFrame(index=df.index)
    out["gap_best"] = (s - (ordered[0] if n else 0.0)) / 100.0
    out["gap_third"] = (s - third) / 100.0
    career = pd.to_numeric(df["career_best"], errors="coerce")
    out["pb_gap"] = ((career - s) / 100.0).fillna(0.0)
    out["no_history"] = career.isna().astype(float)
    sb_dates = pd.to_datetime(df["sb_date"], errors="coerce")
    out["sb_months"] = ((cutoff - sb_dates).dt.days / 30.44).fillna(12.0)
    prior = pd.to_numeric(df["sb_prior_season"], errors="coerce").fillna(0.0)
    prev = pd.to_numeric(df["prev_season_best"], errors="coerce")
    out["yoy"] = ((s - prev) / 100.0).where(prior == 0).fillna(0.0)
    ages = (cutoff - pd.to_datetime(df["dob"], errors="coerce")).dt.days / 365.25
    out["age"] = ages.fillna(ages.median() if ages.notna().any() else 25.0)
    out["sb_prior_season"] = prior
    if any(name in NEW_FEATURES for name in features):
        form = _column(df, "form_score").fillna(s)
        recent = _column(df, "recent_score")
        out["form_gap"] = (form - (ordered[0] if n else 0.0)) / 100.0
        out["recent_delta"] = ((recent - form) / 100.0).fillna(0.0)
        out["no_recent"] = recent.isna().astype(float)
        out["big_podiums"] = _column(df, "big_podiums").fillna(0.0)
        out["h2h_top"] = _column(df, "h2h_top").fillna(0.0)
        out["pb_gap_thin"] = out["pb_gap"] * (_column(df, "races").fillna(0.0) <= THIN_SEASON)
        out["breakout_backed"] = ((form - career).clip(lower=0) / 100.0).fillna(0.0)
    if "consistency" in features:
        out["consistency"] = consistency_of(df)
    if any(name in STRENGTH_GAP_FEATURES for name in features):
        value, _, _ = strength(df)
        by_strength = value.sort_values(ascending=False).to_numpy()
        third_strength = by_strength[2] if n >= 3 else (by_strength[-1] if n else 0.0)
        out["strength_gap_best"] = (value - (by_strength[0] if n else 0.0)) / 100.0
        out["strength_gap_third"] = (value - third_strength) / 100.0
    if any(name in FADE_GAP_FEATURES for name in features):
        level, _ = faded_level(df, cutoff, **(fade or {}))
        by_level = level.sort_values(ascending=False).to_numpy()
        third_level = by_level[2] if n >= 3 else (by_level[-1] if n else 0.0)
        out["fade_gap_best"] = (level - (by_level[0] if n else 0.0)) / 100.0
        out["fade_gap_third"] = (level - third_level) / 100.0
    return out[list(features)].astype(float)


def _column(df, name):
    if name not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[name], errors="coerce")


def consistency_of(df):
    """How settled a season is: how many marks there are, up to CONSISTENT_RACES,
    times how tightly the best of them sit, 1 when they are level and 0 at
    CONSISTENT_SPREAD points apart. Read by strength() and by the old-marks
    family, so both mean the same thing by a steady season."""
    races = _column(df, "races").fillna(1.0).clip(lower=1.0, upper=CONSISTENT_RACES)
    tight = (1.0 - _column(df, "form_spread").fillna(0.0).clip(lower=0.0) / CONSISTENT_SPREAD).clip(lower=0.0)
    return races / CONSISTENT_RACES * tight


def strength(df):
    """(strength, consistency, the weight on the new marks) for each athlete, in
    World Athletics points: the user's rule that new marks count for more.

    The new level is form_score, the mean of the season's best marks, or the
    season best with no races on file. The old level is career_best, the best
    before this season. Consistency is how many marks there are, up to
    CONSISTENT_RACES, times how tightly the best of them sit: 1 when they are
    level, 0 at CONSISTENT_SPREAD points apart. The new marks weigh
    0.5 + 0.5 x consistency, so the old never weigh more than half, and an
    athlete at or above their old level is read on the new marks alone."""
    new = _column(df, "form_score").fillna(pd.to_numeric(df["sb_score"], errors="coerce"))
    old = _column(df, "career_best")
    consistency = consistency_of(df)
    weight = 0.5 + 0.5 * consistency
    value = new.where(old.isna() | (new >= old), weight * new + (1.0 - weight) * old)
    return value, consistency, weight


def fade_weight(ago, last_season=FADE_LAST_SEASON, per_year=FADE_PER_YEAR):
    """How much a best from `ago` seasons back counts: `last_season` for last
    season, times `per_year` for every further year back. Below 1 at every age
    with the defaults, so an old mark never counts as fully as this season."""
    return last_season * per_year ** (np.asarray(ago, dtype=float) - 1.0)


def faded_level(df, cutoff, last_season=FADE_LAST_SEASON, per_year=FADE_PER_YEAR, cap_share=FADE_CAP_SHARE):
    """(level, the parts it is made of) for each athlete in World Athletics
    points: the user's rule for old marks (2026-09-16).

    This season counts in full. The level starts at form_score, the mean of the
    season's best marks, or the season best for an athlete with no races on
    file. An earlier season's best counts only where it stands above that, and
    only at fade_weight() of the distance above it, so the older the mark the
    less it lifts them. The largest of those lifts is the one that counts: they
    do not stack, or three ordinary old seasons would add up to a good one. An
    athlete at or above all of their earlier bests is read on this season alone.

    The lift is then capped at `cap_share` of the distance to the best level in
    the field on this season's marks. That is what holds the user's rule whatever
    the old mark: an old best can close a gap but never carry an athlete past
    someone who is ahead of them this season, so the record-breaker stays in
    front. `parts` carries the season level, the lift, the season it came from
    and its weight, for the reasons shown per athlete."""
    season_year = pd.Timestamp(cutoff).year
    now = _column(df, "form_score").fillna(pd.to_numeric(df["sb_score"], errors="coerce"))
    ones = pd.Series(1.0, index=df.index)
    older_year = _column(df, "older_seasons_best_year")
    buckets = [(ones, _column(df, "prev_season_best")),
               (ones * 2.0, _column(df, "two_seasons_ago_best")),
               ((season_year - older_year).clip(lower=3.0), _column(df, "older_seasons_best"))]
    lift = pd.Series(0.0, index=df.index)
    weight = pd.Series(0.0, index=df.index)
    from_season = pd.Series(np.nan, index=df.index)
    from_best = pd.Series(np.nan, index=df.index)
    for ago, old in buckets:
        w = pd.Series(fade_weight(ago, last_season, per_year), index=df.index)
        gain = (w * (old - now).clip(lower=0.0)).where(ago.notna() & old.notna(), 0.0).fillna(0.0)
        take = gain > lift
        lift, weight = lift.where(~take, gain), weight.where(~take, w)
        from_season = from_season.where(~take, season_year - ago)
        from_best = from_best.where(~take, old)
    room = ((now.max() - now).clip(lower=0.0) * cap_share) if len(now) else lift
    parts = pd.DataFrame({"seasonLevel": now, "lift": lift.clip(upper=room), "fromSeason": from_season,
                          "oldBest": from_best, "weight": weight, "capped": lift > room}, index=df.index)
    return now + parts["lift"], parts


def old_mark_reasons(rows, cutoff, model):
    """How each athlete's old marks counted, in the order of `rows`, for the line
    the page shows them: [{personalBest, fromSeason, percent}], or None for a
    model that does not read old marks this way.

    `percent` is the share of the distance between the old mark and this season
    that actually counted, after the cap, so the page never claims more than the
    model used. `fromSeason` is None for an athlete read on this season alone,
    and `personalBest` says their season best is at or above anything earlier."""
    fade = fade_of(model)
    features = (model or {}).get("features") or []
    if fade is None or not any(name in FADE_GAP_FEATURES for name in features):
        return None
    scored = rows[pd.notna(rows["sb_score"])]
    if scored.empty:
        return []
    _, parts = faded_level(scored, cutoff, **fade)
    career = _column(scored, "career_best")
    season = pd.to_numeric(scored["sb_score"], errors="coerce")
    out = []
    for i in scored.index:
        lift, old = float(parts["lift"].loc[i]), parts["oldBest"].loc[i]
        level = float(parts["seasonLevel"].loc[i])
        counted = lift > 0.05 and pd.notna(old) and float(old) > level
        out.append({
            "personalBest": bool(pd.isna(career.loc[i]) or season.loc[i] >= career.loc[i]),
            "fromSeason": int(parts["fromSeason"].loc[i]) if counted else None,
            "percent": int(round(100 * lift / (float(old) - level))) if counted else 0,
        })
    return out


# ---- the rule test, on made-up finals -----------------------------------------

# The user's rule has two halves, and a setting has to keep both: old marks
# still count, and they never count for as much as recent ones. A test of the
# second half alone would be passed by a setting that ignores old marks
# altogether, which is not what they asked for, so both are checked here.
RULE_YEAR = 2026
RULE_CUTOFF = f"{RULE_YEAR}-09-23"
# The made-up field: A leads on this season's marks, B is 20 points behind with
# an older mark, D is 5 points ahead of B with nothing older to call on.
RULE_A, RULE_B, RULE_D = 0, 1, 2
RULE_SEASON = [1250.0, 1230.0, 1235.0, 1220.0, 1210.0, 1190.0]
RULE_WEIGHT_SETS = 20


def rule_field(old_best, ago=2, races=5, season=None):
    """One made-up final for the rule test.

    A sets a personal best this season and leads the field on 1250. B is on 1230
    but holds `old_best` from `ago` seasons back, in a season of `races` races. D
    is on 1235 with nothing older above their season. Everything else is equal:
    same date, same age, same recent form, no big-meet podiums, nobody has met
    anybody. Only the old mark and B's race count differ, which is what lets the
    test say the old mark alone did it."""
    season = list(season or RULE_SEASON)
    n = len(season)
    prev = [1240.0, 1220.0, 1225.0] + [s - 20 for s in season[3:]]
    two = [1200.0, 1200.0, 1200.0] + [s - 40 for s in season[3:]]
    older = [None, 1180.0, None] + [None] * (n - 3)
    older_year = [None, RULE_YEAR - 4, None] + [None] * (n - 3)
    if ago == 1:
        prev[RULE_B] = old_best
    elif ago == 2:
        two[RULE_B] = old_best
    else:
        older[RULE_B], older_year[RULE_B] = old_best, RULE_YEAR - ago
    career = [1240.0, max(old_best, 1220.0), 1225.0] + [s - 10 for s in season[3:]]
    return pd.DataFrame({
        "athlete_name": ["A breaks records", "B has an old mark", "D has no old mark"]
                        + [f"C{i}" for i in range(n - 3)],
        "sb_score": season, "form_score": season, "recent_score": season,
        "form_spread": [10.0] * n, "races": [5, races] + [5] * (n - 2),
        "big_podiums": [0] * n, "h2h_top": [0.0] * n,
        "sb_date": [pd.Timestamp(f"{RULE_YEAR}-06-01")] * n,
        "sb_prior_season": [0] * n, "dob": [pd.Timestamp("1998-01-01")] * n,
        "career_best": career, "prev_season_best": prev,
        "two_seasons_ago_best": two, "older_seasons_best": older, "older_seasons_best_year": older_year,
    })


# `counts` marks the cases where a big, recent-enough old mark must still be
# worth something: B has to pass D, who is 5 points ahead this season. Without
# them a setting could pass by never counting an old mark at all.
RULE_CASES = (
    [{"label": f"{old:.0f} two seasons ago", "old_best": float(old), "ago": 2, "counts": old >= 1330}
     for old in (1240, 1260, 1280, 1300, 1330, 1400, 1500)]
    + [{"label": f"1330 from {ago} season{'s' if ago > 1 else ''} ago", "old_best": 1330.0, "ago": ago,
        "counts": ago <= 2} for ago in (1, 3, 5, 8)]
    + [{"label": "1330 two seasons ago, one race this season", "old_best": 1330.0, "ago": 2, "races": 1},
       {"label": "1330 two seasons ago, two races this season", "old_best": 1330.0, "ago": 2, "races": 2},
       {"label": "an old best level with the leader", "old_best": 1250.0, "ago": 2},
       {"label": "an old best far above the whole field", "old_best": 1600.0, "ago": 1}]
)


def _bounded_weights(features, bounds, seed):
    """Random weights that respect `bounds`, so the rule is tested against every
    fit the bounds allow rather than against one fitted model."""
    rng = np.random.default_rng(seed)
    weights = []
    for name, value in zip(features, rng.normal(size=len(features))):
        low, high = bounds.get(name, (None, None))
        if high is not None:
            value = -abs(value)
        elif low is not None:
            value = abs(value)
        weights.append(float(value))
    return {"features": list(features), "mean": [0.0] * len(features), "std": [1.0] * len(features),
            "weights": weights}


def _rule_models(model, fade):
    """The models the rule is checked under: the one given, each group's own when
    it was fitted by group, or else every bounds-respecting fit, sampled."""
    if model is None:
        return [_bounded_weights(FADE_FEATURES, FADE_BOUNDS, seed) for seed in range(RULE_WEIGHT_SETS)], fade
    fade = fade if fade is not None else fade_of(model)
    if "byGroup" in model:
        return list(model["byGroup"].values()), fade
    return [model], fade


def check_old_marks_rule(fade=None, model=None, cases=None):
    """Whether the user's rule holds on the made-up finals (RULE_CASES), with the
    old-marks settings `fade`.

    Both halves of what they asked for, on every case:
      * the athlete leading on this season's marks is never passed by an older,
        bigger mark, whatever that mark is and however few races their rival has
        run this season;
      * an old mark that is big and recent enough still counts, so B passes the
        athlete 5 points ahead of them on this season alone.
    Checked on the level faded_level() reads and on the win chances themselves,
    under `model` when one is given and otherwise under RULE_WEIGHT_SETS random
    fits that respect FADE_BOUNDS: a setting passes only if no fit the rule
    allows could break it. This is the gate a tuned setting has to pass before it
    can be kept, whatever it scores."""
    cases = list(cases or RULE_CASES)
    models, fade = _rule_models(model, fade)
    rows_out, passed = [], True
    for case in cases:
        rows = rule_field(case["old_best"], case.get("ago", 2), case.get("races", 5))
        level, parts = faded_level(rows, RULE_CUTOFF, **(fade or {}))
        leader = float(level.iloc[RULE_A] - level.iloc[RULE_B])
        over_d = float(level.iloc[RULE_B] - level.iloc[RULE_D])
        win_leader, win_over_d = [], []
        for m in models:
            X = field_features(rows, RULE_CUTOFF, m.get("features") or FADE_FEATURES, fade).to_numpy()
            win = win_chances(utilities(m, X))
            win_leader.append(float(win[RULE_A] - win[RULE_B]))
            win_over_d.append(float(win[RULE_B] - win[RULE_D]))
        held = leader > 0 and min(win_leader) > 0
        counted = (over_d > 0 and min(win_over_d) > 0) if case.get("counts") else None
        passed = passed and held and (counted is not False)
        rows_out.append({
            "label": case["label"], "oldBest": case["old_best"], "ago": case.get("ago", 2),
            "races": case.get("races", 5), "leaderLevel": round(float(level.iloc[RULE_A]), 1),
            "rivalLevel": round(float(level.iloc[RULE_B]), 1), "lift": round(float(parts["lift"].iloc[RULE_B]), 1),
            "fromSeason": None if pd.isna(parts["fromSeason"].iloc[RULE_B]) else int(parts["fromSeason"].iloc[RULE_B]),
            "weight": round(float(parts["weight"].iloc[RULE_B]), 3),
            "capped": bool(parts["capped"].iloc[RULE_B]),
            "worstWinGap": round(min(win_leader), 4), "leaderStaysAhead": bool(held),
            "oldMarkCounts": None if counted is None else bool(counted),
            "worstCountsGap": None if counted is None else round(min(win_over_d), 4)})
    return {"passed": bool(passed), "fade": dict(fade or {}), "cases": rows_out,
            "fits": len(models),
            "fitsAre": "model given" if model is not None else "random fits inside FADE_BOUNDS"}


def print_rule_test(report):
    print(f"  settings: {report['fade'] or 'the module defaults'}; checked under {report['fits']} "
          f"{report['fitsAre']}")
    print(f"\n  {'case':<44} {'leader':>7} {'rival':>7} {'lift':>6} {'from':>6} {'weight':>7} "
          f"{'ahead':>6} {'counts':>7}")
    for c in report["cases"]:
        counts = "-" if c["oldMarkCounts"] is None else ("yes" if c["oldMarkCounts"] else "NO")
        print(f"  {c['label']:<44} {c['leaderLevel']:>7.1f} {c['rivalLevel']:>7.1f} {c['lift']:>6.1f} "
              f"{c['fromSeason'] or '':>6} {c['weight']:>7.3f} "
              f"{'yes' if c['leaderStaysAhead'] else 'NO':>6} {counts:>7}")
    print(f"\n  the rule held on every case: {report['passed']}")


# ---- the model ----------------------------------------------------------------

def podium_chances(utilities):
    """P(top three) for each athlete under Plackett-Luce, exactly.

    First: e_i / S. Second: sum over who was first. Third: sum over who was
    first and second. Adds up to 3 over any field of four or more; everyone in a
    field of three or fewer is certain of a place."""
    u = np.asarray(utilities, dtype=float)
    n = len(u)
    if n <= 3:
        return np.ones(n)
    e = np.exp(u - u.max())
    total = e.sum()
    p1 = e / total
    first_second = p1[:, None] * e[None, :] / (total - e)[:, None]
    np.fill_diagonal(first_second, 0.0)
    p2 = first_second.sum(axis=0)
    remaining = total - e[:, None] - e[None, :]
    third = first_second[:, :, None] * e[None, None, :] / remaining[:, :, None]
    idx = np.arange(n)
    third[idx, :, idx] = 0.0   # third is not the one who was first
    third[:, idx, idx] = 0.0   # nor the one who was second
    p3 = third.sum(axis=(0, 1))
    return p1 + p2 + p3


def _pack(finals, mean, std):
    width = max(len(f["names"]) for f in finals)
    X = np.zeros((len(finals), width, finals[0]["X"].shape[1]))
    mask = np.zeros((len(finals), width), dtype=bool)
    top = np.zeros((len(finals), 3), dtype=int)
    for i, f in enumerate(finals):
        n = len(f["names"])
        X[i, :n] = (f["X"] - mean) / std
        mask[i, :n] = True
        top[i] = f["top_idx"]
    return X, mask, top


def _loss_and_grad(w, X, mask, top, l2):
    """Negative log-likelihood of each final's observed top three, and its
    gradient. Unavailable slots (padding, and athletes already placed) are
    -inf in the softmax."""
    finals = X.shape[0]
    rows = np.arange(finals)
    u = np.einsum("fnk,k->fn", X, w)
    available = mask.copy()
    loss, grad = 0.0, np.zeros_like(w)
    for step in range(3):
        picked = top[:, step]
        logits = np.where(available, u, -np.inf)
        lse = logsumexp(logits, axis=1)
        probs = np.exp(logits - lse[:, None])
        loss += float(np.sum(lse - u[rows, picked]))
        grad += np.einsum("fn,fnk->k", probs, X) - X[rows, picked].sum(axis=0)
        available[rows, picked] = False
    loss = loss / finals + l2 * float(w @ w)
    grad = grad / finals + 2 * l2 * w
    return loss, grad


def fit(finals, l2=L2, bounds=None):
    """Fit on every trainable final whose top three all have a season score.
    `bounds` is {feature: (low, high)} on that feature's weight, for the signs
    the recency family's rule fixes (RECENCY_BOUNDS). Features are standardised
    by a positive scale, so a sign held on the fitted weight holds on the raw one."""
    usable = [f for f in finals if f["top_idx"] is not None and f.get("trainable", True)]
    if not usable:
        raise ValueError("no trainable final with a scored top three to fit on")
    stacked = np.vstack([f["X"] for f in usable])
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0)
    std[std == 0] = 1.0
    X, mask, top = _pack(usable, mean, std)
    names = list(usable[0].get("features", FEATURES))
    box = [(bounds or {}).get(name, (None, None)) for name in names] if bounds else None
    result = minimize(_loss_and_grad, np.zeros(stacked.shape[1]), args=(X, mask, top, l2),
                      jac=True, method="L-BFGS-B", bounds=box)
    return {"features": names, "mean": mean.tolist(), "std": std.tolist(),
            "weights": result.x.tolist(), "l2": l2, "finals": len(usable),
            "converged": bool(result.success)}


def utilities(model, X):
    """Each athlete's score under the model for one field's feature matrix: the
    log of their strength in Plackett-Luce, comparable only within the field."""
    mean, std, w = (np.asarray(model[k], dtype=float) for k in ("mean", "std", "weights"))
    return ((np.asarray(X, dtype=float) - mean) / std) @ w


def win_chances(utilities):
    """P(first) for each athlete under Plackett-Luce, adding up to 1.

    Shown beside the podium chance since 2026-09-15. Three close athletes can
    each be near-certain of a medal while only one of them wins, so a podium
    chance alone cannot say whether the leader is clear of the rest."""
    u = np.asarray(utilities, dtype=float)
    if len(u) == 0:
        return u
    e = np.exp(u - u.max())
    return e / e.sum()


def predict(model, X):
    """Podium chances for one field's feature matrix."""
    return podium_chances(utilities(model, X))


# ---- finals as the model sees them --------------------------------------------

def load_scored_finals(path=None):
    df = pd.read_csv(path or fd.FINALS_PATH, low_memory=False)
    for col in ("cutoff", "sb_date", "dob"):
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def build_finals(scored, features=FEATURES, fade=None):
    """One dict per final with at least MIN_SCORED scored finalists.

    `podium` holds every medallist's name, scored or not: an unscored medallist
    is a place neither the model nor points could have named, and counting it
    as possible keeps both honest. A shared bronze puts both names on it (23 of
    the 900 championship finals have a tie on the podium). `top_idx` is the scored rows of first,
    second and third, or None when any of the three is unscored.

    `match` is the share of the whole competition's finalists with a score, and
    `trainable` whether it clears field_data.MIN_MATCH. A final below the gate
    is still scored in the backtest, where an athlete we could not find is a
    miss for the model and for points alike. It is only kept out of fitting."""
    has_score = scored["sb_score"].notna()
    rates = has_score.groupby([scored["competition"], scored["year"]]).mean()
    finals = []
    for (competition, year, discipline), g in scored.groupby(["competition", "year", "discipline"]):
        placed = g[g["place"].between(1, 3)].sort_values("place")
        if len(placed) < 3:
            continue
        # The rows arrive in finishing order. Sorted by name instead, so that
        # when two athletes share a score, neither the points ranking nor the
        # model breaks the tie with the result it is trying to predict.
        with_score = (g[g["sb_score"].notna()]
                      .assign(_by_name=lambda d: d["athlete_name"].map(fd.field_key))
                      .sort_values("_by_name", kind="stable").drop(columns="_by_name")
                      .reset_index(drop=True))
        if len(with_score) < MIN_SCORED:
            continue
        X = field_features(with_score, g["cutoff"].iloc[0], features, fade)
        names = with_score["athlete_name"].tolist()
        order = []
        for place in (1, 2, 3):
            hit = with_score.index[with_score["place"] == place]
            order.append(int(hit[0]) if len(hit) else None)
        match = float(rates.loc[(competition, year)])
        finals.append({
            "competition": competition, "year": int(year), "discipline": discipline,
            "group": group_of(discipline), "tier": g["tier"].iloc[0],
            "names": names, "scores": with_score["sb_score"].to_numpy(dtype=float),
            "X": X.to_numpy(), "features": list(features), "podium": set(placed["athlete_name"]),
            "winners": set(placed.loc[placed["place"] == 1, "athlete_name"]),
            "top_idx": order if None not in order else None,
            "match": match, "trainable": match >= fd.MIN_MATCH,
        })
    return finals


def _top3(names, values):
    order = np.argsort(-np.asarray(values, dtype=float), kind="stable")
    return {names[i] for i in order[:3]}


def backtest(finals, l2=L2, shuffle_seed=None, years=None, by_group=False, bounds=None):
    """Walk forward over `years`, TEST_YEARS unless given. Returns (per-final
    results, per-athlete chances). With `shuffle_seed`, every final's feature
    rows are shuffled, for training and testing alike: the control, which
    should find nothing. With `by_group`, each event group in GROUPS is fitted
    on its own finals only."""
    rng = np.random.default_rng(shuffle_seed)
    if shuffle_seed is not None:
        shuffled = []
        for f in finals:
            order = rng.permutation(len(f["names"]))
            shuffled.append({**f, "X": f["X"][order]})
        finals = shuffled

    results, athletes = [], []
    for year in TEST_YEARS if years is None else years:
        test = [f for f in finals if f["year"] == year]
        earlier = [f for f in finals if f["year"] < year]
        pools = ({group: [f for f in earlier if f["group"] == group] for group in GROUPS}
                 if by_group else {None: earlier})
        for group, train in pools.items():
            chunk = [f for f in test if group is None or f["group"] == group]
            if not chunk or not any(f["top_idx"] is not None and f.get("trainable", True) for f in train):
                continue
            model = fit(train, l2, bounds)
            for f in chunk:
                u = utilities(model, f["X"])
                chances = podium_chances(u)
                model_pick = _top3(f["names"], chances)
                points_pick = _top3(f["names"], f["scores"])
                winners = f.get("winners", set())
                results.append({
                    "competition": f["competition"], "year": year, "discipline": f["discipline"],
                    "group": f["group"], "tier": f["tier"],
                    "model_hits": len(model_pick & f["podium"]), "points_hits": len(points_pick & f["podium"]),
                    "model_winner": f["names"][int(np.argmax(chances))] in winners,
                    "points_winner": f["names"][int(np.argmax(f["scores"]))] in winners,
                    # For compare(): how well the model read the order of the top
                    # three, and how sure it was of its favourite.
                    "ll": top3_log_likelihood(u, f["top_idx"]) if f.get("top_idx") is not None else np.nan,
                    "favourite_win": float(win_chances(u).max()),
                })
                for name, chance in zip(f["names"], chances):
                    athletes.append({"chance": float(chance), "medal": name in f["podium"]})
    return pd.DataFrame(results), pd.DataFrame(athletes)


def bootstrap_lower(d, n_boot=BOOTSTRAP, quantile=LOWER_QUANTILE, seed=0):
    d = np.asarray(d, dtype=float)
    if len(d) == 0:
        return None
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    return float(np.quantile(means, quantile))


def ship_decisions(results, control):
    """The pre-registered rule, group by group. See the module docstring."""
    out = {}
    for group in GROUPS:
        g = results[results["group"] == group]
        c = control[control["group"] == group]
        d = (g["model_hits"] - g["points_hits"]).to_numpy()
        asia = g[g["tier"] == "asia"]
        asia_d = (asia["model_hits"] - asia["points_hits"]).to_numpy()
        control_lower = bootstrap_lower((c["model_hits"] - c["points_hits"]).to_numpy())
        lower = bootstrap_lower(d)
        asia_mean = float(asia_d.mean()) if len(asia_d) else None
        passed = bool(
            lower is not None and lower > 0
            and asia_mean is not None and asia_mean >= 0
            and not (control_lower is not None and control_lower > 0)
        )
        out[group] = {
            "finals": int(len(g)), "possible": int(3 * len(g)),
            "modelHits": int(g["model_hits"].sum()), "pointsHits": int(g["points_hits"].sum()),
            "meanDiff": float(d.mean()) if len(d) else None, "lower90": lower,
            "asiaFinals": int(len(asia)), "asiaModelHits": int(asia["model_hits"].sum()),
            "asiaPointsHits": int(asia["points_hits"].sum()), "asiaMeanDiff": asia_mean,
            "controlModelHits": int(c["model_hits"].sum()), "controlLower90": control_lower,
            "passed": passed,
        }
    return out


# ---- comparing a new feature set with today's ---------------------------------

def top3_log_likelihood(u, top_idx):
    """The log-probability of one final's observed first, second and third
    under Plackett-Luce: per final, the quantity fit() maximises."""
    u = np.asarray(u, dtype=float)
    available = np.ones(len(u), dtype=bool)
    total = 0.0
    for i in top_idx:
        total += float(u[i] - logsumexp(u[available]))
        available[i] = False
    return total


def shuffle_columns(finals, columns, seed):
    """The finals with `columns` of each feature matrix permuted within the
    final, together, so those columns stop belonging to their athletes while
    every other column stays put. compare()'s control."""
    rng = np.random.default_rng(seed)
    out = []
    for f in finals:
        X = f["X"].copy()
        X[:, columns] = X[rng.permutation(len(X))][:, columns]
        out.append({**f, "X": X})
    return out


def compare_decision(base, cand, controls):
    """The rule for the race-by-race features, written down on 2026-09-15 before
    compare() first ran. The candidate ships only if all four hold:
      llLower90     the one-sided 90% bootstrap lower bound of its mean gain in
                    top-three log-likelihood per held-out final is above 0;
      hitsNotWorse  it names at least as many medallists per final on average;
      asiaNotWorse  its mean log-likelihood gain on the Asian finals is at least 0;
      controlsFail  no control (the candidate with its new columns shuffled
                    within each final) meets the first condition.
    Log-likelihood rewards confidence in a clear leader and doubt about a close
    field, which a count of medallists cannot see. `base`, `cand` and each
    control are backtest() results for the same finals in the same order."""
    read = base["ll"].notna() & cand["ll"].notna()
    gain = cand["ll"] - base["ll"]
    d_ll = gain[read].to_numpy()
    d_hits = (cand["model_hits"] - base["model_hits"]).to_numpy()
    asia = read & (base["tier"] == "asia")
    d_asia = gain[asia].to_numpy()
    lower = bootstrap_lower(d_ll)
    control_lower = [bootstrap_lower((c["ll"] - base["ll"])[read].to_numpy()) for c in controls]
    conditions = {
        "llLower90": bool(lower is not None and lower > 0),
        "hitsNotWorse": bool(len(d_hits) and d_hits.mean() >= 0),
        "asiaNotWorse": bool(len(d_asia) and d_asia.mean() >= 0),
        "controlsFail": not any(c is not None and c > 0 for c in control_lower),
    }
    return {
        "finals": int(len(base)), "llFinals": int(read.sum()), "asiaFinals": int(asia.sum()),
        "meanLlGain": float(d_ll.mean()) if len(d_ll) else None, "llLower90": lower,
        "meanHitsDiff": float(d_hits.mean()) if len(d_hits) else None,
        "asiaMeanLlGain": float(d_asia.mean()) if len(d_asia) else None,
        "controlLower90": control_lower,
        "conditions": conditions, "ships": all(conditions.values()),
    }


def _favourites(results):
    """How often the favourite won, by how sure the model was of them: clear
    favourites and close fields side by side."""
    out = []
    for lo, hi in ((0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.01)):
        g = results[(results["favourite_win"] >= lo) & (results["favourite_win"] < hi)]
        out.append({"from": lo, "to": min(hi, 1.0), "finals": int(len(g)),
                    "predicted": round(float(g["favourite_win"].mean()), 3) if len(g) else None,
                    "observed": round(float(g["model_winner"].mean()), 3) if len(g) else None})
    return out


def _without(*names):
    return [name for name in FEATURES_V2 if name not in names]


# The candidates tried on DEV_YEARS by --experiments, each against today's
# model. Declared here so the record says what was tried. A new idea is added
# to this list and tried on DEV_YEARS; nothing is tried on HOLDOUT_YEARS.
EXPERIMENTS = {
    "today": {"features": FEATURES},
    "today_l2_strong": {"features": FEATURES, "l2": 0.1},
    "today_by_group": {"features": FEATURES, "by_group": True},
    "v2": {"features": FEATURES_V2},
    "v2_l2_strong": {"features": FEATURES_V2, "l2": 0.1},
    "v2_l2_weak": {"features": FEATURES_V2, "l2": 0.001},
    "v2_by_group": {"features": FEATURES_V2, "by_group": True},
    "v2_recent_28d": {"features": FEATURES_V2, "race": {"recent_days": 28}},
    "v2_recent_70d": {"features": FEATURES_V2, "race": {"recent_days": 70}},
    "v2_form_best_2": {"features": FEATURES_V2, "race": {"form_marks": 2}},
    "v2_form_best_5": {"features": FEATURES_V2, "race": {"form_marks": 5}},
    "v2_big_with_A": {"features": FEATURES_V2, "race": {"big_categories": ["OW", "DF", "GW", "GL", "A"]}},
    "v2_without_h2h": {"features": _without("h2h_top")},
    "v2_without_big_podiums": {"features": _without("big_podiums")},
    "v2_without_form": {"features": _without("form_gap", "breakout_backed")},
    "v2_without_recent": {"features": _without("recent_delta", "no_recent")},
    "v2_without_history_blend": {"features": _without("pb_gap_thin")},
    # The user's rule, new marks over old (strength, RECENCY_BOUNDS). They compete
    # with the rest on accuracy alone: the user wants the most accurate call, and
    # the rule counts in a candidate's favour only when it is also that.
    "recency": {"features": RECENCY_FEATURES, "bounds": RECENCY_BOUNDS, "newOverOld": True},
    "recency_l2_strong": {"features": RECENCY_FEATURES, "bounds": RECENCY_BOUNDS, "l2": 0.1, "newOverOld": True},
    "recency_by_group": {"features": RECENCY_FEATURES, "bounds": RECENCY_BOUNDS, "by_group": True,
                         "newOverOld": True},
    "recency_recent_70d": {"features": RECENCY_FEATURES, "bounds": RECENCY_BOUNDS,
                           "race": {"recent_days": 70}, "newOverOld": True},
    "recency_without_h2h": {"features": [f for f in RECENCY_FEATURES if f != "h2h_top"],
                            "bounds": RECENCY_BOUNDS, "newOverOld": True},
    "recency_without_big_podiums": {"features": [f for f in RECENCY_FEATURES if f != "big_podiums"],
                                    "bounds": RECENCY_BOUNDS, "newOverOld": True},
    # The three ways old marks can count, tested on every past season by
    # run_all_seasons (2026-09-16): v2_form_best_5 as served, recent over old,
    # and this season only, all reading races the same way.
    "recency_form_best_5": {"features": RECENCY_FEATURES, "bounds": RECENCY_BOUNDS,
                            "race": {"form_marks": 5}, "newOverOld": True},
    "season_only": {"features": SEASON_ONLY_FEATURES, "race": {"form_marks": 5}},
    # The user's rule for old marks (2026-09-16), which they chose over raw
    # accuracy: this season in full, an older best only where it still stands
    # above it, faded by its age, and never by enough to pass the athlete ahead
    # this season. The grid over the three settings is tuned on past seasons;
    # the rule holds whichever setting it picks.
    "old_marks": {"features": FADE_FEATURES, "bounds": FADE_BOUNDS, "race": {"form_marks": 5},
                  "fade": {"last_season": FADE_LAST_SEASON, "per_year": FADE_PER_YEAR,
                           "cap_share": FADE_CAP_SHARE},
                  "oldMarks": True},
}
RESULT_KEY = ["year", "competition", "discipline"]
# Every season with at least three seasons of finals before it to learn from
# (2009-2011); 2020 held none. Each is called only from the seasons before it.
ALL_SEASON_YEARS = [year for year in range(2012, 2027) if year != 2020]
# Fixed on 2026-09-16 before any of them ran, today's model first: it is the
# one the others are lined up against.
ALL_SEASON_CANDIDATES = ["v2_form_best_5", "recency_form_best_5", "season_only"]


def spec_scored(scored, spec, races=None):
    """`scored` with its race columns read again when an experiment reads races
    its own way (spec["race"], passed to field_data.race_summary), from `races`,
    which is field_data.load_seasons(RACES_DIR) unless given."""
    if not spec.get("race"):
        return scored
    return fd.attach_races(scored, fd.load_seasons(fd.RACES_DIR) if races is None else races, **spec["race"])


def fit_spec(finals, spec):
    """An experiment's model on `finals`: one model, or one per event group."""
    l2, bounds = spec.get("l2", L2), spec.get("bounds")
    if not spec.get("by_group"):
        return fit(finals, l2, bounds)
    return {"features": list(spec["features"]), "byGroup": {
        group: fit([f for f in finals if f["group"] == group], l2, bounds) for group in GROUPS
        if any(f["group"] == group and f["top_idx"] is not None and f.get("trainable", True) for f in finals)}}


def model_for(model, group):
    """The model that scores an event in `group`: that group's own when the
    model was fitted by group."""
    return model["byGroup"][group] if "byGroup" in model else model


def _weights(model):
    if "byGroup" in model:
        return {group: dict(zip(m["features"], m["weights"])) for group, m in model["byGroup"].items()}
    return dict(zip(model["features"], model["weights"]))


def choose_experiment(rows):
    """The candidate for HOLDOUT_YEARS: the largest mean top-three
    log-likelihood gain on DEV_YEARS among the experiments that name at least as
    many medallists per final as today's model. None when none gains.

    Accuracy alone, as the user asked on 2026-09-15 ("what I want is for the
    accuracy to be the best it can be"), after the experiments had run and
    before any locked-year result was read. It is the rule first committed,
    before any experiment ran. For a short while before that message the choice
    was limited to the newOverOld family, and the locked-year run that produced
    finished before the message arrived; it was set aside unopened (HANDOFF.md)."""
    eligible = [r for r in rows if r["name"] != "today"
                and r.get("meanLlGain") is not None and r["meanLlGain"] > 0
                and r.get("meanHitsDiff") is not None and r["meanHitsDiff"] >= 0]
    return max(eligible, key=lambda r: r["meanLlGain"])["name"] if eligible else None


def compare(scored, candidate="v2", baseline="today", years=DEV_YEARS, controls=CONTROLS, races=None):
    """A candidate from EXPERIMENTS against a baseline on the same held-out
    finals: DEV_YEARS while trying candidates, HOLDOUT_YEARS once for the chosen
    one. Returns (the report, the candidate fitted on every final).

    The controls shuffle the candidate's columns that the baseline lacks, or
    all of them when it adds none. With controls=0 they are skipped and the
    report carries no verdict, as for a first look at an experiment."""
    specs = {"baseline": EXPERIMENTS[baseline], "candidate": EXPERIMENTS[candidate]}
    built = {side: build_finals(spec_scored(scored, spec, races), spec["features"], spec.get("fade"))
             for side, spec in specs.items()}

    def run(side, finals):
        spec = specs[side]
        return backtest(finals, spec.get("l2", L2), years=years, by_group=spec.get("by_group", False),
                        bounds=spec.get("bounds"))

    (base, base_athletes), (cand, cand_athletes) = run("baseline", built["baseline"]), run("candidate", built["candidate"])
    features = specs["candidate"]["features"]
    new = [i for i, name in enumerate(features) if name not in specs["baseline"]["features"]]
    runs = [run("candidate", shuffle_columns(built["candidate"], new or list(range(len(features))), seed))[0]
            for seed in range(controls)]
    # Lined up final by final, since a model fitted by group scores the finals in another order.
    shared = base[RESULT_KEY].merge(cand[RESULT_KEY], on=RESULT_KEY)

    def aligned(results):
        return results.merge(shared, on=RESULT_KEY).sort_values(RESULT_KEY).reset_index(drop=True)

    base, cand, runs = aligned(base), aligned(cand), [aligned(r) for r in runs]
    decision = compare_decision(base, cand, runs)
    if not controls:
        decision["conditions"].pop("controlsFail")
        decision["ships"] = None
    model = fit_spec(built["candidate"], specs["candidate"])
    has_races = (scored["has_races"].fillna(False).astype(bool)[scored["sb_score"].notna()]
                 if "has_races" in scored.columns else None)

    def pair(b, c):
        return {"baseline": _block(b), "candidate": _block(c)}

    report = {
        "builtAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "testYears": list(years), "baselineName": baseline, "candidateName": candidate,
        "features": list(features), "newFeatures": [features[i] for i in new],
        "spec": json.loads(json.dumps(specs["candidate"])),
        "bigCategories": sorted(fd.BIG_CATEGORIES), "recentDays": fd.RECENT_DAYS, "thinSeason": THIN_SEASON,
        "overall": pair(base, cand),
        "byTier": {t: pair(base[base["tier"] == t], cand[cand["tier"] == t]) for t in sorted(base["tier"].unique())},
        "byGroup": {g: pair(base[base["group"] == g], cand[cand["group"] == g]) for g in GROUPS},
        "byYear": {int(y): pair(base[base["year"] == y], cand[cand["year"] == y])
                   for y in sorted(base["year"].unique())},
        "decision": decision,
        "weights": _weights(model),
        "calibration": {"baseline": _calibration(base_athletes), "candidate": _calibration(cand_athletes)},
        "favourites": {"baseline": _favourites(base), "candidate": _favourites(cand)},
        "raceCoverage": None if has_races is None else round(float(has_races.mean()), 3),
    }
    return report, model


def _signed(x, places=4):
    return "n/a" if x is None else f"{x:+.{places}f}"


def print_compare(report):
    d, o = report["decision"], report["overall"]
    b, c = o["baseline"], o["candidate"]
    base, cand = report["baselineName"], report["candidateName"]
    print(f"\n  {b['finals']} held-out finals, {report['testYears'][0]}-{report['testYears'][-1]}: "
          f"{base} {b['model']}%, {cand} {c['model']}% (points {b['points']}%)")
    print(f"  winners named first: {base} {b.get('modelWinners')}, {cand} {c.get('modelWinners')}")
    for label, table in (("tier", report["byTier"]), ("group", report["byGroup"])):
        for name, p in table.items():
            print(f"    {label} {name:<12} {p['baseline']['finals']:>4} finals  {base} {p['baseline']['model']}%  "
                  f"{cand} {p['candidate']['model']}%")
    print(f"\n  top-three log-likelihood gain per final: mean {_signed(d['meanLlGain'])}, 90% lower bound "
          f"{_signed(d['llLower90'])} ({d['llFinals']} finals); Asian finals mean {_signed(d['asiaMeanLlGain'])} "
          f"({d['asiaFinals']}); medallists per final {_signed(d['meanHitsDiff'], 3)}")
    if d["controlLower90"]:
        print("  shuffled controls' lower bounds: " + ", ".join(_signed(x) for x in d["controlLower90"]))
    for name, ok in d["conditions"].items():
        print(f"    {'pass' if ok else 'FAIL'}  {name}")
    verdict = {True: "SHIPS", False: "does not ship, so the call stays as it is",
               None: "no verdict: a first look on the tuning years"}
    print(f"  verdict: {verdict[d['ships']]}")
    weights = report["weights"]
    grouped = weights and all(isinstance(v, dict) for v in weights.values())
    for group, w in (weights.items() if grouped else [("all events", weights)]):
        print(f"\n  weights ({group}): " + ", ".join(f"{k} {_signed(v, 3)}" for k, v in w.items()))
    print(f"  the favourite's win chance -> how often they won ({base} | {cand}):")
    for fb, fc in zip(report["favourites"]["baseline"], report["favourites"]["candidate"]):
        print(f"    {fb['from']:.1f}-{fb['to']:.1f}  {fb['finals']:>4} finals {fb['predicted']} -> {fb['observed']}"
              f"  |  {fc['finals']:>4} finals {fc['predicted']} -> {fc['observed']}")
    if report.get("raceCoverage") is not None:
        print(f"  scored finalists with a race-by-race season on file: {report['raceCoverage']:.1%}")


def _pct(hits, possible):
    return round(100.0 * hits / possible, 1) if possible else None


def _block(df):
    """Podium places named, and winners named first, by the model and by points."""
    possible = 3 * len(df)
    out = {"finals": int(len(df)), "hits": int(df["model_hits"].sum()), "possible": int(possible),
           "model": _pct(df["model_hits"].sum(), possible),
           "points": _pct(df["points_hits"].sum(), possible)}
    if "model_winner" in df.columns:
        out["modelWinners"] = int(df["model_winner"].sum())
        out["pointsWinners"] = int(df["points_winner"].sum())
    return out


def _calibration(athletes):
    """Predicted against observed podium rate, in tenths of chance."""
    calibration = []
    if not athletes.empty:
        bins = np.linspace(0, 1, 11)
        athletes = athletes.assign(bin=np.clip(np.digitize(athletes["chance"], bins) - 1, 0, 9))
        for b, g in athletes.groupby("bin"):
            calibration.append({"from": round(bins[b], 1), "to": round(bins[b + 1], 1), "athletes": int(len(g)),
                                "predicted": round(float(g["chance"].mean()), 3),
                                "observed": round(float(g["medal"].mean()), 3)})
    return calibration


def summarise(results, athletes, control, decisions, coverage=None):
    return {
        "builtAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "testYears": TEST_YEARS, "minScored": MIN_SCORED,
        "overall": _block(results),
        "byTier": {tier: _block(g) for tier, g in results.groupby("tier")},
        "control": _block(control),
        "groups": decisions,
        "calibration": _calibration(athletes),
        "minMatch": fd.MIN_MATCH,
        "coverage": [] if coverage is None else json.loads(coverage.to_json(orient="records")),
    }


def print_report(report):
    o, c = report["overall"], report["control"]
    print(f"\n  {o['finals']} held-out finals, {report['testYears'][0]}-{report['testYears'][-1]}: "
          f"model {o['model']}%  points {o['points']}%   (shuffled control: model {c['model']}%)")
    print(f"  winners named first: model {o.get('modelWinners')}, points {o.get('pointsWinners')}")
    for tier, b in report["byTier"].items():
        print(f"    {tier:<12} {b['finals']:>4} finals  model {b['model']}%  points {b['points']}%  "
              f"winners {b.get('modelWinners')} / {b.get('pointsWinners')}")
    below = [row for row in report["coverage"] if not row["trained"]]
    if below:
        print(f"  Under {report['minMatch']:.0%} of finalists found, scored but not trained on: "
              + ", ".join(f"{fd.competition_label(row['competition'], row['year'])} ({row['scored']:.0%})"
                          for row in below))
    print("\n  group      finals  model  points  mean d  lower90  asia(m/p)   control  verdict")
    for group, g in report["groups"].items():
        print(f"  {group:<10} {g['finals']:>6} {_pct(g['modelHits'], g['possible']):>6} "
              f"{_pct(g['pointsHits'], g['possible']):>6}  {g['meanDiff'] if g['meanDiff'] is None else round(g['meanDiff'], 3):>6} "
              f"{g['lower90'] if g['lower90'] is None else round(g['lower90'], 3):>8}  "
              f"{g['asiaModelHits']:>3}/{g['asiaPointsHits']:<3} ({g['asiaFinals']})  "
              f"{_pct(g['controlModelHits'], g['possible']):>6}  {'SHIPS' if g['passed'] else 'stays on points'}")
    print("\n  calibration (predicted -> observed podium rate):")
    for b in report["calibration"]:
        print(f"    {b['from']:.1f}-{b['to']:.1f}  {b['athletes']:>5} athletes  {b['predicted']:.3f} -> {b['observed']:.3f}")


# ---- serving ------------------------------------------------------------------

def load_model(path=MODEL_PATH):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def spec_of(model):
    """The experiment a saved model was chosen as (EXPERIMENTS), which says how
    it reads races; empty for a model saved by --backtest."""
    return EXPERIMENTS.get((model or {}).get("experiment"), {})


def needs_races(model):
    """Whether a saved model reads each athlete's season race by race."""
    return bool(set((model or {}).get("features") or []) - set(FEATURES))


def fade_of(model):
    """The old-marks settings a saved model was fitted with, from its experiment,
    so an entrant's old mark fades on the call exactly as it did in the backtest.
    None for a model whose family does not read old marks that way."""
    return spec_of(model).get("fade")


def shipped(model, disc_key):
    """(whether this discipline's group passed the ship rule, its evidence)."""
    decision = ((model or {}).get("ship") or {}).get(group_of(disc_key))
    return bool(decision and decision.get("passed")), decision


SERVING_COLUMNS = ["athlete_name", "sb_score", "sb_date", "sb_prior_season", "dob",
                   "career_best", "prev_season_best", "mark", "mark_season", "wa_id"]
SERVING_COLUMNS += [c for c in fd.EARLIER_BEST_COLUMNS if c not in SERVING_COLUMNS]


def serving_rows(disc_key, athletes, season_path, cutoff, year, scores_for=fd.season_scores, extra=(),
                 races=None, race_how=None):
    """The rows field_features needs, for athletes about to compete, read the
    way field_data.attach_scores reads a past final.

    `athletes` is [{name, nat, birthDate?}]. The season best is the best row in
    `season_path` (this season's toplist or a championship's merged snapshot)
    dated before the cut-off. An athlete with none takes last season's best
    from their history, flagged sb_prior_season: training fell back the same
    way, so serving must. History is `scores_for(disc_key)` plus the `extra`
    (path, source) toplists, such as last season's Asian list, and gives the
    bests from earlier seasons too, by how long ago they came
    (field_data.earlier_bests). An athlete with no mark this season or last is
    left out. `mark` and `mark_season` say which mark was used, for the page.

    With `races`, {World Athletics id as text: this season race by race, from
    field_data.fetch_races}, the rows also carry field_data.RACE_COLUMNS, read
    through field_data.race_columns as training reads them, head-to-head
    included: what FEATURES_V2 needs. `wa_id` is the entrant's id, or the one on
    the toplist row their mark came from. `race_how` goes to
    field_data.race_summary, for a model chosen as an experiment that reads
    races its own way (EXPERIMENTS[...]["race"])."""
    season = fd._toplist_rows(season_path, "season")
    history = scores_for(disc_key)
    frames = [f for f in [history] + [fd._toplist_rows(path, source) for path, source in extra] if not f.empty]
    if frames:
        history = pd.concat(frames, ignore_index=True)
    cutoff = pd.Timestamp(cutoff)
    rows = []
    for a in athletes:
        key, nat = fd.field_key(a["name"]), str(a.get("nat") or "").strip()
        wid = int(a["waId"]) if str(a.get("waId") or "").isdigit() else None
        mine = fd.athlete_history(season, key, nat, wid)[0] if not season.empty else season
        mine = mine[mine["date"] < cutoff] if not mine.empty else mine
        past = fd.athlete_history(history, key, nat, wid)[0] if not history.empty else history
        last = past[past["year"] == year - 1] if not past.empty else past
        if not mine.empty:
            best, prior = mine.loc[mine["score"].idxmax()], 0
        elif not last.empty:
            best, prior = last.loc[last["score"].idxmax()], 1
        else:
            continue
        dob = best["dob"] if pd.notna(best["dob"]) else None
        if dob is None and not past.empty and past["dob"].notna().any():
            dob = past["dob"].dropna().iloc[0]
        if dob is None:
            dob = fd.parse_date(a.get("birthDate"))
        if wid is None and "wa_id" in best.index and pd.notna(best["wa_id"]):
            wid = int(best["wa_id"])
        rows.append({
            "athlete_name": a["name"], "sb_score": float(best["score"]), "sb_date": best["date"],
            "sb_prior_season": prior, "dob": dob,
            "mark": best["mark"], "mark_season": int(best["year"]), "wa_id": wid,
            **fd.earlier_bests(past, year),
        })
    out = pd.DataFrame(rows, columns=SERVING_COLUMNS)
    if races is None or out.empty:
        return out
    field = out.assign(discipline=disc_key, cutoff=cutoff)
    return out.join(fd.race_columns(
        field, lambda row: races.get(str(int(row["wa_id"]))) if pd.notna(row["wa_id"]) else None,
        **(race_how or {})))


def field_chances(model, rows, cutoff, disc_key=None):
    """{athlete name: (podium chance, win chance)} for one field, read with the
    model's own features, and with the model for the event's group when it was
    fitted by group. Empty for a field of fewer than three scored athletes,
    where there is no podium to call."""
    rows = rows[rows["sb_score"].notna()].reset_index(drop=True)
    if len(rows) < 3:
        return {}
    features = field_features(rows, cutoff, model.get("features") or FEATURES, fade_of(model))
    u = utilities(model_for(model, group_of(disc_key)), features.to_numpy())
    return {name: (float(p), float(w))
            for name, p, w in zip(rows["athlete_name"], podium_chances(u), win_chances(u))}


def score_field(model, rows, cutoff):
    """{athlete name: podium chance} for one field (see field_chances)."""
    return {name: podium for name, (podium, _) in field_chances(model, rows, cutoff).items()}


def _scored_with_races():
    scored = load_scored_finals()
    missing = [c for c in fd.RACE_COLUMNS if c not in scored.columns]
    if missing:
        print(f"  {fd.FINALS_PATH} has no {', '.join(missing)}: run field_data.py --races, then --report")
        return None
    return scored


def _write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1)


def run_compare():
    """The race-by-race features against today's on DEV_YEARS, with the
    shuffled controls, printed and saved. Never touches the served model."""
    print(f"=== Field model: race by race against today, on {DEV_YEARS[0]}-{DEV_YEARS[-1]} ===")
    scored = _scored_with_races()
    if scored is None:
        return 1
    report, _ = compare(scored)
    print_compare(report)
    _write_json(COMPARE_PATH, report)
    print(f"\n  -> {COMPARE_PATH}")
    return 0


def load_experiments(path=EXPERIMENTS_PATH):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return []


def _experiment_row(report):
    d, o = report["decision"], report["overall"]
    return {"name": report["candidateName"], "ranAt": report["builtAt"], "years": report["testYears"],
            "spec": report["spec"], "finals": o["candidate"]["finals"],
            "candidatePct": o["candidate"]["model"], "baselinePct": o["baseline"]["model"],
            "candidateWinners": o["candidate"].get("modelWinners"),
            "baselineWinners": o["baseline"].get("modelWinners"),
            "meanLlGain": d["meanLlGain"], "llLower90": d["llLower90"], "meanHitsDiff": d["meanHitsDiff"],
            "asiaMeanLlGain": d["asiaMeanLlGain"]}


def run_experiments(names):
    """Each named experiment against today's model on DEV_YEARS. Every run is
    appended to EXPERIMENTS_PATH and kept, so the record says how many things
    were tried before the locked years were used."""
    unknown = [name for name in names if name not in EXPERIMENTS]
    if unknown:
        print(f"  no such experiment: {', '.join(unknown)} (see EXPERIMENTS)")
        return 1
    print(f"=== Field model: experiments against today, on {DEV_YEARS[0]}-{DEV_YEARS[-1]} ===")
    scored = _scored_with_races()
    if scored is None:
        return 1
    races = fd.load_seasons(fd.RACES_DIR) if any(EXPERIMENTS[n].get("race") for n in names) else None
    log = load_experiments()
    print(f"  {'experiment':<26} {'medallists':>10} {'winners':>8} {'LL gain':>9} {'lower90':>9} {'Asia LL':>9}")
    row = None
    for name in names:
        report, _ = compare(scored, candidate=name, controls=0, races=races)
        row = _experiment_row(report)
        log.append(row)
        _write_json(EXPERIMENTS_PATH, log)
        print(f"  {name:<26} {row['candidatePct']:>9}% {row['candidateWinners']:>8} {_signed(row['meanLlGain']):>9} "
              f"{_signed(row['llLower90']):>9} {_signed(row['asiaMeanLlGain']):>9}")
    latest = {r["name"]: r for r in log}
    chosen = choose_experiment(list(latest.values()))
    print(f"\n  today's model on these years: {row['baselinePct']}% of medallists, {row['baselineWinners']} winners")
    print(f"  {len(latest)} experiments on record. By the fixed rule, the candidate for the locked years is "
          f"{chosen or 'none: no experiment gained on the tuning years'}.")
    print(f"  -> {EXPERIMENTS_PATH}")
    return 0


def run_holdout(name, path=HOLDOUT_PATH):
    """The chosen experiment against today's model on HOLDOUT_YEARS, once,
    under compare_decision, with the candidate fitted on every final saved to
    V2_MODEL_PATH. It will not run twice: another look at the locked years,
    after changing anything, would turn them into tuning data as well."""
    if os.path.exists(path):
        print(f"  {path} exists: the locked years have been used and cannot decide anything again")
        return 1
    if name not in EXPERIMENTS:
        print(f"  no such experiment: {name} (see EXPERIMENTS)")
        return 1
    print(f"=== Field model: {name} against today, on the locked years {HOLDOUT_YEARS[0]}-{HOLDOUT_YEARS[-1]} ===")
    scored = _scored_with_races()
    if scored is None:
        return 1
    races = fd.load_seasons(fd.RACES_DIR) if EXPERIMENTS[name].get("race") else None
    report, model = compare(scored, candidate=name, years=HOLDOUT_YEARS, races=races)
    latest = {r["name"]: r for r in load_experiments()}
    report["experimentsTried"] = sorted(latest)
    report["chosenByRule"] = choose_experiment(list(latest.values()))
    report["followsNewOverOld"] = bool(EXPERIMENTS[name].get("newOverOld"))
    print_compare(report)
    print(f"  experiments tried on the tuning years first: {len(latest)}; the rule chose "
          f"{report['chosenByRule']}, and this run scored {name}")
    model.update({"fitAt": report["builtAt"], "experiment": name, "holdout": report["decision"]})
    _write_json(path, report)
    _write_json(V2_MODEL_PATH, model)
    print(f"\n  -> {path}\n  -> {V2_MODEL_PATH}\n  The served model, {MODEL_PATH}, is unchanged.")
    return 0


def choose_all_seasons(rows):
    """The version kept by run_all_seasons, the rule fixed on 2026-09-16 before
    any version ran: the most medallists named, then the most winners named
    first, then the larger mean top-three log-likelihood gain over today's
    model (0 for today's model itself)."""
    return max(rows, key=lambda r: (r["medallists"], r["winners"] or 0, r["meanLlGain"] or 0.0))["name"]


def _all_seasons_row(name, report, side, ll_gain):
    overall = report["overall"][side]
    asia = (report["byTier"].get("asia") or {}).get(side) or {}
    return {"name": name, "spec": json.loads(json.dumps(EXPERIMENTS[name])), "finals": overall["finals"],
            "medallists": overall["hits"], "possible": overall["possible"], "medallistsPct": overall["model"],
            "winners": overall.get("modelWinners"), "pointsPct": overall["points"],
            "pointsWinners": overall.get("pointsWinners"), "meanLlGain": ll_gain,
            "asiaFinals": asia.get("finals"), "asiaPct": asia.get("model"), "asiaPointsPct": asia.get("points"),
            "bySeason": {str(year): pair[side]["model"] for year, pair in report["byYear"].items()}}


def run_all_seasons(names=None, years=None, path=ALL_SEASONS_PATH):
    """Today's model, recent over old and this season only
    (ALL_SEASON_CANDIDATES), each walked forward over every season in
    ALL_SEASON_YEARS on the same finals, and the one choose_all_seasons keeps.

    The user's choice on 2026-09-16: test the ways old marks can count on every
    past season and keep the most accurate, one model for every event. Every
    one of these seasons had already been used for testing or training, so the
    version kept is picked on results already seen; the Asian Games are the
    next finals none of them has seen."""
    names = list(names or ALL_SEASON_CANDIDATES)
    years = list(years or ALL_SEASON_YEARS)
    unknown = [name for name in names if name not in EXPERIMENTS]
    if unknown:
        print(f"  no such experiment: {', '.join(unknown)} (see EXPERIMENTS)")
        return 1
    print(f"=== Field model: {len(names)} versions on every season, {years[0]}-{years[-1]} ===")
    scored = _scored_with_races()
    if scored is None:
        return 1
    races = fd.load_seasons(fd.RACES_DIR) if any(EXPERIMENTS[n].get("race") for n in names) else None
    baseline, rows = names[0], []
    for name in names[1:]:
        report, _ = compare(scored, candidate=name, baseline=baseline, years=years, controls=0, races=races)
        if not rows:
            rows.append(_all_seasons_row(baseline, report, "baseline", 0.0))
        rows.append(_all_seasons_row(name, report, "candidate", report["decision"]["meanLlGain"]))
    chosen = choose_all_seasons(rows)
    _write_json(path, {
        "builtAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "years": years,
        "rule": "most medallists named, then most winners named first, then mean top-three "
                "log-likelihood gain over today's model; fixed before any version ran",
        "candidates": rows, "chosen": chosen})

    print(f"\n  {'version':<22} {'medallists':>16} {'winners':>8} {'LL gain':>9} {'Asian finals':>13}")
    for r in rows:
        print(f"  {r['name']:<22} {r['medallists']:>6} ({r['medallistsPct']:>4}%) {r['winners']:>8} "
              f"{_signed(r['meanLlGain']):>9} {r['asiaPct']:>12}%")
    print(f"  points                 {rows[0]['pointsPct']:>15}% {rows[0]['pointsWinners']:>8}"
          f"{'':>10} {rows[0]['asiaPointsPct']:>12}%")
    print(f"\n  medallists by season: {'':<8}" + " ".join(f"{y:>5}" for y in years))
    for r in rows:
        print(f"  {r['name']:<30}" + " ".join(f"{r['bySeason'].get(str(y), ''):>5}" for y in years))
    print(f"\n  {rows[0]['finals']} finals. Kept by the fixed rule: {chosen}")
    print(f"  -> {path}")
    return 0


def run_rule_test(name="old_marks"):
    """Print the rule test for an experiment's old-marks settings, and for
    information what the served model does with the same made-up finals."""
    if name not in EXPERIMENTS:
        print(f"  no such experiment: {name} (see EXPERIMENTS)")
        return 1
    print(f"=== Field model: the rule for old marks, on made-up finals ({name}) ===")
    report = check_old_marks_rule(EXPERIMENTS[name].get("fade"))
    print_rule_test(report)
    served = load_model()
    if served:
        same = check_old_marks_rule(fade=fade_of(served), model=served)
        broken = [c["label"] for c in same["cases"] if not c["leaderStaysAhead"]]
        print(f"\n  for information, the served model ({served.get('experiment')}): "
              f"{'holds the rule' if same['passed'] else 'breaks it'}"
              + (f" on {len(broken)} of {len(same['cases'])} cases, first at {broken[0]}" if broken else ""))
    return 0 if report["passed"] else 1


# ---- trial and error on the old-marks settings --------------------------------

# How much last season counts, how fast older seasons fade, and how much of the
# gap to the leader an old mark may close. The user asked for these to be found
# by trial and error on past seasons, so every combination below is walked
# forward over every season and recorded. The last three are outside the rule on
# purpose: they say in the record what the rule costs, and choose_old_marks
# cannot keep them.
OLD_MARK_GRID = [{"last_season": last, "per_year": per_year, "cap_share": cap}
                 for last in (0.2, 0.35, 0.5)
                 for per_year in (0.3, 0.5, 0.7)
                 for cap in (0.5, 0.75, 0.9)]
OLD_MARK_OUT_OF_RULE = [
    # Old marks in full, uncapped: closest to the model the user objected to.
    {"last_season": 1.0, "per_year": 1.0, "cap_share": 5.0},
    # Nearly in full, barely fading.
    {"last_season": 0.7, "per_year": 0.9, "cap_share": 2.0},
    # Old marks ignored altogether, which is not the rule either.
    {"last_season": 0.0, "per_year": 0.5, "cap_share": 0.75},
]


def fade_name(fade):
    """A setting's name in the record: last season, the fade per year and the cap."""
    return (f"last{round(fade['last_season'] * 100):d}"
            f"_per{round(fade['per_year'] * 100):d}"
            f"_cap{round(fade['cap_share'] * 100):d}")


def choose_old_marks(rows):
    """The setting kept, by the rule fixed before the grid ran: among those that
    pass the rule test, the most medallists named, then the most winners, then
    the largest mean top-three log-likelihood. None when none passes.

    The rule test comes first by the user's decision of 2026-09-16: the rule for
    old marks holds even at a cost in medallists, so a setting that breaks it
    cannot be kept however accurate it is."""
    passed = [r for r in rows if r["rulePassed"]]
    if not passed:
        return None
    return max(passed, key=lambda r: (r["medallists"], r["winners"], r["meanLl"]))["name"]


def _season_row(name, fade, results, rule_passed):
    """One setting's line in the record: medallists and winners named over every
    season it was walked through, and how sure it was of the order."""
    finals = len(results)
    asia = results[results["tier"] == "asia"]
    return {
        "name": name, "fade": dict(fade), "rulePassed": bool(rule_passed), "finals": finals,
        "medallists": int(results["model_hits"].sum()), "possible": 3 * finals,
        "medallistsPct": round(100 * results["model_hits"].sum() / (3 * finals), 1) if finals else None,
        "winners": int(results["model_winner"].sum()),
        "meanLl": round(float(results["ll"].mean()), 4),
        "pointsPct": round(100 * results["points_hits"].sum() / (3 * finals), 1) if finals else None,
        "pointsWinners": int(results["points_winner"].sum()),
        "asiaFinals": len(asia),
        "asiaPct": round(100 * asia["model_hits"].sum() / (3 * len(asia)), 1) if len(asia) else None,
        "asiaPointsPct": round(100 * asia["points_hits"].sum() / (3 * len(asia)), 1) if len(asia) else None,
        "bySeason": {str(int(year)): round(100 * g["model_hits"].sum() / (3 * len(g)), 1)
                     for year, g in results.groupby("year")},
    }


def run_old_marks(grid=None, years=None, path=OLD_MARKS_PATH, baseline="v2_form_best_5"):
    """Every old-marks setting walked forward over every season in
    ALL_SEASON_YEARS, and the one choose_old_marks keeps.

    The user's rule comes first here, not accuracy: each setting has to pass
    check_old_marks_rule before it can be kept, and the most accurate of those
    that do is the one served, even if it names fewer medallists than the model
    that broke the rule. Every setting tried is written to `path`, the ones
    outside the rule included, so the record says what the rule cost.

    These seasons have all been used for testing or training before, so this
    tunes on results already seen. The Asian Games are the first finals no
    version of the model has seen, and the page says so."""
    grid = list(grid or (OLD_MARK_GRID + OLD_MARK_OUT_OF_RULE))
    years = list(years or ALL_SEASON_YEARS)
    spec = EXPERIMENTS["old_marks"]
    print(f"=== Field model: {len(grid)} old-marks settings on every season, {years[0]}-{years[-1]} ===")
    scored = _scored_with_races()
    if scored is None:
        return 1
    with_races = spec_scored(scored, spec, fd.load_seasons(fd.RACES_DIR))

    base_spec = EXPERIMENTS[baseline]
    base_finals = build_finals(spec_scored(scored, base_spec, None), base_spec["features"],
                               base_spec.get("fade"))
    base = backtest(base_finals, base_spec.get("l2", L2), years=years,
                    by_group=base_spec.get("by_group", False), bounds=base_spec.get("bounds"))[0]
    # Whether the model being replaced keeps the rule, read under the model
    # itself rather than under the new family's settings.
    base_rule = check_old_marks_rule(model=fit_spec(base_finals, base_spec))["passed"]
    base_row = _season_row(baseline, {}, base, base_rule)
    print(f"  the served model, {baseline}, over the same seasons: {base_row['medallistsPct']}% of medallists, "
          f"{base_row['winners']} winners; a points ranking {base_row['pointsPct']}%")

    rows = []
    for i, fade in enumerate(grid, 1):
        name = fade_name(fade)
        rule = check_old_marks_rule(fade)
        results = backtest(build_finals(with_races, spec["features"], fade), spec.get("l2", L2),
                           years=years, bounds=spec["bounds"])[0]
        row = _season_row(name, fade, results, rule["passed"])
        rows.append(row)
        print(f"  {i:>3}/{len(grid)} {name:<24} {row['medallistsPct']:>5}% of medallists, "
              f"{row['winners']:>3} winners, mean LL {row['meanLl']:>8}"
              f"{'' if rule['passed'] else '   OUTSIDE THE RULE, cannot be kept'}")
    chosen = choose_old_marks(rows)
    best_overall = max(rows, key=lambda r: (r["medallists"], r["winners"], r["meanLl"]))["name"]
    _write_json(path, {
        "builtAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "years": years,
        "rule": "the rule for old marks first (check_old_marks_rule), then the most medallists named, "
                "then the most winners, then the mean top-three log-likelihood; fixed before the grid ran",
        "seasonsAlreadyUsed": True, "baseline": base_row, "settings": rows,
        "chosen": chosen, "mostAccurateOfAll": best_overall})

    kept = next((r for r in rows if r["name"] == chosen), None)
    print(f"\n  {len(rows)} settings tried, {sum(r['rulePassed'] for r in rows)} inside the rule.")
    print(f"  kept: {chosen}" + (f" ({kept['fade']})" if kept else ""))
    if kept:
        print(f"    {kept['medallists']} of {kept['possible']} medallists ({kept['medallistsPct']}%), "
              f"{kept['winners']} winners, mean LL {kept['meanLl']}")
        print(f"    against the served model's {base_row['medallistsPct']}% and {base_row['winners']} winners, "
              f"and a points ranking's {base_row['pointsPct']}%")
        print(f"    on the Asian finals: {kept['asiaPct']}% against the served model's {base_row['asiaPct']}%")
    if best_overall != chosen:
        best = next(r for r in rows if r["name"] == best_overall)
        print(f"  the most accurate setting of all was {best_overall} ({best['medallistsPct']}%, "
              f"{best['winners']} winners), and it is outside the rule, so it was not kept: "
              f"the user chose the rule over the numbers.")
    print(f"  every season here had already been used for testing or training; the Asian Games are the "
          f"first finals no version has seen.")
    print(f"  -> {path}")
    return 0


def run_refit(name=None, model_path=MODEL_PATH, previous_path=PREVIOUS_MODEL_PATH):
    """Retrain the served experiment on every final on file and save it as the
    served model, keeping the model it replaces at `previous_path`.

    For after new finals have joined the history (field_data.py --finals, --ids,
    --races and --report): the model learns from them, and the experiment and
    its locked-year record (`holdout`) carry over. It is not a new test. A change
    of experiment or features is one, and goes through --experiments on finals
    the model has not been tested on."""
    current = load_model(model_path) or {}
    name = name or current.get("experiment")
    if name not in EXPERIMENTS:
        print(f"  no such experiment: {name} (see EXPERIMENTS)")
        return 1
    print(f"=== Field model: retraining {name} on every final on file ===")
    scored = _scored_with_races()
    if scored is None:
        return 1
    spec = EXPERIMENTS[name]
    finals = build_finals(spec_scored(scored, spec), spec["features"], spec.get("fade"))
    model = fit_spec(finals, spec)
    model.update({"fitAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "experiment": name,
                  "holdout": current.get("holdout") if current.get("experiment") == name else None})
    if current:
        _write_json(previous_path, current)
    _write_json(model_path, model)
    years = [f["year"] for f in finals]
    print(f"  {len(finals)} finals from {min(years)} to {max(years)}")
    print(f"  -> {model_path}\n  the model it replaces -> {previous_path}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--backtest", action="store_true")
    parser.add_argument("--compare", action="store_true",
                        help="the race-by-race features against today's on DEV_YEARS, with shuffled controls")
    parser.add_argument("--experiments", nargs="*", default=None, metavar="NAME",
                        help="try EXPERIMENTS (all of them when none is named) on DEV_YEARS against today's model")
    parser.add_argument("--holdout", default=None, metavar="NAME",
                        help="score the chosen experiment on HOLDOUT_YEARS, once")
    parser.add_argument("--refit", nargs="?", const="", default=None, metavar="NAME",
                        help="retrain the served experiment (or NAME) on every final on file, after new finals are added")
    parser.add_argument("--all-seasons", action="store_true",
                        help="today's model, recent over old and this season only on every past season")
    parser.add_argument("--rule-test", nargs="?", const="old_marks", default=None, metavar="NAME",
                        help="check the rule for old marks on made-up finals, the gate a tuned setting must pass")
    parser.add_argument("--old-marks", action="store_true",
                        help="try every old-marks setting on every past season and keep the one the rule allows")
    args = parser.parse_args(argv)
    if args.refit is not None:
        return run_refit(args.refit or None)
    if args.all_seasons:
        return run_all_seasons()
    if args.rule_test is not None:
        return run_rule_test(args.rule_test or "old_marks")
    if args.old_marks:
        return run_old_marks()
    if args.compare:
        return run_compare()
    if args.experiments is not None:
        return run_experiments(args.experiments or list(EXPERIMENTS))
    if args.holdout:
        return run_holdout(args.holdout)
    if not args.backtest:
        parser.print_help()
        return 0

    print("=== Field model: walk-forward backtest against points ===")
    scored = load_scored_finals()
    finals = build_finals(scored)
    print(f"  {len(finals)} usable finals, {sum(f['top_idx'] is not None for f in finals)} with a scored podium, "
          f"{sum(f['top_idx'] is not None and f['trainable'] for f in finals)} of those trainable")
    results, athletes = backtest(finals)
    control, _ = backtest(finals, shuffle_seed=7)
    decisions = ship_decisions(results, control)
    report = summarise(results, athletes, control, decisions, fd.coverage(scored))
    print_report(report)

    model = fit(finals)
    model["ship"] = decisions
    model["fitAt"] = report["builtAt"]
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "w", encoding="utf-8") as f:
        json.dump(model, f, indent=1)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    print(f"\n  -> {MODEL_PATH}\n  -> {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
