"""Unit tests for src/train_model.py's pure data-transformation functions.

build_labeled_dataset()/train_and_backtest() are covered separately in
test_fixtures_integration.py, against small monkeypatched fixture files
rather than real data/raw/*.csv and data/dl_final_results.csv on disk.
The two regression tests below (add_season_rank, build_features career_best)
encode real bugs found and fixed in this project: without them, either bug
could silently come back and nothing would catch it.
"""
import pandas as pd
import pytest

import train_model as tm


def test_convert_mark_to_seconds_plain_number():
    assert tm.convert_mark_to_seconds("9.83") == pytest.approx(9.83)


def test_convert_mark_to_seconds_minutes_format():
    assert tm.convert_mark_to_seconds("3:43.73") == pytest.approx(3 * 60 + 43.73)


def test_convert_mark_to_seconds_invalid_returns_none():
    assert tm.convert_mark_to_seconds("DNF") is None
    assert tm.convert_mark_to_seconds("DQ") is None


def test_normalize_name_strips_accents_and_case():
    assert tm.normalize_name("Kristjan ČEH") == tm.normalize_name("KRISTJAN CEH")
    assert tm.normalize_name("Faith KIPYEGON") == "FAITH KIPYEGON"


def test_normalize_name_passthrough_non_string():
    assert tm.normalize_name(None) is None


def test_add_season_rank_field_event_ranks_highest_mark_first():
    """Regression test: add_season_rank() once only special-cased "men_PV"
    for descending rank direction, leaving other field events (including
    women_PV and men_LJ, both already-trained disciplines at the time)
    ranked backwards -- the lowest jump/throw got rank 1 instead of the
    highest."""
    df = pd.DataFrame({
        "discipline": ["men_HJ", "men_HJ", "men_HJ"],
        "year": [2024, 2024, 2024],
        "athlete_name": ["A", "B", "C"],
        "season_best": [2.20, 2.35, 2.28],
    })
    ranked = tm.add_season_rank(df)
    best = ranked.loc[ranked["athlete_name"] == "B"].iloc[0]
    assert best["season_rank"] == 1  # 2.35m is the highest jump -- must be rank 1


def test_add_season_rank_track_event_ranks_lowest_time_first():
    df = pd.DataFrame({
        "discipline": ["men_100m", "men_100m", "men_100m"],
        "year": [2024, 2024, 2024],
        "athlete_name": ["A", "B", "C"],
        "season_best": [9.95, 9.79, 10.05],
    })
    ranked = tm.add_season_rank(df)
    best = ranked.loc[ranked["athlete_name"] == "B"].iloc[0]
    assert best["season_rank"] == 1  # 9.79s is the fastest -- must be rank 1


def _athlete_rows(marks_by_year, athlete="Athlete", country="USA"):
    return pd.DataFrame([
        {"athlete_name": athlete, "Mark": mark, "year": year, "age": 25.0, "country": country}
        for year, marks in marks_by_year.items() for mark in marks
    ])


def test_build_features_career_best_excludes_future_years():
    """Regression test: build_features() once computed career_best from an
    athlete's ENTIRE history, leaking a mark set in a LATER season into an
    earlier year's row. 2021's career_best here must stay bounded to what
    was true as of 2021, even though a faster mark exists in 2023."""
    df = _athlete_rows({2021: ["10.10"], 2022: ["9.90"], 2023: ["9.50"]})
    features = tm.build_features(df, "men_100m")
    row_2021 = features[features["year"] == 2021].iloc[0]
    assert row_2021["career_best"] == pytest.approx(10.10)


def test_build_features_track_season_best_is_fastest_mark():
    df = _athlete_rows({2023: ["10.20", "9.95", "10.05"]})
    features = tm.build_features(df, "men_100m")
    row = features[features["year"] == 2023].iloc[0]
    assert row["season_best"] == pytest.approx(9.95)


def test_build_features_field_event_season_best_is_longest_mark():
    df = _athlete_rows({2023: ["7.80", "8.15", "7.95"]})
    features = tm.build_features(df, "men_LJ")
    row = features[features["year"] == 2023].iloc[0]
    assert row["season_best"] == pytest.approx(8.15)


def test_train_disciplines_and_field_events_are_consistent():
    """Every FIELD_EVENTS key must also be a real trained discipline -- a
    typo'd key here would silently never match anything in add_season_rank/
    build_features's is_track branch."""
    assert tm.FIELD_EVENTS.issubset(set(tm.TRAIN_DISCIPLINES.keys()))


def test_apply_wind_adjustment_no_penalty_within_legal_limit():
    assert tm.apply_wind_adjustment(9.90, wind=0.9, is_field=False) == 9.90
    assert tm.apply_wind_adjustment(8.20, wind=1.0, is_field=True) == 8.20


def test_apply_wind_adjustment_track_event_penalty_increases_mark():
    # A following wind makes a time look artificially fast -- the fair
    # (adjusted) mark must be pushed UP (slower), not down.
    adjusted = tm.apply_wind_adjustment(9.90, wind=2.0, is_field=False)
    assert adjusted > 9.90


def test_apply_wind_adjustment_field_event_penalty_decreases_mark():
    """Regression test: this used to always ADD the penalty regardless of
    event type, which is backwards for a higher-is-better field-event mark
    -- a wind-aided jump would have looked artificially BETTER instead of
    worse. Never actually live before this fix since field events weren't
    in WIND_EVENTS yet, but the bug was real and would have hit the moment
    they were added without this fix alongside it."""
    adjusted = tm.apply_wind_adjustment(8.20, wind=2.0, is_field=True)
    assert adjusted < 8.20


# --- Podium-position breakdown (2026-08-25) ------------------------------
# The headline metric is an order-blind SET intersection, so it cannot say
# that the model finds ~87% of actual winners but only ~37% of actual 3rd
# places. These cover the breakdown reported alongside it -- and, critically,
# that adding it did NOT change what the headline itself means.

def _fold_frame():
    """One discipline: predicted order A,B,C; real podium is A(1st), C(2nd), D(3rd)."""
    return pd.DataFrame({
        "discipline": ["men_100m"] * 4,
        "athlete_name": ["A", "B", "C", "D"],
        "win_probability": [0.9, 0.8, 0.7, 0.1],
        "dl_top3": [1, 0, 1, 1],
        "dl_rank": [1, None, 2, 3],
        "year": [2024] * 4,
    })


def test_walk_forward_folds_still_returns_exactly_two_values():
    """tune_hyperparameters() unpacks a 2-tuple. Collecting the new
    breakdown must stay an optional out-parameter, not a return-shape
    change, or every --tune run breaks."""
    import inspect
    src = inspect.getsource(tm.walk_forward_folds)
    assert "return total_correct, total_possible" in src
    assert "stats=None" in inspect.signature(tm.walk_forward_folds).__str__().replace(" ", "")


def test_position_weighted_score_ranks_winners_above_third():
    """3/2/1 weighting: finding only the winner must beat finding only 3rd."""
    winner_only = {"found": {1: 1, 2: 0, 3: 0}, "total": {1: 1, 2: 1, 3: 1}}
    third_only = {"found": {1: 0, 2: 0, 3: 1}, "total": {1: 1, 2: 1, 3: 1}}

    def weighted(stats):
        c = sum(stats["found"][r] * w for r, w in ((1, 3), (2, 2), (3, 1)))
        p = sum(stats["total"][r] * w for r, w in ((1, 3), (2, 2), (3, 1)))
        return c / p

    assert weighted(winner_only) > weighted(third_only)
    assert weighted(winner_only) == pytest.approx(3 / 6)
    assert weighted(third_only) == pytest.approx(1 / 6)


def test_breakdown_printer_handles_an_empty_stats_dict():
    """A fold set with no labeled podium rows must not raise."""
    tm._print_position_breakdown({})
    tm._print_position_breakdown({"total": {1: 0, 2: 0, 3: 0}, "found": {1: 0, 2: 0, 3: 0}})


def test_flat_metric_is_still_order_blind():
    """Guard the headline's definition: it is a set intersection, so getting
    the right three people in the wrong order is still a perfect score. If
    this ever fails, the historical numbers in HANDOFF stopped being
    comparable."""
    df = _fold_frame()
    disc_df = df.sort_values("win_probability", ascending=False)
    predicted = set(disc_df.head(3)["athlete_name"])
    actual = set(disc_df[disc_df["dl_top3"] == 1]["athlete_name"])
    # predicted A,B,C vs actual A,C,D -> 2 of 3, regardless of ordering
    assert len(predicted & actual) == 2
