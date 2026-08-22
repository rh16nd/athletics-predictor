"""Unit tests for src/train_model.py's pure data-transformation functions.

Deliberately doesn't test build_labeled_dataset()/train_and_backtest() end to
end -- those need real data/raw/*.csv and data/dl_final_results.csv on disk.
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
