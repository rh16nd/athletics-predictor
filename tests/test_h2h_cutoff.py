"""The head-to-head feature must not read races that had not happened yet.

data/h2h/h2h_rates.csv has no time dimension: it aggregates every meeting on
disk, and add_h2h_features read it whole for every label year. Pooled across
531 championship finals that is lookahead, and for 229 of them it is worse than
that -- the scraper's meeting list and the label file overlap, so the 2022
World Championships men's 100m page IS the men_100m 2022 Worlds label. The
model was reading who won the race it was being asked to predict, out of a
feature. Measured at 2.5 points of toplist and 2.0 of field accuracy.

These tests are the pin. Nothing else in the suite would notice the cut-off
being dropped again, because a leak makes every number go UP.
"""
import os

import pandas as pd
import pytest

import train_model as tm


# Two athletes, three meetings. Alpha beat Beta in May; Beta beat Alpha in
# July and again in September. A final held in June may see only the first.
MEETS = pd.DataFrame([
    {"meet": "May Meeting",  "year": 2023, "competition_level": "DL",
     "discipline": "men_100m", "race": "Final", "heat": "", "place": 1.0,
     "athlete": "Alpha Runner", "mark": "9.90"},
    {"meet": "May Meeting",  "year": 2023, "competition_level": "DL",
     "discipline": "men_100m", "race": "Final", "heat": "", "place": 2.0,
     "athlete": "Beta Runner", "mark": "9.95"},
    {"meet": "July Meeting", "year": 2023, "competition_level": "DL",
     "discipline": "men_100m", "race": "Final", "heat": "", "place": 1.0,
     "athlete": "Beta Runner", "mark": "9.88"},
    {"meet": "July Meeting", "year": 2023, "competition_level": "DL",
     "discipline": "men_100m", "race": "Final", "heat": "", "place": 2.0,
     "athlete": "Alpha Runner", "mark": "9.92"},
    {"meet": "Sept Meeting", "year": 2023, "competition_level": "DL",
     "discipline": "men_100m", "race": "Final", "heat": "", "place": 1.0,
     "athlete": "Beta Runner", "mark": "9.85"},
    {"meet": "Sept Meeting", "year": 2023, "competition_level": "DL",
     "discipline": "men_100m", "race": "Final", "heat": "", "place": 2.0,
     "athlete": "Alpha Runner", "mark": "9.99"},
])

DATES = pd.DataFrame([
    {"meet": "May Meeting",  "year": 2023, "competition_level": "DL", "date": "2023-05-10"},
    {"meet": "July Meeting", "year": 2023, "competition_level": "DL", "date": "2023-07-10"},
    {"meet": "Sept Meeting", "year": 2023, "competition_level": "DL", "date": "2023-09-10"},
])


def frame(cutoff):
    """Both athletes pooled for one final held on `cutoff`."""
    return pd.DataFrame([
        {"athlete_name": "Alpha Runner", "discipline": "men_100m", "year": 2023,
         "competition": "Test Final", "cutoff": pd.Timestamp(cutoff)},
        {"athlete_name": "Beta Runner", "discipline": "men_100m", "year": 2023,
         "competition": "Test Final", "cutoff": pd.Timestamp(cutoff)},
    ])


@pytest.fixture
def dated_h2h(tmp_path, monkeypatch):
    results = tmp_path / "meet_results.csv"
    dates = tmp_path / "meet_dates.csv"
    MEETS.to_csv(results, index=False)
    DATES.to_csv(dates, index=False)
    monkeypatch.setattr(tm, "MEET_RESULTS_PATH", str(results))
    monkeypatch.setattr(tm, "MEET_DATES_PATH", str(dates))
    # The records are memoised, so a test that did not clear the cache would
    # silently score against whatever the previous test loaded.
    monkeypatch.setattr(tm, "_DATED_RECORDS", None)
    yield
    tm._DATED_RECORDS = None


def rate_for(df, name):
    return float(df.loc[df["athlete_name"] == name, "h2h_win_rate"].iloc[0])


def test_a_final_in_june_sees_only_the_may_race(dated_h2h):
    """One meeting apiece is below the >=2 threshold, so nothing counts yet --
    which is itself the point: at that date these two had met once."""
    out = tm.add_h2h_features(frame("2023-06-01"))
    assert rate_for(out, "Alpha Runner") == 0.5
    assert rate_for(out, "Beta Runner") == 0.5


def test_a_final_in_august_sees_may_and_july_but_not_september(dated_h2h):
    out = tm.add_h2h_features(frame("2023-08-01"))
    # One win each from two meetings.
    assert rate_for(out, "Alpha Runner") == pytest.approx(0.5)
    assert rate_for(out, "Beta Runner") == pytest.approx(0.5)


def test_a_final_in_october_sees_all_three(dated_h2h):
    out = tm.add_h2h_features(frame("2023-10-01"))
    assert rate_for(out, "Beta Runner") == pytest.approx(2 / 3)
    assert rate_for(out, "Alpha Runner") == pytest.approx(1 / 3)


def test_a_meeting_on_the_final_s_own_day_is_excluded(dated_h2h):
    """The sharp case, and the reason this file exists. A meeting is dated on
    its last day, which for a championship event page is the final's own day,
    so predicting that final must not see it. Cut at 10 September, the
    September meeting is invisible and the record reads 1-1, not 1-2."""
    out = tm.add_h2h_features(frame("2023-09-10"))
    assert rate_for(out, "Beta Runner") == pytest.approx(0.5)
    assert rate_for(out, "Alpha Runner") == pytest.approx(0.5)


def test_an_undated_meeting_is_dropped_not_defaulted(tmp_path, monkeypatch):
    """Dating a meeting by guesswork would decide whether a later final was
    allowed to see it, which is the mistake being fixed. Drop it instead."""
    results = tmp_path / "meet_results.csv"
    dates = tmp_path / "meet_dates.csv"
    MEETS.to_csv(results, index=False)
    DATES[DATES["meet"] != "Sept Meeting"].to_csv(dates, index=False)
    monkeypatch.setattr(tm, "MEET_RESULTS_PATH", str(results))
    monkeypatch.setattr(tm, "MEET_DATES_PATH", str(dates))
    monkeypatch.setattr(tm, "_DATED_RECORDS", None)
    try:
        records = tm.dated_h2h_records()
        assert "Sept Meeting" not in set(records["meet"])
        assert set(records["meet"]) == {"May Meeting", "July Meeting"}
    finally:
        tm._DATED_RECORDS = None


def test_the_diamond_league_only_path_is_unchanged(monkeypatch):
    """No cutoff column means build_labeled_dataset built this frame, and that
    path must keep reading h2h_rates.csv whole -- it is what the deployed
    Diamond-League-only model was trained on."""
    calls = []

    real_read_csv = pd.read_csv

    def spy(path, *args, **kwargs):
        calls.append(str(path))
        return real_read_csv(path, *args, **kwargs)

    monkeypatch.setattr(tm.pd, "read_csv", spy)
    monkeypatch.setattr(tm, "_DATED_RECORDS", None)
    out = tm.add_h2h_features(frame("2023-08-01").drop(columns=["cutoff", "competition"]))
    assert "h2h_win_rate" in out.columns
    assert any(c.endswith("h2h_rates.csv") for c in calls), calls
    assert not any(c.endswith("meet_results.csv") for c in calls), calls
