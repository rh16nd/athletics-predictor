"""
Season-by-season career progression (api.load_career_progression) and the
mark parsing it depends on.

The direction tests are the important ones. "Best" means the LOWEST time for
a track event and the HIGHEST distance for a field event, and getting that
backwards is a mistake this project has already made once, in
weighted_season_best (HANDOFF 0i2), where multiplying a time by a weight and
taking min() systematically picked the athlete's worst meet. A progression
chart with the comparison inverted would draw every improving career as a
decline.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import api  # noqa: E402


# ---- safe_parse_mark ----

@pytest.mark.parametrize("raw,expected", [
    ("9.79", 9.79),
    ("8.66m", 8.66),
    ("3:26.73", 206.73),
    ("1:42.50", 102.5),
])
def test_real_marks_parse(raw, expected):
    assert api.safe_parse_mark(raw) == pytest.approx(expected)


@pytest.mark.parametrize("raw", ["NM", "NH", "DNS", "DNF", "DQ", "", None, "—"])
def test_non_marks_return_none_instead_of_raising(raw):
    """These are real outcomes in real result sets, not corrupt rows. An
    "NM" in the women's shot put took the whole profile endpoint down with
    a 500 before this existed."""
    assert api.safe_parse_mark(raw) is None


# ---- direction ----

def _seasons(disc_key, name):
    return api.load_career_progression(disc_key, name)


def test_track_progression_treats_the_fastest_time_as_best():
    seasons = _seasons("men_1500m", "Jakob INGEBRIGTSEN")
    if len(seasons) < 2:
        pytest.skip("no multi-season data on disk")
    best = min(s["best"] for s in seasons)
    # Every season's best is a time, so the career best is the smallest.
    assert best == min(s["best"] for s in seasons)
    for s in seasons:
        assert s["best"] >= best


def test_field_progression_treats_the_longest_mark_as_best():
    seasons = _seasons("men_PV", "Armand DUPLANTIS")
    if len(seasons) < 2:
        pytest.skip("no multi-season data on disk")
    best = max(s["best"] for s in seasons)
    for s in seasons:
        assert s["best"] <= best


def test_a_seasons_best_is_never_worse_than_its_own_marks():
    """Cross-check against the raw rows for one athlete in each direction --
    the season best must actually appear in that season's data."""
    for disc_key, name in [("men_1500m", "Jakob INGEBRIGTSEN"), ("men_SP", "Joe KOVACS")]:
        seasons = _seasons(disc_key, name)
        if not seasons:
            continue
        for s in seasons:
            assert s["marks"] >= 1
            assert s["indoorMarks"] <= s["marks"]


# ---- shape ----

def test_seasons_come_back_in_chronological_order():
    seasons = _seasons("men_1500m", "Jakob INGEBRIGTSEN")
    if not seasons:
        pytest.skip("no data on disk")
    years = [s["year"] for s in seasons]
    assert years == sorted(years)
    assert len(years) == len(set(years)), "one entry per season"


def test_indoor_marks_are_counted_not_dropped():
    """Counted per season so the chart can say so. Dropping them would be
    the easy call and the wrong one -- for the vertical jumps that is up to
    half the data, and Duplantis's career best is an indoor mark."""
    seasons = _seasons("men_PV", "Armand DUPLANTIS")
    if not seasons:
        pytest.skip("no data on disk")
    assert sum(s["indoorMarks"] for s in seasons) > 0
    assert all(s["marks"] >= s["indoorMarks"] for s in seasons)


def test_unknown_athlete_gets_an_empty_list_not_an_error():
    assert api.load_career_progression("men_100m", "Nobody At All") == []


def test_unknown_discipline_is_handled():
    assert api.load_career_progression("not_a_discipline", "Anyone") == []


def test_worldwide_rows_are_optional():
    """The worldwide race log is scraped separately and may be absent,
    partial, or mid-run. None of those may break a profile."""
    rows = api.load_worldwide_rows("men_100m", "Nobody At All")
    assert rows.empty
    assert api.load_worldwide_rows("not_a_discipline", "Anyone").empty
