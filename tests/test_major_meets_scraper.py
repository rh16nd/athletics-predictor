"""
Pins which meetings major_meets_scraper collects out of each World Athletics
calendar group, against real calendar shapes rather than the network.

The case that matters is 2018. There was no senior outdoor World
Championships that year -- they run 2019, 2022, 2023, 2025 in this window --
yet the scraper collected two meetings from the World Athletics Series group
anyway, for the whole life of the file: an "IAAF World Half Marathon
Championships" and an "IAAF Continental Cup". Both slipped a name-exclusion
list because neither name contains any of its keywords, and the Continental
Cup put 220 rows of a continental TEAM competition into the training data
under a docstring promising World Championships.

The fix reads WA's own rankingCategory ("OW") instead of guessing from the
name, the same way the Continental Tour filter already reads "A". These
tests exist so the keyword list cannot come back.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import major_meets_scraper as mm  # noqa: E402


def _series(monkeypatch, meetings):
    """Run find_year_meetings with only the World Athletics Series group
    populated, so the assertions are about that group's filter alone."""

    def fake_find(year, group_id):
        return meetings if group_id == mm.WORLD_ATHLETICS_SERIES else []

    monkeypatch.setattr(mm, "find_meetings", fake_find)
    return [m for m, _tag in mm.find_year_meetings(2018)]


# The real 2018 World Athletics Series group, as the calendar returns it.
# Every one of these has hasApiResults true, so `hasApiResults` filtering
# does not save us here -- the rankingCategory is the only thing that
# separates them.
SERIES_2018 = [
    {"name": "IAAF Continental Cup", "rankingCategory": "GW"},
    {"name": "IAAF World U20 Championships", "rankingCategory": "C"},
    {"name": "IAAF World Race Walking Team Championships", "rankingCategory": "GW"},
    {"name": "IAAF World Half Marathon Championships", "rankingCategory": "GW"},
    {"name": "IAAF World Indoor Championships", "rankingCategory": "GW"},
]

SERIES_2023 = [
    {"name": "World Athletics Road Running Championships", "rankingCategory": "GW"},
    {"name": "World Athletics Championships, Budapest 2023", "rankingCategory": "OW"},
    {"name": "44th World Athletics Cross Country Championships", "rankingCategory": "GW"},
]


def test_a_year_with_no_senior_outdoor_worlds_collects_none(monkeypatch):
    """2018 had no senior outdoor World Championships. The scraper must
    agree, rather than settling for the nearest thing with 'World' in it."""
    assert _series(monkeypatch, SERIES_2018) == []


def test_the_continental_cup_is_not_a_world_championships(monkeypatch):
    """Named separately because this is the one that actually reached the
    training data -- 220 rows, invisible in the code and obvious in the
    calendar."""
    kept = _series(monkeypatch, SERIES_2018)
    assert not any("Continental Cup" in m["name"] for m in kept)


def test_a_road_championships_is_not_collected_by_the_track_scraper(monkeypatch):
    kept = _series(monkeypatch, SERIES_2018)
    assert not any("Half Marathon" in m["name"] for m in kept)


def test_the_real_worlds_is_still_collected(monkeypatch):
    """The filter has to be narrow without being empty."""
    kept = _series(monkeypatch, SERIES_2023)
    assert [m["name"] for m in kept] == ["World Athletics Championships, Budapest 2023"]


def test_a_name_containing_world_championships_is_not_enough(monkeypatch):
    """The whole failure mode, stated directly: matching on the words is not
    matching on the thing. Both of these read as a World Championships and
    neither is the senior outdoor one."""
    decoys = [
        {"name": "World Athletics Indoor Championships", "rankingCategory": "GW"},
        {"name": "World Athletics U20 Championships", "rankingCategory": "C"},
    ]
    assert _series(monkeypatch, decoys) == []


def test_the_continental_tour_filter_reads_the_same_kind_of_field():
    """Guards the symmetry the fix relies on: both filters read WA's own
    classification rather than the meeting name."""
    assert mm.SERIES_RANKING_CATEGORY == "OW"
