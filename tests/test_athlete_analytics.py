"""
Race-log analytics (src/athlete_analytics.py).

Mostly synthetic fixtures rather than the scraped log: data/worldwide/ is
gitignored, regenerable, and was still being written while these were
authored, so tests that assert on it would pass or fail depending on how
far a scrape had got.

The head-to-head symmetry test is the one that matters most. This module
replaced h2h_rates.csv as the source for the profile's rivals panel
because the two disagreed on the same page -- Kovacs vs Crouser read 1-2
from the old file and 5-23 from the log -- so "derived" now has to be
right, not just deeper.
"""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import athlete_analytics as aa  # noqa: E402


# ---- position parsing ----

@pytest.mark.parametrize("raw,place", [
    ("1", 1),        # historical toplist file
    ("1.", 1),       # worldwide scraper
    ("1.0", 1),      # after pandas reads the column as float
    ("12.", 12),
    ("2f1", 2),      # final, round-qualified
])
def test_finals_parse_to_a_place(raw, place):
    got, is_final = aa.parse_position(raw)
    assert (got, is_final) == (place, True)


@pytest.mark.parametrize("raw", ["1h1", "3sf2", "2q1", "4qf3"])
def test_heats_and_semis_are_not_finals(raw):
    """A win in a heat is not a win. The historical toplist file mixes
    rounds in; the scrapers only ever read races WA labels "Final"."""
    place, is_final = aa.parse_position(raw)
    assert is_final is False
    assert place is None


@pytest.mark.parametrize("raw", [None, "", "nan", float("nan")])
def test_missing_positions_are_missing_not_zero(raw):
    place, is_final = aa.parse_position(raw)
    assert place is None


# ---- mark parsing ----

@pytest.mark.parametrize("raw,expected", [
    ("9.79", 9.79), ("22.58m", 22.58), ("3:26.73", 206.73),
])
def test_marks_parse(raw, expected):
    assert aa.parse_mark_value(raw) == pytest.approx(expected)


@pytest.mark.parametrize("raw", ["NM", "NH", "DNS", "DNF", "DQ", "", None])
def test_non_marks_are_none(raw):
    assert aa.parse_mark_value(raw) is None


# ---- fixtures ----

def _log(rows):
    """rows: (competitor, mark, place, date, meeting, tier, year)"""
    df = pd.DataFrame(rows, columns=["Competitor", "Mark", "place", "date",
                                     "Meeting", "tier", "year"])
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = df["Mark"].map(aa.parse_mark_value)
    df["isFinal"] = True
    df["Venue"] = df["Meeting"]
    df["source"] = "test"
    return df


RACES = _log([
    ("A", "22.00m", 1, "2026-05-01", "Meet One",   "DL", 2026),
    ("B", "21.50m", 2, "2026-05-01", "Meet One",   "DL", 2026),
    ("C", "21.00m", 3, "2026-05-01", "Meet One",   "DL", 2026),
    ("A", "21.80m", 2, "2026-06-01", "Meet Two",   "A",  2026),
    ("B", "22.10m", 1, "2026-06-01", "Meet Two",   "A",  2026),
    ("A", "22.40m", 1, "2026-07-01", "Meet Three", "D",  2026),
    ("C", "21.20m", 4, "2026-07-01", "Meet Three", "D",  2026),
])


# ---- competition record ----

def test_record_counts_wins_and_podiums():
    rec = aa.competition_record(aa.athlete_rows(RACES, "A"))
    assert (rec["races"], rec["wins"], rec["podiums"]) == (3, 2, 3)
    assert rec["winRate"] == pytest.approx(66.7, abs=0.1)
    assert rec["podiumRate"] == 100.0
    assert rec["avgFinish"] == pytest.approx(1.33, abs=0.01)
    assert rec["bestFinish"] == 1


def test_record_breaks_down_by_competition_category():
    rec = aa.competition_record(aa.athlete_rows(RACES, "A"))
    by_tier = {t["tier"]: t for t in rec["byTier"]}
    assert by_tier["DL"]["wins"] == 1
    assert by_tier["A"]["wins"] == 0
    assert by_tier["D"]["wins"] == 1
    assert by_tier["DL"]["label"] == "Diamond League"


def test_top_tier_share_counts_only_the_documented_categories():
    """TOP_TIERS is DL + continental championships + Continental Tour Gold.
    A's three races are one DL, one Gold and one Challenger, so two of
    three count."""
    rec = aa.competition_record(aa.athlete_rows(RACES, "A"))
    assert rec["topTierRaces"] == 2
    assert rec["topTierShare"] == pytest.approx(66.7, abs=0.1)


def test_record_is_none_without_any_finishing_positions():
    rows = aa.athlete_rows(RACES, "A").copy()
    rows["place"] = None
    assert aa.competition_record(rows) is None


# ---- form ----

def test_form_withholds_consistency_below_three_marks():
    """A coefficient of variation from two races is noise with two decimal
    places on it."""
    two = _log([
        ("A", "22.00m", 1, "2026-05-01", "M1", "DL", 2026),
        ("A", "21.00m", 2, "2026-06-01", "M2", "DL", 2026),
    ])
    form = aa.form_by_season(aa.athlete_rows(two, "A"), is_field=True)
    assert form[0]["marks"] == 2
    assert form[0]["consistency"] is None


def test_form_reports_consistency_once_there_are_enough_marks():
    form = aa.form_by_season(aa.athlete_rows(RACES, "A"), is_field=True)
    assert form[0]["marks"] == 3
    assert form[0]["consistency"] is not None
    assert form[0]["consistency"] > 0


def test_form_does_not_publish_a_season_best():
    """load_career_progression owns that number and computes it from the
    toplist, which knows marks this log has never scraped. Publishing both
    put two different figures for one season on one page."""
    form = aa.form_by_season(aa.athlete_rows(RACES, "A"), is_field=True)
    assert "best" not in form[0]
    assert "bestLogged" in form[0]


def test_top3_average_takes_the_best_three_in_the_right_direction():
    # abs=0.001 because the module rounds to three decimals on the way out.
    field = aa.form_by_season(aa.athlete_rows(RACES, "A"), is_field=True)
    assert field[0]["top3Average"] == pytest.approx((22.00 + 21.80 + 22.40) / 3, abs=0.001)

    times = _log([
        ("T", "10.10", 1, "2026-05-01", "M1", "DL", 2026),
        ("T", "9.90", 1, "2026-06-01", "M2", "DL", 2026),
        ("T", "10.50", 3, "2026-07-01", "M3", "DL", 2026),
        ("T", "9.95", 2, "2026-08-01", "M4", "DL", 2026),
    ])
    track = aa.form_by_season(aa.athlete_rows(times, "T"), is_field=False)
    # Best three TIMES are the three fastest, not the three largest.
    assert track[0]["top3Average"] == pytest.approx((9.90 + 9.95 + 10.10) / 3, abs=0.001)


# ---- season shape ----

def test_season_shape_finds_the_month_of_the_best_mark():
    shape = aa.season_shape(aa.athlete_rows(RACES, "A"), 2026, is_field=True)
    assert shape["bestMonth"] == "Jul"     # 22.40m, the farthest
    assert shape["races"] == 3
    assert [m["month"] for m in shape["byMonth"]] == ["May", "Jun", "Jul"]


def test_season_shape_picks_the_fastest_for_a_track_event():
    times = _log([
        ("T", "10.20", 2, "2026-05-01", "M1", "DL", 2026),
        ("T", "9.95", 1, "2026-08-01", "M2", "DL", 2026),
    ])
    shape = aa.season_shape(aa.athlete_rows(times, "T"), 2026, is_field=False)
    assert shape["bestMonth"] == "Aug"


def test_season_shape_is_none_for_a_season_not_contested():
    assert aa.season_shape(aa.athlete_rows(RACES, "A"), 2019, is_field=True) is None


# ---- head to head ----

def test_head_to_head_only_counts_shared_races():
    h2h = {h["name"]: h for h in aa.head_to_head(RACES, "A")}
    # A met B twice (Meet One, Meet Two) and C twice (Meet One, Meet Three).
    assert h2h["B"]["meetings"] == 2
    assert h2h["B"]["wins"] == 1 and h2h["B"]["losses"] == 1
    assert h2h["C"]["meetings"] == 2
    assert h2h["C"]["wins"] == 2 and h2h["C"]["losses"] == 0


def test_head_to_head_is_symmetric():
    """A's wins over B must equal B's losses to A. The old h2h source could
    not be checked this way because both directions were scraped
    separately; a derived record has no excuse."""
    for left, right in [("A", "B"), ("A", "C"), ("B", "C")]:
        a_side = {h["name"]: h for h in aa.head_to_head(RACES, left)}
        b_side = {h["name"]: h for h in aa.head_to_head(RACES, right)}
        if right not in a_side:
            continue
        assert a_side[right]["wins"] == b_side[left]["losses"]
        assert a_side[right]["losses"] == b_side[left]["wins"]
        assert a_side[right]["meetings"] == b_side[left]["meetings"]


def test_head_to_head_can_be_filtered_to_named_opponents():
    only_b = aa.head_to_head(RACES, "A", opponents=["B"])
    assert [h["name"] for h in only_b] == ["B"]


def test_head_to_head_min_meetings_threshold():
    assert aa.head_to_head(RACES, "A", min_meetings=3) == []


def test_head_to_head_of_an_unknown_athlete_is_empty():
    assert aa.head_to_head(RACES, "Nobody") == []


# ---- loading ----

def test_missing_discipline_gives_an_empty_log_not_an_error():
    log = aa.load_race_log("not_a_discipline")
    assert log.empty


def test_build_analytics_returns_none_for_an_unknown_athlete():
    assert aa.build_analytics("men_100m", "Nobody At All", False) is None
