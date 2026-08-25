"""Tests for the qualification race -- api.py's qualification_race() and
meetings_remaining() (HANDOFF item 0l).

This is arithmetic, not a model, and the whole value of it is that its two
verdicts are certain: "out" means the athlete cannot finish inside the cut
even by winning everything left, and "safe" means they cannot be displaced
even if they never score again. These tests pin exactly those boundaries,
including the off-by-one either side of them, because a wrong verdict here
reads as a fact about a real athlete's season.

A Diamond League meeting scores 8-7-6-5-4-3-2-1, so `max_points=8` matches
the real scale; the smaller values used below are just easier to reason
about in a fixture.
"""
from datetime import date

import api


def standings(*points, limit=3, meetings_left=1, max_points=8):
    rows = [{"rank": i + 1, "name": f"A{i + 1}", "country": "USA",
             "events": 1, "points": p} for i, p in enumerate(points)]
    return api.qualification_race(rows, limit, meetings_left, max_points=max_points)


def status_of(race, name):
    return next(r["status"] for r in race["rows"] if r["name"] == name)


# ---- the cut line and the gap to it ----

def test_cut_points_is_the_score_of_the_last_qualifying_place():
    race = standings(20, 15, 12, 9, 4, limit=3)
    assert race["cutPoints"] == 12


def test_gap_is_positive_below_the_line_and_negative_above_it():
    race = standings(20, 15, 12, 9, limit=3)
    gaps = {r["name"]: r["gap"] for r in race["rows"]}
    assert gaps["A1"] == -8   # eight points clear of the cut
    assert gaps["A3"] == 0    # on the line
    assert gaps["A4"] == 3    # three points short


def test_a_thin_table_has_no_cut_line_to_measure_against():
    """Fewer athletes than places means nobody is on the wrong side of a
    line, so a gap would be a made-up number."""
    race = standings(20, 15, limit=3)
    assert race["cutPoints"] is None
    assert all(r["gap"] is None for r in race["rows"])


# ---- "out" is only said when it is arithmetically certain ----

def test_an_athlete_who_cannot_reach_the_places_ahead_is_out():
    # A4 tops out at 3 + 8 = 11; three athletes already hold more than that.
    race = standings(20, 15, 12, 3, limit=3)
    assert status_of(race, "A4") == "out"


def test_an_athlete_who_can_still_reach_the_cut_is_only_chasing():
    # A4 tops out at 5 + 8 = 13, above A3's 12, so the last place is live.
    race = standings(20, 15, 12, 5, limit=3)
    assert status_of(race, "A4") == "chasing"


def test_a_tie_at_the_ceiling_is_not_called_out():
    """A4 can reach exactly A3's total. World Athletics' tie-break isn't in
    this data, so the honest answer is "still alive", not "eliminated"."""
    race = standings(20, 15, 12, 4, limit=3)
    assert status_of(race, "A4") == "chasing"


# ---- "safe" is only said when it is arithmetically certain ----

def test_an_athlete_nobody_can_reach_is_safe():
    race = standings(30, 5, 4, 3, limit=3)
    assert status_of(race, "A1") == "safe"


def test_an_athlete_who_could_be_displaced_is_only_in():
    # Three rivals can each reach A1's 10, which is enough to push them out.
    race = standings(10, 9, 8, 7, limit=3)
    assert status_of(race, "A1") == "in"


def test_being_reachable_by_fewer_rivals_than_there_are_places_is_still_safe():
    """Two rivals can pass A1, but there are three places -- so passing
    them costs A1 nothing."""
    race = standings(10, 9, 8, 1, limit=3)
    assert status_of(race, "A1") == "safe"


# ---- with no meetings left the table is simply the result ----

def test_the_standings_settle_once_no_meetings_remain():
    race = standings(20, 15, 12, 9, 4, limit=3, meetings_left=0)
    assert [r["status"] for r in race["rows"]] == ["safe", "safe", "safe", "out", "out"]


def test_an_athlete_with_no_points_recorded_gets_no_verdict():
    rows = [{"rank": 1, "name": "A1", "points": 20},
            {"rank": 2, "name": "A2", "points": None}]
    race = api.qualification_race(rows, 3, 1)
    assert status_of(race, "A2") == "unknown"
    assert race["rows"][1]["gap"] is None


# ---- meetings_remaining ----

MEETS = [
    {"n": 1, "date": "08 May", "city": "Doha"},
    {"n": 2, "date": "21 Aug", "city": "Lausanne"},
    {"n": 3, "date": "27 Aug", "city": "Zürich"},
    {"n": 4, "date": "04 Sep", "city": "Brussels — Final"},
]


def test_only_meetings_still_to_come_count():
    assert api.meetings_remaining(MEETS, today=date(2026, 8, 25)) == 1


def test_the_final_itself_is_not_a_qualifying_meeting():
    """Nothing is left to qualify through once Zürich is run -- the Final is
    what qualification is FOR, so counting it would always overstate by one."""
    assert api.meetings_remaining(MEETS, today=date(2026, 8, 28)) == 0


def test_a_full_season_ahead_counts_every_scoring_meeting():
    assert api.meetings_remaining(MEETS, today=date(2026, 1, 1)) == 3


# ---- standings_position / points_cut_reason ----

def _detail_env(monkeypatch, rows, limit=8):
    monkeypatch.setattr(api, "load_standings_detail", lambda: {"disciplines": {"men_100m": {
        "qualLimit": limit,
        "standings": [{"rank": r, "name": n, "country": "USA", "events": 2, "points": p}
                      for r, n, p in rows],
    }}})


ROWS = [(i + 1, f"A{i + 1}", 30 - i) for i in range(8)] + [(9, "Noah LYLES", 15)]


def test_standings_position_finds_an_athlete_below_the_cut(monkeypatch):
    _detail_env(monkeypatch, ROWS)
    pos = api.standings_position("men_100m", "Noah LYLES")
    assert pos["rank"] == 9 and pos["points"] == 15
    assert pos["qualLimit"] == 8
    assert pos["cutPoints"] == 23        # the 8th athlete's total
    assert pos["gap"] == 8


def test_standings_position_is_case_insensitive(monkeypatch):
    """Names arrive from a URL path, so the casing is whatever was typed."""
    _detail_env(monkeypatch, ROWS)
    assert api.standings_position("men_100m", "noah lyles")["rank"] == 9


def test_standings_position_is_none_for_someone_with_no_points(monkeypatch):
    """The distinction the whole fix rests on: absent from the table means
    genuinely no Diamond League points, and that is a different sentence
    from "below the cut"."""
    _detail_env(monkeypatch, ROWS)
    assert api.standings_position("men_100m", "Bayanda WALAZA") is None


def test_standings_position_is_none_before_the_scraper_has_run(monkeypatch):
    monkeypatch.setattr(api, "load_standings_detail", lambda: {})
    assert api.standings_position("men_100m", "Noah LYLES") is None


def test_the_reason_states_the_points_and_the_gap():
    reason = api.points_cut_reason("Men's 100m", {
        "rank": 9, "points": 15, "gap": 2, "qualLimit": 8, "status": "chasing"})
    assert "9th" in reason and "15 points" in reason and "2 points short" in reason


def test_one_point_short_is_not_pluralised():
    reason = api.points_cut_reason("Men's 100m", {
        "rank": 9, "points": 16, "gap": 1, "qualLimit": 8, "status": "chasing"})
    assert "1 point short" in reason


def test_an_eliminated_athlete_is_not_told_they_are_a_few_points_short():
    """A gap is only worth quoting while it can still be closed."""
    reason = api.points_cut_reason("Men's 100m", {
        "rank": 18, "points": 3, "gap": 14, "qualLimit": 8, "status": "out"})
    assert "short of the cut" not in reason
    assert "caught up" in reason


def test_level_on_points_is_reported_as_a_tie_break_not_a_gap():
    reason = api.points_cut_reason("Men's 100m", {
        "rank": 9, "points": 17, "gap": 0, "qualLimit": 8, "status": "chasing"})
    assert "tie-break" in reason


def test_ordinal_handles_the_teens():
    assert [api.ordinal(n) for n in (1, 2, 3, 4, 11, 12, 13, 21, 22, 23)] == [
        "1st", "2nd", "3rd", "4th", "11th", "12th", "13th", "21st", "22nd", "23rd"]
