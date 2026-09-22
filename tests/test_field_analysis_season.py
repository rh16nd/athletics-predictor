"""The event page's comparison table ("What separates them") reads each
contender's season the way their own page does (api.field_analysis).

The bug: the table counted this season's races, and took its top-3 average,
steadiness and best month, from the race log alone, while the athlete page
reads World Athletics' season as well. Josh Kerr read "0 races this season"
and no top-3 average on the 1500m page, beside a 3:27.62 in London on his own
page; 85 of the 100 rows checked on 2026-09-22 disagreed with the athlete's
page.

Synthetic race log and profiles throughout.
"""
import json

import pandas as pd
import pytest

import api
import athlete_analytics

YEAR = api.MEETS_YEAR


def _row(name, day, value, place, meeting):
    return {"Competitor": name, "date": pd.Timestamp(day), "year": pd.Timestamp(day).year,
            "value": value, "place": place, "Meeting": meeting, "source": "worldwide", "tier": "GL"}


@pytest.fixture
def world(tmp_path, monkeypatch):
    """Two contenders who met in last season's final. LEAD's race log stops
    last season and his World Athletics page holds this season; CHASE's log
    holds this season and he has no World Athletics page cached."""
    log = pd.DataFrame([
        _row("Lead ATHLETE", "2025-08-01", 208.00, 1, "Final"),
        _row("Chase ATHLETE", "2025-08-01", 209.00, 2, "Final"),
        _row("Chase ATHLETE", f"{YEAR}-05-10", 211.00, 1, "Meet A"),
        _row("Chase ATHLETE", f"{YEAR}-06-10", 210.00, 2, "Meet B"),
        _row("Chase ATHLETE", f"{YEAR}-07-10", 212.00, 3, "Meet C"),
    ])
    monkeypatch.setattr(athlete_analytics, "load_race_log", lambda disc: log)

    def profile(disc, results):
        blob = {"id": "555", "profile": {"resultsByYear": {"resultsByEvent": [
            {"discipline": disc, "results": results}]}}}
        (tmp_path / "555.json").write_text(json.dumps(blob), encoding="utf-8")

    monkeypatch.setattr(api, "ATHLETE_PROFILES_DIR", str(tmp_path))
    monkeypatch.setattr(api, "toplist_entry", lambda disc, name: (
        None, None, "https://worldathletics.org/athletes/athlete=555" if name == "Lead ATHLETE" else None))
    return profile


def _rows(disc):
    fa = api.field_analysis(disc, ["Lead ATHLETE", "Chase ATHLETE"])
    return {c["name"]: c for c in fa["comparison"]}


def test_the_table_counts_the_races_the_athlete_page_counts(world):
    world("1500 Metres", [
        {"date": f"18 JUL {YEAR}", "mark": "3:27.62"},
        {"date": f"01 AUG {YEAR}", "mark": "DNF"},
    ])
    lead = _rows("men_1500m")["Lead ATHLETE"]
    assert lead["seasonRaces"] == 1
    assert lead["top3Average"] == 207.62
    assert lead["bestMonth"] == "Jul"
    # All-time takes this season in: never fewer races than this season.
    assert lead["races"] == 2


def test_a_wind_aided_mark_is_a_race_but_not_a_best(world):
    world("100 Metres", [
        {"date": f"23 JUL {YEAR}", "mark": "9.88", "wind": "+0.3"},
        {"date": f"24 JUL {YEAR}", "mark": "9.76", "wind": "+2.1"},
        {"date": f"24 JUL {YEAR}", "mark": "9.79", "wind": "+0.8"},
    ])
    lead = _rows("men_100m")["Lead ATHLETE"]
    assert lead["seasonRaces"] == 3
    assert lead["top3Average"] == pytest.approx((9.79 + 9.88) / 2, abs=0.001)


def test_the_race_log_stands_when_it_knows_more(world):
    world("1500 Metres", [{"date": f"18 JUL {YEAR}", "mark": "3:27.62"}])
    chase = _rows("men_1500m")["Chase ATHLETE"]
    assert chase["seasonRaces"] == 3
    assert chase["top3Average"] == pytest.approx(211.0)
    assert chase["consistency"] is not None
    assert chase["bestMonth"] == "Jun"
    assert chase["races"] == 4


def test_the_season_chart_is_unchanged_by_the_shared_reader(world):
    world("100 Metres", [
        {"date": f"23 JUL {YEAR}", "mark": "9.88", "wind": "+0.3"},
        {"date": f"24 JUL {YEAR}", "mark": "9.76", "wind": "+2.1"},
    ])
    history, year, condensed, total = api.load_full_season_history("men_100m", "Lead ATHLETE")
    assert (year, condensed, total) == (YEAR, False, 2)
    assert set(history[0]) == {"date", "mark", "markValue", "venue", "resultsScore"}
    assert [h["mark"] for h in history] == ["9.88", "9.76"]
