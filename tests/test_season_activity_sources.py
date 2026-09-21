"""An athlete page's season tiles read the race log and the athlete's World
Athletics season together (api.season_activity), so they agree with the season
chart beside them.

The bug: Puripol Boonson, third pick in the Asian Games 100m, read "0 races
this season" and a blank "last competed" above a chart of his two 2026 races.
The chart read World Athletics' own season; the tiles read only the race log,
which has none of his meetings. 535 linked pages disagreed with themselves the
same way (2026-09-21).

Synthetic profiles in a temporary folder throughout: data/athlete_profiles/ is
gitignored and regenerable.
"""
import json

import pandas as pd
import pytest

import api


def _profile_dir(tmp_path, results_by_event):
    blob = {"id": "555", "profile": {"resultsByYear": {"resultsByEvent": results_by_event}}}
    (tmp_path / "555.json").write_text(json.dumps(blob), encoding="utf-8")
    return str(tmp_path)


@pytest.fixture
def boonson(tmp_path, monkeypatch):
    folder = _profile_dir(tmp_path, [
        {"discipline": "100 Metres", "results": [
            {"date": "01 AUG 2026", "mark": "9.99"},
            {"date": "15 AUG 2026", "mark": "10.35"},
        ]},
        # A later race in another event counts for "last competed" ...
        {"discipline": "200 Metres", "results": [{"date": "20 AUG 2026", "mark": "20.40"}]},
        # ... and a DNS after it does not.
        {"discipline": "4x100 Metres Relay", "results": [{"date": "30 AUG 2026", "mark": "DNS"}]},
        # Nor does last season.
        {"discipline": "60 Metres", "results": [{"date": "01 FEB 2025", "mark": "6.60"}]},
    ])
    monkeypatch.setattr(api, "ATHLETE_PROFILES_DIR", folder)
    monkeypatch.setattr(api, "toplist_entry", lambda disc, name: (None, None, None))
    return "https://worldathletics.org/athletes/athlete=555"


def _log(*dates):
    return pd.DataFrame({"date": [pd.Timestamp(d) for d in dates]})


def test_last_competed_is_the_latest_race_in_any_event(boonson):
    assert api.wa_last_competed("men_100m", "Puripol BOONSON", boonson) == pd.Timestamp("2026-08-20")


def test_the_tiles_count_the_chart_s_races_when_the_race_log_has_none(boonson):
    races, last, days = api.season_activity(
        "men_100m", "Puripol BOONSON", _log(), api.MEETS_YEAR, 2, boonson,
    )
    assert races == 2
    assert last == pd.Timestamp("2026-08-20")
    assert days == (pd.Timestamp.today().normalize() - last).days


def test_the_race_log_wins_when_it_knows_more(boonson):
    log = _log("2026-05-01", "2026-06-01", "2026-09-01")
    races, last, _days = api.season_activity(
        "men_100m", "Puripol BOONSON", log, api.MEETS_YEAR, 2, boonson,
    )
    assert races == 3
    assert last == pd.Timestamp("2026-09-01")


def test_a_chart_from_an_earlier_season_is_not_this_season_s_races(boonson):
    races, _last, _days = api.season_activity(
        "men_100m", "Puripol BOONSON", _log(), api.MEETS_YEAR - 1, 6, boonson,
    )
    assert races == 0


def test_no_profile_and_no_races_leaves_the_tiles_blank(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "ATHLETE_PROFILES_DIR", str(tmp_path))
    monkeypatch.setattr(api, "toplist_entry", lambda disc, name: (None, None, None))
    assert api.season_activity("men_100m", "Nobody HERE", _log(), None, 0, None) == (0, None, None)
