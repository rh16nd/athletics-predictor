"""Unit/integration tests for api.py's build_athlete_profile() and
load_athlete_photo() -- flagged as untested in HANDOFF.md's Next Steps.

build_athlete_profile() is tested end-to-end against the same small,
checked-in fixtures test_fixtures_integration.py already uses
(tests/fixtures/outputs/predictions_latest.csv, tests/fixtures/raw/,
tests/fixtures/h2h/h2h_rates.csv) -- no network, no dependence on real
scraped data. load_athlete_photo() and get_photo_focus() both make real
HTTP calls in production (a GraphQL request and an image download,
respectively); every test here monkeypatches those calls out rather than
hitting the network, since a live World Athletics response isn't something
a unit test should depend on.
"""
import os

import pytest

import api
import dl_final_results_scraper as dlr

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


@pytest.fixture
def fixture_api(monkeypatch):
    monkeypatch.setattr(api, "OUTPUTS_DIR", os.path.join(FIXTURES_DIR, "outputs"))
    monkeypatch.setattr(api, "RAW_DIR", os.path.join(FIXTURES_DIR, "raw"))
    monkeypatch.setattr(api, "H2H_PATH", os.path.join(FIXTURES_DIR, "h2h", "h2h_rates.csv"))
    monkeypatch.setattr(api, "INJURY_FLAGS_PATH", os.path.join(FIXTURES_DIR, "nonexistent_injury_flags.json"))
    # No real photo/face-detection network calls from a unit test -- both
    # are already covered as pure/mocked concerns below and in
    # test_load_athlete_photo_*; keep build_athlete_profile's own tests
    # focused on the profile-assembly logic.
    monkeypatch.setattr(api, "load_athlete_photo", lambda url: None)
    return api


# ---- build_athlete_profile ----

def test_build_athlete_profile_unknown_discipline_returns_none(fixture_api):
    assert fixture_api.build_athlete_profile("not_a_real_discipline", "Alpha SPEEDY") is None


def test_build_athlete_profile_unknown_athlete_returns_none(fixture_api):
    assert fixture_api.build_athlete_profile("men_100m", "Nobody REAL") is None


def test_build_athlete_profile_returns_real_stats_and_history(fixture_api):
    profile = fixture_api.build_athlete_profile("men_100m", "Alpha SPEEDY")

    assert profile["name"] == "Alpha SPEEDY"
    assert profile["disc"] == "Men's 100m"
    assert profile["rank"] == 1
    assert profile["mark"] == pytest.approx(9.85)
    assert profile["prob"] == 45
    assert profile["photoUrl"] is None  # monkeypatched load_athlete_photo
    assert profile["photoFocus"] is None  # get_photo_focus(None) short-circuits, no network

    # Real per-meet history from tests/fixtures/raw/men_100m.csv -- no
    # men_100m_2026_meetings.csv fixture exists, so this exercises the
    # fallback to the athlete's own most recent *historical* season (2023,
    # the max year in the fixture) rather than the current-season path.
    assert profile["historyYear"] == 2023
    assert len(profile["history"]) == 1
    assert profile["history"][0]["mark"] == "9.85"

    # Case-insensitive real head-to-head vs. a top rival, per h2h_rates.csv
    # fixture (10 real meetings, Alpha wins 8).
    h2h = {m["opponent"]: m for m in profile["h2h"]}
    assert h2h["Beta MIDPACK"]["wins"] == 8
    assert h2h["Beta MIDPACK"]["losses"] == 2
    assert h2h["Beta MIDPACK"]["meetings"] == 10


def test_build_athlete_profile_injury_watch_flag_reflected(fixture_api):
    # Gamma MEDIUM's injury_watch=True is set directly in the fixture CSV;
    # with no real injury_flags.json (path points at a nonexistent file),
    # there's no matching evidence entry, so the flag is real but the
    # reason/url are honestly None rather than fabricated.
    profile = fixture_api.build_athlete_profile("men_100m", "Gamma MEDIUM")
    assert profile["injuryWatch"] is True
    assert profile["injuryReason"] is None
    assert profile["injuryUrl"] is None


def test_build_athlete_profile_field_event_uses_field_event_history(fixture_api):
    profile = fixture_api.build_athlete_profile("women_LJ", "Xena JUMPER")
    assert profile["disc"] == "Women's Long Jump"
    assert profile["mark"] == pytest.approx(7.05)
    assert profile["history"][0]["mark"] == "7.05m"


# ---- load_athlete_photo ----

def test_load_athlete_photo_resolves_common_url_format(monkeypatch):
    def fake_graphql(op, variables, query):
        assert variables == {"ids": [14536762]}
        return {"getAthleteActionPictureByIds": [{"id": 14536762, "primaryMediaId": "abc123"}]}
    monkeypatch.setattr(dlr, "graphql", fake_graphql)

    url = api.load_athlete_photo("https://worldathletics.org/athletes/athlete=14536762")
    assert url == "https://assets.aws.worldathletics.org/abc123"


def test_load_athlete_photo_resolves_country_slug_url_format(monkeypatch):
    """The second real WA profile-URL format (~8% of rows, seen for newer/
    junior/collegiate athletes) -- a real bug fix (2026-08-23) was widening
    the regex to also match this shape instead of silently returning None."""
    def fake_graphql(op, variables, query):
        assert variables == {"ids": [14707010]}
        return {"getAthleteActionPictureByIds": [{"id": 14707010, "primaryMediaId": "xyz789"}]}
    monkeypatch.setattr(dlr, "graphql", fake_graphql)

    url = api.load_athlete_photo("https://worldathletics.org/athletes/netherlands/femke-bol-14707010")
    assert url == "https://assets.aws.worldathletics.org/xyz789"


def test_load_athlete_photo_unmatched_url_returns_none_without_network(monkeypatch):
    calls = []
    monkeypatch.setattr(dlr, "graphql", lambda *a, **k: calls.append(1))
    assert api.load_athlete_photo("https://worldathletics.org/search/?q=whoever") is None
    assert calls == []  # never even attempted the network call


def test_load_athlete_photo_non_string_input_returns_none():
    assert api.load_athlete_photo(None) is None


def test_load_athlete_photo_no_photo_on_file_returns_none(monkeypatch):
    monkeypatch.setattr(dlr, "graphql", lambda *a, **k: {"getAthleteActionPictureByIds": [{"id": 1, "primaryMediaId": None}]})
    assert api.load_athlete_photo("https://worldathletics.org/athletes/athlete=1") is None


def test_load_athlete_photo_network_error_returns_none(monkeypatch):
    def raise_error(*a, **k):
        raise RuntimeError("network down")
    monkeypatch.setattr(dlr, "graphql", raise_error)
    assert api.load_athlete_photo("https://worldathletics.org/athletes/athlete=1") is None


# --- Near-miss athletes must not get a finalist profile (2026-08-25) ------
# run.py now writes near-miss athletes into predictions_latest.csv with
# dl_qualified = False and predicted_rank = None. build_athlete_profile used
# to reach them and raise "cannot convert float NaN to integer" on the rank,
# so the endpoint 500'd -- and the frontend only falls back to
# /api/athlete-status on 404, which made every near-miss profile an error
# page. It must return None so that fallback fires.

def test_near_miss_athlete_gets_no_finalist_profile(monkeypatch, tmp_path):
    import pandas as pd
    csv = tmp_path / "predictions_latest.csv"
    pd.DataFrame([
        {"discipline": "Men's 100m", "athlete_name": "Qualified Guy", "predicted_rank": 1,
         "season_best": "9.80", "win_probability": "30%", "dl_qualified": True,
         "nationality": "USA", "injury_watch": False, "profile_url": "https://x"},
        {"discipline": "Men's 100m", "athlete_name": "Noah LYLES", "predicted_rank": None,
         "season_best": "9.79", "win_probability": "8%", "dl_qualified": False,
         "nationality": "USA", "injury_watch": False, "profile_url": "https://x"},
    ]).to_csv(csv, index=False)
    monkeypatch.setattr(api, "OUTPUTS_DIR", str(tmp_path))

    assert api.build_athlete_profile("men_100m", "Noah LYLES") is None
    # the real field is unaffected
    assert api.build_athlete_profile("men_100m", "Qualified Guy") is not None
