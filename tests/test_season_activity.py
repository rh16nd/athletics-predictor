"""Unit tests for src/season_activity.py -- the "meets on record" count that
replaced the Diamond League meeting count on the Track and Field tables.

The number it replaced was true and misleading at the same time: the Diamond
League contests each discipline at only a few of its meetings, so a 0 read as
"did not run" when it meant "did not run a Diamond League one". Rai Benjamin
was third in the world in the 400m hurdles on a 46.67 at Lausanne, with 0
Diamond League meetings, because the men's 400mH was not on that meeting's
Diamond League programme. Both numbers were correct; together they lied.
"""
import os

import pandas as pd
import pytest

import season_activity as sa


@pytest.fixture
def sources(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    wide = tmp_path / "worldwide"
    raw.mkdir()
    wide.mkdir()
    monkeypatch.setattr(sa, "RAW_DIR", str(raw))
    monkeypatch.setattr(sa, "WORLDWIDE_DIR", str(wide))
    return raw, wide


def _csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


def test_counts_distinct_meetings_across_all_three_sources(sources):
    raw, wide = sources
    _csv(raw / "men_400h_2026.csv", [
        {"Competitor": "Rai BENJAMIN", "Date": "21 AUG 2026", "Venue": "Lausanne"},
    ])
    _csv(raw / "men_400h_2026_meetings.csv", [
        {"Competitor": "Karsten WARHOLM", "Date": "12 JUN 2026", "Venue": "Oslo"},
    ])
    _csv(wide / "men_400h.csv", [
        {"Competitor": "Karsten WARHOLM", "Date": "08 AUG 2026", "Venue": "ISTAF", "year": 2026},
    ])
    counts = sa.races_on_record("men_400h", 2026)
    assert counts["RAI BENJAMIN"] == 1
    assert counts["KARSTEN WARHOLM"] == 2


def test_the_same_race_in_two_files_counts_once(sources):
    """The toplist's season-best row and the per-meeting log describe the same
    afternoon. Counting it twice would inflate exactly the athletes who have
    the least racing to show."""
    raw, _ = sources
    row = {"Competitor": "A", "Date": "12 JUN 2026", "Venue": "Oslo"}
    _csv(raw / "men_800m_2026.csv", [row])
    _csv(raw / "men_800m_2026_meetings.csv", [row])
    assert sa.races_on_record("men_800m", 2026)["A"] == 1


def test_a_row_with_no_date_is_dropped_not_counted(sources):
    """Without a date it cannot be de-duplicated against the same race in
    another file, so counting it risks double-counting."""
    raw, _ = sources
    _csv(raw / "men_800m_2026.csv", [
        {"Competitor": "A", "Date": "12 JUN 2026", "Venue": "Oslo"},
        {"Competitor": "A", "Date": "", "Venue": "Somewhere"},
    ])
    assert sa.races_on_record("men_800m", 2026)["A"] == 1


def test_other_seasons_in_the_worldwide_log_are_ignored(sources):
    raw, wide = sources
    _csv(raw / "men_800m_2026.csv", [{"Competitor": "A", "Date": "12 JUN 2026", "Venue": "Oslo"}])
    _csv(wide / "men_800m.csv", [
        {"Competitor": "A", "Date": "01 JUN 2025", "Venue": "Rome", "year": 2025},
        {"Competitor": "A", "Date": "03 JUL 2026", "Venue": "Paris", "year": 2026},
    ])
    assert sa.races_on_record("men_800m", 2026)["A"] == 2


def test_a_discipline_with_no_files_returns_nothing_rather_than_zeros(sources):
    """An empty dict leaves the UI showing an em dash. A dict of zeros would
    claim every athlete sat the season out."""
    assert sa.races_on_record("men_800m", 2026) == {}


def test_the_shipped_rankings_carry_the_count():
    """Guards the column itself: world_rankings.py has to put the number in
    the payload or the table silently falls back to em dashes."""
    import io as _io
    import json
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "world_rankings.json")
    if not os.path.exists(path):
        pytest.skip("data/world_rankings.json not built yet")
    data = json.load(_io.open(path, encoding="utf-8"))
    for key, disc in data.items():
        for view in ("model", "points"):
            for row in disc[view]:
                assert "racesOnRecord" in row, f"{key}/{view}/{row['name']}"
