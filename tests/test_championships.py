"""The championship registry's rules. src/championships.py says why each exists."""
import os

import pytest

import championships as ch


def test_the_registry_is_sound():
    assert ch.validate() == []


def test_every_championship_has_a_page_theme():
    """A tradition the user set: each championship page, and its box on the
    Results page, wears that competition's own colours."""
    assert all(c.get("theme") in ch.THEMES for c in ch.CHAMPIONSHIPS)


def test_a_championship_without_a_theme_is_rejected(monkeypatch):
    monkeypatch.setattr(ch, "CHAMPIONSHIPS", ch.CHAMPIONSHIPS + [{"id": "no-theme-2027", "theme": None}])
    assert any("no-theme-2027" in problem for problem in ch.validate())


def test_current_must_be_registered(monkeypatch):
    monkeypatch.setattr(ch, "CURRENT", "olympics-1896")
    assert any("olympics-1896" in problem for problem in ch.validate())


def test_the_ultimate_keeps_its_data_where_its_frozen_call_is():
    """That projection was frozen before the event and is graded on the Results
    page. Moving the file would break the one guarantee the page rests on."""
    frozen = ch.path(ch.get("ultimate-2026"), "predictions_prefinal.json")
    assert frozen.replace("\\", "/").endswith("data/ultimate/predictions_prefinal.json")
    assert os.path.exists(frozen)


def test_the_asian_games_carries_world_athletics_ids_and_dates():
    ag = ch.get("asian-games-2026")
    assert ag["competitionId"] == 7176091
    assert ag["area"] == "Asia"
    assert (ag["startDate"], ag["endDate"]) == ("2026-09-23", "2026-09-29")


def test_a_championship_without_a_data_directory_has_no_paths():
    assert ch.path(ch.get("dl-final-2026"), "event.json") is None


def test_unknown_ids_fail_loudly():
    with pytest.raises(KeyError):
        ch.get("olympics-1896")
