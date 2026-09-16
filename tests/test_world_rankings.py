"""Unit tests for the Track/Field world-ranking lists (src/world_rankings.py).

Since 2026-09-17 every event's lists come from the championship model
(src/field_model.py): the world's top 20 by points, each with the model rating,
the chance of a top three as if the 20 met in one final. The Diamond League model's rating left these lists
after it named fewer medallists on the same past finals
(src/model_head_to_head.py), and its tests about meetings raced went with it.
"""
import json
import os

import pytest

import world_rankings as wr

ARTIFACT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "world_rankings.json")


def _artifact():
    if not os.path.exists(ARTIFACT):
        pytest.skip("data/world_rankings.json not built yet (python src/world_rankings.py)")
    with open(ARTIFACT, encoding="utf-8") as f:
        return json.load(f)


def test_every_event_gets_the_field_models_view(monkeypatch, tmp_path):
    """The field model gives the top 20 by points a rating each, the chance of
    a top three as if they met in one final, and ranks on it: the hammer and the
    10,000m since 2026-09-15, every event since 2026-09-17. The points list
    carries the same ratings, so the table shows them in either order."""
    (tmp_path / f"men_HT_{wr.YEAR}.csv").write_text(
        "Rank,Mark,WIND,Competitor,DOB,,Pos,,Venue,Date,Results Score,discipline,year,ProfileURL\n"
        "2,80.10,,Second THROWER,01 JAN 2000,POL,1,,Somewhere,01 JUN 2026,1190,men_HT,2026,"
        "https://worldathletics.org/athletes/athlete=2\n"
        "1,82.00,,First THROWER,01 JAN 1999,CAN,1,,Somewhere,01 JUL 2026,1250,men_HT,2026,"
        "https://worldathletics.org/athletes/athlete=1\n"
        "3,78.00,,Third THROWER,01 JAN 1998,HUN,1,,Somewhere,01 MAY 2026,1150,men_HT,2026,"
        "https://worldathletics.org/athletes/athlete=3\n",
        encoding="utf-8")
    monkeypatch.setattr(wr, "RAW_DIR", str(tmp_path))
    monkeypatch.setattr(wr, "races_on_record", lambda key, year: {})
    monkeypatch.setattr(wr.fm, "load_model", lambda: {"weights": [1.0]})
    seen = {}

    def serving_rows(key, athletes, season_path, cutoff, year):
        seen["field"] = [(a["name"], a["waId"]) for a in athletes]
        return "rows"

    monkeypatch.setattr(wr.fm, "serving_rows", serving_rows)
    monkeypatch.setattr(wr.fm, "field_chances", lambda model, rows, cutoff, disc_key=None: {
        "First THROWER": (0.97, 0.30), "Second THROWER": (0.94, 0.25), "Third THROWER": (0.99, 0.45)})

    out = wr.score_discipline("men_HT")
    assert (out["modelAvailable"], out["modelKind"], out["isField"]) == (True, "field", True)
    assert [r["name"] for r in out["points"]] == ["First THROWER", "Second THROWER", "Third THROWER"]
    # The field is the top by points, in that order, matched by id.
    assert seen["field"] == [("First THROWER", 1), ("Second THROWER", 2), ("Third THROWER", 3)]
    assert [(r["name"], r["rank"], r["ratingPct"]) for r in out["model"]] == [
        ("Third THROWER", 1, 99.0), ("First THROWER", 2, 97.0), ("Second THROWER", 3, 94.0)]
    assert [(r["name"], r["rank"], r["ratingPct"]) for r in out["points"]] == [
        ("First THROWER", 1, 97.0), ("Second THROWER", 2, 94.0), ("Third THROWER", 3, 99.0)]
    assert not any(k in row for row in out["model"] + out["points"] for k in ("winPct", "prob", "modelRank"))

    monkeypatch.setattr(wr.fm, "load_model", lambda: None)
    out = wr.score_discipline("men_HT")
    assert (out["modelAvailable"], out["modelKind"], out["model"]) == (False, None, [])
    assert all(r["ratingPct"] is None for r in out["points"])


def test_a_field_model_that_reads_races_gets_the_top_twentys_seasons_read_the_way_it_was_chosen(monkeypatch, tmp_path):
    (tmp_path / f"men_HT_{wr.YEAR}.csv").write_text(
        "Rank,Mark,WIND,Competitor,DOB,,Pos,,Venue,Date,Results Score,discipline,year,ProfileURL\n"
        "1,82.00,,First THROWER,01 JAN 1999,CAN,1,,Somewhere,01 JUL 2026,1250,men_HT,2026,"
        "https://worldathletics.org/athletes/athlete=1\n"
        "2,80.10,,Second THROWER,01 JAN 2000,POL,1,,Somewhere,01 JUN 2026,1190,men_HT,2026,"
        "https://worldathletics.org/athletes/athlete=2\n",
        encoding="utf-8")
    monkeypatch.setattr(wr, "RAW_DIR", str(tmp_path))
    monkeypatch.setattr(wr, "races_on_record", lambda key, year: {})
    fetched, seen = {}, {}

    def current_races(athletes, refresh=False):
        fetched["ids"] = [a["waId"] for a in athletes]
        return {"1": []}

    def serving_rows(key, athletes, season_path, cutoff, year, races=None, race_how=None):
        seen.update(races=races, race_how=race_how)
        return "rows"

    monkeypatch.setattr(wr, "current_races", current_races)
    monkeypatch.setattr(wr.fm, "serving_rows", serving_rows)
    monkeypatch.setattr(wr.fm, "field_chances", lambda model, rows, cutoff, disc_key=None: {
        "First THROWER": (0.9, 0.6), "Second THROWER": (0.8, 0.4)})
    wr.field_model_discipline("men_HT", model={"features": wr.fm.FEATURES_V2, "experiment": "v2_form_best_5"})
    assert fetched["ids"] == [1, 2]
    assert seen == {"races": {"1": []}, "race_how": {"form_marks": 5}}


def test_every_shipped_event_is_the_field_models_and_both_lists_agree():
    """One model on every Track and Field list. An event left on another model
    would show a rating that disagrees with the event page, the athlete page
    and the dashboard, which all read the same one."""
    data = _artifact()
    assert data, "no events in the shipped rankings"
    for key, disc in data.items():
        assert disc["modelKind"] == "field", f"{key} is {disc['modelKind']}"
        assert disc["modelAvailable"] is True, key
        ratings = [r["ratingPct"] for r in disc["model"]]
        assert ratings == sorted(ratings, reverse=True), key
        assert 299 <= sum(ratings) <= 301, (key, sum(ratings))
        rating = {r["name"]: r["ratingPct"] for r in disc["model"]}
        assert set(rating) <= {r["name"] for r in disc["points"]}, key
        for row in disc["points"]:
            assert row["ratingPct"] == rating.get(row["name"]), f"{key}/{row['name']}"
        for row in disc["model"] + disc["points"]:
            assert not {"modelRank", "winPct", "dlRaces"} & set(row), f"{key}/{row['name']}"
