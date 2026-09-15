"""Unit tests for the Track/Field world-ranking lists (src/world_rankings.py).

These pin one specific piece of honesty. The "model" ordering is NOT a ranking
of who is strongest in the world, however much a top-20 table looks like one:
the model was trained on Diamond League Final podiums and every form feature it
reads comes from the circuit, so an athlete who skipped it rates near zero no
matter what they have thrown or run. The UI now says so, and ships each
athlete's Diamond League meeting count in the same row -- so the count, and the
gap it explains, are what the tests here defend.
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


def test_dl_races_is_none_without_a_per_meeting_log(monkeypatch, tmp_path):
    """Without a meetings file, feature_builder falls back to the toplist's row
    count -- which is structurally 1 for every athlete alive. Shipping that as
    "1 Diamond League meeting" would be a fabricated number in the one column
    added to stop numbers being mysterious."""
    monkeypatch.setattr(wr, "RAW_DIR", str(tmp_path))
    assert wr.has_meetings_log("men_800m") is False


def test_dl_races_is_counted_when_the_log_exists(monkeypatch, tmp_path):
    (tmp_path / f"men_800m_{wr.YEAR}_meetings.csv").write_text("Competitor,Mark\n", encoding="utf-8")
    monkeypatch.setattr(wr, "RAW_DIR", str(tmp_path))
    assert wr.has_meetings_log("men_800m") is True


def test_an_event_the_diamond_league_model_has_never_seen_gets_the_field_models_view(monkeypatch, tmp_path):
    """The hammer and the 10,000m. The Diamond League model is never run on them
    (added 2026-09-14): it would score its own defaults. Since 2026-09-15 the
    field model gives the top 20 by points a podium chance each, as if they met
    in one final, and ranks on it."""
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
    monkeypatch.setattr(wr, "build_2026_features", lambda *a, **k: pytest.fail("the Diamond League model ran"))
    monkeypatch.setattr(wr.fm, "load_model", lambda: {"weights": [1.0]})
    seen = {}

    def serving_rows(key, athletes, season_path, cutoff, year):
        seen["field"] = [(a["name"], a["waId"]) for a in athletes]
        return "rows"

    monkeypatch.setattr(wr.fm, "serving_rows", serving_rows)
    monkeypatch.setattr(wr.fm, "field_chances", lambda model, rows, cutoff: {
        "First THROWER": (0.97, 0.30), "Second THROWER": (0.94, 0.25), "Third THROWER": (0.99, 0.45)})

    out = wr.score_discipline("men_HT")
    assert (out["modelAvailable"], out["modelKind"], out["isField"]) == (True, "field", True)
    assert [r["name"] for r in out["points"]] == ["First THROWER", "Second THROWER", "Third THROWER"]
    # The field is the top by points, in that order, matched by id.
    assert seen["field"] == [("First THROWER", 1), ("Second THROWER", 2), ("Third THROWER", 3)]
    assert [(r["name"], r["ratingPct"], r["winPct"]) for r in out["model"]] == [
        ("Third THROWER", 99.0, 45.0), ("First THROWER", 97.0, 30.0), ("Second THROWER", 94.0, 25.0)]
    assert all(r["dlRaces"] is None for r in out["model"] + out["points"])

    monkeypatch.setattr(wr.fm, "load_model", lambda: None)
    out = wr.score_discipline("men_HT")
    assert (out["modelAvailable"], out["modelKind"], out["model"]) == (False, None, [])


def test_every_shipped_row_carries_a_meeting_count():
    """The column is the whole explanation for a low rating. A refresh that
    quietly drops it leaves the ratings looking like verdicts again."""
    data = _artifact()
    for key, disc in data.items():
        for view in ("model", "points"):
            for row in disc[view]:
                assert "dlRaces" in row, f"{key}/{view}/{row['name']} has no dlRaces"
                assert row["dlRaces"] is None or row["dlRaces"] >= 0


def test_the_low_ratings_really_are_the_athletes_who_skipped_the_circuit():
    """The claim the UI copy makes, measured on the shipped data.

    If this ever fails the model has stopped penalising low Diamond League
    exposure -- good news, but it means the subtitle, the tooltip and the
    meetings column are now telling readers something untrue, so fix the copy
    rather than the assertion. Measured 2026-09-07: 1.2% mean rating at zero
    meetings against 17.4% at four, on near-identical mean WA scores."""
    data = _artifact()
    absent, regulars = [], []
    for disc in data.values():
        for row in disc["model"]:
            if row["dlRaces"] is None:
                continue
            (absent if row["dlRaces"] == 0 else regulars if row["dlRaces"] >= 4 else []).append(
                row["ratingPct"]
            )
    if not absent or not regulars:
        pytest.skip("no athletes at one end of the exposure range in this refresh")
    assert sum(absent) / len(absent) < sum(regulars) / len(regulars)
