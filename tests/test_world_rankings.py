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


def test_a_discipline_the_model_has_never_seen_is_ranked_on_points_alone(monkeypatch, tmp_path):
    """The hammer and the 10,000m, added 2026-09-14. A rating for them would be
    the model scoring its own defaults, so the model is never run on them."""
    (tmp_path / f"men_HT_{wr.YEAR}.csv").write_text(
        "Rank,Mark,WIND,Competitor,DOB,,Pos,,Venue,Date,Results Score,discipline,year,ProfileURL\n"
        "2,80.10,,Second THROWER,01 JAN 2000,POL,1,,Somewhere,01 JUN 2026,1190,men_HT,2026,u2\n"
        "1,82.00,,First THROWER,01 JAN 1999,CAN,1,,Somewhere,01 JUL 2026,1250,men_HT,2026,u1\n",
        encoding="utf-8")
    monkeypatch.setattr(wr, "RAW_DIR", str(tmp_path))
    monkeypatch.setattr(wr, "races_on_record", lambda key, year: {})
    monkeypatch.setattr(wr, "build_2026_features", lambda *a, **k: pytest.fail("the model path ran"))

    out = wr.score_discipline("men_HT")
    assert (out["modelAvailable"], out["model"], out["isField"]) == (False, [], True)
    assert [r["name"] for r in out["points"]] == ["First THROWER", "Second THROWER"]
    assert all(r["ratingPct"] is None and r["dlRaces"] is None for r in out["points"])


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
