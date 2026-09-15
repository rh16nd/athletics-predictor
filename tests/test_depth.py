"""
Discipline vs discipline -- the depth index behind /api/depth and
/api/discipline/<key>.

The load-bearing decision here is what the comparison is built ON. The
obvious input is the model's probabilities, and they cannot be used across
disciplines: the target is top-three membership scored per athlete, so a
field's probabilities sum to no fixed total -- 31 to 320 across the 32 real
2026 fields. Ranking events by them would rank the model's per-event
confidence. WA's Results Score is scraped, uniform and cross-comparable, so
it is the only input. The first test pins that reasoning to a real number.
"""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import api  # noqa: E402


@pytest.fixture(scope="module")
def index():
    rows = api.build_depth_index()
    if not rows:
        pytest.skip("no predictions or toplists on disk")
    return rows


def test_probabilities_are_not_comparable_across_disciplines(index):
    """The reason the index is built on WA score instead. If this ever
    fails -- if the per-discipline totals converge on a constant -- the
    probabilities became a shared-out allocation and the choice is worth
    revisiting."""
    track, field = api.load_predictions()
    totals = [sum(a["prob"] for a in d["athletes"]) for d in track + field]
    assert max(totals) - min(totals) > 100


def test_every_discipline_is_ranked_and_ranks_are_dense(index):
    ranks = [r["spreadRank"] for r in index]
    assert ranks == list(range(1, len(index) + 1))


def test_the_index_is_ordered_tightest_first(index):
    spreads = [r["spread"] for r in index]
    assert spreads == sorted(spreads)


def test_spread_is_the_real_distance_between_best_and_weakest(index):
    for r in index:
        assert r["spread"] == r["bestScore"] - r["worstScore"]
        assert r["spread"] >= 0


def test_the_index_reports_how_much_of_each_field_it_scored(index):
    """A spread computed over 5 of 8 athletes is a different claim from one
    computed over all 8, so the count travels with the number."""
    for r in index:
        assert 2 <= r["scored"] <= r["fieldSize"]


def test_a_finalist_outside_the_world_top_100_still_counts_toward_spread():
    """The spread is looked up in the FULL toplist. Dropping a finalist who
    qualified on Diamond League points would remove the weakest athlete from
    the field and make a top-heavy discipline read as level."""
    full = api.load_season_scores()
    if full.empty:
        pytest.skip("no season toplists on disk")
    uniform = api.to_uniform_depth(full)
    deep = full[~full.index.isin(uniform.index)]
    if deep.empty:
        pytest.skip("no discipline was scraped past the uniform depth")
    disc_key = deep["discKey"].iloc[0]
    athletes = [{"name": deep["Competitor"].iloc[0], "prob": 1},
                {"name": full[full["discKey"] == disc_key]["Competitor"].iloc[0], "prob": 2}]
    scored = api._field_scores(full, disc_key, athletes)
    assert len(scored) == 2


# ---- the verdict ----

def test_verdict_splits_into_three_real_groups(index):
    labels = {api.depth_verdict(r["spreadRank"], len(index))["key"] for r in index}
    assert labels == {"level", "mixed", "topHeavy"}


def test_verdict_is_a_tercile_of_the_ranking():
    total = 32
    assert api.depth_verdict(1, total)["key"] == "level"
    assert api.depth_verdict(10, total)["key"] == "level"
    assert api.depth_verdict(11, total)["key"] == "mixed"
    assert api.depth_verdict(21, total)["key"] == "mixed"
    assert api.depth_verdict(22, total)["key"] == "topHeavy"
    assert api.depth_verdict(32, total)["key"] == "topHeavy"


def test_verdict_is_withheld_when_there_is_nothing_to_compare_against():
    assert api.depth_verdict(1, 2) is None


# ---- endpoints ----

@pytest.fixture(scope="module")
def client():
    return api.app.test_client()


def test_depth_endpoint_ranks_all_of_them(client, index):
    payload = client.get("/api/depth").get_json()
    assert payload["total"] == len(index)
    assert payload["toplistDepth"] == api.TOPLIST_DEPTH
    assert all(d["verdict"] for d in payload["disciplines"])


def test_discipline_endpoint_agrees_with_the_index(client, index):
    """The page's own verdict has to be the one the index gives it --
    recomputing depth per route is how two pages start disagreeing."""
    target = index[0]
    payload = client.get(f"/api/discipline/{target['discKey']}").get_json()
    assert payload["depth"]["spreadRank"] == target["spreadRank"]
    assert payload["depth"]["spread"] == target["spread"]
    assert payload["depth"]["of"] == len(index)


def test_discipline_endpoint_exposes_the_scores_behind_the_spread(client, index):
    """The spread is inspectable, not asserted: the per-athlete scores are
    returned so a reader can see where it comes from."""
    target = index[0]
    payload = client.get(f"/api/discipline/{target['discKey']}").get_json()
    scores = [s["score"] for s in payload["scores"]]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] - scores[-1] == target["spread"]


def test_unknown_discipline_is_a_404(client):
    assert client.get("/api/discipline/men_marathon").status_code == 404


def test_discipline_payload_is_json_serialisable(client, index):
    import json
    json.dumps(client.get(f"/api/discipline/{index[0]['discKey']}").get_json())


def test_an_event_with_no_final_gets_a_page_from_the_worlds_top_athletes(client, monkeypatch):
    """The hammer and the 10,000m (2026-09-15). The field is the world's top N
    on points, N being a Final's size in that kind of event. Its spread is set
    against the Finals' without joining them, the chance beside each athlete is
    the field model's from the Track and Field list, and no storyline reads a
    Diamond League Final the event never had."""
    points = [{"rank": i, "name": f"Thrower {i}", "nat": "POL", "mark": f"{80 - i}.00",
               "score": 1260 - 10 * i} for i in range(1, 9)]
    monkeypatch.setattr(api, "load_world_rankings", lambda: {"men_HT": {
        "points": points,
        "model": [{"name": "Thrower 2", "ratingPct": 71.0}, {"name": "Thrower 1", "ratingPct": 64.5}]}})
    monkeypatch.setattr(api, "load_season_scores", lambda year=None: pd.DataFrame([
        {"Competitor": p["name"], "Mark": p["mark"], "Results Score": p["score"], "Venue": "Somewhere",
         "Date": "01 JUN 2026", "Rank": p["rank"], "discKey": "men_HT", "indoor": False}
        for p in points]))
    monkeypatch.setattr(api, "build_depth_index", lambda year=None: [{"spread": s} for s in (20, 40, 80, 120)])
    monkeypatch.setattr(api, "build_discipline_trajectories", lambda disc_key, athletes: [])
    monkeypatch.setattr(api.athlete_analytics, "build_field_analysis", lambda *args: None)

    res = client.get("/api/discipline/men_HT")
    assert res.status_code == 200
    payload = res.get_json()
    assert (payload["fieldSource"], payload["modelKind"], payload["storylines"]) == ("toplist", "field", [])
    assert [a["name"] for a in payload["athletes"]] == [f"Thrower {i}" for i in range(1, 7)]
    depth = payload["depth"]
    # Six throwers from 1250 down to 1200: a 50-point spread, wider than two of
    # the four finals and tighter than the other two.
    assert (depth["fieldSize"], depth["spread"], depth["spreadRank"], depth["finalsWider"], depth["of"]) == \
        (6, 50, 3, 2, 4)
    assert depth["favouriteProb"] == 71.0
    assert {s["name"]: s["prob"] for s in payload["scores"]}["Thrower 3"] is None
