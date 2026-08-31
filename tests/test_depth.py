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
