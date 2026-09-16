"""The head-to-head test's rule and its choice of finals (src/model_head_to_head.py)."""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import model_head_to_head as h2h  # noqa: E402


def group(dl, champ):
    return {"dl": {"medallists": dl}, "championship": {"medallists": champ}}


def test_the_rule_switches_only_when_the_championship_model_is_not_worse_on_both():
    assert h2h.decide({"championship": group(100, 100), "all": group(300, 300)})["verdict"] == "switch"
    # Better on championships, worse once Diamond League Finals are counted: the user's call.
    assert h2h.decide({"championship": group(100, 110), "all": group(300, 290)})["verdict"] == "userDecides"
    # Worse on the championships the user cares about most: keep what is there.
    assert h2h.decide({"championship": group(110, 100), "all": group(300, 320)})["verdict"] == "keep"


def test_only_finals_both_models_hold_are_compared_and_tour_meetings_are_not():
    labels = pd.DataFrame({
        "year": [2022, 2022, 2022, 2022, 2019],
        "discipline": ["men_100m", "men_100m", "men_100m", "men_100m", "men_100m"],
        "competition": ["Diamond League Final", "World Athletics Championships, Oregon 2022",
                        "European Athletics Championships", "Paavo Nurmi Games", "Diamond League Final"],
        "tier": ["dl_final", "global", "continental", "tour", "dl_final"]})
    field_finals = labels.iloc[[0, 1, 3, 4]].assign(tier=["dl_final", "global", "tour", "dl_final"])
    finals = h2h.universe(labels, field_finals)
    # The Europeans are missing from the championship model's history, the tour
    # meeting is outside the test, and 2019 is before the scored years.
    assert finals == [(2022, "men_100m", "Diamond League Final", "dlFinal"),
                      (2022, "men_100m", "World Athletics Championships, Oregon 2022", "championship")]


def test_a_final_a_model_could_not_call_counts_as_three_misses():
    rows = [{"dl": (2, True), "championship": (0, False), "points": (1, False),
             "dlCalled": True, "championshipCalled": False},
            {"dl": (1, False), "championship": (3, True), "points": (2, True),
             "dlCalled": True, "championshipCalled": True}]
    out = h2h.summarise(rows)
    assert out["possible"] == 6
    assert (out["dl"]["medallists"], out["championship"]["medallists"]) == (3, 3)
    assert out["notCalled"] == {"dl": 0, "championship": 1}
