"""Unit tests for the pooled final labels (src/final_labels.py) and the
date cut-off that makes them safe to train on (train_model.build_features).

Both things these pin fail SILENTLY. A name key that does not match produces a
training set with thousands of podiums and zero positive labels; a cut-off that
does not cut produces a model that scores an athlete on marks set after the
race it is predicting, and reports a lovely accuracy number for it. Neither
raises. Neither is visible in a diff.
"""
import os

import pandas as pd
import pytest

import final_labels as fl
import train_model as tm

LABELS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "data", "labels", "finals.csv")


def _labels():
    if not os.path.exists(LABELS_PATH):
        pytest.skip("data/labels/finals.csv not built yet (python src/final_labels.py)")
    return pd.read_csv(LABELS_PATH, parse_dates=["date"])


def test_name_key_matches_the_trainer_exactly():
    """The join key. These two functions live in different files and are only
    ever correct together -- when they disagreed, every one of 2,940 podium
    rows silently failed to match a feature row and the model trained on a
    label column that was all zeros."""
    for name in ["Emmanuel WANYONYI", "Sandra PERKOVIĆ", "Armand DUPLANTIS",
                 "Yaroslava MAHUCHIKH", "Kristjan ČEH", "  Noah LYLES  "]:
        assert fl.normalize_name(name) == tm.normalize_name(name), name


def test_every_final_has_a_complete_podium():
    """A final with two medallists on record teaches that whoever actually
    finished third did not medal, which is a fabricated negative. Incomplete
    podiums are dropped rather than kept."""
    labels = _labels()
    sizes = labels.groupby(["discipline", "year", "competition"]).size()
    assert set(sizes.unique()) == {3}, f"finals without exactly 3 podium rows: {sizes[sizes != 3]}"


def test_places_are_one_two_three():
    labels = _labels()
    per_final = labels.groupby(["discipline", "year", "competition"])["place"].apply(
        lambda s: sorted(s.tolist()))
    assert all(p == [1, 2, 3] for p in per_final), "a final without a clean 1/2/3"


def test_the_pool_is_wider_than_the_diamond_league():
    """The entire point of the exercise. If this ever falls back to
    dl_final only, the model has quietly returned to answering one
    competition's question."""
    labels = _labels()
    tiers = set(labels["tier"])
    assert {"dl_final", "global"} <= tiers, tiers
    finals = labels.groupby(["discipline", "year", "competition"]).ngroups
    dl_finals = labels[labels["tier"] == "dl_final"].groupby(
        ["discipline", "year", "competition"]).ngroups
    assert finals > 3 * dl_finals, f"{finals} finals against {dl_finals} DL ones"


def test_tier_of_reads_the_competition_name():
    assert fl.tier_of("The XXXIII Olympic Games") == "global"
    assert fl.tier_of("World Athletics Championships, Tokyo 2025") == "global"
    assert fl.tier_of("European Athletics Championships") == "continental"
    assert fl.tier_of("IAAF Continental Cup") == "continental"
    assert fl.tier_of("Diamond League Final") == "dl_final"
    assert fl.tier_of("60th Ostrava Golden Spike") == "tour"


def _one_athlete_season():
    """One athlete, four races across 2024, each slower than the last so the
    season best is unambiguous about WHEN it was set."""
    return pd.DataFrame({
        "athlete_name": ["A"] * 4,
        "Mark": ["10.20", "10.10", "9.90", "9.80"],
        "date": pd.to_datetime(["2024-05-01", "2024-06-01", "2024-08-01", "2024-08-20"]),
        "year": [2024] * 4,
        "age": [25.0] * 4,
        "country": ["KEN"] * 4,
    })


def test_a_cutoff_hides_every_mark_set_after_the_final():
    """The leakage test. A July final must not be scored on an August mark --
    that is not a prediction, it is the answer read backwards out of the
    future, and it is exactly what a hard-coded 1 September cut-off did to
    every championship held before September."""
    df = _one_athlete_season()
    july = tm.build_features(df, "men_100m", [(2024, "World Championships", "2024-07-01")])
    assert len(july) == 1
    row = july.iloc[0]
    # Only the May and June races existed on 1 July.
    assert row["season_best"] == 10.10, row["season_best"]
    assert row["meets_count"] == 2, row["meets_count"]


def test_without_a_cutoff_the_whole_season_is_visible():
    """The Diamond-League-only path, unchanged: one row per athlete-year over
    the full season. Kept so the model that is actually deployed can still be
    reproduced exactly."""
    df = _one_athlete_season()
    full = tm.build_features(df, "men_100m")
    row = full[full["year"] == 2024].iloc[0]
    assert row["season_best"] == 9.80
    assert row["meets_count"] == 4
    assert "competition" not in full.columns


def test_an_athlete_with_no_marks_before_the_final_gets_no_row():
    """Not a zero-filled row: an athlete who had not competed yet has no form
    to read, and inventing one is how a feature becomes a fabricated fact."""
    df = _one_athlete_season()
    early = tm.build_features(df, "men_100m", [(2024, "Some Final", "2024-04-01")])
    assert early.empty


def test_career_best_is_also_cut_at_the_final():
    """career_best and pb_gap are as capable of leaking as season_best -- a
    personal best set in August is not a fact available in July."""
    df = _one_athlete_season()
    july = tm.build_features(df, "men_100m", [(2024, "World Championships", "2024-07-01")])
    assert july.iloc[0]["career_best"] == 10.10


def test_pooled_actually_switches_the_dataset(monkeypatch):
    """Plumbing, because this exact wiring silently failed twice while it was
    being built: the branch was written into the wrong function and --pooled
    ran the Diamond-League-only path, reporting its numbers under a heading
    that said "pooled finals". Nothing raised. The accuracy was simply the old
    accuracy with a new label on it."""
    called = []
    monkeypatch.setattr(tm, "build_pooled_dataset", lambda tiers=None: called.append("pooled") or _stub())
    monkeypatch.setattr(tm, "build_labeled_dataset", lambda: called.append("dl") or _stub())
    monkeypatch.setattr(tm, "add_season_rank", lambda d: d)
    monkeypatch.setattr(tm, "add_new_features", lambda d: d)
    monkeypatch.setattr(tm, "add_h2h_features", lambda d: d)
    monkeypatch.setattr(tm, "mark_final_field", lambda d: d)
    monkeypatch.setattr(tm, "walk_forward_folds",
                        lambda *a, **k: (0, 0))

    tm.train_and_backtest(["season_best"], pooled=True)
    assert called == ["pooled"], called
    called.clear()
    tm.train_and_backtest(["season_best"], pooled=False)
    assert called == ["dl"], called


def _stub():
    return pd.DataFrame({
        "athlete_name": ["A", "B"], "discipline": ["men_100m"] * 2,
        "year": [2024, 2024], "season_best": [9.9, 10.1], "dl_top3": [1, 0],
        "dl_rank": [1, 0],
    })


# ---------------------------------------------------------------------------
# Tier names across BOTH eras. The training set was extended back to 2009 on
# 2026-09-08, and those seasons use the IAAF-era names: "IAAF World
# Championships" rather than "World Athletics Championships", and "Barcelona
# European Championships" rather than "European Athletics Championships".
#
# Neither matched the original patterns, so all five pre-2018 Worlds and three
# of the four pre-2018 Euros would have been filed as `tour` -- and `tour` is
# exactly what the shipped model's --tiers argument excludes. The labels would
# have been scraped, written, and then silently dropped.
#
# Every string below is real, from WA's calendar for 2009-2025.
# ---------------------------------------------------------------------------

import pytest


@pytest.mark.parametrize("competition", [
    "IAAF World Championships",                    # 2009, 2011, 2013, 2015
    "IAAF World Championships in Athletics",       # 2017, 2019
    "World Athletics Championships, Oregon 2022",  # 2022
    "The XXX Olympic Games",                       # 2012
    "The XXXIII Olympic Games",                    # 2024
])
def test_every_era_of_a_global_final_is_global(competition):
    assert fl.tier_of(competition) == "global"


@pytest.mark.parametrize("competition", [
    "Barcelona European Championships",   # 2010
    "Helsinki European Championships",    # 2012
    "Zürich European Championships",  # 2014
    "European Athletics Championships",   # 2016 onwards
    "IAAF Continental Cup",               # 2018
])
def test_every_era_of_a_continental_final_is_continental(competition):
    assert fl.tier_of(competition) == "continental"


@pytest.mark.parametrize("competition", [
    "Paavo Nurmi Games",
    "60th Ostrava Golden Spike",
    "World Athletics Continental Tour - Beijing",
    "USATF Golden Games",
    "Gyulai István Memorial - Hungarian Athletics Grand Prix",
])
def test_an_invitational_is_still_only_tour(competition):
    """The widened patterns must not start swallowing the tier they were
    widened past. "World Athletics Continental Tour - Beijing" is the one that
    comes closest, and it is not a World Championships."""
    assert fl.tier_of(competition) == "tour"
