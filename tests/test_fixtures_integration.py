"""Integration tests for build_labeled_dataset()/train_and_backtest() (train_model.py)
and load_predictions() (api.py) -- the three functions HANDOFF.md's Next Steps
flagged as untested because they normally read real data/raw/*.csv,
data/dl_final_results.csv, data/h2h/h2h_rates.csv, and outputs/predictions_latest.csv
straight off disk. Every path they read is monkeypatched to tests/fixtures/ instead,
so these run against small, controlled, checked-in data -- no network, no
dependence on whatever the real scraped files currently contain.

Fixture design (tests/fixtures/raw/{men_100m,women_LJ}.csv,
tests/fixtures/dl_final_results.csv): 5 athletes per discipline across
2021-2023, season-best strictly improving each year, with the same 3
athletes finishing top-3 at the (fixture) DL Final every year in men_100m
and women_LJ alike -- one track discipline, one field discipline, so both
of build_features()'s is_track branches get real end-to-end coverage.
men_100m's 2023 3rd place is deliberately given to "Delta OUTSIDER", an
athlete absent from the toplist fixture entirely -- the same "real athlete
missing from the toplist" shape documented in HANDOFF.md's Known
Limitations (Ngetich/Mabry), to confirm build_labeled_dataset() handles an
unmatched label without crashing rather than only ever being exercised by
real, occasionally-messy scraped data.
"""
import os

import pandas as pd
import pytest

import api
import train_model as tm

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")

FEATURE_COLS = [
    "season_best", "career_best", "pb_gap", "meets_count", "consistency",
    "yoy_improvement", "age", "season_rank", "season_percentile",
    "weighted_season_best", "wind_adj_season_best",
    "recent_trend", "days_since_last", "h2h_win_rate",
]


@pytest.fixture
def fixture_train_model(monkeypatch):
    """Points train_model's data paths + discipline/year scope at the small
    fixtures instead of the real project data."""
    monkeypatch.setattr(tm, "RAW_DIR", os.path.join(FIXTURES_DIR, "raw"))
    monkeypatch.setattr(tm, "DL_RESULTS_PATH", os.path.join(FIXTURES_DIR, "dl_final_results.csv"))
    monkeypatch.setattr(tm, "H2H_PATH", os.path.join(FIXTURES_DIR, "h2h", "h2h_rates.csv"))
    monkeypatch.setattr(tm, "TRAIN_DISCIPLINES", {"men_100m": "Men 100m", "women_LJ": "Women Long Jump"})
    monkeypatch.setattr(tm, "LABEL_YEARS", [2021, 2022, 2023])
    return tm


def test_build_labeled_dataset_merges_real_labels_onto_features(fixture_train_model):
    labeled = fixture_train_model.build_labeled_dataset()

    # Every fixture discipline-year is DL-Final-contested, so nothing should
    # have been dropped by the contested-year filter.
    assert set(labeled["discipline"].unique()) == {"men_100m", "women_LJ"}
    assert set(labeled["year"].unique()) == {2021, 2022, 2023}

    men_2023 = labeled[(labeled["discipline"] == "men_100m") & (labeled["year"] == 2023)]
    top3_names = set(men_2023.loc[men_2023["dl_top3"] == 1, "athlete_name"])
    assert top3_names == {"Alpha SPEEDY", "Beta MIDPACK"}  # "Gamma MEDIUM" ran 3rd-best but
    # didn't actually medal at the (fixture) Final -- "Delta OUTSIDER" did, and isn't in the
    # toplist fixture at all, so correctly contributes no dl_top3=1 row anywhere.

    women_2023 = labeled[(labeled["discipline"] == "women_LJ") & (labeled["year"] == 2023)]
    assert set(women_2023.loc[women_2023["dl_top3"] == 1, "athlete_name"]) == \
        {"Xena JUMPER", "Yara HOPPER", "Zoe LEAPER"}


def test_train_and_backtest_runs_end_to_end_on_fixture_data(fixture_train_model):
    """Doesn't assert on save_artifacts()/outputs/ -- train_and_backtest()
    itself never writes to disk, only returns
    (model, scaler, accuracy_pct, field_pct).

    `field_pct` is the same predictions scored against the athletes who
    actually contested the Final, rather than the whole toplist -- the task
    run.py performs (2026-08-25). It is None only when the fixture has no
    Final field to restrict to."""
    model, scaler, accuracy_pct, field_pct = fixture_train_model.train_and_backtest(
        FEATURE_COLS, label="fixture test")

    assert 0.0 <= accuracy_pct <= 100.0
    assert field_pct is None or 0.0 <= field_pct <= 100.0
    assert hasattr(model, "predict_proba")
    assert hasattr(scaler, "transform")

    # The fixture's signal is clean (the same 3 athletes are both the best
    # season and the actual DL Final medalists every year) -- a model that's
    # at least learning *something* real from season_rank/season_percentile
    # should beat a coin flip on this easy a walk-forward fold.
    assert accuracy_pct > 50.0


def test_load_predictions_reads_fixture_csv(monkeypatch):
    monkeypatch.setattr(api, "OUTPUTS_DIR", os.path.join(FIXTURES_DIR, "outputs"))
    monkeypatch.setattr(api, "INJURY_FLAGS_PATH", os.path.join(FIXTURES_DIR, "nonexistent_injury_flags.json"))

    track, field = api.load_predictions()

    men_100m = next(d for d in track if d["id"] == "men_100m")
    assert [a["name"] for a in men_100m["athletes"]] == ["Alpha SPEEDY", "Beta MIDPACK", "Gamma MEDIUM"]
    assert men_100m["athletes"][0]["prob"] == 45
    assert men_100m["athletes"][2]["injuryWatch"] is True  # fixture's 3rd-place row sets injury_watch=True

    women_lj = next(d for d in field if d["id"] == "women_LJ")
    assert women_lj["athletes"][0]["mark"] == "7.05m"


def test_load_predictions_missing_file_returns_none(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "OUTPUTS_DIR", str(tmp_path))  # empty dir, no predictions_latest.csv
    assert api.load_predictions() == (None, None)
