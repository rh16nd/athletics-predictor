"""A career best has to include the season being scored.

`build_2026_features` reads career bests from data/raw/<key>.csv, which stops at
2025 -- the current season lives in <key>_2026.csv. So an athlete running the
fastest race of their life this year was handed a career best from a slower
year, and `pb_gap`, which the trainer defines as "how far off your best you
are", came out as "how much you have improved". Same number, opposite meaning,
and the model reads it with the meaning it was trained on.

Measured on the real 2026 data before the fix: **885 of 3,993 athletes, 22.2%,
in every one of the 32 disciplines.** Alison dos Santos, who ran 45.80 against a
46.29 best from earlier years, was scored as 0.49s off his peak while setting a
world record. Reported by the user, who asked why the world record holder was
not the model's top pick in his event.

Nothing else in the suite catches this: the feature is present, correctly named,
and plausible-looking. Only its meaning is wrong.
"""
import os

import pandas as pd
import pytest

import feature_builder as fb
import train_model as tm

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "career_best")


@pytest.fixture
def improving_athletes(tmp_path, monkeypatch):
    """One track discipline and one field discipline, each with an athlete who
    has just set a lifetime best and one who has not."""
    raw = tmp_path
    # Track: lower is better. RECORD BREAKER improves on 46.29 with 45.80.
    pd.DataFrame([
        {"Competitor": "Record BREAKER", "Mark": "46.29", "year": 2024},
        {"Competitor": "Record BREAKER", "Mark": "46.55", "year": 2025},
        {"Competitor": "Past PEAK", "Mark": "45.94", "year": 2024},
    ]).to_csv(raw / "men_400h.csv", index=False)
    pd.DataFrame([
        {"Rank": 1, "Mark": "45.80", "WIND": "", "Competitor": "Record BREAKER",
         "DOB": "01 Jan 1999", "Venue": "Test", "Date": "10 Aug 2026",
         "discipline": "men_400h", "year": 2026},
        {"Rank": 2, "Mark": "46.52", "WIND": "", "Competitor": "Past PEAK",
         "DOB": "01 Jan 1996", "Venue": "Test", "Date": "10 Aug 2026",
         "discipline": "men_400h", "year": 2026},
    ]).to_csv(raw / "men_400h_2026.csv", index=False)

    # Field: HIGHER is better, so the same bug needs the opposite comparison.
    pd.DataFrame([
        {"Competitor": "Long JUMPER", "Mark": "8.10", "year": 2024},
    ]).to_csv(raw / "men_LJ.csv", index=False)
    pd.DataFrame([
        {"Rank": 1, "Mark": "8.45", "WIND": "", "Competitor": "Long JUMPER",
         "DOB": "01 Jan 1999", "Venue": "Test", "Date": "10 Aug 2026",
         "discipline": "men_LJ", "year": 2026},
    ]).to_csv(raw / "men_LJ_2026.csv", index=False)

    monkeypatch.setattr(fb, "RAW_DIR", str(raw))
    return raw


def row(feat, name):
    return feat[feat["athlete_name"] == name].iloc[0]


def test_a_lifetime_best_this_season_becomes_the_career_best(improving_athletes):
    breaker = row(fb.build_2026_features("men_400h"), "Record BREAKER")
    assert breaker["season_best"] == pytest.approx(45.80)
    assert breaker["career_best"] == pytest.approx(45.80)
    assert breaker["pb_gap"] == pytest.approx(0.0)


def test_an_athlete_off_their_peak_still_reads_the_gap(improving_athletes):
    """The fix must not flatten pb_gap to zero for everybody -- it is only the
    impossible direction that was wrong."""
    past = row(fb.build_2026_features("men_400h"), "Past PEAK")
    assert past["career_best"] == pytest.approx(45.94)
    assert past["pb_gap"] == pytest.approx(0.58)


def test_the_field_event_direction_is_not_reversed(improving_athletes):
    """Higher is better here, so "best" is a max. Taking a min would hand a
    jumper their WORST mark as a career best -- the same class of direction bug
    add_season_rank once had for the whole pole vault."""
    jumper = row(fb.build_2026_features("men_LJ"), "Long JUMPER")
    assert jumper["career_best"] == pytest.approx(8.45)
    assert jumper["pb_gap"] == pytest.approx(0.0)


def test_no_athlete_can_have_a_career_best_worse_than_their_season_best(improving_athletes):
    """The invariant, stated directly. It holds by construction in training,
    where build_features takes career_best over marks up to AND INCLUDING the
    label year, so it must hold at serve time too."""
    for key, is_field in (("men_400h", False), ("men_LJ", True)):
        feat = fb.build_2026_features(key)
        impossible = ((feat["season_best"] > feat["career_best"]) if is_field
                      else (feat["season_best"] < feat["career_best"]))
        assert not impossible.any(), feat[impossible][
            ["athlete_name", "season_best", "career_best"]]


def test_training_never_produces_the_impossible_direction():
    """Proves the invariant is the trainer's, not an invention of this test:
    build_features computes career_best over a SUPERSET of the season's marks,
    so it can never come out worse."""
    df = pd.DataFrame([
        {"athlete_name": "Record BREAKER", "Mark": "46.29", "year": 2024,
         "date": pd.Timestamp("2024-07-01"), "age": 25.0, "country": "BRA"},
        {"athlete_name": "Record BREAKER", "Mark": "45.80", "year": 2026,
         "date": pd.Timestamp("2026-08-10"), "age": 27.0, "country": "BRA"},
    ])
    feat = tm.build_features(df, "men_400h", [(2026, "Test Final", "2026-09-01")])
    assert feat.iloc[0]["career_best"] == pytest.approx(45.80)
    assert feat.iloc[0]["pb_gap"] == pytest.approx(0.0)
