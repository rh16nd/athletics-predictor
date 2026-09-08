"""Unit tests for the Ultimate Championship projection
(src/ultimate_predictions.py and the qualified-field fetch it rests on).

The thing worth guarding here is the boundary between what World Athletics
says and what the model says. The FIELD is WA's -- who qualified, and how --
and the ORDER is ours. A projection that quietly invented an entrant, or
quietly dropped one, would look exactly like a correct one.
"""
import ultimate_scraper as us
import ultimate_predictions as up


def test_disciplines_we_hold_no_data_for_do_not_get_a_key():
    """The Ultimate contests a men's hammer throw. We have no hammer toplist,
    no history and no model for it. Mapping it to "men_HT" anyway would hand
    the projector a discipline it cannot score and make an honest gap look
    like a bug."""
    assert us.disc_key_from_wa("Men's Hammer Throw") is None
    assert us.disc_key_from_wa("Mixed 4x400 Metres Relay") is None
    assert us.disc_key_from_wa("Men's 400 Metres Hurdles") == "men_400h"
    assert us.disc_key_from_wa("Women's Triple Jump") == "women_TJ"


def test_only_athletes_wa_marks_qualified_are_kept(monkeypatch):
    """WA's qualification endpoint returns everyone it considered, not just
    those who got in -- 57 rows for a 17-place event. Treating the whole list
    as the field would invent one three times the real size."""
    events = [{"genderCode": "M", "eventId": 1, "disciplineName": "Men's 800 Metres"}]

    def fake_graphql(op, variables, query):
        if variables.get("e") is None:
            return {"getChampionshipQualifications": {"events": events}}
        return {"getChampionshipQualifications": {
            "disciplineName": "Men's 800 Metres", "entryNumber": 2,
            "qualifications": [
                {"name": "IN ONE", "countryCode": "KEN", "qualified": True,
                 "qualifiedBy": "Qualified by Wild Card", "qualificationPosition": 1,
                 "score": None, "competitorIaafId": 1},
                {"name": "IN TWO", "countryCode": "CAN", "qualified": True,
                 "qualifiedBy": "Qualified by World Rankings", "qualificationPosition": 2,
                 "score": 1290, "competitorIaafId": 2},
                {"name": "MISSED OUT", "countryCode": "GBR", "qualified": False,
                 "qualifiedBy": None, "qualificationPosition": 3,
                 "score": 1200, "competitorIaafId": 3},
            ]}}

    monkeypatch.setattr(us.dlr, "graphql", fake_graphql)
    field = us.fetch_qualified_field()
    assert len(field) == 1
    names = [a["name"] for a in field[0]["athletes"]]
    assert names == ["IN ONE", "IN TWO"], "a non-qualifier reached the field"
    assert field[0]["athletes"][0]["qualifiedBy"] == "Qualified by Wild Card"


def test_an_event_we_cannot_score_is_skipped_not_faked(monkeypatch):
    event = {"discKey": None, "disciplineLabel": "Men's Hammer Throw",
             "athletes": [{"name": "A"}]}
    assert up.project_event(event, None, None, []) is None


def test_qualified_athletes_with_no_mark_are_named_not_dropped(monkeypatch):
    """A qualified athlete the model cannot score is a hole in the projection.
    Counting them silently would let a missing favourite pass unnoticed; the
    page names them instead."""
    import pandas as pd

    event = {
        "discKey": "men_800m", "disciplineLabel": "Men's 800 Metres", "sex": "M",
        "places": 2,
        "athletes": [
            {"name": "HAS A MARK", "nat": "KEN", "qualifiedBy": "Wild Card",
             "position": 1, "rankingScore": None},
            {"name": "NO MARK", "nat": "USA", "qualifiedBy": "World Rankings",
             "position": 2, "rankingScore": 1300},
        ],
    }
    monkeypatch.setattr(up, "build_2026_features",
                        lambda k: pd.DataFrame({"athlete_name": ["Has A Mark"], "x": [1.0]}))
    monkeypatch.setattr(up, "add_h2h", lambda df, k: df.assign(h2h_win_rate=0.5))

    import numpy as np

    class Model:
        def predict_proba(self, X):
            return np.array([[0.4, 0.6]])

    class Scaler:
        def transform(self, X):
            return X

    out = up.project_event(event, Model(), Scaler(), ["x"])
    assert out["scored"] == 1
    assert out["qualified"] == 2
    assert out["unscored"] == ["NO MARK"]
    assert out["athletes"][0]["name"] == "Has A Mark"
    assert out["athletes"][0]["podiumChance"] == 60.0
