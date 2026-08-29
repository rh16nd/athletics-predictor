"""
Cross-discipline performance stats (/api/stats), built on World Athletics'
Results Score.

The indoor tests carry the most weight here. WA writes indoor marks into its
outdoor season toplists tagged only by a "(i)" venue suffix, and on the real
2026 data that is 13% of all rows and ~47% of the men's high jump. The site
keeps those marks deliberately -- for a vault or a shot put indoors is
arguably the truer measure of ability -- so the only thing standing between
a reader and an unlabelled indoor mark is the detector below.
"""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import api  # noqa: E402


# ---- the indoor detector ----

@pytest.mark.parametrize("venue", [
    "IFU Arena, Uppsala (SWE) (i)",          # WA's actual suffix format
    "Complexe Kindarena, Rouen (FRA) (i)",
    "Randal Tyson Indoor Center, Fayetteville, AR (USA)",
    "Atletická hala, Ostrava (CZE) (i)",
])
def test_indoor_venues_are_detected(venue):
    assert api.is_indoor_venue(venue) is True


@pytest.mark.parametrize("venue", [
    "National Stadium, Kingston (JAM)",
    "Letzigrund, Zürich (SUI)",
    "Hayward Field, Eugene, OR (USA)",
    "Stadion Śląski, Chorzów (POL)",
    None,
    "",
])
def test_outdoor_venues_are_not_flagged(venue):
    assert api.is_indoor_venue(venue) is False


def test_a_venue_merely_containing_the_letter_i_is_not_indoor():
    """The pattern anchors on a parenthesised "(i)" at the end, not a bare
    letter -- otherwise every venue with an (ITA)/(IND) country code, or the
    word "Stadio", would be called indoor."""
    for venue in ["Stadio Olimpico, Roma (ITA)", "Kalinga Stadium, Bhubaneswar (IND)",
                  "Suhaim bin Hamad Stadium, Doha (QAT)"]:
        assert api.is_indoor_venue(venue) is False


# ---- build_stats ----

@pytest.fixture(scope="module")
def stats():
    return api.build_stats()


def test_stats_ranks_across_disciplines(stats):
    """The whole point of using WA's score: the top of the list is not one
    event's leaderboard."""
    top = stats["topPerformances"]
    assert len(top) > 1
    assert len({p["discKey"] for p in top}) > 1
    scores = [p["score"] for p in top]
    assert scores == sorted(scores, reverse=True)


def test_every_performance_says_whether_it_was_indoors(stats):
    assert all(isinstance(p["indoor"], bool) for p in stats["topPerformances"])


def test_indoor_marks_are_kept_not_filtered(stats):
    """A regression guard with teeth: the highest-scoring mark in the real
    2026 data is Duplantis's 6.31 in Uppsala, set INDOORS. If someone
    "cleans" indoor rows out of load_season_scores, the site quietly loses
    the best performance of the season."""
    assert stats["indoor"]["rows"] > 0
    assert 0 < stats["indoor"]["share"] < 100


def test_discipline_depth_covers_every_discipline_with_a_toplist(stats):
    depth = stats["disciplineDepth"]
    assert len(depth) > 0
    for d in depth:
        assert d["medianScore"] <= d["topScore"]
        assert d["athletes"] > 0
        assert 0 <= d["indoorShare"] <= 100
    medians = [d["medianScore"] for d in depth]
    assert medians == sorted(medians, reverse=True)


def test_payload_is_json_serialisable(stats):
    """pandas means every aggregate is a numpy scalar until it is cast, and
    Flask's jsonify refuses np.float64 -- this caught a real 500."""
    import json
    json.dumps(stats)


# ---- athlete_score_context ----

def test_score_context_reports_both_percentiles():
    df = api.load_season_scores()
    if df.empty:
        pytest.skip("no season toplists on disk")
    row = df.nlargest(1, "Results Score").iloc[0]
    ctx = api.athlete_score_context(row["discKey"], row["Competitor"])
    assert ctx is not None
    # The best mark in the whole dataset tops both scales.
    assert ctx["percentile"] == pytest.approx(100.0, abs=0.2)
    assert ctx["discPercentile"] == pytest.approx(100.0, abs=0.2)
    assert ctx["score"] == int(row["Results Score"])


def test_score_context_is_none_for_an_unknown_athlete():
    assert api.athlete_score_context("men_100m", "Nobody At All") is None


def test_cross_discipline_percentile_differs_from_within_discipline():
    """Both numbers are reported because they answer different questions --
    if they were always equal one of them would be dead weight."""
    df = api.load_season_scores()
    if df.empty:
        pytest.skip("no season toplists on disk")
    differing = 0
    for _, row in df.sample(min(40, len(df)), random_state=0).iterrows():
        ctx = api.athlete_score_context(row["discKey"], row["Competitor"])
        if ctx and abs(ctx["percentile"] - ctx["discPercentile"]) > 1:
            differing += 1
    assert differing > 0


def test_season_scores_carry_a_score_for_every_row():
    df = api.load_season_scores()
    if df.empty:
        pytest.skip("no season toplists on disk")
    assert df["Results Score"].notna().all()
    assert pd.api.types.is_bool_dtype(df["indoor"])


# ---- stats for athletes run.py never scored ----

def test_unscored_athletes_get_a_career_best_and_pb_gap():
    """Reported for Dina Asher-Smith, who is not in predictions_latest.csv
    at all -- yet has 41 races on record, 8 seasons of history and a 10.83
    career best. Only the SCORE needs the model; career best, PB gap and
    meeting count are facts already on disk, and the page was showing
    dashes over data it had."""
    import pandas as pd
    scored = set(pd.read_csv(
        os.path.join(os.path.dirname(__file__), "..", "outputs", "predictions_latest.csv")
    )["athlete_name"].dropna())
    if "Dina ASHER-SMITH" in scored:
        pytest.skip("she is scored in this snapshot; the branch under test is the unscored one")

    out = api.athlete_field_status("women_100m", "Dina ASHER-SMITH")
    assert out["careerBest"] is not None
    assert out["pbGap"] is not None
    assert out["meetsCount"] is not None


def test_pb_gap_matches_feature_builders_definition():
    """`abs(season_best - career_best)`. If this drifts, the same label means
    two different things on a scored and an unscored athlete's page."""
    out = api.athlete_field_status("women_100m", "Dina ASHER-SMITH")
    if out["pbGap"] is None:
        pytest.skip("no data on disk for this athlete")
    season = api.safe_parse_mark(out["seasonBest"])
    career = api.safe_parse_mark(out["careerBest"])
    assert out["pbGap"] == pytest.approx(abs(season - career), abs=0.002)


def test_career_best_is_the_best_mark_not_the_latest():
    """Track means lowest. Asher-Smith's 2026 best is 11.10 and her career
    best is 10.83 -- taking the most recent season would report the wrong
    number, and taking max() would report the worst one."""
    out = api.athlete_field_status("women_100m", "Dina ASHER-SMITH")
    if out["careerBest"] is None:
        pytest.skip("no data on disk for this athlete")
    career = api.safe_parse_mark(out["careerBest"])
    seasons = api.load_career_progression("women_100m", "Dina ASHER-SMITH")
    assert career == pytest.approx(min(s["best"] for s in seasons), abs=0.002)


def test_dl_meetings_count_is_zero_not_none_when_the_file_has_no_rows():
    """For an athlete outside the field that zero is the answer, and often
    the reason. None would render as "unknown", which is a different claim."""
    assert api.dl_meetings_count("women_100m", "Nobody At All") == 0
    assert api.dl_meetings_count("not_a_discipline", "Anyone") is None
