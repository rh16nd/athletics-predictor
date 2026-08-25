"""Unit tests for src/injury_checker.py's pure matching logic.

HANDOFF listed the injury checker as untested because its scraping needs
Selenium and live network. That is true of fetch_headlines/check_injuries,
but the parts that decide whether a real athlete gets REMOVED from the
predictions -- keyword matching, recovery estimation, and attribution -- are
pure string functions and are exactly where a bug does the most damage.

These exist because of a real, live false positive found 2026-08-25: Cole
Hocker was removed from the men's 1500m predictions entirely on the strength
of the headline "Jakob Ingebrigtsen Is Back. His First Big 1500m Test: Cole
Hocker" -- "back" matched as a body part, and the article was about someone
else.
"""
import injury_checker as ic

REAL_FALSE_POSITIVE = (
    "jakob ingebrigtsen is back. his first big 1500m test: cole hocker"
)


def test_the_headline_that_wrongly_removed_cole_hocker_matches_nothing():
    matched = ic.match_keywords(REAL_FALSE_POSITIVE)
    assert matched["remove"] == []
    assert matched["watch"] == []
    assert ic.estimate_recovery_weeks(REAL_FALSE_POSITIVE) is None


def test_returning_phrasings_are_not_injuries():
    for headline in (
        "noah lyles is back to winning ways",
        "back-to-back wins for the world champion",
        "she bounced back from a disappointing final",
        "he is back in the field for brussels",
    ):
        assert ic.match_keywords(headline)["watch"] == [], headline
        assert ic.estimate_recovery_weeks(headline) is None, headline


def test_real_back_injuries_are_still_caught():
    for headline, expected in (
        ("cole hocker out with a back injury", "back injury"),
        ("sprinter nursing a lower back problem", "lower back"),
        ("back spasm forces withdrawal", "back spasm"),
    ):
        assert expected in ic.match_keywords(headline)["watch"], headline
        assert ic.estimate_recovery_weeks(headline) == (2, 8), headline


def test_other_body_parts_are_unaffected():
    assert "hamstring" in ic.match_keywords("hamstring strain ends his season")["watch"]
    assert ic.estimate_recovery_weeks("achilles rupture") == (12, 26)  # severity upgrade


def test_severity_words_widen_and_narrow_the_estimate():
    base = ic.estimate_recovery_weeks("calf strain")
    worse = ic.estimate_recovery_weeks("torn calf")
    milder = ic.estimate_recovery_weeks("minor calf tightness")
    assert worse[1] > base[1]
    assert milder[1] <= base[1]


def test_remove_keywords_still_trigger_removal():
    assert "withdraws" in ic.match_keywords("champion withdraws from brussels")["remove"]
    assert "ruled out" in ic.match_keywords("ruled out for the season")["remove"]


# --- attribution ---------------------------------------------------------

def test_keyword_is_attributed_to_the_nearest_named_athlete():
    headline = "jakob ingebrigtsen has a hamstring injury. next test: cole hocker"
    names = ["jakob ingebrigtsen", "cole hocker"]
    assert ic.keyword_is_about(headline, "jakob ingebrigtsen", names, "hamstring")
    assert not ic.keyword_is_about(headline, "cole hocker", names, "hamstring")


def test_a_single_named_athlete_always_owns_the_keyword():
    headline = "cole hocker has a hamstring injury"
    assert ic.keyword_is_about(headline, "cole hocker", ["cole hocker"], "hamstring")


def test_attribution_is_false_when_the_athlete_is_not_named():
    headline = "jakob ingebrigtsen has a hamstring injury"
    assert not ic.keyword_is_about(headline, "cole hocker", ["jakob ingebrigtsen"], "hamstring")


def test_unknown_keyword_does_not_crash_attribution():
    assert ic.keyword_is_about("some headline", "someone", ["someone"], "not-a-keyword")
