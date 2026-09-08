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


# --- The field the checker watches (added 2026-09-08) -------------------------
#
# The checker read data/standings.json alone, which was right while the Diamond
# League Final was the site's next event and silently wrong afterwards: once the
# site pivoted to the Ultimate Championship, 143 of that field's 306 athletes had
# never raced the circuit, so nothing ever looked at them and an injured athlete
# kept their projection. These pin the widened field and the target date that
# decides "watch" vs "remove".

import json


def _write(tmp_path, standings=None, projections=None, event=None):
    """Points the module's paths at a temp directory and returns nothing."""
    data = tmp_path / "data"
    (data / "ultimate").mkdir(parents=True, exist_ok=True)
    if standings is not None:
        (data / "standings.json").write_text(json.dumps(standings), encoding="utf-8")
    if projections is not None:
        (data / "ultimate" / "predictions.json").write_text(
            json.dumps({"projections": projections}), encoding="utf-8")
    if event is not None:
        (data / "ultimate" / "event.json").write_text(json.dumps(event), encoding="utf-8")
    ic.STANDINGS_PATH = str(data / "standings.json")
    ic.ULTIMATE_PREDICTIONS_PATH = str(data / "ultimate" / "predictions.json")
    ic.ULTIMATE_EVENT_PATH = str(data / "ultimate" / "event.json")


def test_the_ultimate_field_is_watched_not_just_the_diamond_league(tmp_path, monkeypatch):
    monkeypatch.setattr(ic, "STANDINGS_PATH", "", raising=False)
    _write(
        tmp_path,
        standings={"men_100m": ["Oblique SEVILLE"]},
        projections=[{"discKey": "men_800m", "athletes": [{"name": "Josh HOEY"}],
                      "unscored": ["Emmanuel WANYONYI"]}],
    )
    athletes = ic.load_qualified_athletes()
    # The Diamond League name still counts...
    assert "Oblique Seville" in athletes
    # ...and so does an Ultimate qualifier who never raced the circuit, which is
    # the whole point of the change.
    assert "Josh Hoey" in athletes
    # An athlete WA says has qualified but the model could not score is still a
    # real person on a start line.
    assert "Emmanuel Wanyonyi" in athletes


def test_a_name_in_both_competitions_carries_both_disciplines(tmp_path, monkeypatch):
    monkeypatch.setattr(ic, "STANDINGS_PATH", "", raising=False)
    _write(
        tmp_path,
        standings={"men_1500m": ["Cole HOCKER"]},
        projections=[{"discKey": "men_5000m", "athletes": [{"name": "Cole HOCKER"}]}],
    )
    assert ic.load_qualified_athletes()["Cole Hocker"] == {"men_1500m", "men_5000m"}


def test_ultimate_alone_is_survivable(tmp_path, monkeypatch):
    monkeypatch.setattr(ic, "STANDINGS_PATH", "", raising=False)
    _write(tmp_path, projections=[{"discKey": "men_100m", "athletes": [{"name": "Noah LYLES"}]}])
    assert ic.load_qualified_athletes() == {"Noah Lyles": {"men_100m"}}


def test_the_diamond_league_alone_is_survivable(tmp_path, monkeypatch):
    monkeypatch.setattr(ic, "STANDINGS_PATH", "", raising=False)
    _write(tmp_path, standings={"men_100m": ["Noah LYLES"]})
    assert ic.load_qualified_athletes() == {"Noah Lyles": {"men_100m"}}


def test_neither_source_means_skip_not_an_empty_all_clear(tmp_path, monkeypatch):
    monkeypatch.setattr(ic, "STANDINGS_PATH", "", raising=False)
    _write(tmp_path)
    assert ic.load_qualified_athletes() == {}


def test_the_target_date_follows_the_next_competition(tmp_path, monkeypatch):
    monkeypatch.setattr(ic, "STANDINGS_PATH", "", raising=False)
    _write(tmp_path, event={"startDate": "2026-09-11"})
    assert ic.target_date().isoformat() == "2026-09-11"


def test_a_missing_event_file_falls_back_rather_than_crashing(tmp_path, monkeypatch):
    monkeypatch.setattr(ic, "STANDINGS_PATH", "", raising=False)
    _write(tmp_path)
    assert ic.target_date() == ic.DL_FINAL_DATE


# --- The widened search (added 2026-09-08) -----------------------------------
#
# Three front pages yielded 179 headlines, one of which mentioned an injury, so
# Noah Lyles ending his season and Duplantis withdrawing were simply never seen.
# Google News fixed the recall and broke the precision; these pin both halves.

def test_a_hyphenated_name_survives_a_headline_that_drops_the_hyphen():
    # The real miss: WA has "Tara Davis-Woodhall", LetsRun wrote
    # "Tara Davis Woodhall Injured In Car Wreck", and exact matching saw nothing.
    assert ic.normalize_for_match("Tara DAVIS-WOODHALL") == "tara davis woodhall"


def test_accents_are_folded_so_a_headline_spelling_still_matches():
    assert ic.normalize_for_match("Kristjan ČEH") == "kristjan ceh"
    assert ic.normalize_for_match("Yeral NUÑEZ") == "yeral nunez"


def test_a_surname_shared_by_two_athletes_is_never_an_alias():
    # Three Davises in the field. Matching "Davis Injured" to one at random is
    # the Cole Hocker mistake with extra steps.
    aliases = ic.build_aliases({"Tamari Davis": {"women_100m"},
                                "Tara Davis-Woodhall": {"women_LJ"}})
    assert "davis" not in aliases
    assert aliases["woodhall"] == ("Tara Davis-Woodhall", True)
    assert aliases["tara davis woodhall"] == ("Tara Davis-Woodhall", False)


def test_a_given_name_never_becomes_an_alias():
    aliases = ic.build_aliases({"Tara Davis-Woodhall": {"women_LJ"}})
    assert "tara" not in aliases


def test_name_particles_and_common_words_are_not_aliases():
    aliases = ic.build_aliases({
        "Alida Van Daalen": {"women_DT"},
        "Someone Long": {"men_LJ"},
        "Another Young": {"men_100m"},
    })
    assert "van" not in aliases
    assert "long" not in aliases
    assert "young" not in aliases
    assert aliases["daalen"] == ("Alida Van Daalen", True)


def test_a_surname_only_hit_needs_the_headline_to_mention_the_sport():
    # "Latics boss Caldwell provides positive Chapman injury update" is a
    # footballer; "Caudery out for the season" (European Athletics) is not.
    assert not ic.reads_like_athletics(
        ic.normalize_for_match("Latics boss Caldwell provides positive Chapman injury update"))
    assert ic.reads_like_athletics(
        ic.normalize_for_match("World indoor champion Caudery out for the season - athletics"))


def test_other_sports_are_rejected_outright():
    for headline in ("Cowboy Basketball's Jennings out for the season",
                     "Marshall Baseball Ends Season at the SBC Championship",
                     "Injury disappointment for Jones in the football club cup clash"):
        assert ic.is_other_sport(ic.normalize_for_match(headline)), headline


def test_club_names_are_not_used_as_a_filter():
    # "united" would reject "United States", "villa" would reject "Villanueva".
    assert not ic.is_other_sport(ic.normalize_for_match("United States athlete ruled out"))
    assert not ic.is_other_sport(ic.normalize_for_match("Villanueva withdraws from the 800m"))


def test_a_comeback_story_is_not_an_injury_report():
    for headline in (
        "'I'm not just an athlete' - Wightman on injury bounce back",
        "analysis of Jazmin Sawyers and her success since her achilles injury",
        "Hodgkinson returns from injury scare for the showdown",
    ):
        assert ic.is_returning_story(ic.normalize_for_match(headline)), headline


def test_a_withdrawal_still_counts_even_when_it_mentions_returning():
    # Suppression must never swallow a real one: this says he is injured AND out.
    norm = ic.normalize_for_match(
        "Injured Omanyala pulls out of Diamond League final and World Ultimate Championships")
    assert ic.match_keywords(norm)["remove"], "the withdrawal must still register"


def test_the_publisher_suffix_google_appends_is_not_matchable_text():
    # "Charlton Athletic Football Club" flagged Devynne Charlton until the
    # " - Publisher" tail was stripped before name matching.
    assert ic._published_date("Mon, 08 Sep 2026 09:00:00 GMT") == "2026-09-08"
    assert ic._published_date("not a date") is None
    assert ic._published_date(None) is None


# --- Returns override older withdrawals (added 2026-09-08) -------------------
#
# Reported by the user: an athlete was listed as out while more recent coverage
# said they were back. Keely Hodgkinson withdrew from Zurich (24 Aug) and from
# another race (2 Sep); "Keely Hodgkinson returns from injury scare" ran on
# 7 Sep and the flag stood anyway.

RETURN_HEADLINE = "keely hodgkinson returns from injury scare for werro and broeders bol showdown"


def test_a_return_is_attributed_by_the_return_phrase_not_the_injury_word():
    # The word "injury" is nearest Werro, so the keyword rule gives it to her.
    # "returns from injury" is nearest Hodgkinson, and she is the one who is back.
    assert ic.returning_is_about(RETURN_HEADLINE, "keely hodgkinson", ["werro", "broeders", "bol"])
    assert not ic.returning_is_about(RETURN_HEADLINE, "werro", ["keely hodgkinson", "broeders"])


def test_a_return_is_not_credited_to_an_athlete_merely_present():
    assert not ic.returning_is_about(RETURN_HEADLINE, "broeders bol", ["keely hodgkinson", "werro"])


def test_an_athletes_own_surname_does_not_beat_their_full_name():
    # `other_names` must exclude the target's own aliases. Passing "hodgkinson"
    # in as an "other" would put a nearer name next to the phrase and reject
    # every return she has.
    assert ic.returning_is_about(RETURN_HEADLINE, "keely hodgkinson", ["werro"])


def test_a_headline_with_no_return_phrase_is_not_a_return():
    assert not ic.returning_is_about(
        "keely hodgkinson withdraws from zurich diamond league", "keely hodgkinson", [])


# --- Naming the competition is what makes an out an out (added 2026-09-08) ----
#
# Reported by the user, about Duplantis: he was pulled out of the pole vault
# projection on a report saying he withdrew from the DIAMOND LEAGUE FINAL, a
# meeting already run, which says nothing about Budapest. The surname cap keeps
# a one-word name match from ever becoming a withdrawal -- right in general, and
# wrong when the headline also names the competition.

def _terms():
    return {"budapest", "ultimate championship", "ultimate championships",
            "world athletics ultimate championship"}


def _is_out(headline):
    n = ic.normalize_for_match(headline)
    return (ic.names_target_event(n, _terms())
            and any(w in n for w in ic.OUT_OF_EVENT_WORDS))


def test_naming_the_championship_and_an_absence_is_an_out():
    for headline in (
        "Injury holds back Cuban gem Jorge Hodelin: will not compete in the World Ultimate Championship",
        "Injured Omanyala pulls out of Diamond League final and World Ultimate Championships",
        # Only "surgery" matched here, a watch word -- this athlete sat in the
        # projection as a doubt while the headline said he was gone.
        "Sachin Yadav Undergoes Elbow Surgery, Misses World Athletics Ultimate Championships",
    ):
        assert _is_out(headline), headline


def test_withdrawing_from_a_different_meeting_is_not_an_out():
    # The case that started this. It is a real withdrawal from a real meeting,
    # and that meeting is not the one being projected.
    assert not _is_out(
        "Duplantis withdraws from Diamond League finals after thigh discomfort during warm-up")
    assert not _is_out("Day 2 Brussels Diamond League Final Results")


def test_naming_the_championship_without_an_absence_is_not_an_out():
    assert not _is_out("Duplantis targets another world record at the Ultimate Championship")


def test_the_event_terms_are_read_from_the_event_file(tmp_path, monkeypatch):
    # Not hard-coded, so this follows the site when the next championship takes
    # the tab.
    monkeypatch.setattr(ic, "STANDINGS_PATH", "", raising=False)
    _write(tmp_path, event={"name": "World Athletics Ultimate Championship",
                            "shortName": "Budapest 26", "city": "Budapest"})
    terms = ic.target_event_terms()
    assert "ultimate championship" in terms
    assert "budapest" in terms


def test_a_missing_event_file_yields_no_terms_rather_than_crashing(tmp_path, monkeypatch):
    monkeypatch.setattr(ic, "STANDINGS_PATH", "", raising=False)
    _write(tmp_path)
    assert ic.target_event_terms() == set()
