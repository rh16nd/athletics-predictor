"""Unit tests for api.py's pure functions -- the ones that don't need a
running Flask server or real outputs/predictions_latest.csv on disk.

Two of these are regression tests for real gaps found and fixed in this
project: injury_evidence()/build_removed_athletes() exist because the
injury-watch flag and "removed" athletes were being computed correctly but
never reaching the dashboard at all -- see api.py's module history.
"""
from datetime import date

import api


def test_parse_mark_seconds():
    assert api.parse_mark("9.79") == 9.79


def test_parse_mark_minutes_format():
    assert api.parse_mark("1:41.84") == 101.84


def test_format_mark_field_event():
    assert api.format_mark(8.66, "men_LJ") == "8.66m"


def test_format_mark_middle_distance():
    assert api.format_mark(101.84, "men_800m") == "1:41.84"


def test_format_mark_sprint():
    assert api.format_mark(9.79, "men_100m") == "9.79"


def test_compute_meet_statuses_marks_past_meets_done_and_next_meet_next():
    meets = [
        {"n": 1, "date": "08 May", "city": "Doha"},
        {"n": 2, "date": "10 Jul", "city": "Monaco"},
        {"n": 3, "date": "27 Aug", "city": "Zürich"},
        {"n": 4, "date": "04 Sep", "city": "Brussels — Final"},
    ]
    result = api.compute_meet_statuses(meets, today=date(2026, 8, 22))
    statuses = {m["city"]: m["status"] for m in result}
    assert statuses["Doha"] == "done"
    assert statuses["Monaco"] == "done"
    assert statuses["Zürich"] == "next"  # first meet on/after today
    assert statuses["Brussels — Final"] == "final"  # always "final" regardless of date


def test_compute_meet_statuses_only_first_upcoming_meet_is_next():
    meets = [
        {"n": 1, "date": "01 Jan", "city": "Past"},
        {"n": 2, "date": "01 Dec", "city": "First upcoming"},
        {"n": 3, "date": "15 Dec", "city": "Second upcoming"},
        {"n": 4, "date": "31 Dec", "city": "Final"},
    ]
    result = api.compute_meet_statuses(meets, today=date(2026, 6, 1))
    statuses = {m["city"]: m["status"] for m in result}
    assert statuses["First upcoming"] == "next"
    assert statuses["Second upcoming"] == "upcoming"


def test_normalize_athlete_name_matches_all_caps_surname_format():
    assert api.normalize_athlete_name("Shericka JACKSON") == "Shericka Jackson"


def test_injury_evidence_extracts_headline_and_url():
    entry = {
        "status": "watch",
        "matches": [{"headline": "Full Lausanne Results", "url": "https://example.com/a", "source": "letsrun_results"}],
    }
    reason, url = api.injury_evidence(entry)
    assert "Full Lausanne Results" in reason
    assert "letsrun" in reason
    assert url == "https://example.com/a"


def test_injury_evidence_no_matches_returns_none():
    assert api.injury_evidence({"status": "watch", "matches": []}) == (None, None)
    assert api.injury_evidence(None) == (None, None)


def test_build_removed_athletes_only_includes_remove_status():
    injury_flags = {
        "Watched Athlete": {"status": "watch", "disciplines": ["men_100m"], "matches": []},
        "Removed Athlete": {
            "status": "remove",
            "disciplines": ["women_200m"],
            "matches": [{"headline": "withdraws injured", "url": "https://example.com/b", "source": "letsrun"}],
        },
    }
    removed = api.build_removed_athletes(injury_flags)
    assert len(removed) == 1
    assert removed[0]["name"] == "Removed Athlete"
    assert removed[0]["disciplines"] == ["Women's 200m"]
    assert removed[0]["url"] == "https://example.com/b"


def _discipline(label, prob, injury_watch=False):
    return {
        "label": label,
        "athletes": [{
            "name": f"{label} Winner", "mark": "10.00", "prob": prob, "waUrl": "https://x",
            "injuryWatch": injury_watch, "injuryReason": None, "injuryUrl": None,
        }],
    }


def test_build_top_winners_sorted_by_probability_and_capped_at_six():
    discs = [_discipline(f"Event {i}", prob=i * 10) for i in range(1, 9)]
    winners = api.build_top_winners(discs, [])
    assert len(winners) == 6
    assert [w["prob"] for w in winners] == sorted((w["prob"] for w in winners), reverse=True)
    assert winners[0]["disc"] == "Event 8"  # highest probability (80) first


def test_build_top_winners_skips_disciplines_with_no_athletes():
    discs = [_discipline("Has athletes", prob=50), {"label": "Empty", "athletes": []}]
    winners = api.build_top_winners(discs, [])
    assert len(winners) == 1
    assert winners[0]["disc"] == "Has athletes"


def test_build_confidence_sorted_descending_by_top_athlete_probability():
    discs = [_discipline("Low", prob=20), _discipline("High", prob=90)]
    confidence = api.build_confidence(discs, [])
    assert confidence[0]["disc"] == "High"
    assert confidence[0]["value"] == 90


# --- Storyline ordering (2026-08-24) -------------------------------------
# The Projections page features exactly one storyline at large scale. It used
# to be whichever generator ran first, so a routine "#2 First Final
# appearance" outranked "12-0, the head-to-head leader is NOT the model's
# pick". These cover the computed rule that replaced that fixed order.

def _story(kind, stat, *athletes):
    return {"type": kind, "title": kind, "stat": stat, "text": "", "athletes": list(athletes)}


def test_rivalry_whose_h2h_leader_is_not_the_model_pick_outranks_a_debutant():
    stories = [
        _story("debutant", "#2", "Debutant"),
        _story("rivalry", "12-0", "Lyles", "Seville"),
    ]
    ranked = api.rank_storylines(stories, prob_leader="Seville", prob_top3=["Seville", "Lyles"])
    assert ranked[0]["type"] == "rivalry"


def test_rivalry_led_by_the_model_pick_is_not_treated_as_surprising():
    story = _story("rivalry", "12-0", "Seville", "Lyles")
    assert api.storyline_surprise(story, "Seville", ["Seville"]) == api.SURPRISE_NOTABLE


def test_level_head_to_head_record_has_no_leader_to_contradict_the_model():
    story = _story("rivalry", "5-5", "Lyles", "Seville")
    assert api.storyline_surprise(story, "Seville", ["Seville"]) == api.SURPRISE_NOTABLE


def test_injury_flagged_favourite_outranks_an_injury_flagged_also_ran():
    favourite = _story("injury_watch", "#1", "Duplantis")
    also_ran = _story("injury_watch", "#7", "Somebody")
    assert api.storyline_surprise(favourite, "Duplantis", ["Duplantis"]) > api.storyline_surprise(
        also_ran, "Duplantis", ["Duplantis"]
    )


def test_debutant_projected_to_win_outranks_a_debutant_further_down():
    winning = _story("debutant", "#1", "Newcomer")
    third = _story("debutant", "#3", "Newcomer")
    assert api.storyline_surprise(winning, "Newcomer", ["Newcomer"]) == api.SURPRISE_DEFIES_PRIOR
    assert api.storyline_surprise(third, "Favourite", ["Favourite"]) == api.SURPRISE_CONTEXT


def test_head_to_head_counter_evidence_outranks_the_same_athletes_debut():
    """The real Men's 100m case: Seville is the model's pick AND a first-time
    finalist AND 0-12 against Lyles. A debut is an absence of prior finals,
    not evidence against the pick, so the head-to-head has to win the
    featured slot -- these used to tie and lose on generator order."""
    stories = [
        _story("debutant", "#2", "Oblique SEVILLE"),
        _story("rivalry", "12-0", "Noah LYLES", "Oblique SEVILLE"),
    ]
    ranked = api.rank_storylines(
        stories, prob_leader="Oblique SEVILLE", prob_top3=["Oblique SEVILLE", "Noah LYLES"]
    )
    assert [s["type"] for s in ranked] == ["rivalry", "debutant"]


def test_tight_photo_finish_outranks_a_looser_one():
    tight = _story("photo_finish", "1pt gap", "A", "B")
    loose = _story("photo_finish", "5pt gap", "A", "B")
    assert api.storyline_surprise(tight, "A", ["A"]) > api.storyline_surprise(loose, "A", ["A"])


def test_equally_surprising_storylines_keep_the_generators_original_order():
    stories = [
        _story("returning_champion", "2024", "Champ"),
        _story("hot_streak", "-0.20s", "Riser"),
    ]
    ranked = api.rank_storylines(stories, "Champ", ["Champ"])
    assert [s["type"] for s in ranked] == ["returning_champion", "hot_streak"]


def test_ranking_still_caps_the_list_but_promotes_before_it_truncates():
    stories = [
        _story("debutant", "#3", "Third"),
        _story("returning_champion", "2024", "Champ"),
        _story("hot_streak", "-0.20s", "Riser"),
        _story("photo_finish", "4pt gap", "Fav", "Second"),
        _story("rivalry", "9-1", "Challenger", "Fav"),
    ]
    ranked = api.rank_storylines(stories, prob_leader="Fav", prob_top3=["Fav", "Challenger"])
    assert len(ranked) == 4
    assert ranked[0]["type"] == "rivalry"
    # The unsurprising #3 debutant is the one that falls off the end.
    assert "debutant" not in [s["type"] for s in ranked]


# --- The model's pick vs. the best mark (2026-08-24) ----------------------
# `athletes` is ordered by real season-best mark, which disagrees with win
# probability in 15 of 32 real disciplines. Reading athletes[0] as "the
# favourite" has been a bug in four separate places in this project; these
# pin the shared helper that replaced the last two.

def _disc_two(label, first, second):
    """A discipline whose best MARK and best PROBABILITY are different people,
    in the real list order (mark-sorted, so the weaker pick comes first)."""
    def athlete(name, prob):
        return {
            "name": name, "mark": "10.00", "prob": prob, "waUrl": "https://x",
            "injuryWatch": False, "injuryReason": None, "injuryUrl": None,
        }
    return {"id": label, "label": label, "athletes": [athlete(*first), athlete(*second)]}


def test_discipline_favourite_picks_highest_probability_not_best_mark():
    disc = _disc_two("Men's 100m", ("Noah LYLES", 16), ("Oblique SEVILLE", 27))
    assert api.discipline_favourite(disc)["name"] == "Oblique SEVILLE"


def test_discipline_favourite_breaks_ties_on_season_best_mark():
    disc = _disc_two("Men's 100m", ("Best Mark", 20), ("Slower Mark", 20))
    assert api.discipline_favourite(disc)["name"] == "Best Mark"


def test_discipline_favourite_returns_none_for_an_empty_field():
    assert api.discipline_favourite({"label": "Empty", "athletes": []}) is None


def test_top_winners_names_the_model_pick_not_the_fastest_athlete():
    discs = [_disc_two("Men's 100m", ("Noah LYLES", 16), ("Oblique SEVILLE", 27))]
    winners = api.build_top_winners(discs, [])
    assert [w["name"] for w in winners] == ["Oblique SEVILLE"]
    assert winners[0]["prob"] == 27


def test_top_winners_no_longer_drops_a_discipline_by_under_reporting_it():
    """The real regression: Winfred Yavi at 52% was missing from the dashboard
    entirely because her discipline was scored on its best mark (31%)."""
    discs = [
        _disc_two("Women's 3000m SC", ("Peruth CHEMUTAI", 31), ("Winfred YAVI", 52)),
        _discipline("Filler A", prob=45),
        _discipline("Filler B", prob=40),
    ]
    winners = api.build_top_winners(discs, [])
    assert winners[0]["name"] == "Winfred YAVI"
    assert winners[0]["prob"] == 52


def test_confidence_scores_a_discipline_on_its_top_pick_not_its_top_mark():
    discs = [_disc_two("Women's 3000m SC", ("Peruth CHEMUTAI", 31), ("Winfred YAVI", 52))]
    assert api.build_confidence(discs, []) == [
        {"disc": "Women's 3000m SC", "discKey": "Women's 3000m SC", "value": 52}
    ]


def test_confidence_carries_the_discipline_key_for_linking():
    """The dashboard links its least-sure rows straight to the discipline
    page, so the key has to travel with the label rather than the caller
    matching on a display string."""
    discs = [_disc_two("men_100m", ("Noah LYLES", 16), ("Oblique SEVILLE", 27))]
    assert api.build_confidence(discs, [])[0]["discKey"] == "men_100m"


def test_confidence_survives_a_discipline_with_no_id():
    """Costs the link, not the whole /api/predictions payload."""
    row = api.build_confidence([_discipline("No id here", prob=40)], [])[0]
    assert row["discKey"] is None
    assert row["value"] == 40


# --- Athlete search + "why not in the field" (2026-08-25) -----------------
# predictions_latest.csv holds ~230 projected finalists out of ~3,700 ranked
# athletes, so before this an athlete outside the field simply had no page.
# The question that exposed it: "why isn't Lyles in the 100m?" -- he is world
# #1 at 9.79 and genuinely ineligible (no Diamond League points in the
# event). These pin the honest-reason logic, which mirrors run.py's real
# selection order rather than guessing.

def _status_env(monkeypatch, *, in_field=False, standings=None, injury=None,
                toplist=("9.79", 1), detail=None, bio=None, scored=None,
                field_names=None, h2h=None):
    monkeypatch.setattr(api, "load_standings", lambda: standings or {})
    # The FULL standings table, which knows the difference between "has no
    # Diamond League points" and "has points but is below the cut". Stubbed
    # empty by default so each test states its own case rather than
    # depending on whatever the last scrape happened to put on disk.
    monkeypatch.setattr(api, "load_standings_detail", lambda: detail or {})
    # The profile-enrichment lookups read real files (data/raw/*.csv,
    # outputs/predictions_latest.csv) -- stubbed so these tests describe
    # their own case instead of quietly depending on the last scrape.
    monkeypatch.setattr(api, "toplist_bio", lambda k, n: bio or {})
    monkeypatch.setattr(api, "scored_prediction_row", lambda k, n: scored)
    monkeypatch.setattr(api, "projected_field_names", lambda k, limit=6: field_names or [])
    monkeypatch.setattr(api, "load_h2h_vs_rivals", lambda k, n, rivals: h2h or [])
    monkeypatch.setattr(api, "load_injury_flags", lambda: injury or {})
    monkeypatch.setattr(api, "toplist_entry", lambda k, n: (*toplist, "https://wa/x"))
    monkeypatch.setattr(api, "load_athlete_history", lambda k, n: ([], None))
    if in_field:
        disc = {"id": "men_100m", "label": "Men's 100m",
                "athletes": [{"name": "Noah LYLES", "mark": "9.79"}]}
        monkeypatch.setattr(api, "load_predictions", lambda: ([disc], []))
    else:
        monkeypatch.setattr(api, "load_predictions", lambda: ([], []))


def test_athlete_in_the_field_reports_no_exclusion_reason(monkeypatch):
    _status_env(monkeypatch, in_field=True)
    out = api.athlete_field_status("men_100m", "Noah LYLES")
    assert out["inField"] is True
    assert out["reason"] is None


def _detail(*rows, limit=8):
    """A men's 100m standings table, [(rank, name, points), ...]."""
    return {"disciplines": {"men_100m": {
        "qualLimit": limit,
        "standings": [{"rank": r, "name": n, "country": "USA", "events": 2, "points": p}
                      for r, n, p in rows],
    }}}


def test_an_athlete_with_no_dl_points_at_all_is_told_exactly_that(monkeypatch):
    """Fastest in the world, never scored in the event -- genuinely absent
    from WA's standings table, not merely below the cut."""
    _status_env(monkeypatch, standings={"men_100m": ["Oblique SEVILLE"]},
                detail=_detail((1, "Oblique SEVILLE", 23)))
    out = api.athlete_field_status("men_100m", "Noah LYLES")
    assert out["inField"] is False
    assert out["reasonCode"] == "not_in_standings"
    assert out["dl"] is None
    assert out["worldRank"] == 1


def test_an_athlete_below_the_cut_is_not_told_they_have_no_points(monkeypatch):
    """The real 2026-08-25 bug. standings.json is truncated to the
    qualifying places, so being absent from it was read as "never scored" --
    and the site told readers Noah Lyles had no Diamond League points in the
    100m while he sat 9th on 15, two short of the cut."""
    _status_env(
        monkeypatch,
        standings={"men_100m": ["Oblique SEVILLE"]},
        detail=_detail(*[(i + 1, f"A{i + 1}", 30 - i) for i in range(8)],
                       (9, "Noah LYLES", 15), limit=8),
    )
    out = api.athlete_field_status("men_100m", "Noah LYLES")
    assert out["reasonCode"] == "outside_points_cut"
    assert out["dl"]["points"] == 15
    assert out["dl"]["rank"] == 9
    assert "15 points" in out["reason"]
    assert "no Diamond League points" not in out["reason"]


def test_injury_removal_is_reported_with_its_evidence(monkeypatch):
    """The real Hocker case -- and the page shows the headline, so a bad
    flag is visible to the reader instead of silently deleting an athlete."""
    injury = {"Cole Hocker": {"status": "remove", "matches": [
        {"headline": "Some headline", "url": "https://x", "source": "letsrun"}]}}
    _status_env(monkeypatch, standings={"men_1500m": ["Cole HOCKER"]}, injury=injury)
    out = api.athlete_field_status("men_1500m", "Cole HOCKER")
    assert out["reasonCode"] == "injury_removed"
    assert "Some headline" in (out["injuryReason"] or "")


def test_in_standings_but_outside_the_cut(monkeypatch):
    _status_env(monkeypatch, standings={"men_100m": ["Noah LYLES"]})
    out = api.athlete_field_status("men_100m", "Noah LYLES")
    assert out["reasonCode"] == "outside_cut"


def test_no_season_mark_is_its_own_reason(monkeypatch):
    _status_env(monkeypatch, standings={"men_100m": ["Noah LYLES"]}, toplist=(None, None))
    out = api.athlete_field_status("men_100m", "Noah LYLES")
    assert out["reasonCode"] == "no_data"


def test_search_needs_at_least_two_characters():
    assert api.search_athletes("") == []
    assert api.search_athletes("a") == []


# --- Near-miss athletes must never contaminate the predictions ------------
# run.py now exports athletes beyond the confirmed DL field with
# dl_qualified = False so the site can show "who'd be a threat if they got
# in". Every model-derived figure (top winners, confidence, storylines, the
# favourite) reads disc["athletes"], so a near-miss athlete leaking into
# that list would be presented as a projected finalist.

def _disc_with_near_miss():
    def a(name, prob, rank):
        return {"name": name, "mark": "9.80", "prob": prob, "waUrl": "https://x",
                "rank": rank, "injuryWatch": False, "injuryReason": None, "injuryUrl": None}
    return {
        "id": "men_100m", "label": "Men's 100m",
        "athletes": [a("Qualified One", 30, 1), a("Qualified Two", 20, 2)],
        "nearMiss": [a("Noah LYLES", 95, 1)],  # deliberately the highest prob
    }


def test_top_winners_ignores_near_miss_athletes():
    """A near-miss athlete with a huge probability must not become the
    'model's #1 pick' -- he is not in the Final."""
    winners = api.build_top_winners([_disc_with_near_miss()], [])
    names = [w["name"] for w in winners]
    assert "Noah LYLES" not in names
    assert names == ["Qualified One"]


def test_confidence_ignores_near_miss_athletes():
    conf = api.build_confidence([_disc_with_near_miss()], [])
    assert conf == [{"disc": "Men's 100m", "discKey": "men_100m", "value": 30}]


def test_discipline_favourite_ignores_near_miss_athletes():
    fav = api.discipline_favourite(_disc_with_near_miss())
    assert fav["name"] == "Qualified One"


def test_a_removal_with_no_headline_still_appears_in_the_news(monkeypatch):
    """The news feed is now the only place withdrawn athletes are listed --
    the dashboard's separate "Removed from predictions" panel was pure
    duplication and was deleted. So a removal must never be droppable just
    because its match carries no usable headline."""
    monkeypatch.setattr(api, "load_injury_flags", lambda: {
        "Ghost Athlete": {
            "status": "remove",
            "disciplines": ["men_1500m"],
            "matches": [{"url": "https://x", "keywords": ["withdraws"]}],  # no headline
        }
    })
    items = api.build_news()
    assert [i["athlete"] for i in items] == ["Ghost Athlete"]
    assert items[0]["status"] == "remove"
    assert items[0]["keywords"] == ["withdraws"]


def test_news_still_dedupes_one_article_across_several_athletes(monkeypatch):
    shared = {"headline": "Two athletes out", "url": "https://same", "source": "letsrun"}
    monkeypatch.setattr(api, "load_injury_flags", lambda: {
        "A": {"status": "watch", "disciplines": [], "matches": [dict(shared)]},
        "B": {"status": "watch", "disciplines": [], "matches": [dict(shared)]},
    })
    assert len(api.build_news()) == 1


# ---- non-qualified athlete profiles (HANDOFF 0k) ----
# These pages used to be a stub: a reason, a season best and a chart. The
# same real stats the in-field profile shows are available for most of these
# athletes and are now carried over. What is NOT carried over is the
# finalist-only material -- a predicted rank, and a probability presented as
# a forecast about a Final they are not in.

def test_a_scored_near_miss_athlete_gets_the_full_season_stats(monkeypatch):
    import pandas as pd
    row = pd.Series({
        "athlete_name": "Noah LYLES", "nationality": "USA", "season_best": "9.79",
        "win_probability": "7%", "career_best": 9.79, "pb_gap": 0.0, "age": 29.1,
        "meets_count": 2, "days_since_last": 58.0,
    })
    _status_env(monkeypatch, standings={"men_100m": ["Oblique SEVILLE"]}, scored=row)
    out = api.athlete_field_status("men_100m", "Noah LYLES")
    assert out["careerBest"] == 9.79
    assert out["age"] == 29.1
    assert out["meetsCount"] == 2
    assert out["nat"] == "USA"


def test_a_scored_near_miss_athletes_probability_is_marked_hypothetical(monkeypatch):
    """run.py really does score this athlete with the same forest, so the
    number is real -- but it is a conditional one. It travels under a name
    the finalist payload doesn't use, so it cannot be rendered as `prob`
    and read as a forecast about the Final."""
    import pandas as pd
    row = pd.Series({"athlete_name": "Noah LYLES", "nationality": "USA",
                     "win_probability": "7%", "career_best": 9.79, "pb_gap": 0.0,
                     "age": 29.1, "meets_count": 2, "days_since_last": 58.0})
    _status_env(monkeypatch, standings={"men_100m": ["Oblique SEVILLE"]}, scored=row)
    out = api.athlete_field_status("men_100m", "Noah LYLES")
    assert out["hypotheticalProb"] == 7
    assert "prob" not in out
    assert "rank" not in out


def test_an_unscored_athlete_still_gets_name_and_age_from_the_toplist(monkeypatch):
    """Anyone below the near-miss group has no prediction row at all
    (Ingebrigtsen, 14th in the 1500m standings, is the live case). Facts the
    scrape already collected shouldn't be blank because of that.

    This assertion used to read `careerBest is None`, which encoded the old
    behaviour: no prediction row meant no career best. That was reported as
    a bug against Dina Asher-Smith, who is not in predictions_latest.csv at
    all yet has 41 races and eight seasons on record. Only the SCORE needs
    the model, so `hypotheticalProb` stays None while career best is now
    derived from the race log. Career best is asserted to be either a real
    mark or None (the fixture here has no race log to read), never to be
    absent by construction."""
    _status_env(monkeypatch, standings={"men_1500m": ["Yared NUGUSE"]},
                bio={"nat": "NOR", "age": 25.9})
    out = api.athlete_field_status("men_1500m", "Jakob INGEBRIGTSEN")
    assert out["nat"] == "NOR"
    assert out["age"] == 25.9
    assert out["careerBest"] is None or isinstance(out["careerBest"], str)
    # The one thing that genuinely cannot exist without a prediction row.
    assert out["hypotheticalProb"] is None


def test_head_to_head_is_measured_against_the_athletes_who_qualified(monkeypatch):
    """The opponents are the projected field, not the near-miss group -- the
    question this panel answers is "how do they do against the ones who got
    in?"."""
    captured = {}

    def fake_h2h(disc_key, name, rivals):
        captured["rivals"] = rivals
        return [{"opponent": rivals[0], "wins": 12, "losses": 0, "meetings": 12}]

    monkeypatch.setattr(api, "load_h2h_vs_rivals", fake_h2h)
    _status_env(monkeypatch, standings={"men_100m": ["Oblique SEVILLE"]},
                field_names=["Oblique SEVILLE", "Akani SIMBINE"])
    monkeypatch.setattr(api, "load_h2h_vs_rivals", fake_h2h)
    out = api.athlete_field_status("men_100m", "Noah LYLES")
    assert captured["rivals"] == ["Oblique SEVILLE", "Akani SIMBINE"]
    assert out["h2h"][0]["opponent"] == "Oblique SEVILLE"
