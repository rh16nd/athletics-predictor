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
    assert api.build_confidence(discs, []) == [{"disc": "Women's 3000m SC", "value": 52}]


# --- Athlete search + "why not in the field" (2026-08-25) -----------------
# predictions_latest.csv holds ~230 projected finalists out of ~3,700 ranked
# athletes, so before this an athlete outside the field simply had no page.
# The question that exposed it: "why isn't Lyles in the 100m?" -- he is world
# #1 at 9.79 and genuinely ineligible (no Diamond League points in the
# event). These pin the honest-reason logic, which mirrors run.py's real
# selection order rather than guessing.

def _status_env(monkeypatch, *, in_field=False, standings=None, injury=None, toplist=("9.79", 1)):
    monkeypatch.setattr(api, "load_standings", lambda: standings or {})
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


def test_not_in_dl_standings_is_reported_as_the_reason(monkeypatch):
    """The real Lyles case: fastest in the world, no DL points in the event."""
    _status_env(monkeypatch, standings={"men_100m": ["Oblique SEVILLE"]})
    out = api.athlete_field_status("men_100m", "Noah LYLES")
    assert out["inField"] is False
    assert out["reasonCode"] == "not_in_standings"
    assert out["worldRank"] == 1


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
    assert conf == [{"disc": "Men's 100m", "value": 30}]


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
