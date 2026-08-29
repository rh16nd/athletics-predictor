"""
Honours, world ranking and personal bests as World Athletics states them
(src/athlete_career.py).

Synthetic profiles throughout: data/athlete_profiles/ is gitignored and
regenerable, so tests asserting on it would pass or fail depending on when
it was last fetched.

The headline tests carry the weight. That one line renders the word
"champion" next to an athlete's name, and the data behind it mixes Olympic
titles with national ones, NCAA titles and World U20 golds indiscriminately
-- 354 national golds against 20 Olympic ones across the surveyed field.
Getting the tiers wrong would not look like a bug, it would look like a
claim.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import athlete_career as ac  # noqa: E402


def _profile(honour_groups=(), rankings=(), bests=()):
    return {
        "honours": [
            {"categoryName": name,
             "results": [{"competition": name, "mark": "45.00", "place": f"{p}."} for p in places]}
            for name, places in honour_groups
        ],
        "worldRankings": {"current": [{"eventGroup": e, "place": p} for e, p in rankings]},
        "personalBests": {"results": list(bests)},
    }


# ---- headline tiers ----

def test_global_titles_headline():
    groups = ac.honours(_profile([("Olympic Games", [1, 1]), ("World Championships", [1])]))
    assert ac.headline(groups) == "2× Olympic champion · World champion"


def test_a_single_medal_is_not_pluralised():
    groups = ac.honours(_profile([("Olympic Games", [1])]))
    assert ac.headline(groups) == "Olympic champion"


def test_only_the_best_colour_in_a_category_is_said():
    """An athlete with gold and bronze at the same championship is its
    champion; listing both would read as hedging."""
    groups = ac.honours(_profile([("World Championships", [1, 3, 3])]))
    assert ac.headline(groups) == "World champion"


def test_silver_and_bronze_are_named_not_softened():
    groups = ac.honours(_profile([("World Championships", [2, 2])]))
    assert ac.headline(groups) == "2× World silver"


def test_national_titles_never_headline():
    """354 national golds appear across the surveyed field against 20
    Olympic ones. Rendering "champion" off one would be a claim, not a
    stat."""
    groups = ac.honours(_profile([("National Championships", [1, 1, 1, 1, 1])]))
    assert ac.headline(groups) is None


@pytest.mark.parametrize("category", [
    "World U20 Championships", "European U23 Championships",
    "World U18 Championships", "NCAA Championships",
    "National Indoor Championships",
])
def test_age_group_and_collegiate_titles_never_headline(category):
    groups = ac.honours(_profile([(category, [1, 1])]))
    assert ac.headline(groups) is None


def test_diamond_league_meeting_wins_never_headline():
    """"Diamond League" without "Final" counts individual MEETING wins --
    392 of them across the field. A meeting win is not a title."""
    groups = ac.honours(_profile([("Diamond League", [1] * 11)]))
    assert ac.headline(groups) is None


def test_diamond_league_final_does_headline():
    groups = ac.honours(_profile([("Diamond League Final", [1, 1])]))
    assert ac.headline(groups) == "2× Diamond League Final champion"


# ---- the continental fallback ----

def test_continental_titles_are_used_only_when_there_is_nothing_global():
    groups = ac.honours(_profile([
        ("Olympic Games", [1]), ("Commonwealth Games", [1]),
    ]))
    assert ac.headline(groups) == "Olympic champion"


def test_a_continental_title_headlines_when_nothing_global_exists():
    groups = ac.honours(_profile([("Commonwealth Games", [1])]))
    assert ac.headline(groups) == "Commonwealth champion"


def test_a_continental_title_is_always_named():
    """Never shortened to "champion" -- the whole point of the second tier
    is that a reader can tell it apart from a global one at a glance."""
    for category, expected in [("European Championships", "European champion"),
                               ("African Championships", "African champion"),
                               ("Asian Games", "Asian Games champion")]:
        groups = ac.honours(_profile([(category, [1])]))
        assert ac.headline(groups) == expected


def test_no_podium_anywhere_gets_no_consolation_phrase():
    groups = ac.honours(_profile([("Olympic Games", [6, 7]), ("World Championships", [5])]))
    assert ac.headline(groups) is None


# ---- ordering ----

def test_honours_are_ordered_by_championship_not_by_medal_count():
    """Sorting on gold count alone put "Diamond League 11" and "National
    Championships 5" above "Olympic Games 2" on a real athlete's page,
    reading as though eleven meeting wins outrank two Olympic titles."""
    groups = ac.honours(_profile([
        ("Diamond League", [1] * 11),
        ("National Championships", [1] * 5),
        ("Olympic Games", [1, 1]),
        ("World Championships", [1, 2]),
    ]))
    assert [g["category"] for g in groups][:3] == [
        "Olympic Games", "World Championships", "Diamond League",
    ]


def test_medal_counts_only_count_podiums():
    groups = ac.honours(_profile([("Olympic Games", [1, 2, 3, 4, 8])]))
    g = groups[0]
    assert (g["gold"], g["silver"], g["bronze"], g["podiums"]) == (1, 1, 1, 3)
    assert len(g["results"]) == 5, "a non-podium finish stays in the detail"


# ---- world ranking ----

def test_world_ranking_sorts_by_place_and_separates_the_overall_row():
    r = ac.world_ranking(_profile(rankings=[
        ("Men's 200m", 5), ("Men's Overall Ranking", 22), ("Men's 100m", 2),
    ]))
    assert [e["event"] for e in r["events"]] == ["Men's 100m", "Men's 200m"]
    assert r["best"]["place"] == 2
    assert r["overall"] == 22


def test_world_ranking_is_empty_not_crashing_without_data():
    r = ac.world_ranking({})
    assert r["events"] == [] and r["best"] is None and r["overall"] is None


# ---- personal bests ----

def test_indoor_personal_bests_are_flagged():
    bests = ac.personal_bests(_profile(bests=[
        {"discipline": "60 Metres", "mark": "6.43", "venue": "Albuquerque, NM (USA) (i)", "date": "17 FEB 2024"},
        {"discipline": "100 Metres", "mark": "9.79", "venue": "Stade de France, Paris (FRA)", "date": "04 AUG 2024"},
    ]))
    assert bests[0]["indoor"] is True
    assert bests[1]["indoor"] is False


def test_unknown_athlete_has_no_career_block():
    assert ac.build_career("Nobody At All") is None
