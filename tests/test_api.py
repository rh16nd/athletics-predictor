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
