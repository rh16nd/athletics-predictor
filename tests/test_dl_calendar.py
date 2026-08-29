"""
Pins api.py's hand-typed MEETS list to World Athletics' own Diamond League
calendar, and covers the split-Final case that list used to get wrong by
construction.

Everything here runs against committed snapshots
(tests/fixtures/wa_dl_calendar_*.json, regenerated with
`python src/dl_calendar.py --year YYYY --snapshot`), so no test touches the
network. The 2019 snapshot is kept specifically because that season's Final
was split across two meetings -- it is the real-data counterexample to
"the Final is the last meeting of the season".
"""
import os
import sys
from datetime import date

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import api  # noqa: E402
import dl_calendar  # noqa: E402


@pytest.fixture(scope="module")
def calendar_2026():
    return dl_calendar.load_snapshot(2026)


@pytest.fixture(scope="module")
def calendar_2019():
    return dl_calendar.load_snapshot(2019)


# ---- MEETS vs World Athletics ----

def test_meets_matches_world_athletics_calendar(calendar_2026):
    """The Schedule page renders MEETS to readers as fact. It had drifted --
    an "08 May Doha" opener that never existed, Doha's real 19 Jun date
    missing, Paris and Eugene each a day or two out -- and nothing caught
    it because MEETS only feeds display and, by late season, arithmetic
    that every past date answers identically."""
    problems = dl_calendar.diff_against_meets(calendar_2026, api.MEETS)
    assert problems == [], "api.MEETS disagrees with WA's calendar:\n  " + "\n  ".join(problems)


def test_every_meeting_world_athletics_lists_is_in_meets(calendar_2026):
    assert len(api.MEETS) == len(calendar_2026)


def test_the_final_is_whatever_world_athletics_tags_DF(calendar_2026):
    wa_finals = [m for m in calendar_2026 if m["rankingCategory"] == "DF"]
    flagged = [m for m in api.MEETS if m.get("final")]
    assert len(wa_finals) == 1, "2026 is a single Final; re-check if this changes"
    assert len(flagged) == len(wa_finals)
    assert dl_calendar.day_month(wa_finals[0]["startDate"]) == flagged[0]["date"]


# ---- the split Final ----

def test_2019_really_was_a_split_final(calendar_2019):
    """Real data, not a hypothetical: two meetings that season both carry
    ranking category DF. This is why `final` is an explicit flag rather
    than "the last entry"."""
    finals = [m for m in calendar_2019 if m["rankingCategory"] == "DF"]
    assert len(finals) == 2
    assert [m["city"] for m in finals] == ["Zürich", "Bruxelles"]


def test_both_legs_of_a_split_final_are_final_not_qualifying(calendar_2019):
    """The trap the old last-index rule set: with two DF meetings, the
    first leg is not the last entry, so it was scored as a scoring meeting
    -- and meetings_remaining() would then claim Diamond League points were
    still winnable AT a Final."""
    meets = [
        {"n": i + 1,
         "date": dl_calendar.day_month(m["startDate"]),
         "city": m["city"],
         **({"final": True} if m["rankingCategory"] == "DF" else {})}
        for i, m in enumerate(calendar_2019)
    ]
    statuses = api.compute_meet_statuses(meets, today=date(2026, 8, 1))
    finals = [m for m in statuses if m["status"] == "final"]
    assert [m["city"] for m in finals] == ["Zürich", "Bruxelles"]
    # Nothing left to qualify through once the regular season is run.
    assert api.meetings_remaining(meets, today=date(2026, 8, 26)) == 0


def test_last_entry_is_still_the_final_when_nothing_is_flagged():
    """Backward compatibility: bare lists without the flag keep the old
    meaning, which is what the other meets tests still pass."""
    meets = [
        {"n": 1, "date": "01 May", "city": "A"},
        {"n": 2, "date": "01 Sep", "city": "B — Final"},
    ]
    statuses = api.compute_meet_statuses(meets, today=date(2026, 6, 1))
    assert [m["status"] for m in statuses] == ["done", "final"]


# ---- two-day meetings ----

def test_a_two_day_meeting_is_not_done_on_its_first_morning():
    """Zürich ran 26-27 Aug. Judged on its start date it would have gone
    grey on the site while its second day was still being contested."""
    meets = [
        {"n": 1, "date": "26 Aug", "dateEnd": "27 Aug", "city": "Zürich"},
        {"n": 2, "date": "04 Sep", "city": "Brussels — Final", "final": True},
    ]
    on_day_two = api.compute_meet_statuses(meets, today=date(2026, 8, 27))
    assert on_day_two[0]["status"] == "next"
    after = api.compute_meet_statuses(meets, today=date(2026, 8, 28))
    assert after[0]["status"] == "done"


def test_meets_two_day_entries_span_consecutive_days():
    for meet in api.MEETS:
        if not meet.get("dateEnd"):
            continue
        start = api._meet_date(meet["date"])
        end = api._meet_date(meet["dateEnd"])
        assert start is not None and end is not None
        assert 0 < (end - start).days <= 2, f"{meet['city']} spans {start}..{end}"
