"""Unit tests for src/feature_builder.py -- the real feature-building logic
extracted out of run.py (2026-08-24) specifically so it could be tested
without importing run.py directly. run.py has no `if __name__ == "__main__":`
guard, so importing it kicks off a real ~1hr live scrape (see HANDOFF.md's
"near-miss" note) -- these tests import feature_builder instead, which has
no top-level side effects at all.

Fixture design (tests/fixtures/raw2026/): mirrors the real data/raw/
directory's three-file-per-discipline layout (the live toplist snapshot
{key}_2026.csv, the real per-meeting file {key}_2026_meetings.csv, and the
historical training file {key}.csv all coexist in one directory) since
build_2026_features() reads all three by convention from one RAW_DIR.

men_100m covers the "everything present" and "in toplist but not in
meetings file" cases in one discipline: Alpha SPEEDY has real rows in all
three files (meetings file's most recent race postdates the toplist's
best-mark date, the same shape as the real Rai Benjamin bug this feature
was fixed for); Beta NOMEETINGS is in the toplist but has zero rows in the
meetings file -- a regression case for the real Jessica Hull bug
(meets_count must report a real 0, not the toplist's structural "1" every
athlete gets from a one-row-per-athlete snapshot).

women_LJ (a field event, no meetings file, no historical file at all)
covers the pre-meetings-scraper fallback path still used for any
discipline that file doesn't exist for yet.
"""
import os
from datetime import date

import pytest

import feature_builder as fb

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "raw2026")


@pytest.fixture
def fixture_raw_dir(monkeypatch):
    monkeypatch.setattr(fb, "RAW_DIR", FIXTURES_DIR)
    return fb


# ---- get_qual_limit ----

def test_get_qual_limit_field_event_is_6():
    assert fb.get_qual_limit("men_LJ") == 6


def test_get_qual_limit_long_distance_is_10():
    assert fb.get_qual_limit("men_5000m") == 10


def test_get_qual_limit_default_is_8():
    assert fb.get_qual_limit("men_100m") == 8


# ---- parse_mark ----

def test_parse_mark_seconds():
    assert fb.parse_mark("9.79") == 9.79


def test_parse_mark_minutes_seconds():
    assert fb.parse_mark("1:41.84") == pytest.approx(101.84)


def test_parse_mark_invalid_returns_none():
    assert fb.parse_mark("DNF") is None


# ---- seconds_to_time ----

def test_seconds_to_time_field_event():
    assert fb.seconds_to_time(8.66, "men_LJ") == "8.66m"


def test_seconds_to_time_middle_distance():
    assert fb.seconds_to_time(101.84, "men_800m") == "1:41.84"


def test_seconds_to_time_sprint():
    assert fb.seconds_to_time(9.79, "men_100m") == "9.79"


# ---- build_2026_features ----

def test_build_2026_features_missing_toplist_returns_empty_dataframe(fixture_raw_dir):
    assert fixture_raw_dir.build_2026_features("men_800m").empty


def test_build_2026_features_uses_real_meetings_file_not_toplist_snapshot(fixture_raw_dir):
    """Alpha SPEEDY: real regression case for the "days since last" bug --
    the toplist's own row is dated 10 Aug (that happened to be their best
    mark), but their real most recent race (from the meetings file) was 20
    Aug. days_since_last must reflect the real last race, not the best-mark
    date. meets_count must be the real per-meeting count (3), not the
    toplist's structural 1."""
    feat = fixture_raw_dir.build_2026_features("men_100m")
    row = feat[feat["athlete_name"] == "Alpha SPEEDY"].iloc[0]

    assert row["meets_count"] == 3
    expected_days = (date.today() - date(2026, 8, 20)).days
    assert row["days_since_last"] == expected_days

    # career_best from the historical file (min across 2023/2025 = 9.75,
    # track event), not the toplist's own current-season mark (9.85).
    assert row["career_best"] == pytest.approx(9.75)
    assert row["pb_gap"] == pytest.approx(0.10, abs=1e-6)
    # yoy_improvement: prev_season_best (2025 = 9.90) - season_best (9.85),
    # positive means the athlete got faster.
    assert row["yoy_improvement"] == pytest.approx(0.05, abs=1e-6)


def test_build_2026_features_athlete_absent_from_meetings_file_gets_real_zero(fixture_raw_dir):
    """Regression test for the real Jessica Hull bug: an athlete who is in
    the toplist but has zero real rows in the meetings file must get a real
    verified meets_count of 0, not the toplist's structural "1" (every
    athlete gets exactly one row in a season-best snapshot regardless of
    how many times they actually raced)."""
    feat = fixture_raw_dir.build_2026_features("men_100m")
    row = feat[feat["athlete_name"] == "Beta NOMEETINGS"].iloc[0]

    assert row["meets_count"] == 0
    assert row["days_since_last"] == 999  # real sentinel: no current-season meeting data on record


def test_build_2026_features_field_event_falls_back_to_toplist_without_meetings_file(fixture_raw_dir):
    """women_LJ has no _2026_meetings.csv and no historical .csv fixture at
    all -- exercises the pre-meetings-scraper fallback path any discipline
    still uses before current_season_scraper.py has ever been run for it."""
    feat = fixture_raw_dir.build_2026_features("women_LJ")
    row = feat[feat["athlete_name"] == "Gamma JUMPER"].iloc[0]

    assert row["meets_count"] == 1  # structural fallback: one row in the toplist snapshot
    assert row["career_best"] == pytest.approx(row["season_best"])  # no historical file to diverge from
    assert row["season_best"] == pytest.approx(7.05)
    # Falls back to computing days_since_last from the toplist's own Date
    # column (05 Jul 2026) rather than the 999 sentinel, since the toplist
    # itself does carry a real date even with no dedicated meetings file.
    expected_days = (date.today() - date(2026, 7, 5)).days
    assert row["days_since_last"] == expected_days
