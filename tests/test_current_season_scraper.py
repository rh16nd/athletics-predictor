"""Regression test for a real bug found and fixed 2026-08-24: the
"recognized athlete" filter current_season_scraper.py uses to drop
field-fillers only checked a discipline's historical toplist
(2018-2025), which silently dropped real, currently-ranked athletes with
no multi-year history in that specific discipline -- confirmed live for
Femke Bol, who switched from the 400m Hurdles to the 800m for 2026 and is
genuinely ranked #3 in predictions_latest.csv, but has zero historical
presence in women_800m.csv, so every one of her real 2026 races was being
filtered out as "unrecognized."

get_recognized_names() was split out specifically so this fix (union the
historical set with this season's own live toplist) is unit-testable
without a live scrape -- find_season_meetings()/scrape_meeting() (the rest
of current_season_scraper.py) both hit the real World Athletics API and
aren't exercised here.
"""
import os

import current_season_scraper as ccs
import dl_final_results_scraper as dlr

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "raw2026")


def test_historical_only_filter_misses_the_event_switcher():
    """Confirms the bug actually existed as understood: an athlete with no
    historical presence in this discipline (Epsilon SWITCHER, standing in
    for Femke Bol's real 2026 event switch) is invisible to the old,
    historical-only check on its own."""
    recognized = dlr.load_recognized_names("women_800m", FIXTURES_DIR)
    assert "Delta VETERAN" in recognized
    assert "Epsilon SWITCHER" not in recognized


def test_get_recognized_names_includes_current_season_toplist_athletes():
    """The real fix: unioning in this season's own live toplist recovers
    the event-switcher/breakout athlete without losing anyone the
    historical check already covered."""
    recognized = ccs.get_recognized_names("women_800m", 2026, FIXTURES_DIR)
    assert "Delta VETERAN" in recognized  # still recognized via history
    assert "Epsilon SWITCHER" in recognized  # now recognized via this season's own toplist


def test_get_recognized_names_missing_current_toplist_falls_back_to_historical_only():
    """men_HJ.csv (historical) exists in this fixture dir but men_HJ_2026.csv
    (this season's live toplist) does not -- confirms this doesn't crash
    when a discipline's live snapshot isn't present yet, it just can't
    recover anyone beyond the historical set."""
    recognized = ccs.get_recognized_names("men_HJ", 2026, FIXTURES_DIR)
    assert recognized == {"Zeta HIGHJUMPER"}
