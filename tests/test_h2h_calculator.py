"""Regression tests for the head-to-head calculator (HANDOFF 0o).

The bug these pin was live for months and reached a user-facing panel:
`h2h_rates.csv` claimed Yared Nuguse had beaten Jakob Ingebrigtsen 12-0 in
the 1500m. Three causes compounded, and each gets a test here.

The fixture shape is the real 2023 World Championships one that made the
false record: a meet page carries heats, semis and the final as separate
rows, `h2h_scraper.py` does not record which round a row came from, so
Ingebrigtsen appears as [1, 13] and Nuguse as [4, 1, 5]. The old code kept
each athlete's LAST row and compared 13 against 5.
"""
import pandas as pd

import h2h_calculator as hc


def results(*rows):
    """rows of (meet, athlete, place) for one discipline."""
    return pd.DataFrame(
        [{"discipline": "men_1500m", "meet": m, "athlete": a, "place": p} for m, a, p in rows]
    )


WORLDS = results(
    ("2023 World Championships", "Jakob Ingebrigtsen", 1.0),
    ("2023 World Championships", "Jakob Ingebrigtsen", 13.0),
    ("2023 World Championships", "Yared Nuguse", 4.0),
    ("2023 World Championships", "Yared Nuguse", 1.0),
    ("2023 World Championships", "Yared Nuguse", 5.0),
)


def rec(h2h, a, b):
    # calculate_h2h returns a bare DataFrame() when nothing was comparable,
    # so it has no columns to filter on.
    if h2h.empty:
        return None
    hit = h2h[(h2h["athlete_a"] == a) & (h2h["athlete_b"] == b)]
    return None if hit.empty else (int(hit.iloc[0]["wins"]), int(hit.iloc[0]["meetings"]))


# ---- cause 1: only the last row per athlete survived ----

def test_an_athletes_best_place_is_the_one_that_counts():
    kept = hc.one_row_per_athlete(WORLDS[WORLDS["athlete"] == "Yared Nuguse"])
    assert len(kept) == 1
    assert kept.iloc[0]["place"] == 1.0


def test_a_heat_number_can_no_longer_stand_in_for_a_final_placing():
    """The exact false record: comparing Ingebrigtsen's 13 against Nuguse's
    5 handed Nuguse a championship Ingebrigtsen won. Both actually won a
    round, so the honest answer is now no record at all -- not a wrong one."""
    h2h = hc.calculate_h2h(WORLDS)
    assert rec(h2h, "Yared Nuguse", "Jakob Ingebrigtsen") is None
    assert rec(h2h, "Jakob Ingebrigtsen", "Yared Nuguse") is None


# ---- cause 2: equal places are not a result ----

def test_equal_places_produce_no_record():
    """Two athletes on the same placing can only have been in different
    heats. Scoring it either way would invent a meeting."""
    h2h = hc.calculate_h2h(results(
        ("2024 Meet", "Alpha Speedy", 1.0),
        ("2024 Meet", "Beta Quick", 1.0),
    ))
    assert h2h.empty


# ---- cause 3: one meeting was counted many times ----

def test_one_meeting_counts_once_not_once_per_row_pair():
    """The old loop walked the ROW list, so a 2-row athlete against a 3-row
    athlete produced up to 6 "meetings" -- the same comparison repeated,
    which is why 64% of pairs with 5+ meetings came out as perfect sweeps."""
    h2h = hc.calculate_h2h(results(
        ("2024 Meet", "Alpha Speedy", 1.0),
        ("2024 Meet", "Alpha Speedy", 4.0),
        ("2024 Meet", "Beta Quick", 2.0),
        ("2024 Meet", "Beta Quick", 7.0),
        ("2024 Meet", "Beta Quick", 9.0),
    ))
    assert rec(h2h, "Alpha Speedy", "Beta Quick") == (1, 1)


# ---- the record still reads correctly in both directions ----

def test_both_directions_are_stored_and_agree():
    h2h = hc.calculate_h2h(results(
        ("2024 A", "Alpha Speedy", 1.0), ("2024 A", "Beta Quick", 2.0),
        ("2024 B", "Alpha Speedy", 3.0), ("2024 B", "Beta Quick", 1.0),
        ("2024 C", "Alpha Speedy", 1.0), ("2024 C", "Beta Quick", 5.0),
    ))
    assert rec(h2h, "Alpha Speedy", "Beta Quick") == (2, 3)
    assert rec(h2h, "Beta Quick", "Alpha Speedy") == (1, 3)


def test_rows_with_no_place_are_ignored_not_treated_as_a_win():
    h2h = hc.calculate_h2h(results(
        ("2024 A", "Alpha Speedy", 1.0),
        ("2024 A", "Beta Quick", float("nan")),
    ))
    assert h2h.empty
