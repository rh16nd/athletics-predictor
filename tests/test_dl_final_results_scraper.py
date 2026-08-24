"""Unit tests for src/dl_final_results_scraper.py's pure name-mapping logic.

Doesn't test scrape_year()/graphql() -- those need a live network call to
World Athletics' API. strip_gender_prefix/resolve_discipline_key were
extracted specifically to make this real bug testable: the mapping table
used to be keyed on the discipline name alone ("100 Metres") while the
API's actual event field includes the "Men's "/"Women's " prefix
("Men's 100 Metres"), so every single lookup silently missed until this
was found and fixed.
"""
import os

import dl_final_results_scraper as scraper
import train_model as tm


def test_strip_gender_prefix_removes_leading_possessive():
    assert scraper.strip_gender_prefix("Men's 100 Metres") == "100 Metres"
    assert scraper.strip_gender_prefix("Women's High Jump") == "High Jump"


def test_resolve_discipline_key_matches_real_api_event_names():
    assert scraper.resolve_discipline_key("M", "Men's 100 Metres") == "men_100m"
    assert scraper.resolve_discipline_key("W", "Women's Pole Vault") == "women_PV"


def test_mile_counts_as_1500m_only_when_the_caller_asks():
    """The DL Final substitutes the Mile for the 1500m in some years, so the
    Final labeller must treat them as one event or it loses real ground-truth
    labels. Every other caller builds a per-meeting time series, where a Mile
    is ~16-17s slower and mixing them corrupts the series -- see the module's
    MILE_AS_1500_KEY comment for the real case this came from."""
    assert scraper.resolve_discipline_key("M", "Men's Mile", mile_as_1500=True) == "men_1500m"
    assert scraper.resolve_discipline_key("W", "Women's Mile", mile_as_1500=True) == "women_1500m"


def test_mile_is_not_a_1500m_by_default():
    # The default is the safe one: opting IN is what the Final scraper does.
    assert scraper.resolve_discipline_key("M", "Men's Mile") is None
    assert scraper.resolve_discipline_key("W", "Women's Mile") is None


def test_a_real_1500m_resolves_either_way():
    for flag in (True, False):
        assert scraper.resolve_discipline_key("M", "Men's 1500 Metres", mile_as_1500=flag) == "men_1500m"
        assert scraper.resolve_discipline_key("W", "Women's 1500 Metres", mile_as_1500=flag) == "women_1500m"


def test_the_flag_does_not_leak_into_other_events():
    # Guards against a future edit routing everything through the override map.
    assert scraper.resolve_discipline_key("M", "Men's 100 Metres", mile_as_1500=True) == "men_100m"
    assert scraper.resolve_discipline_key("M", "Men's 4x100 Metres Relay", mile_as_1500=True) is None


def test_per_meeting_scrapers_do_not_opt_into_the_mile_substitution():
    """The bug was not in the mapping itself but in WHO shared it: the two
    per-meeting scrapers call the same helper the Final labeller does. Reading
    their source keeps that guarantee visible even though the call sites pass
    no flag at all."""
    # Read the files rather than importing them: both set
    # sys.stdout = TextIOWrapper(...) at module scope, which detaches the
    # stream pytest is capturing into and tears down the whole session.
    src_dir = os.path.join(os.path.dirname(__file__), "..", "src")
    for filename in ("season_results_scraper.py", "major_meets_scraper.py"):
        with open(os.path.join(src_dir, filename), encoding="utf-8") as fh:
            body = fh.read()
        assert "resolve_discipline_key" in body, filename
        assert "mile_as_1500=True" not in body, (
            f"{filename} must not opt into Mile-as-1500m: its rows are a time series"
        )


def test_resolve_discipline_key_returns_none_for_unmapped_events():
    # Relays, non-standard extra races (e.g. a flat "3000 Metres" alongside
    # the Mile), and 5km road-race substitutes for the 5000m are all
    # deliberately unmapped -- absence here is what makes a discipline
    # "not contested that year" auto-detected rather than hand-flagged.
    assert scraper.resolve_discipline_key("M", "Men's 4x100 Metres Relay") is None
    assert scraper.resolve_discipline_key("M", "Men's 5 Kilometres Road") is None
    assert scraper.resolve_discipline_key("M", "Men's 3000 Metres") is None


def test_wa_event_to_key_covers_every_trained_discipline():
    mapped_keys = set(scraper.WA_EVENT_TO_KEY.values())
    assert set(tm.TRAIN_DISCIPLINES.keys()) == mapped_keys
