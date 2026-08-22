"""Unit tests for src/dl_final_results_scraper.py's pure name-mapping logic.

Doesn't test scrape_year()/graphql() -- those need a live network call to
World Athletics' API. strip_gender_prefix/resolve_discipline_key were
extracted specifically to make this real bug testable: the mapping table
used to be keyed on the discipline name alone ("100 Metres") while the
API's actual event field includes the "Men's "/"Women's " prefix
("Men's 100 Metres"), so every single lookup silently missed until this
was found and fixed.
"""
import dl_final_results_scraper as scraper
import train_model as tm


def test_strip_gender_prefix_removes_leading_possessive():
    assert scraper.strip_gender_prefix("Men's 100 Metres") == "100 Metres"
    assert scraper.strip_gender_prefix("Women's High Jump") == "High Jump"


def test_resolve_discipline_key_matches_real_api_event_names():
    assert scraper.resolve_discipline_key("M", "Men's 100 Metres") == "men_100m"
    assert scraper.resolve_discipline_key("W", "Women's Pole Vault") == "women_PV"


def test_resolve_discipline_key_mile_maps_to_1500m():
    # Some years' Final substitutes the Mile for the standard 1500m.
    assert scraper.resolve_discipline_key("M", "Men's Mile") == "men_1500m"
    assert scraper.resolve_discipline_key("M", "Men's 1500 Metres") == "men_1500m"


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
