"""Unit tests for src/dl_final_results_scraper.py's pure name-mapping logic, and
for graphql()'s recovery when World Athletics moves its data server.

scrape_year() isn't tested -- it needs a live network call to World Athletics'
API. strip_gender_prefix/resolve_discipline_key were extracted specifically to
make this real bug testable: the mapping table used to be keyed on the
discipline name alone ("100 Metres") while the API's actual event field includes
the "Men's "/"Women's " prefix ("Men's 100 Metres"), so every single lookup
silently missed until this was found and fixed. The server-move tests at the
bottom run against a fake network.
"""
import json
import os

import pytest

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
    import feature_builder

    mapped_keys = set(scraper.WA_EVENT_TO_KEY.values())
    trained = set(tm.TRAIN_DISCIPLINES.keys())
    assert trained <= mapped_keys
    # The only keys beyond the trained set are the points-only disciplines,
    # mapped (2026-09-14) so a championship's results can be read for them.
    assert mapped_keys - trained == set(feature_builder.POINTS_ONLY_DISCIPLINES)


# ---- When World Athletics moves its data server ----------------------------
# graphql-prod-4881 stopped resolving during the 2026 Ultimate Championship and
# 4888 two days later. Every fetch failed, each failure looked like "nothing
# published yet", and the refresh button stopped working without saying why.

class _Reply:
    def __init__(self, text="", body=None):
        self.text = text
        self._body = body

    def raise_for_status(self):
        pass

    def json(self):
        if self._body is None:
            raise ValueError("not JSON")
        return self._body


def test_endpoints_in_script_reads_the_host_and_keys_from_a_page_bundle():
    bundle = ('e="https://graphql-prod-4892.edge.aws.worldathletics.org/graphql",'
              'k=["da2-rtp5hipy7bbkhab4h7k5xpjy5y","da2-qcbtpq2oifcclb773zdrfjsbsi"]')
    hosts, keys = scraper.endpoints_in_script(bundle)
    assert hosts == ["graphql-prod-4892.edge.aws.worldathletics.org"]
    assert keys == ["da2-qcbtpq2oifcclb773zdrfjsbsi", "da2-rtp5hipy7bbkhab4h7k5xpjy5y"]


def test_graphql_retries_on_the_new_server_after_a_move(monkeypatch):
    asked = []

    def fake_post(url, json=None, headers=None, timeout=None):
        asked.append(url)
        if "old-host" in url:
            raise scraper.requests.ConnectionError("name does not resolve")
        return _Reply(body={"data": {"ok": True}})

    def fake_discover():
        scraper.GRAPHQL_URL = "https://new-host/graphql"
        return True

    monkeypatch.setattr(scraper, "GRAPHQL_URL", "https://old-host/graphql")
    monkeypatch.setattr(scraper.requests, "post", fake_post)
    monkeypatch.setattr(scraper, "discover_endpoint", fake_discover)
    assert scraper.graphql("op", {}, "query") == {"ok": True}
    assert asked == ["https://old-host/graphql", "https://new-host/graphql"]


def test_an_html_error_page_is_treated_like_a_moved_server(monkeypatch):
    replies = iter([_Reply(text="<!DOCTYPE HTML>"), _Reply(body={"data": {"ok": True}})])
    looked_up = []
    monkeypatch.setattr(scraper.requests, "post", lambda *a, **k: next(replies))
    monkeypatch.setattr(scraper, "discover_endpoint", lambda: looked_up.append(1) or True)
    assert scraper.graphql("op", {}, "query") == {"ok": True}
    assert looked_up == [1]


def test_graphql_gives_up_when_no_new_server_is_found(monkeypatch):
    def fake_post(*args, **kwargs):
        raise scraper.requests.ConnectionError("down")

    monkeypatch.setattr(scraper.requests, "post", fake_post)
    monkeypatch.setattr(scraper, "discover_endpoint", lambda: False)
    with pytest.raises(scraper.requests.ConnectionError):
        scraper.graphql("op", {}, "query")


def test_a_graphql_error_in_a_real_reply_is_not_retried(monkeypatch):
    """A wrong query is our bug, not a moved server. Looking for a new server
    would hide it behind a slow, pointless lookup."""
    monkeypatch.setattr(scraper.requests, "post",
                        lambda *a, **k: _Reply(body={"errors": [{"message": "bad field"}]}))
    monkeypatch.setattr(scraper, "discover_endpoint",
                        lambda: pytest.fail("a GraphQL error must not trigger a server lookup"))
    with pytest.raises(RuntimeError):
        scraper.graphql("op", {}, "query")


def test_discovery_runs_once_per_process(monkeypatch):
    """A scrape makes dozens of calls. When WA itself is down, one lookup is
    enough; one per call would hammer the site for nothing."""
    fetched = []

    def fake_get(url, headers=None, timeout=None):
        fetched.append(url)
        raise scraper.requests.ConnectionError("site down")

    monkeypatch.setattr(scraper, "_discovery_tried", [False])
    monkeypatch.setattr(scraper.requests, "get", fake_get)
    assert scraper.discover_endpoint() is False
    assert scraper.discover_endpoint() is False
    assert fetched == [scraper.WA_HOME]


def test_a_key_answering_with_an_html_page_is_not_used(monkeypatch):
    """Only some of the keys in WA's bundle work; the rest get an HTML error
    page, which is retried briefly (CloudFront does that to bursts too) and
    then written off."""
    monkeypatch.setattr(scraper.requests, "post", lambda *a, **k: _Reply(text="<!DOCTYPE HTML>"))
    monkeypatch.setattr(scraper.time, "sleep", lambda seconds: None)
    assert scraper._answers("https://host/graphql", "da2-" + "a" * 26) is False


def test_discovery_switches_to_a_working_pair_and_remembers_it(monkeypatch, tmp_path):
    key = "da2-" + "b" * 26
    home = _Reply(text='<script src="/_next/static/chunks/app-1.js"></script>')
    bundle = _Reply(text=f'"graphql-prod-9999.edge.aws.worldathletics.org" "{key}"')
    cache = tmp_path / "wa_graphql.json"

    monkeypatch.setattr(scraper.requests, "get",
                        lambda url, **kwargs: home if url == scraper.WA_HOME else bundle)
    monkeypatch.setattr(scraper, "_answers", lambda url, k: True)
    monkeypatch.setattr(scraper, "_discovery_tried", [False])
    monkeypatch.setattr(scraper, "ENDPOINT_CACHE", str(cache))
    # Recorded so the real values come back after the test, since a successful
    # discovery rewrites all three module globals.
    monkeypatch.setattr(scraper, "GRAPHQL_URL", scraper.GRAPHQL_URL)
    monkeypatch.setattr(scraper, "API_KEY", scraper.API_KEY)
    monkeypatch.setattr(scraper, "HEADERS", scraper.HEADERS)

    assert scraper.discover_endpoint() is True
    assert scraper.GRAPHQL_URL == "https://graphql-prod-9999.edge.aws.worldathletics.org/graphql"
    assert scraper.HEADERS["x-api-key"] == key
    assert json.loads(cache.read_text(encoding="utf-8")) == {
        "url": "https://graphql-prod-9999.edge.aws.worldathletics.org/graphql", "key": key,
    }
