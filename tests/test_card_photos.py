"""Unit tests for the card photo cache (src/warm_card_photos.py and the
attach helper in api.py).

The cache exists so a card view does not resolve photos live. What has to hold
is that it caches an absence as an absence, never caches a FAILURE as one, and
that the API attaches from it without reaching the network at all -- a cache
that quietly falls back to a live lookup would reintroduce exactly the
hundreds-of-round-trips problem it was built to remove.
"""
import json

import api
import warm_card_photos as w


def test_an_athlete_with_no_photo_is_cached_as_such(tmp_path, monkeypatch):
    """World Athletics has no headshot for roughly a third of athletes and
    Commons covers about half of those. A null is a fact about the athlete, and
    caching it is what stops every later run re-asking two external APIs for a
    picture that does not exist."""
    monkeypatch.setattr(w, "CACHE_PATH", str(tmp_path / "cache.json"))
    monkeypatch.setattr(w, "card_athletes", lambda: [("https://wa/athlete=1", "NO PHOTO")])
    monkeypatch.setattr(w.api, "load_athlete_photo", lambda url: None)
    monkeypatch.setattr(w.api, "load_wikimedia_photo", lambda url: None)
    w.main()
    cache = json.loads((tmp_path / "cache.json").read_text(encoding="utf-8"))
    assert cache["https://wa/athlete=1"]["url"] is None
    assert cache["https://wa/athlete=1"]["credit"] is None


def test_a_failed_lookup_is_not_cached_as_no_photo(tmp_path, monkeypatch):
    """The dangerous case. If a network error were stored as {"url": None} the
    next run would skip that athlete forever and their card would be blank for
    good, with nothing anywhere saying why."""
    monkeypatch.setattr(w, "CACHE_PATH", str(tmp_path / "cache.json"))
    monkeypatch.setattr(w, "card_athletes", lambda: [("https://wa/athlete=2", "FLAKY")])

    def boom(url):
        raise RuntimeError("connection reset")

    monkeypatch.setattr(w.api, "load_athlete_photo", boom)
    w.main()
    cache = json.loads((tmp_path / "cache.json").read_text(encoding="utf-8"))
    assert "https://wa/athlete=2" not in cache


def test_already_cached_athletes_are_not_re_resolved(tmp_path, monkeypatch):
    monkeypatch.setattr(w, "CACHE_PATH", str(tmp_path / "cache.json"))
    (tmp_path / "cache.json").write_text(
        json.dumps({"https://wa/athlete=3": {"url": "https://img/x.jpg", "credit": None}}),
        encoding="utf-8")
    monkeypatch.setattr(w, "card_athletes", lambda: [("https://wa/athlete=3", "CACHED")])
    calls = []
    monkeypatch.setattr(w.api, "load_athlete_photo",
                        lambda url: calls.append(url) or None)
    monkeypatch.setattr(w.api, "load_wikimedia_photo", lambda url: None)
    w.main()
    assert calls == [], "a cached athlete was resolved again"


def test_attach_adds_nothing_when_the_athlete_is_not_cached(monkeypatch):
    """A card with no photo must render a monogram, not reach for the network.
    Absence has to stay absence all the way to the payload."""
    monkeypatch.setattr(api, "_CARD_PHOTOS", {})
    row = {"name": "A", "profileUrl": "https://wa/athlete=9"}
    assert api.attach_card_photo(row) == {"name": "A", "profileUrl": "https://wa/athlete=9"}


def test_attach_uses_the_cache_and_carries_the_credit(monkeypatch):
    monkeypatch.setattr(api, "_CARD_PHOTOS", {
        "https://wa/athlete=9": {"url": "https://img/a.jpg",
                                 "credit": {"author": "Someone", "license": "CC BY-SA 4.0"}},
    })
    row = api.attach_card_photo({"name": "A", "profileUrl": "https://wa/athlete=9"})
    assert row["photoUrl"] == "https://img/a.jpg"
    assert row["photoCredit"]["license"] == "CC BY-SA 4.0"


def test_world_athletics_wins_whenever_they_have_a_photo(monkeypatch):
    """The user's rule, stated explicitly: if World Athletics has a photo, use
    it; Wikipedia is only for athletes they have nothing for.

    An earlier version preferred a Commons portrait whenever the WA shot had no
    detectable face in it. That was overruled, and this pins the decision so it
    does not drift back on the next person who notices an action photo with the
    athlete's head turned away."""
    monkeypatch.setattr(w.api, "load_athlete_photo", lambda url: "https://wa/photo.jpg")
    monkeypatch.setattr(w.api, "load_wikimedia_photo",
                        lambda url: {"url": "https://commons/portrait.jpg",
                                     "credit": {"author": "X", "license": "CC BY"}})
    # Even with a Commons portrait sitting right there, and regardless of what
    # face detection would say about the WA one.
    url, credit, how = w.best_photo("https://wa/athlete=1")
    assert url == "https://wa/photo.jpg"
    assert credit is None
    assert how == "wa"


def test_wikimedia_is_used_only_when_world_athletics_has_nothing(monkeypatch):
    monkeypatch.setattr(w.api, "load_athlete_photo", lambda url: None)
    monkeypatch.setattr(w.api, "load_wikimedia_photo",
                        lambda url: {"url": "https://commons/portrait.jpg",
                                     "credit": {"author": "X", "license": "CC BY"}})
    url, credit, how = w.best_photo("https://wa/athlete=2")
    assert url == "https://commons/portrait.jpg"
    assert credit["license"] == "CC BY", "the licence credit has to travel with it"
    assert how == "wikimedia"


def test_no_photo_anywhere_is_not_an_error(monkeypatch):
    monkeypatch.setattr(w.api, "load_athlete_photo", lambda url: None)
    monkeypatch.setattr(w.api, "load_wikimedia_photo", lambda url: None)
    assert w.best_photo("https://wa/athlete=3") == (None, None, "none")
