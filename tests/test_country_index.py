"""Unit tests for the country view: src/country_index.py and api.py's
country search / endpoints.

The country page is the one view that puts athletes from DIFFERENT events in a
single ordered list, which makes it the one place the project's recurring
cross-discipline bug can reappear -- so the ordering rule is pinned here.
"""
import api
import country_index as ci


def _country(code, name, score, athletes=1):
    return {
        "code": code,
        "name": name,
        "topScore": score,
        "athleteCount": athletes,
        "disciplineCount": 1,
        "area": "Europe",
    }


def test_search_matches_a_country_name_not_just_athletes(monkeypatch):
    """The whole point of the feature: before it, typing "Jamaica" matched no
    athlete name and so returned nothing at all."""
    monkeypatch.setattr(api, "load_countries", lambda: {"JAM": _country("JAM", "Jamaica", 1276)})
    hits = api.search_countries("jamaica")
    assert [h["code"] for h in hits] == ["JAM"]


def test_search_ranks_an_exact_code_above_a_matching_name(monkeypatch):
    """A real collision from our own data: "ISL" is Iceland's code, while
    Iran is listed as "Islamic Republic Of Iran" -- whose name also starts
    "Isl". Someone typing three letters means the code, so Iceland wins."""
    monkeypatch.setattr(api, "load_countries", lambda: {
        "ISL": _country("ISL", "Iceland", 1100),
        "IRI": _country("IRI", "Islamic Republic Of Iran", 1200),
    })
    hits = api.search_countries("isl")
    assert [h["code"] for h in hits] == ["ISL", "IRI"]

def test_search_prefers_a_name_that_starts_with_the_query(monkeypatch):
    monkeypatch.setattr(api, "load_countries", lambda: {
        "USA": _country("USA", "United States", 1313),
        "UAE": _country("UAE", "United Arab Emirates", 900),
        "GBR": _country("GBR", "Great Britain & NI", 1250),
    })
    hits = api.search_countries("united")
    assert [h["code"] for h in hits] == ["USA", "UAE"]


def test_search_ignores_a_query_that_is_too_short(monkeypatch):
    monkeypatch.setattr(api, "load_countries", lambda: {"JAM": _country("JAM", "Jamaica", 1)})
    assert api.search_countries("j") == []
    assert api.search_countries("") == []


def test_search_survives_a_missing_country_index(monkeypatch):
    """The index is a built file; before it exists the rest of search must
    still work rather than raising."""
    monkeypatch.setattr(api, "load_countries", lambda: None)
    assert api.search_countries("jamaica") == []


def test_country_athletes_are_ordered_by_results_score(monkeypatch):
    """Ordered by World Athletics Results Score, NEVER by podium probability.

    Podium probabilities are per-discipline -- across the 32 events they sum to
    anywhere between 31 and 320 -- so they rank athletes within one event and
    are meaningless between events. A country page inevitably puts a shot putter
    next to a 400m runner, which is exactly the comparison that has produced a
    real bug in this project before.
    """
    rows = {
        "JAM": [
            {"name": "Low", "score": 1100, "disc": "Men's 100m", "discKey": "men_100m"},
            {"name": "High", "score": 1276, "disc": "Men's Shot Put", "discKey": "men_SP"},
            {"name": "None", "score": None, "disc": "Men's 200m", "discKey": "men_200m"},
        ]
    }
    monkeypatch.setattr(ci, "athletes_by_country", lambda: {
        k: sorted(v, key=lambda a: (a["score"] is None, -(a["score"] or 0), a["disc"]))
        for k, v in rows.items()
    })
    monkeypatch.setattr(ci, "country_names", lambda: {"JAM": {"name": "Jamaica", "area": "Americas"}})
    monkeypatch.setattr(ci, "ultimate_by_country", lambda: {})
    out = ci.build()
    assert [a["name"] for a in out["JAM"]["athletes"]] == ["High", "Low", "None"]
    assert out["JAM"]["topScore"] == 1276


def test_a_country_lists_everyone_with_a_page_not_only_its_world_top_100(monkeypatch, tmp_path):
    """Until 2026-09-15 a nation's page read the world toplist alone, so 362 of
    the 518 Asian Games entrants with a page were on no country page and 20 of
    their nations had no page. The championship snapshot adds the rest, a world
    row wins, and an entrant whose page is built from the call comes last with
    no mark."""
    header = "Rank,Mark,WIND,Competitor,DOB,,Pos,,Venue,Date,Results Score,discipline,year,ProfileURL\n"
    lyles = "1,9.79,+0.8,Noah LYLES,18 JUL 1997,USA,1,,New York,24 JUL 2026,1280,men_100m,2026,u-lyles\n"
    adam = ",11.02,+1.1,Ibadulla ADAM,14 JAN 2002,MDV,1,,Gaborone,27 JUN 2026,880,men_100m,2026,u-adam\n"
    world = tmp_path / "world.csv"
    world.write_text(header + lyles, encoding="utf-8")
    snapshot = tmp_path / "snapshot.csv"
    snapshot.write_text(header + lyles + adam, encoding="utf-8")
    monkeypatch.setattr(api, "DISC_LABELS", {"men_100m": "Men's 100m"})
    monkeypatch.setattr(api, "season_snapshot_paths", lambda disc_key: [str(world), str(snapshot)])
    monkeypatch.setattr(api, "championship_page_entrants", lambda: [
        {"name": "Hassan SAEED", "discKey": "men_100m", "nat": "MDV", "profileUrl": "u-saeed"},
        {"name": "Ibadulla ADAM", "discKey": "men_100m", "nat": "MDV", "profileUrl": "u-adam"}])

    out = ci.athletes_by_country()
    assert [(a["name"], a["worldRank"]) for a in out["USA"]] == [("Noah LYLES", 1)]
    assert [(a["name"], a["mark"], a["score"], a["worldRank"], a["profileUrl"]) for a in out["MDV"]] == [
        ("Ibadulla ADAM", "11.02", 880, None, "u-adam"), ("Hassan SAEED", None, None, None, "u-saeed")]


def test_the_championship_entrants_with_a_page_carry_their_nation_until_it_ends(monkeypatch):
    from datetime import date

    monkeypatch.setattr(api.championships, "current", lambda: {"id": "asian-games-2026", "endDate": "2026-09-29"})
    monkeypatch.setattr(api, "load_event_predictions", lambda champ_id: {"projections": [{
        "discKey": "men_100m",
        "athletes": [{"name": "Ranked ONE", "nat": "PAK", "profileUrl": "u-1", "hasPage": True}],
        "unranked": [{"name": "No PAGE", "nat": "AFG", "profileUrl": None, "hasPage": False}]}]})
    assert api.championship_page_entrants(today=date(2026, 9, 20)) == [
        {"name": "Ranked ONE", "discKey": "men_100m", "nat": "PAK", "profileUrl": "u-1"}]
    assert api.championship_page_entrants(today=date(2026, 9, 30)) == []


def test_country_falls_back_to_its_code_when_the_name_lookup_fails(monkeypatch):
    """WA's country list is a live call. If it fails the page must still work,
    showing the 3-letter code rather than failing the whole build."""
    monkeypatch.setattr(ci, "athletes_by_country", lambda: {"XXX": [{"name": "A", "score": 1, "disc": "d", "discKey": "k"}]})
    monkeypatch.setattr(ci, "country_names", lambda: {})
    monkeypatch.setattr(ci, "ultimate_by_country", lambda: {})
    out = ci.build()
    assert out["XXX"]["name"] == "XXX"
