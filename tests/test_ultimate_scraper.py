"""Unit tests for src/ultimate_scraper.py.

The network calls aren't tested -- they need World Athletics' live API -- but
the two things that silently produced WRONG data are, because both failed by
looking exactly like "nothing published yet" rather than like an error.
"""
import dl_final_results_scraper as dlr
import ultimate_scraper as us


def test_fetch_timetable_queries_the_waw_event_id_not_the_competition_id(monkeypatch):
    """getEventTimetable keys on the WAW event id, NOT the competition id that
    getCalendarCompetitionResults uses. They are different numbers for the same
    meeting, and WA's API accepts either without complaint -- the wrong one just
    returns an empty timetable, which is indistinguishable from "WA hasn't
    published the start lists yet".

    That is why this is pinned: with the competition id, fieldPublished could
    never have become True once WA published, and the Ultimate page would have
    sat on "start lists aren't up" through the entire championship without ever
    raising an error.
    """
    seen = {}

    def fake_graphql(operation, variables, query):
        seen.update(variables)
        return {"getEventTimetable": []}

    monkeypatch.setattr(dlr, "graphql", fake_graphql)
    us.fetch_timetable()
    assert seen["e"] == us.WAW_EVENT_ID
    assert seen["e"] != us.COMPETITION_ID


def test_fetch_timetable_survives_a_failing_api(monkeypatch):
    """A fetch that raises must read as "not published", never crash the build:
    the whole scraper is designed to run repeatedly before the data exists."""

    def boom(*a, **k):
        raise RuntimeError("GraphQL error")

    monkeypatch.setattr(dlr, "graphql", boom)
    assert us.fetch_timetable() == []


def test_resolve_disc_keys_refuses_to_guess(monkeypatch):
    """An athlete we rank in several events is resolved by the event World
    Athletics says they won; anything ambiguous is left unlinked rather than
    pointed at the wrong page."""
    monkeypatch.setattr(
        us,
        "_athlete_discipline_index",
        lambda: {1: {"men_100m": "Noah LYLES", "men_200m": "Noah LYLES"}, 2: {"men_HT": "A N Other"}},
    )
    out = us.resolve_disc_keys([
        # ranked in two events, WA names one of them -> resolved
        {"waId": 1, "disciplineLabel": "200M", "name": "NOAH LYLES"},
        # a sprint double names two events at once -> not guessable
        {"waId": 1, "disciplineLabel": "100M & 200M", "name": "NOAH LYLES"},
        # an event we don't rank them in -> left alone
        {"waId": 2, "disciplineLabel": "SHOT PUT", "name": "A N OTHER"},
        # unknown athlete entirely
        {"waId": 999, "disciplineLabel": "100M", "name": "NOBODY"},
    ])
    assert out[0]["discKey"] == "men_200m"
    assert out[0]["linkName"] == "Noah LYLES"
    assert out[1]["discKey"] is None
    assert out[2]["discKey"] is None
    assert out[3]["discKey"] is None


def test_resolve_disc_keys_uses_our_name_not_world_athletics_wording():
    """WA's card shows the popular name; our athlete route is keyed on the
    registered one. Linking on their string 404s -- verified on Duplantis
    ("MONDO" vs "Armand"), Kebinatshipi and Perez Hernandez."""
    idx = {7: {"men_PV": "Armand DUPLANTIS"}}
    original = us._athlete_discipline_index
    us._athlete_discipline_index = lambda: idx
    try:
        out = us.resolve_disc_keys([{"waId": 7, "disciplineLabel": "POLE VAULT", "name": "MONDO DUPLANTIS"}])
    finally:
        us._athlete_discipline_index = original
    assert out[0]["linkName"] == "Armand DUPLANTIS"
    assert out[0]["name"] == "MONDO DUPLANTIS"


def test_relay_key_matches_the_two_names_wa_gives_one_event():
    """The results feed says "Mixed 4x100 Metres Relay"; the timetable says
    "4x100 Metres Relay". Squads are keyed off one and teams off the other, so
    they only join if both reduce to the same thing."""
    assert us._relay_key("Mixed 4x100 Metres Relay") == us._relay_key("4x100 Metres Relay")
    assert us._relay_key("Mixed 4x400 Metres Relay") == "4x400"
    assert us._relay_key("4 x 100 Metres Relay") == "4x100"


def test_relay_squads_count_appearances_across_rounds_and_meets(monkeypatch):
    """Appearances are what separate a fixture of the squad from a one-off.

    The mock answers per (event, discipline) the way WA does, because that is
    the behaviour under test: one athlete runs at BOTH meets, another only at
    the older one. Most meets wins, then most rounds -- which is how Bryce
    Deadmon (World Relays and the 2025 Worlds) sorts above a single-meet pick.
    """
    def phase(name, members):
        return {"sexCode": "X", "phaseName": name,
                "discipline": {"name": "4x400 Metres Relay", "isRelay": True},
                "units": [{"results": [{"resultCountryCode": "USA", "teamMembers": members}]}]}

    def member(name, wa, leg):
        return {"competitorName": name, "competitorId_WA": wa, "competitorOrder": leg}

    def fake_graphql(operation, variables, query):
        # The 4x100 is not run at a World Championships, so it answers empty --
        # exactly what WA does, and it must not invent a squad.
        if variables["d"] != "4X4":
            return {"getEventTimetableWithContent": []}
        if variables["e"] == us.WORLD_RELAYS_EVENT_ID:      # 2026
            return {"getEventTimetableWithContent": [
                phase("Round 1", [member("Both Meets", 1, 1), member("New Face", 2, 3)]),
                phase("Final", [member("Both Meets", 1, 1), member("New Face", 2, 3)]),
            ]}
        return {"getEventTimetableWithContent": [                # 2025
            phase("Final", [member("Both Meets", 1, 1), member("Older Pick", 3, 3)]),
        ]}

    monkeypatch.setattr(dlr, "graphql", fake_graphql)
    squads = us.relay_squads()
    usa = squads[("USA", "4x400")]
    # Two meets beats more rounds at one meet.
    assert [a["name"] for a in usa] == ["Both Meets", "New Face", "Older Pick"]
    assert usa[0]["meets"] == [2026, 2025]
    assert usa[0]["rounds"] == 3
    assert usa[1]["meets"] == [2026]
    assert usa[2]["meets"] == [2025]
    # An event a meet does not contest must not appear at all.
    assert ("USA", "4x100") not in squads

def test_relay_squads_survive_a_failing_api(monkeypatch):
    """Squads are an enrichment: losing them must cost the names, not the
    relay teams themselves."""
    def boom(*a, **k):
        raise RuntimeError("GraphQL error")
    monkeypatch.setattr(dlr, "graphql", boom)
    assert us.relay_squads() == {}


def test_startlists_stay_empty_until_wa_populates_the_api(monkeypatch):
    """World Athletics announced the Ultimate's final entry lists (381 athletes,
    65 federations) on 2026-09-07 while their competition API still returned
    nothing for them: all 40 phases flagged isStartlistPublished false, and
    asking for a start list directly gave a row of nulls.

    The scraper must report that honestly rather than reach for the press
    release and turn prose into a field. An empty list here is the correct
    answer to "who is entered" when nothing machine-readable says."""
    import ultimate_scraper as us

    def fake_graphql(op, variables, query):
        if op == "UcPhases":
            return {"getEventPhases": [
                {"phaseCode": "f", "phaseName": "Final", "sexCode": "M",
                 "isStartlistPublished": False,
                 "discipline": {"name": "100 Metres", "nameUrlSlug": "100-metres"}},
            ]}
        if op == "UcStartlist":
            return {"getEventPhaseByDiscipline": {
                "phaseName": None, "isStartlistPublished": None, "units": None}}
        raise AssertionError(op)

    monkeypatch.setattr(us.dlr, "graphql", fake_graphql)
    assert us.fetch_startlists() == []


def test_startlists_are_read_once_wa_publishes_them(monkeypatch):
    """The other half: when the API does carry entries, they come through with
    the fields the site needs. Verified against the live schema -- the start
    list rows really are named competitorName / competitorCountryCode /
    seasonBestMark / worldRanking / bib."""
    import ultimate_scraper as us

    def fake_graphql(op, variables, query):
        if op == "UcPhases":
            return {"getEventPhases": [
                {"phaseCode": "f", "phaseName": "Final", "sexCode": "M",
                 "isStartlistPublished": True,
                 "discipline": {"name": "800 Metres", "nameUrlSlug": "800-metres"}},
            ]}
        assert variables["d"] == "800-metres", "disciplineCode must be the URL slug"
        assert variables["p"] == "f", "phaseCode is lower case"
        return {"getEventPhaseByDiscipline": {
            "phaseName": "Final", "isStartlistPublished": True,
            "units": [{"unitId": 1, "unitTypeName": "Final", "startlist": [
                {"competitorName": "Emmanuel WANYONYI", "competitorCountryCode": "KEN",
                 "seasonBestMark": "1:41.76", "worldRanking": 1, "bib": "101"},
            ]}]}}

    monkeypatch.setattr(us.dlr, "graphql", fake_graphql)
    lists = us.fetch_startlists()
    assert len(lists) == 1
    assert lists[0]["discipline"] == "800 Metres"
    assert lists[0]["entries"][0]["competitorName"] == "Emmanuel WANYONYI"


def test_the_entry_check_writes_nothing_while_they_are_unpublished(monkeypatch, capsys):
    """A scheduled check that quietly rewrote event.json on every run, or that
    reported a network failure as "not published", would be worse than no check
    at all. Both cases have to be distinguishable in the first line."""
    import check_ultimate_entries as chk

    monkeypatch.setattr(chk.us, "fetch_startlists", lambda: [])
    monkeypatch.setattr(chk.us, "fetch_timetable", lambda: [{}] * 40)
    written = []
    monkeypatch.setattr(chk.us, "build_event", lambda: written.append(1) or {})
    assert chk.main() == 0
    assert capsys.readouterr().out.startswith("NOT YET:")
    assert not written, "must not rebuild event.json when there is nothing to add"


def test_a_failed_check_is_not_reported_as_not_published(monkeypatch, capsys):
    import check_ultimate_entries as chk

    def boom():
        raise RuntimeError("GraphQL 502")

    monkeypatch.setattr(chk.us, "fetch_startlists", boom)
    assert chk.main() == 0
    assert capsys.readouterr().out.startswith("CHECK FAILED:")


def test_published_entries_rebuild_the_event_file(monkeypatch, tmp_path, capsys):
    import check_ultimate_entries as chk

    monkeypatch.setattr(chk.us, "fetch_startlists", lambda: [
        {"discipline": "800 Metres", "sex": "M", "phase": "Final",
         "entries": [{"competitorName": "A"}, {"competitorName": "B"}]},
    ])
    monkeypatch.setattr(chk.us, "build_event", lambda: {"ok": True})
    monkeypatch.setattr(chk.us, "OUT_PATH", str(tmp_path / "event.json"))
    assert chk.main() == 0
    out = capsys.readouterr().out
    assert out.startswith("ENTRIES PUBLISHED: 2 athletes across 1 events")
    assert (tmp_path / "event.json").exists()
