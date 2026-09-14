"""Tests for the Asian Games field (src/asian_games_scraper.py) and its call
(src/asian_games_predictions.py).

Two boundaries are worth guarding. The FIELD is the organisers' entry list, and
each entrant's season mark is World Athletics': matching one to the other wrong
would score the wrong person, and would look exactly like a right answer. The
CALL is made one way per event, by the user's 6-of-8 rule, and a model chance
must never share a list with a points ranking.
"""
import json
import zlib

import pytest
import requests

import asian_games_predictions as agp
import asian_games_scraper as ags


# ---- the portal's replies ---------------------------------------------------

def test_a_reply_is_inflated_the_way_the_portal_does_it():
    """Checked against the live API on 2026-09-14: a zlib stream written out as
    text, one character per byte, which the site's own page inflates with pako."""
    payload = {"participants": [{"Name": "TADA Shuhei", "Org": "JPN"}]}
    wire = zlib.compress(json.dumps(payload).encode("utf-8")).decode("latin-1").encode("utf-8")
    assert ags.decode(wire) == payload


def test_a_plain_json_error_reply_still_reads():
    assert ags.decode(b'{"code": 404, "message": "This service does not exist"}')["code"] == 404


@pytest.mark.parametrize("ev_key, expected", [
    ("M.100M--------------", "men_100m"),
    ("W.100MHURD----------", "women_100h"),
    ("M.110MHURD----------", "men_110h"),
    ("W.3000MST-----------", "women_3000sc"),
    ("M.PLEVAULT----------", "men_PV"),
    ("W.TRPLJUMP----------", "women_TJ"),
    ("M.HAMMER------------", None),
    ("M.10000M------------", None),
    ("W.HEPTATH-----------", None),
    ("X.4X400M------------", None),
])
def test_portal_event_codes_map_to_our_disciplines(ev_key, expected):
    assert ags.disc_key(ev_key) == expected


def test_events_we_do_not_call_say_why():
    assert ags.not_called_reason({"EvKey": "X.4X400M------------", "IsTeam": True}) == "relay"
    assert ags.not_called_reason({"EvKey": "M.MARATHON----------", "IsTeam": False}) == "noData"


# ---- the entry list ---------------------------------------------------------

def test_the_field_is_the_entry_list_and_ids_come_from_each_federation():
    listing = {
        "orgs": [{"Key": "JPN"}, {"Key": "IND"}],
        "participants": [
            {"Reg": "1", "Org": "JPN", "Type": "A", "Name": "TADA Shuhei",
             "Inscriptions": [{"EvKey": "M.100M--------------"}, {"EvKey": "M.200M--------------"}]},
            {"Reg": "2", "Org": "IND", "Type": "A", "Name": "CHOPRA Neeraj",
             "Inscriptions": [{"EvKey": "M.JAVELIN-----------"}]},
            {"Reg": "9", "Org": "JPN", "Type": "T", "Name": "Japan",
             "Inscriptions": [{"EvKey": "M.4X100M------------"}]},
        ],
    }
    jpn = {"Events": [{"Partics": [{"Reg": "1", "IFId": "1001", "GivenName": "Shuhei",
                                     "FamilyName": "TADA", "BirthDateRaw": "1996-06-24"}]}]}

    def get(route):
        if route == "ATH/entries/list":
            return listing
        if route == "ATH/entries/org/JPN":
            return jpn
        raise requests.ConnectionError(route)

    by_event, failed = ags.fetch_entries(get)
    assert failed == ["IND"]
    assert [e["waId"] for e in by_event["M.100M--------------"]] == [1001]
    assert by_event["M.200M--------------"][0]["waId"] == 1001
    # A federation page that failed costs the id, not the place in the field.
    assert by_event["M.JAVELIN-----------"][0]["entryName"] == "CHOPRA Neeraj"
    assert by_event["M.JAVELIN-----------"][0]["waId"] is None
    assert "M.4X100M------------" not in by_event


# ---- the Asian toplist ------------------------------------------------------

TH = ("<tr><th>Rank</th><th>Mark</th><th>WIND</th><th>Competitor</th><th>DOB</th><th></th>"
      "<th>Pos</th><th></th><th>Venue</th><th>Date</th><th>Results Score</th></tr>")


def page(*rows):
    body = "".join(
        f'<tr><td>{rank}</td><td>{mark}</td><td></td><td><a href="/athletes/athlete={wid}">{name}</a></td>'
        f"<td>01 JAN 2000</td><td>{nat}</td><td>1</td><td></td><td>Somewhere (JPN)</td>"
        f"<td>01 JUN 2026</td><td>{score}</td></tr>"
        for rank, mark, name, nat, wid, score in rows)
    return f"<table>{TH}{body}</table>"


def test_the_asian_list_pages_with_an_ampersand_and_stops_once_every_entrant_is_found():
    urls = []
    pages = {
        1: page((1, "10.00", "Shuhei TADA", "JPN", 1001, 1200)),
        2: page((1, "10.00", "Shuhei TADA", "JPN", 1001, 1200), (101, "10.40", "Xin LI", "CHN", 1002, 1050)),
    }

    def fetch(url):
        urls.append(url)
        return pages.get(int(url.rsplit("page=", 1)[1]) if "page=" in url else 1, "<html></html>")

    entrants = [{"waId": 1001, "entryName": "TADA Shuhei", "nat": "JPN"},
                {"waId": 1002, "entryName": "LI Xin", "nat": "CHN"}]
    header, rows = ags.scrape_asian_toplist("men_100m", entrants, ags.ASIAN_NATIONS, fetch=fetch)
    assert urls[1].endswith("region=asia&page=2")
    assert len(urls) == 2
    # Once each, though page 2 repeats the tie on its boundary.
    assert [r[3] for r in rows] == ["Shuhei TADA", "Xin LI"]
    assert header[-3:] == ["discipline", "year", "ProfileURL"]


def test_an_athlete_from_outside_asia_means_the_area_filter_did_not_apply():
    def fetch(url):
        return page((1, "9.79", "Noah LYLES", "USA", 1, 1280))

    with pytest.raises(ValueError, match="USA"):
        ags.scrape_asian_toplist("men_100m", [], ags.ASIAN_NATIONS, fetch=fetch)


HEADER = ["Rank", "Mark", "WIND", "Competitor", "DOB", "", "Pos", "", "Venue", "Date",
          "Results Score", "discipline", "year", "ProfileURL"]


def row(rank, mark, name, nat, wid, score):
    return [str(rank), mark, "", name, "01 JAN 2000", nat, "1", "", "Somewhere", "01 JUN 2026",
            str(score), "men_100m", "2026",
            f"https://worldathletics.org/athletes/athlete={wid}" if wid else ""]


def test_an_entrant_is_matched_by_world_athletics_id_whatever_the_name_order():
    rows = [row(1, "10.00", "Shuhei TADA", "JPN", 1001, 1200)]
    [(athlete, matched)] = ags.match_entrants(
        [{"entryName": "TADA Shuhei", "nat": "JPN", "waId": 1001}], HEADER, rows)
    assert matched is rows[0]
    assert (athlete["name"], athlete["score"], athlete["matchedBy"]) == ("Shuhei TADA", 1200, "waId")


def test_without_an_id_name_and_nation_must_both_agree():
    rows = [row(1, "10.00", "Shuhei TADA", "JPN", 1001, 1200)]
    same = {"entryName": "TADA Shuhei", "nat": "JPN", "waId": None}
    namesake = {"entryName": "TADA Shuhei", "nat": "KOR", "waId": None}
    (a1, r1), (a2, r2) = ags.match_entrants([same, namesake], HEADER, rows)
    assert r1 is rows[0] and a1["matchedBy"] == "name"
    assert r2 is None and a2["score"] is None


def test_an_entrant_with_an_id_is_never_matched_to_a_namesake():
    rows = [row(1, "10.00", "Shuhei TADA", "JPN", 1001, 1200)]
    [(athlete, matched)] = ags.match_entrants(
        [{"entryName": "TADA Shuhei", "nat": "JPN", "waId": 2002, "givenName": "Shuhei", "familyName": "TADA"}],
        HEADER, rows)
    assert matched is None and athlete["score"] is None
    # Written the site's way even when unmatched, and still linked to WA.
    assert athlete["name"] == "Shuhei TADA"
    assert athlete["profileUrl"].endswith("athlete=2002")


def test_ids_are_read_from_both_kinds_of_profile_link():
    """The world toplists link /athletes/athlete=ID and the area toplists
    /athletes/<country>/<name>-ID. Reading only the first form left entrants
    with a good id unmatched, found on the first real run (2026-09-14)."""
    assert ags.wa_id("https://worldathletics.org/athletes/athlete=14536762") == 14536762
    assert ags.wa_id("https://worldathletics.org/athletes/pr-of-china/wenjie-wang-15074930") == 15074930
    assert ags.wa_id("") is None


def test_an_id_matches_an_area_list_row_linked_by_name():
    rows = [row(1, "13:40.00", "Wenjie WANG", "CHN", None, 1100)]
    rows[0][-1] = "https://worldathletics.org/athletes/pr-of-china/wenjie-wang-15074930"
    [(athlete, matched)] = ags.match_entrants(
        [{"entryName": "WANG Wenjie", "nat": "CHN", "waId": 15074930}], HEADER, rows)
    assert matched is rows[0] and athlete["matchedBy"] == "waId"


def test_a_middle_name_on_the_toplist_does_not_stop_a_name_match():
    """Real 2026 shape: the entry list has "ALZOFAIRI Ebrahim", the toplist
    "Ebrahim Remaid ALZOFAIRI", and the entry carries no id."""
    rows = [row(4, "1:47.10", "Ebrahim Remaid ALZOFAIRI", "KUW", None, 1080)]
    entrant = {"entryName": "ALZOFAIRI Ebrahim", "familyName": "ALZOFAIRI", "givenName": "Ebrahim",
               "nat": "KUW", "waId": None}
    [(athlete, matched)] = ags.match_entrants([entrant], HEADER, rows)
    assert matched is rows[0] and athlete["matchedBy"] == "nameParts"


def test_a_loose_name_match_with_two_candidates_is_no_match():
    rows = [row(1, "10.30", "Ali HAJI", "BRN", None, 1100),
            row(2, "10.35", "Ali Hassan HAJI", "BRN", None, 1090)]
    entrant = {"entryName": "HAJI", "familyName": "HAJI", "givenName": "", "nat": "BRN", "waId": None}
    [(athlete, matched)] = ags.match_entrants([entrant], HEADER, rows)
    assert matched is None


def test_a_row_matched_by_id_is_not_also_given_to_an_entrant_without_one():
    rows = [row(1, "10.00", "Shuhei TADA", "JPN", 1001, 1200)]
    with_id = {"entryName": "TADA Shuhei", "nat": "JPN", "waId": 1001}
    without = {"entryName": "TADA Shuhei", "familyName": "TADA", "givenName": "Shuhei",
               "nat": "JPN", "waId": None}
    (a1, r1), (a2, r2) = ags.match_entrants([with_id, without], HEADER, rows)
    assert r1 is rows[0] and r2 is None


def test_the_snapshot_keeps_the_world_list_and_adds_only_entrants_it_lacks():
    world = [row(1, "9.79", "Noah LYLES", "USA", 1, 1280), row(2, "9.95", "Shuhei TADA", "JPN", 1001, 1230)]
    already = row(1, "9.95", "Shuhei TADA", "JPN", 1001, 1230)
    missing = row(7, "10.21", "Xin LI", "CHN", 1002, 1120)
    merged, added = ags.merge_snapshot(HEADER, world, HEADER, [already, missing])
    assert merged[:2] == world
    assert [r[3] for r in added] == ["Xin LI"]
    assert added[0][0] == "", "an Asian rank is not a world rank"


def test_toplists_with_different_columns_are_not_merged():
    with pytest.raises(ValueError):
        ags.merge_snapshot(HEADER, [], HEADER[:-1], [])


def test_a_run_that_lost_the_field_or_results_is_not_saved():
    saved = {"field": [{"discKey": "men_100m"}], "results": [{"discipline": "men_100m"}]}
    assert ags.lost_data({"field": [], "results": []}, saved) == "field, results"
    assert ags.lost_data(saved, saved) is None
    assert ags.lost_data({"field": [], "results": []}, None) is None


# ---- the call: one method per event -----------------------------------------

def athletes(*spec):
    """(name, score) pairs in Asian-list order."""
    return [{"name": n, "nat": "JPN", "score": s, "mark": "10.00", "asiaRank": i + 1}
            for i, (n, s) in enumerate(spec)]


def test_six_of_the_top_eight_with_history_is_a_model_event():
    field = athletes(*[(f"K{i}", 1300 - i) for i in range(6)], ("N1", 1200), ("N2", 1199))
    method, evidence = agp.choose_method(field, {f"K{i}" for i in range(6)})
    assert method == "model"
    assert len(evidence["withHistory"]) == 6 and len(evidence["considered"]) == 8


def test_five_of_the_top_eight_is_points_whoever_has_history_further_down():
    field = athletes(*[(f"K{i}", 1300 - i) for i in range(5)],
                     ("N1", 1200), ("N2", 1199), ("N3", 1198), ("K5", 1100), ("K6", 1099))
    method, evidence = agp.choose_method(field, {f"K{i}" for i in range(7)})
    assert method == "points"
    assert "K5" not in evidence["considered"]


def test_an_entrant_with_no_season_mark_cannot_be_in_the_top_eight():
    field = athletes(*[(f"K{i}", 1300 - i) for i in range(6)]) + [{"name": "NOMARK", "nat": "JPN", "score": None}]
    method, evidence = agp.choose_method(field, {f"K{i}" for i in range(6)})
    assert "NOMARK" not in evidence["considered"]
    assert method == "model"


def test_a_points_event_states_an_order_and_no_chances():
    event = {"discKey": "men_JT", "disciplineLabel": "Men's Javelin Throw", "sex": "M",
             "athletes": athletes(("Second ONE", 1150), ("First ONE", 1240))
                         + [{"name": "No MARK", "nat": "PAK", "score": None}]}
    out = agp.points_call(event)
    assert [a["name"] for a in out["athletes"]] == ["First ONE", "Second ONE"]
    assert all(a["podiumChance"] is None for a in out["athletes"])
    assert out["unscored"] == ["No MARK"]


def test_no_event_mixes_a_model_chance_with_a_points_ranking(tmp_path):
    model_event = {"discKey": "men_100m", "disciplineLabel": "Men's 100m", "sex": "M",
                   "athletes": athletes(*[(f"K{i}", 1300 - i) for i in range(8)])}
    points_event = {"discKey": "men_JT", "disciplineLabel": "Men's Javelin Throw", "sex": "M",
                    "athletes": athletes(*[(f"N{i}", 1200 - i) for i in range(8)])}
    seen_snapshots = []

    def project(entry, model, scaler, cols, snapshot_path=None):
        seen_snapshots.append(snapshot_path)
        return {"discKey": entry["discKey"], "unscored": [],
                "athletes": [{"rank": i, "name": a["name"], "podiumChance": 50.0 - i}
                             for i, a in enumerate(entry["athletes"], 1)]}

    known = {"men_100m": {f"K{i}" for i in range(8)}, "men_JT": set()}
    out = agp.build({"field": [model_event, points_event],
                     "notCalled": [{"label": "Men's Hammer Throw", "reason": "noData"}]},
                    None, None, [], snapshot_dir=str(tmp_path), known_for=known.get, project=project,
                    world_for=lambda key: set())
    by_key = {p["discKey"]: p for p in out["projections"]}

    assert by_key["men_100m"]["method"] == "model"
    assert all(a["podiumChance"] is not None for a in by_key["men_100m"]["athletes"])
    assert by_key["men_JT"]["method"] == "points"
    assert by_key["men_JT"]["methodEvidence"]["reason"] == "history"
    assert by_key["men_100m"]["methodEvidence"]["reason"] is None
    assert all(a["podiumChance"] is None for a in by_key["men_JT"]["athletes"])
    # The model reads the merged snapshot, never the world one.
    assert seen_snapshots == [str(tmp_path / "men_100m_2026.csv")]
    assert out["notCalled"][0]["reason"] == "noData"


def test_an_athlete_with_no_page_here_links_to_world_athletics_instead():
    event = {"athletes": [
        {"name": "On LIST", "profileUrl": "https://worldathletics.org/athletes/athlete=1"},
        {"name": "Off LIST", "profileUrl": "https://worldathletics.org/athletes/japan/off-list-2"},
    ]}
    out = agp.link_athletes({"athletes": [{"name": "On LIST"}, {"name": "Off LIST"}]}, event, {"ON LIST"})
    assert [a["hasPage"] for a in out["athletes"]] == [True, False]
    assert out["athletes"][1]["profileUrl"].endswith("off-list-2")


def test_a_favourite_under_the_floor_sends_the_event_to_points(tmp_path):
    """The real 2026 case in miniature: the women's triple jump passed the
    history rule and the model's favourite read 0.4%."""
    event = {"discKey": "women_TJ", "disciplineLabel": "Women's Triple Jump", "sex": "W",
             "athletes": athletes(*[(f"K{i}", 1150 - i) for i in range(8)])}

    def project(entry, model, scaler, cols, snapshot_path=None):
        return {"discKey": entry["discKey"], "unscored": [],
                "athletes": [{"rank": i, "name": a["name"], "podiumChance": round(0.4 - i / 100, 2)}
                             for i, a in enumerate(entry["athletes"], 1)]}

    out = agp.build({"field": [event]}, None, None, [], snapshot_dir=str(tmp_path),
                    known_for=lambda key: {f"K{i}" for i in range(8)}, project=project,
                    world_for=lambda key: set())
    [call] = out["projections"]
    assert call["method"] == "points"
    assert (call["methodEvidence"]["reason"], call["methodEvidence"]["topChance"]) == ("floor", 0.39)
    assert all(a["podiumChance"] is None for a in call["athletes"])
    assert out["rule"]["floor"] == agp.MODEL_FLOOR


def test_a_model_event_the_model_cannot_score_stops_the_build():
    event = {"discKey": "men_100m", "athletes": athletes(("K0", 1300))}
    with pytest.raises(RuntimeError):
        agp.model_call(event, None, None, [], "missing.csv", project=lambda *a, **k: None)


# ---- grading ----------------------------------------------------------------

def test_a_points_call_is_graded_without_inventing_a_chance(tmp_path, monkeypatch):
    import api

    frozen = tmp_path / "predictions_prefinal.json"
    frozen.write_text(json.dumps({"frozenAt": "2026-09-22T00:00:00Z", "projections": [
        {"discKey": "men_JT", "method": "points",
         "athletes": [{"rank": 1, "name": "First ONE", "podiumChance": None}]}]}), encoding="utf-8")
    monkeypatch.setattr(api, "championship_file", lambda champ_id, filename: str(frozen))

    idx, frozen_at = api.load_event_prefinal("asian-games-2026")
    assert idx["men_JT"]["First One"]["prob"] is None
    assert idx["men_JT"]["First One"]["rank"] == 1
    assert frozen_at == "2026-09-22T00:00:00Z"


def test_the_summary_carries_what_the_nav_needs_and_not_the_field(monkeypatch):
    """The nav tab, landing badge, dashboard band and schedule read this on
    every page. The whole payload runs to hundreds of kilobytes."""
    import api

    event = {"name": "20th Asian Games", "shortName": "Aichi-Nagoya 2026", "venue": "Paloma Mizuho Stadium",
             "city": "Nagoya", "country": "JPN", "eventCount": 50,
             "field": [{"discKey": "men_100m"}], "results": []}
    monkeypatch.setattr(api, "load_event", lambda champ_id: event)
    out = api.championship_summary("asian-games-2026")
    assert (out["theme"], out["navKey"]) == ("asianGames", "nav.asianGames")
    assert (out["city"], out["venue"], out["eventCount"]) == ("Nagoya", "Paloma Mizuho Stadium", 50)
    assert "field" not in out and "results" not in out
