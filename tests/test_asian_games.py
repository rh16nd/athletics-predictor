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
    # Called on points since 2026-09-14; before that, "noData".
    ("M.HAMMER------------", "men_HT"),
    ("W.10000M------------", "women_10000m"),
    ("M.MARATHON----------", None),
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


# ---- entrants the Asian list does not place ----------------------------------

def test_world_athletics_birth_dates_read_the_way_the_entry_list_writes_them():
    assert ags.birth_date_iso("07 DEC 2001") == "2001-12-07"
    assert ags.birth_date_iso("2004") == "2004"
    assert ags.birth_date_iso("") is None


def wa_candidate(wid, given, family, born, nat="LAO", gender="Men"):
    return {"aaAthleteId": str(wid), "givenName": given, "familyName": family,
            "birthDate": born, "gender": gender, "country": nat}


def test_an_entry_without_an_id_is_found_by_its_date_of_birth_whatever_the_spelling():
    """The real 2026 case: the entry list writes PHOMPHAKDI, World Athletics
    PHOMPAKDI, and both say 7 December 2001."""
    entrant = {"entryName": "PHOMPHAKDI Sorsy", "familyName": "PHOMPHAKDI", "givenName": "Sorsy",
               "nat": "LAO", "birthDate": "2001-12-07", "waId": None}
    queries = []

    def search(query, nat):
        queries.append((query, nat))
        return [wa_candidate(14972201, "Sorsy", "PHOMPAKDI", "07 DEC 2001"),
                wa_candidate(14714329, "Sengpheth", "PHOMPHADY", "30 MAY 1996")]

    assert ags.find_wa_id(entrant, "M", search) == 14972201
    assert queries[0] == ("PHOMPHAKDI", "LAO")


def test_a_namesake_born_another_day_or_of_the_other_sex_is_not_the_entrant():
    entrant = {"entryName": "SALEM Saeed", "familyName": "SALEM", "givenName": "Saeed",
               "nat": "QAT", "birthDate": "2000-01-01", "waId": None}

    def search(query, nat):
        return [wa_candidate(1, "Saeed", "SALEM", "02 JAN 2000", nat="QAT"),
                wa_candidate(2, "Saeed", "SALEM", "01 JAN 2000", nat="QAT", gender="Women")]

    assert ags.find_wa_id(entrant, "M", search) is None


def test_a_birth_year_alone_needs_both_names_and_a_shared_particle_is_not_a_name():
    entrant = {"entryName": "AL GHALBAN Ahmed", "familyName": "AL GHALBAN", "givenName": "Ahmed",
               "nat": "PLE", "birthDate": "2004-03-02", "waId": None}
    assert ags.find_wa_id(entrant, "M", lambda q, nat: [wa_candidate(7, "Ahmed", "AL GHALBAN", "2004", nat="PLE")]) == 7
    assert ags.find_wa_id(entrant, "M", lambda q, nat: [wa_candidate(8, "Ahmed", "AL HASSAN", "2004", nat="PLE")]) is None


def test_the_two_lists_may_swap_which_name_is_the_family_name():
    """Real 2026 shape: the entry list writes "DAVAANYAM Myagmarsuren" of
    Mongolia, World Athletics has given name Dawaanyam and family name
    MYAGMARSUREN, both born on 6 April 1999. (The id here is made up.)"""
    entrant = {"entryName": "DAVAANYAM Myagmarsuren", "familyName": "DAVAANYAM", "givenName": "Myagmarsuren",
               "nat": "MGL", "birthDate": "1999-04-06", "waId": None}

    def search(query, nat):
        return [wa_candidate(900001, "Dawaanyam", "MYAGMARSUREN", "06 APR 1999", nat="MGL")]

    assert ags.find_wa_id(entrant, "M", search) == 900001


def test_two_athletes_who_both_fit_is_no_match():
    entrant = {"entryName": "ALI Abdul", "familyName": "ALI", "givenName": "Abdul",
               "nat": "UAE", "birthDate": "1999-05-05", "waId": None}

    def search(query, nat):
        return [wa_candidate(1, "Abdul", "ALI", "05 MAY 1999", nat="UAE"),
                wa_candidate(2, "Abdul Quddus", "ALI", "05 MAY 1999", nat="UAE")]

    assert ags.find_wa_id(entrant, "M", search) is None


def wa_profile(*groups, given="Ibadulla", family="ADAM", born="14 JAN 2002"):
    return {"basicData": {"birthDate": born, "givenName": given, "familyName": family},
            "resultsByYear": {"resultsByEvent": [
                {"discipline": d, "indoor": indoor, "results": results} for d, indoor, results in groups]}}


def wa_result(mark, score, date="27 JUN 2026", legal=True):
    return {"date": date, "venue": "National Stadium, Gaborone (BOT)", "place": "1.", "mark": mark,
            "wind": "+1.1", "resultScore": score, "notLegal": not legal}


def test_a_season_best_is_the_best_legal_outdoor_score_this_season_in_that_event():
    """Real shape: Ibadulla ADAM of the Maldives, 11.02 (880) in Gaborone and
    below the top 300 in Asia."""
    profile = wa_profile(
        ("100 Metres", None, [wa_result("11.31", 797, "08 APR 2026"), wa_result("11.02", 880),
                              wa_result("10.95", 901, legal=False), wa_result("10.90", 915, "20 JUL 2025")]),
        ("100 Metres", True, [wa_result("10.80", 990)]),
        ("200 Metres", None, [wa_result("21.90", 960)]),
    )
    best = ags.season_best(profile, "men_100m")
    assert (best["mark"], best["resultScore"]) == ("11.02", 880)
    assert ags.season_best(profile, "men_400m") is None


def test_an_indoor_mark_is_left_out_though_world_athletics_never_sets_the_indoor_flag():
    """Every event group World Athletics sent on 2026-09-15 had `indoor` empty.
    An indoor result says so only with "(i)" after its venue, sometimes with a
    series label after that."""
    indoor = {**wa_result("46.10", 1180, "14 FEB 2026"), "venue": "Tianjin (CHN) (i)"}
    tour = {**wa_result("46.00", 1190, "21 FEB 2026"),
            "venue": "Gallur, Madrid (ESP) (i) - World Athletics Indoor Tour"}
    profile = wa_profile(("400 Metres", None, [indoor, tour, wa_result("46.50", 1150)]))
    assert ags.season_best(profile, "men_400m")["resultScore"] == 1150
    assert not ags.is_indoor({"venue": "Stade Louis II, Fontvieille (MON)", "competition": "Herculis (ind.)"})


def test_unplaced_entrants_are_looked_up_and_the_rest_say_why():
    rows = [row(1, "10.00", "Shuhei TADA", "JPN", 1001, 1200),
            row(250, "10.80", "Kin Wa CHAN", "MAC", 1003, 980)]
    entrants = [
        {"entryName": "TADA Shuhei", "nat": "JPN", "waId": 1001},
        # No id, and a family name spelt differently from the list's.
        {"entryName": "CHANG Kin Wa", "familyName": "CHANG", "givenName": "Kin Wa",
         "nat": "MAC", "birthDate": "2001-02-03", "waId": None},
        {"entryName": "ADAM Ibadulla", "familyName": "ADAM", "givenName": "Ibadulla",
         "nat": "MDV", "waId": 14852443},
        {"entryName": "NOOR ZAHI Sha", "familyName": "NOOR ZAHI", "givenName": "Sha",
         "nat": "AFG", "birthDate": "1991-03-21", "waId": None},
        {"entryName": "HASSAN Saaid", "familyName": "HASSAN", "givenName": "Saaid",
         "nat": "MDV", "waId": 14427032},
    ]

    def search(query, nat):
        return [wa_candidate(1003, "Kin Wa", "CHAN", "03 FEB 2001", nat="MAC")] if nat == "MAC" else []

    profiles = {14852443: wa_profile(("100 Metres", None, [wa_result("11.02", 880)])),
                14427032: wa_profile(("200 Metres", None, [wa_result("22.10", 900)]), given="Saaid", family="HASAN")}
    fetched = []

    def season(athlete_id):
        fetched.append(athlete_id)
        return profiles[athlete_id]

    out = ags.complete_matches(ags.match_entrants(entrants, HEADER, rows), entrants, HEADER, rows,
                               "men_100m", "M", search, season)
    by = {a["entryName"]: (a, r) for a, r in out}

    assert by["TADA Shuhei"][0]["unranked"] is None and by["TADA Shuhei"][1] is rows[0]
    chan, chan_row = by["CHANG Kin Wa"]
    assert chan_row is rows[1] and (chan["matchedBy"], chan["waId"], chan["score"]) == ("search", 1003, 980)
    adam, adam_row = by["ADAM Ibadulla"]
    assert (adam["matchedBy"], adam["mark"], adam["score"], adam["asiaRank"]) == ("profile", "11.02", 880, None)
    # A row the snapshot, the model and the athlete pages read like any other,
    # with no rank: this athlete is on no list.
    assert adam_row[0] == "" and adam_row[3] == "Ibadulla ADAM" and adam_row[ags.NAT] == "MDV"
    assert adam_row[HEADER.index("Results Score")] == "880" and adam_row[-1].endswith("athlete=14852443")
    assert by["NOOR ZAHI Sha"][0]["unranked"] == "notFound"
    # Unranked, and still under World Athletics' spelling of the name.
    assert (by["HASSAN Saaid"][0]["unranked"], by["HASSAN Saaid"][0]["name"]) == ("noMark", "Saaid HASAN")
    assert fetched == [14852443, 14427032]


def test_an_athlete_entered_twice_is_searched_once_and_a_failed_lookup_says_so():
    entrant = {"entryName": "CHAABAN Omar", "familyName": "CHAABAN", "givenName": "Omar",
               "nat": "PLE", "birthDate": "2006-08-18", "waId": None}
    unmatched = [({"name": "Omar CHAABAN", "entryName": "CHAABAN Omar", "waId": None, "profileUrl": None}, None)]
    queries, cache = [], {}

    def search(query, nat):
        queries.append(query)
        return []

    for key in ("men_100m", "men_200m"):
        ags.complete_matches(unmatched, [entrant], HEADER, [], key, "M", search, None, cache)
    assert queries == ["CHAABAN", "Omar CHAABAN", "Omar"]

    def down(query, nat):
        raise requests.ConnectionError("down")

    [(athlete, matched)] = ags.complete_matches(unmatched, [entrant], HEADER, [], "men_100m", "M", down, None)
    assert matched is None and athlete["unranked"] == "lookupFailed"


def test_an_athlete_takes_the_world_lists_spelling_where_their_id_is_on_it():
    """Real 2026 case: "Taepoong NAM" on the Asian list and "Tae-poong NAM" on
    the world list, one id. The snapshot keeps the world row, so under the Asian
    spelling the model could not score him in a model event."""
    world = [row(53, "80.35", "Tae-poong NAM", "KOR", 14910385, 1150)]
    asian_row = row(12, "80.35", "Taepoong NAM", "KOR", 14910385, 1150)
    matches = [({"name": "Taepoong NAM", "waId": 14910385}, asian_row),
               ({"name": "Xin LI", "waId": 1002}, None)]
    out = ags.world_spelling(matches, HEADER, world)
    assert [a["name"] for a, _ in out] == ["Tae-poong NAM", "Xin LI"]
    assert out[0][1] is asian_row


def test_a_single_named_entrant_shows_no_placeholder_dot():
    assert ags.display_name({"givenName": "Sangay", "familyName": ".", "entryName": "Sangay"}) == "Sangay"
    assert ags.display_name({"givenName": "Shuhei", "familyName": "Tada"}) == "Shuhei TADA"


# ---- the call: one method per event -----------------------------------------

def athletes(*spec):
    """(name, score) pairs in Asian-list order."""
    return [{"name": n, "nat": "JPN", "score": s, "mark": "10.00", "asiaRank": i + 1}
            for i, (n, s) in enumerate(spec)]


def unit_model():
    """A field model that reads nothing but the gap to the best mark."""
    import field_model as fm

    width = len(fm.FEATURES)
    return {"features": fm.FEATURES, "mean": [0.0] * width, "std": [1.0] * width,
            "weights": [1.0] + [0.0] * (width - 1)}


def fake_rows(scores, mark_season=2026):
    """field_model.serving_rows for the entrants named in `scores`."""
    import field_model as fm
    import pandas as pd

    def rows_for(key, entrants, season_path, cutoff, year, extra=()):
        rows = [{"athlete_name": a["name"], "sb_score": float(scores[a["name"]]),
                 "sb_date": pd.Timestamp("2026-06-01"), "sb_prior_season": int(mark_season != year),
                 "dob": pd.Timestamp("1998-01-01"), "career_best": scores[a["name"]] + 10.0,
                 "prev_season_best": scores[a["name"]] - 10.0, "mark": "70.00", "mark_season": mark_season}
                for a in entrants if a["name"] in scores]
        return pd.DataFrame(rows, columns=fm.SERVING_COLUMNS)
    return rows_for


def call_with(event, rows_for, tmp_path, report_path=None):
    return agp.build(event, unit_model(), snapshot_dir=str(tmp_path), asia_dir=str(tmp_path), cutoff="2026-09-23",
                     pages_for=lambda key: set(), last_for=lambda key: {"id": {}, "name": {}}, rows_for=rows_for,
                     report_path=report_path or str(tmp_path / "no_report.json"))


def test_every_event_with_three_marks_is_called_by_the_model_and_its_chances_add_to_300(tmp_path):
    scores = {"First ONE": 1240, "Second ONE": 1150, "Third ONE": 1100, "Fourth ONE": 1050}
    event = {"field": [{"discKey": "men_JT", "disciplineLabel": "Men's Javelin Throw", "sex": "M",
                        "athletes": athletes(*scores.items())
                        + [{"name": "No MARK", "nat": "PAK", "score": None, "unranked": "noMark"}]}],
             "notCalled": [{"label": "Men's Marathon", "reason": "noData"}]}
    out = call_with(event, fake_rows(scores), tmp_path)
    [call] = out["projections"]
    assert (call["method"], call["methodEvidence"]["reason"]) == ("model", None)
    assert [a["name"] for a in call["athletes"]] == list(scores)
    assert sum(a["podiumChance"] for a in call["athletes"]) == pytest.approx(300, abs=0.2)
    assert [(u["name"], u["reason"]) for u in call["unranked"]] == [("No MARK", "noMark")]
    assert (out["rule"]["method"], out["rule"]["cutoff"], out["rule"]["backtest"]) == ("field", "2026-09-23", None)
    assert out["notCalled"][0]["reason"] == "noData"


def test_an_entrant_read_on_last_seasons_mark_is_tagged_with_its_year(tmp_path):
    import pandas as pd

    def rows_for(key, entrants, season_path, cutoff, year, extra=()):
        now = fake_rows({"Now ONE": 1100, "Now TWO": 1090})(key, entrants, season_path, cutoff, year)
        then = fake_rows({"Last YEAR": 1180}, mark_season=2025)(key, entrants, season_path, cutoff, year)
        return pd.concat([now, then], ignore_index=True)

    event = {"field": [{"discKey": "women_10000m", "athletes": athletes(("Now ONE", 1100), ("Now TWO", 1090))
                        + [{"name": "Last YEAR", "nat": "BRN", "score": None, "unranked": "noMark"}]}]}
    [call] = call_with(event, rows_for, tmp_path)["projections"]
    assert {a["name"]: a["markSeason"] for a in call["athletes"]} == {"Last YEAR": 2025, "Now ONE": None, "Now TWO": None}
    assert call["unranked"] == []


def test_an_event_with_fewer_than_three_marks_is_ranked_on_points_and_says_why(tmp_path):
    scores = {"Only ONE": 1233, "Only TWO": 1226}
    event = {"field": [{"discKey": "women_HT", "athletes": athletes(*scores.items())}]}
    [call] = call_with(event, fake_rows(scores), tmp_path)["projections"]
    assert (call["method"], call["methodEvidence"]["reason"]) == ("points", "tooFew")
    assert all(a["podiumChance"] is None for a in call["athletes"])


def test_the_page_states_the_models_test_from_the_report_file(tmp_path):
    report = tmp_path / "field_model_report.json"
    report.write_text(json.dumps({"testYears": [2021, 2022, 2023, 2024, 2025],
                                  "overall": {"finals": 511, "model": 64.9, "points": 64.1},
                                  "byTier": {"asia": {"finals": 108, "model": 64.8, "points": 67.0}}}),
                      encoding="utf-8")
    assert agp.backtest_summary(str(report)) == {
        "finals": 511, "model": 64.9, "points": 64.1, "asiaFinals": 108, "asiaModel": 64.8, "asiaPoints": 67.0,
        "years": [2021, 2022, 2023, 2024, 2025]}
    assert agp.backtest_summary(str(tmp_path / "missing.json")) is None


def test_the_page_states_the_locked_year_test_only_for_the_model_that_took_it(tmp_path):
    held = tmp_path / "holdout.json"
    held.write_text(json.dumps({
        "candidateName": "v2_form_best_5", "testYears": [2024, 2025],
        "overall": {"baseline": {"finals": 206, "model": 62.9, "points": 62.0},
                    "candidate": {"finals": 206, "model": 64.4, "points": 62.0}},
        "byTier": {"asia": {"baseline": {"finals": 36, "model": 65.7, "points": 70.4},
                            "candidate": {"finals": 36, "model": 67.6, "points": 70.4}}}}), encoding="utf-8")
    missing = str(tmp_path / "missing.json")
    assert agp.backtest_summary(missing, {"experiment": "v2_form_best_5"}, str(held)) == {
        "finals": 206, "model": 64.4, "previous": 62.9, "points": 62.0,
        "asiaFinals": 36, "asiaModel": 67.6, "asiaPrevious": 65.7, "asiaPoints": 70.4, "years": [2024, 2025]}
    # Another model, or one saved by --backtest, is never described by that test.
    assert agp.backtest_summary(missing, {"experiment": "recency"}, str(held)) is None
    assert agp.backtest_summary(missing, None, str(held)) is None


def test_a_model_that_reads_races_is_served_them_the_way_it_was_chosen_and_scored_on_its_features():
    import pandas as pd

    seen = {}

    def rows_for(key, entrants, season_path, cutoff, year, extra=(), races=None, race_how=None):
        seen.update(races=races, race_how=race_how)
        return pd.DataFrame({
            "athlete_name": ["A", "B", "C"], "sb_score": [1200.0, 1150.0, 1100.0],
            "sb_date": pd.Timestamp("2026-06-01"), "sb_prior_season": 0, "dob": pd.Timestamp("1998-01-01"),
            "career_best": [1210.0, 1160.0, 1110.0], "prev_season_best": [1190.0, 1140.0, 1090.0],
            "mark": ["1:44.0", "1:45.0", "1:46.0"], "mark_season": 2026, "wa_id": [1, 2, 3],
            "form_score": [1190.0, 1150.0, 1080.0], "form_spread": 10.0, "recent_score": [1195.0, None, 1090.0],
            "races": 6, "big_podiums": [2, 0, 0], "h2h_top": [1.0, 0.0, -1.0], "has_races": True})

    features = agp.fm.FEATURES_V2
    model = {"features": features, "experiment": "v2_form_best_5",
             "mean": [0.0] * len(features), "std": [1.0] * len(features), "weights": [0.0] * len(features)}
    model["weights"][features.index("form_gap")] = 1.0
    event = {"discKey": "men_800m", "athletes": [{"name": n, "nat": "KEN", "waId": i} for i, n in enumerate("ABC", 1)]}
    out = agp.field_call(event, model, "snapshot.csv", "last.csv", "2026-09-23", rows_for=rows_for, races={"1": []})
    assert seen == {"races": {"1": []}, "race_how": {"form_marks": 5}}
    assert [a["name"] for a in out["athletes"]] == ["A", "B", "C"]


def test_a_points_event_states_an_order_and_no_chances():
    event = {"discKey": "men_JT", "disciplineLabel": "Men's Javelin Throw", "sex": "M",
             "athletes": athletes(("Second ONE", 1150), ("First ONE", 1240))
                         + [{"name": "No MARK", "nat": "PAK", "score": None}]}
    out = agp.points_call(event)
    assert [a["name"] for a in out["athletes"]] == ["First ONE", "Second ONE"]
    assert all(a["podiumChance"] is None for a in out["athletes"])
    assert out["unscored"] == ["No MARK"]


def test_every_entrant_with_a_world_athletics_profile_has_a_page():
    """Ranked or not, and on the snapshot or not: the page is built from the
    call and the profile. Only an entrant matched to no profile has none."""
    event = {"athletes": [
        {"name": "On LIST", "profileUrl": "https://worldathletics.org/athletes/athlete=1"},
        {"name": "Off LIST", "profileUrl": "https://worldathletics.org/athletes/japan/off-list-2"},
        {"name": "No PROFILE", "profileUrl": None},
    ]}
    out = agp.link_athletes({
        "athletes": [{"name": "On LIST"}, {"name": "Off LIST"}, {"name": "No PROFILE"}],
        "unranked": [{"name": "No MARK", "profileUrl": "https://worldathletics.org/athletes/athlete=5"},
                     {"name": "Not FOUND", "profileUrl": None}],
    }, event, {"ON LIST"})
    assert [a["hasPage"] for a in out["athletes"]] == [True, True, False]
    assert out["athletes"][1]["profileUrl"].endswith("off-list-2")
    assert [u["hasPage"] for u in out["unranked"]] == [True, False]


def test_every_entrant_a_call_cannot_rank_comes_with_the_reason():
    event = {"discKey": "men_100m", "athletes": athletes(("Ranked ONE", 1200)) + [
        {"name": "Not FOUND", "nat": "AFG", "score": None, "unranked": "notFound", "profileUrl": None},
        {"name": "No MARK", "nat": "MDV", "score": None, "unranked": "noMark",
         "profileUrl": "https://worldathletics.org/athletes/athlete=5"}]}
    out = agp.points_call(event)
    assert [(u["name"], u["reason"]) for u in out["unranked"]] == [("Not FOUND", "notFound"), ("No MARK", "noMark")]
    assert out["unranked"][1]["profileUrl"].endswith("athlete=5")
    # A mark the model could not read is a reason of its own.
    assert agp.unranked_detail(event, ["Ranked ONE"])[0]["reason"] == "notScored"


# ---- the mark an entrant is ranked on -----------------------------------------

def last_list(tmp_path, key, *entries):
    """Last season's Asian list on disk, from (name, nat, mark, score, WA id)."""
    rows = []
    for i, (name, nat, mark, score, wid) in enumerate(entries, 1):
        cells = row(i, mark, name, nat, wid, score)
        cells[-3:-1] = [key, str(ags.LAST_YEAR)]
        rows.append(cells)
    ags.write_rows(str(tmp_path / f"{key}_{ags.LAST_YEAR}.csv"), HEADER, rows)
    return agp.last_season_marks(key, asia_dir=str(tmp_path))


def test_an_entrant_with_no_mark_this_season_is_ranked_on_last_seasons_best_and_says_so(tmp_path):
    last = last_list(tmp_path, "men_JT", ("Out LASTYEAR", "IND", "84.00", 1210, 101),
                     ("Ranked NOW", "PAK", "90.00", 1300, 102))
    field = [{"name": "Ranked NOW", "nat": "PAK", "waId": 102, "score": 1150, "mark": "80.00"},
             {"name": "Out LASTYEAR", "nat": "IND", "waId": 101, "score": None, "mark": None, "unranked": "noMark"},
             {"name": "Nobody ANYWHERE", "nat": "SRI", "score": None, "unranked": "noMark"}]
    out = {a["name"]: a for a in agp.last_season_rule(field, "men_JT", last)}
    # Outside the 5000m and 10,000m, a better mark last season does not replace this season's.
    assert (out["Ranked NOW"]["score"], out["Ranked NOW"]["markSeason"]) == (1150, None)
    assert (out["Out LASTYEAR"]["score"], out["Out LASTYEAR"]["mark"], out["Out LASTYEAR"]["markSeason"]) \
        == (1210, "84.00", ags.LAST_YEAR)
    assert out["Out LASTYEAR"]["unranked"] is None and out["Out LASTYEAR"]["seasonScore"] is None
    assert (out["Nobody ANYWHERE"]["score"], out["Nobody ANYWHERE"]["unranked"]) == (None, "noMark")
    call = agp.points_call({"discKey": "men_JT", "athletes": list(out.values())})
    assert [(a["name"], a["markSeason"]) for a in call["athletes"]] == [
        ("Out LASTYEAR", ags.LAST_YEAR), ("Ranked NOW", None)]
    assert [u["name"] for u in call["unranked"]] == ["Nobody ANYWHERE"]


def test_in_the_5000m_and_10000m_everyone_takes_the_better_of_the_two_seasons(tmp_path):
    last = last_list(tmp_path, "women_10000m", ("Raced ONCE", "BRN", "30:40.00", 1180, None),
                     ("Better NOW", "JPN", "32:00.00", 1090, None))
    field = [{"name": "Raced ONCE", "nat": "BRN", "score": 1050, "mark": "33:10.00"},
             {"name": "Better NOW", "nat": "JPN", "score": 1120, "mark": "31:30.00"}]
    out = {a["name"]: a for a in agp.last_season_rule(field, "women_10000m", last)}
    assert (out["Raced ONCE"]["score"], out["Raced ONCE"]["markSeason"], out["Raced ONCE"]["seasonScore"]) \
        == (1180, ags.LAST_YEAR, 1050)
    assert (out["Better NOW"]["score"], out["Better NOW"]["markSeason"]) == (1120, None)


def test_last_season_is_found_by_id_or_by_a_name_spelt_or_ordered_differently(tmp_path):
    last = last_list(tmp_path, "men_JT", ("Tae-poong NAM", "KOR", "80.00", 1150, None),
                     ("MYAGMARSUREN Dawaanyam", "MGL", "60.00", 900, None),
                     ("Some NAME", "IND", "70.00", 1000, 555))
    field = [{"name": "Taepoong NAM", "nat": "KOR", "score": None},
             {"name": "Dawaanyam MYAGMARSUREN", "nat": "MGL", "score": None},
             {"name": "Other SPELLING", "nat": "IND", "waId": 555, "score": None},
             {"name": "Tae-poong NAM", "nat": "PRK", "score": None}]
    assert [a["score"] for a in agp.last_season_rule(field, "men_JT", last)] == [1150, 900, 1000, None]


def test_a_failed_lookup_is_not_papered_over_with_last_season(tmp_path):
    last = last_list(tmp_path, "men_JT", ("Lookup FAILED", "IND", "80.00", 1150, 7))
    [a] = agp.last_season_rule([{"name": "Lookup FAILED", "nat": "IND", "waId": 7, "score": None,
                                 "unranked": "lookupFailed"}], "men_JT", last)
    assert (a["score"], a["unranked"]) == (None, "lookupFailed")


def test_last_seasons_asian_list_is_read_from_its_own_page_down_to_the_last_page():
    urls = []

    def fetch(url):
        urls.append(url)
        return page((1, "13.40", "Only LASTYEAR", "CHN", 9, 1180))

    header, rows = ags.scrape_asian_toplist("men_110h", None, ags.ASIAN_NATIONS, fetch=fetch, year=ags.LAST_YEAR)
    assert len(urls) == ags.MAX_PAGES and all(f"/{ags.LAST_YEAR}?" in u for u in urls)
    assert len(rows) == 1 and rows[0][-2] == str(ags.LAST_YEAR)
    assert ags.asian_toplist_url("men_100m").endswith(f"/{ags.YEAR}?{ags.ASIA_QUERY}")


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


def test_an_athlete_page_carries_the_call_until_the_championship_ends(monkeypatch):
    import api
    from datetime import date

    preds = {"projections": [{
        "discKey": "men_100m", "method": "points", "qualified": 3,
        "athletes": [{"rank": 2, "name": "Sorsy PHOMPAKDI", "rankingScore": 877, "podiumChance": None,
                      "mark": "10.61", "markSeason": 2025, "nat": "LAO", "profileUrl": "u-sorsy"}],
        "unranked": [{"name": "Omar CHAABAN", "reason": "noMark", "nat": "PLE", "profileUrl": "u-omar"}]}]}
    monkeypatch.setattr(api, "load_event_predictions", lambda champ_id: preds)
    monkeypatch.setattr(api.championships, "current", lambda: {
        "id": "asian-games-2026", "labelKey": "results.meet.asianGames", "theme": "asianGames",
        "endDate": "2026-09-29"})
    during = date(2026, 9, 20)

    call = api.championship_call("men_100m", "sorsy phompakdi", today=during)
    assert (call["method"], call["rank"], call["rankingScore"], call["ranked"], call["entered"]) == \
        ("points", 2, 877, 1, 3)
    assert (call["mark"], call["markSeason"], call["nat"], call["profileUrl"]) == ("10.61", 2025, "LAO", "u-sorsy")
    unranked = api.championship_call("men_100m", "Omar CHAABAN", today=during)
    assert (unranked["unranked"], unranked["mark"], unranked["profileUrl"]) == ("noMark", None, "u-omar")
    assert api.championship_call("men_200m", "Sorsy PHOMPAKDI", today=during) is None
    assert api.championship_call("men_100m", "Sorsy PHOMPAKDI", today=date(2026, 9, 30)) is None


def test_an_entrant_with_no_row_this_season_gets_a_page_built_from_the_call(monkeypatch):
    """The 23 entrants ranked on a 2025 mark and the unranked ones with a World
    Athletics profile had no page until 2026-09-15. Their results and photo come
    from the profile the call carries, and the mark is labelled with its year."""
    import api

    call = {"id": "asian-games-2026", "labelKey": "results.meet.asianGames", "theme": "asianGames",
            "method": "model", "entered": 12, "ranked": 10, "rank": 9, "podiumChance": 2.1,
            "rankingScore": 1040, "unranked": None, "mark": "46.10", "markSeason": 2025,
            "nat": "PAK", "profileUrl": "https://worldathletics.org/athletes/athlete=14800001"}
    seen = {}

    def history(disc_key, name, url):
        seen["history"] = url
        return [], None, False, 0

    def photo(url):
        seen["photo"] = url
        return None, None

    monkeypatch.setattr(api, "toplist_entry", lambda disc_key, name: (None, None, None))
    monkeypatch.setattr(api, "toplist_bio", lambda disc_key, name: {})
    monkeypatch.setattr(api, "championship_call", lambda disc_key, name: call)
    monkeypatch.setattr(api, "load_predictions", lambda: ([], []))
    monkeypatch.setattr(api, "scored_prediction_row", lambda disc_key, name: None)
    monkeypatch.setattr(api, "load_career_progression", lambda disc_key, name: [{"year": 2024, "best": 45.8}])
    monkeypatch.setattr(api, "load_athlete_history", history)
    monkeypatch.setattr(api, "resolve_athlete_photo", photo)
    monkeypatch.setattr(api, "athlete_score_context", lambda disc_key, name: None)
    monkeypatch.setattr(api, "projected_field_names", lambda disc_key: [])
    monkeypatch.setattr(api.athlete_analytics, "build_analytics", lambda *args: None)
    monkeypatch.setattr(api.athlete_career, "build_career", lambda name: None)

    out = api.athlete_field_status("men_400m", "Shajar ABBAS")
    assert (out["seasonBest"], out["seasonBestYear"], out["nat"]) == ("46.10", 2025, "PAK")
    assert out["waUrl"] == call["profileUrl"]
    assert seen == {"history": call["profileUrl"], "photo": call["profileUrl"]}
    # A 2025 mark is no season best to measure the gap to a career best from.
    assert out["careerBest"] is not None and out["pbGap"] is None
    assert out["reasonCode"] == "championship_entrant" and out["championship"] is call


def test_an_entrant_on_no_world_toplist_is_read_from_the_championship_snapshot(monkeypatch, tmp_path):
    """What gives the 281 ranked entrants who are on no world toplist a page."""
    import api

    header = "Rank,Mark,WIND,Competitor,DOB,,Pos,,Venue,Date,Results Score,discipline,year,ProfileURL\n"
    lyles = "1,9.79,+0.8,Noah LYLES,18 JUL 1997,USA,1,,New York,24 JUL 2026,1280,men_100m,2026,u-lyles\n"
    adam = (",11.02,+1.1,Ibadulla ADAM,14 JAN 2002,MDV,1,,Gaborone,27 JUN 2026,880,men_100m,2026,"
            "https://worldathletics.org/athletes/athlete=14852443\n")
    world = tmp_path / "raw"
    world.mkdir()
    (world / f"men_100m_{api.MEETS_YEAR}.csv").write_text(header + lyles, encoding="utf-8")
    snap = tmp_path / "ag" / "raw"
    snap.mkdir(parents=True)
    (snap / f"men_100m_{api.MEETS_YEAR}.csv").write_text(header + lyles + adam, encoding="utf-8")
    monkeypatch.setattr(api, "RAW_DIR", str(world))
    monkeypatch.setattr(api.championships, "CHAMPIONSHIPS", [{"id": "x", "dataDir": str(tmp_path / "ag")}])

    assert api.toplist_entry("men_100m", "Noah LYLES") == ("9.79", 1, "u-lyles")
    assert api.toplist_entry("men_100m", "Ibadulla ADAM") == (
        "11.02", None, "https://worldathletics.org/athletes/athlete=14852443")
    assert api.toplist_bio("men_100m", "Ibadulla ADAM")["nat"] == "MDV"
    rows = {(name, key, rank) for name, key, _mark, rank in api.build_search_index()["athletes"]}
    assert ("Ibadulla ADAM", "men_100m", None) in rows
    # Once each, though the snapshot repeats the world list.
    assert sum(1 for name, key, _ in rows if name == "Noah LYLES") == 1


def test_the_profile_fetch_takes_every_entrant_with_an_id(monkeypatch, tmp_path):
    """Entrants on a 2025 mark or none have pages since 2026-09-15, so their
    profiles are fetched too."""
    import athlete_profile_scraper as aps
    import championships

    event = tmp_path / "event.json"
    event.write_text(json.dumps({"field": [{"athletes": [
        {"name": "With MARK", "waId": 1, "score": 1100},
        {"name": "No MARK", "waId": 2, "score": None},
        {"name": "No ID", "waId": None, "score": None}]}]}), encoding="utf-8")
    monkeypatch.setattr(championships, "get", lambda champ_id: {"id": champ_id})
    monkeypatch.setattr(championships, "path", lambda champ, filename: str(event))
    assert aps.championship_athletes("asian-games-2026") == {"With MARK": "1", "No MARK": "2"}


def test_search_finds_an_entrant_whose_page_is_built_from_the_call(monkeypatch, tmp_path):
    """Pages for entrants in no snapshot came on 2026-09-15; search holds them
    for as long as the page is built, and with no mark."""
    import api
    from datetime import date

    monkeypatch.setattr(api.championships, "current", lambda: {"id": "asian-games-2026", "endDate": "2026-09-29"})
    monkeypatch.setattr(api, "load_event_predictions", lambda champ_id: {"projections": [{
        "discKey": "men_200m",
        "athletes": [{"name": "Shajar ABBAS", "hasPage": True}, {"name": "No PROFILE", "hasPage": False}],
        "unranked": [{"name": "Omar CHAABAN", "hasPage": True}]}]})
    assert api.championship_page_names(today=date(2026, 9, 20)) == {"men_200m": ["Shajar ABBAS", "Omar CHAABAN"]}
    assert api.championship_page_names(today=date(2026, 9, 30)) == {}

    monkeypatch.setattr(api, "RAW_DIR", str(tmp_path))
    monkeypatch.setattr(api.championships, "CHAMPIONSHIPS", [])
    monkeypatch.setattr(api, "championship_page_names", lambda: {"men_200m": ["Shajar ABBAS", "Omar CHAABAN"]})
    rows = api.build_search_index()["athletes"]
    assert ["Shajar ABBAS", "men_200m", None, None] in rows
    assert ["Omar CHAABAN", "men_200m", None, None] in rows


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
