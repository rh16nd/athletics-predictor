"""Tests for the field model (src/field_model.py) and the data it learns from
(src/field_data.py).

Everything here is synthetic or faked: no network, nothing read from data/.
The things worth guarding are the ones that would look like a working model
while being wrong -- a cut-off that lets in a mark from the championship itself,
chances that do not add up, features that secretly depend on the world list,
and a ship rule that lets a model through without beating points.
"""
import itertools

import numpy as np
import pandas as pd
import pytest

import field_data as fd
import field_model as fm
import ultimate_scraper as us


# ---- podium chances -----------------------------------------------------------

def brute_force_top3(u):
    """P(top three) by enumerating every ordered top three under Plackett-Luce."""
    e = np.exp(np.asarray(u, dtype=float))
    n = len(e)
    p = np.zeros(n)
    for a, b, c in itertools.permutations(range(n), 3):
        pa = e[a] / e.sum()
        pb = e[b] / (e.sum() - e[a])
        pc = e[c] / (e.sum() - e[a] - e[b])
        prob = pa * pb * pc
        p[[a, b, c]] += prob
    return p


def test_podium_chances_are_exact_add_up_to_three_and_rise_with_utility():
    u = [2.0, 1.2, 0.5, 0.0, -0.7, -1.5]
    chances = fm.podium_chances(u)
    assert np.allclose(chances, brute_force_top3(u))
    assert chances.sum() == pytest.approx(3.0)
    assert all(a > b for a, b in zip(chances, chances[1:]))


def test_a_field_of_three_or_fewer_is_certain():
    assert list(fm.podium_chances([0.3, -1.0, 2.0])) == [1.0, 1.0, 1.0]


# ---- features -----------------------------------------------------------------

def field(scores, cutoff="2023-09-29", **extra):
    n = len(scores)
    return pd.DataFrame({
        "athlete_name": [f"A{i}" for i in range(n)],
        "sb_score": scores,
        "sb_date": extra.get("sb_date", [pd.Timestamp("2023-06-01")] * n),
        "sb_prior_season": extra.get("prior", [0] * n),
        "career_best": extra.get("career", [s + 10 for s in scores]),
        "prev_season_best": extra.get("prev", [s - 20 for s in scores]),
        "dob": extra.get("dob", [pd.Timestamp("1998-01-01")] * n),
    })


def test_features_depend_on_the_field_not_on_the_world_scale():
    """An Asian field 200 points below a world final, spaced the same way,
    reads the same: its leader is a leader."""
    low = fm.field_features(field([1000, 980, 950, 900, 880]), "2023-09-29")
    high = fm.field_features(field([1200, 1180, 1150, 1100, 1080]), "2023-09-29")
    assert np.allclose(low.to_numpy(), high.to_numpy())
    assert low["gap_best"].iloc[0] == 0 and low["gap_third"].iloc[2] == 0


def test_slower_entrants_beside_a_final_do_not_change_how_its_athletes_read():
    """Trained on finals, asked about entry lists: the eight who would make a
    final must read the same with twenty slower entrants beside them."""
    top = [1200, 1190, 1180, 1150, 1140, 1120, 1110, 1100]
    final = fm.field_features(field(top), "2023-09-29").to_numpy()
    entry_list = fm.field_features(field(top + list(range(1000, 800, -10))), "2023-09-29").to_numpy()
    assert np.allclose(final, entry_list[:8])


def test_a_season_best_from_last_season_carries_no_improvement_and_is_flagged():
    rows = field([1100, 1050, 1000], prior=[0, 1, 0])
    f = fm.field_features(rows, "2023-09-29")
    assert f["sb_prior_season"].tolist() == [0.0, 1.0, 0.0]
    assert f["yoy"].iloc[1] == 0.0 and f["yoy"].iloc[0] == pytest.approx(0.2)


def test_no_history_is_its_own_signal_rather_than_a_zero_gap_alone():
    rows = field([1100, 1050, 1000], career=[1110, None, 1005])
    f = fm.field_features(rows, "2023-09-29")
    assert f["no_history"].tolist() == [0.0, 1.0, 0.0]


# ---- season scores and the cut-off --------------------------------------------

def history(*rows):
    return pd.DataFrame([{"key": k, "nat": n, "year": y, "score": s, "date": pd.Timestamp(d),
                          "dob": pd.Timestamp("1995-05-05"), "name": k, "mark": "", "source": "asia"}
                         for k, n, y, s, d in rows])


def test_a_season_best_the_toplist_cannot_date_before_the_cut_off_comes_from_the_profile_or_not_at_all():
    """Sending these finalists straight to last season was a leak: whether a
    toplist best came after the cut-off depends on the championship itself."""
    cutoff = pd.Timestamp("2015-08-22")
    h = history(("EARLY", "KEN", 2015, 1200, "2015-06-01"), ("EARLY", "KEN", 2014, 1180, "2014-06-01"),
                ("PEAKED", "KEN", 2015, 1230, "2015-08-25"), ("PEAKED", "KEN", 2014, 1190, "2014-06-01"),
                ("NOPROFILE", "KEN", 2015, 1225, "2015-08-25"), ("NOPROFILE", "KEN", 2014, 1195, "2014-06-01"),
                ("INJURED", "KEN", 2015, 1210, "2015-09-05"), ("INJURED", "KEN", 2014, 1205, "2014-06-01"))
    finals = pd.DataFrame({"discipline": "men_800m", "cutoff": cutoff, "year": 2015, "nationality": "KEN",
                           "athlete_name": ["EARLY", "PEAKED", "NOPROFILE", "INJURED"],
                           "athlete_id": [1, 2, None, 4]})
    seasons = {2015: {"2": [{"discipline": "800 Metres", "indoor": False, "results": [
                          {"date": "01 JUL 2015", "mark": "1:44.00", "notLegal": False, "resultScore": 1195},
                          {"date": "25 AUG 2015", "mark": "1:42.90", "notLegal": False, "resultScore": 1230}]}],
                      "4": []}}
    out = fd.attach_scores(finals, scores_for=lambda key: h, seasons=seasons).set_index("athlete_name")
    columns = ["sb_score", "sb_source", "sb_prior_season"]
    assert out.loc["EARLY", columns].tolist() == [1200.0, "toplist", 0]
    assert out.loc["PEAKED", columns].tolist() == [1195.0, "profile", 0]
    assert pd.isna(out.loc["NOPROFILE", "sb_score"]) and bool(out.loc["NOPROFILE", "needs_profile"])
    assert out.loc["INJURED", columns].tolist() == [1205.0, "lastSeason", 1]
    assert not bool(out.loc["EARLY", "needs_profile"])


def test_a_profile_mark_counts_only_before_the_cut_off_outdoors_legal_and_electronically_timed():
    events = [
        {"discipline": "800 Metres", "indoor": True, "results": [
            {"date": "01 FEB 2015", "mark": "1:44.00", "notLegal": False, "resultScore": 1250}]},
        {"discipline": "800 Metres", "indoor": False, "results": [
            {"date": "13 JUN 2015", "mark": "1:43.58", "notLegal": False, "resultScore": 1217},
            {"date": "31 JUL 2015", "mark": "1:43.2h", "notLegal": False, "resultScore": 1230},
            {"date": "01 AUG 2015", "mark": "1:43.00", "notLegal": True, "resultScore": 1235},
            {"date": "22 AUG 2015", "mark": "1:43.10", "notLegal": False, "resultScore": 1232},
            {"date": "26 MAY 2015", "mark": "DNF", "notLegal": False, "resultScore": 0}]},
        {"discipline": "600 Metres", "indoor": False, "results": [
            {"date": "01 MAY 2015", "mark": "1:13.00", "notLegal": False, "resultScore": 1300}]},
    ]
    assert fd.profile_best(events, "men_800m", pd.Timestamp("2015-08-22")) == (1217.0, pd.Timestamp("2015-06-13"))
    assert fd.profile_best(events, "men_800m", pd.Timestamp("2015-03-01")) is None


def test_finalist_ids_come_from_each_competitions_results_feed(tmp_path):
    assert "urlSlug" in fd.IDS_QUERY
    assert fd.slug_id("islamic-republic-of-iran/hassan-taftian-14421587") == 14421587
    assert fd.slug_id(None) is None and fd.slug_id("no-number-here") is None

    def query(competition_id, day):
        if day is None:
            return {"options": {"days": [{"day": 1}]}}
        return {"eventTitles": [{"events": [{"gender": "M", "event": "Men's 100 Metres", "races": [
            {"race": "Final", "results": [{"competitor": {
                "name": "Hassan TAFTIAN", "urlSlug": "islamic-republic-of-iran/hassan-taftian-14421587",
                "birthDate": "04 MAY 1993"}}]}]}]}]}

    taftian = (fd.field_key("Hassan TAFTIAN"))
    assert fd.competition_athletes(7147637, query=query) == {
        ("men_100m", taftian): (14421587, pd.Timestamp("1993-05-04"))}

    finals = pd.DataFrame({"competition": ["Asian Games 2023", "Diamond League Final", "Diamond League Final"],
                           "competition_id": [7147637, None, None], "year": [2023, 2019, 2019],
                           "discipline": "men_100m", "athlete_name": ["Hassan TAFTIAN", "Early FINAL", "Late FINAL"]})
    meetings = {7147637: {("men_100m", taftian): (14421587, None)},
                1: {("men_100m", fd.field_key("Early FINAL")): (11, None)},
                2: {("men_100m", fd.field_key("Late FINAL")): (22, None)}}
    ids = fd.finalist_ids(finals, athletes_at=lambda m: meetings[m], dl_meetings=lambda year: [{"id": 1}, {"id": 2}])
    assert ids.set_index("athlete_name")["athlete_id"].to_dict() == {
        "Hassan TAFTIAN": 14421587, "Early FINAL": 11, "Late FINAL": 22}


def test_a_season_that_fails_to_download_is_asked_for_again_and_an_empty_one_is_not(tmp_path):
    calls = []

    def fetch(athlete, year):
        calls.append((athlete, year))
        if athlete == 2:
            raise ConnectionError("CloudFront error page")
        return []

    assert fd.fetch_seasons({(1, 2015), (2, 2015)}, fetch=fetch, seasons_dir=str(tmp_path)) == 1
    assert fd.load_seasons(str(tmp_path)) == {2015: {"1": []}}
    fd.fetch_seasons({(1, 2015), (2, 2015)}, fetch=fetch, seasons_dir=str(tmp_path))
    assert calls == [(1, 2015), (2, 2015), (2, 2015)]


def test_names_match_through_punctuation_and_by_nationality_first():
    assert fd.field_key("Tae-poong NAM") == fd.field_key("Taepoong NAM")
    h = history(("SAMBA", "QAT", 2023, 1250, "2023-05-01"), ("SAMBA", "MTN", 2016, 1100, "2016-05-01"))
    rows, how = fd.athlete_history(h, "SAMBA", "QAT")
    assert (len(rows), how) == (1, "nameNat")
    # A name held under two nationalities is not guessed at.
    assert fd.athlete_history(h, "SAMBA", "BRN")[1] is None
    single = history(("ALI", "BRN", 2019, 1100, "2019-05-01"))
    assert fd.athlete_history(single, "ALI", "KEN")[1] == "name"


# ---- finals -------------------------------------------------------------------

def test_a_competition_with_no_results_is_skipped_and_the_whole_order_is_kept():
    def fetch(field=None, competition_id=None):
        if competition_id == 2:
            return []
        return [{"discipline": "men_100m", "athlete_name": "A", "place": "1.", "mark": "10.0", "nationality": "JPN"},
                {"discipline": "men_100m", "athlete_name": "B", "place": "DNF", "mark": "DNF", "nationality": "CHN"}]

    comps = [{"id": 1, "competition": "Asian Games 2018", "year": 2018, "tier": "asia"},
             {"id": 2, "competition": "Asian Championships 2019", "year": 2019, "tier": "asia"}]
    finals, missing = fd.competition_finals(comps, fetch=fetch, start=lambda cid: pd.Timestamp("2018-08-25"))
    assert missing == ["Asian Championships 2019"]
    assert finals["athlete_name"].tolist() == ["A", "B"]
    assert finals["place"].tolist()[0] == 1 and pd.isna(finals["place"].tolist()[1])


TH = ("<tr><th>Rank</th><th>Mark</th><th>WIND</th><th>Competitor</th><th>DOB</th><th></th>"
      "<th>Pos</th><th></th><th>Venue</th><th>Date</th><th>Results Score</th></tr>")


def toplist_page(nations):
    body = "".join(
        f'<tr><td>{i + 1}</td><td>10.{i:02d}</td><td></td><td><a href="/athletes/athlete={i}">Athlete {i}</a></td>'
        f"<td>01 JAN 2000</td><td>{nat}</td><td>1</td><td></td><td>Somewhere</td><td>01 JUN 2019</td><td>1100</td></tr>"
        for i, nat in enumerate(nations))
    return f"<table>{TH}{body}</table>"


def test_a_past_area_list_that_is_really_the_world_list_is_refused():
    with pytest.raises(ValueError, match="area filter"):
        fd.area_toplist("men_100m", 2019, "asia", fetch=lambda url: toplist_page(["USA", "JAM", "KEN", "JPN"]), pause=0)
    header, rows = fd.area_toplist("men_100m", 2019, "asia", fetch=lambda url: toplist_page(["JPN", "CHN", "QAT"]), pause=0)
    assert len(rows) == 3 and header[-3:] == ["discipline", "year", "ProfileURL"]
    assert "region=europe" in fd.area_toplist_url("women_HT", 2012, "europe")
    with pytest.raises(ValueError, match="outside europe"):
        fd.area_toplist("men_100m", 2012, "europe", fetch=lambda url: toplist_page(["JPN", "CHN", "GBR"]), pause=0)
    _, rows = fd.area_toplist("men_100m", 2012, "europe", fetch=lambda url: toplist_page(["GBR", "GER", "ANA"]), pause=0)
    assert len(rows) == 3


def test_a_discipline_with_a_season_that_failed_is_not_saved_so_a_rerun_fetches_it(tmp_path):
    def fetch(url):
        if "/2016?" in url:
            raise ConnectionError("reset by peer")
        return toplist_page(["GBR", "FRA"])

    fd.scrape_area_toplists(["men_100m"], "europe", years=[2015, 2016], fetch=fetch, out_dir=str(tmp_path), pause=0)
    assert not (tmp_path / "men_100m.csv").exists()
    fd.scrape_area_toplists(["men_100m"], "europe", years=[2015], fetch=fetch, out_dir=str(tmp_path), pause=0)
    assert len(pd.read_csv(tmp_path / "men_100m.csv")) == 2


def scored_final(competition, year, group_event, n=6, podium=("A0", "A1", "A2"), unscored=()):
    rows = []
    for i in range(n):
        name = f"A{i}"
        rows.append({
            "competition": competition, "year": year, "discipline": f"men_{group_event}", "tier": "asia",
            "cutoff": pd.Timestamp(f"{year}-09-01"), "athlete_name": name,
            "place": (podium.index(name) + 1) if name in podium else i + 4,
            "sb_score": None if name in unscored else 1200 - 10 * i,
            "sb_date": pd.Timestamp(f"{year}-06-01"), "sb_prior_season": 0,
            "career_best": 1210 - 10 * i, "prev_season_best": 1190 - 10 * i, "dob": pd.Timestamp("1998-01-01"),
        })
    return pd.DataFrame(rows)


def test_an_unscored_medallist_is_still_a_place_to_find_and_thin_fields_are_skipped():
    data = pd.concat([
        scored_final("Asian Games", 2023, "100m", n=7, unscored=("A0",)),
        scored_final("Asian Championships", 2023, "200m", n=5, unscored=("A3",)),
    ])
    [final] = fm.build_finals(data)
    assert final["discipline"] == "men_100m"
    assert "A0" in final["podium"] and "A0" not in final["names"]
    assert final["winners"] == {"A0"}
    assert final["top_idx"] is None


def test_a_competition_with_too_few_finalists_found_is_scored_but_not_trained_on():
    found = scored_final("Asian Games", 2022, "100m", n=8)
    # 6 of 8 found is 75%: under the gate, yet a field with a scored podium.
    thin = scored_final("Asian Championships", 2022, "200m", n=8, unscored=("A6", "A7"))
    # 17 of 20 is exactly the gate.
    edge = scored_final("Asian Championships", 2023, "400m", n=20, unscored=("A17", "A18", "A19"))
    data = pd.concat([found, thin, edge], ignore_index=True)
    finals = {(f["competition"], f["year"]): f for f in fm.build_finals(data)}
    assert finals[("Asian Games", 2022)]["trainable"] is True
    assert finals[("Asian Championships", 2022)]["trainable"] is False
    assert finals[("Asian Championships", 2022)]["top_idx"] is not None
    assert finals[("Asian Championships", 2023)]["trainable"] is True
    assert fm.fit(list(finals.values()))["finals"] == 2
    with pytest.raises(ValueError, match="trainable"):
        fm.fit([finals[("Asian Championships", 2022)]])
    table = fd.coverage(data).set_index(["competition", "year"])
    assert table["trained"].to_dict() == {("Asian Championships", 2022): False,
                                          ("Asian Championships", 2023): True,
                                          ("Asian Games", 2022): True}


def test_a_shared_bronze_puts_both_medallists_on_the_podium():
    data = scored_final("World Championships", 2015, "100m", n=8)
    data.loc[data["athlete_name"] == "A3", "place"] = 3
    [final] = fm.build_finals(data)
    assert final["podium"] == {"A0", "A1", "A2", "A3"}


def test_a_tie_on_points_is_not_broken_by_the_finishing_order_the_rows_arrive_in():
    data = scored_final("World Championships", 2023, "100m", n=8)
    data.loc[data["athlete_name"] == "A5", "sb_score"] = 1180  # level with A2, the bronze medallist
    in_order = fm.build_finals(data)[0]
    reversed_rows = fm.build_finals(data.iloc[::-1])[0]
    assert in_order["names"] == reversed_rows["names"]
    assert np.allclose(in_order["scores"], reversed_rows["scores"])
    assert np.allclose(in_order["X"], reversed_rows["X"])
    assert in_order["top_idx"] == reversed_rows["top_idx"]


def test_the_seasons_ranked_names_pick_the_championship_final_over_a_fuller_masters_race():
    """The 2015 Worlds had two races labelled "Men's 800 Metres" Final. The
    masters one had 10 finishers to the real one's 8, and was kept."""
    histories = {"men_800m": pd.DataFrame({"name": ["David RUDISHA", "Nijel AMOS", "David RUDISHA"],
                                           "year": [2015, 2014, 2014]})}
    seen = {}

    def fetch(field=None, competition_id=None):
        seen[competition_id] = field
        return [{"discipline": "men_800m", "athlete_name": "David RUDISHA", "place": "1.", "mark": "1:45.84"}]

    comps = [{"id": 7078726, "competition": "IAAF World Championships", "year": 2015, "tier": "global"}]
    fd.competition_finals(comps, fetch=fetch, start=lambda cid: pd.Timestamp("2015-08-22"),
                          field_for=lambda year: fd.season_field(year, histories))
    assert seen[7078726] == [{"discKey": "men_800m", "athletes": [{"name": "David RUDISHA"}]}]

    masters = {"race": "Final", "results": [{"competitor": {"name": f"Masters {i}"}} for i in range(10)]}
    real = {"race": "Final", "results": [{"competitor": {"name": "David RUDISHA"}}]
            + [{"competitor": {"name": f"Finalist {i}"}} for i in range(7)]}
    entrants = us.entrants_by_key(seen[7078726])["men_800m"]
    assert us.pick_final([masters, real], entrants) is real


def test_the_winner_check_names_a_final_whose_saved_winner_is_not_the_major_meet_winner(tmp_path):
    (tmp_path / "men_800m.csv").write_text(
        "Rank,Mark,WIND,Competitor,DOB,Nat,Pos,Unnamed: 7,Venue,Date,Results Score,discipline,year,ProfileURL,source\n"
        ",1:45.84,,David RUDISHA,,KEN,1.,,IAAF World Championships,,,,2015,,major_meet\n"
        ",1:46.08,,Adam KSZCZOT,,POL,2.,,IAAF World Championships,,,,2015,,major_meet\n",
        encoding="utf-8")
    saved = pd.DataFrame({"competition": "IAAF World Championships", "year": 2015, "tier": "global",
                          "discipline": "men_800m", "athlete_name": ["David HEATH", "Michael SHERAR"],
                          "place": [1, 2]})
    checked, disagree = fd.winner_disagreements(saved, raw_dir=str(tmp_path))
    assert checked == 1
    assert disagree == [{"competition": "IAAF World Championships", "year": 2015, "discipline": "men_800m",
                         "majorMeet": ["DAVIDRUDISHA"], "saved": ["DAVIDHEATH"]}]
    fixed = saved.assign(athlete_name=["David RUDISHA", "Adam KSZCZOT"])
    assert fd.winner_disagreements(fixed, raw_dir=str(tmp_path)) == (1, [])


# ---- the ship rule ------------------------------------------------------------

def synthetic_finals(seed=3, per_year=12):
    """Finals where the podium follows improvement on last season far more
    than the season-best score, so a model that reads both should beat points."""
    rng = np.random.default_rng(seed)
    finals = []
    for year in range(2015, 2026):
        for j in range(per_year):
            n = 8
            scores = rng.uniform(1100, 1200, n)
            yoy = rng.normal(0, 40, n)
            rows = pd.DataFrame({
                "athlete_name": [f"{year}-{j}-{i}" for i in range(n)], "sb_score": scores,
                "sb_date": [pd.Timestamp(f"{year}-06-01")] * n, "sb_prior_season": [0] * n,
                "career_best": scores + 5, "prev_season_best": scores - yoy, "dob": [pd.Timestamp("1998-01-01")] * n,
            })
            X = fm.field_features(rows, f"{year}-09-01").to_numpy()
            utility = 0.08 * yoy + 0.002 * (scores - 1150)
            order = np.argsort(-(utility + rng.gumbel(size=n)))
            names = rows["athlete_name"].tolist()
            finals.append({"competition": f"C{j % 2}", "year": year, "discipline": "men_100m", "group": "sprints",
                           "tier": "asia" if j % 2 else "global", "names": names, "scores": scores, "X": X,
                           "podium": {names[i] for i in order[:3]}, "top_idx": [int(i) for i in order[:3]]})
    return finals


def test_a_model_that_beats_points_ships_and_its_shuffled_control_does_not():
    finals = synthetic_finals()
    results, athletes = fm.backtest(finals)
    control, _ = fm.backtest(finals, shuffle_seed=7)
    decision = fm.ship_decisions(results, control)["sprints"]
    assert decision["modelHits"] > decision["pointsHits"]
    assert decision["passed"] is True
    assert decision["controlLower90"] <= 0
    # Chances still add up to three per field.
    assert athletes["chance"].sum() == pytest.approx(3.0 * len(results), rel=1e-6)


def results_frame(model, points, tiers):
    return pd.DataFrame({"group": "throws", "tier": tiers, "model_hits": model, "points_hits": points})


def test_no_better_than_points_stays_on_points():
    same = results_frame([1, 2, 2, 1] * 10, [1, 2, 2, 1] * 10, ["asia", "global"] * 20)
    assert fm.ship_decisions(same, same.assign(model_hits=0))["throws"]["passed"] is False


def test_a_win_that_loses_on_the_asian_finals_stays_on_points():
    # Better on every global final, worse on every Asian one.
    wins_globally = results_frame([3] * 30 + [1] * 10, [2] * 30 + [2] * 10, ["global"] * 30 + ["asia"] * 10)
    control = wins_globally.assign(model_hits=wins_globally["points_hits"])
    assert fm.ship_decisions(wins_globally, control)["throws"]["passed"] is False


def test_a_control_that_also_passes_means_nothing_ships():
    better = results_frame([3] * 40, [2] * 40, ["asia", "global"] * 20)
    assert fm.ship_decisions(better, better)["throws"]["passed"] is False


# ---- serving ------------------------------------------------------------------

def test_a_field_is_scored_from_this_seasons_list_and_its_chances_add_to_three(tmp_path):
    season = tmp_path / "men_100m_2026.csv"
    season.write_text(
        "Rank,Mark,WIND,Competitor,DOB,,Pos,,Venue,Date,Results Score,discipline,year,ProfileURL\n"
        "1,10.00,,Shuhei TADA,24 JUN 1996,JPN,1,,Tokyo,01 JUN 2026,1200,men_100m,2026,u1\n"
        "2,10.10,,Xin LI,01 JAN 2000,CHN,1,,Beijing,01 MAY 2026,1170,men_100m,2026,u2\n"
        "3,10.20,,Ali AHMED,01 JAN 2001,QAT,1,,Doha,01 JUL 2026,1150,men_100m,2026,u3\n"
        "4,10.30,,Late MARK,01 JAN 2001,KOR,1,,Seoul,30 SEP 2026,1140,men_100m,2026,u4\n",
        encoding="utf-8")
    athletes = [{"name": "Shuhei TADA", "nat": "JPN"}, {"name": "Xin LI", "nat": "CHN"},
                {"name": "Ali AHMED", "nat": "QAT"}, {"name": "Late MARK", "nat": "KOR"},
                {"name": "Not LISTED", "nat": "MGL"}]
    empty = lambda key: fd.season_scores(key, raw_dir=str(tmp_path), area_dirs={"asia": str(tmp_path)})  # noqa: E731
    rows = fm.serving_rows("men_100m", athletes, str(season), "2026-09-23", 2026, scores_for=empty)
    # A mark set after the cut-off, and an athlete on no list, are not in the field.
    assert rows["athlete_name"].tolist() == ["Shuhei TADA", "Xin LI", "Ali AHMED"]
    model = {"features": fm.FEATURES, "mean": [0.0] * len(fm.FEATURES), "std": [1.0] * len(fm.FEATURES),
             "weights": [1.0] + [0.0] * (len(fm.FEATURES) - 1)}
    chances = fm.score_field(model, rows.assign(athlete_name=rows["athlete_name"]), "2026-09-23")
    assert chances == {name: 1.0 for name in rows["athlete_name"]}


def test_the_scored_seasons_match_the_existing_model():
    import train_model as tm

    assert fm.TEST_YEARS == [y for y in tm.LABEL_YEARS if y >= tm.FIRST_TEST_YEAR]
    assert fm.group_of("women_10000m") == "distance" and fm.group_of("men_HT") == "throws"
