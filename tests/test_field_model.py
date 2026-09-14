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
    assert low["gap_best"].iloc[0] == 0 and low["rank_frac"].iloc[0] == 0


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


def test_a_mark_on_or_after_the_cut_off_does_not_count_and_last_season_stands_in():
    h = history(("NAM", "KOR", 2023, 1150, "2023-09-29"), ("NAM", "KOR", 2022, 1120, "2022-07-01"))
    assert fd.score_as_of(h, 2023, pd.Timestamp("2023-09-29")) == (1120.0, pd.Timestamp("2022-07-01"), 1)
    assert fd.score_as_of(h, 2023, pd.Timestamp("2023-09-30"))[0] == 1150.0


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


def test_a_past_asian_list_that_is_really_the_world_list_is_refused():
    with pytest.raises(ValueError, match="area filter"):
        fd.asia_toplist("men_100m", 2019, fetch=lambda url: toplist_page(["USA", "JAM", "KEN", "JPN"]), pause=0)
    header, rows = fd.asia_toplist("men_100m", 2019, fetch=lambda url: toplist_page(["JPN", "CHN", "QAT"]), pause=0)
    assert len(rows) == 3 and header[-3:] == ["discipline", "year", "ProfileURL"]


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
    assert final["top_idx"] is None


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
    empty = lambda key: fd.season_scores(key, raw_dir=str(tmp_path), asia_dir=str(tmp_path))  # noqa: E731
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
