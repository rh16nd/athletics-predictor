"""Tests for the field model's race-by-race features (2026-09-15): form, recent
form, podiums at big meetings, head-to-head against the field's strongest, and
the comparison with today's model, whose rule was written down before it ran.

Synthetic data only. What they guard is what would look like a working model
while being wrong: a race from the championship itself counting, a heat or an
indoor meeting counting as a big podium, one shared final counted twice, a
feature that moves when slower entrants are added, and a comparison that ships
a candidate its own shuffled control can match.
"""
import numpy as np
import pandas as pd
import pytest

import field_data as fd
import field_model as fm

CUTOFF = pd.Timestamp("2015-08-22")


def result(date, score, place="1.", race="F", category="GW", competition="Meeting, Somewhere (KEN)",
           legal=True, mark="1:44.00"):
    return {"date": date, "competition": competition, "category": category, "race": race,
            "place": place, "mark": mark, "notLegal": not legal, "resultScore": score}


def season(*results, discipline="800 Metres"):
    return [{"discipline": discipline, "indoor": None, "results": list(results)}]


# ---- one athlete's season, race by race --------------------------------------

def test_a_race_on_or_after_the_cut_off_counts_for_nothing():
    s = season(result("01 JUL 2015", 1200, competition="Paris"),
               result("22 AUG 2015", 1300, competition="Beijing Worlds", category="OW"),
               result("25 AUG 2015", 1310, competition="Beijing Worlds", category="OW"))
    summary = fd.race_summary(s, "men_800m", CUTOFF)
    assert summary["form_score"] == 1200 and summary["races"] == 1
    assert summary["recent_score"] is None
    assert summary["big_podiums"] == 1
    assert list(summary["finals"]) == [("Paris", pd.Timestamp("2015-07-01"), "F")]


def test_form_is_the_mean_of_the_best_three_so_one_outlying_mark_moves_it_less_than_the_season_best():
    s = season(result("01 JUN 2015", 1180), result("15 JUN 2015", 1170), result("01 JUL 2015", 1175),
               result("20 JUL 2015", 1260))
    summary = fd.race_summary(s, "men_800m", CUTOFF)
    assert summary["form_score"] == pytest.approx((1260 + 1180 + 1175) / 3)
    assert summary["races"] == 4


def test_recent_form_is_the_best_score_in_the_six_weeks_before_the_cut_off():
    s = season(result("01 JUN 2015", 1250), result("15 JUL 2015", 1190), result("08 AUG 2015", 1210))
    assert fd.race_summary(s, "men_800m", CUTOFF)["recent_score"] == 1210


def test_only_a_final_outdoors_at_a_big_meeting_is_a_big_podium_and_a_place_counts_when_the_mark_does_not():
    s = season(
        result("01 JUN 2015", 1200, place="1.", race="H2", category="OW"),   # a heat
        result("05 JUN 2015", 1200, place="2.", race="F", category="A"),     # below the big tiers
        result("10 JUN 2015", 1200, place="3.", race="F1", category="DF"),   # a final run in sections
        result("20 JUN 2015", 0, place="", race="F", category="GW", mark="DNF"),
        result("01 JUL 2015", 1230, place="1.", race="F", category="GL", legal=False),
        result("05 JUL 2015", 1240, place="1.", race="F", category="GW",
               competition="Birmingham Indoor Grand Prix (i)"),
    )
    summary = fd.race_summary(s, "men_800m", CUTOFF)
    assert summary["big_podiums"] == 2   # the DF section and the GL win
    assert summary["races"] == 3         # the marks from the heat, the A final and the DF section


def test_big_podiums_are_capped():
    s = season(*[result(f"{day:02d} JUN 2015", 1200) for day in range(1, 9)])
    assert fd.race_summary(s, "men_800m", CUTOFF)["big_podiums"] == fd.BIG_PODIUM_CAP


def test_a_season_with_nothing_in_the_event_before_the_cut_off_has_no_summary():
    assert fd.race_summary(season(result("01 JUN 2015", 1200), discipline="1500 Metres"), "men_800m", CUTOFF) is None
    assert fd.race_summary(season(result("01 SEP 2015", 1200)), "men_800m", CUTOFF) is None
    assert fd.race_summary(None, "men_800m", CUTOFF) is None


# ---- head-to-head --------------------------------------------------------------

def test_head_to_head_is_against_the_three_strongest_others_and_each_shared_final_counts_once():
    paris = ("Paris", pd.Timestamp("2015-07-04"), "F")
    oslo = ("Oslo", pd.Timestamp("2015-06-11"), "F")
    rome = ("Rome", pd.Timestamp("2015-06-04"), "F")
    finals = {"A": {paris: 1, oslo: 2}, "B": {paris: 2, oslo: 1, rome: 1},
              "C": {paris: 3, rome: 2}, "D": {rome: 3}, "E": {paris: 4}}
    scores = {"A": 1250, "B": 1240, "C": 1230, "D": 1220, "E": 1100}
    h2h = fd.h2h_top(finals, scores)
    assert h2h["A"] == pytest.approx(1 / 3)    # beat B and C in Paris, lost to B in Oslo
    assert h2h["B"] == pytest.approx(3 / 5)
    assert h2h["C"] == pytest.approx(-2 / 4)
    assert h2h["D"] == pytest.approx(-1.0)     # lost to B and C in Rome, never met A
    assert h2h["E"] == pytest.approx(-1.0)     # E's rivals are A, B and C

    more = fd.h2h_top({**finals, "F": {paris: 5}, "G": {oslo: 3}}, {**scores, "F": 1000, "G": 990})
    assert all(more[k] == pytest.approx(h2h[k]) for k in h2h)
    assert fd.h2h_top({"A": {}, "B": {}}, {"A": 1, "B": 2}) == {"A": 0.0, "B": 0.0}


def test_race_columns_come_from_each_finalists_own_season_and_head_to_head_stays_inside_the_final():
    scored = pd.DataFrame({
        "competition": "Worlds", "year": 2015, "discipline": "men_800m", "cutoff": CUTOFF, "tier": "global",
        "athlete_name": ["A", "B", "C"], "athlete_id": [1, 2, None], "sb_score": [1250.0, 1240.0, 1230.0]})
    paris = result("04 JUL 2015", 1230, competition="Paris")
    races = {2015: {"1": season(paris), "2": season({**paris, "place": "2.", "resultScore": 1220})}}
    out = fd.attach_races(scored, races).set_index("athlete_name")
    assert (out.loc["A", "h2h_top"], out.loc["B", "h2h_top"]) == (1.0, -1.0)
    assert (out.loc["A", "big_podiums"], out.loc["B", "big_podiums"]) == (1, 1)
    assert bool(out.loc["A", "has_races"]) and not bool(out.loc["C", "has_races"])
    assert pd.isna(out.loc["C", "form_score"]) and out.loc["C", "h2h_top"] == 0.0


# ---- features ------------------------------------------------------------------

def v2_field(scores, **extra):
    n = len(scores)
    return pd.DataFrame({
        "athlete_name": [f"A{i}" for i in range(n)],
        "sb_score": scores,
        "sb_date": [pd.Timestamp("2023-08-01")] * n,
        "sb_prior_season": extra.get("prior", [0] * n),
        "career_best": extra.get("career", [s + 10 for s in scores]),
        "prev_season_best": [s - 20 for s in scores],
        "dob": [pd.Timestamp("1998-01-01")] * n,
        "form_score": extra.get("form", [s - 15 for s in scores]),
        "recent_score": extra.get("recent", [s - 5 for s in scores]),
        "races": extra.get("races", [6] * n),
        "big_podiums": extra.get("podiums", [0] * n),
        "h2h_top": extra.get("h2h", [0.0] * n),
    })


def test_the_new_features_do_not_move_when_slower_entrants_are_added_beside_a_final():
    """Even when a slower entrant's form beats a leader's: the gap is taken from
    the field's best season score, which slower entrants cannot change."""
    top = [1200, 1190, 1180, 1150, 1140, 1120, 1110, 1100]
    forms = [900] + [s - 15 for s in top[1:]]
    final = fm.field_features(v2_field(top, form=forms), "2023-09-29", fm.FEATURES_V2).to_numpy()
    slower = list(range(1000, 800, -10))
    entry = fm.field_features(v2_field(top + slower, form=forms + slower), "2023-09-29", fm.FEATURES_V2)
    assert np.allclose(final, entry.to_numpy()[:8])
    assert list(entry.columns) == fm.FEATURES_V2


def test_missing_race_data_falls_back_to_the_season_best_with_nothing_recent():
    rows = v2_field([1200, 1150, 1100], form=[None, 1140, 1090], recent=[None, None, 1110],
                    races=[None, 1, 6], podiums=[None, 2, 0], h2h=[None, 0.5, -0.5])
    f = fm.field_features(rows, "2023-09-29", fm.FEATURES_V2)
    assert f["form_gap"].iloc[0] == 0.0
    assert f[["recent_delta", "big_podiums", "h2h_top"]].iloc[0].tolist() == [0.0, 0.0, 0.0]
    assert f["no_recent"].tolist() == [1.0, 1.0, 0.0]
    assert f["recent_delta"].iloc[2] == pytest.approx(0.2)


def test_history_counts_more_in_a_thin_season_and_a_breakout_counts_only_as_far_as_form_backs_it():
    rows = v2_field([1200, 1200, 1200], career=[1260, 1260, 1150], form=[1180, 1180, 1190], races=[2, 7, 7])
    f = fm.field_features(rows, "2023-09-29", fm.FEATURES_V2)
    assert f["pb_gap_thin"].tolist() == pytest.approx([0.6, 0.0, 0.0])
    assert f["breakout_backed"].tolist() == pytest.approx([0.0, 0.0, 0.4])


def test_todays_features_are_unchanged_by_default():
    assert list(fm.field_features(v2_field([1200, 1150, 1100]), "2023-09-29").columns) == fm.FEATURES
    assert fm.FEATURES_V2[:len(fm.FEATURES)] == fm.FEATURES


# ---- the comparison ------------------------------------------------------------

def test_the_top_three_log_likelihood_is_exact():
    u = np.array([2.0, 1.0, 0.0, -1.0])
    e = np.exp(u)
    expected = np.log(e[0] / e.sum()) + np.log(e[1] / (e.sum() - e[0])) + np.log(e[2] / (e.sum() - e[0] - e[1]))
    assert fm.top3_log_likelihood(u, [0, 1, 2]) == pytest.approx(expected)


def test_the_control_shuffles_only_the_new_columns_and_keeps_them_together():
    X = np.arange(40, dtype=float).reshape(10, 4)
    out = fm.shuffle_columns([{"names": list("abcdefghij"), "X": X}], [2, 3], seed=0)[0]["X"]
    assert np.array_equal(out[:, :2], X[:, :2])
    assert not np.array_equal(out[:, 2:], X[:, 2:])
    assert {tuple(r) for r in out[:, 2:]} == {tuple(r) for r in X[:, 2:]}


def results(ll, hits, asia=40):
    n = len(hits)
    return pd.DataFrame({"model_hits": hits, "ll": ll, "tier": ["asia"] * asia + ["global"] * (n - asia),
                         "group": "sprints", "competition": "C", "year": 2023,
                         "discipline": [f"d{i}" for i in range(n)]})


def comparison(n=200):
    rng = np.random.default_rng(1)
    base = results(list(-3.0 + rng.normal(0, 0.2, n)), [2] * n)
    better = results(list(base["ll"] + 0.1), [2] * n)
    controls = [results(list(base["ll"] - 0.02), [2] * n) for _ in range(5)]
    return base, better, controls


def test_the_comparison_ships_a_candidate_that_reads_the_top_three_better_and_names_as_many_medallists():
    base, better, controls = comparison()
    decision = fm.compare_decision(base, better, controls)
    assert decision["ships"] and all(decision["conditions"].values())


def test_fewer_medallists_named_means_nothing_ships():
    base, better, controls = comparison()
    fewer = better.assign(model_hits=[1] * len(better))
    assert not fm.compare_decision(base, fewer, controls)["ships"]


def test_a_candidate_worse_on_the_asian_finals_does_not_ship():
    base, better, controls = comparison()
    worse_in_asia = better.assign(ll=np.where(better["tier"] == "asia", base["ll"] - 0.5, better["ll"]))
    assert not fm.compare_decision(base, worse_in_asia, controls)["ships"]


def test_a_shuffled_control_that_also_passes_means_nothing_ships():
    base, better, controls = comparison()
    assert not fm.compare_decision(base, better, controls[:4] + [better])["ships"]


# ---- serving -------------------------------------------------------------------

def test_an_entrant_is_served_race_by_race_exactly_as_a_finalist_was_trained(tmp_path):
    path = tmp_path / "men_800m_2026.csv"
    path.write_text(
        "Rank,Mark,WIND,Competitor,DOB,,Pos,,Venue,Date,Results Score,discipline,year,ProfileURL\n"
        "1,1:43.00,,Fast RUNNER,01 JAN 2000,KEN,1,,Somewhere,01 JUL 2026,1250,men_800m,2026,"
        "https://worldathletics.org/athletes/athlete=11\n"
        "2,1:44.00,,Slow RUNNER,01 JAN 2000,KEN,1,,Somewhere,02 JUL 2026,1220,men_800m,2026,"
        "https://worldathletics.org/athletes/athlete=22\n", encoding="utf-8")
    zurich = result("28 AUG 2026", 1250, competition="Weltklasse Zürich", category="DF")
    races = {"11": season(zurich), "22": season({**zurich, "place": "2.", "resultScore": 1220})}
    # The first entrant has an id on the entry list; the second takes the toplist's.
    athletes = [{"name": "Fast RUNNER", "nat": "KEN", "waId": 11}, {"name": "Slow RUNNER", "nat": "KEN"}]
    served = fm.serving_rows("men_800m", athletes, str(path), "2026-09-23", 2026,
                             scores_for=lambda key: pd.DataFrame(), races=races)
    assert served["wa_id"].tolist() == [11, 22]

    trained = fd.attach_races(pd.DataFrame({
        "competition": "Asian Games", "year": 2026, "discipline": "men_800m", "cutoff": pd.Timestamp("2026-09-23"),
        "athlete_name": ["Fast RUNNER", "Slow RUNNER"], "athlete_id": [11, 22], "sb_score": [1250.0, 1220.0]}),
        {2026: races})
    columns = fd.RACE_COLUMNS
    assert (served.set_index("athlete_name")[columns].to_dict("index")
            == trained.set_index("athlete_name")[columns].to_dict("index"))
    f = fm.field_features(served, "2026-09-23", fm.FEATURES_V2)
    assert f["h2h_top"].tolist() == [1.0, -1.0] and f["big_podiums"].tolist() == [1.0, 1.0]


def test_a_refresh_fetches_a_season_again_for_a_field_still_competing(tmp_path):
    calls = []

    def fetch(athlete, year):
        calls.append((athlete, year))
        return []

    fd.fetch_seasons({(1, 2026)}, fetch=fetch, seasons_dir=str(tmp_path))
    fd.fetch_seasons({(1, 2026)}, fetch=fetch, seasons_dir=str(tmp_path))
    fd.fetch_seasons({(1, 2026)}, fetch=fetch, seasons_dir=str(tmp_path), refresh=True)
    assert calls == [(1, 2026), (1, 2026)]


# ---- trying candidates on the tuning years, and the locked years -----------------

def test_the_tuning_years_and_the_locked_years_are_apart_and_cover_the_test_years():
    assert not set(fm.DEV_YEARS) & set(fm.HOLDOUT_YEARS)
    assert sorted(fm.DEV_YEARS + fm.HOLDOUT_YEARS) == fm.TEST_YEARS
    assert max(fm.DEV_YEARS) < min(fm.HOLDOUT_YEARS)


def opposing_finals(years, per_year=30, seed=0):
    """Finals of six decided by one feature: the higher value wins in sprints,
    the lower in throws. One pooled model cannot read both."""
    rng = np.random.default_rng(seed)
    finals = []
    for year in years:
        for group in ("sprints", "throws"):
            for i in range(per_year):
                x = rng.normal(size=(6, 1))
                order = np.argsort(-x[:, 0] if group == "sprints" else x[:, 0])
                names = [f"a{j}" for j in range(6)]
                finals.append({"competition": f"C{i}", "year": year, "discipline": f"{group}_{i}", "group": group,
                               "tier": "global", "names": names, "scores": x[:, 0], "X": x, "features": ["x"],
                               "podium": {names[j] for j in order[:3]}, "winners": {names[order[0]]},
                               "top_idx": [int(j) for j in order[:3]], "trainable": True})
    return finals


def test_a_backtest_scores_only_the_years_it_is_given():
    results, _ = fm.backtest(opposing_finals([2019, 2020, 2021, 2022]), years=[2021])
    assert set(results["year"]) == {2021}


def test_a_model_fitted_by_group_reads_events_that_pull_opposite_ways():
    finals = opposing_finals([2019, 2020, 2021])
    pooled, _ = fm.backtest(finals, years=[2021])
    grouped, _ = fm.backtest(finals, years=[2021], by_group=True)
    assert len(grouped) == len(pooled)
    assert grouped["model_hits"].mean() > pooled["model_hits"].mean() + 0.5


def test_how_races_are_read_can_be_varied_for_an_experiment():
    s = season(result("01 JUN 2015", 1250), result("15 JUL 2015", 1190), result("08 AUG 2015", 1210, category="A"))
    assert fd.race_summary(s, "men_800m", CUTOFF, recent_days=20)["recent_score"] == 1210
    assert fd.race_summary(s, "men_800m", CUTOFF, recent_days=90)["recent_score"] == 1250
    assert fd.race_summary(s, "men_800m", CUTOFF, form_marks=1)["form_score"] == 1250
    assert fd.race_summary(s, "men_800m", CUTOFF)["big_podiums"] == 2
    assert fd.race_summary(s, "men_800m", CUTOFF, big_categories=("GW", "A"))["big_podiums"] == 3


def test_an_experiment_that_reads_races_its_own_way_recomputes_the_race_columns():
    scored = pd.DataFrame({
        "competition": "Worlds", "year": 2015, "discipline": "men_800m", "cutoff": CUTOFF, "tier": "global",
        "athlete_name": ["A", "B"], "athlete_id": [1, 2], "sb_score": [1250.0, 1240.0]})
    races = {2015: {"1": season(result("01 JUN 2015", 1250)), "2": season(result("02 JUN 2015", 1240))}}
    usual = fd.attach_races(scored, races)
    varied = fm.spec_scored(usual, {"features": fm.FEATURES_V2, "race": {"recent_days": 90}}, races)
    assert pd.isna(usual.loc[0, "recent_score"]) and varied.loc[0, "recent_score"] == 1250
    assert fm.spec_scored(usual, {"features": fm.FEATURES_V2}, races) is usual


def test_every_experiment_uses_known_features_and_known_ways_of_reading_races():
    assert fm.EXPERIMENTS["today"]["features"] == fm.FEATURES
    for name, spec in fm.EXPERIMENTS.items():
        assert set(spec["features"]) <= set(fm.FEATURES_V2), name
        assert set(spec.get("race", {})) <= {"recent_days", "form_marks", "big_categories"}, name


def test_the_candidate_for_the_locked_years_gains_on_the_tuning_years_without_naming_fewer_medallists():
    rows = [{"name": "today", "meanLlGain": 0.0, "meanHitsDiff": 0.0},
            {"name": "biggest_gain_fewer_medallists", "meanLlGain": 0.09, "meanHitsDiff": -0.01},
            {"name": "gains", "meanLlGain": 0.05, "meanHitsDiff": 0.0},
            {"name": "gains_less", "meanLlGain": 0.02, "meanHitsDiff": 0.04}]
    assert fm.choose_experiment(rows) == "gains"
    assert fm.choose_experiment([{"name": "worse", "meanLlGain": -0.01, "meanHitsDiff": 0.1}]) is None


def test_the_locked_years_are_scored_once(tmp_path, monkeypatch):
    used = tmp_path / "holdout.json"
    used.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(fm, "load_scored_finals", lambda: pytest.fail("the locked years were read a second time"))
    assert fm.run_holdout("v2", path=str(used)) == 1
