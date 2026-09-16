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
    assert summary["form_spread"] == 1260 - 1175
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
        "two_seasons_ago_best": extra.get("two", [None] * n),
        "older_seasons_best": extra.get("older", [None] * n),
        "older_seasons_best_year": extra.get("older_year", [None] * n),
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
        assert set(spec["features"]) <= set(fm.ALL_FEATURES), name
        assert set(spec.get("race", {})) <= {"recent_days", "form_marks", "big_categories"}, name


def test_the_candidate_for_the_locked_years_is_the_most_accurate_that_names_no_fewer_medallists():
    """Accuracy alone decides (the user, 2026-09-15): a candidate built on the
    new-over-old rule wins only by also being the most accurate."""
    rows = [{"name": "today", "meanLlGain": 0.0, "meanHitsDiff": 0.0},
            {"name": "recency_by_group", "meanLlGain": 0.30, "meanHitsDiff": -0.01},   # names fewer medallists
            {"name": "v2", "meanLlGain": 0.20, "meanHitsDiff": 0.10},
            {"name": "recency", "meanLlGain": 0.05, "meanHitsDiff": 0.0}]
    assert fm.choose_experiment(rows) == "v2"
    assert fm.choose_experiment([rows[0], {"name": "recency", "meanLlGain": -0.01, "meanHitsDiff": 0.1}]) is None


# ---- the user's rule: new marks over old ------------------------------------------

def test_old_marks_never_outweigh_new_ones_and_a_consistent_season_is_judged_on_itself():
    rows = pd.DataFrame({
        "sb_score": [1200.0] * 5,
        "form_score": [1180.0, 1180.0, 1180.0, 1250.0, 1180.0],
        "form_spread": [10.0, 10.0, 80.0, 10.0, 10.0],
        "races": [5, 1, 5, 5, 5],
        "career_best": [1300.0, 1300.0, 1300.0, 1210.0, None],
    })
    value, consistency, weight = fm.strength(rows)
    # Five marks ten points apart: consistency 0.8, and the new marks weigh 0.9.
    assert consistency[0] == pytest.approx(0.8) and weight[0] == pytest.approx(0.9)
    assert value[0] == pytest.approx(0.9 * 1180 + 0.1 * 1300)
    # One mark: consistency 0.16, and the new marks still weigh more than half.
    assert weight[1] == pytest.approx(0.58)
    # Best marks eighty points apart: the old marks count, but only as much as the new.
    assert weight[2] == pytest.approx(0.5) and value[2] == pytest.approx((1180 + 1300) / 2)
    # Better now than before, or no past at all: read on the new marks alone.
    assert value[3] == 1250.0 and value[4] == 1180.0
    assert (weight >= 0.5).all()


def test_the_rule_holds_the_signs_the_fit_is_not_allowed_to_cross():
    """Being read on last season's mark predicts a podium in these finals, and a
    free fit rewards it; held by the rule, its weight cannot rise above zero."""
    sprints = [f for f in opposing_finals([2019, 2020], per_year=40) if f["group"] == "sprints"]
    tempting = [dict(f, X=np.hstack([f["X"], (f["X"] > 0).astype(float)]), features=["x", "sb_prior_season"])
                for f in sprints]
    assert fm.fit(tempting)["weights"][1] > 0
    assert fm.fit(tempting, bounds=fm.RECENCY_BOUNDS)["weights"][1] <= 1e-9


def test_the_recency_features_do_not_move_when_weaker_entrants_are_added():
    top = [1200, 1190, 1180, 1150, 1140, 1120, 1110, 1100]
    final = fm.field_features(v2_field(top), "2023-09-29", fm.RECENCY_FEATURES).to_numpy()
    entry = fm.field_features(v2_field(top + list(range(1000, 800, -10))), "2023-09-29", fm.RECENCY_FEATURES)
    assert np.allclose(final, entry.to_numpy()[:8])
    assert list(entry.columns) == fm.RECENCY_FEATURES


def test_the_experiments_marked_new_over_old_are_built_on_the_rule():
    history_first = {"pb_gap", "yoy", "pb_gap_thin", "breakout_backed"}
    marked = {name: spec for name, spec in fm.EXPERIMENTS.items() if spec.get("newOverOld")}
    assert "recency" in marked
    for name, spec in marked.items():
        assert not history_first & set(spec["features"]), name
        assert spec["bounds"] == fm.RECENCY_BOUNDS, name
    assert not any(fm.EXPERIMENTS[name].get("newOverOld") for name in ("today", "v2"))


def test_the_three_versions_tested_on_every_season_differ_only_in_how_old_marks_count():
    """The user's choice on 2026-09-16: today's model, recent over old and this
    season only, each reading races the same way as one model for every event."""
    assert fm.ALL_SEASON_CANDIDATES[0] == "v2_form_best_5"
    today, recent, season_only = (fm.EXPERIMENTS[name] for name in fm.ALL_SEASON_CANDIDATES)
    for spec in (today, recent, season_only):
        assert spec["race"] == {"form_marks": 5} and not spec.get("by_group") and "l2" not in spec
    assert not set(fm.OLD_MARK_FEATURES) & set(season_only["features"])
    assert set(season_only["features"]) | set(fm.OLD_MARK_FEATURES) == set(fm.FEATURES_V2)
    assert "sb_prior_season" in season_only["features"]   # last season only when there is no mark this season
    assert recent.get("newOverOld") and recent["bounds"] == fm.RECENCY_BOUNDS
    assert fm.ALL_SEASON_YEARS[0] == 2012 and fm.ALL_SEASON_YEARS[-1] == 2026 and 2020 not in fm.ALL_SEASON_YEARS


def test_the_version_kept_names_the_most_medallists_then_the_most_winners_then_reads_the_order_best():
    rows = [{"name": "v2_form_best_5", "medallists": 2400, "winners": 700, "meanLlGain": 0.0},
            {"name": "recency_form_best_5", "medallists": 2390, "winners": 720, "meanLlGain": 0.05},
            {"name": "season_only", "medallists": 2400, "winners": 690, "meanLlGain": 0.2}]
    assert fm.choose_all_seasons(rows) == "v2_form_best_5"
    rows[2]["winners"] = 700
    assert fm.choose_all_seasons(rows) == "season_only"


def test_every_version_is_recorded_on_the_same_finals_with_the_one_kept(tmp_path, monkeypatch):
    import json

    def block(hits):
        return {"finals": 10, "hits": hits, "possible": 30, "model": round(100 * hits / 30, 1), "points": 60.0,
                "modelWinners": 6, "pointsWinners": 5}

    def fake_compare(scored, candidate, baseline, years, controls, races):
        assert (baseline, controls) == ("v2_form_best_5", 0)
        pair = {"baseline": block(20), "candidate": block({"recency_form_best_5": 19, "season_only": 21}[candidate])}
        return {"overall": pair, "byTier": {"asia": pair}, "byYear": {2025: pair},
                "decision": {"meanLlGain": 0.1}}, None

    monkeypatch.setattr(fm, "_scored_with_races", lambda: pd.DataFrame())
    monkeypatch.setattr(fm, "compare", fake_compare)
    monkeypatch.setattr(fd, "load_seasons", lambda *args, **kwargs: {})
    path = tmp_path / "all_seasons.json"
    assert fm.run_all_seasons(years=[2025], path=str(path)) == 0
    report = json.loads(path.read_text(encoding="utf-8"))
    assert [r["name"] for r in report["candidates"]] == fm.ALL_SEASON_CANDIDATES
    assert [r["medallists"] for r in report["candidates"]] == [20, 19, 21]
    assert report["candidates"][0]["meanLlGain"] == 0.0 and report["candidates"][2]["bySeason"] == {"2025": 70.0}
    assert (report["chosen"], report["years"]) == ("season_only", [2025])


def test_a_refit_retrains_the_served_experiment_and_keeps_its_test_record_and_the_model_it_replaces(
        tmp_path, monkeypatch):
    import json

    served, previous = tmp_path / "field_model.json", tmp_path / "field_model_previous.json"
    served.write_text(json.dumps({"experiment": "today", "holdout": {"ships": True}}), encoding="utf-8")
    sprints = [f for f in opposing_finals([2019, 2020]) if f["group"] == "sprints"]
    monkeypatch.setattr(fm, "_scored_with_races", lambda: pd.DataFrame())
    monkeypatch.setattr(fm, "build_finals", lambda scored, features, fade=None: sprints)

    assert fm.run_refit(model_path=str(served), previous_path=str(previous)) == 0
    model = json.loads(served.read_text(encoding="utf-8"))
    assert (model["experiment"], model["holdout"], model["finals"]) == ("today", {"ships": True}, len(sprints))
    assert json.loads(previous.read_text(encoding="utf-8")) == {"experiment": "today", "holdout": {"ships": True}}
    assert fm.run_refit("no_such_experiment", model_path=str(served), previous_path=str(previous)) == 1


def test_the_locked_years_are_scored_once(tmp_path, monkeypatch):
    used = tmp_path / "holdout.json"
    used.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(fm, "load_scored_finals", lambda: pytest.fail("the locked years were read a second time"))
    assert fm.run_holdout("v2", path=str(used)) == 1


# ---- the user's rule for old marks: this season first, and old marks fade ---------

def levels_field(**columns):
    """A leader on 1300 and one athlete on 1200 this season, with whatever
    earlier bests the case needs. The gap to the leader is wide, so the cap on
    how far an old mark may lift an athlete does not come into these numbers."""
    rows = {"sb_score": [1300.0, 1200.0], "form_score": [1300.0, 1200.0], "races": [5, 5],
            "form_spread": [0.0, 0.0], "prev_season_best": [None, None], "two_seasons_ago_best": [None, None],
            "older_seasons_best": [None, None], "older_seasons_best_year": [None, None]}
    rows.update(columns)
    return pd.DataFrame(rows)


def test_this_season_counts_in_full_and_an_older_best_counts_for_less_the_older_it_is():
    """The user's rule (2026-09-16), at the settings the grid chose: the same
    1240 best lifts an athlete on 1200 by 14 points from last season, 4.2 from
    two seasons ago and 0.38 from four."""
    last, parts = fm.faded_level(levels_field(prev_season_best=[None, 1240.0]), "2026-09-23")
    assert last.iloc[1] == pytest.approx(1214.0)
    assert (parts["fromSeason"].iloc[1], parts["weight"].iloc[1]) == (2025, pytest.approx(0.35))
    two, _ = fm.faded_level(levels_field(two_seasons_ago_best=[None, 1240.0]), "2026-09-23")
    assert two.iloc[1] == pytest.approx(1204.2)
    four, parts = fm.faded_level(levels_field(older_seasons_best=[None, 1240.0],
                                              older_seasons_best_year=[None, 2022]), "2026-09-23")
    assert four.iloc[1] == pytest.approx(1200.378)
    assert parts["fromSeason"].iloc[1] == 2022
    # Nothing older stands above this season: read on this season alone.
    below, parts = fm.faded_level(levels_field(prev_season_best=[None, 1150.0],
                                              two_seasons_ago_best=[None, 1100.0]), "2026-09-23")
    assert below.iloc[1] == pytest.approx(1200.0) and pd.isna(parts["fromSeason"].iloc[1])
    # Old seasons do not stack: the largest lift counts, not the sum of them.
    stacked, parts = fm.faded_level(levels_field(prev_season_best=[None, 1240.0],
                                                two_seasons_ago_best=[None, 1400.0]), "2026-09-23")
    assert stacked.iloc[1] == pytest.approx(1221.0) and parts["fromSeason"].iloc[1] == 2024


def test_an_old_mark_can_close_a_gap_but_never_pass_the_leader_on_this_seasons_marks():
    """The user's case, and the reason the rule is structure and not a weight: an
    athlete breaking records this season is not beaten by a mark from two years
    ago, whatever that mark was. The served model reverses this pair once the old
    best passes about 1280 (HANDOFF.md)."""
    season = [1250.0, 1230.0, 1220.0, 1210.0, 1200.0, 1190.0]
    for old_best in (1260.0, 1300.0, 1330.0, 1500.0):
        rows = v2_field(season, form=season, two=[1200.0, old_best] + [None] * 4)
        level, parts = fm.faded_level(rows, "2026-09-23")
        assert level.iloc[0] > level.iloc[1], old_best
        # The lift stops at three quarters of the 20 points between them.
        assert parts["lift"].iloc[1] <= 15.0 + 1e-9
        f = fm.field_features(rows, "2026-09-23", fm.FADE_FEATURES)
        assert f["fade_gap_best"].iloc[0] == 0.0 and f["fade_gap_best"].iloc[1] < 0.0

    # An athlete with one race and a big old mark: what strength() read the other
    # way round, since it capped old marks at half but never by their age.
    thin = v2_field(season, form=season, races=[5, 1] + [5] * 4, two=[1200.0, 1330.0] + [None] * 4,
                    career=[1250.0, 1330.0] + [s + 10 for s in season[2:]])
    faded = fm.faded_level(thin, "2026-09-23")[0]
    assert faded.iloc[0] > faded.iloc[1] and faded.iloc[1] == pytest.approx(1240.5)
    strength = fm.strength(thin)[0]
    assert strength.iloc[1] > strength.iloc[0] and strength.iloc[1] == pytest.approx(1270.0)


def test_the_old_marks_features_do_not_move_when_weaker_entrants_are_added():
    top = [1200, 1190, 1180, 1150, 1140, 1120, 1110, 1100]
    old = [s + 100 for s in top]
    final = fm.field_features(v2_field(top, two=old), "2023-09-29", fm.FADE_FEATURES).to_numpy()
    slower = list(range(1000, 800, -10))
    entry = fm.field_features(v2_field(top + slower, two=old + [None] * len(slower)),
                              "2023-09-29", fm.FADE_FEATURES)
    assert np.allclose(final, entry.to_numpy()[:8])
    assert list(entry.columns) == fm.FADE_FEATURES


def test_the_old_marks_experiment_is_built_on_the_rule_and_not_on_the_fit():
    spec = fm.EXPERIMENTS["old_marks"]
    history_first = {"pb_gap", "yoy", "pb_gap_thin", "breakout_backed"}
    assert not history_first & set(spec["features"])
    assert spec["bounds"] == fm.FADE_BOUNDS and spec.get("oldMarks")
    assert set(spec["fade"]) == {"last_season", "per_year", "cap_share"}
    # The signs the rule needs, and every one of them on a feature it reads.
    assert fm.FADE_BOUNDS["fade_gap_best"] == (0.0, None) and fm.FADE_BOUNDS["sb_prior_season"] == (None, 0.0)
    assert set(fm.FADE_BOUNDS) <= set(fm.FADE_FEATURES)
    assert set(fm.FADE_FEATURES) <= set(fm.ALL_FEATURES)
    assert fm.fade_weight(1) > fm.fade_weight(2) > fm.fade_weight(5) and fm.fade_weight(1) < 1.0


def test_the_old_marks_rule_holds_the_sign_the_fit_is_not_allowed_to_cross():
    """As RECENCY_BOUNDS does: a free fit rewards being read on last season's
    mark, and the rule will not let that weight rise above zero."""
    sprints = [f for f in opposing_finals([2019, 2020], per_year=40) if f["group"] == "sprints"]
    tempting = [dict(f, X=np.hstack([f["X"], (f["X"] > 0).astype(float)]), features=["x", "sb_prior_season"])
                for f in sprints]
    assert fm.fit(tempting)["weights"][1] > 0
    assert fm.fit(tempting, bounds=fm.FADE_BOUNDS)["weights"][1] <= 1e-9


def test_the_call_fades_an_old_mark_exactly_as_the_backtest_did():
    """The settings travel with the model, so serving cannot read an old mark
    one way while the backtest read it another."""
    assert fm.fade_of({"experiment": "old_marks"}) == fm.EXPERIMENTS["old_marks"]["fade"]
    assert fm.fade_of({"experiment": "v2_form_best_5"}) is None and fm.fade_of({}) is None
    season = [1250.0, 1230.0, 1220.0, 1210.0, 1200.0, 1190.0]
    rows = v2_field(season, form=season, two=[1200.0, 1300.0] + [None] * 4)
    default = fm.field_features(rows, "2026-09-23", fm.FADE_FEATURES)
    nothing = fm.field_features(rows, "2026-09-23", fm.FADE_FEATURES,
                               {"last_season": 0.0, "per_year": 0.5, "cap_share": 0.75})
    assert default["fade_gap_best"].iloc[1] > nothing["fade_gap_best"].iloc[1]
    assert nothing["fade_gap_best"].iloc[1] == pytest.approx(-0.20)


def test_the_rule_test_passes_with_the_settings_the_family_ships_with():
    """Both halves of what the user asked for, on every made-up case: the leader
    on this season's marks is never passed, and a big recent-enough old mark
    still counts. Checked under every fit FADE_BOUNDS allows, sampled."""
    report = fm.check_old_marks_rule(fm.EXPERIMENTS["old_marks"]["fade"])
    assert report["passed"] and report["fits"] == fm.RULE_WEIGHT_SETS
    assert all(c["leaderStaysAhead"] for c in report["cases"])
    counted = [c for c in report["cases"] if c["oldMarkCounts"] is not None]
    assert len(counted) >= 4 and all(c["oldMarkCounts"] for c in counted)
    # The lift fades with age: the same 1330 mark is worth less the older it is.
    by_age = {c["ago"]: c["lift"] for c in report["cases"] if c["oldBest"] == 1330.0 and c["races"] == 5}
    assert by_age[3] > by_age[5] > by_age[8] and by_age[8] < 1.0


def test_a_setting_that_would_let_an_old_mark_win_fails_the_rule_test():
    """The gate, not a description: a cap loose enough to carry an athlete past
    the leader, or a weight above 1, cannot be kept however accurate it is."""
    loose = fm.check_old_marks_rule({"last_season": 0.35, "per_year": 0.5, "cap_share": 5.0})
    assert not loose["passed"]
    assert [c["label"] for c in loose["cases"] if not c["leaderStaysAhead"]]
    assert not fm.check_old_marks_rule({"last_season": 1.2, "per_year": 0.9, "cap_share": 5.0})["passed"]


def test_a_setting_that_ignores_old_marks_fails_the_rule_test():
    """The other half of the rule. Old performances still count, so a setting
    that never lets them count is not the rule either, however safe it looks."""
    ignored = fm.check_old_marks_rule({"last_season": 0.0})
    assert not ignored["passed"]
    assert all(c["lift"] == 0.0 for c in ignored["cases"])
    assert all(c["oldMarkCounts"] is False for c in ignored["cases"] if c["oldMarkCounts"] is not None)


def test_the_rule_test_catches_the_reading_the_user_objected_to():
    """A model that rewards a career best far above the season, which is what
    put a 1230 athlete with a 1330 from two years ago ahead of a 1250 personal
    best (HANDOFF.md), is caught by the same gate."""
    weights = [1.0 if name == "pb_gap" else 0.0 for name in fm.FEATURES_V2]
    old_way = {"features": fm.FEATURES_V2, "mean": [0.0] * len(weights), "std": [1.0] * len(weights),
               "weights": weights}
    report = fm.check_old_marks_rule(model=old_way)
    assert not report["passed"] and report["fitsAre"] == "model given"
    beaten = [c["label"] for c in report["cases"] if not c["leaderStaysAhead"]]
    assert "1330 two seasons ago" in beaten


def test_the_made_up_final_differs_only_in_the_old_mark():
    """What lets the test blame the old mark: A, B and D share every other
    number, so nothing else can explain a change of order."""
    rows = fm.rule_field(1330.0)
    same = ["sb_date", "sb_prior_season", "dob", "big_podiums", "h2h_top", "form_spread", "races"]
    assert all(rows[col].nunique() == 1 for col in same)
    assert (rows["form_score"] == rows["sb_score"]).all() and (rows["recent_score"] == rows["sb_score"]).all()
    assert rows["sb_score"].iloc[fm.RULE_A] == max(rows["sb_score"])
    assert rows["two_seasons_ago_best"].iloc[fm.RULE_B] == 1330.0
    # The rival is behind this season and the third athlete sits between them.
    assert rows["sb_score"].iloc[fm.RULE_B] < rows["sb_score"].iloc[fm.RULE_D] < rows["sb_score"].iloc[fm.RULE_A]
    # A thin season changes the race count and nothing else about the field.
    thin = fm.rule_field(1330.0, races=1)
    assert thin["races"].iloc[fm.RULE_B] == 1
    assert thin.drop(columns="races").equals(rows.drop(columns="races"))


def test_the_reason_says_which_old_mark_counted_and_how_much_of_it():
    """The line the page shows, as fields rather than a sentence, so it can be
    said in English and in French. The share is the one that survived the cap,
    so the page never claims more of an old mark than the model used."""
    rows = fm.rule_field(1330.0)
    reasons = fm.old_mark_reasons(rows, fm.RULE_CUTOFF, {"features": fm.FADE_FEATURES,
                                                         "experiment": "old_marks"})
    assert reasons[fm.RULE_A] == {"personalBest": True, "fromSeason": None, "percent": 0}
    # A best from two seasons back counts at 10.5% of the 100 points between it
    # and this season, under the cap, and the reason says 10.
    assert reasons[fm.RULE_B] == {"personalBest": False, "fromSeason": 2024, "percent": 10}
    assert reasons[fm.RULE_D]["fromSeason"] is None
    # A model that does not read old marks this way has no such reason to give.
    assert fm.old_mark_reasons(rows, fm.RULE_CUTOFF,
                               {"features": fm.FEATURES_V2, "experiment": "v2_form_best_5"}) is None


def test_the_reason_is_in_the_order_of_the_rows_it_was_asked_about():
    """The call sorts its rows by chance before writing them out, so a reason
    read back by position has to follow that order, not the order they arrived."""
    rows = fm.rule_field(1330.0)
    flipped = rows.iloc[::-1]
    forward = fm.old_mark_reasons(rows, fm.RULE_CUTOFF, {"features": fm.FADE_FEATURES,
                                                         "experiment": "old_marks"})
    backward = fm.old_mark_reasons(flipped, fm.RULE_CUTOFF, {"features": fm.FADE_FEATURES,
                                                             "experiment": "old_marks"})
    assert backward == forward[::-1]
