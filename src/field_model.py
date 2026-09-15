"""
field_model.py -- podium chances for any field, each athlete rated against the
others in the same final.

WHY
The model in train_model.py answers "who podiums at a Diamond League-calibre
final, given the world top 100". Asked about an Asian Games field it scores most
entrants against the world list and against Diamond League activity they never
had, so their chances come out under 1%, and the Asian Games call sent 28 of 36
events to a plain points ranking (2026-09-14). This model never looks outside the
field it is asked about.

FEATURES (field_features; the same function trains and serves)
From each athlete's season-best Results Score before the competition, measured
against the rest of the field: gap to the best and gap to the third best, which
stay the same however many slower athletes are added. From their own history:
career best before this season, change on last season, how long ago the season
best was set, age. Plus whether they had no mark this season before the
competition, so that last season's stands in, and whether they have any
history at all. No Diamond League meetings, no head-to-head, no world rank.

MODEL
Plackett-Luce: a softmax over the field on a linear score of those features,
fitted by maximum likelihood on the order of each final's top three. A podium
chance is the exact probability of finishing in the top three under that model,
so a field's chances add up to three podium places.

THE SHIP RULE (fixed before any backtest was run, user decision 2026-09-14)
The model is used for an event group only where it beats simply taking the top
three by season-best points, on finals it was not trained on (walk-forward,
scored 2021-2025). Per group, with d = model hits minus points hits per held-out
final: the one-sided 90% bootstrap lower bound of the mean d must be above 0,
the mean d on the Asian finals alone must be at least 0, and the same model
trained on features shuffled within each field must NOT pass. A group that
fails stays on points.

Usage:
    python src/field_model.py --backtest    # report + ship decisions, fits and saves the model
    python src/field_model.py --compare     # today's features against race by race (compare_decision)
Reads data/field/finals.csv (field_data.py --report). --backtest writes
outputs/field_model.json and outputs/field_model_report.json; --compare writes
outputs/field_model_compare.json and outputs/field_model_v2.json, and leaves
the served model alone.
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import field_data as fd  # noqa: E402

BASE_DIR = fd.BASE_DIR
MODEL_PATH = os.path.join(BASE_DIR, "outputs", "field_model.json")
REPORT_PATH = os.path.join(BASE_DIR, "outputs", "field_model_report.json")
COMPARE_PATH = os.path.join(BASE_DIR, "outputs", "field_model_compare.json")
V2_MODEL_PATH = os.path.join(BASE_DIR, "outputs", "field_model_v2.json")

# The same scored seasons as train_model.FIRST_TEST_YEAR onwards, so the two
# models' numbers cover the same years (tests/test_field_model.py pins it).
TEST_YEARS = [2021, 2022, 2023, 2024, 2025]
# A final is used only when this many of its finalists have a season score.
# Fewer, and "rank within the field" describes a handful of names, not a field.
MIN_SCORED = 5
L2 = 0.01
BOOTSTRAP = 10_000
LOWER_QUANTILE = 0.10

GROUPS = {
    "sprints": {"100m", "200m", "400m", "110h", "100h", "400h"},
    "distance": {"800m", "1500m", "5000m", "10000m", "3000sc"},
    "jumps": {"HJ", "PV", "LJ", "TJ"},
    "throws": {"SP", "DT", "JT", "HT"},
}

# No feature depends on how many weaker athletes share the field. The model
# learns on finals but is asked about entry lists (the Asian Games) and a world
# top 20 (Track/Field), both longer than a final, and a z-score or a rank divided
# by field size would read the same athlete differently in each. Both were in
# the first version and were taken out on 2026-09-14, before any backtest ran.
FEATURES = ["gap_best", "gap_third", "pb_gap", "no_history", "sb_months", "yoy", "age", "sb_prior_season"]
# Race by race, from field_data.race_summary and h2h_top (2026-09-15). The user
# asked for a call that moves after big meetings, rates a clear leader clearly
# and leans on history when a season is thin. compare() measures these against
# FEATURES, under a rule fixed before it first ran.
NEW_FEATURES = ["form_gap", "recent_delta", "no_recent", "big_podiums", "h2h_top", "pb_gap_thin",
                "breakout_backed"]
FEATURES_V2 = FEATURES + NEW_FEATURES
# A season of this many marks or fewer is thin.
THIN_SEASON = 2
CONTROLS = 5


def group_of(disc_key):
    event = str(disc_key).split("_", 1)[-1]
    return next((name for name, events in GROUPS.items() if event in events), None)


# ---- features -----------------------------------------------------------------

def field_features(rows, cutoff, features=FEATURES):
    """`features` for every row of one field that has a season score: today's
    FEATURES unless asked for FEATURES_V2.

    `rows` needs sb_score, sb_date, sb_prior_season, career_best,
    prev_season_best and dob, and the race-by-race features read
    field_data.RACE_COLUMNS as well. Scores are divided by 100 so a coefficient
    reads per 100 World Athletics points. An athlete with no age takes the
    field's median, and 25 when nobody in the field has one.

    The race-by-race features (2026-09-15), none of which moves when slower
    athletes join the field:
      form_gap         form, the mean of the best three scores, below the field's
                       best season score; the season best stands in for an
                       athlete with no races on file
      recent_delta     the best score of the last six weeks against form, 0 with none
      no_recent        no score in those six weeks
      big_podiums      podiums in finals at big meetings this season
      h2h_top          net record against the field's three strongest others
      pb_gap_thin      pb_gap in a season of THIN_SEASON marks or fewer, else 0:
                       history counts for more when the season says little
      breakout_backed  how far form sits above the career best: a breakout the
                       races back up, where one big mark barely moves form"""
    df = rows[pd.notna(rows["sb_score"])].copy()
    cutoff = pd.Timestamp(cutoff)
    s = df["sb_score"].astype(float)
    n = len(df)
    ordered = s.sort_values(ascending=False).to_numpy()
    third = ordered[2] if n >= 3 else (ordered[-1] if n else 0.0)

    out = pd.DataFrame(index=df.index)
    out["gap_best"] = (s - (ordered[0] if n else 0.0)) / 100.0
    out["gap_third"] = (s - third) / 100.0
    career = pd.to_numeric(df["career_best"], errors="coerce")
    out["pb_gap"] = ((career - s) / 100.0).fillna(0.0)
    out["no_history"] = career.isna().astype(float)
    sb_dates = pd.to_datetime(df["sb_date"], errors="coerce")
    out["sb_months"] = ((cutoff - sb_dates).dt.days / 30.44).fillna(12.0)
    prior = pd.to_numeric(df["sb_prior_season"], errors="coerce").fillna(0.0)
    prev = pd.to_numeric(df["prev_season_best"], errors="coerce")
    out["yoy"] = ((s - prev) / 100.0).where(prior == 0).fillna(0.0)
    ages = (cutoff - pd.to_datetime(df["dob"], errors="coerce")).dt.days / 365.25
    out["age"] = ages.fillna(ages.median() if ages.notna().any() else 25.0)
    out["sb_prior_season"] = prior
    if any(name in NEW_FEATURES for name in features):
        def column(name):
            if name not in df.columns:
                return pd.Series(np.nan, index=df.index)
            return pd.to_numeric(df[name], errors="coerce")

        form = column("form_score").fillna(s)
        recent = column("recent_score")
        out["form_gap"] = (form - (ordered[0] if n else 0.0)) / 100.0
        out["recent_delta"] = ((recent - form) / 100.0).fillna(0.0)
        out["no_recent"] = recent.isna().astype(float)
        out["big_podiums"] = column("big_podiums").fillna(0.0)
        out["h2h_top"] = column("h2h_top").fillna(0.0)
        out["pb_gap_thin"] = out["pb_gap"] * (column("races").fillna(0.0) <= THIN_SEASON)
        out["breakout_backed"] = ((form - career).clip(lower=0) / 100.0).fillna(0.0)
    return out[list(features)].astype(float)


# ---- the model ----------------------------------------------------------------

def podium_chances(utilities):
    """P(top three) for each athlete under Plackett-Luce, exactly.

    First: e_i / S. Second: sum over who was first. Third: sum over who was
    first and second. Adds up to 3 over any field of four or more; everyone in a
    field of three or fewer is certain of a place."""
    u = np.asarray(utilities, dtype=float)
    n = len(u)
    if n <= 3:
        return np.ones(n)
    e = np.exp(u - u.max())
    total = e.sum()
    p1 = e / total
    first_second = p1[:, None] * e[None, :] / (total - e)[:, None]
    np.fill_diagonal(first_second, 0.0)
    p2 = first_second.sum(axis=0)
    remaining = total - e[:, None] - e[None, :]
    third = first_second[:, :, None] * e[None, None, :] / remaining[:, :, None]
    idx = np.arange(n)
    third[idx, :, idx] = 0.0   # third is not the one who was first
    third[:, idx, idx] = 0.0   # nor the one who was second
    p3 = third.sum(axis=(0, 1))
    return p1 + p2 + p3


def _pack(finals, mean, std):
    width = max(len(f["names"]) for f in finals)
    X = np.zeros((len(finals), width, finals[0]["X"].shape[1]))
    mask = np.zeros((len(finals), width), dtype=bool)
    top = np.zeros((len(finals), 3), dtype=int)
    for i, f in enumerate(finals):
        n = len(f["names"])
        X[i, :n] = (f["X"] - mean) / std
        mask[i, :n] = True
        top[i] = f["top_idx"]
    return X, mask, top


def _loss_and_grad(w, X, mask, top, l2):
    """Negative log-likelihood of each final's observed top three, and its
    gradient. Unavailable slots (padding, and athletes already placed) are
    -inf in the softmax."""
    finals = X.shape[0]
    rows = np.arange(finals)
    u = np.einsum("fnk,k->fn", X, w)
    available = mask.copy()
    loss, grad = 0.0, np.zeros_like(w)
    for step in range(3):
        picked = top[:, step]
        logits = np.where(available, u, -np.inf)
        lse = logsumexp(logits, axis=1)
        probs = np.exp(logits - lse[:, None])
        loss += float(np.sum(lse - u[rows, picked]))
        grad += np.einsum("fn,fnk->k", probs, X) - X[rows, picked].sum(axis=0)
        available[rows, picked] = False
    loss = loss / finals + l2 * float(w @ w)
    grad = grad / finals + 2 * l2 * w
    return loss, grad


def fit(finals, l2=L2):
    """Fit on every trainable final whose top three all have a season score."""
    usable = [f for f in finals if f["top_idx"] is not None and f.get("trainable", True)]
    if not usable:
        raise ValueError("no trainable final with a scored top three to fit on")
    stacked = np.vstack([f["X"] for f in usable])
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0)
    std[std == 0] = 1.0
    X, mask, top = _pack(usable, mean, std)
    result = minimize(_loss_and_grad, np.zeros(stacked.shape[1]), args=(X, mask, top, l2),
                      jac=True, method="L-BFGS-B")
    return {"features": list(usable[0].get("features", FEATURES)), "mean": mean.tolist(), "std": std.tolist(),
            "weights": result.x.tolist(), "l2": l2, "finals": len(usable),
            "converged": bool(result.success)}


def utilities(model, X):
    """Each athlete's score under the model for one field's feature matrix: the
    log of their strength in Plackett-Luce, comparable only within the field."""
    mean, std, w = (np.asarray(model[k], dtype=float) for k in ("mean", "std", "weights"))
    return ((np.asarray(X, dtype=float) - mean) / std) @ w


def win_chances(utilities):
    """P(first) for each athlete under Plackett-Luce, adding up to 1.

    Shown beside the podium chance since 2026-09-15. Three close athletes can
    each be near-certain of a medal while only one of them wins, so a podium
    chance alone cannot say whether the leader is clear of the rest."""
    u = np.asarray(utilities, dtype=float)
    if len(u) == 0:
        return u
    e = np.exp(u - u.max())
    return e / e.sum()


def predict(model, X):
    """Podium chances for one field's feature matrix."""
    return podium_chances(utilities(model, X))


# ---- finals as the model sees them --------------------------------------------

def load_scored_finals(path=None):
    df = pd.read_csv(path or fd.FINALS_PATH, low_memory=False)
    for col in ("cutoff", "sb_date", "dob"):
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def build_finals(scored, features=FEATURES):
    """One dict per final with at least MIN_SCORED scored finalists.

    `podium` holds every medallist's name, scored or not: an unscored medallist
    is a place neither the model nor points could have named, and counting it
    as possible keeps both honest. A shared bronze puts both names on it (23 of
    the 900 championship finals have a tie on the podium). `top_idx` is the scored rows of first,
    second and third, or None when any of the three is unscored.

    `match` is the share of the whole competition's finalists with a score, and
    `trainable` whether it clears field_data.MIN_MATCH. A final below the gate
    is still scored in the backtest, where an athlete we could not find is a
    miss for the model and for points alike. It is only kept out of fitting."""
    has_score = scored["sb_score"].notna()
    rates = has_score.groupby([scored["competition"], scored["year"]]).mean()
    finals = []
    for (competition, year, discipline), g in scored.groupby(["competition", "year", "discipline"]):
        placed = g[g["place"].between(1, 3)].sort_values("place")
        if len(placed) < 3:
            continue
        # The rows arrive in finishing order. Sorted by name instead, so that
        # when two athletes share a score, neither the points ranking nor the
        # model breaks the tie with the result it is trying to predict.
        with_score = (g[g["sb_score"].notna()]
                      .assign(_by_name=lambda d: d["athlete_name"].map(fd.field_key))
                      .sort_values("_by_name", kind="stable").drop(columns="_by_name")
                      .reset_index(drop=True))
        if len(with_score) < MIN_SCORED:
            continue
        X = field_features(with_score, g["cutoff"].iloc[0], features)
        names = with_score["athlete_name"].tolist()
        order = []
        for place in (1, 2, 3):
            hit = with_score.index[with_score["place"] == place]
            order.append(int(hit[0]) if len(hit) else None)
        match = float(rates.loc[(competition, year)])
        finals.append({
            "competition": competition, "year": int(year), "discipline": discipline,
            "group": group_of(discipline), "tier": g["tier"].iloc[0],
            "names": names, "scores": with_score["sb_score"].to_numpy(dtype=float),
            "X": X.to_numpy(), "features": list(features), "podium": set(placed["athlete_name"]),
            "winners": set(placed.loc[placed["place"] == 1, "athlete_name"]),
            "top_idx": order if None not in order else None,
            "match": match, "trainable": match >= fd.MIN_MATCH,
        })
    return finals


def _top3(names, values):
    order = np.argsort(-np.asarray(values, dtype=float), kind="stable")
    return {names[i] for i in order[:3]}


def backtest(finals, l2=L2, shuffle_seed=None):
    """Walk forward over TEST_YEARS. Returns (per-final results, per-athlete
    chances). With `shuffle_seed`, every final's feature rows are shuffled, for
    training and testing alike: the control, which should find nothing."""
    rng = np.random.default_rng(shuffle_seed)
    if shuffle_seed is not None:
        shuffled = []
        for f in finals:
            order = rng.permutation(len(f["names"]))
            shuffled.append({**f, "X": f["X"][order]})
        finals = shuffled

    results, athletes = [], []
    for year in TEST_YEARS:
        train = [f for f in finals if f["year"] < year]
        test = [f for f in finals if f["year"] == year]
        if not test or not any(f["top_idx"] is not None and f.get("trainable", True) for f in train):
            continue
        model = fit(train, l2)
        for f in test:
            u = utilities(model, f["X"])
            chances = podium_chances(u)
            model_pick = _top3(f["names"], chances)
            points_pick = _top3(f["names"], f["scores"])
            winners = f.get("winners", set())
            results.append({
                "competition": f["competition"], "year": year, "discipline": f["discipline"],
                "group": f["group"], "tier": f["tier"],
                "model_hits": len(model_pick & f["podium"]), "points_hits": len(points_pick & f["podium"]),
                "model_winner": f["names"][int(np.argmax(chances))] in winners,
                "points_winner": f["names"][int(np.argmax(f["scores"]))] in winners,
                # For compare(): how well the model read the order of the top
                # three, and how sure it was of its favourite.
                "ll": top3_log_likelihood(u, f["top_idx"]) if f.get("top_idx") is not None else np.nan,
                "favourite_win": float(win_chances(u).max()),
            })
            for name, chance in zip(f["names"], chances):
                athletes.append({"chance": float(chance), "medal": name in f["podium"]})
    return pd.DataFrame(results), pd.DataFrame(athletes)


def bootstrap_lower(d, n_boot=BOOTSTRAP, quantile=LOWER_QUANTILE, seed=0):
    d = np.asarray(d, dtype=float)
    if len(d) == 0:
        return None
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    return float(np.quantile(means, quantile))


def ship_decisions(results, control):
    """The pre-registered rule, group by group. See the module docstring."""
    out = {}
    for group in GROUPS:
        g = results[results["group"] == group]
        c = control[control["group"] == group]
        d = (g["model_hits"] - g["points_hits"]).to_numpy()
        asia = g[g["tier"] == "asia"]
        asia_d = (asia["model_hits"] - asia["points_hits"]).to_numpy()
        control_lower = bootstrap_lower((c["model_hits"] - c["points_hits"]).to_numpy())
        lower = bootstrap_lower(d)
        asia_mean = float(asia_d.mean()) if len(asia_d) else None
        passed = bool(
            lower is not None and lower > 0
            and asia_mean is not None and asia_mean >= 0
            and not (control_lower is not None and control_lower > 0)
        )
        out[group] = {
            "finals": int(len(g)), "possible": int(3 * len(g)),
            "modelHits": int(g["model_hits"].sum()), "pointsHits": int(g["points_hits"].sum()),
            "meanDiff": float(d.mean()) if len(d) else None, "lower90": lower,
            "asiaFinals": int(len(asia)), "asiaModelHits": int(asia["model_hits"].sum()),
            "asiaPointsHits": int(asia["points_hits"].sum()), "asiaMeanDiff": asia_mean,
            "controlModelHits": int(c["model_hits"].sum()), "controlLower90": control_lower,
            "passed": passed,
        }
    return out


# ---- comparing a new feature set with today's ---------------------------------

def top3_log_likelihood(u, top_idx):
    """The log-probability of one final's observed first, second and third
    under Plackett-Luce: per final, the quantity fit() maximises."""
    u = np.asarray(u, dtype=float)
    available = np.ones(len(u), dtype=bool)
    total = 0.0
    for i in top_idx:
        total += float(u[i] - logsumexp(u[available]))
        available[i] = False
    return total


def shuffle_columns(finals, columns, seed):
    """The finals with `columns` of each feature matrix permuted within the
    final, together, so those columns stop belonging to their athletes while
    every other column stays put. compare()'s control."""
    rng = np.random.default_rng(seed)
    out = []
    for f in finals:
        X = f["X"].copy()
        X[:, columns] = X[rng.permutation(len(X))][:, columns]
        out.append({**f, "X": X})
    return out


def compare_decision(base, cand, controls):
    """The rule for the race-by-race features, written down on 2026-09-15 before
    compare() first ran. The candidate ships only if all four hold:
      llLower90     the one-sided 90% bootstrap lower bound of its mean gain in
                    top-three log-likelihood per held-out final is above 0;
      hitsNotWorse  it names at least as many medallists per final on average;
      asiaNotWorse  its mean log-likelihood gain on the Asian finals is at least 0;
      controlsFail  no control (the candidate with its new columns shuffled
                    within each final) meets the first condition.
    Log-likelihood rewards confidence in a clear leader and doubt about a close
    field, which a count of medallists cannot see. `base`, `cand` and each
    control are backtest() results for the same finals in the same order."""
    read = base["ll"].notna() & cand["ll"].notna()
    gain = cand["ll"] - base["ll"]
    d_ll = gain[read].to_numpy()
    d_hits = (cand["model_hits"] - base["model_hits"]).to_numpy()
    asia = read & (base["tier"] == "asia")
    d_asia = gain[asia].to_numpy()
    lower = bootstrap_lower(d_ll)
    control_lower = [bootstrap_lower((c["ll"] - base["ll"])[read].to_numpy()) for c in controls]
    conditions = {
        "llLower90": bool(lower is not None and lower > 0),
        "hitsNotWorse": bool(len(d_hits) and d_hits.mean() >= 0),
        "asiaNotWorse": bool(len(d_asia) and d_asia.mean() >= 0),
        "controlsFail": not any(c is not None and c > 0 for c in control_lower),
    }
    return {
        "finals": int(len(base)), "llFinals": int(read.sum()), "asiaFinals": int(asia.sum()),
        "meanLlGain": float(d_ll.mean()) if len(d_ll) else None, "llLower90": lower,
        "meanHitsDiff": float(d_hits.mean()) if len(d_hits) else None,
        "asiaMeanLlGain": float(d_asia.mean()) if len(d_asia) else None,
        "controlLower90": control_lower,
        "conditions": conditions, "ships": all(conditions.values()),
    }


def _favourites(results):
    """How often the favourite won, by how sure the model was of them: clear
    favourites and close fields side by side."""
    out = []
    for lo, hi in ((0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.01)):
        g = results[(results["favourite_win"] >= lo) & (results["favourite_win"] < hi)]
        out.append({"from": lo, "to": min(hi, 1.0), "finals": int(len(g)),
                    "predicted": round(float(g["favourite_win"].mean()), 3) if len(g) else None,
                    "observed": round(float(g["model_winner"].mean()), 3) if len(g) else None})
    return out


def compare(scored, l2=L2, controls=CONTROLS):
    """Today's FEATURES against FEATURES_V2 on the same held-out finals.
    Returns (the report, the candidate fitted on every final)."""
    base_finals, cand_finals = build_finals(scored, FEATURES), build_finals(scored, FEATURES_V2)
    keys = [(f["competition"], f["year"], f["discipline"]) for f in base_finals]
    if keys != [(f["competition"], f["year"], f["discipline"]) for f in cand_finals]:
        raise ValueError("the two feature sets were built on different finals")
    base, base_athletes = backtest(base_finals, l2)
    cand, cand_athletes = backtest(cand_finals, l2)
    new = [FEATURES_V2.index(name) for name in NEW_FEATURES]
    runs = [backtest(shuffle_columns(cand_finals, new, seed), l2)[0] for seed in range(controls)]
    model = fit(cand_finals, l2)
    has_races = (scored["has_races"].fillna(False).astype(bool)[scored["sb_score"].notna()]
                 if "has_races" in scored.columns else None)

    def pair(b, c):
        return {"baseline": _block(b), "candidate": _block(c)}

    report = {
        "builtAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "testYears": TEST_YEARS, "features": FEATURES_V2, "newFeatures": NEW_FEATURES,
        "bigCategories": sorted(fd.BIG_CATEGORIES), "recentDays": fd.RECENT_DAYS, "thinSeason": THIN_SEASON,
        "overall": pair(base, cand),
        "byTier": {t: pair(base[base["tier"] == t], cand[cand["tier"] == t]) for t in sorted(base["tier"].unique())},
        "byGroup": {g: pair(base[base["group"] == g], cand[cand["group"] == g]) for g in GROUPS},
        "decision": compare_decision(base, cand, runs),
        "weights": dict(zip(model["features"], model["weights"])),
        "calibration": {"baseline": _calibration(base_athletes), "candidate": _calibration(cand_athletes)},
        "favourites": {"baseline": _favourites(base), "candidate": _favourites(cand)},
        "raceCoverage": None if has_races is None else round(float(has_races.mean()), 3),
    }
    return report, model


def print_compare(report):
    def signed(x, places=4):
        return "n/a" if x is None else f"{x:+.{places}f}"

    d, o = report["decision"], report["overall"]
    b, c = o["baseline"], o["candidate"]
    print(f"\n  {b['finals']} held-out finals, {report['testYears'][0]}-{report['testYears'][-1]}: "
          f"today {b['model']}%, race by race {c['model']}% (points {b['points']}%)")
    print(f"  winners named first: today {b.get('modelWinners')}, race by race {c.get('modelWinners')}")
    for label, table in (("tier", report["byTier"]), ("group", report["byGroup"])):
        for name, p in table.items():
            print(f"    {label} {name:<12} {p['baseline']['finals']:>4} finals  today {p['baseline']['model']}%  "
                  f"race by race {p['candidate']['model']}%")
    print(f"\n  top-three log-likelihood gain per final: mean {signed(d['meanLlGain'])}, 90% lower bound "
          f"{signed(d['llLower90'])} ({d['llFinals']} finals); Asian finals mean {signed(d['asiaMeanLlGain'])} "
          f"({d['asiaFinals']}); medallists per final {signed(d['meanHitsDiff'], 3)}")
    print("  shuffled controls' lower bounds: " + ", ".join(signed(x) for x in d["controlLower90"]))
    for name, ok in d["conditions"].items():
        print(f"    {'pass' if ok else 'FAIL'}  {name}")
    print(f"  verdict: {'SHIPS' if d['ships'] else 'does not ship, so the call stays as it is'}")
    print("\n  weights: " + ", ".join(f"{k} {signed(v, 3)}" for k, v in report["weights"].items()))
    print("  the favourite's win chance -> how often they won (today | race by race):")
    for fb, fc in zip(report["favourites"]["baseline"], report["favourites"]["candidate"]):
        print(f"    {fb['from']:.1f}-{fb['to']:.1f}  {fb['finals']:>4} finals {fb['predicted']} -> {fb['observed']}"
              f"  |  {fc['finals']:>4} finals {fc['predicted']} -> {fc['observed']}")
    if report.get("raceCoverage") is not None:
        print(f"  scored finalists with a race-by-race season on file: {report['raceCoverage']:.1%}")


def _pct(hits, possible):
    return round(100.0 * hits / possible, 1) if possible else None


def _block(df):
    """Podium places named, and winners named first, by the model and by points."""
    possible = 3 * len(df)
    out = {"finals": int(len(df)),
           "model": _pct(df["model_hits"].sum(), possible),
           "points": _pct(df["points_hits"].sum(), possible)}
    if "model_winner" in df.columns:
        out["modelWinners"] = int(df["model_winner"].sum())
        out["pointsWinners"] = int(df["points_winner"].sum())
    return out


def _calibration(athletes):
    """Predicted against observed podium rate, in tenths of chance."""
    calibration = []
    if not athletes.empty:
        bins = np.linspace(0, 1, 11)
        athletes = athletes.assign(bin=np.clip(np.digitize(athletes["chance"], bins) - 1, 0, 9))
        for b, g in athletes.groupby("bin"):
            calibration.append({"from": round(bins[b], 1), "to": round(bins[b + 1], 1), "athletes": int(len(g)),
                                "predicted": round(float(g["chance"].mean()), 3),
                                "observed": round(float(g["medal"].mean()), 3)})
    return calibration


def summarise(results, athletes, control, decisions, coverage=None):
    return {
        "builtAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "testYears": TEST_YEARS, "minScored": MIN_SCORED,
        "overall": _block(results),
        "byTier": {tier: _block(g) for tier, g in results.groupby("tier")},
        "control": _block(control),
        "groups": decisions,
        "calibration": _calibration(athletes),
        "minMatch": fd.MIN_MATCH,
        "coverage": [] if coverage is None else json.loads(coverage.to_json(orient="records")),
    }


def print_report(report):
    o, c = report["overall"], report["control"]
    print(f"\n  {o['finals']} held-out finals, {report['testYears'][0]}-{report['testYears'][-1]}: "
          f"model {o['model']}%  points {o['points']}%   (shuffled control: model {c['model']}%)")
    print(f"  winners named first: model {o.get('modelWinners')}, points {o.get('pointsWinners')}")
    for tier, b in report["byTier"].items():
        print(f"    {tier:<12} {b['finals']:>4} finals  model {b['model']}%  points {b['points']}%  "
              f"winners {b.get('modelWinners')} / {b.get('pointsWinners')}")
    below = [row for row in report["coverage"] if not row["trained"]]
    if below:
        print(f"  Under {report['minMatch']:.0%} of finalists found, scored but not trained on: "
              + ", ".join(f"{fd.competition_label(row['competition'], row['year'])} ({row['scored']:.0%})"
                          for row in below))
    print("\n  group      finals  model  points  mean d  lower90  asia(m/p)   control  verdict")
    for group, g in report["groups"].items():
        print(f"  {group:<10} {g['finals']:>6} {_pct(g['modelHits'], g['possible']):>6} "
              f"{_pct(g['pointsHits'], g['possible']):>6}  {g['meanDiff'] if g['meanDiff'] is None else round(g['meanDiff'], 3):>6} "
              f"{g['lower90'] if g['lower90'] is None else round(g['lower90'], 3):>8}  "
              f"{g['asiaModelHits']:>3}/{g['asiaPointsHits']:<3} ({g['asiaFinals']})  "
              f"{_pct(g['controlModelHits'], g['possible']):>6}  {'SHIPS' if g['passed'] else 'stays on points'}")
    print("\n  calibration (predicted -> observed podium rate):")
    for b in report["calibration"]:
        print(f"    {b['from']:.1f}-{b['to']:.1f}  {b['athletes']:>5} athletes  {b['predicted']:.3f} -> {b['observed']:.3f}")


# ---- serving ------------------------------------------------------------------

def load_model(path=MODEL_PATH):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def shipped(model, disc_key):
    """(whether this discipline's group passed the ship rule, its evidence)."""
    decision = ((model or {}).get("ship") or {}).get(group_of(disc_key))
    return bool(decision and decision.get("passed")), decision


SERVING_COLUMNS = ["athlete_name", "sb_score", "sb_date", "sb_prior_season", "dob",
                   "career_best", "prev_season_best", "mark", "mark_season", "wa_id"]


def serving_rows(disc_key, athletes, season_path, cutoff, year, scores_for=fd.season_scores, extra=(),
                 races=None):
    """The rows field_features needs, for athletes about to compete, read the
    way field_data.attach_scores reads a past final.

    `athletes` is [{name, nat, birthDate?}]. The season best is the best row in
    `season_path` (this season's toplist or a championship's merged snapshot)
    dated before the cut-off. An athlete with none takes last season's best
    from their history, flagged sb_prior_season: training fell back the same
    way, so serving must. History is `scores_for(disc_key)` plus the `extra`
    (path, source) toplists, such as last season's Asian list, and gives the
    career best and last season's best too. An athlete with neither mark is
    left out. `mark` and `mark_season` say which mark was used, for the page.

    With `races`, {World Athletics id as text: this season race by race, from
    field_data.fetch_races}, the rows also carry field_data.RACE_COLUMNS, read
    through field_data.race_columns as training reads them, head-to-head
    included: what FEATURES_V2 needs. `wa_id` is the entrant's id, or the one on
    the toplist row their mark came from."""
    season = fd._toplist_rows(season_path, "season")
    history = scores_for(disc_key)
    frames = [f for f in [history] + [fd._toplist_rows(path, source) for path, source in extra] if not f.empty]
    if frames:
        history = pd.concat(frames, ignore_index=True)
    cutoff = pd.Timestamp(cutoff)
    rows = []
    for a in athletes:
        key, nat = fd.field_key(a["name"]), str(a.get("nat") or "").strip()
        wid = int(a["waId"]) if str(a.get("waId") or "").isdigit() else None
        mine = fd.athlete_history(season, key, nat, wid)[0] if not season.empty else season
        mine = mine[mine["date"] < cutoff] if not mine.empty else mine
        past = fd.athlete_history(history, key, nat, wid)[0] if not history.empty else history
        earlier = past[past["year"] < year] if not past.empty else past
        last = past[past["year"] == year - 1] if not past.empty else past
        if not mine.empty:
            best, prior = mine.loc[mine["score"].idxmax()], 0
        elif not last.empty:
            best, prior = last.loc[last["score"].idxmax()], 1
        else:
            continue
        dob = best["dob"] if pd.notna(best["dob"]) else None
        if dob is None and not past.empty and past["dob"].notna().any():
            dob = past["dob"].dropna().iloc[0]
        if dob is None:
            dob = fd.parse_date(a.get("birthDate"))
        if wid is None and "wa_id" in best.index and pd.notna(best["wa_id"]):
            wid = int(best["wa_id"])
        rows.append({
            "athlete_name": a["name"], "sb_score": float(best["score"]), "sb_date": best["date"],
            "sb_prior_season": prior, "dob": dob,
            "career_best": float(earlier["score"].max()) if not earlier.empty else None,
            "prev_season_best": float(last["score"].max()) if not last.empty else None,
            "mark": best["mark"], "mark_season": int(best["year"]), "wa_id": wid,
        })
    out = pd.DataFrame(rows, columns=SERVING_COLUMNS)
    if races is None or out.empty:
        return out
    field = out.assign(discipline=disc_key, cutoff=cutoff)
    return out.join(fd.race_columns(
        field, lambda row: races.get(str(int(row["wa_id"]))) if pd.notna(row["wa_id"]) else None))


def field_chances(model, rows, cutoff):
    """{athlete name: (podium chance, win chance)} for one field. Empty for a
    field of fewer than three scored athletes, where there is no podium to call."""
    rows = rows[rows["sb_score"].notna()].reset_index(drop=True)
    if len(rows) < 3:
        return {}
    u = utilities(model, field_features(rows, cutoff).to_numpy())
    return {name: (float(p), float(w))
            for name, p, w in zip(rows["athlete_name"], podium_chances(u), win_chances(u))}


def score_field(model, rows, cutoff):
    """{athlete name: podium chance} for one field (see field_chances)."""
    return {name: podium for name, (podium, _) in field_chances(model, rows, cutoff).items()}


def run_compare():
    """compare() on data/field/finals.csv, printed and saved beside the served
    model, never over it: promoting the candidate is a separate step, taken
    only after its result has been read."""
    print("=== Field model: today's features against race by race, on the same held-out finals ===")
    scored = load_scored_finals()
    missing = [c for c in fd.RACE_COLUMNS if c not in scored.columns]
    if missing:
        print(f"  {fd.FINALS_PATH} has no {', '.join(missing)}: run field_data.py --races, then --report")
        return 1
    report, model = compare(scored)
    print_compare(report)
    model["fitAt"] = report["builtAt"]
    model["compare"] = report["decision"]
    os.makedirs(os.path.dirname(COMPARE_PATH), exist_ok=True)
    with open(COMPARE_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    with open(V2_MODEL_PATH, "w", encoding="utf-8") as f:
        json.dump(model, f, indent=1)
    print(f"\n  -> {COMPARE_PATH}\n  -> {V2_MODEL_PATH}\n  The served model, {MODEL_PATH}, is unchanged.")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--backtest", action="store_true")
    parser.add_argument("--compare", action="store_true",
                        help="today's features against the race-by-race ones, on the same held-out finals")
    args = parser.parse_args(argv)
    if args.compare:
        return run_compare()
    if not args.backtest:
        parser.print_help()
        return 0

    print("=== Field model: walk-forward backtest against points ===")
    scored = load_scored_finals()
    finals = build_finals(scored)
    print(f"  {len(finals)} usable finals, {sum(f['top_idx'] is not None for f in finals)} with a scored podium, "
          f"{sum(f['top_idx'] is not None and f['trainable'] for f in finals)} of those trainable")
    results, athletes = backtest(finals)
    control, _ = backtest(finals, shuffle_seed=7)
    decisions = ship_decisions(results, control)
    report = summarise(results, athletes, control, decisions, fd.coverage(scored))
    print_report(report)

    model = fit(finals)
    model["ship"] = decisions
    model["fitAt"] = report["builtAt"]
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "w", encoding="utf-8") as f:
        json.dump(model, f, indent=1)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    print(f"\n  -> {MODEL_PATH}\n  -> {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
