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
Reads data/field/finals.csv (field_data.py --report). Writes
outputs/field_model.json and outputs/field_model_report.json.
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


def group_of(disc_key):
    event = str(disc_key).split("_", 1)[-1]
    return next((name for name, events in GROUPS.items() if event in events), None)


# ---- features -----------------------------------------------------------------

def field_features(rows, cutoff):
    """FEATURES for every row of one field that has a season score.

    `rows` needs sb_score, sb_date, sb_prior_season, career_best,
    prev_season_best and dob. Scores are divided by 100 so a coefficient reads
    per 100 World Athletics points. An athlete with no age takes the field's
    median, and 25 when nobody in the field has one."""
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
    return out[FEATURES].astype(float)


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
    X = np.zeros((len(finals), width, len(FEATURES)))
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
    result = minimize(_loss_and_grad, np.zeros(len(FEATURES)), args=(X, mask, top, l2),
                      jac=True, method="L-BFGS-B")
    return {"features": list(FEATURES), "mean": mean.tolist(), "std": std.tolist(),
            "weights": result.x.tolist(), "l2": l2, "finals": len(usable),
            "converged": bool(result.success)}


def predict(model, X):
    """Podium chances for one field's feature matrix."""
    mean, std, w = (np.asarray(model[k], dtype=float) for k in ("mean", "std", "weights"))
    return podium_chances(((np.asarray(X, dtype=float) - mean) / std) @ w)


# ---- finals as the model sees them --------------------------------------------

def load_scored_finals(path=None):
    df = pd.read_csv(path or fd.FINALS_PATH, low_memory=False)
    for col in ("cutoff", "sb_date", "dob"):
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def build_finals(scored):
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
        with_score = g[g["sb_score"].notna()].reset_index(drop=True)
        if len(with_score) < MIN_SCORED:
            continue
        features = field_features(with_score, g["cutoff"].iloc[0])
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
            "X": features.to_numpy(), "podium": set(placed["athlete_name"]),
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
            chances = predict(model, f["X"])
            model_pick = _top3(f["names"], chances)
            points_pick = _top3(f["names"], f["scores"])
            winners = f.get("winners", set())
            results.append({
                "competition": f["competition"], "year": year, "discipline": f["discipline"],
                "group": f["group"], "tier": f["tier"],
                "model_hits": len(model_pick & f["podium"]), "points_hits": len(points_pick & f["podium"]),
                "model_winner": f["names"][int(np.argmax(chances))] in winners,
                "points_winner": f["names"][int(np.argmax(f["scores"]))] in winners,
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


def _pct(hits, possible):
    return round(100.0 * hits / possible, 1) if possible else None


def summarise(results, athletes, control, decisions, coverage=None):
    def block(df):
        possible = 3 * len(df)
        out = {"finals": int(len(df)),
               "model": _pct(df["model_hits"].sum(), possible),
               "points": _pct(df["points_hits"].sum(), possible)}
        if "model_winner" in df.columns:
            out["modelWinners"] = int(df["model_winner"].sum())
            out["pointsWinners"] = int(df["points_winner"].sum())
        return out

    calibration = []
    if not athletes.empty:
        bins = np.linspace(0, 1, 11)
        athletes = athletes.assign(bin=np.clip(np.digitize(athletes["chance"], bins) - 1, 0, 9))
        for b, g in athletes.groupby("bin"):
            calibration.append({"from": round(bins[b], 1), "to": round(bins[b + 1], 1), "athletes": int(len(g)),
                                "predicted": round(float(g["chance"].mean()), 3),
                                "observed": round(float(g["medal"].mean()), 3)})
    return {
        "builtAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "testYears": TEST_YEARS, "minScored": MIN_SCORED,
        "overall": block(results),
        "byTier": {tier: block(g) for tier, g in results.groupby("tier")},
        "control": block(control),
        "groups": decisions,
        "calibration": calibration,
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


def serving_rows(disc_key, athletes, season_path, cutoff, year, scores_for=fd.season_scores):
    """The rows field_features needs, for athletes about to compete.

    `athletes` is [{name, nat}]. Their season best, its date and date of birth
    come from `season_path` (this season's toplist or a championship's merged
    snapshot); career best and last season's best from the same history the
    backtest read. Athletes not found in `season_path` are left out."""
    season = fd._toplist_rows(season_path, "season")
    history = scores_for(disc_key)
    rows = []
    for a in athletes:
        key, nat = fd.field_key(a["name"]), str(a.get("nat") or "").strip()
        mine, _how = fd.athlete_history(season, key, nat)
        mine = mine[mine["date"] < pd.Timestamp(cutoff)] if not mine.empty else mine
        if mine.empty:
            continue
        best = mine.loc[mine["score"].idxmax()]
        past, _ = fd.athlete_history(history, key, nat)
        earlier = past[past["year"] < year]
        last = past[past["year"] == year - 1]
        dob = best["dob"] if pd.notna(best["dob"]) else (past["dob"].dropna().iloc[0] if past["dob"].notna().any() else None)
        rows.append({
            "athlete_name": a["name"], "sb_score": float(best["score"]), "sb_date": best["date"],
            "sb_prior_season": 0, "dob": dob,
            "career_best": float(earlier["score"].max()) if not earlier.empty else None,
            "prev_season_best": float(last["score"].max()) if not last.empty else None,
        })
    return pd.DataFrame(rows, columns=["athlete_name", "sb_score", "sb_date", "sb_prior_season",
                                       "dob", "career_best", "prev_season_best"])


def score_field(model, rows, cutoff):
    """{athlete name: podium chance} for one field. Empty for a field of fewer
    than three scored athletes, where there is no podium to call."""
    rows = rows[rows["sb_score"].notna()].reset_index(drop=True)
    if len(rows) < 3:
        return {}
    chances = predict(model, field_features(rows, cutoff).to_numpy())
    return dict(zip(rows["athlete_name"], (float(c) for c in chances)))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--backtest", action="store_true")
    args = parser.parse_args(argv)
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
