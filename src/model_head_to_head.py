"""
model_head_to_head.py -- the Diamond League model against the championship
model, on the same past finals, each season called only from the seasons
before it.

WHY
The site runs two models. The Diamond League model (train_model.py, a random
forest) rates the world's top 20 on Track and Field, picks the dashboard
favourites and gives the athlete page figure. The championship model
(field_model.py, `old_marks`) calls the Asian Games and rates the hammer and the
10,000m. The Diamond League model learned that athletes who race the circuit a
lot reach Diamond League Finals, so its rating follows meetings raced more than
marks (HANDOFF.md, 2026-09-07). On 2026-09-16 the user asked whether the
championship model should replace it across the site before the Asian Games:
"I'd rather have it be a full-on meet predictor, like championship meets".

THE TEST
Every final both models have results for, 2021 to 2025, the years the Diamond
League model's own walk-forward scores: Diamond League Finals, Olympic Games,
World Championships and European Championships. Each model is refitted for each
year on earlier seasons only, exactly as it is backtested on its own:
  * the Diamond League model as deployed: the 18 features in
    outputs/feature_cols.pkl, DEFAULT_MODEL_PARAMS, trained on the pooled
    Diamond League, global and continental finals, and scored among the
    athletes who actually contested each final;
  * the championship model as served: EXPERIMENTS["old_marks"], its fade
    settings and bounds.
A final a model could not call counts as three misses for it, and a medallist it
had no data on counts as a miss, the convention both backtests already use. A
ranking by World Athletics points is shown beside them for reference.

THE RULE (fixed 2026-09-16, before this file first ran)
The championship model replaces the Diamond League model across the site if
  1. it names at least as many medallists on the championship finals
     (Olympics, World Championships, European Championships), and
  2. it names at least as many medallists across all of them, Diamond League
     Finals included.
If 1 holds and 2 does not, the decision is the user's: they said championship
meets matter most, and the numbers will say what the switch costs on Diamond
League Finals. Winners named are reported and decide nothing.

WHAT THIS CANNOT SAY
Both models were tuned on these same seasons: the Diamond League model's
hyperparameters by walk-forward search, the championship model's settings by
the old-marks grid. So this compares them on equal footing, but neither number
is a result on finals the models had never been shaped by. The Asian Games are
that test, for the championship model.

Usage:
    python src/model_head_to_head.py      # writes outputs/model_head_to_head.json
"""
import json
import os
import pickle
import sys
from datetime import datetime, timezone

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import field_data as fd  # noqa: E402
import field_model as fm  # noqa: E402

BASE_DIR = fd.BASE_DIR
OUT_PATH = os.path.join(BASE_DIR, "outputs", "model_head_to_head.json")
LABELS_PATH = os.path.join(BASE_DIR, "data", "labels", "finals.csv")
YEARS = [2021, 2022, 2023, 2024, 2025]
DL_TIERS = ["dl_final", "global", "continental"]
KEY = ["year", "discipline", "competition"]


def kind_of(tier, competition):
    """"championship" or "dlFinal" for a final in the comparison, None for one
    outside it (the Continental Tour, the Continental Cup, the Asian finals)."""
    if tier == "dl_final":
        return "dlFinal"
    if tier == "global":
        return "championship"
    if tier == "continental" and "European" in str(competition):
        return "championship"
    return None


def universe(labels, field_finals):
    """The finals in both models' histories, 2021-2025: [(year, discipline,
    competition, kind)]. Only finals both label sets hold, so neither model is
    scored on a final the other never had."""
    ours = {(int(r.year), r.discipline, r.competition): kind_of(r.tier, r.competition)
            for r in labels[labels["year"].isin(YEARS)].drop_duplicates(KEY).itertuples()}
    theirs = {(int(r.year), r.discipline, r.competition)
              for r in field_finals[field_finals["year"].isin(YEARS)].drop_duplicates(KEY).itertuples()}
    return sorted((y, d, c, k) for (y, d, c), k in ours.items() if k and (y, d, c) in theirs)


def dl_model_calls(features):
    """{(year, discipline, competition): (medallists named, winner named)} from
    the Diamond League model, refitted for each year on earlier label years only,
    and scored among the athletes who contested the final, as train_model's
    _score_fold scores it."""
    import train_model as tm

    full = tm.build_pooled_dataset(DL_TIERS)
    full = tm.add_h2h_features(full)
    full = full.dropna(subset=features)
    full = tm.mark_final_field(full)
    calls = {}
    for year in YEARS:
        train = full[full["year"].isin([y for y in tm.LABEL_YEARS if y < year])]
        test = full[full["year"] == year].copy()
        if train.empty or test.empty:
            continue
        scaler = StandardScaler()
        model = RandomForestClassifier(**tm.DEFAULT_MODEL_PARAMS)
        model.fit(scaler.fit_transform(train[features]), train["dl_top3"])
        test["p"] = model.predict_proba(scaler.transform(test[features]))[:, 1]
        for (discipline, competition), final in test.groupby(["discipline", "competition"]):
            field = final[final["in_final_field"]].sort_values("p", ascending=False)
            if field.empty:
                continue
            picks = field.head(3)["athlete_name"].tolist()
            medallists = set(field.loc[field["dl_top3"] == 1, "athlete_name"])
            winners = set(field.loc[field["dl_rank"] == 1, "athlete_name"]) if "dl_rank" in field else set()
            calls[(int(year), discipline, competition)] = (len(set(picks) & medallists), picks[0] in winners)
    return calls


def field_model_calls():
    """{(year, discipline, competition): (model medallists, model winner, points
    medallists, points winner)} from the championship model as served, walked
    forward over YEARS."""
    spec = fm.EXPERIMENTS["old_marks"]
    scored = fm._scored_with_races()
    if scored is None:
        raise SystemExit("finals.csv has no race columns: run field_data.py --races, then --report")
    with_races = fm.spec_scored(scored, spec, fd.load_seasons(fd.RACES_DIR))
    finals = fm.build_finals(with_races, spec["features"], spec["fade"])
    results, _ = fm.backtest(finals, spec.get("l2", fm.L2), years=YEARS, bounds=spec["bounds"])
    return {(int(r.year), r.discipline, r.competition):
            (int(r.model_hits), bool(r.model_winner), int(r.points_hits), bool(r.points_winner))
            for r in results.itertuples()}


def summarise(rows):
    """Totals for one group of finals: medallists named out of three per final,
    and winners named, for each side."""
    n = len(rows)
    out = {"finals": n, "possible": 3 * n}
    for side in ("dl", "championship", "points"):
        hits = sum(r[side][0] for r in rows)
        out[side] = {"medallists": hits, "medallistsPct": round(100 * hits / (3 * n), 1) if n else None,
                     "winners": sum(1 for r in rows if r[side][1])}
    out["notCalled"] = {"dl": sum(1 for r in rows if not r["dlCalled"]),
                        "championship": sum(1 for r in rows if not r["championshipCalled"])}
    return out


def decide(groups):
    """The rule in the module docstring, applied to summarise()'s groups."""
    champ, overall = groups["championship"], groups["all"]
    beats_on_championships = champ["championship"]["medallists"] >= champ["dl"]["medallists"]
    beats_overall = overall["championship"]["medallists"] >= overall["dl"]["medallists"]
    if beats_on_championships and beats_overall:
        verdict = "switch"
    elif beats_on_championships:
        verdict = "userDecides"
    else:
        verdict = "keep"
    return {"verdict": verdict, "championshipFinalsNotWorse": beats_on_championships,
            "allFinalsNotWorse": beats_overall}


def main():
    features = pickle.load(open(os.path.join(BASE_DIR, "outputs", "feature_cols.pkl"), "rb"))
    labels = pd.read_csv(LABELS_PATH, low_memory=False)
    field_finals = pd.read_csv(fd.FINALS_PATH, low_memory=False, usecols=KEY + ["tier"])
    finals = universe(labels, field_finals)
    print(f"=== Diamond League model against the championship model, {YEARS[0]}-{YEARS[-1]} ===")
    print(f"  {len(finals)} finals in both histories "
          f"({sum(1 for f in finals if f[3] == 'championship')} championship, "
          f"{sum(1 for f in finals if f[3] == 'dlFinal')} Diamond League Finals)")

    print("  the Diamond League model, refitted year by year...")
    dl = dl_model_calls(features)
    print("  the championship model, refitted year by year...")
    ch = field_model_calls()

    rows = []
    for year, discipline, competition, kind in finals:
        key = (year, discipline, competition)
        d, c = dl.get(key), ch.get(key)
        rows.append({"year": year, "discipline": discipline, "competition": competition, "kind": kind,
                     "dl": d or (0, False), "championship": (c[0], c[1]) if c else (0, False),
                     "points": (c[2], c[3]) if c else (0, False),
                     "dlCalled": d is not None, "championshipCalled": c is not None})
    groups = {"championship": summarise([r for r in rows if r["kind"] == "championship"]),
              "dlFinal": summarise([r for r in rows if r["kind"] == "dlFinal"]),
              "all": summarise(rows)}
    decision = decide(groups)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"builtAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "years": YEARS,
                   "rule": "switch if the championship model names at least as many medallists on the "
                           "championship finals and across all finals; if only the first, the user decides; "
                           "fixed before the first run",
                   "tunedOnTheseSeasons": True, "groups": groups, "decision": decision,
                   "finals": [{**r, "dl": list(r["dl"]), "championship": list(r["championship"]),
                               "points": list(r["points"])} for r in rows]}, f, indent=1)

    labels_for = {"championship": "Olympics, Worlds, Europeans", "dlFinal": "Diamond League Finals",
                  "all": "All of them"}
    print(f"\n  {'finals':<30} {'DL model':>18} {'championship model':>20} {'points':>16}")
    for name, g in groups.items():
        cell = lambda s: f"{g[s]['medallists']:>4} ({g[s]['medallistsPct']:>4}%) {g[s]['winners']:>3}w"  # noqa: E731
        print(f"  {labels_for[name] + ' (' + str(g['finals']) + ')':<30} {cell('dl'):>18} "
              f"{cell('championship'):>20} {cell('points'):>16}")
    print(f"\n  finals a model could not call: DL model {groups['all']['notCalled']['dl']}, "
          f"championship model {groups['all']['notCalled']['championship']}")
    print(f"  by the rule fixed before this ran: {decision['verdict']}")
    print("  both models were tuned on these seasons; the Asian Games are the unseen test")
    print(f"  -> {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
