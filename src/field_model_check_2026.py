"""
field_model_check_2026.py -- the field model on the 2026 championships it was
not fitted on: the Diamond League Final (Brussels, 4 September) and the World
Athletics Ultimate Championship (Budapest, 11-13 September).

Reported beside field_model.compare(), and not part of its rule: 57 finals are
too few to decide anything, and the rule was fixed before any of these numbers
existed. Each final is read the way field_data reads every past final: ids
from the competition's own results feed, season bests dated before its first
day (the toplist, then the athlete's profile, then last season), and races
before that day. So an athlete whose season best came at the championship is
read on their best before it, not sent to last season, which is what a toplist
refreshed after the Ultimate did to the first look at those finals on
2026-09-15.

Today's feature set and the experiment (the served one unless another is named)
are fitted only on the finals before 2026 in data/field/finals.csv, so the
check means the same after these championships join the history. The points
ranking is scored beside them, and on the Ultimate the old Diamond League
model's frozen call (the Diamond League Final's is graded on the Results page).

Run on 2026-09-15 for the Ultimate, and on 2026-09-16 with the Diamond League
Final added, before either joined the history.

Usage, after field_data.py --races and --report:
    python src/field_model_check_2026.py [EXPERIMENT]
Fetches the finalists' 2026 seasons into data/field/races_2026/ and writes
outputs/field_model_check_2026.json.
"""
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import field_data as fd  # noqa: E402
import field_model as fm  # noqa: E402

YEAR = 2026
FROZEN_CALL_PATH = os.path.join(fd.BASE_DIR, "data", "ultimate", "predictions_prefinal.json")
RACES_DIR = os.path.join(fd.FIELD_DIR, "races_2026")
OUT_PATH = os.path.join(fd.BASE_DIR, "outputs", "field_model_check_2026.json")


def championships(dl_finals=fd.dl_finals, event_path=None):
    """[(key, competition, finals)] for each YEAR championship with results on disk."""
    out = []
    dl = dl_finals()
    dl = dl[dl["year"] == YEAR]
    if not dl.empty:
        out.append(("dlFinal", "Diamond League Final", dl))
    event_path = event_path or fd.ULTIMATE_EVENT_PATH
    if os.path.exists(event_path):
        with open(event_path, encoding="utf-8") as f:
            ultimate = fd.ultimate_finals(json.load(f))
        if not ultimate.empty:
            out.append(("ultimate", fd.ULTIMATE_COMPETITION, ultimate))
    return out


def training_history(history, year=YEAR):
    """The finals a model checked on `year` may learn from: every season before it."""
    return history[history["year"] < year]


def score(model, finals):
    """Per final: medallists named, winner named first, the favourite and how
    sure the model was of them, and the top-three log-likelihood."""
    out = []
    for f in finals:
        u = fm.utilities(fm.model_for(model, f["group"]), f["X"])
        chances = fm.podium_chances(u)
        favourite = f["names"][int(np.argmax(chances))]
        out.append({"discipline": f["discipline"],
                    "hits": len(fm._top3(f["names"], chances) & f["podium"]),
                    "pointsHits": len(fm._top3(f["names"], f["scores"]) & f["podium"]),
                    "winner": favourite in f["winners"], "favourite": favourite,
                    "favouriteWin": round(float(fm.win_chances(u).max()), 3),
                    "ll": fm.top3_log_likelihood(u, f["top_idx"]) if f["top_idx"] is not None else None})
    return out


def frozen_call(finals):
    """The old model's call frozen before the Ultimate, scored on the same finals."""
    with open(FROZEN_CALL_PATH, encoding="utf-8") as f:
        projections = {p["discKey"]: p for p in json.load(f)["projections"]}
    hits = winners = 0
    for f in finals:
        called = [a for a in (projections.get(f["discipline"]) or {}).get("athletes") or []
                  if a.get("podiumChance") is not None]
        order = [fd.field_key(a["name"]) for a in sorted(called, key=lambda a: -a["podiumChance"])]
        podium = {fd.field_key(n) for n in f["podium"]}
        hits += len(set(order[:3]) & podium)
        winners += bool(order) and order[0] in {fd.field_key(n) for n in f["winners"]}
    return {"hits": hits, "winners": winners}


def totals(rows):
    ll = [r["ll"] for r in rows if r["ll"] is not None]
    return {"finals": len(rows), "hits": sum(r["hits"] for r in rows), "possible": 3 * len(rows),
            "winners": sum(r["winner"] for r in rows), "pointsHits": sum(r["pointsHits"] for r in rows),
            "meanLl": round(float(np.mean(ll)), 4) if ll else None, "llFinals": len(ll)}


def print_block(block, name):
    t, r = block["today"]["totals"], block["candidate"]["totals"]
    print(f"\n  {t['finals']} finals, {t['possible']} podium places")
    print(f"    today's model      {t['hits']} medallists, {t['winners']} winners, mean log-likelihood {t['meanLl']}")
    print(f"    {name:<18} {r['hits']} medallists, {r['winners']} winners, mean log-likelihood {r['meanLl']}")
    print(f"    points             {t['pointsHits']} medallists")
    if "frozenCall" in block:
        print(f"    old frozen call    {block['frozenCall']['hits']} medallists, "
              f"{block['frozenCall']['winners']} winners")
    print(f"\n  event            winner named first (today | {name}), favourite's win chance")
    for a, b in zip(block["today"]["finals"], block["candidate"]["finals"]):
        print(f"    {a['discipline']:<14} {a['favourite'][:22]:<22} {'Y' if a['winner'] else 'n'} {a['favouriteWin']:.2f}"
              f"  |  {b['favourite'][:22]:<22} {'Y' if b['winner'] else 'n'} {b['favouriteWin']:.2f}")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    name = argv[0] if argv else (fm.load_model() or {}).get("experiment") or "v2"
    if name not in fm.EXPERIMENTS:
        print(f"  no such experiment: {name} (see field_model.EXPERIMENTS)")
        return 1
    history = fm.load_scored_finals()
    missing = [c for c in fd.RACE_COLUMNS if c not in history.columns]
    if missing:
        print(f"  {fd.FINALS_PATH} has no {', '.join(missing)}: run field_data.py --races, then --report")
        return 1
    history = training_history(history)
    history_races = fd.load_seasons(fd.RACES_DIR) if fm.EXPERIMENTS[name].get("race") else None
    models = {}
    for label, spec_name in (("today", "today"), ("candidate", name)):
        spec = fm.EXPERIMENTS[spec_name]
        finals = fm.build_finals(fm.spec_scored(history, spec, history_races), spec["features"])
        models[label] = (spec_name, spec, fm.fit_spec(finals, spec))
    first, last = int(history["year"].min()), int(history["year"].max())
    print(f"=== Today's features and {name}, both fitted on the finals of {first}-{last} ===")

    report = {"builtAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "experiment": name,
              "trainedOn": [first, last], "competitions": {}}
    both = {"today": [], "candidate": []}
    for key, competition, finals in championships():
        cutoff = pd.Timestamp(finals["cutoff"].iloc[0])
        print(f"\n=== {competition} {YEAR}: {finals['discipline'].nunique()} finals, {len(finals)} finalists, "
              f"cut off {cutoff.date()} ===")
        ids = fd.finalist_ids(finals)
        finals = finals.merge(ids, on=["competition", "year", "discipline", "athlete_name"], how="left")
        fd.fetch_seasons({(a, YEAR) for a in finals["athlete_id"].dropna()}, fetch=fd.fetch_races,
                         seasons_dir=RACES_DIR)
        races = fd.load_seasons(RACES_DIR)
        scored = fd.attach_races(fd.attach_scores(finals, scores_for=fd.season_scores_with(YEAR), seasons=races),
                                 races)
        print("  Where each season best came from: "
              + ", ".join(f"{k} {v}" for k, v in scored["sb_source"].fillna("unscored").value_counts().items()))
        block = {"competition": competition, "cutoff": str(cutoff.date())}
        for label, (spec_name, spec, model) in models.items():
            rows = score(model, fm.build_finals(fm.spec_scored(scored, spec, races), spec["features"]))
            block[label] = {"experiment": spec_name, "totals": totals(rows), "finals": rows}
            both[label] += rows
        if key == "ultimate" and os.path.exists(FROZEN_CALL_PATH):
            block["frozenCall"] = frozen_call(fm.build_finals(scored, fm.FEATURES))
        report["competitions"][key] = block
        print_block(block, name)

    report["both"] = {label: totals(rows) for label, rows in both.items()}
    t, r = report["both"]["today"], report["both"]["candidate"]
    print(f"\n=== Both championships: {t['finals']} finals, {t['possible']} podium places ===")
    print(f"    today's model      {t['hits']} medallists, {t['winners']} winners, mean log-likelihood {t['meanLl']}")
    print(f"    {name:<18} {r['hits']} medallists, {r['winners']} winners, mean log-likelihood {r['meanLl']}")
    print(f"    points             {t['pointsHits']} medallists")
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(f"\n  -> {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
