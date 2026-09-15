"""
field_model_check_2026.py -- today's field model and the race-by-race one on the
2026 World Athletics Ultimate Championship, which neither was fitted on.

Reported beside field_model.compare(), and not part of its rule: 25 finals are
too few to decide anything, and the rule was fixed before either number
existed. The Ultimate's finals are read the way field_data reads every past
final: ids from the competition's own results feed, season bests dated before
its first day (the toplist, then the athlete's profile, then last season),
and races before that day. So an athlete whose season best came in Budapest
is read on their best before it, not sent to last season, which is what a
toplist refreshed after the Ultimate did to the first look at these finals
on 2026-09-15.

Both feature sets are fitted on every 2009-2025 final in data/field/finals.csv,
and the points ranking and the old Diamond League model's frozen call are
scored beside them.

Usage, after field_data.py --races and --report:
    python src/field_model_check_2026.py
Fetches the Ultimate finalists' 2026 seasons into data/field/races_2026/ and
writes outputs/field_model_check_2026.json.
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
COMPETITION = "World Athletics Ultimate Championship"
EVENT_PATH = os.path.join(fd.BASE_DIR, "data", "ultimate", "event.json")
FROZEN_CALL_PATH = os.path.join(fd.BASE_DIR, "data", "ultimate", "predictions_prefinal.json")
RACES_DIR = os.path.join(fd.FIELD_DIR, "races_2026")
OUT_PATH = os.path.join(fd.BASE_DIR, "outputs", "field_model_check_2026.json")


def ultimate_finals(event):
    """The Ultimate's results as finals rows, cut off at its first day."""
    results = pd.DataFrame(event.get("results") or [])
    return pd.DataFrame({
        "competition": COMPETITION, "competition_id": event["competitionId"], "tier": "global",
        "year": YEAR, "cutoff": pd.Timestamp(event["startDate"]),
        "discipline": results["discipline"], "athlete_name": results["athlete_name"],
        "nationality": results["nationality"], "place": pd.to_numeric(results["place"], errors="coerce"),
        "mark": results["mark"],
    })


def this_season_scores(key):
    """Toplist history plus this season's world list, which season_scores
    leaves out because every past final had its season in the history."""
    return fd.season_scores(key, extra=[(os.path.join(fd.RAW_DIR, f"{key}_{YEAR}.csv"), "world")])


def score(model, finals):
    """Per final: medallists named, winner named first, the favourite and how
    sure the model was of them, and the top-three log-likelihood."""
    out = []
    for f in finals:
        u = fm.utilities(model, f["X"])
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


def main():
    with open(EVENT_PATH, encoding="utf-8") as f:
        event = json.load(f)
    finals = ultimate_finals(event)
    print(f"=== The {YEAR} Ultimate: {finals['discipline'].nunique()} finals, {len(finals)} finalists ===")
    ids = fd.finalist_ids(finals)
    finals = finals.merge(ids, on=["competition", "year", "discipline", "athlete_name"], how="left")
    fd.fetch_seasons({(a, YEAR) for a in finals["athlete_id"].dropna()}, fetch=fd.fetch_races, seasons_dir=RACES_DIR)
    races = fd.load_seasons(RACES_DIR)
    scored = fd.attach_races(fd.attach_scores(finals, scores_for=this_season_scores, seasons=races), races)
    print("  Where each season best came from: "
          + ", ".join(f"{k} {v}" for k, v in scored["sb_source"].fillna("unscored").value_counts().items()))

    history = fm.load_scored_finals()
    missing = [c for c in fd.RACE_COLUMNS if c not in history.columns]
    if missing:
        print(f"  {fd.FINALS_PATH} has no {', '.join(missing)}: run field_data.py --races, then --report")
        return 1
    report = {"builtAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
              "competition": COMPETITION, "cutoff": event["startDate"]}
    for label, features in (("today", fm.FEATURES), ("raceByRace", fm.FEATURES_V2)):
        model = fm.fit(fm.build_finals(history, features))
        rows = score(model, fm.build_finals(scored, features))
        report[label] = {"totals": totals(rows), "finals": rows}
    report["frozenCall"] = frozen_call(fm.build_finals(scored, fm.FEATURES))

    t, r = report["today"]["totals"], report["raceByRace"]["totals"]
    print(f"\n  {t['finals']} finals, {t['possible']} podium places")
    print(f"    today's features   {t['hits']} medallists, {t['winners']} winners, mean log-likelihood {t['meanLl']}")
    print(f"    race by race       {r['hits']} medallists, {r['winners']} winners, mean log-likelihood {r['meanLl']}")
    print(f"    points             {t['pointsHits']} medallists")
    print(f"    old frozen call    {report['frozenCall']['hits']} medallists, {report['frozenCall']['winners']} winners")
    print("\n  event            winner named first (today | race by race), favourite's win chance")
    for a, b in zip(report["today"]["finals"], report["raceByRace"]["finals"]):
        print(f"    {a['discipline']:<14} {a['favourite'][:22]:<22} {'Y' if a['winner'] else 'n'} {a['favouriteWin']:.2f}"
              f"  |  {b['favourite'][:22]:<22} {'Y' if b['winner'] else 'n'} {b['favouriteWin']:.2f}")
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(f"\n  -> {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
