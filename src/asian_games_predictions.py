"""
asian_games_predictions.py -- the call for each Asian Games event we can call,
made one way per event.

THE RULE (set by the user on 2026-09-14)
An event is called by the model when at least 6 of its top 8 entrants by World
Athletics Results Score have real history with us: a row in data/raw/{key}.csv
(toplists and championship finals back to 2008) or in
data/raw/{key}_2026_meetings.csv (this season's Diamond League meetings).
Otherwise the whole event is ranked by Results Score.

THE FLOOR (added by the user the same day)
An event that passes on history still goes to points when the model's own
favourite has less than a MODEL_FLOOR podium chance. With the history rule
alone, 5 of the 12 model events had a favourite under 1.5%: the women's triple
jump read 0.4%, 0.3% and 0.2%. The model learned on world-class finals and rates
most of an Asian field as long shots, so numbers that small are an order with
nothing to say about who is likely to medal.

Why a rule: the model reads form from data we hold, and we hold little on most
Asian athletes. One with no history is scored on defaults (no meetings,
head-to-head 0.5), which is not a view of them. When most of an event's
contenders are like that, a model ranking would be defaults sorted against
defaults, and World Athletics' own scoring is the honest call.

Why never mixed: a podium chance and a points ranking are different kinds of
statement. In one list, a 41% would sit beside a 1190 and invite a comparison
neither supports. Each event says which one it is.

Usage:
    python src/asian_games_predictions.py
Reads data/asian_games_2026/event.json (asian_games_scraper.py) and writes
data/asian_games_2026/predictions.json.
"""
import json
import os
import sys
from datetime import datetime, timezone

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import asian_games_scraper as ags  # noqa: E402
import ultimate_predictions as up  # noqa: E402

OUT_PATH = os.path.join(ags.OUT_DIR, "predictions.json")
MODEL_OF = 8
MODEL_NEEDED = 6
# Percent. See THE FLOOR above.
MODEL_FLOOR = 5.0


def _key(name):
    return str(name or "").upper().strip()


def history_names(key, raw_dir=None):
    """Everyone with a row in the discipline's history or this season's
    meetings log, as upper-cased toplist names."""
    raw_dir = raw_dir or ags.WORLD_RAW_DIR
    names = set()
    for filename in (f"{key}.csv", f"{key}_{ags.YEAR}_meetings.csv"):
        path = os.path.join(raw_dir, filename)
        if os.path.exists(path):
            names.update(_key(n) for n in pd.read_csv(path, usecols=["Competitor"])["Competitor"].dropna())
    return names


def _by_points(athletes):
    """Entrants with a 2026 mark, best Results Score first. Ties keep the Asian
    list's own order."""
    return sorted((a for a in athletes if a.get("score") is not None),
                  key=lambda a: (-a["score"], a.get("asiaRank") or 10 ** 6))


def choose_method(athletes, known):
    """("model" | "points", the evidence the page shows). An entrant with no
    2026 mark has no Results Score, so cannot be among the top eight."""
    top = _by_points(athletes)[:MODEL_OF]
    with_history = [a["name"] for a in top if _key(a["name"]) in known]
    method = "model" if len(with_history) >= MODEL_NEEDED else "points"
    return method, {
        "considered": [a["name"] for a in top],
        "withHistory": with_history,
        "needed": MODEL_NEEDED,
        "of": MODEL_OF,
    }


def points_call(event):
    """The field in Results Score order, in project_event's shape, with no
    podium chance anywhere: a ranking by points states an order, not odds."""
    scored = _by_points(event["athletes"])
    return {
        "discKey": event["discKey"],
        "disciplineLabel": event.get("disciplineLabel"),
        "sex": event.get("sex"),
        "places": None,
        "qualified": len(event["athletes"]),
        "scored": len(scored),
        "unscored": [a["name"] for a in event["athletes"] if a.get("score") is None],
        "athletes": [{"rank": i, "name": a["name"], "nat": a.get("nat"), "qualifiedBy": None,
                      "rankingScore": a["score"], "mark": a.get("mark"), "podiumChance": None}
                     for i, a in enumerate(scored, 1)],
        "notEntered": [],
        "fieldSource": "entries",
    }


def model_call(event, model, scaler, feature_cols, snapshot_path, project=up.project_event):
    """The Ultimate's projection, run on the merged snapshot. An entrant who is
    not in it (no 2026 mark found) comes back in `unscored`, named."""
    entry = {
        "discKey": event["discKey"],
        "disciplineLabel": event.get("disciplineLabel"),
        "sex": event.get("sex"),
        "places": None,
        "athletes": [{"name": a["name"], "nat": a.get("nat"), "qualifiedBy": None,
                      "rankingScore": a.get("score")} for a in event["athletes"]],
        "fieldSource": "entries",
    }
    out = project(entry, model, scaler, feature_cols, snapshot_path=snapshot_path)
    if out is None:
        # The rule gave this event to the model because six of its top eight
        # have history, so there is always someone to score. None means the
        # snapshot is missing, which is a broken run, not a points event.
        raise RuntimeError(f"{event['discKey']}: the model scored nobody; is {snapshot_path} there?")
    marks = {_key(a["name"]): a.get("mark") for a in event["athletes"]}
    for athlete in out["athletes"]:
        athlete["mark"] = marks.get(_key(athlete["name"]))
    return out


def world_toplist_names(key, raw_dir=None):
    """Everyone on the discipline's world toplist, the athletes the site has a
    page for."""
    path = os.path.join(raw_dir or ags.WORLD_RAW_DIR, f"{key}_{ags.YEAR}.csv")
    if not os.path.exists(path):
        return set()
    return {_key(n) for n in pd.read_csv(path, usecols=["Competitor"])["Competitor"].dropna()}


def link_athletes(out, event, on_site):
    """Each ranked athlete's World Athletics profile, and whether the site has a
    page for them. Athlete pages are built from the world toplists, so an Asian
    entrant who is on none would otherwise be linked to a page that cannot load."""
    profiles = {_key(a["name"]): a.get("profileUrl") for a in event["athletes"]}
    for athlete in out["athletes"]:
        athlete["profileUrl"] = profiles.get(_key(athlete["name"]))
        athlete["hasPage"] = _key(athlete["name"]) in on_site
    return out


def build(event, model, scaler, feature_cols, snapshot_dir=None, known_for=history_names,
          project=up.project_event, world_for=world_toplist_names):
    snapshot_dir = snapshot_dir or ags.SNAPSHOT_DIR
    projections = []
    for ev in event.get("field") or []:
        key = ev["discKey"]
        method, evidence = choose_method(ev["athletes"], known_for(key))
        evidence["reason"] = None if method == "model" else "history"
        if method == "model":
            out = model_call(ev, model, scaler, feature_cols,
                             os.path.join(snapshot_dir, f"{key}_{ags.YEAR}.csv"), project)
            evidence["topChance"] = max((a["podiumChance"] for a in out["athletes"]), default=0.0)
            evidence["floor"] = MODEL_FLOOR
            if evidence["topChance"] < MODEL_FLOOR:
                method, evidence["reason"] = "points", "floor"
                out = points_call(ev)
        else:
            out = points_call(ev)
        out["method"] = method
        out["methodEvidence"] = evidence
        projections.append(link_athletes(out, ev, world_for(key)))
    return {
        "rule": {"needed": MODEL_NEEDED, "of": MODEL_OF, "floor": MODEL_FLOOR},
        "builtAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "projections": projections,
        "notCalled": event.get("notCalled") or [],
    }


if __name__ == "__main__":
    print("=== Calling the Asian Games ===")
    with open(ags.OUT_PATH, encoding="utf-8") as f:
        saved_event = json.load(f)
    model, scaler, cols = up.load_model()
    payload = build(saved_event, model, scaler, cols)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)

    projections = payload["projections"]
    by_method = {m: [p for p in projections if p["method"] == m] for m in ("model", "points")}
    print(f"  {len(by_method['model'])} events by the model, {len(by_method['points'])} by points, "
          f"{len(payload['notCalled'])} not called")
    for p in projections:
        ev = p["methodEvidence"]
        top = ", ".join(
            f"{a['name']} {a['podiumChance']}%" if p["method"] == "model" else f"{a['name']} {a['rankingScore']}"
            for a in p["athletes"][:3])
        print(f"    {p['disciplineLabel']:<30} {p['method']:<6} "
              f"{len(ev['withHistory'])}/{len(ev['considered'])} with history  {top}")
    print(f"\n  -> {OUT_PATH}")
