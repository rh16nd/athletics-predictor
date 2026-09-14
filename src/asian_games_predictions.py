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

THE MARK AN ENTRANT IS RANKED ON (asked for by the user on 2026-09-15)
Their best mark in the event this season. An entrant with no mark this season
is ranked on last season's best instead, tagged with its year, and in the 5000m
and 10,000m every entrant takes the better of the two seasons. Both were
measured on past championship finals before they were built: see
last_season_rule.

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
Reads data/asian_games_2026/event.json and asia/{key}_2025.csv
(asian_games_scraper.py) and writes data/asian_games_2026/predictions.json.
"""
import json
import os
import re
import sys
import unicodedata
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
# Events where every entrant takes the better of this season's and last
# season's best. See last_season_rule.
BETTER_OF_TWO = frozenset({"5000m", "10000m"})


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


# ---- the mark each entrant is ranked on ------------------------------------------

def name_keys(name, nat):
    """Ways to meet one athlete on another list, always with their nation: the
    name without punctuation ("Tae-poong NAM" and "Taepoong NAM"), and for two
    words or more the words in any order, since the lists can swap which name
    is the family name."""
    plain = "".join(c for c in unicodedata.normalize("NFD", str(name or "")) if not unicodedata.combining(c))
    keys = [("flat", re.sub(r"[^A-Z0-9]", "", plain.upper()), nat)]
    words = ags._tokens(name)
    if len(words) >= 2:
        keys.append(("words", " ".join(sorted(words)), nat))
    return keys


def last_season_marks(key, asia_dir=None):
    """Last season's Asian list for one discipline, as {"id": {WA id: (score,
    mark)}, "name": {name key: (score, mark)}}, the best row per athlete.
    Empty when the list was never fetched (asian_games_scraper.py --last-season)."""
    path = os.path.join(asia_dir or ags.ASIA_DIR, f"{key}_{ags.LAST_YEAR}.csv")
    by_id, by_name = {}, {}
    if not os.path.exists(path):
        return {"id": by_id, "name": by_name}
    header, rows = ags.read_rows(path)
    comp, mark, score = (header.index(c) for c in ("Competitor", "Mark", "Results Score"))
    for row in rows:
        value = ags._int(row[score])
        if value is None:
            continue
        idents = [(by_id, ags.wa_id(row[-1]))] + [(by_name, k) for k in name_keys(row[comp], row[ags.NAT])]
        for table, ident in idents:
            if ident is not None and (ident not in table or value > table[ident][0]):
                table[ident] = (value, row[mark])
    return {"id": by_id, "name": by_name}


def last_season_rule(athletes, key, last):
    """The entrants, each with the mark they are ranked on.

    `score` and `mark` become that mark, `seasonScore` and `seasonMark` keep
    this season's, and `markSeason` is last season's year when the mark is from
    then (None otherwise). An entrant with no mark this season takes last
    season's best from its Asian list. In BETTER_OF_TWO events every entrant
    takes the better of the two.

    Measured on 1,115 past championship finals before it was built
    (2026-09-15), counting the podium places a top three by points named:
      - on this season's marks alone, 92 medallists had no mark to rank them
        by. Falling back to last season left 7, and named more places: a mean
        +0.013 a final, 90% interval +0.002 to +0.024.
      - In the 10,000m the better of two seasons named 51.3% of places against
        39.3% on this season alone, and in the 5000m 57.9% against 50.9%. A
        distance runner may race the distance once a year.
      - Applied to every event it did worse in the 1500m (47.9% against 52.6%)
        and the javelin (56.2% against 60.9%), so it stops at those two. Those
        two were chosen after seeing every event's numbers.
    An entrant whose lookup failed is left as it is: a failed request says
    nothing about their season."""
    event = str(key).split("_", 1)[-1]
    out = []
    for athlete in athletes:
        a = {**athlete, "seasonScore": athlete.get("score"), "seasonMark": athlete.get("mark"), "markSeason": None}
        if a.get("unranked") == "lookupFailed":
            out.append(a)
            continue
        found = last["id"].get(ags._int(a.get("waId"))) if a.get("waId") else None
        for ident in ([] if found else name_keys(a.get("name"), a.get("nat"))):
            found = last["name"].get(ident)
            if found:
                break
        if found and (a["score"] is None or (event in BETTER_OF_TWO and found[0] > a["score"])):
            a.update({"score": found[0], "mark": found[1], "markSeason": ags.LAST_YEAR, "unranked": None})
        out.append(a)
    return out


def _by_points(athletes):
    """Entrants with a mark to rank them on, best Results Score first. Ties keep
    the Asian list's own order."""
    return sorted((a for a in athletes if a.get("score") is not None),
                  key=lambda a: (-a["score"], a.get("asiaRank") or 10 ** 6))


def choose_method(athletes, known):
    """("model" | "points", the evidence the page shows). An entrant with no
    mark from either season has no Results Score, so cannot be among the top
    eight."""
    top = _by_points(athletes)[:MODEL_OF]
    with_history = [a["name"] for a in top if _key(a["name"]) in known]
    method = "model" if len(with_history) >= MODEL_NEEDED else "points"
    return method, {
        "considered": [a["name"] for a in top],
        "withHistory": with_history,
        "needed": MODEL_NEEDED,
        "of": MODEL_OF,
    }


def unranked_detail(event, names):
    """The entrants a call could not rank, each with the reason, for the rows at
    the foot of the event's table. The reason is the scraper's ("notFound",
    "noMark", "lookupFailed"), or, in a model event, "lastSeasonOnly" for an
    entrant whose only mark is last season's, which the model cannot read, and
    "notScored" for one with a mark this season the model still could not
    score."""
    by_name = {_key(a["name"]): a for a in event["athletes"]}
    out = []
    for name in names:
        a = by_name.get(_key(name), {})
        if a.get("unranked"):
            reason = a["unranked"]
        elif a.get("markSeason") and a.get("seasonScore") is None:
            reason = "lastSeasonOnly"
        elif a.get("score") is not None:
            reason = "notScored"
        else:
            reason = "noMark"
        out.append({"name": name, "nat": a.get("nat"), "reason": reason, "profileUrl": a.get("profileUrl")})
    return out


def points_call(event):
    """The field in Results Score order, in project_event's shape, with no
    podium chance anywhere: a ranking by points states an order, not odds."""
    scored = _by_points(event["athletes"])
    unscored = [a["name"] for a in event["athletes"] if a.get("score") is None]
    return {
        "discKey": event["discKey"],
        "disciplineLabel": event.get("disciplineLabel"),
        "sex": event.get("sex"),
        "places": None,
        "qualified": len(event["athletes"]),
        "scored": len(scored),
        "unscored": unscored,
        "unranked": unranked_detail(event, unscored),
        "athletes": [{"rank": i, "name": a["name"], "nat": a.get("nat"), "qualifiedBy": None,
                      "rankingScore": a["score"], "mark": a.get("mark"), "markSeason": a.get("markSeason"),
                      "podiumChance": None}
                     for i, a in enumerate(scored, 1)],
        "notEntered": [],
        "fieldSource": "entries",
    }


def model_call(event, model, scaler, feature_cols, snapshot_path, project=up.project_event):
    """The Ultimate's projection, run on the merged snapshot. An entrant who is
    not in it (no 2026 mark found) comes back in `unscored`, named. The model
    reads this season only, so last season's marks play no part here."""
    entry = {
        "discKey": event["discKey"],
        "disciplineLabel": event.get("disciplineLabel"),
        "sex": event.get("sex"),
        "places": None,
        "athletes": [{"name": a["name"], "nat": a.get("nat"), "qualifiedBy": None,
                      "rankingScore": a.get("seasonScore", a.get("score"))} for a in event["athletes"]],
        "fieldSource": "entries",
    }
    out = project(entry, model, scaler, feature_cols, snapshot_path=snapshot_path)
    if out is None:
        # The rule gave this event to the model because six of its top eight
        # have history, so there is always someone to score. None means the
        # snapshot is missing, which is a broken run, not a points event.
        raise RuntimeError(f"{event['discKey']}: the model scored nobody; is {snapshot_path} there?")
    marks = {_key(a["name"]): a.get("seasonMark", a.get("mark")) for a in event["athletes"]}
    for athlete in out["athletes"]:
        athlete["mark"] = marks.get(_key(athlete["name"]))
        athlete["markSeason"] = None
    out["unranked"] = unranked_detail(event, out.get("unscored") or [])
    return out


def snapshot_names(key, snapshot_dir=None):
    """Everyone in the discipline's merged snapshot: the world toplist plus the
    entrants added to it. The athletes the site has a page for.

    It was the world toplist alone until 2026-09-14, which left 281 of the 419
    ranked entrants linking out to World Athletics. api.py's athlete pages now
    read this snapshot too, so everyone in it has a page."""
    path = os.path.join(snapshot_dir or ags.SNAPSHOT_DIR, f"{key}_{ags.YEAR}.csv")
    if not os.path.exists(path):
        return set()
    return {_key(n) for n in pd.read_csv(path, usecols=["Competitor"])["Competitor"].dropna()}


def link_athletes(out, event, on_site):
    """Each ranked athlete's World Athletics profile, and whether the site has a
    page for them. An entrant with no row in the snapshot has no page, and is
    linked to World Athletics rather than to a page that cannot load."""
    profiles = {_key(a["name"]): a.get("profileUrl") for a in event["athletes"]}
    for athlete in out["athletes"]:
        athlete["profileUrl"] = profiles.get(_key(athlete["name"]))
        athlete["hasPage"] = _key(athlete["name"]) in on_site
    return out


def build(event, model, scaler, feature_cols, snapshot_dir=None, known_for=history_names,
          project=up.project_event, pages_for=snapshot_names, last_for=last_season_marks):
    snapshot_dir = snapshot_dir or ags.SNAPSHOT_DIR
    projections = []
    for ev in event.get("field") or []:
        key = ev["discKey"]
        ev = {**ev, "athletes": last_season_rule(ev["athletes"], key, last_for(key))}
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
        projections.append(link_athletes(out, ev, pages_for(key)))
    return {
        "rule": {"needed": MODEL_NEEDED, "of": MODEL_OF, "floor": MODEL_FLOOR,
                 "lastSeason": ags.LAST_YEAR, "betterOfTwo": sorted(BETTER_OF_TWO)},
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
    from_last = sum(1 for p in projections for a in p["athletes"] if a.get("markSeason"))
    print(f"  {len(by_method['model'])} events by the model, {len(by_method['points'])} by points, "
          f"{len(payload['notCalled'])} not called; {from_last} athletes ranked on a {ags.LAST_YEAR} mark")
    for p in projections:
        ev = p["methodEvidence"]
        top = ", ".join(
            f"{a['name']} {a['podiumChance']}%" if p["method"] == "model" else f"{a['name']} {a['rankingScore']}"
            for a in p["athletes"][:3])
        last = sum(1 for a in p["athletes"] if a.get("markSeason"))
        print(f"    {p['disciplineLabel']:<30} {p['method']:<6} "
              f"{len(ev['withHistory'])}/{len(ev['considered'])} with history  "
              f"{last:>2} on {ags.LAST_YEAR}  {len(p.get('unranked') or []):>2} unranked  {top}")
    print(f"\n  -> {OUT_PATH}")
