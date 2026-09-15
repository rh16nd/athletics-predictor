"""
asian_games_predictions.py -- the call for each Asian Games event we can call.

THE MODEL (set by the user on 2026-09-15)
Every event is called by the field model (src/field_model.py). It rates each
entrant against the others entered, on what was known before the Games: their
best mark this season, or last season's when they have none this year, their
career best, last season's best, how long ago the best mark was set, and their
age. An athlete with one race this year is read on their record rather than
written off. Each entrant gets a chance of a podium, and a field's chances add
up to 300, one hundred for each medal.

HOW IT TESTED, SAID PLAINLY ON THE PAGE
On 511 held-out championship finals it named 64.9% of the medallists, and a
ranking by World Athletics points named 64.1%: level, not better (+0.025 a
final, 90% interval -0.018 to +0.068). On the Asian finals it was behind, 210
podium places to 217, also inside the noise. The payload's `rule.backtest` is
read from outputs/field_model_report.json, so the page states whatever the
last backtest said.

It replaced, the same day, a call made two ways: the Diamond League model where
6 of an event's top 8 had a record we hold and its favourite a 5% chance, and a
ranking on points everywhere else, which was 28 of the 36 events.

THE MARK SHOWN
The mark the model read each entrant on: this season's best, or last season's
tagged with its year (`markSeason`). An event with fewer than MIN_SCORED
entrants to read has no podium to call, and is ranked on points instead, saying
why.

Usage:
    python src/asian_games_predictions.py
Reads data/asian_games_2026/event.json, raw/{key}_2026.csv and asia/{key}_2025.csv
(asian_games_scraper.py) and outputs/field_model.json (field_model.py), and
writes data/asian_games_2026/predictions.json.
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
import championships  # noqa: E402
import field_model as fm  # noqa: E402

OUT_PATH = os.path.join(ags.OUT_DIR, "predictions.json")
CHAMP = championships.get("asian-games-2026")
# Fewer entrants with a mark to read than this, and there is no podium to call.
MIN_SCORED = 3
# In a points fallback, events where every entrant takes the better of this
# season's and last season's best. See last_season_rule.
BETTER_OF_TWO = frozenset({"5000m", "10000m"})


def _key(name):
    return str(name or "").upper().strip()


# ---- last season ------------------------------------------------------------------

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
    """The entrants, each with the mark a points ranking would rank them on.

    `score` and `mark` become that mark, `seasonScore` and `seasonMark` keep
    this season's, and `markSeason` is last season's year when the mark is from
    then (None otherwise). An entrant with no mark this season takes last
    season's best from its Asian list, which also clears their "no mark"
    reason. In BETTER_OF_TWO events every entrant takes the better of the two.

    Measured on 1,115 past championship finals before it was built
    (2026-09-15), counting the podium places a top three by points named:
      - on this season's marks alone, 92 medallists had no mark to rank them
        by. Falling back to last season left 7, and named more places: a mean
        +0.013 a final, 90% interval +0.002 to +0.024.
      - In the 10,000m the better of two seasons named 51.3% of places against
        39.3% on this season alone, and in the 5000m 57.9% against 50.9%.
      - Applied to every event it did worse in the 1500m (47.9% against 52.6%)
        and the javelin (56.2% against 60.9%), so it stops at those two. Those
        two were chosen after seeing every event's numbers.
    The model reads last season through its own fallback (field_model.
    serving_rows); this rule orders a points fallback and settles who is
    unranked. An entrant whose lookup failed is left as it is: a failed
    request says nothing about their season."""
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


# ---- the call -----------------------------------------------------------------------

def _by_points(athletes):
    """Entrants with a mark to rank them on, best Results Score first. Ties keep
    the Asian list's own order."""
    return sorted((a for a in athletes if a.get("score") is not None),
                  key=lambda a: (-a["score"], a.get("asiaRank") or 10 ** 6))


def unranked_detail(event, names):
    """The entrants a call could not rank, each with the reason, for the rows at
    the foot of the event's table. The reason is the scraper's ("notFound",
    "noMark", "lookupFailed"), or "notScored" for an entrant with a mark the
    model still could not read."""
    by_name = {_key(a["name"]): a for a in event["athletes"]}
    out = []
    for name in names:
        a = by_name.get(_key(name), {})
        reason = a.get("unranked") or ("notScored" if a.get("score") is not None else "noMark")
        out.append({"name": name, "nat": a.get("nat"), "reason": reason, "profileUrl": a.get("profileUrl")})
    return out


def field_call(event, model, snapshot_path, last_path, cutoff, year=ags.YEAR, rows_for=fm.serving_rows):
    """One event called by the field model, in ultimate_predictions.project_event's
    shape, or None when fewer than MIN_SCORED entrants have a mark to read."""
    key = event["discKey"]
    entrants = [{"name": a["name"], "nat": a.get("nat"), "waId": a.get("waId")} for a in event["athletes"]]
    rows = rows_for(key, entrants, snapshot_path, cutoff, year, extra=[(last_path, "asiaLastSeason")])
    rows = rows[rows["sb_score"].notna()].reset_index(drop=True)
    if len(rows) < MIN_SCORED:
        return None
    u = fm.utilities(model, fm.field_features(rows, cutoff).to_numpy())
    rows = (rows.assign(chance=fm.podium_chances(u), win=fm.win_chances(u))
            .sort_values(["chance", "sb_score"], ascending=False, kind="stable"))
    nats = {_key(a["name"]): a.get("nat") for a in event["athletes"]}
    athletes = [{
        "rank": rank, "name": r.athlete_name, "nat": nats.get(_key(r.athlete_name)), "qualifiedBy": None,
        "rankingScore": int(r.sb_score), "mark": r.mark,
        "markSeason": int(r.mark_season) if int(r.mark_season) != year else None,
        "podiumChance": round(float(r.chance) * 100, 1),
        "winChance": round(float(r.win) * 100, 1),
    } for rank, r in enumerate(rows.itertuples(), 1)]
    scored = {_key(a["name"]) for a in athletes}
    unscored = [a["name"] for a in event["athletes"] if _key(a["name"]) not in scored]
    return {
        "discKey": key,
        "disciplineLabel": event.get("disciplineLabel"),
        "sex": event.get("sex"),
        "places": None,
        "qualified": len(event["athletes"]),
        "scored": len(athletes),
        "unscored": unscored,
        "unranked": unranked_detail(event, unscored),
        "athletes": athletes,
        "notEntered": [],
        "fieldSource": "entries",
    }


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
    """Each entrant's World Athletics profile, and whether the site has a page
    for them, for the ranked rows and the unranked rows alike.

    A page needs a row in the snapshot or a World Athletics profile. Since
    2026-09-15 api.py builds a page from the call for an entrant with no row
    this season (a 2025 mark, or no mark at all) and reads their results, photo
    and career from the profile. Before that, 23 ranked entrants and every
    unranked one linked out to World Athletics. An entrant matched to no profile
    has nothing to put on a page and links nowhere."""
    profiles = {_key(a["name"]): a.get("profileUrl") for a in event["athletes"]}
    for athlete in out["athletes"] + (out.get("unranked") or []):
        athlete["profileUrl"] = athlete.get("profileUrl") or profiles.get(_key(athlete["name"]))
        athlete["hasPage"] = _key(athlete["name"]) in on_site or bool(athlete["profileUrl"])
    return out


def backtest_summary(report_path=fm.REPORT_PATH):
    """The field model's test against points, for the page to state, or None
    when there is no report."""
    try:
        with open(report_path, encoding="utf-8") as f:
            report = json.load(f)
    except (OSError, ValueError):
        return None
    overall = report.get("overall") or {}
    asia = (report.get("byTier") or {}).get("asia") or {}
    return {"finals": overall.get("finals"), "model": overall.get("model"), "points": overall.get("points"),
            "asiaFinals": asia.get("finals"), "asiaModel": asia.get("model"), "asiaPoints": asia.get("points"),
            "years": report.get("testYears")}


def build(event, model, snapshot_dir=None, asia_dir=None, cutoff=None, pages_for=snapshot_names,
          last_for=last_season_marks, rows_for=fm.serving_rows, report_path=fm.REPORT_PATH):
    snapshot_dir = snapshot_dir or ags.SNAPSHOT_DIR
    asia_dir = asia_dir or ags.ASIA_DIR
    cutoff = cutoff or CHAMP["startDate"]
    projections = []
    for ev in event.get("field") or []:
        key = ev["discKey"]
        ev = {**ev, "athletes": last_season_rule(ev["athletes"], key, last_for(key))}
        out = field_call(ev, model, os.path.join(snapshot_dir, f"{key}_{ags.YEAR}.csv"),
                         os.path.join(asia_dir, f"{key}_{ags.LAST_YEAR}.csv"), cutoff, rows_for=rows_for)
        if out is None:
            out = points_call(ev)
            out["method"], out["methodEvidence"] = "points", {"reason": "tooFew", "needed": MIN_SCORED}
        else:
            out["method"], out["methodEvidence"] = "model", {"reason": None}
        projections.append(link_athletes(out, ev, pages_for(key)))
    return {
        "rule": {"method": "field", "cutoff": str(cutoff), "backtest": backtest_summary(report_path)},
        "builtAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "projections": projections,
        "notCalled": event.get("notCalled") or [],
    }


if __name__ == "__main__":
    print("=== Calling the Asian Games ===")
    with open(ags.OUT_PATH, encoding="utf-8") as f:
        saved_event = json.load(f)
    field_model = fm.load_model()
    if field_model is None:
        sys.exit(f"  no field model at {fm.MODEL_PATH}; run python src/field_model.py --backtest first")
    payload = build(saved_event, field_model)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)

    projections = payload["projections"]
    by_method = {m: [p for p in projections if p["method"] == m] for m in ("model", "points")}
    from_last = sum(1 for p in projections for a in p["athletes"] if a.get("markSeason"))
    print(f"  {len(by_method['model'])} events by the model, {len(by_method['points'])} on points "
          f"(too few marks), {len(payload['notCalled'])} not called; "
          f"{from_last} athletes read on a {ags.LAST_YEAR} mark")
    for p in projections:
        if p["method"] == "model":
            total = sum(a["podiumChance"] for a in p["athletes"])
            top = ", ".join(f"{a['name']} {a['podiumChance']}%" for a in p["athletes"][:3])
        else:
            total, top = 0, ", ".join(f"{a['name']} {a['rankingScore']}" for a in p["athletes"][:3])
        last = sum(1 for a in p["athletes"] if a.get("markSeason"))
        print(f"    {p['disciplineLabel']:<30} {p['method']:<6} {len(p['athletes']):>3} called  "
              f"sum {total:6.1f}  {last:>2} on {ags.LAST_YEAR}  {len(p.get('unranked') or []):>2} unranked  {top}")
    print(f"\n  -> {OUT_PATH}")
