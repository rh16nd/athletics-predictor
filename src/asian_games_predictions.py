"""
asian_games_predictions.py -- the call for each Asian Games event we can call.

THE MODEL (set by the user on 2026-09-15, and replaced that evening)
Every event is called by the field model (src/field_model.py) in its
race-by-race form, the experiment v2_form_best_5. It rates each entrant against
the others entered, on what was known before the Games: this season's form
(the mean of their best five marks), how they have raced in the last six weeks,
podiums in finals at the biggest meetings, their record against the field's
three strongest in finals they shared, and, as before, their season best (last
season's when they have none this year), career best, last season and age.
Each entrant gets a chance of a podium, the field's adding up to 300, and a
chance of winning, adding up to 100.

It reads each entrant's season race by race, fetched into
data/asian_games_2026/races (field_data.fetch_races) before the call is built.
Run with --refresh-races before the freeze, so the meetings since the last
fetch count.

HOW IT TESTED, SAID PLAINLY ON THE PAGE
Chosen from 23 candidates on the finals of 2021-2023, then tested once on the
206 finals of 2024-2025, kept aside while it was built: 64.4% of the medallists
against 62.9% for the model it replaced and 62.0% for points. On the 36 Asian
finals, 67.6% against 65.7%, though points named 70.4% there. The payload's
`rule.backtest` is read from outputs/field_model_holdout.json when the served
model is the one that report tested (backtest_summary).

Earlier on 2026-09-15 the field model read season bests alone and tested level
with points (511 finals, 64.9% against 64.1%). Before that, the Diamond League
model called the 8 events where 6 of the top 8 had a record we hold, and points
the other 28.

THE MARK SHOWN
The mark the model read each entrant on: this season's best, or last season's
tagged with its year (`markSeason`). An event with fewer than MIN_SCORED
entrants to read has no podium to call, and is ranked on points instead, saying
why.

Usage:
    python src/asian_games_predictions.py [--refresh-races]
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
import field_data as fd  # noqa: E402
import field_model as fm  # noqa: E402

OUT_PATH = os.path.join(ags.OUT_DIR, "predictions.json")
RACES_DIR = os.path.join(ags.OUT_DIR, "races")
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


def field_call(event, model, snapshot_path, last_path, cutoff, year=ags.YEAR, rows_for=fm.serving_rows,
               races=None):
    """One event called by the field model, in ultimate_predictions.project_event's
    shape, or None when fewer than MIN_SCORED entrants have a mark to read.

    `races` is {World Athletics id: this season race by race}, for a model that
    reads races (field_model.needs_races), read the way its experiment reads them."""
    key = event["discKey"]
    entrants = [{"name": a["name"], "nat": a.get("nat"), "waId": a.get("waId")} for a in event["athletes"]]
    how = {"races": races, "race_how": fm.spec_of(model).get("race")} if races is not None else {}
    rows = rows_for(key, entrants, snapshot_path, cutoff, year, extra=[(last_path, "asiaLastSeason")], **how)
    rows = rows[rows["sb_score"].notna()].reset_index(drop=True)
    if len(rows) < MIN_SCORED:
        return None
    features = fm.field_features(rows, cutoff, model.get("features") or fm.FEATURES)
    u = fm.utilities(fm.model_for(model, fm.group_of(key)), features.to_numpy())
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


def _read_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def backtest_summary(report_path=fm.REPORT_PATH, model=None, holdout_path=fm.HOLDOUT_PATH,
                     all_seasons_path=fm.ALL_SEASONS_PATH):
    """How the served field model tested, for the page to state, or None.

    First, when the served model is the version field_model.py --all-seasons
    kept, that test (`method` "allSeasons"): every past season, each called only
    from the seasons before it, and how many versions it was chosen among. Then,
    for a model chosen as an experiment and tested once on the locked years
    (field_model.py --holdout), that test: against the model it replaced
    (`previous`) and against points, on finals kept aside while it was built.
    Only when the report tested this model; otherwise the older backtest
    against points in `report_path`."""
    served = (model or {}).get("experiment")
    seasons = _read_json(all_seasons_path)
    if seasons and served and seasons.get("chosen") == served:
        row = next((r for r in seasons.get("candidates") or [] if r.get("name") == served), None)
        if row:
            return {"method": "allSeasons", "finals": row["finals"], "model": row["medallistsPct"],
                    "points": row["pointsPct"], "asiaFinals": row.get("asiaFinals"),
                    "asiaModel": row.get("asiaPct"), "asiaPoints": row.get("asiaPointsPct"),
                    "years": seasons.get("years"), "versions": len(seasons.get("candidates") or [])}
    held = _read_json(holdout_path)
    if held and (model or {}).get("experiment") and held.get("candidateName") == model["experiment"]:
        overall, asia = held["overall"], (held.get("byTier") or {}).get("asia") or {}
        return {"finals": overall["candidate"]["finals"], "model": overall["candidate"]["model"],
                "previous": overall["baseline"]["model"], "points": overall["baseline"]["points"],
                "asiaFinals": (asia.get("candidate") or {}).get("finals"),
                "asiaModel": (asia.get("candidate") or {}).get("model"),
                "asiaPrevious": (asia.get("baseline") or {}).get("model"),
                "asiaPoints": (asia.get("baseline") or {}).get("points"),
                "years": held.get("testYears")}
    report = _read_json(report_path)
    if report is None:
        return None
    overall = report.get("overall") or {}
    asia = (report.get("byTier") or {}).get("asia") or {}
    return {"finals": overall.get("finals"), "model": overall.get("model"), "points": overall.get("points"),
            "asiaFinals": asia.get("finals"), "asiaModel": asia.get("model"), "asiaPoints": asia.get("points"),
            "years": report.get("testYears")}


def build(event, model, snapshot_dir=None, asia_dir=None, cutoff=None, pages_for=snapshot_names,
          last_for=last_season_marks, rows_for=fm.serving_rows, report_path=fm.REPORT_PATH, races=None,
          holdout_path=fm.HOLDOUT_PATH, all_seasons_path=fm.ALL_SEASONS_PATH):
    snapshot_dir = snapshot_dir or ags.SNAPSHOT_DIR
    asia_dir = asia_dir or ags.ASIA_DIR
    cutoff = cutoff or CHAMP["startDate"]
    projections = []
    for ev in event.get("field") or []:
        key = ev["discKey"]
        ev = {**ev, "athletes": last_season_rule(ev["athletes"], key, last_for(key))}
        out = field_call(ev, model, os.path.join(snapshot_dir, f"{key}_{ags.YEAR}.csv"),
                         os.path.join(asia_dir, f"{key}_{ags.LAST_YEAR}.csv"), cutoff, rows_for=rows_for,
                         races=races)
        if out is None:
            out = points_call(ev)
            out["method"], out["methodEvidence"] = "points", {"reason": "tooFew", "needed": MIN_SCORED}
        else:
            out["method"], out["methodEvidence"] = "model", {"reason": None}
        projections.append(link_athletes(out, ev, pages_for(key)))
    return {
        "rule": {"method": "field", "cutoff": str(cutoff),
                 "backtest": backtest_summary(report_path, model, holdout_path, all_seasons_path)},
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
    races = None
    if fm.needs_races(field_model):
        wanted = {(int(i), ags.YEAR) for ev in saved_event.get("field") or [] for a in ev["athletes"]
                  for i in [a.get("waId") or ags.wa_id(a.get("profileUrl"))] if str(i or "").isdigit()}
        print(f"  The model reads races: {len(wanted)} entrants' {ags.YEAR} seasons in {RACES_DIR}")
        if fd.fetch_seasons(wanted, fetch=fd.fetch_races, seasons_dir=RACES_DIR,
                            refresh="--refresh-races" in sys.argv):
            sys.exit("  some seasons failed to download: run again, since a call missing races is a weaker call")
        races = fd.load_seasons(RACES_DIR).get(ags.YEAR, {})
    payload = build(saved_event, field_model, races=races)
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
