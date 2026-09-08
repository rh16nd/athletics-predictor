"""
ultimate_predictions.py -- the model's projected podium for every event at the
World Athletics Ultimate Championship, Budapest, 11-13 September 2026.

WHY THIS EXISTS SEPARATELY FROM run.py
run.py projects the Diamond League Final: it takes the athletes in WA's Diamond
League standings and scores them. That Final has been run, the standings are
history, and the site's next event is a championship with a completely
different field decided by completely different rules. Pointing run.py at
Budapest would have meant teaching it a second, unrelated way to choose a
field, inside a script that is already five commands deep in a refresh. This
does the one new thing instead.

THE FIELD IS REAL, NOT PROJECTED
This is the part that matters. WA publishes the Ultimate's qualification list
through their own API -- `getChampionshipQualifications` -- and it says who is
in, how they got in (a wild card for the Olympic, world and Diamond League
champions; World Athletics Rankings for the rest) and what ranking score they
got in on. 344 athletes across 28 events, fetched by
ultimate_scraper.fetch_qualified_field().

So nothing here guesses at who will be on the start line. Everyone scored below
is an athlete World Athletics says has qualified. Where an event is missing it
is missing honestly:
  * the two mixed relays -- the model is built on individual events and has no
    history for a national team, which is a documented refusal, not a gap
  * the men's hammer throw -- the Ultimate contests it, we hold no hammer data
    at all, so there is nothing to score

THE MODEL
The same one the rest of the site uses (outputs/), now trained on 531 real
championship finals rather than 215 Diamond League Finals -- which is what
makes it able to say anything about Budapest at all. Head-to-head is computed
WITHIN the Ultimate field, so it answers "how has this athlete done against
these opponents" rather than against a Diamond League pool none of them may
have met in.

Usage:
    python src/ultimate_predictions.py
Writes data/ultimate/predictions.json.
"""
import json
import os
import pickle
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from feature_builder import build_2026_features  # noqa: E402
import ultimate_scraper as us  # noqa: E402
from world_rankings import add_h2h  # noqa: E402

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUTPUTS_DIR = os.environ.get("PODIUMCALL_MODEL_DIR", os.path.join(BASE_DIR, "outputs"))
OUT_PATH = os.path.join(BASE_DIR, "data", "ultimate", "predictions.json")


def load_model():
    with open(os.path.join(OUTPUTS_DIR, "model_rf.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(OUTPUTS_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(OUTPUTS_DIR, "feature_cols.pkl"), "rb") as f:
        cols = pickle.load(f)
    return model, scaler, cols


def _key(name):
    return str(name).upper().strip()


def project_event(event, model, scaler, feature_cols):
    """One event's projected podium, or None when it cannot be scored honestly.

    Returns the whole qualified field ranked, not just three: the podium is
    what the site leads with, but a reader looking at a 16-strong championship
    field wants to see where everyone else landed."""
    key = event.get("discKey")
    if not key:
        return None

    df = build_2026_features(key)
    if df.empty:
        return None

    entered = {_key(a["name"]): a for a in event["athletes"]}
    df = df[df["athlete_name"].map(lambda n: _key(n) in entered)].copy()
    if df.empty:
        return None

    # h2h across THIS field, which is the whole point of computing it here
    # rather than reusing a Diamond League pool.
    df = add_h2h(df, key)
    df = df.dropna(subset=feature_cols)
    if df.empty:
        return None

    df["prob"] = model.predict_proba(scaler.transform(df[feature_cols]))[:, 1]
    df = df.sort_values("prob", ascending=False)

    ranked = []
    for i, (_, r) in enumerate(df.iterrows(), 1):
        meta = entered[_key(r["athlete_name"])]
        ranked.append({
            "rank": i,
            "name": r["athlete_name"],
            "nat": meta.get("nat"),
            "qualifiedBy": meta.get("qualifiedBy"),
            "rankingScore": meta.get("rankingScore"),
            "podiumChance": round(float(r["prob"]) * 100, 1),
        })

    # Who WA says is in but we could not score, named rather than counted: a
    # missing favourite is the difference between a projection worth reading
    # and one that is quietly wrong.
    scored = {_key(a["name"]) for a in ranked}
    return {
        "discKey": key,
        "disciplineLabel": event.get("disciplineLabel"),
        "sex": event.get("sex"),
        "places": event.get("places"),
        "qualified": len(event["athletes"]),
        "scored": len(ranked),
        "unscored": [a["name"] for a in event["athletes"] if _key(a["name"]) not in scored],
        "athletes": ranked,
        # Qualified for this event and NOT on the entry list. Named rather than
        # silently dropped: several of them were this projection's own top
        # picks yesterday, and a reader who remembers that deserves to be told
        # where they went.
        "notEntered": event.get("notEntered") or [],
        "fieldSource": event.get("fieldSource") or "qualification",
    }


EVENT_PATH = os.path.join(BASE_DIR, "data", "ultimate", "event.json")


def double_entries(field):
    """{athlete key: [other event labels]}, and which of those CANNOT both be
    contested.

    WA's qualification list says who is ELIGIBLE, not who is entered, and 11
    athletes qualified in two events. Listing all of them in both projections
    reads as a claim that they will run both. Most of them plausibly will: a
    sprinter in the 100m and 200m on different evenings is an ordinary
    championship double, and so is 1500m/5000m two days apart.

    One is not. Rai Benjamin is qualified in the 400m and the 400m hurdles, and
    WA's own timetable puts the 400m FINAL and the 400mH SEMI-FINAL in the same
    session on 12 September, 99 minutes apart. Reported by the user, whose
    information was that the previews have him in the flat 400m only.

    The rule is that one: two events sharing a session. It is WA's own
    scheduling, not a judgement about the athlete, and measured across the whole
    field it flags exactly the one case and none of the legitimate doubles.
    Which event he actually runs is NOT decided here -- entries are not
    published (`isStartlistPublished` is false on all 40 phases), so the page
    says what is known and lets the reader draw the conclusion."""
    try:
        with open(EVENT_PATH, encoding="utf-8") as f:
            timetable = json.load(f).get("timetable") or []
    except (OSError, json.JSONDecodeError):
        timetable = []

    sex_word = {"M": "Men", "W": "Women"}
    sessions = {}
    for phase in timetable:
        name = (phase.get("discipline") or {}).get("name")
        stamp = phase.get("phaseDateAndTime") or ""
        if not name or not stamp:
            continue
        slot = (stamp[:10], phase.get("phaseSessionName"))
        sessions.setdefault((phase.get("sexName"), name), set()).add(slot)

    def slots(event):
        label = event.get("disciplineLabel") or ""
        discipline = label.split("'s ", 1)[-1] if "'s " in label else label
        return sessions.get((sex_word.get(event.get("sex")), discipline), set())

    entries = {}
    for event in field:
        for athlete in event.get("athletes") or []:
            entries.setdefault(_key(athlete["name"]), []).append(event)

    out = {}
    for key, events in entries.items():
        if len(events) < 2:
            continue
        out[key] = [
            {"label": event.get("disciplineLabel"),
             "discKey": event.get("discKey"),
             # True when this event shares a session with another of theirs, so
             # contesting both is not a hard double but an impossible one.
             "clashes": any(slots(event) & slots(other)
                            for other in events if other is not event)}
            for event in events
        ]
    return out


def build():
    model, scaler, feature_cols = load_model()
    field = us.fetch_qualified_field()

    # WA's ENTRY list, when it exists, outranks their qualification list.
    # Qualification says who is eligible; entries say who is running, and on
    # 2026-09-08 they disagreed about 19 athletes each way -- including Tara
    # Davis-Woodhall, who was this projection's number one in the women's long
    # jump at 56.6% and is not entered. Falling back to qualification when the
    # entry list is unavailable is right; preferring it when both exist is not.
    entries = us.fetch_entry_lists()
    if entries:
        field = [us.apply_entry_list(e, entries[e["disciplineLabel"]])
                 if e.get("disciplineLabel") in entries else e
                 for e in field]
        swapped = sum(1 for e in field if e.get("fieldSource") == "entries")
        dropped = sum(len(e.get("notEntered") or []) for e in field)
        print(f"  entry list applied to {swapped} events; "
              f"{dropped} qualified athletes are not entered")

    doubles = double_entries(field)
    projections = []
    skipped = []
    for event in field:
        out = project_event(event, model, scaler, feature_cols)
        if out:
            for athlete in out["athletes"]:
                others = [o for o in doubles.get(_key(athlete["name"]), [])
                          if o.get("discKey") != out["discKey"]]
                if others:
                    athlete["alsoQualifiedIn"] = others
            projections.append(out)
        else:
            skipped.append(event.get("disciplineLabel"))
    return projections, skipped


if __name__ == "__main__":
    print("=== Projecting the Ultimate Championship ===")
    projections, skipped = build()
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"projections": projections}, f, ensure_ascii=False, indent=1)

    total_unscored = sum(len(p["unscored"]) for p in projections)
    print(f"  {len(projections)} events projected, {len(skipped)} not scoreable")
    if skipped:
        print(f"    not scoreable: {', '.join(str(s) for s in skipped)}")
    print(f"  {sum(p['scored'] for p in projections)} qualified athletes scored, "
          f"{total_unscored} qualified but unscoreable (no 2026 mark on file)")
    for p in projections[:6]:
        top = ", ".join(f"{a['name']} {a['podiumChance']}%" for a in p["athletes"][:3])
        print(f"    {p['disciplineLabel']:<28} {top}")
    print(f"\n  -> {os.path.abspath(OUT_PATH)}")
