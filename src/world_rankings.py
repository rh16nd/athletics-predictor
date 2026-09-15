"""
world_rankings.py -- per-discipline "best in the world" lists for the Track and
Field pages, in two orderings the UI toggles between:

  points : World Athletics performance points (the Results Score of each
           athlete's season best) -- an objective ranking by what they've run.
  model  : the trained model's own view. The model rates podium probability
           WITHIN a field, so this is computed by scoring the whole discipline
           pool with the same model+features run.py uses and ranking by that
           probability.

           It is NOT a ranking of who is strongest in the world, and the UI
           must not present it as one. The model was trained on Diamond League
           Final podiums and its form features are all computed from
           DL-circuit data, so an athlete who skipped the circuit rates near
           zero however fast they have run. Measured over the 32 model top-20s
           (2026-09-07): mean rating rises 1.2% -> 28.8% going from 0 to 5 DL
           meetings while mean WA score barely moves (1172 -> 1231), and the
           rating correlates better with meetings contested (Spearman 0.75)
           than with the marks themselves (0.65). Hence dlRaces below: the
           count is shipped next to the rating so a low number explains
           itself instead of looking like a verdict on the athlete.

           The hammer and the 10,000m are not Diamond League events, so that
           model has never seen them. Their model view comes from the field
           model instead (src/field_model.py, 2026-09-15): each of the top 20
           by points gets a podium chance, as if the 20 met in one final.
           `modelKind` says which model made a list.

Both are built from data we already scrape (the season toplists), so this runs
in the normal refresh with no new sources. Writes data/world_rankings.json:
  { "<disc_key>": { "isField": bool, "modelAvailable": bool, "modelKind": str,
                    "model": [rows...], "points": [rows...] } }
each row: { rank, name, nat, mark, score, ratingPct, dlRaces,
            racesOnRecord, profileUrl }.

Usage:
    python src/world_rankings.py
    python src/world_rankings.py --only men_HT women_HT   # rewrite just these
"""
import argparse
import glob
import json
import os
import pickle
import sys
from datetime import date

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import asian_games_scraper as ags  # noqa: E402
import field_model as fm  # noqa: E402
from feature_builder import FIELD_EVENTS, POINTS_ONLY_DISCIPLINES, RAW_DIR, build_2026_features  # noqa: E402
from season_activity import races_on_record  # noqa: E402

# Both overridable so an experimental model can be driven through the real
# site without displacing the deployed one. A pooled model written over
# outputs/ would silently become the live model at the next refresh, which is
# not a thing to discover afterwards.
#   PODIUMCALL_MODEL_DIR     where to load model_rf/scaler/feature_cols from
#   PODIUMCALL_RANKINGS_OUT  where to write the rankings JSON
OUTPUTS_DIR = os.environ.get(
    "PODIUMCALL_MODEL_DIR", os.path.join(os.path.dirname(__file__), "..", "outputs"))
OUT_PATH = os.environ.get(
    "PODIUMCALL_RANKINGS_OUT",
    os.path.join(os.path.dirname(__file__), "..", "data", "world_rankings.json"))
YEAR = 2026
TOP_N = 20

with open(os.path.join(OUTPUTS_DIR, "model_rf.pkl"), "rb") as f:
    MODEL = pickle.load(f)
with open(os.path.join(OUTPUTS_DIR, "scaler.pkl"), "rb") as f:
    SCALER = pickle.load(f)
with open(os.path.join(OUTPUTS_DIR, "feature_cols.pkl"), "rb") as f:
    FEAT_COLS = pickle.load(f)


def discipline_keys():
    keys = []
    for path in glob.glob(os.path.join(RAW_DIR, f"*_{YEAR}.csv")):
        base = os.path.basename(path)[: -len(f"_{YEAR}.csv")]
        keys.append(base)
    return sorted(keys)


def toplist_meta(key):
    """name -> {mark, nat, score, url} from the raw toplist, taking each
    athlete's best (first) row. Nationality has a blank header in WA's export,
    so it is read by position (column 5), same as everywhere else."""
    path = os.path.join(RAW_DIR, f"{key}_{YEAR}.csv")
    raw = pd.read_csv(path).dropna(subset=["Competitor"])
    meta = {}
    for _, r in raw.iterrows():
        name = r["Competitor"]
        if name in meta:
            continue
        try:
            score = int(r["Results Score"]) if pd.notna(r["Results Score"]) else None
        except (ValueError, TypeError):
            score = None
        meta[name] = {
            "mark": str(r["Mark"]) if pd.notna(r["Mark"]) else None,
            "nat": str(r.iloc[5]) if pd.notna(r.iloc[5]) else None,
            "score": score,
            "url": str(r["ProfileURL"]) if pd.notna(r["ProfileURL"]) else None,
        }
    return meta


def add_h2h(df, key):
    """h2h_win_rate as a model input, computed over this pool -- the same
    case-insensitive lookup run.py uses (h2h_rates.csv is normal-case, the
    toplist is ALL-CAPS-surname)."""
    h2h_path = os.path.join(os.path.dirname(__file__), "..", "data", "h2h", "h2h_rates.csv")
    if not os.path.exists(h2h_path):
        df["h2h_win_rate"] = 0.5
        return df
    h2h = pd.read_csv(h2h_path)
    disc = h2h[(h2h["discipline"] == key) & (h2h["meetings"] >= 2)].copy()
    lookup = {}
    for _, r in disc.iterrows():
        lookup.setdefault(str(r["athlete_a"]).lower(), {})[str(r["athlete_b"]).lower()] = r["win_rate"]
    names_lower = df["athlete_name"].str.lower().tolist()

    def rate(n):
        rates = [lookup[n][o] for o in names_lower if o != n and n in lookup and o in lookup[n]]
        return sum(rates) / len(rates) if rates else 0.5

    df["h2h_win_rate"] = [rate(n) for n in names_lower]
    return df


def has_meetings_log(key):
    """Whether this discipline has a real per-meeting file. Without one,
    feature_builder falls back to the toplist's row count, which is
    structurally 1 for every athlete -- a number that says nothing. dlRaces
    is None in that case rather than a confident-looking 1."""
    return os.path.exists(os.path.join(RAW_DIR, f"{key}_{YEAR}_meetings.csv"))


def ranked(rows, sort_key, reverse=True):
    """The top TOP_N rows by one key, best first, with missing values last."""
    ordered = sorted(rows, key=lambda x: (x[sort_key] is None, x[sort_key] or 0), reverse=reverse)
    # None-last even when reverse=True: push missing values to the bottom
    ordered = [x for x in ordered if x[sort_key] is not None] + [x for x in ordered if x[sort_key] is None]
    out = []
    for i, x in enumerate(ordered[:TOP_N], 1):
        out.append({
            "rank": i, "name": x["name"], "nat": x["nat"], "mark": x["mark"],
            "score": x["score"], "ratingPct": x["ratingPct"],
            "dlRaces": x["dlRaces"], "racesOnRecord": x["racesOnRecord"],
            "profileUrl": x["profileUrl"],
        })
    return out


def field_model_rows(key, rows, today=None, model=None):
    """The field model's view of one event: its top 20 by points, each with a
    podium chance as if the 20 met in one final, best first. Empty without a
    saved field model, or with fewer than three athletes it can read.

    `ratingPct` is that chance, so the 20 add up to 300: one hundred for each
    podium place. The model reads each athlete's season best, career best, last
    season and age against the rest of the field (field_model.FEATURES). In its
    backtest it named about as many medallists as a points ranking, not more,
    and the page says so."""
    model = model if model is not None else fm.load_model()
    if not model:
        return []
    field = ranked(rows, "score")
    by_name = {r["name"]: r for r in rows}
    # The end of today, so a mark set today still counts as this season's.
    cutoff = pd.Timestamp(today or date.today()) + pd.Timedelta(days=1)
    athletes = [{"name": r["name"], "nat": r["nat"], "waId": ags.wa_id(r["profileUrl"])} for r in field]
    served = fm.serving_rows(key, athletes, os.path.join(RAW_DIR, f"{key}_{YEAR}.csv"), cutoff, YEAR)
    chances = fm.score_field(model, served, cutoff)
    scored = [{**by_name[name], "ratingPct": round(chance * 100, 1), "prob": chance}
              for name, chance in chances.items() if name in by_name]
    return ranked(scored, "prob")


def field_model_discipline(key, today=None, model=None):
    """An event the Diamond League model has never seen: the hammer and the
    10,000m, added 2026-09-14. Ranked on points, with the field model's view
    beside it since 2026-09-15.

    The Diamond League model is still never run on them. It would score athletes
    with no meetings log and no history, every one on the same defaults, and the
    order that came out would be the toplist's own order with a percentage
    beside it. `modelAvailable` is false only when the field model has nothing
    to say."""
    meta = toplist_meta(key)
    if not meta:
        return None
    activity = races_on_record(key, YEAR)
    rows = [{
        "name": name, "nat": m["nat"], "mark": m["mark"], "score": m["score"],
        "ratingPct": None, "dlRaces": None,
        "racesOnRecord": activity.get(str(name).upper().strip()),
        "profileUrl": m["url"],
    } for name, m in meta.items()]
    field = field_model_rows(key, rows, today, model)
    return {
        "isField": key in FIELD_EVENTS,
        "modelAvailable": bool(field),
        "modelKind": "field" if field else None,
        "model": field,
        "points": ranked(rows, "score"),
    }


def score_discipline(key):
    if key in POINTS_ONLY_DISCIPLINES:
        return field_model_discipline(key)
    df = build_2026_features(key)
    if df.empty:
        return None
    df = add_h2h(df, key)
    df = df.dropna(subset=FEAT_COLS)
    if df.empty:
        return None
    df["prob"] = MODEL.predict_proba(SCALER.transform(df[FEAT_COLS]))[:, 1]

    real_meets = has_meetings_log(key)
    # How many times we can SEE this athlete contest the discipline, across
    # the toplist, the Diamond League per-meeting log and the worldwide race
    # log. A floor rather than a census -- see season_activity.py.
    activity = races_on_record(key, YEAR)
    meta = toplist_meta(key)
    rows = []
    for _, r in df.iterrows():
        m = meta.get(r["athlete_name"], {})
        rows.append({
            "name": r["athlete_name"],
            "nat": m.get("nat"),
            "mark": m.get("mark"),
            "score": m.get("score"),
            "ratingPct": round(float(r["prob"]) * 100, 1),
            "prob": float(r["prob"]),
            # Diamond League meetings contested. Kept in the payload because
            # it is what the model's meets_count feature actually reads, but
            # NOT what the site shows: the Diamond League contests each
            # discipline at only a few of its meetings, so a 0 here means "no
            # Diamond League 400mH", not "did not run".
            "dlRaces": int(r["meets_count"]) if real_meets else None,
            # What the site shows instead: every race we can see, from all
            # three sources. Rai Benjamin reads 0 above and 1 here, and the
            # second number is the one that tells a reader his third place is
            # built on a single run.
            "racesOnRecord": activity.get(str(r["athlete_name"]).upper().strip()),
            "profileUrl": m.get("url"),
        })

    return {
        "isField": key in FIELD_EVENTS,
        "modelAvailable": True,
        "modelKind": "form",
        "model": ranked(rows, "prob"),
        "points": ranked(rows, "score"),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Per-discipline top-20 lists for the Track and Field pages.")
    parser.add_argument("--only", nargs="+", metavar="KEY",
                        help="rebuild just these disciplines and keep the rest of the saved file as it is")
    args = parser.parse_args(argv)

    print("=== Building world rankings (points + model) per discipline ===")
    out = {}
    if args.only:
        try:
            with open(OUT_PATH, encoding="utf-8") as f:
                out = json.load(f)
        except (OSError, ValueError):
            out = {}
    for key in args.only or discipline_keys():
        res = score_discipline(key)
        if res:
            out[key] = res
            model_top = res["model"][0]["name"] if res["model"] else "(points only)"
            print(f"  {key}: model#1={model_top} ({res.get('modelKind')})  points#1={res['points'][0]['name']}")
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"\n  {len(out)} disciplines -> {os.path.abspath(OUT_PATH)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
