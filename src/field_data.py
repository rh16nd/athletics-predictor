"""
field_data.py -- the finals and the season scores the field model learns from.

WHY A SEPARATE PIPELINE
The podium model in train_model.py learned from the world top 100 and the
Diamond League. Asked about an Asian Games field, it reads most entrants as
world long shots, their chances come out under 1%, and on 2026-09-14 the Asian
Games call sent 28 of 36 events to a plain points ranking. The field model
(src/field_model.py) rates each athlete against the rest of their own final
instead. It needs two things, and this file builds both:

  1. FINALS, with everyone who contested them and where they placed: Diamond
     League Finals from disk, and the Olympics, Worlds and Europeans
     (2009-2025), the Asian Games (2018, 2023) and the Asian Championships
     (2019, 2023, 2025) read from World Athletics. The championships are read
     live rather than from the major_meet rows in data/raw, because those keep
     only athletes who were once in a world top 100, which is most of an Asian
     final's field gone.
  2. SEASON SCORES: each athlete's best Results Score in each season, with its
     date, from the world toplists (top 100, data/raw) and the Asian area
     toplists (up to 300 deep, 2015-2025, data/field/asia).

Nothing here writes to data/raw, so the existing model never sees these rows.

THE CUT-OFF
Every final is paired with the day its competition STARTED, and only marks
dated strictly before it count. A World Championships final is on day 8; a
season best set in its heat on day 6 is not something anyone knew before the
championship began, and the Asian Games call is frozen before its first day.

Usage:
    python src/field_data.py --asia-toplists [--only k1,k2] [--refresh]
    python src/field_data.py --finals          # championships, world and Asian
    python src/field_data.py --report          # join to season scores, print coverage
"""
import argparse
import csv
import os
import re
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import asian_games_scraper as ags  # noqa: E402
import dl_final_results_scraper as dlr  # noqa: E402
import final_labels  # noqa: E402
import live_fetcher  # noqa: E402
import ultimate_scraper as us  # noqa: E402

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
FIELD_DIR = os.path.join(BASE_DIR, "data", "field")
ASIA_DIR = os.path.join(FIELD_DIR, "asia")
CHAMPIONSHIP_FINALS_PATH = os.path.join(FIELD_DIR, "championship_finals.csv")
FINALS_PATH = os.path.join(FIELD_DIR, "finals.csv")
DL_RESULTS = os.path.join(BASE_DIR, "data", "dl_final_results.csv")

ASIA_YEARS = range(2015, 2026)
ASIA_PAGES = 3
REQUEST_PAUSE = 0.3
# Nationality sits under a blank header, fifth column from zero, on every
# World Athletics toplist, world or area, with or without wind.
NAT = 5

# Checked live on 2026-09-14 with getCalendarEvents: each has API results.
ASIAN_COMPETITIONS = [
    {"id": 7121814, "competition": "Asian Games 2018", "year": 2018, "tier": "asia"},
    {"id": 7147637, "competition": "Asian Games 2023", "year": 2023, "tier": "asia"},
    {"id": 7129855, "competition": "Asian Championships 2019", "year": 2019, "tier": "asia"},
    {"id": 7185337, "competition": "Asian Championships 2023", "year": 2023, "tier": "asia"},
    {"id": 7216290, "competition": "Asian Championships 2025", "year": 2025, "tier": "asia"},
]

WORLD_YEARS = [y for y in range(2009, 2026) if y != 2020]
# major_meets_scraper's labels for the championships the deployed model trains
# on. Continental Tour Gold meetings are left out, as they are there.
WORLD_LABEL_TIERS = {"Olympics": "global", "World Champs": "global", "European Champs": "continental"}

DAYS_QUERY = """query getCalendarCompetitionResults($competitionId: Int, $day: Int, $eventId: Int) {
  getCalendarCompetitionResults(competitionId: $competitionId, day: $day, eventId: $eventId) {
    options { days { day date } }
  }
}"""

FINAL_COLUMNS = ["competition", "competition_id", "tier", "year", "cutoff", "discipline",
                 "athlete_name", "nationality", "place", "mark"]


def field_key(name):
    """An athlete name reduced to letters and digits. The world and area
    toplists and the results pages spell one athlete more than one way
    ("Tae-poong NAM", "Taepoong NAM"), and the difference is punctuation."""
    return re.sub(r"[^A-Z0-9]", "", str(final_labels.normalize_name(str(name or ""))))


def parse_date(value):
    when = pd.to_datetime(value, format="%d %b %Y", errors="coerce")
    if pd.isna(when):
        when = pd.to_datetime(value, errors="coerce")
    return None if pd.isna(when) else when.normalize()


def _place(value):
    text = str(value or "").strip().rstrip(".")
    return int(text) if text.isdigit() else None


# ---- Asian area toplists ------------------------------------------------------

def asia_toplist_url(key, year):
    base = live_fetcher.DISCIPLINE_URLS[key].rsplit("/", 1)[0]
    return f"{base}/{year}?{ags.ASIA_QUERY}"


def asia_toplist(key, year, fetch=ags.fetch_page, pages=ASIA_PAGES, pause=REQUEST_PAUSE):
    """(header, rows) of one season's Asian list for one discipline, or
    (None, []) when World Athletics serves no table.

    Raises when more than 5% of the rows are from outside Asia: that means the
    area filter was ignored and the world list came back, which would quietly
    fill the Asian history with everyone else."""
    url = asia_toplist_url(key, year)
    header, rows, seen = None, [], set()
    for page in range(1, pages + 1):
        page_headers, page_rows, urls = live_fetcher.parse_toplist_html(
            fetch(live_fetcher.toplist_page_url(url, page)))
        if pause:
            time.sleep(pause)
        if not page_rows:
            break
        if header is None:
            header = list(page_headers) + ["discipline", "year", "ProfileURL"]
            comp = page_headers.index("Competitor")
        width = len(page_headers)
        for cells, profile in zip(page_rows, urls):
            cells = (list(cells) + [""] * width)[:width]
            ident = profile or f"{cells[comp]}|{cells[NAT]}"
            if ident in seen:
                continue
            seen.add(ident)
            rows.append(cells + [key, str(year), profile or ""])
        if len(page_rows) < 100:
            break
    outside = [r for r in rows if r[NAT] and r[NAT] not in ags.ASIAN_NATIONS]
    if rows and len(outside) > 0.05 * len(rows):
        raise ValueError(f"{key} {year}: {len(outside)} of {len(rows)} rows are from outside Asia; "
                         "World Athletics did not apply the area filter")
    return header, rows


def scrape_asia_toplists(keys, years=ASIA_YEARS, refresh=False, fetch=ags.fetch_page):
    """One file per discipline, every season in it. A discipline whose file
    exists is skipped unless `refresh`, so a run that stops can be restarted."""
    os.makedirs(ASIA_DIR, exist_ok=True)
    for key in keys:
        path = os.path.join(ASIA_DIR, f"{key}.csv")
        if os.path.exists(path) and not refresh:
            print(f"  {key}: on disk, skipped")
            continue
        header, rows = None, []
        for year in years:
            try:
                year_header, year_rows = asia_toplist(key, year, fetch=fetch)
            except Exception as exc:  # one season must not end a 36-discipline run
                print(f"  {key} {year}: FAILED ({str(exc)[:80]})")
                continue
            if year_header is None:
                print(f"  {key} {year}: no table")
                continue
            if header is None:
                header = year_header
            elif year_header != header:
                print(f"  {key} {year}: columns differ from {years[0]}'s, season skipped")
                continue
            rows.extend(year_rows)
        if header is None:
            print(f"  {key}: nothing fetched, no file written")
            continue
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)
        print(f"  {key}: {len(rows)} rows across {len({r[-2] for r in rows})} seasons")


# ---- finals -------------------------------------------------------------------

def competition_start(competition_id, query=None):
    """The first day of a competition, from World Athletics' own day list."""
    query = query or (lambda cid: dlr.graphql(
        "getCalendarCompetitionResults", {"competitionId": cid, "day": None, "eventId": None},
        DAYS_QUERY)["getCalendarCompetitionResults"])
    days = [parse_date(d.get("date")) for d in (query(competition_id).get("options") or {}).get("days") or []]
    days = [d for d in days if d is not None]
    if not days:
        raise RuntimeError(f"competition {competition_id}: World Athletics lists no dates")
    return min(days)


def world_competitions(years=WORLD_YEARS, find=None):
    """The Olympics, senior Worlds and Europeans of each season, found the way
    major_meets_scraper finds them. Imported here rather than at the top: that
    module rewraps sys.stdout when imported, which tears down a pytest run."""
    if find is None:
        import major_meets_scraper as mms
        find = mms.find_year_meetings
    comps = []
    for year in years:
        for meeting, label in find(year):
            tier = WORLD_LABEL_TIERS.get(label)
            if tier:
                comps.append({"id": meeting["id"], "competition": meeting["name"], "year": year, "tier": tier})
    return comps


def competition_finals(competitions, fetch=us.fetch_results, start=competition_start):
    """Every finalist of every discipline we hold, at each competition.

    fetch_results with no field takes the fullest race called a Final, so this
    is the whole finishing order, the athletes who did not finish included:
    they were in the field, and a model that never sees them never learns how
    often the favourite fails to. A competition that returns nothing is named
    and skipped, not written as an empty field."""
    rows, missing = [], []
    for comp in competitions:
        found = fetch(field=None, competition_id=comp["id"])
        if not found:
            missing.append(comp["competition"])
            print(f"  {comp['competition']} {comp['year']}: no results came back, skipped")
            continue
        cutoff = start(comp["id"])
        for r in found:
            rows.append({
                "competition": comp["competition"], "competition_id": comp["id"], "tier": comp["tier"],
                "year": comp["year"], "cutoff": cutoff, "discipline": r["discipline"],
                "athlete_name": r["athlete_name"], "nationality": r.get("nationality"),
                "place": _place(r.get("place")), "mark": r.get("mark"),
            })
        print(f"  {comp['competition']} {comp['year']}: {len(found)} finalists, starts {cutoff.date()}")
    return pd.DataFrame(rows, columns=FINAL_COLUMNS), missing


def dl_final_starts(find=dlr.find_final_competition_ids):
    """{year: first day of that year's Diamond League Final}. 2018 and 2019 held
    it in two cities two weeks apart, so the earlier one is the cut-off."""
    starts = {}
    for year in final_labels.dl_final_dates():
        meetings = find(int(year))
        dates = [parse_date(m.get("startDate")) for m in meetings]
        dates = [d for d in dates if d is not None]
        if dates:
            starts[int(year)] = min(dates)
    return starts


def dl_finals(starts=None):
    """Diamond League Finals from data/dl_final_results.csv, which already holds
    each whole field, cut off at the first day of that year's Final."""
    starts = dl_final_starts() if starts is None else starts
    dl = pd.read_csv(DL_RESULTS)
    dl["cutoff"] = dl["year"].map(lambda y: starts.get(int(y)))
    dl = dl.dropna(subset=["cutoff"])
    return pd.DataFrame({
        "competition": "Diamond League Final", "competition_id": None, "tier": "dl_final",
        "year": dl["year"].astype(int), "cutoff": dl["cutoff"], "discipline": dl["discipline"],
        "athlete_name": dl["athlete_name"], "nationality": dl["nationality"],
        "place": dl["place"].map(_place), "mark": dl["mark"],
    })[FINAL_COLUMNS]


# ---- season scores ------------------------------------------------------------

def _toplist_rows(path, source):
    try:
        df = pd.read_csv(path, low_memory=False)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return pd.DataFrame()
    if "source" in df.columns:
        df = df[df["source"].isna() | (df["source"] == "toplist")]
    if df.empty or "Results Score" not in df.columns:
        return pd.DataFrame()
    out = pd.DataFrame({
        "key": df["Competitor"].map(field_key),
        "nat": df.iloc[:, NAT].astype(str).str.strip(),
        "year": pd.to_numeric(df["year"], errors="coerce"),
        "score": pd.to_numeric(df["Results Score"], errors="coerce"),
        "date": df["Date"].map(parse_date),
        "dob": df["DOB"].map(parse_date),
        "name": df["Competitor"],
        "mark": df["Mark"].astype(str),
        "source": source,
    })
    return out.dropna(subset=["year", "score", "date"])


def season_scores(key, raw_dir=RAW_DIR, asia_dir=ASIA_DIR, extra=()):
    """Every toplist row we hold for one discipline: world history, Asian
    history, and any `extra` (path, source) pairs, such as this season's lists.
    One athlete can appear in both lists for one season; both rows are kept and
    the reader takes the best."""
    frames = [_toplist_rows(os.path.join(raw_dir, f"{key}.csv"), "world"),
              _toplist_rows(os.path.join(asia_dir, f"{key}.csv"), "asia")]
    frames += [_toplist_rows(path, source) for path, source in extra]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame(columns=["key", "nat", "year", "score", "date", "dob", "name", "mark", "source"])
    out = pd.concat(frames, ignore_index=True)
    out["year"] = out["year"].astype(int)
    return out


def athlete_history(scores, key, nat):
    """This athlete's toplist rows. By name and nationality first; by name
    alone only when the name belongs to one nationality, since a switch of
    allegiance (to Bahrain or Qatar, say) otherwise loses every earlier season."""
    same = scores[(scores["key"] == key) & (scores["nat"] == nat)]
    if not same.empty:
        return same, "nameNat"
    by_name = scores[scores["key"] == key]
    if not by_name.empty and by_name["nat"].nunique() == 1:
        return by_name, "name"
    return by_name.iloc[0:0], None


def score_as_of(history, year, cutoff):
    """(season best score, its date, taken from the season before) as of the
    cut-off, or (None, None, None).

    A toplist holds one row per athlete per season, their best. When that best
    came on or after the cut-off, what they had run before it is not on any
    list, and the previous season's best stands in, flagged."""
    this = history[(history["year"] == year) & (history["date"] < cutoff)]
    if not this.empty:
        best = this.loc[this["score"].idxmax()]
        return float(best["score"]), best["date"], 0
    prev = history[history["year"] == year - 1]
    if not prev.empty:
        best = prev.loc[prev["score"].idxmax()]
        return float(best["score"]), best["date"], 1
    return None, None, None


def attach_scores(finals, scores_for=season_scores):
    """The finals with each finalist's pre-competition numbers: season best,
    its date, career best before this season, last season's best, date of
    birth, and how the history was matched. Unmatched finalists keep NaNs."""
    out = []
    for key, group in finals.groupby("discipline"):
        scores = scores_for(key)
        rows = group.to_dict("records")
        for row in rows:
            cutoff = pd.Timestamp(row["cutoff"])
            year = int(row["year"])
            history, how = athlete_history(scores, field_key(row["athlete_name"]), str(row["nationality"] or "").strip())
            sb, sb_date, prior = score_as_of(history, year, cutoff)
            earlier = history[history["year"] < year]
            last = history[history["year"] == year - 1]
            dob = history["dob"].dropna()
            row.update({
                "sb_score": sb, "sb_date": sb_date, "sb_prior_season": prior,
                "career_best": float(earlier["score"].max()) if not earlier.empty else None,
                "prev_season_best": float(last["score"].max()) if not last.empty else None,
                "dob": dob.iloc[0] if not dob.empty else None,
                "matched_by": how if sb is not None else None,
            })
            out.append(row)
    return pd.DataFrame(out)


def coverage(scored):
    """Per competition: finals, finalists, the share with a season score, and
    the share of podium places whose athlete has one."""
    df = scored.copy()
    df["scored"] = df["sb_score"].notna()
    df["podium"] = df["place"].between(1, 3)
    table = df.groupby(["tier", "competition", "year"]).agg(
        finals=("discipline", "nunique"), finalists=("athlete_name", "size"),
        scored=("scored", "mean"),
        podium_scored=("scored", lambda s: s[df.loc[s.index, "podium"]].mean()),
    ).reset_index()
    return table.sort_values(["tier", "year", "competition"])


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--asia-toplists", action="store_true")
    parser.add_argument("--finals", action="store_true")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--only", default=None, help="comma-separated discipline keys")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args(argv)
    keys = args.only.split(",") if args.only else list(live_fetcher.DISCIPLINE_URLS)

    if args.asia_toplists:
        print("=== Asian area toplists, 2015-2025 ===")
        scrape_asia_toplists(keys, refresh=args.refresh)
    if args.finals:
        print("=== Championship finals: Olympics, Worlds, Europeans, Asian Games, Asian Championships ===")
        finals, missing = competition_finals(world_competitions() + ASIAN_COMPETITIONS)
        asian_missing = [c["competition"] for c in ASIAN_COMPETITIONS if c["competition"] in missing]
        if asian_missing:
            print(f"  NOT SAVED: no results for {', '.join(asian_missing)}; the Asian finals are the point")
            return 1
        os.makedirs(FIELD_DIR, exist_ok=True)
        finals.to_csv(CHAMPIONSHIP_FINALS_PATH, index=False)
        print(f"  {len(finals)} rows, {finals.groupby(['competition', 'year', 'discipline']).ngroups} finals "
              f"-> {CHAMPIONSHIP_FINALS_PATH}")
    if args.report:
        print("=== Finals joined to season scores ===")
        finals = dl_finals()
        if os.path.exists(CHAMPIONSHIP_FINALS_PATH):
            champs = pd.read_csv(CHAMPIONSHIP_FINALS_PATH, parse_dates=["cutoff"])
            finals = pd.concat([finals, champs], ignore_index=True)
        finals = finals[finals["discipline"].isin(keys)]
        scored = attach_scores(finals)
        os.makedirs(FIELD_DIR, exist_ok=True)
        scored.to_csv(FINALS_PATH, index=False)
        table = coverage(scored)
        with pd.option_context("display.max_rows", 200, "display.width", 160):
            print(table.to_string(index=False, formatters={
                "scored": "{:.0%}".format, "podium_scored": "{:.0%}".format}))
        print(f"  {len(scored)} rows -> {FINALS_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
