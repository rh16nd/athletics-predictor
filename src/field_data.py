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
  2. SEASON SCORES: each finalist's best Results Score before the competition
     started. From the toplists where they can say it (world top 100 in
     data/raw; area lists up to 300 deep, Asia 2015-2025 in data/field/asia and
     Europe 2009-2024 in data/field/europe), and from the athlete's own World
     Athletics profile where they cannot (data/field/seasons). See
     attach_scores for which is used when, and why the order matters.

Nothing here writes to data/raw, so the existing model never sees these rows.

THE CUT-OFF
Every final is paired with the day its competition STARTED, and only marks
dated strictly before it count. A World Championships final is on day 8; a
season best set in its heat on day 6 is not something anyone knew before the
championship began, and the Asian Games call is frozen before its first day.

Usage, in order:
    python src/field_data.py --asia-toplists [--only k1,k2] [--refresh]
    python src/field_data.py --europe-toplists [--only k1,k2] [--refresh]
    python src/field_data.py --finals          # championships, world and Asian
    python src/field_data.py --ids             # every finalist's World Athletics id
    python src/field_data.py --seasons [--years 2015,2017]   # profiles the toplists need
    python src/field_data.py --races [--years 2015,2017]     # every finalist's races, with places
    python src/field_data.py --report          # join to season scores, print coverage
"""
import argparse
import csv
import json
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
EUROPE_DIR = os.path.join(FIELD_DIR, "europe")
SEASONS_DIR = os.path.join(FIELD_DIR, "seasons")
RACES_DIR = os.path.join(FIELD_DIR, "races")
CHAMPIONSHIP_FINALS_PATH = os.path.join(FIELD_DIR, "championship_finals.csv")
IDS_PATH = os.path.join(FIELD_DIR, "finalist_ids.csv")
FINALS_PATH = os.path.join(FIELD_DIR, "finals.csv")
DL_RESULTS = os.path.join(BASE_DIR, "data", "dl_final_results.csv")

AREA_PAGES = 3
REQUEST_PAUSE = 0.3
# Nationality sits under a blank header, fifth column from zero, on every
# World Athletics toplist, world or area, with or without wind.
NAT = 5
# A competition is trained on only when at least this share of its finalists
# has a season score (the plan, 2026-09-14). Below it, "rank within the field"
# is a rank among the few we found, not among the athletes who ran.
MIN_MATCH = 0.85

# European Athletics' 51 member federations, and ANA, the code World Athletics'
# lists give authorised neutral athletes.
EUROPEAN_NATIONS = frozenset({
    "ALB", "AND", "ARM", "AUT", "AZE", "BEL", "BIH", "BLR", "BUL", "CRO", "CYP",
    "CZE", "DEN", "ESP", "EST", "FIN", "FRA", "GBR", "GEO", "GER", "GIB", "GRE",
    "HUN", "IRL", "ISL", "ISR", "ITA", "KOS", "LAT", "LIE", "LTU", "LUX", "MDA",
    "MKD", "MLT", "MNE", "MON", "NED", "NOR", "POL", "POR", "ROU", "RUS", "SLO",
    "SMR", "SRB", "SUI", "SVK", "SWE", "TUR", "UKR", "ANA",
})

# Each area list: its query, the nations it may hold, where it is saved, and
# the seasons fetched. Every championship in the finals has its own season and
# the one before it on file: Asia's from 2018, the Europeans' from 2010.
AREAS = {
    "asia": {"query": ags.ASIA_QUERY, "nations": ags.ASIAN_NATIONS, "dir": ASIA_DIR,
             "years": range(2015, 2026)},
    "europe": {"query": "regionType=area&region=europe", "nations": EUROPEAN_NATIONS, "dir": EUROPE_DIR,
               "years": range(2009, 2025)},
}

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

# The results feed with each competitor's profile slug, which ends in their id
# ("islamic-republic-of-iran/hassan-taftian-14421587"), and date of birth.
IDS_QUERY = dlr.RESULTS_QUERY.replace("competitor { name }", "competitor { name urlSlug birthDate }")

# One athlete's results in one season, grouped by event. Checked live on
# 2026-09-14: David Rudisha's 2015 800m came back with 11 dated, scored results,
# 7 of them before the World Championships began.
SEASON_RESULTS_QUERY = """query SeasonResults($id: Int, $year: Int) {
  getSingleCompetitorResultsDiscipline(id: $id, resultsByYear: $year) {
    resultsByEvent { discipline indoor results { date mark notLegal resultScore } }
  }
}"""

# The same season with each result's meeting, its World Athletics category (OW,
# DF, GW, GL, then A to F), its round ("F" for a final) and the place, for the
# field model's form, big-meet and head-to-head features. Checked live on
# 2026-09-15: Abderrahman Samba's 2026 400m hurdles came back as "GCC Games",
# category D, race F, place "1.", and Xiamen as GW, F, "5.".
RACES_QUERY = SEASON_RESULTS_QUERY.replace(
    "results { date mark notLegal resultScore }",
    "results { date competition category race place mark notLegal resultScore }")

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
    """A finishing place as an int, or None. pandas reads a place column that
    holds only "1.", "2." and so on as floats, and str(1.0) is not digits."""
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    text = str(value or "").strip().rstrip(".")
    return int(text) if text.isdigit() else None


def competition_label(name, year):
    """"European Athletics Championships 2018", without doubling the year for
    names that already carry it ("Asian Games 2018")."""
    return str(name) if str(year) in str(name) else f"{name} {year}"


# ---- area toplists ------------------------------------------------------------

def area_toplist_url(key, year, area):
    base = live_fetcher.DISCIPLINE_URLS[key].rsplit("/", 1)[0]
    return f"{base}/{year}?{AREAS[area]['query']}"


def area_toplist(key, year, area, fetch=ags.fetch_page, pages=AREA_PAGES, pause=REQUEST_PAUSE):
    """(header, rows) of one season's area list for one discipline, or
    (None, []) when World Athletics serves no table.

    Raises when more than 5% of the rows are from outside the area: that means
    the area filter was ignored and the world list came back, which would
    quietly fill the area's history with everyone else."""
    url = area_toplist_url(key, year, area)
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
    outside = [r for r in rows if r[NAT] and r[NAT] not in AREAS[area]["nations"]]
    if rows and len(outside) > 0.05 * len(rows):
        raise ValueError(f"{key} {year}: {len(outside)} of {len(rows)} rows are from outside {area}; "
                         "World Athletics did not apply the area filter")
    return header, rows


def scrape_area_toplists(keys, area, years=None, refresh=False, fetch=ags.fetch_page,
                         out_dir=None, pause=REQUEST_PAUSE):
    """One file per discipline, every season in it. A discipline whose file
    exists is skipped unless `refresh`, so a run that stops can be restarted.
    A discipline with a season that failed to download is not saved at all, so
    the restart fetches it again rather than skipping a file with a hole in it."""
    out_dir = out_dir or AREAS[area]["dir"]
    years = AREAS[area]["years"] if years is None else years
    os.makedirs(out_dir, exist_ok=True)
    for key in keys:
        path = os.path.join(out_dir, f"{key}.csv")
        if os.path.exists(path) and not refresh:
            print(f"  {key}: on disk, skipped")
            continue
        header, rows, failed = None, [], []
        for year in years:
            try:
                year_header, year_rows = area_toplist(key, year, area, fetch=fetch, pause=pause)
            except Exception as exc:  # one season must not end a 36-discipline run
                print(f"  {key} {year}: FAILED ({str(exc)[:80]})")
                failed.append(year)
                continue
            if year_header is None:
                print(f"  {key} {year}: no table")
                continue
            if header is None:
                header = year_header
            elif year_header != header:
                print(f"  {key} {year}: columns differ from the first season's, season skipped")
                continue
            rows.extend(year_rows)
        if failed:
            print(f"  {key}: NOT SAVED, {len(failed)} season(s) failed; run again to fetch it")
            continue
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


def season_field(year, histories):
    """Each discipline's names on this season's toplists, in the shape
    fetch_results takes as a qualified field. `histories` is {key: season_scores}.

    Used only to tell a championship's own final from another race labelled
    Final at the same meeting. Without it pick_final keeps the fuller race, and
    at the 2015 Worlds that was a masters 800m won in 2:00.92 and a masters
    400m won in 60.05, not Rudisha's and Felix's finals."""
    field = []
    for key, scores in histories.items():
        names = [] if scores.empty else pd.unique(scores.loc[scores["year"] == year, "name"])
        field.append({"discKey": key, "athletes": [{"name": name} for name in names]})
    return field


def competition_finals(competitions, fetch=us.fetch_results, start=competition_start, field_for=lambda year: None):
    """Every finalist of every discipline we hold, at each competition.

    This is the whole finishing order, the athletes who did not finish
    included: they were in the field, and a model that never sees them never
    learns how often the favourite fails to. `field_for(year)` gives the names
    that pick the championship's race when two are labelled Final (see
    season_field). A competition that returns nothing is named and skipped,
    not written as an empty field."""
    rows, missing = [], []
    for comp in competitions:
        found = fetch(field=field_for(comp["year"]), competition_id=comp["id"])
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


def all_finals(keys=None):
    """Diamond League Finals and the championship finals on disk, together."""
    finals = dl_finals()
    if os.path.exists(CHAMPIONSHIP_FINALS_PATH):
        champs = pd.read_csv(CHAMPIONSHIP_FINALS_PATH, parse_dates=["cutoff"])
        finals = pd.concat([finals, champs], ignore_index=True)
    return finals if keys is None else finals[finals["discipline"].isin(keys)]


# ---- finalists' World Athletics ids --------------------------------------------

def slug_id(slug):
    """14421587 from "islamic-republic-of-iran/hassan-taftian-14421587"."""
    tail = str(slug or "").rsplit("-", 1)[-1]
    return int(tail) if tail.isdigit() else None


def _results_day(competition_id, day):
    time.sleep(REQUEST_PAUSE)
    return dlr.graphql("getCalendarCompetitionResults",
                       {"competitionId": competition_id, "day": day, "eventId": None},
                       IDS_QUERY)["getCalendarCompetitionResults"]


def competition_athletes(competition_id, query=_results_day):
    """{(discipline key, field_key(name)): (athlete id, date of birth)} for
    everyone with a result in any race at one competition."""
    first = query(competition_id, None)
    days = [d["day"] for d in (first.get("options") or {}).get("days") or []]
    found = {}
    for data in ([query(competition_id, day) for day in days] if days else [first]):
        for group in data.get("eventTitles") or []:
            for event in group.get("events") or []:
                key = dlr.resolve_discipline_key(event.get("gender"), event.get("event"), mile_as_1500=True)
                if key is None:
                    continue
                for race in event.get("races") or []:
                    for result in race.get("results") or []:
                        who = result.get("competitor") or {}
                        athlete = slug_id(who.get("urlSlug"))
                        if athlete and who.get("name"):
                            found.setdefault((key, field_key(who["name"])), (athlete, parse_date(who.get("birthDate"))))
    return found


def finalist_ids(finals, athletes_at=competition_athletes, dl_meetings=dlr.find_final_competition_ids):
    """One row per finalist: competition, year, discipline, athlete_name,
    athlete_id and birth_date, read from that competition's own results feed.
    A Diamond League Final is read from each of its meetings (two in 2018 and
    2019)."""
    rows = []
    for (competition, year), group in finals.groupby(["competition", "year"]):
        competition_id = group["competition_id"].iloc[0]
        if pd.notna(competition_id):
            meetings = [int(competition_id)]
        else:
            meetings = [int(m["id"]) for m in dl_meetings(int(year))]
        found = {}
        for meeting in meetings:
            for who, value in athletes_at(meeting).items():
                found.setdefault(who, value)
        hits = 0
        for row in group.itertuples():
            athlete, born = found.get((row.discipline, field_key(row.athlete_name)), (None, None))
            hits += athlete is not None
            rows.append({"competition": competition, "year": int(year), "discipline": row.discipline,
                         "athlete_name": row.athlete_name, "athlete_id": athlete, "birth_date": born})
        print(f"  {competition_label(competition, year)}: {hits} of {len(group)} finalists have an id")
    return pd.DataFrame(rows, columns=["competition", "year", "discipline", "athlete_name", "athlete_id", "birth_date"])


def with_ids(finals, ids_path=IDS_PATH):
    """`finals` with athlete_id and birth_date joined on, blank before --ids has run."""
    if not os.path.exists(ids_path):
        return finals.assign(athlete_id=None, birth_date=None)
    keys = ["competition", "year", "discipline", "athlete_name"]
    ids = pd.read_csv(ids_path).drop_duplicates(subset=keys)
    return finals.merge(ids, on=keys, how="left")


# ---- athletes' own season results ---------------------------------------------

def load_seasons(seasons_dir=SEASONS_DIR):
    """{year: {athlete id as text: that season's results by event}}."""
    seasons = {}
    if not os.path.isdir(seasons_dir):
        return seasons
    for name in os.listdir(seasons_dir):
        if name.endswith(".json") and name[:-5].isdigit():
            with open(os.path.join(seasons_dir, name), encoding="utf-8") as f:
                seasons[int(name[:-5])] = json.load(f)
    return seasons


def _save_season(year, athletes, seasons_dir):
    os.makedirs(seasons_dir, exist_ok=True)
    path = os.path.join(seasons_dir, f"{year}.json")
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(athletes, f)
    os.replace(path + ".tmp", path)


def fetch_season(athlete_id, year, query=SEASON_RESULTS_QUERY):
    time.sleep(ags.LOOKUP_PAUSE)
    data = dlr.graphql("SeasonResults", {"id": int(athlete_id), "year": int(year)}, query)
    return (data.get("getSingleCompetitorResultsDiscipline") or {}).get("resultsByEvent") or []


def fetch_races(athlete_id, year):
    """One season with each result's meeting, category, round and place."""
    return fetch_season(athlete_id, year, RACES_QUERY)


def fetch_seasons(wanted, fetch=fetch_season, seasons_dir=SEASONS_DIR, save_every=100, refresh=False):
    """Each (athlete id, year) in `wanted` that is not on disk, saved one file
    per year so that workers given different years never write the same file.
    A request that fails is not saved, so the next run asks again; a season
    with no results is saved empty, so it is not asked twice. Returns the
    number that failed.

    With `refresh`, every season in `wanted` is fetched again: an entry list's
    season is still going on, and the call before a championship must see the
    meetings run since the last fetch."""
    seasons = load_seasons(seasons_dir)
    wanted = {(int(a), int(y)) for a, y in wanted}
    todo = sorted(w for w in wanted if refresh or str(w[0]) not in seasons.get(w[1], {}))
    print(f"  {len(wanted)} athlete-seasons wanted, {len(wanted) - len(todo)} on disk, {len(todo)} to fetch")
    failed, dirty = 0, set()
    for i, (athlete, year) in enumerate(todo, 1):
        try:
            seasons.setdefault(year, {})[str(athlete)] = fetch(athlete, year)
            dirty.add(year)
        except Exception as exc:  # one athlete must not end a run of thousands
            failed += 1
            if failed <= 10:
                print(f"    {athlete} {year}: FAILED ({str(exc)[:80]})")
        if i % save_every == 0 or i == len(todo):
            for y in dirty:
                _save_season(y, seasons[y], seasons_dir)
            dirty.clear()
        if i % 500 == 0 or i == len(todo):
            print(f"  {i} of {len(todo)} fetched, {failed} failed")
    return failed


def profile_best(events, key, cutoff):
    """(best Results Score, its date) among one season's results in one event
    that were legal, outdoors, electronically timed and dated before the
    cut-off; None when there are none. A profile's resultScore is the toplist's
    Results Score (asian_games_scraper.season_best checked it), and toplists
    take only electronic times, so a hand time ("1:44.2h") is left out.

    Indoors comes from the "(i)" World Athletics puts at the end of a meeting's
    name (ags.is_indoor), because the profile's own flag is always empty. The
    seasons in data/field/seasons were fetched without meeting names, so only a
    season from --races (data/field/races) can be filtered (found 2026-09-15)."""
    wanted = ags.WA_EVENT_NAMES.get(key)
    best = None
    for group in events or []:
        if group.get("discipline") != wanted:
            continue
        for result in group.get("results") or []:
            score, when = result.get("resultScore"), parse_date(result.get("date"))
            if (result.get("notLegal") or not score or when is None or when >= cutoff
                    or ags.is_indoor(result, group)
                    or str(result.get("mark") or "").strip().lower().endswith("h")):
                continue
            if best is None or score > best[0]:
                best = (float(score), when)
    return best


# ---- race by race: form, big meetings and head-to-head --------------------------

# World Athletics' meeting categories, top tier first: OW is the Olympics and the
# World Championships, DF the Diamond League Final meetings, and GW and GL hold
# the Diamond League meetings and area championships such as the Europeans
# (2012's GL results open with the European Championships, Zurich, Eugene,
# Lausanne and Monaco). A, the tier below, holds the Commonwealth Games, the 2018
# Asian Games and Continental Tour Silver meetings. Fixed on 2026-09-15 from the
# category counts of the first 29,472 results fetched, before any comparison
# was run, and not to be changed after one has been.
BIG_CATEGORIES = frozenset({"OW", "DF", "GW", "GL"})
# Recent form is the six weeks before the cut-off: a championship's lead-in
# (the Diamond League Final and the Ultimate, before the 2026 Asian Games).
RECENT_DAYS = 42
FORM_MARKS = 3
BIG_PODIUM_CAP = 5
# A final's round is "F", or "F1", "F2" when it is run in sections. Heats are
# "H1", qualifying rounds "Q1", semi-finals "SF1".
_FINAL_ROUND = re.compile(r"^F\d*$")
RACE_COLUMNS = ["form_score", "recent_score", "races", "big_podiums", "h2h_top", "has_races"]


def race_summary(events, key, cutoff):
    """One athlete's season in one event before the cut-off, race by race, or
    None when it holds nothing there.

    Only outdoor results dated strictly before the cut-off count, and for the
    scores only legal, electronically timed ones, as in profile_best:
      form_score    the mean of the best FORM_MARKS scores, so one outlying mark
                    moves it less than it moves the season best;
      recent_score  the best score in the RECENT_DAYS before the cut-off, or None;
      races         how many scores there are.
    A place counts whether or not its mark does:
      big_podiums   places 1 to 3 in a final at a BIG_CATEGORIES meeting,
                    capped at BIG_PODIUM_CAP;
      finals        {(meeting, date, round): place} for every final, for h2h_top."""
    wanted = ags.WA_EVENT_NAMES.get(key)
    cutoff = pd.Timestamp(cutoff)
    scores, recent, podiums, finals = [], None, 0, {}
    for group in events or []:
        if group.get("discipline") != wanted:
            continue
        for result in group.get("results") or []:
            when = parse_date(result.get("date"))
            if when is None or when >= cutoff or ags.is_indoor(result, group):
                continue
            place, round_ = _place(result.get("place")), str(result.get("race") or "")
            if place is not None and _FINAL_ROUND.match(round_):
                finals[(str(result.get("competition") or ""), when, round_)] = place
                if place <= 3 and result.get("category") in BIG_CATEGORIES:
                    podiums += 1
            score = result.get("resultScore")
            if (result.get("notLegal") or not score
                    or str(result.get("mark") or "").strip().lower().endswith("h")):
                continue
            scores.append(float(score))
            if (cutoff - when).days <= RECENT_DAYS:
                recent = float(score) if recent is None else max(recent, float(score))
    if not scores and not finals:
        return None
    best = sorted(scores, reverse=True)[:FORM_MARKS]
    return {"form_score": sum(best) / len(best) if best else None, "recent_score": recent,
            "races": len(scores), "big_podiums": min(podiums, BIG_PODIUM_CAP), "finals": finals}


def h2h_top(finals_by_athlete, scores):
    """{athlete: net head-to-head record against the field's strongest}.

    `finals_by_athlete` is {athlete: race_summary's finals} and `scores`
    {athlete: season score or None}, for one field. Each athlete is set against
    the three highest season scores in the field other than their own, in the
    finals both ran: (wins - losses) / meetings, and 0 when they never met.
    Against those three alone, so adding slower entrants changes nobody's
    number. A final counts once for each pair, and a shared place is neither a
    win nor a loss."""
    ranked = sorted((a for a, s in scores.items() if s is not None and not pd.isna(s)),
                    key=lambda a: (-float(scores[a]), str(a)))
    out = {}
    for athlete in scores:
        mine = finals_by_athlete.get(athlete) or {}
        wins = losses = 0
        for rival in [r for r in ranked if r != athlete][:3]:
            theirs = finals_by_athlete.get(rival) or {}
            for race in mine.keys() & theirs.keys():
                wins += mine[race] < theirs[race]
                losses += mine[race] > theirs[race]
        out[athlete] = (wins - losses) / (wins + losses) if wins + losses else 0.0
    return out


def race_columns(field, season_for):
    """RACE_COLUMNS for one field, a final or an entry list: a DataFrame with
    discipline, cutoff and sb_score. `season_for(row)` gives that athlete's
    season race by race (fetch_races), or None. Training and serving both read
    races through here, so the two cannot drift apart."""
    summaries = {}
    for idx, row in field.iterrows():
        events = season_for(row)
        summaries[idx] = race_summary(events, row["discipline"], row["cutoff"]) if events is not None else None
    h2h = h2h_top({i: (s or {}).get("finals") or {} for i, s in summaries.items()},
                  {i: field.at[i, "sb_score"] for i in field.index})

    def value(i, name):
        return (summaries[i] or {}).get(name)

    return pd.DataFrame({
        "form_score": [value(i, "form_score") for i in field.index],
        "recent_score": [value(i, "recent_score") for i in field.index],
        "races": [value(i, "races") for i in field.index],
        "big_podiums": [value(i, "big_podiums") for i in field.index],
        "h2h_top": [h2h[i] for i in field.index],
        "has_races": [summaries[i] is not None for i in field.index],
    }, index=field.index)


def attach_races(scored, races=None):
    """The scored finals with RACE_COLUMNS, each final read on its own, so
    head-to-head stays inside it. `races` is load_seasons(RACES_DIR)."""
    races = {} if races is None else races
    scored = scored.drop(columns=[c for c in RACE_COLUMNS if c in scored.columns])
    if scored.empty:
        return scored.assign(**{c: None for c in RACE_COLUMNS})
    parts = [race_columns(final, lambda row: _season_for(races, row.get("athlete_id"), row["year"]))
             for _, final in scored.groupby(["competition", "year", "discipline"], sort=False)]
    return scored.join(pd.concat(parts))


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
        "wa_id": df["ProfileURL"].map(ags.wa_id) if "ProfileURL" in df.columns else None,
        "source": source,
    })
    return out.dropna(subset=["year", "score", "date"])


def season_scores(key, raw_dir=RAW_DIR, area_dirs=None, extra=()):
    """Every toplist row we hold for one discipline: world history, each area's
    history (`area_dirs`, {area: directory}, every area in AREAS by default),
    and any `extra` (path, source) pairs, such as this season's lists. One
    athlete can appear in more than one list for one season; every row is kept
    and the reader takes the best."""
    area_dirs = {area: spec["dir"] for area, spec in AREAS.items()} if area_dirs is None else area_dirs
    frames = [_toplist_rows(os.path.join(raw_dir, f"{key}.csv"), "world")]
    frames += [_toplist_rows(os.path.join(directory, f"{key}.csv"), area) for area, directory in area_dirs.items()]
    frames += [_toplist_rows(path, source) for path, source in extra]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame(columns=["key", "nat", "year", "score", "date", "dob", "name", "mark", "wa_id", "source"])
    out = pd.concat(frames, ignore_index=True)
    out["year"] = out["year"].astype(int)
    return out


def athlete_history(scores, key, nat, wa_id=None):
    """This athlete's toplist rows. By World Athletics id first, when one is
    given and the rows carry one; then by name and nationality; then by name
    alone only when the name belongs to one nationality, since a switch of
    allegiance (to Bahrain or Qatar, say) otherwise loses every earlier season.

    The id comes first because the lists spell one athlete more than one way:
    serving the Asian Games by name alone missed seven entrants' 2025 marks
    that the entry list's ids found (2026-09-15)."""
    if wa_id is not None and "wa_id" in scores.columns:
        by_id = scores[scores["wa_id"] == wa_id]
        if not by_id.empty:
            return by_id, "id"
    same = scores[(scores["key"] == key) & (scores["nat"] == nat)]
    if not same.empty:
        return same, "nameNat"
    by_name = scores[scores["key"] == key]
    if not by_name.empty and by_name["nat"].nunique() == 1:
        return by_name, "name"
    return by_name.iloc[0:0], None


def _season_for(seasons, athlete_id, year):
    """One athlete's fetched season, or None when it was never fetched."""
    try:
        return seasons.get(int(year), {}).get(str(int(float(athlete_id))))
    except (TypeError, ValueError):
        return None


def attach_scores(finals, scores_for=season_scores, seasons=None):
    """The finals with each finalist's numbers as they stood when the
    competition started.

    The season best is, in this order:
      1. the toplist's season best, when it is dated before the cut-off. A
         toplist holds one best per athlete per season, so that mark IS their
         best before the cut-off;
      2. otherwise their profile's best legal outdoor mark in the event before
         the cut-off (`seasons`, see fetch_seasons and profile_best);
      3. otherwise, when the profile has no such mark, last season's best,
         flagged sb_prior_season.
    A finalist who needs step 2 and has no profile on file is left unscored.

    Going straight from 1 to 3 was a leak, and the first version did it. Whether
    a toplist best came after the cut-off, or whether an athlete made the
    season's list at all, depends on how the championship itself went: on the
    2026-09-14 data, finalists sent to last season that way won a medal 37.9% of
    the time and the rest 23.4%. The flag told the model who peaked in the
    final, and the points ranking scored those same athletes on last year.

    Career best before this season, last season's best and date of birth come
    from earlier seasons' toplists, all of it known before the competition.
    `needs_profile` marks the finalists step 1 could not date."""
    seasons = {} if seasons is None else seasons
    out = []
    for key, group in finals.groupby("discipline"):
        scores = scores_for(key)
        for row in group.to_dict("records"):
            cutoff = pd.Timestamp(row["cutoff"])
            year = int(row["year"])
            history, how = athlete_history(scores, field_key(row["athlete_name"]), str(row["nationality"] or "").strip())
            before = history[(history["year"] == year) & (history["date"] < cutoff)]
            earlier = history[history["year"] < year]
            last = history[history["year"] == year - 1]
            sb = sb_date = prior = source = None
            if not before.empty:
                best = before.loc[before["score"].idxmax()]
                sb, sb_date, prior, source = float(best["score"]), best["date"], 0, "toplist"
            else:
                season = _season_for(seasons, row.get("athlete_id"), year)
                found = profile_best(season, key, cutoff) if season is not None else None
                if found is not None:
                    (sb, sb_date), prior, source = found, 0, "profile"
                elif season is not None and not last.empty:
                    best = last.loc[last["score"].idxmax()]
                    sb, sb_date, prior, source = float(best["score"]), best["date"], 1, "lastSeason"
            dob = history["dob"].dropna()
            row.update({
                "sb_score": sb, "sb_date": sb_date, "sb_prior_season": prior, "sb_source": source,
                "needs_profile": before.empty,
                "career_best": float(earlier["score"].max()) if not earlier.empty else None,
                "prev_season_best": float(last["score"].max()) if not last.empty else None,
                "dob": dob.iloc[0] if not dob.empty else parse_date(row.get("birth_date")),
                "matched_by": how,
            })
            out.append(row)
    return pd.DataFrame(out)


def coverage(scored):
    """Per competition: finals, finalists, the share with a season score, the
    share of podium places whose athlete has one, and whether that clears the
    MIN_MATCH gate for training."""
    df = scored.copy()
    df["scored"] = df["sb_score"].notna()
    df["podium"] = df["place"].between(1, 3)
    table = df.groupby(["tier", "competition", "year"]).agg(
        finals=("discipline", "nunique"), finalists=("athlete_name", "size"),
        scored=("scored", "mean"),
        podium_scored=("scored", lambda s: s[df.loc[s.index, "podium"]].mean()),
    ).reset_index()
    table["trained"] = table["scored"] >= MIN_MATCH
    return table.sort_values(["tier", "year", "competition"])


def winner_disagreements(champs, raw_dir=RAW_DIR):
    """(finals checked, [disagreements]): each Olympics, Worlds and Europeans
    winner in `champs` against the major_meet rows in data/raw, which
    major_meets_scraper chose on its own. This check, not a test, caught the
    2015 masters races; a disagreement is a race to look at by hand."""
    world = champs[champs["tier"] != "asia"]
    names = set(world["competition"])
    checked, disagree = 0, []
    for key in sorted(world["discipline"].unique()):
        try:
            raw = pd.read_csv(os.path.join(raw_dir, f"{key}.csv"), low_memory=False)
        except OSError:
            continue
        if "source" not in raw.columns:
            continue
        meets = raw[(raw["source"] == "major_meet") & raw["Venue"].isin(names)]
        for (venue, year), g in meets.groupby(["Venue", "year"]):
            before = set(g.loc[g["Pos"].map(_place) == 1, "Competitor"].map(field_key))
            if not before:
                continue
            final = world[(world["competition"] == venue) & (world["year"] == year) & (world["discipline"] == key)]
            saved = set(final.loc[final["place"] == 1, "athlete_name"].map(field_key))
            checked += 1
            if not before & saved:
                disagree.append({"competition": venue, "year": int(year), "discipline": key,
                                 "majorMeet": sorted(before), "saved": sorted(saved)})
    return checked, disagree


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--asia-toplists", action="store_true")
    parser.add_argument("--europe-toplists", action="store_true")
    parser.add_argument("--finals", action="store_true")
    parser.add_argument("--check-winners", action="store_true")
    parser.add_argument("--ids", action="store_true")
    parser.add_argument("--seasons", action="store_true")
    parser.add_argument("--races", action="store_true")
    parser.add_argument("--years", default=None, help="comma-separated seasons, for --seasons and --races")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--only", default=None, help="comma-separated discipline keys")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args(argv)
    keys = args.only.split(",") if args.only else list(live_fetcher.DISCIPLINE_URLS)

    for area, flag in (("asia", args.asia_toplists), ("europe", args.europe_toplists)):
        if flag:
            years = AREAS[area]["years"]
            print(f"=== {area.title()} area toplists, {years[0]}-{years[-1]} ===")
            scrape_area_toplists(keys, area, refresh=args.refresh)
    if args.finals:
        print("=== Championship finals: Olympics, Worlds, Europeans, Asian Games, Asian Championships ===")
        histories = {key: season_scores(key) for key in live_fetcher.DISCIPLINE_URLS}
        finals, missing = competition_finals(world_competitions() + ASIAN_COMPETITIONS,
                                             field_for=lambda year: season_field(year, histories))
        asian_missing = [c["competition"] for c in ASIAN_COMPETITIONS if c["competition"] in missing]
        if asian_missing:
            print(f"  NOT SAVED: no results for {', '.join(asian_missing)}; the Asian finals are the point")
            return 1
        os.makedirs(FIELD_DIR, exist_ok=True)
        finals.to_csv(CHAMPIONSHIP_FINALS_PATH, index=False)
        print(f"  {len(finals)} rows, {finals.groupby(['competition', 'year', 'discipline']).ngroups} finals "
              f"-> {CHAMPIONSHIP_FINALS_PATH}")
    if (args.finals or args.check_winners) and os.path.exists(CHAMPIONSHIP_FINALS_PATH):
        checked, disagree = winner_disagreements(pd.read_csv(CHAMPIONSHIP_FINALS_PATH, low_memory=False))
        print(f"  Winner check: {checked} world finals against the major_meet rows, {len(disagree)} disagree")
        for d in disagree:
            print(f"    {d['competition']} {d['year']} {d['discipline']}: "
                  f"major_meet {d['majorMeet']}, saved {d['saved']}")
    if args.ids:
        print("=== Every finalist's World Athletics id, from each competition's results ===")
        ids = finalist_ids(all_finals())
        os.makedirs(FIELD_DIR, exist_ok=True)
        ids.to_csv(IDS_PATH, index=False)
        print(f"  {int(ids['athlete_id'].notna().sum())} of {len(ids)} finalists have an id -> {IDS_PATH}")
    if args.seasons:
        print("=== Profile seasons for finalists the toplists cannot date before their cut-off ===")
        first_pass = attach_scores(with_ids(all_finals()))
        need = first_pass[first_pass["needs_profile"]]
        wanted = {(a, y) for a, y in zip(need["athlete_id"], need["year"]) if pd.notna(a)}
        if args.years:
            years = {int(y) for y in args.years.split(",")}
            wanted = {w for w in wanted if int(w[1]) in years}
        print(f"  {len(need)} of {len(first_pass)} finalists need a profile; "
              f"{int(need['athlete_id'].isna().sum())} of those have no id and stay unscored")
        fetch_seasons(wanted)
    if args.races:
        print("=== Every finalist's season before their final, race by race ===")
        finals = with_ids(all_finals())
        wanted = {(a, y) for a, y in zip(finals["athlete_id"], finals["year"]) if pd.notna(a)}
        if args.years:
            years = {int(y) for y in args.years.split(",")}
            wanted = {w for w in wanted if int(w[1]) in years}
        print(f"  {len(finals)} finalists, {int(finals['athlete_id'].isna().sum())} with no id")
        fetch_seasons(wanted, fetch=fetch_races, seasons_dir=RACES_DIR)
    if args.report:
        print("=== Finals joined to season scores ===")
        races = load_seasons(RACES_DIR)
        # A season fetched race by race carries the meeting names that tell an
        # indoor mark apart (ags.is_indoor), so it replaces the same season from
        # --seasons, which cannot be filtered.
        seasons = load_seasons()
        for year, athletes in races.items():
            seasons.setdefault(year, {}).update(athletes)
        scored = attach_races(attach_scores(with_ids(all_finals(keys)), seasons=seasons), races)
        os.makedirs(FIELD_DIR, exist_ok=True)
        scored.to_csv(FINALS_PATH, index=False)
        table = coverage(scored)
        with pd.option_context("display.max_rows", 200, "display.width", 160):
            print(table.to_string(index=False, formatters={
                "scored": "{:.0%}".format, "podium_scored": "{:.0%}".format}))
        below = table[~table["trained"]]
        if not below.empty:
            print(f"  Under {MIN_MATCH:.0%} found, scored but not trained on: "
                  + ", ".join(f"{competition_label(r.competition, r.year)} ({r.scored:.0%})"
                              for r in below.itertuples()))
        print("  Where each season best came from: "
              + ", ".join(f"{k} {v}" for k, v in scored["sb_source"].fillna("unscored").value_counts().items()))
        undated = scored["needs_profile"] & scored["sb_score"].isna()
        print(f"  {int(undated.sum())} finalists are unscored: no mark before their cut-off on any list or on "
              f"their profile, and none on last season's lists (or no profile season on file)")
        print(f"  Race by race: {int(scored['has_races'].sum())} of {len(scored)} finalists have a season "
              f"in their event on file ({RACES_DIR})")
        print(f"  {len(scored)} rows -> {FINALS_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
