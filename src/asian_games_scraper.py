"""
asian_games_scraper.py -- the field, the season marks and, from 23 September,
the results for athletics at the 20th Asian Games, Aichi-Nagoya 2026.

THE FIELD IS THE OFFICIAL ENTRY LIST
The organisers' results site (results.asiangames2026.org) is a front end for an
API at back.results.asiangames2026.org. Nine days out it already served the
athletics entries: 851 athletes from 41 federations, each with the events they
are entered in and, for most, their World Athletics id. So the field is not
guessed. The plan's stand-in, the top of the Asian list capped at two per
nation, was dropped the day the entry list was found (2026-09-14).

The API sends each reply as a zlib stream written out as text, one character
per byte. The site inflates it with pako; decode() does the same.

SEASON MARKS
World Athletics' Asian area toplist (the world list plus
?regionType=area&region=asia) gives each entrant's 2026 best and its Results
Score. Entrants are matched to it by World Athletics id, and by name and nation
only when the entry carries no id. The two sources order names differently: the
entry list writes "TADA Shuhei", the toplist "Shuhei TADA".

THE MERGED SNAPSHOT
The model was trained on world toplists. Scored against the Asian list alone,
the Asian number one would rank first and look like a world leader. So for each
discipline this writes data/asian_games_2026/raw/{key}_2026.csv: the world
snapshot as it is, plus the Asian row of every ENTRANT the world list lacks.
Entrants only, not the whole Asian list, because every added row enlarges the
population that season_percentile and field_gap are measured against.

Usage:
    python src/asian_games_scraper.py                  # entries, marks, snapshots, results
    python src/asian_games_scraper.py --results-only   # during the Games: results into the saved file
Writes data/asian_games_2026/event.json, asia/{key}_2026.csv and raw/{key}_2026.csv.
"""
import argparse
import csv
import json
import os
import re
import sys
import zlib
from datetime import datetime, timezone

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import championships  # noqa: E402
import live_fetcher  # noqa: E402
import ultimate_scraper as us  # noqa: E402

CHAMP = championships.get("asian-games-2026")
COMPETITION_ID = CHAMP["competitionId"]
OUT_DIR = os.path.join(championships.BASE_DIR, CHAMP["dataDir"])
OUT_PATH = os.path.join(OUT_DIR, "event.json")
ASIA_DIR = os.path.join(OUT_DIR, "asia")
SNAPSHOT_DIR = os.path.join(OUT_DIR, "raw")
WORLD_RAW_DIR = os.path.join(championships.BASE_DIR, "data", "raw")
YEAR = 2026

EVENT_META = {
    "competitionId": COMPETITION_ID,
    "name": "20th Asian Games",
    "shortName": "Aichi-Nagoya 2026",
    "venue": "Paloma Mizuho Stadium",
    "city": "Nagoya",
    "country": "JPN",
    "startDate": CHAMP["startDate"],
    "endDate": CHAMP["endDate"],
    "timezone": "Asia/Tokyo",
}

PORTAL_SITE = "https://results.asiangames2026.org"
PORTAL_API = "https://back.results.asiangames2026.org"
PORTAL_CHAMP = "AG2026"
PORTAL_HEADERS = {
    "User-Agent": us.MINISITE_HEADERS["User-Agent"],
    "Accept": "application/json, text/plain, */*",
    "Origin": PORTAL_SITE,
    "Referer": PORTAL_SITE + "/",
}

ASIA_QUERY = "regionType=area&region=asia"
# Far enough down the Asian list for any entrant who could matter to a call. An
# entrant ranked below 300 in Asia is not a contender, and is named as having no
# season mark on file rather than chased further.
MAX_PAGES = 3
# Nationality sits under a blank header, so it is read by position, as
# everywhere else that reads a toplist.
NAT = 5

# The portal's event codes ("M.110MHURD----------") for the 32 disciplines we
# hold data for. Anything else is not called; see not_called_reason().
EVENT_CODES = {
    "100M": "100m", "200M": "200m", "400M": "400m", "800M": "800m",
    "1500M": "1500m", "5000M": "5000m", "3000MST": "3000sc",
    "110MHURD": "110h", "100MHURD": "100h", "400MHURD": "400h",
    "HIGHJUMP": "HJ", "PLEVAULT": "PV", "LONGJUMP": "LJ", "TRPLJUMP": "TJ",
    "SHOTPUT": "SP", "DISCUS": "DT", "JAVELIN": "JT",
}
SEXES = {"M": "men", "W": "women"}

# World Athletics' Asian area, the Asian Athletics Association's members. The
# federations on the entry list are added at run time, so a code spelled
# differently here cannot fail a scrape that is fine. Palestine is listed under
# both codes in use.
ASIAN_NATIONS = frozenset({
    "AFG", "BAN", "BHU", "BRN", "BRU", "CAM", "CHN", "HKG", "INA", "IND", "IRI",
    "IRQ", "JOR", "JPN", "KAZ", "KGZ", "KOR", "KSA", "KUW", "LAO", "LBN", "MAC",
    "MAS", "MDV", "MGL", "MYA", "NEP", "OMA", "PAK", "PAL", "PLE", "PHI", "PRK",
    "QAT", "SGP", "SRI", "SYR", "THA", "TJK", "TKM", "TLS", "TPE", "UAE", "UZB",
    "VIE", "YEM",
})


def decode(raw):
    """A portal reply's JSON. Error replies, such as a 404 for a route the
    portal does not have, come as plain JSON, so that is tried second."""
    try:
        return json.loads(zlib.decompress(raw.decode("utf-8").encode("latin-1")).decode("utf-8"))
    except (UnicodeError, zlib.error, ValueError):
        return json.loads(raw.decode("utf-8"))


def portal_get(route):
    r = requests.get(f"{PORTAL_API}/s/{PORTAL_CHAMP}/en/{route}", headers=PORTAL_HEADERS, timeout=60)
    r.raise_for_status()
    return decode(r.content)


def disc_key(ev_key):
    """'M.110MHURD----------' -> 'men_110h', or None for an event we hold no data for."""
    sex, _, code = str(ev_key or "").partition(".")
    code = code.rstrip("-")
    if sex not in SEXES or code not in EVENT_CODES:
        return None
    key = f"{SEXES[sex]}_{EVENT_CODES[code]}"
    return key if key in live_fetcher.DISCIPLINE_URLS else None


def not_called_reason(event):
    """Why an event gets no call. A relay is a national team and the model is
    built on individuals. The rest (10,000m, marathons, walks, hammer, combined
    events) are events we hold no toplist, history or model for."""
    return "relay" if event.get("IsTeam") else "noData"


def fetch_programme(get=portal_get):
    """Every athletics event, in the organisers' order."""
    return get("ATH/disc/data").get("Events") or []


def fetch_entries(get=portal_get):
    """({portal event code: [entrant]}, federations whose detail page failed).

    entries/list is the whole list in one reply and decides who is entered.
    Each federation's page adds the World Athletics id and date of birth, which
    the list leaves out. A federation page that fails costs its athletes their
    id, not their place in the field: they are then matched by name and nation."""
    listing = get("ATH/entries/list")
    details, failed = {}, []
    for org in [o.get("Key") for o in listing.get("orgs") or [] if o.get("Key")]:
        try:
            page = get(f"ATH/entries/org/{org}")
        except (requests.RequestException, ValueError):
            failed.append(org)
            continue
        for ev in page.get("Events") or []:
            for p in ev.get("Partics") or []:
                details[str(p.get("Reg"))] = p

    by_event = {}
    for part in listing.get("participants") or []:
        if part.get("Type") != "A":      # relay teams
            continue
        d = details.get(str(part.get("Reg")), {})
        ifid = str(d.get("IFId") or "").strip()
        entrant = {
            "entryName": part.get("Name"),
            "givenName": d.get("GivenName"),
            "familyName": d.get("FamilyName"),
            "nat": part.get("Org"),
            "waId": int(ifid) if ifid.isdigit() else None,
            "birthDate": d.get("BirthDateRaw") or None,
        }
        for ins in part.get("Inscriptions") or []:
            if ins.get("EvKey"):
                by_event.setdefault(ins["EvKey"], []).append(entrant)
    return by_event, failed


def display_name(entrant):
    """"Shuhei TADA", the order World Athletics and the rest of the site use."""
    given = (entrant.get("givenName") or "").strip()
    family = (entrant.get("familyName") or "").strip()
    if given and family:
        return f"{given} {family.upper()}"
    return entrant.get("entryName") or ""


def wa_id(url):
    """The World Athletics id in a profile link. The world toplists link
    /athletes/athlete=14536762 and the area toplists link
    /athletes/pr-of-china/wenjie-wang-15074930. Reading only the first form
    matched almost nobody on the Asian list by id, on the first real run."""
    text = str(url or "")
    m = re.search(r"athlete=(\d+)", text) or re.search(r"-(\d+)/?$", text)
    return int(m.group(1)) if m else None


def _int(text):
    try:
        return int(float(str(text).strip()))
    except (TypeError, ValueError):
        return None


def fetch_page(url):
    r = requests.get(url, headers=us.MINISITE_HEADERS, timeout=60)
    r.raise_for_status()
    return r.text


def asian_toplist_url(key):
    return f"{live_fetcher.DISCIPLINE_URLS[key]}?{ASIA_QUERY}"


def scrape_asian_toplist(key, entrants, allowed, fetch=fetch_page, max_pages=MAX_PAGES):
    """(header, rows) of one discipline's 2026 Asian list, in the world
    snapshot's own column layout, or (None, []) when WA serves no table.

    Pages until every entrant is found or max_pages. Raises if any athlete on
    the list is from outside Asia: that means WA ignored the area filter and
    served the world list, which would silently rank the field against it."""
    url = asian_toplist_url(key)
    wanted_ids = {e["waId"] for e in entrants if e.get("waId")}
    wanted_names = {us.entry_name_key(e.get("entryName") or "") for e in entrants if not e.get("waId")}
    header, rows, seen = None, [], set()
    for page in range(1, max_pages + 1):
        page_headers, page_rows, urls = live_fetcher.parse_toplist_html(
            fetch(live_fetcher.toplist_page_url(url, page)))
        if not page_rows:
            break
        if header is None:
            header = list(page_headers) + ["discipline", "year", "ProfileURL"]
            comp = header.index("Competitor")
        width = len(page_headers)
        for cells, profile in zip(page_rows, urls):
            cells = (list(cells) + [""] * width)[:width]
            # A tie on a page boundary can put one athlete on two pages.
            ident = profile or f"{cells[comp]}|{cells[NAT]}"
            if ident in seen:
                continue
            seen.add(ident)
            rows.append(cells + [key, str(YEAR), profile or ""])
        found_ids = {wa_id(r[-1]) for r in rows}
        found_names = {us.entry_name_key(r[comp]) for r in rows}
        if wanted_ids <= found_ids and wanted_names <= found_names:
            break

    foreign = sorted({r[NAT] for r in rows if r[NAT] and r[NAT] not in allowed})
    if foreign:
        raise ValueError(f"{key}: the Asian list has athletes from {', '.join(foreign)}; "
                         "World Athletics did not apply the area filter")
    return header, rows


def _name_parts(name):
    """(family, given) token sets of a toplist name. World Athletics writes the
    family name in capitals: "Ebrahim Remaid ALZOFAIRI"."""
    words = str(name or "").split()
    family_flags = [w.isupper() and len(w.strip(".")) > 1 for w in words]
    family = " ".join(w for w, is_family in zip(words, family_flags) if is_family)
    given = " ".join(w for w, is_family in zip(words, family_flags) if not is_family)
    return us.entry_name_key(family), us.entry_name_key(given)


def same_person_by_name(entrant, toplist_name):
    """Whether an entry with no id can be the athlete on a toplist row: the same
    family name, and one side's given names all present on the other. That lets
    a middle name one source leaves out still match ("ALZOFAIRI Ebrahim" and
    "Ebrahim Remaid ALZOFAIRI"), and nothing looser."""
    family_row, given_row = _name_parts(toplist_name)
    entry = (entrant.get("entryName") or "").split()
    if entrant.get("familyName"):
        family = us.entry_name_key(entrant["familyName"])
        given = us.entry_name_key(entrant.get("givenName") or "")
    else:
        family = us.entry_name_key(entry[0] if entry else "")
        given = us.entry_name_key(" ".join(entry[1:]))
    if not family or family != family_row:
        return False
    return given <= given_row or given_row <= given


def match_entrants(entrants, header, rows):
    """[(athlete, toplist row or None)] for one event's entrants.

    An entrant with a World Athletics id is matched by that id and nothing else:
    if the id is not on the list the athlete is not on it, and a row with the
    same name would be somebody else. An entrant without one is matched by name
    within their own nation, on a row no id has claimed, and only when exactly
    one row fits: the same name first, then same_person_by_name()."""
    comp, mark, score, rank = (header.index(c) for c in ("Competitor", "Mark", "Results Score", "Rank"))
    by_id = {}
    for r in rows:
        rid = wa_id(r[-1])
        if rid:
            by_id.setdefault(rid, r)
    claimed = {id(by_id[e["waId"]]) for e in entrants if e.get("waId") in by_id}

    out = []
    for e in entrants:
        if e.get("waId"):
            row = by_id.get(e["waId"])
            how = "waId" if row is not None else None
        else:
            open_rows = [r for r in rows if id(r) not in claimed and r[NAT] == e.get("nat")]
            wanted = us.entry_name_key(e.get("entryName") or "")
            exact = [r for r in open_rows if us.entry_name_key(r[comp]) == wanted]
            fits = exact or [r for r in open_rows if same_person_by_name(e, r[comp])]
            row = fits[0] if len(fits) == 1 else None
            how = None if row is None else ("name" if exact else "nameParts")
            if row is not None:
                claimed.add(id(row))
        rid = e.get("waId") or (wa_id(row[-1]) if row is not None else None)
        out.append(({
            # The toplist's spelling when there is one: it is the name the
            # snapshot, the model and the athlete pages all key on.
            "name": row[comp] if row is not None else display_name(e),
            "entryName": e.get("entryName"),
            "nat": e.get("nat"),
            "waId": rid,
            "mark": row[mark] if row is not None else None,
            "score": _int(row[score]) if row is not None else None,
            "asiaRank": _int(row[rank]) if row is not None else None,
            "matchedBy": how,
            "profileUrl": ((row[-1] or None) if row is not None else None)
                          or (f"https://worldathletics.org/athletes/athlete={rid}" if rid else None),
        }, row))
    return out


def merge_snapshot(world_header, world_rows, asian_header, entrant_rows):
    """(merged rows, added rows): the world snapshot unchanged and first, then
    the Asian row of each entrant it lacks. Refuses toplists whose columns
    differ, since rows are copied by position."""
    if world_header != asian_header:
        raise ValueError(f"toplist columns differ: world {world_header} vs Asian {asian_header}")
    comp = world_header.index("Competitor")
    world_ids = {wa_id(r[-1]) for r in world_rows} - {None}
    world_names = {r[comp].strip().upper() for r in world_rows}
    added, seen = [], set()
    for row in entrant_rows:
        rid, name = wa_id(row[-1]), row[comp].strip().upper()
        # A name already on the world list is skipped even without an id match:
        # two rows under one name would be merged into one athlete downstream.
        if (rid and rid in world_ids) or name in world_names or (rid or name) in seen:
            continue
        seen.add(rid or name)
        # An Asian rank means nothing on the world list, so it is left blank.
        added.append([""] + list(row[1:]))
    return list(world_rows) + added, added


def read_rows(path):
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        return header, [row for row in reader if row]


def write_rows(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def _now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_event(get=portal_get, fetch=fetch_page, results=us.fetch_results, world_dir=None):
    """(event payload, {csv path: (header, rows)}). Nothing is written here, so
    a run that fails part-way leaves every saved file as it was."""
    world_dir = world_dir or WORLD_RAW_DIR
    programme = fetch_programme(get)
    entries, failed = fetch_entries(get)
    allowed = ASIAN_NATIONS | {e["nat"] for group in entries.values() for e in group if e.get("nat")}

    field, not_called, files = [], [], {}
    for ev in programme:
        ev_key = ev.get("EvKey")
        entrants = entries.get(ev_key) or []
        key = disc_key(ev_key)
        if key is None or not entrants:
            not_called.append({"evKey": ev_key, "label": ev.get("Desc"),
                               "reason": not_called_reason(ev) if key is None else "noEntries",
                               "entrants": len(entrants)})
            continue

        header, rows = scrape_asian_toplist(key, entrants, allowed, fetch)
        if header is None:
            raise RuntimeError(f"{key}: World Athletics served no Asian toplist")
        matches = match_entrants(entrants, header, rows)
        world_header, world_rows = read_rows(os.path.join(world_dir, f"{key}_{YEAR}.csv"))
        merged, added = merge_snapshot(world_header, world_rows, header,
                                       [row for _, row in matches if row is not None])
        files[os.path.join(ASIA_DIR, f"{key}_{YEAR}.csv")] = (header, rows)
        files[os.path.join(SNAPSHOT_DIR, f"{key}_{YEAR}.csv")] = (world_header, merged)

        athletes = [athlete for athlete, _ in matches]
        comp = header.index("Competitor")
        field.append({
            "evKey": ev_key,
            "discKey": key,
            "disciplineLabel": ev.get("Desc"),
            "sex": ev_key[0],
            "athletes": athletes,
            "withSeasonMark": sum(1 for a in athletes if a["score"] is not None),
            "addedToSnapshot": [row[comp] for row in added],
            "fieldSource": "entries",
        })

    found = results(field, competition_id=COMPETITION_ID)
    people = {(e.get("nat"), e.get("entryName")) for group in entries.values() for e in group}
    event = {
        **EVENT_META,
        "eventCount": len(programme),
        "entriesSource": PORTAL_SITE,
        "entriesFetchedAt": _now(),
        "entrants": len(people),
        "federations": len({nat for nat, _ in people}),
        # Federations whose athletes were matched by name because their page
        # with World Athletics ids did not load. Empty on a clean run.
        "federationsWithoutIds": failed,
        "fieldPublished": bool(field),
        "resultsAvailable": bool(found),
        "field": field,
        "notCalled": not_called,
        "results": found,
    }
    return event, files


def add_results(saved, results=us.fetch_results):
    """The saved event with fresh results and its field untouched. What the
    refresh button runs during the Games: re-reading 32 toplists on every press
    would be slow, and would move the snapshots the frozen call was made from."""
    found = results(saved.get("field"), competition_id=COMPETITION_ID)
    return {**saved, "results": found, "resultsAvailable": bool(found), "resultsFetchedAt": _now()}


def lost_data(new, old):
    """What this run dropped that the saved file has, or None. A field or a
    result does not un-publish, so losing one means a fetch failed."""
    lost = [key for key in ("field", "results") if (old or {}).get(key) and not new.get(key)]
    return ", ".join(lost) or None


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-only", action="store_true",
                        help="Add results to the saved event.json and change nothing else.")
    args = parser.parse_args(argv)

    try:
        with open(OUT_PATH, encoding="utf-8") as f:
            saved = json.load(f)
    except (OSError, json.JSONDecodeError):
        saved = None

    if args.results_only:
        if not (saved or {}).get("field"):
            sys.exit(f"  no saved field in {OUT_PATH}; run without --results-only first")
        event, files = add_results(saved), {}
    else:
        event, files = build_event()

    lost = lost_data(event, saved)
    if lost:
        print(f"  NOT SAVED: this run has no {lost}, which {OUT_PATH} still has. "
              "A fetch failed; the saved files are kept.")
        return 1

    for path, (header, rows) in files.items():
        write_rows(path, header, rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(event, f, indent=1, ensure_ascii=False)

    field = event.get("field") or []
    print(f"  {event.get('entrants')} entrants from {event.get('federations')} federations; "
          f"{len(field)} events called, {len(event.get('notCalled') or [])} not")
    if event.get("federationsWithoutIds"):
        print(f"  WARNING: no World Athletics ids for {', '.join(event['federationsWithoutIds'])}")
    for ev in field:
        print(f"    {ev['disciplineLabel']:<30} {len(ev['athletes']):>3} entered, "
              f"{ev['withSeasonMark']:>3} with a 2026 mark, {len(ev['addedToSnapshot']):>2} added to the snapshot")
    print(f"  results: {len(event.get('results') or [])} rows")
    print(f"  Saved -> {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
