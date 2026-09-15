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

ENTRANTS THE ASIAN LIST DOES NOT PLACE
On the first real run 179 of 599 entries had no mark: 156 carried no World
Athletics id, so could only be matched by name, and 23 had one but sat below the
top 300 in Asia. complete_matches() looks each of them up at World Athletics
directly: an id through WA's own athlete search, confirmed by date of birth, and
a season best from the athlete's profile, whose resultScore is the toplist's
Results Score. Whoever is still unranked carries the reason.

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
import time
import unicodedata
import zlib
from datetime import datetime, timezone

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import championships  # noqa: E402
import dl_final_results_scraper as dlr  # noqa: E402
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

# The portal's event codes ("M.110MHURD----------") for the disciplines we hold
# a 2026 toplist for. Anything else is not called; see not_called_reason().
# The 10,000m and the hammer joined on 2026-09-14. With no race history for
# either, the 6-of-8 rule can only ever rank them on points.
EVENT_CODES = {
    "100M": "100m", "200M": "200m", "400M": "400m", "800M": "800m",
    "1500M": "1500m", "5000M": "5000m", "10000M": "10000m", "3000MST": "3000sc",
    "110MHURD": "110h", "100MHURD": "100h", "400MHURD": "400h",
    "HIGHJUMP": "HJ", "PLEVAULT": "PV", "LONGJUMP": "LJ", "TRPLJUMP": "TJ",
    "SHOTPUT": "SP", "DISCUS": "DT", "JAVELIN": "JT", "HAMMER": "HT",
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
    built on individuals. The rest (marathons, walks, combined events) are
    events we hold no toplist, history or model for."""
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
    """"Shuhei TADA", the order World Athletics and the rest of the site use.
    A "." in either part holds the place of a name a single-named athlete does
    not have: the entry list gives Sangay of Bhutan the family name ".", which
    read "Sangay ." on the page."""
    given = (entrant.get("givenName") or "").strip().strip(".").strip()
    family = (entrant.get("familyName") or "").strip().strip(".").strip()
    if given and family:
        return f"{given} {family.upper()}"
    return given or family.upper() or (entrant.get("entryName") or "")


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


# Last season's Asian list ranks an entrant with no mark this season. See
# asian_games_predictions.last_season_rule.
LAST_YEAR = YEAR - 1


def asian_toplist_url(key, year=YEAR):
    base = live_fetcher.DISCIPLINE_URLS[key].rsplit("/", 1)[0]
    return f"{base}/{year}?{ASIA_QUERY}"


def scrape_asian_toplist(key, entrants, allowed, fetch=fetch_page, max_pages=MAX_PAGES, year=YEAR):
    """(header, rows) of one discipline's Asian list for `year`, in the world
    snapshot's own column layout, or (None, []) when WA serves no table.

    Pages until every entrant is found or max_pages. With no `entrants` (last
    season's list, read for whoever is on it) it pages to max_pages. Raises if
    any athlete on the list is from outside Asia: that means WA ignored the
    area filter and served the world list, which would silently rank the field
    against it."""
    url = asian_toplist_url(key, year)
    entrants = entrants or []
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
            rows.append(cells + [key, str(year), profile or ""])
        found_ids = {wa_id(r[-1]) for r in rows}
        found_names = {us.entry_name_key(r[comp]) for r in rows}
        if entrants and wanted_ids <= found_ids and wanted_names <= found_names:
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


SEARCH_QUERY = """query SearchCompetitors($query: String, $countryCode: String) {
  searchCompetitors(query: $query, countryCode: $countryCode) {
    aaAthleteId familyName givenName birthDate gender country
  }
}"""

SEASON_QUERY = """query SeasonResults($id: Int) {
  getSingleCompetitor(id: $id) {
    basicData { birthDate givenName familyName }
    resultsByYear {
      resultsByEvent {
        discipline indoor
        results { date venue place mark wind resultScore notLegal }
      }
    }
  }
}"""

# World Athletics' own name for each discipline ("100 Metres", "10,000 Metres"),
# for reading a profile. The inverse of the map their results are read through.
WA_EVENT_NAMES = {key: name for (_sex, name), key in dlr.WA_EVENT_TO_KEY.items()}
# The pause athlete_profile_scraper.py keeps between GraphQL requests. WA's
# CloudFront answers a burst with an HTML error page.
LOOKUP_PAUSE = 0.35
_MONTHS = {m: i for i, m in enumerate(
    ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"), 1)}


def wa_search(query, nat):
    time.sleep(LOOKUP_PAUSE)
    data = dlr.graphql("SearchCompetitors", {"query": query, "countryCode": nat}, SEARCH_QUERY)
    return data.get("searchCompetitors") or []


def wa_season(athlete_id):
    time.sleep(LOOKUP_PAUSE)
    return dlr.graphql("SeasonResults", {"id": int(athlete_id)}, SEASON_QUERY).get("getSingleCompetitor")


def birth_date_iso(text):
    """'18 AUG 2006' -> '2006-08-18'. A year on its own stays a year: World
    Athletics knows only the birth year of some athletes."""
    parts = str(text or "").split()
    if len(parts) == 3 and parts[0].isdigit() and parts[1].upper() in _MONTHS and parts[2].isdigit():
        return f"{parts[2]}-{_MONTHS[parts[1].upper()]:02d}-{int(parts[0]):02d}"
    if len(parts) == 1 and len(parts[0]) == 4 and parts[0].isdigit():
        return parts[0]
    return None


# Name particles shared by unrelated people: "AL GHALBAN" and "AL HASSAN" agree
# on "AL" and nothing else, which is no agreement at all.
_NAME_PARTICLES = frozenset({"AL", "EL", "BIN", "BINT", "IBN", "ABU", "BEN", "DE", "DA", "DI", "VAN"})


def _tokens(text):
    s = unicodedata.normalize("NFD", str(text or ""))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return {w for w in re.split(r"[^A-Z0-9]+", s.upper()) if len(w) > 1 and w not in _NAME_PARTICLES}


def find_wa_id(entrant, sex, search=wa_search):
    """The World Athletics id of an entrant the entry list gave none, or None.

    Looked up in WA's own athlete search within the entrant's nation, and taken
    only when exactly one result is the same sex, was born on the same day and
    shares a name with the entry. The date does the work. WA spells Sorsy
    PHOMPHAKDI of Laos "PHOMPAKDI", and a search on the entry's spelling still
    finds him, born on 7 December 2001 as the entry says. Where WA knows only a
    birth year, two names must agree.

    Which part is the family name is not compared. For Mongolia the two lists
    swap them: the entry "DAVAANYAM Myagmarsuren" is given name Dawaanyam and
    family name MYAGMARSUREN at WA, same birth date. Comparing part with part
    turned that away on the first run with this lookup (2026-09-14)."""
    birth = entrant.get("birthDate")
    nat = entrant.get("nat")
    if not birth or not nat:
        return None
    entry_words = (entrant.get("entryName") or "").split()
    family = entrant.get("familyName") or (entry_words[0] if entry_words else "")
    given = entrant.get("givenName") or " ".join(entry_words[1:])
    wanted = _tokens(family) | _tokens(given)
    for query in dict.fromkeys(q for q in (family, f"{given} {family}".strip(), given) if q):
        fits = set()
        for c in search(query, nat):
            if c.get("country") != nat or not str(c.get("aaAthleteId") or "").isdigit():
                continue
            if sex and str(c.get("gender") or "")[:1].upper() != sex:
                continue
            born = birth_date_iso(c.get("birthDate"))
            shared = wanted & (_tokens(c.get("familyName")) | _tokens(c.get("givenName")))
            if born == birth and shared:
                fits.add(int(c["aaAthleteId"]))
            elif born and len(born) == 4 and birth.startswith(born) and len(shared) >= 2:
                fits.add(int(c["aaAthleteId"]))
        if len(fits) == 1:
            return fits.pop()
        if fits:
            return None
    return None


# World Athletics leaves a profile's `indoor` flag empty on every event group
# (all 3,293 groups fetched on 2026-09-15, and all 14,897 in data/field/seasons)
# and lists indoor results beside outdoor ones, marked only by "(i)" after the
# venue: "Antequera (i)", but also "Madrid Indoor Meeting, Gallur, Madrid (i) -
# World Athletics Indoor Tour", where a series label follows it. 112 of the
# first 29,472 fetched results were that second kind, so "(i)" is read as a
# token anywhere, not only at the end. athlete_career.py reads the "(i)" too.
_INDOOR_MARK = re.compile(r"(?:^|\s)\(i\)(?=\s|$)")


def is_indoor(result, group=None):
    """Whether one profile result was set indoors."""
    if group is not None and group.get("indoor"):
        return True
    return any(_INDOOR_MARK.search(str(result.get(field) or "")) for field in ("venue", "competition"))


def season_best(profile, key, year=YEAR):
    """An athlete's best legal outdoor result this season in one discipline, as
    their World Athletics profile has it, or None. Indoors is read by
    is_indoor, not by the profile's flag.

    The profile's resultScore is the toplist's Results Score. Checked on
    2026-09-14 on two men's 100m entrants who are on the Asian list: 1169 and
    1066 in both places. So a mark found here ranks on the same scale."""
    wanted = WA_EVENT_NAMES.get(key)
    best = None
    for group in ((profile or {}).get("resultsByYear") or {}).get("resultsByEvent") or []:
        if group.get("discipline") != wanted:
            continue
        for result in group.get("results") or []:
            score = result.get("resultScore")
            if (result.get("notLegal") or not score or is_indoor(result, group)
                    or not str(result.get("date") or "").endswith(str(year))):
                continue
            if best is None or score > best["resultScore"]:
                best = result
    return best


def profile_name(profile, fallback):
    """The athlete's name as World Athletics writes it, "Sorsy PHOMPAKDI". Their
    results carry this spelling, and the results are graded by name."""
    basic = (profile or {}).get("basicData") or {}
    given, family = (basic.get("givenName") or "").strip(), (basic.get("familyName") or "").strip()
    return f"{given} {family.upper()}" if given and family else fallback


def profile_row(header, key, name, nat, athlete_id, profile, result):
    """A toplist-shaped row for a mark read off a profile, so the merged
    snapshot, the model and the athlete pages treat it like any other row. The
    rank stays blank: this athlete is on no list, Asian or world."""
    cells = [""] * (len(header) - 3)
    values = {
        "Mark": result.get("mark"), "WIND": result.get("wind"), "Competitor": name,
        "DOB": ((profile or {}).get("basicData") or {}).get("birthDate"),
        "Pos": str(result.get("place") or "").rstrip("."), "Venue": result.get("venue"),
        "Date": result.get("date"), "Results Score": str(result.get("resultScore")),
    }
    for col, value in values.items():
        if col in header[:len(cells)]:
            cells[header.index(col)] = value or ""
    cells[NAT] = nat or ""
    return cells + [key, str(YEAR), f"https://worldathletics.org/athletes/athlete={athlete_id}"]


def complete_matches(matches, entrants, header, rows, key, sex, search=wa_search, season=wa_season, cache=None):
    """match_entrants()'s [(athlete, row)], with each entrant the Asian list did
    not place looked up at World Athletics directly:

      1. no id: WA's athlete search, confirmed by date of birth (find_wa_id);
      2. an id that is on the list after all, under a spelling the name match
         missed: that toplist row;
      3. otherwise the athlete's profile, their best legal outdoor result this
         season, as a toplist-shaped row (profile_row).

    Every athlete gets `unranked`: None when ranked, "notFound" when WA has no
    athlete to match, "noMark" when their profile has no 2026 result in this
    event, and "lookupFailed" when a request failed. `cache` is shared across
    events, so an athlete entered twice is looked up once."""
    cache = {} if cache is None else cache
    comp, mark_col, score_col, rank_col = (header.index(c) for c in ("Competitor", "Mark", "Results Score", "Rank"))
    by_id = {}
    for r in rows:
        rid = wa_id(r[-1])
        if rid:
            by_id.setdefault(rid, r)
    claimed = {id(r) for _, r in matches if r is not None}

    out = []
    for (athlete, row), entrant in zip(matches, entrants):
        athlete = {**athlete, "unranked": None}
        if row is not None:
            out.append((athlete, row))
            continue
        try:
            athlete_id = athlete.get("waId")
            if not athlete_id:
                ident = ("search", sex, entrant.get("nat"), entrant.get("entryName"))
                if ident not in cache:
                    cache[ident] = find_wa_id(entrant, sex, search)
                athlete_id = cache[ident]
            if not athlete_id:
                out.append(({**athlete, "unranked": "notFound"}, None))
                continue
            athlete["waId"] = athlete_id
            athlete["profileUrl"] = athlete.get("profileUrl") or f"https://worldathletics.org/athletes/athlete={athlete_id}"

            listed = by_id.get(athlete_id)
            if listed is not None and id(listed) not in claimed:
                claimed.add(id(listed))
                athlete.update(name=listed[comp], mark=listed[mark_col], score=_int(listed[score_col]),
                               asiaRank=_int(listed[rank_col]), matchedBy="search",
                               profileUrl=listed[-1] or athlete["profileUrl"])
                out.append((athlete, listed))
                continue

            if ("profile", athlete_id) not in cache:
                cache[("profile", athlete_id)] = season(athlete_id)
            profile = cache[("profile", athlete_id)]
            best = season_best(profile, key)
            if best is None:
                # Under World Athletics' spelling, as wherever they are ranked.
                # Otherwise one Mongolian entrant read as two people: "Dawaanyam
                # MYAGMARSUREN" ranked in the 1500m, "Myagmarsuren DAVAANYAM"
                # unranked in the 800m.
                out.append(({**athlete, "name": profile_name(profile, athlete["name"]),
                             "unranked": "noMark"}, None))
                continue
            name = profile_name(profile, athlete["name"])
            athlete.update(name=name, mark=best.get("mark"), score=int(best["resultScore"]),
                           asiaRank=None, matchedBy="profile")
            out.append((athlete, profile_row(header, key, name, athlete.get("nat"), athlete_id, profile, best)))
        except (requests.RequestException, RuntimeError, ValueError, KeyError, TypeError):
            out.append(({**athlete, "unranked": "lookupFailed"}, None))
    return out


def world_spelling(matches, world_header, world_rows):
    """[(athlete, row)] with each athlete under the world toplist's spelling of
    their name, wherever their World Athletics id is on that list.

    The two lists can spell one athlete two ways: "Taepoong NAM" in Asia and
    "Tae-poong NAM" worldwide, "Yu sun JEONG" and "Yusun JEONG". merge_snapshot
    keeps the world row, while the model, the athlete pages and the grading of
    results all key on the name. Under the Asian spelling the men's javelin call,
    a model event, could not score Taepoong NAM, and Yusun JEONG had no page
    (found on the first run with lookups, 2026-09-14)."""
    comp = world_header.index("Competitor")
    by_id = {}
    for r in world_rows:
        rid = wa_id(r[-1])
        if rid:
            by_id.setdefault(rid, r[comp])
    return [({**athlete, "name": by_id.get(athlete.get("waId"), athlete["name"])}, row)
            for athlete, row in matches]


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


def build_event(get=portal_get, fetch=fetch_page, results=us.fetch_results, world_dir=None,
                search=wa_search, season=wa_season):
    """(event payload, {csv path: (header, rows)}). Nothing is written here, so
    a run that fails part-way leaves every saved file as it was."""
    world_dir = world_dir or WORLD_RAW_DIR
    programme = fetch_programme(get)
    entries, failed = fetch_entries(get)
    allowed = ASIAN_NATIONS | {e["nat"] for group in entries.values() for e in group if e.get("nat")}

    field, not_called, files, lookups = [], [], {}, {}
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
        last_header, last_rows = scrape_asian_toplist(key, None, allowed, fetch, year=LAST_YEAR)
        if last_header is not None:
            files[os.path.join(ASIA_DIR, f"{key}_{LAST_YEAR}.csv")] = (last_header, last_rows)
        matches = complete_matches(match_entrants(entrants, header, rows), entrants, header, rows,
                                   key, ev_key[0], search, season, lookups)
        world_header, world_rows = read_rows(os.path.join(world_dir, f"{key}_{YEAR}.csv"))
        matches = world_spelling(matches, world_header, world_rows)
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
            # Marks read off a World Athletics profile, not the Asian list.
            "markFromProfile": sum(1 for a in athletes if a.get("matchedBy") == "profile"),
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
        # Entrants whose World Athletics lookup failed outright, so their
        # "unranked" may be a network error rather than a fact. 0 on a clean run.
        "lookupFailures": sum(1 for ev in field for a in ev["athletes"] if a.get("unranked") == "lookupFailed"),
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
    parser.add_argument("--last-season", action="store_true",
                        help=f"Fetch only the {LAST_YEAR} Asian lists for the saved field's events.")
    args = parser.parse_args(argv)

    try:
        with open(OUT_PATH, encoding="utf-8") as f:
            saved = json.load(f)
    except (OSError, json.JSONDecodeError):
        saved = None

    if args.last_season:
        field = (saved or {}).get("field") or []
        if not field:
            sys.exit(f"  no saved field in {OUT_PATH}; run without --last-season first")
        allowed = ASIAN_NATIONS | {a["nat"] for ev in field for a in ev["athletes"] if a.get("nat")}
        for ev in field:
            header, rows = scrape_asian_toplist(ev["discKey"], None, allowed, year=LAST_YEAR)
            if header is None:
                print(f"    {ev['discKey']}: World Athletics served no {LAST_YEAR} list")
                continue
            write_rows(os.path.join(ASIA_DIR, f"{ev['discKey']}_{LAST_YEAR}.csv"), header, rows)
            print(f"    {ev['discKey']}: {len(rows)} athletes on the {LAST_YEAR} Asian list")
        return 0

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
              f"{ev['withSeasonMark']:>3} with a 2026 mark ({ev.get('markFromProfile', 0):>2} from a profile), "
              f"{len(ev['addedToSnapshot']):>2} added to the snapshot")
    reasons = {}
    for ev in field:
        for a in ev["athletes"]:
            if a.get("unranked"):
                reasons[a["unranked"]] = reasons.get(a["unranked"], 0) + 1
    print(f"  unranked entries by reason: {reasons or 'none'}")
    if event.get("lookupFailures"):
        print(f"  WARNING: {event['lookupFailures']} World Athletics lookups failed; re-run before freezing")
    print(f"  results: {len(event.get('results') or [])} rows")
    print(f"  Saved -> {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
