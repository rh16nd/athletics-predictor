"""
ultimate_scraper.py -- data for the World Athletics Ultimate Championship
(Budapest, 11-13 September 2026), the site's focus now that the 2026 Diamond
League is over.

It reads from World Athletics' own public GraphQL API (the same endpoint and
public x-api-key the DL Final scraper uses -- see dl_final_results_scraper.py)
plus a few facts verified from WA's competition minisite and Wikipedia. It is
built to be run repeatedly: the timetable, start lists and results are empty
until WA publishes them (the event is days away as of this writing), so the
scraper degrades to "not published yet" rather than inventing anything.

    Verified event identity (WA minisite __NEXT_DATA__, 2026-09-06):
      WAW event id      8592
      competition id    7212925   (eventId_WA -- the getCalendarCompetitionResults key)
      dates             2026-09-11 .. 2026-09-13  (Europe/Budapest, evening sessions)
      venue             National Athletics Centre, Budapest (HUN)
    Format (WA + en.wikipedia.org/wiki/2026_World_Athletics_Ultimate_Championship):
      28 events, US$10M purse, fields of 16 (track) / 8 (field).
      Qualification: direct slots to the 2024 Olympic champion, the 2025 World
      champion and the 2026 Diamond League Final winner per event; the rest by
      World Athletics Rankings (2 Sep 2025 - 1 Sep 2026).

Usage:
    python src/ultimate_scraper.py
Writes data/ultimate/event.json.

Start lists: see fetch_startlists() -- wired and schema-verified, empty until
World Athletics puts the entries in their API rather than only in a press
release.
"""
import csv
import json
import os
import re
import sys
import unicodedata
from html import unescape as hunescape

import requests

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import dl_final_results_scraper as dlr  # noqa: E402 -- reuse graphql()/HEADERS

COMPETITION_ID = 7212925          # getCalendarCompetitionResults key (eventId_WA)
WAW_EVENT_ID = 8592
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "ultimate")
OUT_PATH = os.path.join(OUT_DIR, "event.json")
DL_FINAL_RESULTS = os.path.join(os.path.dirname(__file__), "..", "data", "dl_final_2026_results.csv")

# Verified facts (see module docstring for sources). Kept here rather than
# scraped every run because they are fixed for this edition; the dynamic data
# (timetable/field/results) is fetched live below.
EVENT_META = {
    "competitionId": COMPETITION_ID,
    "name": "World Athletics Ultimate Championship",
    "shortName": "Budapest 26",
    "venue": "National Athletics Centre",
    "city": "Budapest",
    "country": "HUN",
    "startDate": "2026-09-11",
    "endDate": "2026-09-13",
    "timezone": "Europe/Budapest",
    "eventCount": 28,
    "prizeUSD": 10_000_000,
    "trackFieldSize": 16,
    "fieldFieldSize": 8,
    "sessions": 3,
}

TIMETABLE_QUERY = """query UcTimetable($e: Int) {
  getEventTimetable(eventId: $e) {
    phaseName sexName phaseDateAndTime phaseSessionName
    isStartlistPublished isResultPublished
    discipline { name isTrack isField }
  }
}"""

# The phase list, which carries the arguments a start list has to be asked for
# by. Separate from the timetable above because getEventTimetable does not
# return phaseCode or the discipline slug, and both are required below.
PHASES_QUERY = """query UcPhases($e: Int) {
  getEventPhases(eventId: $e) {
    phaseCode phaseName sexCode isStartlistPublished
    discipline { name nameUrlSlug }
  }
}"""

# One phase's start list. Verified against the live schema 2026-09-07 by
# introspection, which WA leaves open (176 queries): `disciplineCode` wants the
# discipline's URL SLUG ("100-metres"), not its id ("100") or its name, and
# `phaseCode` is lower case ("f", "sf"). Passing the id returns an unhandled
# Lambda error; passing the name returns a row of nulls. Neither looks like a
# wrong argument, which is why this is written down.
STARTLIST_QUERY = """query UcStartlist($e: Int, $d: String, $s: String, $p: String) {
  getEventPhaseByDiscipline(eventId: $e, disciplineCode: $d, sexCode: $s, phaseCode: $p) {
    phaseName isStartlistPublished
    units {
      unitId unitTypeName
      startlist {
        competitorName competitorCountryCode competitorId_WA
        personalBestMark seasonBestMark worldRanking bib startlistOrder
      }
    }
  }
}"""


def fetch_timetable():
    """The Ultimate's session timetable, or [] until WA publishes it. Each row
    is one phase (an event's round), carrying whether its start list / result
    is up yet -- which is how we detect the field going live.

    NOTE the id: getEventTimetable keys on the WAW event id (WAW_EVENT_ID),
    NOT the competition id that getCalendarCompetitionResults uses. They are
    different numbers for the same meeting and WA's API accepts either without
    complaint -- the wrong one simply returns an empty timetable, which reads
    exactly like "not published yet". This was passing COMPETITION_ID, so
    fieldPublished could never have flipped to True once WA published; the page
    would have sat on "start lists aren't up" through the whole championship.
    Confirmed against the 2026 World Athletics Relays, where the event id (8656)
    returns 18 phases and its competition id (7216920) returns none."""
    try:
        data = dlr.graphql("UcTimetable", {"e": WAW_EVENT_ID}, TIMETABLE_QUERY)
        return data.get("getEventTimetable") or []
    except Exception as e:
        print(f"  timetable fetch failed ({str(e)[:60]}) -- treating as not published")
        return []


def entrants_by_key(field):
    """{discipline key: {entrant names, lowercased}} from the qualified field,
    used to tell the championship's own race apart from any other race sharing
    the meeting. Empty when the field has not been published."""
    out = {}
    for disc in field or []:
        key = disc.get("discKey")
        if not key:
            continue
        out.setdefault(key, set()).update(
            (a.get("name") or "").strip().lower() for a in disc.get("athletes") or []
        )
    return out


def pick_final(races, entrants):
    """Which of several races labelled "Final" is the one that decides the
    championship?

    Taking the first was wrong, and a dry run against a finished meeting proved
    it rather than a review catching it: pointed at the 2026 Diamond League
    Final, this returned the BELGIAN NATIONAL women's 100m -- 12.21, 12.39,
    12.56 -- instead of the Diamond League race won in 11.06. Both are labelled
    "Final" in the same feed, and the national one came first.

    dl_final_results_scraper avoids that by keeping only groups whose
    rankingCategory is "DF", which is a Diamond League concept and no use at a
    championship. The discriminator that works anywhere is the entry list: the
    real race is the one contested by the athletes who qualified. Falls back to
    the fullest race, then the first, when there is no field to compare against.
    """
    finals = [r for r in races if r.get("race") == "Final"]
    if len(finals) <= 1:
        return finals[0] if finals else None

    def score(race):
        names = {
            ((res.get("competitor") or {}).get("name") or "").strip().lower()
            for res in race.get("results") or []
        }
        return (len(names & entrants), len(race.get("results") or []))

    best = max(finals, key=score)
    print(f"    {len(finals)} races labelled Final; kept the one with "
          f"{score(best)[0]} of the entered athletes")
    return best


def fetch_results(field=None):
    """Final results per contested event, or [] until the meeting runs. Reuses
    the DL Final scraper's exact per-day/per-event walk (getCalendarCompetitionResults),
    with the Mile counted as the 1500m the same way a championship Final needs.

    `field` is fetch_qualified_field()'s output, used only to choose between
    several races that all call themselves the Final -- see pick_final."""
    rows = []
    entrants = entrants_by_key(field)
    try:
        day_data = dlr.graphql(
            "getCalendarCompetitionResults",
            {"competitionId": COMPETITION_ID, "day": None, "eventId": None},
            dlr.RESULTS_QUERY,
        )["getCalendarCompetitionResults"]
        days = [d["day"] for d in day_data["options"]["days"]] or [None]
        # Every candidate race first, then one choice per discipline. Deciding
        # as we walk meant whichever group the feed happened to list first won,
        # which is exactly how the national race got in.
        candidates = {}
        for day in days:
            data = dlr.graphql(
                "getCalendarCompetitionResults",
                {"competitionId": COMPETITION_ID, "day": day, "eventId": None},
                dlr.RESULTS_QUERY,
            )["getCalendarCompetitionResults"]
            for group in data["eventTitles"]:
                for event in group["events"]:
                    key = dlr.resolve_discipline_key(event["gender"], event["event"], mile_as_1500=True)
                    if key is None:
                        continue
                    candidates.setdefault(key, []).extend(event["races"])
        for key, races in candidates.items():
            final = pick_final(races, entrants.get(key, set()))
            if final is None:
                continue
            for res in final["results"]:
                name = (res.get("competitor") or {}).get("name")
                if not name:
                    continue
                rows.append({
                    "discipline": key,
                    "athlete_name": name,
                    "place": (res.get("place") or "").rstrip("."),
                    "mark": res.get("mark"),
                    "nationality": res.get("nationality"),
                })
    except Exception as e:
        print(f"  results fetch failed ({str(e)[:60]}) -- treating as not run yet")
        return []
    return rows


MINISITE_URL = "https://worldathletics.org/competitions/world-athletics-ultimate-championship"

# WA fronts the minisite with CloudFront, which refuses requests without a
# browser-shaped User-Agent. This is a public marketing page, fetched once per
# run, and parsed from its own embedded JSON rather than its rendered markup.
MINISITE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-GB,en;q=0.9",
}

# ---------------------------------------------------------------------------
# THE OFFICIAL ENTRY LIST.
#
# fetch_startlists() below asks the competition API and gets nothing: all 40
# phases report isStartlistPublished false three days out. But the entry list
# DOES exist -- as a PDF linked from a press release, revised hourly, and it is
# materially different from the qualification list this file already reads.
#
# Qualification says who is ELIGIBLE. Entries say who is RUNNING. Checked
# 2026-09-08 against the 7 September 15:30 CEST revision: 27 athletes we were
# projecting are not entered, including **Tara Davis-Woodhall, our number one
# in the women's long jump at 56.6%**, and 26 entered athletes were missing
# from our field entirely. Reported by the user, who had seen the previews and
# knew Rai Benjamin was in the flat 400m and not the hurdles -- which is
# exactly what this list says, and what the qualification list could not.
# ---------------------------------------------------------------------------
ENTRY_RELEASE_HINT = "final-entry-lists"
ENTRY_DOC_RE = re.compile(r'href="(https://assets\.aws\.worldathletics\.org/document/[^"]+\.pdf)"'
                          r'[^>]*>\s*entries by event and world ranking', re.I)
_ENTRY_ANY_DOC = re.compile(r'https://assets\.aws\.worldathletics\.org/document/[^"\s\\]+\.pdf')

# "100 Metres16" -- the event name with its athlete count glued on, because the
# PDF is two-column and pypdf flattens it. "MEN51" is the sex divider.
ENTRY_SEX_RE = re.compile(r"^(MEN|WOMEN)(\d+)$")
ENTRY_EVENT_RE = re.compile(
    r"^([\d,]*\s*(?:Metres Hurdles|Metres Steeplechase|Metres)"
    r"|High Jump|Pole Vault|Long Jump|Triple Jump|Shot Put"
    r"|Discus Throw|Hammer Throw|Javelin Throw)(\d+)$")
# bib, name, 3-letter nation, then a mark
ENTRY_ATHLETE_RE = re.compile(r"^(\d{2,3})\s+(.+?)\s+([A-Z]{3})\s+[\d.:]")


def find_entry_list_pdf():
    """The URL of WA's current "entries by event" PDF, or None.

    Found by following the press release rather than hard-coding a document id:
    the list is revised hourly and each revision is a new document."""
    try:
        r = requests.get(MINISITE_URL + "/2026/news", headers=MINISITE_HEADERS, timeout=40)
        page = r.text if r.status_code == 200 else ""
        link = re.search(rf'href="([^"]*{ENTRY_RELEASE_HINT}[^"]*)"', page)
        url = link.group(1) if link else None
        if url and url.startswith("/"):
            url = "https://worldathletics.org" + url
        if not url:
            r = requests.get(MINISITE_URL, headers=MINISITE_HEADERS, timeout=40)
            link = re.search(rf'href="([^"]*{ENTRY_RELEASE_HINT}[^"]*)"', r.text)
            url = link.group(1) if link else None
        if not url:
            print("  entry list: no press release found")
            return None
        art = requests.get(url, headers=MINISITE_HEADERS, timeout=40)
        if art.status_code != 200:
            print(f"  entry list: press release HTTP {art.status_code}")
            return None
        body = art.text
        m = ENTRY_DOC_RE.search(body)
        if m:
            return m.group(1)
        # Fall back to the first document link, but SAY SO -- the three PDFs
        # are the same entries sorted three ways, so the wrong one is still
        # the right athletes, but a silent fallback is how a parser starts
        # reading a federation listing as an event listing.
        any_doc = _ENTRY_ANY_DOC.search(body)
        if any_doc:
            print("  entry list: named link not found, using the first document link")
            return any_doc.group(0)
    except Exception as e:
        print(f"  entry list: lookup failed ({str(e)[:60]})")
    return None


def parse_entry_pdf(data):
    """{event label: [athlete names]} from the entry-list PDF's bytes.

    Returns {} rather than a partial field if any event's parsed count
    disagrees with the count the PDF prints for itself. A short entry list is
    indistinguishable from a set of withdrawals downstream, and inventing
    withdrawals is worse than having no entry list at all."""
    try:
        from pypdf import PdfReader
    except ImportError:
        print("  entry list: pypdf not installed, skipping")
        return {}
    import io as _io
    try:
        reader = PdfReader(_io.BytesIO(data))
        text = "\n".join(p.extract_text() or "" for p in reader.pages)
    except Exception as e:
        print(f"  entry list: could not read the PDF ({str(e)[:60]})")
        return {}

    sex = event = None
    out, expected = {}, {}
    for line in (l.strip() for l in text.splitlines()):
        if not line:
            continue
        m = ENTRY_SEX_RE.match(line.upper())
        if m:
            sex = m.group(1).title()
            continue
        m = ENTRY_EVENT_RE.match(line)
        if m and sex:
            event = f"{sex}'s {m.group(1).strip()}"
            expected[event] = int(m.group(2))
            out.setdefault(event, [])
            continue
        m = ENTRY_ATHLETE_RE.match(line)
        if m and event:
            out[event].append(m.group(2).strip())

    mismatched = [e for e, names in out.items() if len(names) != expected.get(e)]
    if mismatched or not out:
        print(f"  entry list: parse disagrees with the PDF's own counts "
              f"({len(mismatched)} events) -- discarding rather than guessing")
        return {}
    return out


def entry_name_key(name):
    """An order-insensitive key for matching a name across the two sources.

    WA's PDF writes some names surname-first ("NAKAJIMA Yuki Joseph", "YAN
    Ziyi") where the qualification API has them given-name-first. Matching on
    the string made seven athletes read as BOTH a withdrawal and a new entry --
    which would have deleted seven people who are running and invented seven
    who are not. A token set is immune to the ordering and to the diacritics."""
    s = unicodedata.normalize("NFKD", str(name))
    s = "".join(c for c in s if not unicodedata.combining(c)).upper()
    return frozenset(t for t in s.replace(".", " ").replace("-", " ").split() if t)


def apply_entry_list(event, entered):
    """One event's field, rebuilt from who is actually ENTERED.

    Qualification says who is eligible; entries say who is running, and on
    2026-09-08 they disagreed about 19 athletes per side. Keeps each qualified
    athlete's route and ranking score where WA still lists them, carries over
    anyone entered who never appeared in the qualification list (replacements
    do not get a qualification route), and records the rest as `notEntered` so
    the page can name them instead of just dropping them."""
    by_key = {entry_name_key(a["name"]): a for a in event.get("athletes") or []}
    kept, added = [], []
    for name in entered:
        k = entry_name_key(name)
        if k in by_key:
            kept.append(by_key.pop(k))
        else:
            added.append({"name": name, "nat": None, "qualifiedBy": None,
                          "position": None, "rankingScore": None, "waId": None})
    return {**event,
            "athletes": kept + added,
            "notEntered": [a["name"] for a in by_key.values()],
            "fieldSource": "entries"}


def fetch_entry_lists():
    """{event label: [names]} from WA's published entry list, or {}."""
    url = find_entry_list_pdf()
    if not url:
        return {}
    try:
        r = requests.get(url, headers=MINISITE_HEADERS, timeout=60)
    except Exception as e:
        print(f"  entry list: download failed ({str(e)[:60]})")
        return {}
    if r.status_code != 200:
        print(f"  entry list: document HTTP {r.status_code}")
        return {}
    entries = parse_entry_pdf(r.content)
    if entries:
        print(f"  entry list: {len(entries)} events, "
              f"{sum(len(v) for v in entries.values())} athletes ({url.rsplit('/', 1)[-1]})")
    return entries


# Each carousel on the minisite is a content module titled by qualification
# route. Matched on the title rather than the array index, so WA reordering
# the page does not silently swap Olympic champions for World champions.
ROUTE_TITLES = {
    "olympic champions": "olympic",
    "world champions": "world",
}


def _text(html):
    """Visible text of a small HTML fragment, entities resolved."""
    return " ".join(hunescape(re.sub(r"<[^>]+>", " ", html or "")).split())


def champion_qualifiers():
    """The direct qualifiers World Athletics has NAMED: the 2024 Olympic
    champions and the 2025 World champions, each of whom holds a slot in their
    event. Read from the minisite's own __NEXT_DATA__ payload -- the same
    structured JSON the page renders itself from -- so we parse data rather
    than scraped layout.

    Returns [] on any failure. The Ultimate page is built to show whatever
    routes it has, so a fetch that fails costs the named champions, not the
    page."""
    try:
        r = requests.get(MINISITE_URL, headers=MINISITE_HEADERS, timeout=40)
        r.raise_for_status()
        m = re.search(
            r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
            r.text, re.S,
        )
        if not m:
            print("  minisite: no __NEXT_DATA__ block -- page shape changed")
            return []
        modules = json.loads(m.group(1))["props"]["pageProps"]["page"]["contentModules"]
    except Exception as e:
        print(f"  champion qualifiers fetch failed ({str(e)[:70]}) -- skipping")
        return []

    out = []
    for mod in modules:
        route = ROUTE_TITLES.get(str(mod.get("title") or "").strip().lower())
        if not route:
            continue
        note = _text(mod.get("subtitle"))          # "Qualified by winning in Paris"
        for img in mod.get("images") or []:
            name = " ".join(str(img.get("title") or "").split())
            sub = img.get("subtitle") or ""
            href = re.search(r'href="([^"]+)"', sub)
            # "POLE VAULT * SWE" -- discipline and nation, bullet-separated.
            parts = [p.strip() for p in _text(sub).split("•")]
            url = href.group(1) if href else None
            wa_id = None
            if url:
                mid = re.search(r"-(\d+)/?$", url)
                wa_id = int(mid.group(1)) if mid else None
            if not name:
                continue
            out.append({
                "route": route,
                "routeNote": note,
                "name": name,
                "disciplineLabel": parts[0] if parts else None,
                "nationality": parts[1] if len(parts) > 1 else None,
                "profileUrl": url,
                "waId": wa_id,
            })
    return out

# WA writes event names in prose ("POLE VAULT"); our discipline keys use short
# codes ("men_PV"). Gender is deliberately NOT in this map -- the minisite card
# doesn't carry it, so it is recovered from whichever of our disciplines the
# athlete actually appears in.
EVENT_CODES = {
    "100M": "100m", "200M": "200m", "400M": "400m", "800M": "800m",
    "1500M": "1500m", "MILE": "1500m", "5000M": "5000m",
    "3000M STEEPLECHASE": "3000sc", "STEEPLECHASE": "3000sc",
    "110M HURDLES": "110h", "100M HURDLES": "100h", "400M HURDLES": "400h",
    "HIGH JUMP": "HJ", "POLE VAULT": "PV", "LONG JUMP": "LJ",
    "TRIPLE JUMP": "TJ", "SHOT PUT": "SP", "DISCUS THROW": "DT",
    "JAVELIN THROW": "JT", "HAMMER THROW": "HT",
}


def _athlete_discipline_index():
    """{World Athletics id: {our discipline key: name}} from the world rankings
    we already build. Used to turn a named champion into a link into our own
    pages. Returns {} if that file has not been built yet."""
    path = os.path.join(os.path.dirname(__file__), "..", "data", "world_rankings.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        wr = json.load(f)
    idx = {}
    for key, disc in wr.items():
        for lst in ("model", "points"):
            for row in disc.get(lst, []):
                m = re.search(r"athlete=(\d+)", str(row.get("profileUrl"))) or re.search(
                    r"-(\d+)/?$", str(row.get("profileUrl")))
                if m:
                    idx.setdefault(int(m.group(1)), {})[key] = row.get("name")
    return idx


def resolve_disc_keys(qualifiers):
    """Attach our own discipline key to each named champion where it can be
    established beyond doubt, so the page can link them.

    Deliberately conservative. An athlete who ranks in several of our events
    (Lyles in both sprints) is resolved by the event WA says they won; one
    whose card names two events (a sprint double) or whose event we do not
    cover at all (the hammer is not a Diamond League discipline) is left
    unlinked rather than guessed at."""
    idx = _athlete_discipline_index()
    for q in qualifiers:
        q["discKey"] = None
        q["linkName"] = None
        candidates = idx.get(q.get("waId")) or {}
        code = EVENT_CODES.get((q.get("disciplineLabel") or "").upper())
        if not candidates or not code:
            continue
        hits = [k for k in candidates if k.split("_", 1)[-1] == code]
        if len(hits) == 1:
            q["discKey"] = hits[0]
            # Our own spelling of the name, which is what the athlete route
            # is keyed on. WA's card shows the popular name ("MONDO
            # DUPLANTIS") where our data has the registered one ("Armand
            # DUPLANTIS"), so linking on their string would 404.
            q["linkName"] = candidates[hits[0]]
    return qualifiers

def dl_final_qualifiers():
    """The 2026 Diamond League Final winners, who each hold a direct slot at the
    Ultimate. This is real data we already have (data/dl_final_2026_results.csv,
    place 1), so the Ultimate page can show a genuine list of qualified stars
    before WA's full field is published."""
    if not os.path.exists(DL_FINAL_RESULTS):
        return []
    out = []
    for row in csv.DictReader(open(DL_FINAL_RESULTS, encoding="utf-8")):
        if str(row.get("place", "")).strip() == "1":
            out.append({
                "discipline": row["discipline"],
                "athlete_name": row["athlete_name"],
                "nationality": row["nationality"],
            })
    return out


# The Ultimate contests two relays, both MIXED, and they qualify by a route of
# their own: the 2026 World Athletics Relays (Gaborone, 2 May), where the top
# six teams per event took the places. One team per country. Hungary hold host
# places in both. Nothing about this runs through the model -- see relay_teams.
WORLD_RELAYS_COMPETITION_ID = 7216920
WORLD_RELAYS = {
    "name": "World Athletics Relays",
    "city": "Gaborone",
    "country": "BOT",
    "date": "2026-05-02",
}
RELAY_QUALIFY_PLACES = 6
RELAY_HOST_WILDCARDS = ["HUN"]


WORLD_RELAYS_EVENT_ID = 8656      # WAW event id -- NOT the competition id

# Meets we read relay squads from, most recent first. A nation fields a relay
# SQUAD rather than its four fastest individuals, so who it actually ran before
# is the best evidence of who it will run again.
#
# The 2024 Olympics are deliberately absent: WA carries the phases (event 8691)
# but zero unit results for them -- it is an IOC-run meet and the detail never
# reaches this API. Checked, not assumed.
RELAY_SOURCE_MEETS = [
    {"eventId": WORLD_RELAYS_EVENT_ID, "name": "World Athletics Relays", "year": 2026},
    {"eventId": 8137, "name": "World Athletics Championships", "year": 2025},
]

# WA's own discipline ids. Passing one keeps the response inside the API's 6MB
# limit -- a whole World Championships in one call exceeds it and errors.
RELAY_DISCIPLINE_CODES = ("4X1", "4X4")

TIMETABLE_CONTENT_QUERY = """query TimetableWithContent($e: Int, $d: String) {
  getEventTimetableWithContent(eventId: $e, disciplineCode: $d) {
    phaseName sexCode discipline { name isRelay }
    units { results { resultCountryCode
      teamMembers { competitorName competitorId_WA competitorOrder } } }
  }
}"""


def _relay_key(event_name):
    """"Mixed 4x100 Metres Relay" and "4x100 Metres Relay" are the same event
    under two names -- the results feed prefixes the sex, the timetable does
    not. Reduce both to "4x100" so squads can be matched to teams."""
    m = re.search(r"4\s*x\s*(\d+)", event_name or "", re.I)
    return f"4x{m.group(1)}" if m else (event_name or "").strip().lower()


def relay_squads():
    """{(country code, "4x100"): [athlete, ...]} -- who each nation actually
    ran, across every round of the qualifying meet.

    This answers "which athletes does a country use in the relay": not its four
    fastest individuals on paper, but the people it actually put on the track.
    Counting APPEARANCES across rounds is what separates a fixed squad member
    from a one-round substitution -- Jamaica ran the same four in both of their
    4x100 rounds, while the USA rotated their third leg.

    Reached through getEventTimetableWithContent, which takes only the event id.
    The per-discipline phase query needs a disciplineCode WA does not document
    and rejects in every form its own timetable reports."""
    squads = {}
    for meet in RELAY_SOURCE_MEETS:
        for code in RELAY_DISCIPLINE_CODES:
            try:
                phases = dlr.graphql(
                    "TimetableWithContent",
                    {"e": meet["eventId"], "d": code},
                    TIMETABLE_CONTENT_QUERY,
                )["getEventTimetableWithContent"] or []
            except Exception as e:
                print(f"  squads {meet['year']} {code} failed ({str(e)[:50]}) -- skipping")
                continue
            for phase in phases:
                disc = phase.get("discipline") or {}
                if phase.get("sexCode") != "X" or not disc.get("isRelay"):
                    continue
                key_event = _relay_key(disc.get("name"))
                for unit in phase.get("units") or []:
                    for res in unit.get("results") or []:
                        nat = res.get("resultCountryCode")
                        if not nat:
                            continue
                        bucket = squads.setdefault((nat, key_event), {})
                        for m in res.get("teamMembers") or []:
                            name = m.get("competitorName")
                            if not name:
                                continue
                            entry = bucket.setdefault(name, {
                                "name": name,
                                "waId": m.get("competitorId_WA"),
                                "rounds": 0,
                                "legs": [],
                                "meets": [],
                            })
                            entry["rounds"] += 1
                            leg = m.get("competitorOrder")
                            if leg and leg not in entry["legs"]:
                                entry["legs"].append(leg)
                            if meet["year"] not in entry["meets"]:
                                entry["meets"].append(meet["year"])
    return {
        k: sorted(
            v.values(),
            key=lambda a: (-len(a["meets"]), -a["rounds"], a["legs"][0] if a["legs"] else 99),
        )
        for k, v in squads.items()
    }


def relay_teams():
    """The two mixed relays and who qualified, read from the World Athletics
    Relays finals.

    Relays are the one part of the championship the model says nothing about,
    and that is a data fact rather than a choice: the Diamond League has no
    relays, so there is no training history, no toplist and no head-to-head for
    a national team. What we can report is what actually happened in the
    qualifying meet, which is real and checkable.

    The result rows name the TEAM ("Jamaica"), not the four runners -- WA does
    expose leg runners, but through a different query than this one."""
    out = {}
    squads = relay_squads()
    try:
        day_data = dlr.graphql(
            "getCalendarCompetitionResults",
            {"competitionId": WORLD_RELAYS_COMPETITION_ID, "day": None, "eventId": None},
            dlr.RESULTS_QUERY,
        )["getCalendarCompetitionResults"]
        days = [d["day"] for d in day_data["options"]["days"]] or [None]
        for day in days:
            data = dlr.graphql(
                "getCalendarCompetitionResults",
                {"competitionId": WORLD_RELAYS_COMPETITION_ID, "day": day, "eventId": None},
                dlr.RESULTS_QUERY,
            )["getCalendarCompetitionResults"]
            for group in data["eventTitles"]:
                for event in group["events"]:
                    if event.get("gender") != "X":      # X = mixed
                        continue
                    finals = [r for r in event["races"] if r["race"] == "Final"]
                    if not finals:
                        continue
                    teams = []
                    for res in finals[0]["results"]:
                        place_raw = (res.get("place") or "").rstrip(".").strip()
                        place = int(place_raw) if place_raw.isdigit() else None
                        nat = res.get("nationality")
                        teams.append({
                            "nationality": nat,
                            "country": (res.get("competitor") or {}).get("name"),
                            "place": place,
                            # DNF teams keep their status word rather than a time.
                            "mark": res.get("mark"),
                            "qualified": place is not None and place <= RELAY_QUALIFY_PLACES,
                            # The athletes this nation actually ran, most-used first.
                            "squad": squads.get((nat, _relay_key(event["event"]))) or [],
                        })
                    out[event["event"]] = {
                        "event": event["event"],
                        "teams": teams,
                        "wildcards": RELAY_HOST_WILDCARDS,
                    }
    except Exception as e:
        print(f"  relay fetch failed ({str(e)[:60]}) -- skipping relays")
        return []
    return [
        {**v, "qualifier": WORLD_RELAYS, "qualifyPlaces": RELAY_QUALIFY_PLACES}
        for _, v in sorted(out.items())
    ]


def fetch_startlists():
    """Who is entered, per event, or [] until WA populates it.

    STATUS 2026-09-07: World Athletics' own front page says "FINAL ENTRY LISTS
    PUBLISHED -- 381 athletes from 65 federations", and their competition API
    does not have them. Every one of the 40 phases reports
    isStartlistPublished false, and asking for a start list directly returns a
    row of nulls. The entry list exists as a press release, not as competition
    data.

    So this returns nothing today, on purpose, rather than reaching for the
    press release and parsing prose into a start list. It is wired up and
    verified against the live schema so that the day WA populates the API the
    field simply appears -- no rediscovery of which argument spelling works.
    """
    phases = []
    try:
        phases = dlr.graphql("UcPhases", {"e": WAW_EVENT_ID}, PHASES_QUERY)["getEventPhases"] or []
    except Exception as e:
        print(f"  phase list fetch failed ({str(e)[:60]}) -- no start lists")
        return []

    out = []
    for phase in phases:
        disc = (phase.get("discipline") or {})
        slug = disc.get("nameUrlSlug")
        if not slug or not phase.get("phaseCode"):
            continue
        try:
            data = dlr.graphql("UcStartlist", {
                "e": WAW_EVENT_ID, "d": slug,
                "s": phase.get("sexCode"), "p": phase.get("phaseCode"),
            }, STARTLIST_QUERY)["getEventPhaseByDiscipline"]
        except Exception:
            continue
        if not data:
            continue
        entries = [
            athlete
            for unit in (data.get("units") or [])
            for athlete in (unit.get("startlist") or [])
        ]
        if entries:
            out.append({
                "discipline": disc.get("name"),
                "sex": phase.get("sexCode"),
                "phase": phase.get("phaseName"),
                "entries": entries,
            })
    return out


QUALIFICATION_QUERY = """query UcQualification($c: Int!, $e: Int) {
  getChampionshipQualifications(competitionId: $c, eventId: $e) {
    disciplineName entryNumber
    events { genderCode eventId disciplineName }
    qualifications {
      name countryCode qualified qualifiedBy qualificationPosition
      score place competitorIaafId
    }
  }
}"""

# WA writes the discipline as "Men's 400 Metres Hurdles"; our keys are
# "men_400h". The gender prefix and the possessive are stripped, then the rest
# runs through EVENT_CODES above -- the same map the named-champion resolver
# uses, so the two cannot drift.
_DISC_PREFIX = {"Men's ": "men", "Women's ": "women", "Mixed ": None}


def disc_key_from_wa(discipline_name):
    """'Men's 400 Metres Hurdles' -> 'men_400h', or None when we do not cover
    the event. The Ultimate contests a men's hammer throw and we hold no
    hammer data at all, so None is a real answer here, not a parsing failure."""
    label = str(discipline_name or "").strip()
    for prefix, sex in _DISC_PREFIX.items():
        if label.startswith(prefix):
            if sex is None:
                return None
            rest = label[len(prefix):]
            # WA spells it out ("400 Metres Hurdles"); EVENT_CODES is keyed on
            # the short form ("400M HURDLES").
            short = rest.upper().replace(" METRES", "M").replace("M HURDLES", "M HURDLES")
            code = EVENT_CODES.get(short)
            if code is None:
                code = EVENT_CODES.get(rest.upper())
            if not code:
                return None
            key = f"{sex}_{code}"
            # A name we can parse is not the same as an event we hold data for.
            # The Ultimate contests a men's hammer throw, which maps cleanly to
            # "men_HT" and for which we have no toplist, no history and no
            # model. Returning the key anyway would hand the projector a
            # discipline it cannot score and make it look like a bug.
            return key if _have_season_data(key) else None
    return None


def _have_season_data(disc_key):
    return os.path.exists(os.path.join(
        os.path.dirname(__file__), "..", "data", "raw", f"{disc_key}_2026.csv"))


def fetch_qualified_field():
    """Who has actually qualified for the Ultimate, per event.

    THIS IS THE REAL FIELD, and it is a better source than the entry list WA
    announced in a press release: it is machine-readable, it says HOW each
    athlete got in (a wild card for the champions, world rankings for the
    rest), and it carries the ranking score they got in on.

    Only rows WA marks qualified=True are kept. The endpoint also returns the
    athletes who did not make the cut -- 57 rows for a 17-place event -- and
    treating those as entrants would invent a field twice the real size."""
    try:
        head = dlr.graphql("UcQualification", {"c": COMPETITION_ID, "e": None},
                           QUALIFICATION_QUERY)["getChampionshipQualifications"]
    except Exception as e:
        print(f"  qualification fetch failed ({str(e)[:60]}) -- no field")
        return []
    events = (head or {}).get("events") or []

    out = []
    for ev in events:
        try:
            data = dlr.graphql("UcQualification",
                               {"c": COMPETITION_ID, "e": ev.get("eventId")},
                               QUALIFICATION_QUERY)["getChampionshipQualifications"]
        except Exception:
            continue
        if not data:
            continue
        qualified = [q for q in (data.get("qualifications") or []) if q.get("qualified")]
        if not qualified:
            continue
        out.append({
            "disciplineLabel": data.get("disciplineName") or ev.get("disciplineName"),
            "discKey": disc_key_from_wa(ev.get("disciplineName")),
            "sex": ev.get("genderCode"),
            "eventId": ev.get("eventId"),
            "places": data.get("entryNumber"),
            "athletes": [
                {
                    "name": q.get("name"),
                    "nat": q.get("countryCode"),
                    "qualifiedBy": q.get("qualifiedBy"),
                    "position": q.get("qualificationPosition"),
                    "rankingScore": q.get("score"),
                    "waId": q.get("competitorIaafId"),
                }
                for q in qualified
            ],
        })
    return out


def build_event():
    timetable = fetch_timetable()
    startlists = fetch_startlists()
    # The field is fetched BEFORE the results because the results need it: it
    # is what tells the championship's own final apart from any other race in
    # the same feed calling itself a Final. See pick_final.
    field = fetch_qualified_field()
    results = fetch_results(field)
    # WA's own flag, from either feed. It is the thing that flips when they
    # populate the API, and it is currently false everywhere even though the
    # entry lists are announced -- see fetch_startlists.
    field_published = (any(p.get("isStartlistPublished") for p in timetable)
                       or bool(startlists) or bool(field))
    return {
        **EVENT_META,
        "fieldPublished": field_published,
        "resultsAvailable": bool(results),
        "dlFinalQualifiers": dl_final_qualifiers(),
        "namedQualifiers": resolve_disc_keys(champion_qualifiers()),
        "relays": relay_teams(),
        "timetable": timetable,
        "startlists": startlists,
        "qualifiedField": field,
        "results": results,
    }


def lost_data(new, old):
    """What a fresh build dropped that the saved file still has, or None.

    Every fetch above swallows its own failure and returns empty, which is right
    for a meeting that has not published yet and wrong for a network that is
    down. When WA retired its GraphQL host in September 2026 every call failed
    at once, and the build saved a file with no timetable, no field and no
    relays over the real one -- the next static build would have emptied the
    championship page. A published timetable or field does not un-publish, so
    losing one means a fetch broke, not that the meeting changed."""
    lost = [key for key in ("timetable", "qualifiedField", "relays")
            if (old or {}).get(key) and not new.get(key)]
    return ", ".join(lost) or None


if __name__ == "__main__":
    print("=== Building World Athletics Ultimate Championship data ===")
    event = build_event()
    try:
        with open(OUT_PATH, encoding="utf-8") as f:
            previous = json.load(f)
    except (OSError, json.JSONDecodeError):
        previous = None
    lost = lost_data(event, previous)
    if lost:
        print(f"  NOT SAVED: this build has no {lost}, which {os.path.abspath(OUT_PATH)} "
              "still has. A fetch failed; the saved file is kept.")
        raise SystemExit(1)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(event, f, indent=2, ensure_ascii=False)
    named = event["namedQualifiers"]
    linked = sum(1 for q in named if q.get("discKey"))
    print(f"  fieldPublished={event['fieldPublished']} resultsAvailable={event['resultsAvailable']} "
          f"dlFinalQualifiers={len(event['dlFinalQualifiers'])} timetablePhases={len(event['timetable'])}")
    print(f"  namedQualifiers={len(named)} "
          f"(olympic={sum(1 for q in named if q['route'] == 'olympic')}, "
          f"world={sum(1 for q in named if q['route'] == 'world')}); "
          f"{linked} linked to one of our disciplines")
    print(f"  relays={len(event['relays'])} "
          f"({sum(sum(1 for t in r['teams'] if t['qualified']) for r in event['relays'])} qualified teams)")
    print(f"  Saved -> {os.path.abspath(OUT_PATH)}")
