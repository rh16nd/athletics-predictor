"""
injury_checker.py — scrapes athletics news sources for injury/withdrawal
mentions of the athletes the site currently projects, and writes
data/injury_flags.json.

The field is BOTH competitions: World Athletics' Diamond League standings
(data/standings.json) and the Ultimate Championship's qualified field
(data/ultimate/predictions.json). It was the standings alone until 2026-09-08,
which was correct only while the Diamond League Final was the next event —
after the pivot to Budapest it left 143 of the Ultimate's 306 athletes
unwatched, so an injured athlete kept their projection with nothing to show
for it. See load_qualified_athletes().

Sources: LetsRun.com, Athletics Weekly, World Athletics news. Two of the three
need a HEADFUL browser, so this cannot run unattended.

Run standalone, after live_fetcher.py and/or ultimate_predictions.py have
produced the field:
    python src/injury_checker.py
"""
import io
import json
import os
import re
import sys
import time
import unicodedata
from functools import lru_cache
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

def _force_utf8_stdout():
    """Windows consoles default to cp1252 and choke on athlete names.

    Deliberately NOT done at import time. Several modules in src/ do this,
    and each one wraps `sys.stdout.buffer` -- so when two of them are
    imported into the same process (which happens as soon as two test
    modules import two of them) both wrappers own the same underlying
    buffer, the first to be garbage-collected closes it, and every
    subsequent write dies with "I/O operation on closed file". pytest.ini
    keeps `-s` for the same family of reasons; this stays a __main__-only
    concern so the module is safely importable."""
    if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("utf"):
        return
    # Guarded: several modules in src/ do this, and each wraps the SAME
# sys.stdout.buffer. With two of them imported into one process the first
# wrapper to be garbage-collected closes the buffer under the second, and
# every later write dies with "I/O operation on closed file" -- which took
# down the whole pytest run on 2026-08-25. After the first wrap the
# encoding is already utf-8, so this becomes a no-op.
if not (sys.stdout.encoding or "").lower().startswith("utf"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
STANDINGS_PATH = os.path.join(DATA_DIR, "standings.json")
ULTIMATE_PREDICTIONS_PATH = os.path.join(DATA_DIR, "ultimate", "predictions.json")
ULTIMATE_EVENT_PATH = os.path.join(DATA_DIR, "ultimate", "event.json")
OUTPUT_PATH = os.path.join(DATA_DIR, "injury_flags.json")

# LetsRun and Athletics Weekly both sit behind bot-detection that blocks
# headless Chrome — they need a headful browser. World Athletics is fine
# headless (same as live_fetcher.py).
SOURCES = [
    {"name": "letsrun",         "url": "https://www.letsrun.com/news/",      "headless": False, "wait_seconds": 6},
    {"name": "athleticsweekly", "url": "https://athleticsweekly.com/news/",  "headless": False, "wait_seconds": 6},
    {"name": "worldathletics",  "url": "https://worldathletics.org/news",    "headless": True,  "wait_seconds": 5},
]

# Google News, as a fourth source and by far the widest one.
#
# The three above are FRONT PAGES. Between them they yielded 179 headlines on
# 2026-09-08, exactly ONE of which mentioned an injury -- and Athletics Weekly
# has started refusing us outright (ConnectionResetError, headful included). So
# the check was reading a keyhole: Noah Lyles ending his season and missing the
# Ultimate, and Duplantis withdrawing, were both simply never in front of it.
#
# This is a search rather than a front page, it is plain XML over HTTPS with no
# browser and nothing to block, and it reaches Athletics Weekly's own reporting
# through Google's index even while their site turns us away. It brings in
# other sports too -- baseball and football use the same words -- which costs
# nothing, because a headline only matters here if it also names an athlete in
# the field.
GOOGLE_NEWS_URL = "https://news.google.com/rss/search"
GOOGLE_NEWS_QUERIES = [
    'athletics injury OR injured OR withdraws OR "out for the season"',
    '"track and field" injury OR withdrawal OR "ruled out"',
    '"World Athletics" injury OR withdraws OR "will not compete"',
    '"Ultimate Championship" Budapest injury OR withdraw OR "pulls out"',
    '"out for the season" OR "season is over" OR "ends season" athlete',
    # Returns, deliberately. Knowing an athlete is BACK needs its own search:
    # the queries above look for people going out, so a comeback never turned
    # up and a withdrawal stood even after later coverage overtook it. Keely
    # Hodgkinson was still listed as out on 2026-09-08 from a 2 Sep withdrawal,
    # five days after "Hodgkinson returns from injury scare" ran.
    'athletics "returns from injury" OR "back from injury" OR "return to racing"',
    '"track and field" comeback OR "recovered from" OR "back in action"',
]
GOOGLE_NEWS_UA = "Mozilla/5.0 (compatible; PodiumCall injury check)"

# How far back a search result may be. A news search has no sense of season:
# without this the first widened run flagged Josh Kerr off a 2025 World
# Championships report and Lachlan Kennedy off a 2025 withdrawal, both long
# since irrelevant. The front-page sources need no equivalent because a front
# page is recent by construction.
GOOGLE_NEWS_MAX_AGE_DAYS = 30

# Other sports use these words about their own athletes, and enough of our
# surnames are shared with footballers and basketball players that a bare
# surname plus "injury" matches constantly. One widened run produced Abbey
# Caldwell from a Wigan Athletic report, Devynne Charlton from Charlton
# Athletic FC, and Gabrielle Jennings from Oklahoma State basketball. A
# headline carrying any of these is not about track and field.
OTHER_SPORT_WORDS = [
    "football", "soccer", "basketball", "baseball", "hockey", "cricket",
    "rugby", "netball", "golf", "tennis", "nfl", "nba", "mlb", "nhl", "wnba",
    "premier league", "champions league", "quarterback", "touchdown",
    "bullpen", "innings", "midfielder", "defender", "goalkeeper", "striker",
    "football club", "wicket", "scrum", "linebacker", "pitcher", "shortstop",
    "bullpen", "dugout", "playoffs", "draft pick",
]
# Club names are deliberately NOT in that list. There is always another club,
# and as substrings they are actively unsafe: "united" is in "United States",
# "villa" is in "Villanueva". The general guard is the one below instead.

# A BARE SURNAME is weak evidence, so it only counts when the headline also
# reads like athletics. A full name needs no corroboration. This is what
# separates "Caudery out for the season" (European Athletics, real) from
# "Latics boss Caldwell provides positive Chapman injury update" (Wigan
# Today, a footballer) without having to know what Wigan Athletic is.
# An article about an athlete COMING BACK is not a report that they are out.
# The word "injury" is genuinely in it, which is why keyword matching alone
# cannot tell the difference: "'I'm not just an athlete' - Wightman on injury
# bounce back" and "analysis of Jazmin Sawyers and her success since her
# achilles injury" both flagged, and the second was escalated to a REMOVE
# because "achilles" carries a 6-20 week estimate whether the athlete is
# entering that window or coming out of it.
#
# These suppress a flag ONLY when no explicit withdrawal word is present, so
# "Injured Omanyala pulls out of Diamond League final" still flags: it says
# both that he was injured and that he is out.
RETURNING_PHRASES = [
    "bounce back", "bounced back", "bounces back", "bouncing back",
    "back from injury", "return from injury", "returns from injury",
    "returning from injury", "recovered from", "recovers from", "comeback",
    "makes his return", "makes her return", "back in action", "fit again",
    "return to action", "on the mend", "back to winning", "cleared to return",
    "since her injury", "since his injury", "since the injury", "success since",
    "injury bounce", "back to full fitness", "return to racing",
]

ATHLETICS_CONTEXT = [
    "athletics", "athlete", "track and field", "sprint", "sprinter", "runner",
    "metres", "meters", "hurdles", "steeplechase", "high jump", "long jump",
    "triple jump", "pole vault", "shot put", "discus", "javelin", "hammer",
    "heptathlon", "decathlon", "marathon", "diamond league", "world athletics",
    "olympic", "olympics", "world championships", "european championships",
    "ultimate championship", "relay", "personal best", "world record",
    "100m", "200m", "400m", "800m", "1500m", "5000m", "10000m", "3000m",
]

# Keyword sets are matched as whole words against lowercased headlines.
# REMOVE = athlete is confirmed out of competition -> drop from predictions.
# WATCH  = injury-adjacent mention that isn't a confirmed withdrawal -> keep
#          in predictions but surface a warning (may still get escalated to
#          REMOVE below if the estimated recovery time won't clear in time
#          for the final).
REMOVE_KEYWORDS = [
    "withdraw", "withdrawn", "withdraws", "withdrew", "pulls out", "pulled out",
    "ruled out", "out for the season", "out for the year", "will not compete",
    "will miss", "scratched", "did not start", "forced to withdraw",
    "ends season", "season is over", "retires", "retirement",
    # Contractions and the phrasings a withdrawal actually gets written in.
    # Headlines say "won't compete", not "will not compete", and an athlete
    # whose year is finished is usually described as ending or shutting down a
    # season rather than withdrawing from a meeting. Hyphens are normalised to
    # spaces before matching, so "season-ending" is covered by "season ending".
    "won't compete", "wont compete", "won't race", "will not race",
    "ends his season", "ends her season", "ends the season",
    "shuts down season", "shuts down his season", "shuts down her season",
    "season ending", "out of the championships", "out of the world championships",
    "out of budapest", "misses the championships", "no longer competing",
]
GENERIC_WATCH_KEYWORDS = [
    "injury", "injured", "surgery", "setback", "sidelined", "recovering",
    "fitness concern", "doubtful", "injury scare",
]

# Body-part keywords double as (a) a watch trigger on their own -- "Jackson's
# calf tightens up" is injury-adjacent even without the word "injury" -- and
# (b) the input to the recovery-time estimate below. (min_weeks, max_weeks)
# are rough, literature-typical recovery windows for a moderate strain/sprain
# of that body part -- NOT a medical diagnosis. Real recovery time depends
# heavily on grade/severity that vague news text usually doesn't specify,
# which is why these ranges are wide and skew conservative (short end) so we
# don't over-remove on a guess.
INJURY_RECOVERY_WEEKS = {
    "hamstring":       (2, 10),
    "achilles":        (6, 20),
    "calf":            (1, 5),
    "groin":           (2, 6),
    "quad":            (2, 6),
    "quadriceps":      (2, 6),
    "knee":            (3, 12),
    "ankle":           (1, 6),
    "foot":            (4, 10),
    "stress fracture": (6, 12),
    "hip":             (2, 8),
    # "back" is NOT here as a bare word, deliberately. In athletics writing it
    # almost always means returning -- "is back", "back-to-back", "bounce
    # back", "back in the field" -- and matching it as a body part caused a
    # real, live false positive on 2026-08-25: the headline "Jakob
    # Ingebrigtsen Is Back. His First Big 1500m Test: Cole Hocker" flagged a
    # 2-8 week recovery and REMOVED Cole Hocker from the men's 1500m
    # predictions entirely. Only unambiguous body-part phrasings count.
    "back injury":     (2, 8),
    "back problem":    (2, 8),
    "back issue":      (2, 8),
    "back strain":     (2, 8),
    "back spasm":      (2, 8),
    "back pain":       (2, 8),
    "lower back":      (2, 8),
}
WATCH_KEYWORDS = GENERIC_WATCH_KEYWORDS + list(INJURY_RECOVERY_WEEKS.keys())

# Words that shift the estimate up or down regardless of body part.
SEVERITY_UP_WORDS = ["surgery", "torn", "tear", "rupture", "ruptured", "operation"]
SEVERITY_DOWN_WORDS = ["minor", "slight", "small", "tweak", "niggle", "precaution", "tightness"]

DL_FINAL_DATE = date(2026, 9, 4)  # Brussels DL Final — run, and now history


# Ways of saying an athlete will not be at a named competition. Only ever
# consulted alongside the competition's own name (see names_target_event), so
# these can be looser than REMOVE_KEYWORDS -- "misses" on its own would match
# "misses the podium", but "misses ... World Athletics Ultimate Championships"
# is exactly what it looks like. That headline is why the list exists: Sachin
# Yadav's withdrawal matched only "surgery", a watch word, and sat in the
# projection as a doubt rather than an absence.
OUT_OF_EVENT_WORDS = [
    "misses", "miss", "will not compete", "wont compete", "won't compete",
    "withdraws", "withdrawn", "withdrew", "pulls out", "pulled out",
    "ruled out", "out of", "absent from", "not travel", "skip", "skips",
]


def target_event_terms():
    """Normalised names for the competition being projected.

    Read from the event file rather than hard-coded, so this follows the site
    when the next championship takes the tab."""
    terms = set()
    try:
        with open(ULTIMATE_EVENT_PATH, encoding="utf-8") as f:
            event = json.load(f)
    except (OSError, json.JSONDecodeError):
        return terms
    for key in ("name", "shortName", "city"):
        value = normalize_for_match(event.get(key) or "")
        if value:
            terms.add(value)
    # "World Athletics Ultimate Championship" is written a dozen ways; the two
    # words that survive all of them are the ones worth matching on.
    full = normalize_for_match(event.get("name") or "")
    if "ultimate" in full:
        terms.add("ultimate championship")
        terms.add("ultimate championships")
    return terms


def names_target_event(headline_norm, terms):
    return any(term in headline_norm for term in terms)


def target_date():
    """The competition an injury is measured against.

    This used to be the Brussels Final, hard-coded. Once that was run the date
    went NEGATIVE, and `days_to_final` drives the one decision that matters
    here: a recovery estimate longer than the wait upgrades a "watch" to a
    "remove". Against a date in the past every estimate is longer than the
    wait, so a fresh check would have quietly removed every athlete it found
    any injury mention for. Reads the real next competition instead, and only
    falls back to the old constant if the file is missing."""
    try:
        with open(ULTIMATE_EVENT_PATH, encoding="utf-8") as f:
            start = json.load(f).get("startDate")
        if start:
            return date.fromisoformat(start)
    except (OSError, json.JSONDecodeError, ValueError):
        pass
    return DL_FINAL_DATE

_KEYWORD_RE = {
    kw: re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE)
    for kw in REMOVE_KEYWORDS + WATCH_KEYWORDS
}

_RETURN_RE = {p: re.compile(r"\b" + re.escape(p) + r"\b") for p in RETURNING_PHRASES}

_SOURCE_HEADLESS = {s["name"]: s["headless"] for s in SOURCES}
_SOURCE_BASE_URL = {
    "letsrun": "https://www.letsrun.com",
    "athleticsweekly": "https://athleticsweekly.com",
    "worldathletics": "https://worldathletics.org",
}

# Full meet-results recaps (as opposed to narrative news coverage) list every
# athlete's result, including DNF/DNS/DQ -- a much more reliable signal than
# hoping a headline happens to name a specific athlete. Identified by a
# headline containing "results" plus a known DL city.
DL_CITIES = [
    "doha", "shanghai", "suzhou", "shaoxing", "rabat", "florence", "paris",
    "oslo", "lausanne", "stockholm", "silesia", "monaco", "london",
    "zurich", "zürich", "brussels", "eugene", "birmingham", "rome", "xiamen",
]


def create_driver(headless=True):
    options = Options()
    if headless:
        options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--log-level=3")
    options.add_argument("--window-size=1920,1080")
    return webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options,
    )


def _add(athletes, name, discipline_key):
    if not name:
        return
    normalized = " ".join(str(name).split()).title()
    athletes.setdefault(normalized, set()).add(discipline_key)


def load_qualified_athletes():
    """Returns {normalized athlete name -> set of discipline keys}: everyone the
    site currently shows a projection for.

    BOTH competitions, deliberately. This read only the Diamond League
    standings, which was right while the Final was the site's next event and
    silently wrong the moment it wasn't: the Ultimate's field is World
    Athletics' own qualification list, and 143 of its 306 athletes qualified on
    world rankings without racing the circuit. None of them were ever looked
    at, so an injured athlete simply kept their projection. Measured on
    2026-09-08, which is what sent this back for repair.

    The Ultimate names come from data/ultimate/predictions.json rather than a
    live call, because this already needs a headful browser for two of its
    three sources and should not also depend on WA's API being up. Both files
    are optional: with neither, the caller skips rather than pretending."""
    athletes = {}

    if os.path.exists(STANDINGS_PATH):
        try:
            with open(STANDINGS_PATH, encoding="utf-8") as f:
                standings = json.load(f)
            for discipline_key, names in standings.items():
                for name in names:
                    _add(athletes, name, discipline_key)
        except (OSError, json.JSONDecodeError):
            pass

    if os.path.exists(ULTIMATE_PREDICTIONS_PATH):
        try:
            with open(ULTIMATE_PREDICTIONS_PATH, encoding="utf-8") as f:
                projections = json.load(f).get("projections") or []
            for event in projections:
                key = event.get("discKey")
                for athlete in event.get("athletes") or []:
                    _add(athletes, athlete.get("name"), key)
                # The unscored are qualified too — they just have no 2026 mark
                # to project. An injury still matters to a reader looking for
                # them on the start list.
                for name in event.get("unscored") or []:
                    _add(athletes, name, key)
        except (OSError, json.JSONDecodeError):
            pass

    return athletes


def _published_date(raw):
    """RFC-822 pubDate -> "YYYY-MM-DD", or None if it will not parse."""
    if not raw:
        return None
    try:
        return parsedate_to_datetime(raw).date().isoformat()
    except (TypeError, ValueError):
        return None


def fetch_google_news(query, limit=100):
    """Headlines from one Google News search, as {headline, url, source}.

    Plain XML over HTTPS -- no browser, nothing to be blocked by, and cheap
    enough to run several queries. `<source>` carries the real publisher, so a
    reader sees "European Athletics" rather than "googlenews" even though the
    link is Google's redirect (which resolves to the article)."""
    url = f"{GOOGLE_NEWS_URL}?q={urllib.parse.quote(query)}&hl=en-GB&gl=GB&ceid=GB:en"
    request = urllib.request.Request(url, headers={"User-Agent": GOOGLE_NEWS_UA})
    with urllib.request.urlopen(request, timeout=25) as response:
        root = ET.fromstring(response.read())

    cutoff = datetime.now(timezone.utc) - timedelta(days=GOOGLE_NEWS_MAX_AGE_DAYS)
    out = []
    for item in root.findall("./channel/item")[:limit]:
        headline = (item.findtext("title") or "").strip()
        if not headline:
            continue

        published = item.findtext("pubDate")
        if published:
            try:
                if parsedate_to_datetime(published) < cutoff:
                    continue
            except (TypeError, ValueError):
                pass  # unparseable date: keep it rather than lose a real story

        # Google appends " - Publisher" to every title. Left on, the publisher
        # becomes matchable text: "Charlton Athletic Football Club" flagged
        # Devynne Charlton. Kept for display, dropped before name matching.
        publisher = (item.findtext("source") or "google news").strip()
        if publisher and headline.endswith(f" - {publisher}"):
            headline = headline[: -len(f" - {publisher}")].strip()

        out.append({
            "headline": headline,
            "url": (item.findtext("link") or "").strip(),
            "source": publisher,
            # Carried through to the flag so a reader can see how old the news
            # is. "Is this current?" is the first question a withdrawal raises.
            "published": _published_date(published),
        })
    return out


def fetch_headlines(source):
    driver = create_driver(headless=source["headless"])
    headlines = []
    try:
        driver.get(source["url"])
        try:
            WebDriverWait(driver, source["wait_seconds"]).until(
                EC.presence_of_element_located((By.TAG_NAME, "a"))
            )
        except Exception:
            pass
        time.sleep(source["wait_seconds"])

        soup = BeautifulSoup(driver.page_source, "html.parser")
        for a in soup.find_all("a", href=True):
            text = a.get_text(strip=True)
            if len(text) < 12:
                continue
            headlines.append({"headline": text, "url": a["href"], "source": source["name"]})
    finally:
        driver.quit()
    return headlines


def match_keywords(headline_lower):
    matched = {"remove": [], "watch": []}
    for kw in REMOVE_KEYWORDS:
        if _KEYWORD_RE[kw].search(headline_lower):
            matched["remove"].append(kw)
    for kw in WATCH_KEYWORDS:
        if _KEYWORD_RE[kw].search(headline_lower):
            matched["watch"].append(kw)
    return matched


# Surnames that are also ordinary English words, or too short to be safe on
# their own. Eight of the 345 unique surnames in the watched field land here.
# Without this, "long jump star ruled out" flags an athlete surnamed Long.
AMBIGUOUS_SURNAMES = {
    # Ordinary English words. Without this, "long jump star ruled out" flags an
    # athlete surnamed Long, and "back on form" flags one surnamed Back.
    "abe", "bell", "brown", "walker", "white", "young", "hall", "long",
    "king", "cook", "green", "black", "rose", "price", "bird", "sharp", "best",
    "field", "track", "west", "may", "will", "march", "back", "park", "wood",
    "stone", "hill", "gold", "ann", "ian",
    # Name particles. They belong to a surname, they are not one, and they are
    # shared by everyone who has one.
    "van", "von", "der", "den", "del", "dos", "das", "de", "da", "la", "le",
    "el", "al", "bin", "ben", "abu", "mac", "mc", "st",
}


def normalize_for_match(text):
    """Lowercase, unaccented, and with hyphens and apostrophes read as spaces.

    Headlines do not spell names the way a database does. World Athletics has
    "Tara Davis-Woodhall"; LetsRun wrote "Tara Davis Woodhall Injured In Car
    Wreck" and the exact-substring test missed the Olympic long jump champion
    entirely. Same for accents -- the field holds "Hernández", a headline
    writes "Hernandez"."""
    folded = unicodedata.normalize("NFKD", str(text))
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    folded = re.sub(r"[‐-―\-'’.]", " ", folded.lower())
    return re.sub(r"\s+", " ", folded).strip()


def build_aliases(athletes):
    """{alias -> (athlete name, surname_only)} for every way a headline might
    name someone in the field.

    Two kinds of alias: the full name, and a bare surname where that surname
    belongs to exactly one watched athlete and is not an ordinary word or a
    name particle. The surname is what headlines actually print -- "Duplantis
    withdraws", not "Armand Duplantis withdraws", and the field does not even
    hold the name he is usually called by ("Mondo").

    Ownership is counted over EVERY token of a name except the given name, not
    just the last one. A first version keyed on the last token only, so
    "Tara Davis-Woodhall" registered as "woodhall" and left "davis" looking
    unique to Tamari Davis -- which would have attributed a headline about Tara
    to Tamari. Shared tokens are dropped rather than guessed at; that is the
    Cole Hocker mistake with extra steps."""
    aliases = {}
    token_owners = {}
    for name in athletes:
        # [1:] because a given name is not a surname: "Tara" must not become an
        # alias, or every headline about any Tara flags this one.
        for token in normalize_for_match(name).split()[1:]:
            token_owners.setdefault(token, set()).add(name)

    for name in athletes:
        full = normalize_for_match(name)
        if full:
            aliases[full] = (name, False)
    for token, owners in token_owners.items():
        if len(owners) != 1 or token in AMBIGUOUS_SURNAMES or len(token) < 3:
            continue
        aliases.setdefault(token, (next(iter(owners)), True))
    return aliases


@lru_cache(maxsize=4096)
def _word_re(alias):
    """Whole-word pattern for one alias, compiled once and reused."""
    return re.compile(r"\b" + re.escape(alias) + r"\b")


def is_other_sport(headline_norm):
    """True when the headline is plainly about a different sport."""
    return any(word in headline_norm for word in OTHER_SPORT_WORDS)


def reads_like_athletics(headline_norm):
    """Does this headline mention the sport at all?"""
    return any(word in headline_norm for word in ATHLETICS_CONTEXT)


def is_returning_story(headline_norm):
    """Is this about an athlete coming BACK from an injury rather than going
    into one? See RETURNING_PHRASES."""
    return any(phrase in headline_norm for phrase in RETURNING_PHRASES)


def returning_is_about(headline_norm, target, other_names):
    """Is the RETURN describing `target`, or another athlete in the headline?

    Same nearest-name rule as keyword_is_about, but anchored on the return
    phrase instead of the injury word -- which is the whole point. In "Keely
    Hodgkinson returns from injury scare for Werro and Broeders-Bol showdown"
    the word "injury" is nearest Werro, so the keyword rule hands it to her;
    "returns from injury" is nearest Hodgkinson, and she is the one who is back.

    `other_names` must exclude the target's own aliases, or her surname beats
    her full name and every return is rejected."""
    positions = [m.start() for m in
                 (rx.search(headline_norm) for rx in _RETURN_RE.values()) if m]
    if not positions:
        return False
    anchor = min(positions)
    target_pos = headline_norm.find(target)
    if target_pos < 0:
        return False
    target_dist = abs(anchor - target_pos)
    for other in other_names:
        pos = headline_norm.find(other)
        if pos >= 0 and abs(anchor - pos) < target_dist:
            return False
    return True


def names_in(headline_norm, aliases):
    """Which aliases this headline actually names, as whole words."""
    found = []
    for alias in aliases:
        if alias in headline_norm and _word_re(alias).search(headline_norm):
            found.append(alias)
    return found


def keyword_is_about(headline_lower, target_lower, other_names_lower, keyword):
    """Is `keyword` plausibly describing `target_lower`, or someone else?

    Matching used to require only that the athlete's name and an injury word
    both appeared somewhere in the same headline. That flagged everyone
    mentioned, including the person the article is contrasting the injured
    athlete AGAINST. The real case that exposed it: "Jakob Ingebrigtsen Is
    Back. His First Big 1500m Test: Cole Hocker" removed Hocker.

    Attribute the keyword to whichever known athlete is named closest to it.
    Ties and single-name headlines resolve to the target, so this only ever
    rejects when some OTHER athlete is genuinely nearer."""
    kw_match = _KEYWORD_RE[keyword].search(headline_lower) if keyword in _KEYWORD_RE else None
    if kw_match is None:
        return True
    kw_pos = kw_match.start()
    target_pos = headline_lower.find(target_lower)
    if target_pos < 0:
        return False
    target_dist = abs(kw_pos - target_pos)
    for other in other_names_lower:
        if other == target_lower:
            continue
        pos = headline_lower.find(other)
        if pos >= 0 and abs(kw_pos - pos) < target_dist:
            return False
    return True


def estimate_recovery_weeks(text_lower):
    """Rough recovery-time estimate from injury-adjacent text. This is a
    heuristic, not a medical diagnosis -- vague news text ("hamstring issue")
    rarely gives a grade/severity, so treat the returned range as a wide,
    conservative guess. Returns (min_weeks, max_weeks) for the most severe
    body part matched, or None if no recognized body part is named."""
    best = None
    for body_part, weeks in INJURY_RECOVERY_WEEKS.items():
        if re.search(r"\b" + re.escape(body_part) + r"\b", text_lower):
            if best is None or weeks[1] > best[1]:
                best = weeks
    if best is None:
        return None

    min_w, max_w = best
    if any(re.search(r"\b" + re.escape(w) + r"\b", text_lower) for w in SEVERITY_UP_WORDS):
        min_w, max_w = max(min_w, 12), max(max_w, 26)
    elif any(re.search(r"\b" + re.escape(w) + r"\b", text_lower) for w in SEVERITY_DOWN_WORDS):
        min_w, max_w = max(1, min_w // 2), max(1, max_w // 2)
    return (min_w, max_w)


def upgrade_status(entry, new_status):
    """Status only ever moves toward 'remove', never back down from it."""
    if new_status == "remove":
        entry["status"] = "remove"
    elif entry["status"] != "remove":
        entry["status"] = new_status


def find_results_articles(headlines, max_articles=3):
    """Picks out headlines that look like a full meet-results recap rather
    than an ordinary news story."""
    matches = []
    seen_urls = set()
    for item in headlines:
        # Only the sites we scrape directly. A Google News hit is a redirect to
        # someone else's page in an unknown layout, and this step opens the
        # article in a real browser to read a results table out of it.
        if item.get("source", "").replace("_results", "") not in _SOURCE_BASE_URL:
            continue
        headline_lower = item["headline"].lower()
        if "results" not in headline_lower:
            continue
        if not any(city in headline_lower for city in DL_CITIES):
            continue
        if item["url"] in seen_urls:
            continue
        seen_urls.add(item["url"])
        matches.append(item)
        if len(matches) >= max_articles:
            break
    return matches


def fetch_article_text(url, source_name):
    if not url.startswith("http"):
        url = _SOURCE_BASE_URL.get(source_name, "") + url
    driver = create_driver(headless=_SOURCE_HEADLESS.get(source_name, False))
    try:
        driver.get(url)
        time.sleep(6)
        soup = BeautifulSoup(driver.page_source, "html.parser")
        return soup.get_text(" ", strip=True)
    finally:
        driver.quit()


def find_dnf_athletes(article_text, athlete_names):
    """Scans results-article body text for an athlete's name immediately
    followed by DNF, returning {name: recovery_estimate_or_None}. DNS is
    deliberately NOT treated as a signal here -- an athlete can skip a race
    for rest, travel, or scheduling reasons that have nothing to do with
    injury. DQ is ignored too -- a disqualification is a rules call by
    officials and has no bearing on health.

    The recovery estimate looks at a wider window around the name (not just
    the DNF itself) in case the same recap article separately describes what
    happened, e.g. "... DNF ... appeared to pull up with a hamstring issue".
    """
    text_lower = article_text.lower()
    found = {}
    for name in athlete_names:
        name_lower = name.lower()
        idx = text_lower.find(name_lower)
        if idx == -1:
            continue
        dnf_window = text_lower[idx: idx + len(name_lower) + 40]
        if not re.search(r"\bdnf\b", dnf_window):
            continue
        context_window = text_lower[max(0, idx - 100): idx + len(name_lower) + 300]
        found[name] = estimate_recovery_weeks(context_window)
    return found


def check_injuries():
    athletes = load_qualified_athletes()
    if not athletes:
        print("  No field found (run live_fetcher.py and/or ultimate_predictions.py first). Skipping.")
        result = {"checked_at": datetime.now(timezone.utc).isoformat(), "sources_ok": [], "athletes": {}}
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        return result

    all_headlines = []
    sources_ok = []
    for source in SOURCES:
        print(f"  Scraping {source['name']}...")
        try:
            headlines = fetch_headlines(source)
            print(f"    {len(headlines)} headlines fetched")
            all_headlines.extend(headlines)
            sources_ok.append(source["name"])
        except Exception as e:
            print(f"    WARNING: {source['name']} failed ({e})")

    # The wide net. Deduped by headline text, because several queries overlap
    # and the same story is syndicated under the same title.
    print("  Searching Google News...")
    seen_headlines = {h["headline"] for h in all_headlines}
    google_found = 0
    for query in GOOGLE_NEWS_QUERIES:
        try:
            for item in fetch_google_news(query):
                if item["headline"] in seen_headlines:
                    continue
                seen_headlines.add(item["headline"])
                all_headlines.append(item)
                google_found += 1
        except Exception as e:
            print(f"    WARNING: query {query[:40]!r} failed ({e})")
    if google_found:
        print(f"    {google_found} additional headlines")
        sources_ok.append("googlenews")

    days_to_final = (target_date() - date.today()).days

    flags = {}
    event_terms = target_event_terms()
    # name -> newest date on a story saying this athlete is BACK. Kept so a
    # withdrawal can be overtaken by later news rather than standing forever.
    returned = {}
    aliases = build_aliases(athletes)
    # Normalise each headline once: 700-odd aliases against several hundred
    # headlines is the hot loop, and the fold is the same for every alias.
    prepared = [(item, normalize_for_match(item["headline"])) for item in all_headlines]

    for name, discipline_keys in athletes.items():
        mine = [a for a, (owner, _) in aliases.items() if owner == name]
        if not mine:
            continue
        for item, headline_norm in prepared:
            if is_other_sport(headline_norm):
                continue
            here = names_in(headline_norm, aliases)
            # Prefer a full-name match over a bare surname when the headline
            # carries both, so "Noah Lyles ends season" is treated as the
            # strong evidence it is rather than as a surname sighting.
            ours = [a for a in here if a in mine]
            hit = next((a for a in ours if not aliases[a][1]), None) or (ours[0] if ours else None)
            if hit is None:
                continue
            # A bare surname on a headline that never mentions the sport is
            # almost always somebody else with the same name.
            if aliases[hit][1] and not reads_like_athletics(headline_norm):
                continue
            # A surname on its own is weaker evidence than a full name, so it
            # can raise a flag but never a removal. "Duplantis withdraws" is
            # almost certainly the right Duplantis; it is still one word.
            surname_only = aliases[hit][1]
            matched = match_keywords(headline_norm)
            # A headline naming two athletes must not flag both. Keep only the
            # keywords this athlete is actually the nearest named subject of
            # -- see keyword_is_about() for the real case that motivated it.
            others = [a for a in here if aliases[a][0] != name]
            if others:
                matched = {
                    bucket: [kw for kw in kws
                             if keyword_is_about(headline_norm, hit, [hit] + others, kw)]
                    for bucket, kws in matched.items()
                }
            # Returns are handled BEFORE the keyword test, and on their own
            # attribution. A comeback still only counts as news if the story
            # does not also say the athlete is out.
            if is_returning_story(headline_norm) and returning_is_about(headline_norm, hit, others):
                when = item.get("published")
                if when and when > returned.get(name, ""):
                    returned[name] = when
                if not matched["remove"]:
                    continue
            if not matched["remove"] and not matched["watch"]:
                continue
            status = "watch" if surname_only else ("remove" if matched["remove"] else "watch")
            # A headline that names the competition AND says the athlete is not
            # in it is not a doubt, whichever way the name matched.
            if (names_target_event(headline_norm, event_terms)
                    and any(w in headline_norm for w in OUT_OF_EVENT_WORDS)):
                status = "remove"

            recovery = estimate_recovery_weeks(headline_norm)
            likely_out_for_final = recovery is not None and (recovery[0] * 7 > days_to_final)
            if likely_out_for_final and not surname_only and not is_returning_story(headline_norm):
                status = "remove"

            entry = flags.setdefault(name, {
                "status": "watch",
                "disciplines": sorted(discipline_keys),
                "matches": [],
            })
            upgrade_status(entry, status)
            match_record = {
                "headline": item["headline"],
                "url": item["url"],
                "source": item["source"],
                "keywords": matched["remove"] + matched["watch"],
            }
            if item.get("published"):
                match_record["published"] = item["published"]
            if recovery is not None:
                match_record["estimated_recovery_weeks"] = list(recovery)
                match_record["likely_out_for_final"] = likely_out_for_final
            entry["matches"].append(match_record)

    # Cross-check qualified athletes against recent full meet-results recaps
    # for a DNF -- catches cases like a mid-race injury that never gets its
    # own headline (it's just a bare "DNF" in a results table).
    results_articles = find_results_articles(all_headlines)
    for article in results_articles:
        print(f"  Checking meet results: {article['headline']}")
        try:
            text = fetch_article_text(article["url"], article["source"])
        except Exception as e:
            print(f"    WARNING: failed to fetch results article ({e})")
            continue
        for name, recovery in find_dnf_athletes(text, athletes.keys()).items():
            likely_out_for_final = recovery is not None and (recovery[0] * 7 > days_to_final)
            status = "remove" if likely_out_for_final else "watch"

            entry = flags.setdefault(name, {
                "status": "watch",
                "disciplines": sorted(athletes[name]),
                "matches": [],
            })
            upgrade_status(entry, status)
            match_record = {
                "headline": article["headline"],
                "url": article["url"],
                "source": f"{article['source']}_results",
                "keywords": ["dnf"],
            }
            if recovery is not None:
                match_record["estimated_recovery_weeks"] = list(recovery)
                match_record["likely_out_for_final"] = likely_out_for_final
            entry["matches"].append(match_record)

    # A flag stands only until the athlete is reported back. Compared on dates
    # rather than assumed: a withdrawal with no date cannot be overtaken,
    # because there is no way to tell which came first.
    superseded = []
    for name in list(flags):
        back = returned.get(name)
        if not back:
            continue
        newest_flag = max((m.get("published") or "") for m in flags[name]["matches"])
        if newest_flag and back > newest_flag:
            superseded.append({"athlete": name, "flagged_on": newest_flag, "returned_on": back})
            del flags[name]
    for entry in superseded:
        print(f"  CLEARED {entry['athlete']}: flagged {entry['flagged_on']}, "
              f"reported back {entry['returned_on']}")

    result = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "sources_ok": sources_ok,
        "days_to_final": days_to_final,
        "athletes": flags,
        # Recorded rather than dropped silently: "why is X not flagged?" should
        # have an answer in the file.
        "superseded": superseded,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result


def load_injury_flags():
    """Returns {athlete_name -> {status, disciplines, matches}}, or {} if unavailable."""
    if not os.path.exists(OUTPUT_PATH):
        return {}
    try:
        with open(OUTPUT_PATH, encoding="utf-8") as f:
            return json.load(f).get("athletes", {})
    except (json.JSONDecodeError, OSError):
        return {}


if __name__ == "__main__":
    _force_utf8_stdout()
    print("=== Checking for athlete injuries / withdrawals ===")
    outcome = check_injuries()
    flagged = outcome["athletes"]
    if flagged:
        print(f"\n  {len(flagged)} athlete(s) flagged:")
        for name, info in flagged.items():
            print(f"    [{info['status'].upper()}] {name} - {info['matches'][0]['headline']}")
    else:
        print("\n  No injury/withdrawal mentions found.")
