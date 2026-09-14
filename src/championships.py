"""
championships.py -- every championship the site has called or is calling, where
each one's data lives, and which one is current.

The site used to name the Ultimate Championship in a dozen places: its data
directory in api.py, the freeze script, the refresh button and the static build.
Moving on to the Asian Games meant either copying all of that or saying it once.
This says it once.

validate() enforces two rules, and a test calls it:

  * Every championship names a page THEME. It is a tradition the user set when the
    Asian Games page was planned: each championship page is dressed in that
    competition's own colours, and its box on the Results page wears the same
    theme (Diamond League blue, the Ultimate's black and purple with gold, the
    Asian Games from its emblem). A championship without one is not finished.
  * CURRENT names a registered championship. It is what the championship page,
    the refresh button and the freeze script act on when not told otherwise.
"""
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

THEMES = ("diamondLeague", "ultimate", "asianGames")

CHAMPIONSHIPS = [
    {
        # Its comparison predates this registry: it reads outputs/ and
        # data/dl_final_2026_results.csv rather than a championship directory.
        "id": "dl-final-2026",
        "labelKey": "results.meet.dlFinal",
        "venue": "Brussels",
        "startDate": "2026-09-04",
        "endDate": "2026-09-04",
        "theme": "diamondLeague",
        "competitionId": None,
        "area": None,
        "dataDir": None,
        "scraper": None,
    },
    {
        "id": "ultimate-2026",
        "labelKey": "results.meet.ultimate",
        "venue": "Budapest",
        "startDate": "2026-09-11",
        "endDate": "2026-09-13",
        "theme": "ultimate",
        "competitionId": 7212925,
        "area": None,
        "dataDir": os.path.join("data", "ultimate"),
        "scraper": os.path.join("src", "ultimate_scraper.py"),
    },
    {
        # World Athletics calendar, checked 2026-09-14: "20th Asian Games",
        # category A (Area Senior Games), Mizuho Stadium, Nagoya.
        "id": "asian-games-2026",
        "labelKey": "results.meet.asianGames",
        "venue": "Nagoya",
        "startDate": "2026-09-23",
        "endDate": "2026-09-29",
        "theme": "asianGames",
        "competitionId": 7176091,
        "area": "Asia",
        "dataDir": os.path.join("data", "asian_games_2026"),
        "scraper": None,
    },
]

# The Ultimate stays current until the Asian Games has a field and a call to
# show. Flipping this is what moves the championship page, the refresh button
# and the freeze script on to the next competition.
CURRENT = "ultimate-2026"


def get(champ_id):
    for champ in CHAMPIONSHIPS:
        if champ["id"] == champ_id:
            return champ
    raise KeyError(f"no championship {champ_id!r}; registered: {[c['id'] for c in CHAMPIONSHIPS]}")


def current():
    return get(CURRENT)


def path(champ, filename):
    """A file in the championship's data directory, or None for one without."""
    if not champ.get("dataDir"):
        return None
    return os.path.join(BASE_DIR, champ["dataDir"], filename)


def validate():
    """The registry's rules, as a list of problems. Empty when it is sound."""
    problems = []
    ids = [c.get("id") for c in CHAMPIONSHIPS]
    if len(ids) != len(set(ids)):
        problems.append("duplicate championship ids")
    for champ in CHAMPIONSHIPS:
        if champ.get("theme") not in THEMES:
            problems.append(f"{champ.get('id')} has no page theme (expected one of {THEMES})")
        if champ.get("competitionId") and not champ.get("dataDir"):
            problems.append(f"{champ.get('id')} reads World Athletics but has nowhere to keep the data")
    if CURRENT not in ids:
        problems.append(f"CURRENT {CURRENT!r} is not a registered championship")
    return problems
