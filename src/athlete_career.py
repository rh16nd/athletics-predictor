"""
athlete_career.py -- what an athlete has already won, and where World
Athletics ranks them, from the profiles src/athlete_profile_scraper.py
fetches.

WHY THIS IS A SEPARATE MODULE FROM athlete_analytics
----------------------------------------------------
athlete_analytics computes things FROM the race log: win rates, form,
head-to-head. Everything here is stated by World Athletics directly --
medals, titles, ranking places -- and is read, not derived. Keeping the two
apart means a bug in one cannot quietly change the other's numbers, and it
keeps obvious the fact that these are claims WA makes rather than claims
this project computes.

THE CATEGORY LABEL IS LOAD-BEARING
----------------------------------
`honours` comes back as anonymous groups of results. The first fetch did not
request `categoryName` and the data was unusable for anything but a raw
count: 3,787 first places across the field, which mixes Olympic titles with
national championships and Diamond League Final wins indiscriminately.
With the label, WA names each group itself -- "Olympic Games", "World
Championships", "Diamond League Final", "World Indoor Championships",
"World Athletics Relays" -- so a claim like "Olympic champion" is WA's own
classification rather than this module's guess at one from a competition
string.

Nothing here feeds the model.
"""
import json
import os
import re

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROFILE_DIR = os.path.join(BASE_DIR, "data", "athlete_profiles")
INDEX_PATH = os.path.join(PROFILE_DIR, "_index.json")

# Which honour categories may appear in the one-line summary, in two tiers.
# Surveyed from the real data rather than assumed: 51 distinct categories
# appear across the fetched profiles.
#
# GLOBAL is what settles an argument on its own. CONTINENTAL is a real
# senior title and is used only when an athlete has nothing global, always
# named precisely ("Commonwealth champion", never just "champion") so it can
# never be mistaken for one.
GLOBAL_CATEGORIES = [
    ("Olympic Games",              "Olympic"),
    ("World Championships",        "World"),
    ("Diamond League Final",       "Diamond League Final"),
    ("World Indoor Championships", "World Indoor"),
]

CONTINENTAL_CATEGORIES = [
    ("European Championships",       "European"),
    ("Commonwealth Games",           "Commonwealth"),
    ("Asian Games",                  "Asian Games"),
    ("Asian Championships",          "Asian"),
    ("African Championships",        "African"),
    ("Pan American Games",           "Pan American"),
    ("NACAC Championships",          "NACAC"),
    ("South American Championships", "South American"),
    ("Oceania Championships",        "Oceania"),
]

# Never headlined, whatever the medal count. An age-group title is not a
# senior credential, and the field is full of them: World U20 appears on 37
# of the surveyed athletes, European U23 on 22. "National Championships" is
# the single most common category of all (87 athletes, 354 golds) and
# "Diamond League" without "Final" counts individual MEETING wins, 392 of
# them -- neither is a title in the sense this line implies.
NEVER_HEADLINE = re.compile(r"U1[0-9]|U2[0-9]|youth|junior|NCAA|National", re.I)

MEDAL_WORDS = {1: "champion", 2: "silver", 3: "bronze"}

_PLACE = re.compile(r"^(\d+)")
_cache = {}


def _place(value):
    match = _PLACE.match(str(value or "").strip())
    return int(match.group(1)) if match else None


def _index():
    try:
        with open(INDEX_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _load_profile(athlete_name):
    """Profiles are keyed by World Athletics id on disk, so the index is the
    only way from a name to a file. Cached on the index's mtime so a
    re-fetch is picked up without a restart -- the same reason
    api.load_season_scores keys its cache on file state rather than on a
    name."""
    try:
        stamp = os.path.getmtime(INDEX_PATH)
    except OSError:
        return None
    if _cache.get("_stamp") != stamp:
        _cache.clear()
        _cache["_stamp"] = stamp
        _cache["_byname"] = {v.get("name"): k for k, v in _index().items()}

    athlete_id = (_cache.get("_byname") or {}).get(athlete_name)
    if athlete_id is None:
        return None
    if athlete_id in _cache:
        return _cache[athlete_id]
    try:
        with open(os.path.join(PROFILE_DIR, f"{athlete_id}.json"), encoding="utf-8") as f:
            profile = json.load(f).get("profile")
    except (OSError, ValueError):
        profile = None
    _cache[athlete_id] = profile
    return profile


def world_ranking(profile):
    """WA's own current ranking place per event group, best event first.
    The "Overall Ranking" row is kept separately -- it is a different scale
    and would sort nonsensically alongside per-event places."""
    current = ((profile or {}).get("worldRankings") or {}).get("current") or []
    events, overall = [], None
    for row in current:
        group = row.get("eventGroup") or ""
        place = row.get("place")
        if place is None:
            continue
        if "Overall" in group:
            overall = int(place)
        else:
            events.append({"event": group, "place": int(place)})
    events.sort(key=lambda e: e["place"])
    return {"events": events, "overall": overall, "best": events[0] if events else None}


def honours(profile):
    """Every honour WA lists, grouped under its own category label, with a
    medal count per category. Podium places only for the counts: a sixth
    place at the Olympics is a real result and stays in the list, but it is
    not something to headline."""
    groups = []
    for group in (profile or {}).get("honours") or []:
        results = group.get("results") or []
        if not results:
            continue
        medals = {1: 0, 2: 0, 3: 0}
        entries = []
        for result in results:
            place = _place(result.get("place"))
            if place in medals:
                medals[place] += 1
            entries.append({
                "competition": result.get("competition"),
                "mark":        result.get("mark"),
                "place":       place,
            })
        groups.append({
            "category": group.get("categoryName"),
            "results":  entries,
            "gold":     medals[1],
            "silver":   medals[2],
            "bronze":   medals[3],
            "podiums":  sum(medals.values()),
        })
    # Ordered by what the championship IS, not by how many were won there.
    # Sorting on gold count alone put "Diamond League 11" and "National
    # Championships 5" above "Olympic Games 2" on Dalilah Muhammad's page,
    # which reads as though eleven meeting wins outrank two Olympic titles.
    # Tier first, gold count only to break ties inside a tier.
    def rank(group):
        category = group["category"] or ""
        for i, (name, _) in enumerate(GLOBAL_CATEGORIES):
            if category == name:
                return (0, i)
        for i, (name, _) in enumerate(CONTINENTAL_CATEGORIES):
            if category == name:
                return (1, i)
        return (2, 0)

    groups.sort(key=lambda g: (*rank(g), -g["gold"], -g["podiums"]))
    return groups


def _headline_parts(groups, tiers):
    by_category = {g["category"]: g for g in groups}
    parts = []
    for category, label in tiers:
        group = by_category.get(category)
        if not group or not group["podiums"]:
            continue
        if NEVER_HEADLINE.search(category):
            continue
        for place in (1, 2, 3):
            count = (group["gold"] if place == 1
                     else group["silver"] if place == 2 else group["bronze"])
            if not count:
                continue
            times = "" if count == 1 else f"{count}× "
            parts.append(f"{times}{label} {MEDAL_WORDS[place]}")
            break   # the best colour in a category is the one worth saying
    return parts


def headline(groups):
    """One line an analyst can read at a glance: the strongest thing this
    athlete has done, in World Athletics' own words.

    Global titles first. Only if there are none does it fall back to
    continental ones, and those are always named ("Commonwealth champion")
    rather than shortened to "champion", so the two can never be confused.
    Age-group, national and NCAA titles are excluded outright -- a World U20
    gold is a real result and stays in the full list below, but rendering it
    as "champion" beside an Olympic one would be a lie of omission.

    Returns None rather than inventing a consolation phrase."""
    parts = _headline_parts(groups, GLOBAL_CATEGORIES)
    if not parts:
        parts = _headline_parts(groups, CONTINENTAL_CATEGORIES)
    return " · ".join(parts[:3]) or None


def personal_bests(profile, limit=None):
    """Every discipline WA holds a PB for, newest-dated first. The count is
    the interesting part on its own: 257 of 323 athletes have PBs in four or
    more events, which tells you whether a contender is a specialist or a
    doubler before you read a single mark."""
    results = ((profile or {}).get("personalBests") or {}).get("results") or []
    bests = [{
        "discipline": r.get("discipline"),
        "mark":       r.get("mark"),
        "venue":      r.get("venue"),
        "date":       r.get("date"),
        # WA writes indoor marks into the same list, tagged only by a "(i)"
        # venue suffix -- the same trap as the outdoor toplists.
        "indoor":     bool(re.search(r"\(i\)\s*$|\bindoor\b", str(r.get("venue") or ""), re.I)),
    } for r in results]
    return bests if limit is None else bests[:limit]


def build_career(athlete_name):
    """The whole read-from-WA block for one athlete, or None when no profile
    has been fetched for them (src/athlete_profile_scraper.py covers the
    athletes the site renders pages for, not all 7,628 in the race log)."""
    profile = _load_profile(athlete_name)
    if not profile:
        return None
    groups = honours(profile)
    ranking = world_ranking(profile)
    bests = personal_bests(profile)
    if not groups and not ranking["events"] and not bests:
        return None
    return {
        "headline":      headline(groups),
        "honours":       groups,
        "worldRanking":  ranking,
        "personalBests": bests,
        "eventCount":    len({b["discipline"] for b in bests if b["discipline"]}),
    }
