"""
warm_card_photos.py -- resolve the headshots the card views need, once, into
data/card_photo_cache.json.

WHY A CACHE
api.py resolves an athlete's photo live: a GraphQL call to World Athletics,
then a Wikidata + Commons lookup if WA has none. That is fine on a profile
page, which is one athlete. It is not fine on a view made of thirty cards, and
it is much worse across 137 country pages -- it would mean hundreds of
round-trips to two external APIs on every page load, for a picture.

So the resolution runs here instead, ahead of time, and the API attaches URLs
from the cache with no network at all. Same shape as the two photo caches this
project already keeps (wikimedia_photo_cache.json, photo_focus_cache.json) and
committed for the same reason: a fresh clone should not have to re-scrape a
face it already knows.

WHO GETS WARMED
Only athletes who actually appear on a card:
  * the model's top-rated athlete in each of the 32 disciplines (the dashboard)
  * each nation's three best athletes (the country pages)
Everyone else is resolved on demand as before. About 400 athletes, which is a
few minutes and then never again until the next data refresh.

WORLD ATHLETICS FIRST, COMMONS ONLY AS A FALLBACK -- the same order the
profile pages use, so an athlete never shows one photo on a card and another on
their own page. Face detection still runs, but it decides how the photo is
CROPPED, not which photo is used.

NO PHOTO IS A REAL ANSWER. World Athletics has none for roughly a third of
athletes and Commons covers about half of those, so a cached `null` is a fact
about the athlete, not a failed lookup, and it is stored as such so the next
run does not retry it. The cards show a monogram, which is what WA's own
rankings cards do.

Usage:
    python src/warm_card_photos.py            # fill in what is missing
    python src/warm_card_photos.py --recheck  # also retry the known misses
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import api  # noqa: E402

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CACHE_PATH = os.path.join(BASE_DIR, "data", "card_photo_cache.json")
RANKINGS_PATH = os.path.join(BASE_DIR, "data", "world_rankings.json")
COUNTRIES_PATH = os.path.join(BASE_DIR, "data", "countries.json")


def load_cache():
    try:
        with open(CACHE_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def save_cache(cache):
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=1, sort_keys=True)


def card_athletes():
    """[(profileUrl, name)] for every athlete a card view shows, de-duplicated.

    Keyed on the profile URL because that is what the photo resolvers take and
    what carries the World Athletics id -- names collide and get respelled,
    ids do not."""
    wanted = {}

    try:
        with open(RANKINGS_PATH, encoding="utf-8") as f:
            rankings = json.load(f)
    except (OSError, json.JSONDecodeError):
        rankings = {}
    for disc in rankings.values():
        for row in (disc.get("model") or [])[:1]:
            if row.get("profileUrl"):
                wanted[row["profileUrl"]] = row.get("name")

    try:
        with open(COUNTRIES_PATH, encoding="utf-8") as f:
            countries = json.load(f)
    except (OSError, json.JSONDecodeError):
        countries = {}
    # A dict keyed by IOC code, each carrying its athletes already ordered by
    # World Athletics score -- so the first three are the three the country
    # page leads with.
    for nation in countries.values():
        for athlete in (nation.get("athletes") or [])[:3]:
            if athlete.get("profileUrl"):
                wanted[athlete["profileUrl"]] = athlete.get("name")
    return list(wanted.items())


def best_photo(profile_url):
    """The photo for a card: World Athletics' own if they have one, a freely
    licensed Wikimedia Commons photo only when they do not.

    WORLD ATHLETICS WINS WHENEVER THEY HAVE ONE. This is the same order
    api.resolve_athlete_photo uses for profile pages, kept deliberately
    identical so an athlete does not appear with one photo on a card and a
    different one on their own page.

    An earlier version of this preferred a Commons portrait whenever the WA
    photo had no face detectable in it -- 32 of the 342 card athletes. The user
    ruled against it: WA's photo is theirs, it is uniform, and it is the right
    picture even when the athlete is mid-stride with their head turned.
    Commons is the fallback for athletes WA has nothing for, and nothing more.

    What DOES still come from the face detection is the crop. Where a face is
    found in the WA action shot, the card centres on it instead of guessing;
    where none is found it falls back to a top-biased crop. That changes how
    the photo is framed, never which photo is used."""
    wa = api.load_athlete_photo(profile_url)
    if wa:
        return wa, None, "wa"
    wm = api.load_wikimedia_photo(profile_url)
    if wm and wm.get("url"):
        return wm["url"], wm.get("credit"), "wikimedia"
    return None, None, "none"


def main(recheck=False):
    cache = load_cache()
    targets = card_athletes()
    print(f"=== Warming card photos: {len(targets)} athletes on card views ===")

    resolved = misses = skipped = 0
    for i, (url, name) in enumerate(targets, 1):
        if url in cache and not (recheck and cache[url].get("url") is None):
            skipped += 1
            continue
        try:
            photo, credit, how = best_photo(url)
        except Exception as e:
            # A failed lookup is NOT a "no photo" answer and must not be
            # cached as one, or the next run will trust it and never retry.
            print(f"  [{i}/{len(targets)}] {name}: lookup failed ({type(e).__name__})")
            continue
        cache[url] = {"url": photo, "credit": credit, "source": how}
        if photo:
            resolved += 1
        else:
            misses += 1
        if i % 25 == 0:
            save_cache(cache)
            print(f"  ...{i}/{len(targets)}")

    save_cache(cache)
    have = sum(1 for v in cache.values() if v.get("url"))
    sources = {}
    for v in cache.values():
        sources[v.get("source") or "?"] = sources.get(v.get("source") or "?", 0) + 1
    print("  chosen by: " + ", ".join(f"{k} {n}" for k, n in sorted(sources.items())))
    print(f"\n  {resolved} newly resolved, {misses} confirmed to have no photo, "
          f"{skipped} already cached")
    # Guarded: a run where every lookup failed leaves the cache empty, and
    # a crash on the summary line would bury the reason under a traceback.
    pct = f" ({100.0 * have / len(cache):.0f}%)" if cache else ""
    print(f"  {have}/{len(cache)} cached athletes have a photo{pct}")
    print(f"  -> {os.path.abspath(CACHE_PATH)}")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--recheck", action="store_true",
                   help="Retry athletes previously found to have no photo. Commons gains "
                        "images over time, so this is worth running occasionally -- but it "
                        "is slow, hence not the default.")
    sys.exit(main(p.parse_args().recheck))
