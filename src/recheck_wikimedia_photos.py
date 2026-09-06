"""
recheck_wikimedia_photos.py -- re-queries Wikimedia Commons for the athletes
the site currently shows with NO photo at all (no World Athletics asset AND no
cached Wikimedia fallback), picking up photos that Commons/Wikidata have added
since data/wikimedia_photo_cache.json was last built.

Why this is needed: load_wikimedia_photo() caches negatives ("this athlete has
no Commons photo") so a miss isn't re-queried on every page view. That is right
for serving, but it means a photo added later is never noticed. This script
clears and re-checks only those negative/absent entries.

Two safety properties, both deliberate:
  * The match stays EXACT -- Wikidata P1146 == World Athletics id -- so we never
    attach the wrong person's face. Same path as api.load_wikimedia_photo.
  * A transient failure (429/timeout) is NEVER written as a false "no photo".
    The SPARQL call is retried; on persistent failure the entry is left as-is
    for a later run, exactly like warm_photo_focus.py does for images.

Only athletes WITHOUT a WA photo are re-checked (WA's own asset stays primary,
so an athlete who already has one gains nothing from a Wikimedia fallback).

Usage:
    python src/recheck_wikimedia_photos.py
Then run:  python src/warm_photo_focus.py   (to face-focus any new photos)
"""
import csv
import json
import os
import re
import sys
import time

import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import api  # noqa: E402

OUTPUTS = os.path.join(os.path.dirname(__file__), "..", "outputs")
DATA = os.path.join(os.path.dirname(__file__), "..", "data")
RETRY_WAITS = [5, 15, 30]


def competitor_id(url):
    if not isinstance(url, str):
        return None
    m = re.search(r"athlete=(\d+)", url) or re.search(r"-(\d+)/?$", url)
    return int(m.group(1)) if m else None


def displayed_athletes():
    """{wa_id -> (name, profile_url)} for everyone the site renders: the
    projected field + near-miss (predictions_latest) and every Final finisher
    (via their toplist profile URL, which is where the outside finishers come
    from)."""
    out = {}
    pred = pd.read_csv(os.path.join(OUTPUTS, "predictions_latest.csv"))
    for _, r in pred.iterrows():
        c = competitor_id(r.get("profile_url"))
        if c:
            out.setdefault(c, (r["athlete_name"], r["profile_url"]))
    results_path = os.path.join(DATA, "dl_final_2026_results.csv")
    if os.path.exists(results_path):
        for row in csv.DictReader(open(results_path, encoding="utf-8")):
            _m, _r, wa = api.toplist_entry(row["discipline"], row["athlete_name"])
            c = competitor_id(wa)
            if c and c not in out:
                out[c] = (row["athlete_name"], wa)
    # The Track/Field pages now show the world top-20 per discipline
    # (src/world_rankings.py), which is a far bigger cast than the projected
    # Diamond League field -- and most of it had never been checked for a
    # Commons photo. Include it, or the re-check misses the athletes the site
    # actually renders now.
    wr_path = os.path.join(DATA, "world_rankings.json")
    if os.path.exists(wr_path):
        with open(wr_path, encoding="utf-8") as f:
            wr = json.load(f)
        for disc in wr.values():
            for lst in ("model", "points"):
                for r in disc.get(lst, []):
                    c = competitor_id(r.get("profileUrl"))
                    if c and c not in out:
                        out[c] = (r.get("name"), r.get("profileUrl"))
    return out


def has_wa_photo(ids):
    """Set of competitor ids with a live World Athletics photo, batched."""
    have = set()
    for i in range(0, len(ids), 60):
        chunk = ids[i:i + 60]
        try:
            data = api.dlr.graphql("GetAthletePhoto", {"ids": chunk}, api.PHOTO_QUERY)
            for e in (data.get("getAthleteActionPictureByIds") or []):
                if e and e.get("primaryMediaId"):
                    have.add(int(e["id"]))
        except Exception as ex:
            print(f"  WA batch error: {str(ex)[:60]}")
        time.sleep(0.4)
    return have


def wikidata_title(wa_id):
    """('ok', title|None) on a successful query, ('err', None) on persistent
    failure -- so a transient error is never mistaken for 'no image'."""
    query = (
        'SELECT ?image WHERE { ?item wdt:P1146 "%s". '
        "OPTIONAL { ?item wdt:P18 ?image. } } LIMIT 1" % wa_id
    )
    for attempt in range(len(RETRY_WAITS) + 1):
        try:
            r = requests.get(
                "https://query.wikidata.org/sparql",
                params={"format": "json", "query": query},
                headers=api.WM_HEADERS,
                timeout=25,
            )
            if r.status_code in (429, 503) and attempt < len(RETRY_WAITS):
                time.sleep(RETRY_WAITS[attempt])
                continue
            r.raise_for_status()
            bindings = r.json()["results"]["bindings"]
        except Exception:
            if attempt < len(RETRY_WAITS):
                time.sleep(RETRY_WAITS[attempt])
                continue
            return "err", None
        if not bindings or "image" not in bindings[0]:
            return "ok", None
        import urllib.parse
        filename = urllib.parse.unquote(bindings[0]["image"]["value"].rsplit("/", 1)[-1])
        return "ok", "File:" + filename
    return "err", None


def main():
    athletes = displayed_athletes()
    ids = list(athletes)
    print(f"{len(ids)} displayed athletes; checking World Athletics photos...")
    wa = has_wa_photo(ids)
    print(f"  {len(wa)} have a WA photo; {len(ids) - len(wa)} do not")

    cache = api._load_json_cache(api.WIKIMEDIA_CACHE_PATH)
    # Re-check only the no-photo-anywhere athletes: no WA asset, and no cached
    # Wikimedia photo yet (a null entry or no entry at all).
    targets = [
        c for c in ids
        if c not in wa and not (cache.get(str(c)))
    ]
    print(f"  {len(targets)} athletes have NO photo anywhere -- re-checking Commons\n")

    found = still_none = errors = 0
    for i, c in enumerate(targets, 1):
        name = athletes[c][0]
        status, title = wikidata_title(c)
        if status == "err":
            errors += 1
            print(f"  [{i}/{len(targets)}] {name}: query failed, left for next run")
            time.sleep(1.2)
            continue
        result = api._commons_photo(title) if title else None
        cache[str(c)] = result  # definitive: overwrite the stale negative
        if result:
            found += 1
            print(f"  [{i}/{len(targets)}] {name}: FOUND {result['url'].split('/')[-1][:45]}")
        else:
            still_none += 1
        api._save_json_cache(api.WIKIMEDIA_CACHE_PATH, cache)
        time.sleep(1.2)

    print(
        f"\n  new photos found: {found}   still none: {still_none}   "
        f"query failures (left for next run): {errors}"
    )
    print(f"  Saved -> {os.path.abspath(api.WIKIMEDIA_CACHE_PATH)}")
    if found:
        print("  Next: python src/warm_photo_focus.py  (face-focus the new photos)")


if __name__ == "__main__":
    print("=== Re-checking Wikimedia Commons for athletes with no photo ===")
    main()
