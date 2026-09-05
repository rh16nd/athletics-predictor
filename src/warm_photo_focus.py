"""
warm_photo_focus.py -- pre-computes the face-focus point for every Wikimedia
Commons fallback photo and writes it into data/photo_focus_cache.json, so the
live site serves a correct face crop on the first view instead of relying on
lazy, per-view detection (which, before the User-Agent fix in api.py's
get_photo_focus, silently 403'd on Wikimedia's image hosts and left every
fallback photo on the top-biased "forehead" crop).

Run it after refreshing data/wikimedia_photo_cache.json (i.e. whenever new
fallback photos are added). It OVERWRITES each Wikimedia photo's focus entry
so a stale null from a previously-failed fetch is recomputed. World Athletics'
own photos already have correct cached focus and are left untouched.

Usage:
    python src/warm_photo_focus.py
"""
import json
import os
import sys
import time

import requests

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import api  # noqa: E402 -- reuse the exact detector + cache the server uses

WM_CACHE = os.path.join(os.path.dirname(__file__), "..", "data", "wikimedia_photo_cache.json")

DELAY = 1.5          # seconds between requests -- be a polite Wikimedia citizen
RETRY_WAITS = [5, 15, 30]  # backoff on 429/503 rather than giving up on a face


def wikimedia_photo_urls():
    with open(WM_CACHE, encoding="utf-8") as f:
        cache = json.load(f)
    urls = []
    for entry in cache.values():
        if entry and isinstance(entry, dict) and entry.get("url"):
            urls.append(entry["url"])
    return urls


def fetch(url):
    """Bytes for a photo, retrying on rate-limit/temporary errors. Returns None
    only on a real, permanent failure -- so a transient 429 never gets cached
    as a false 'no-face'."""
    for attempt in range(len(RETRY_WAITS) + 1):
        try:
            r = requests.get(url, headers=api.WM_HEADERS, timeout=25)
            if r.status_code in (429, 503) and attempt < len(RETRY_WAITS):
                time.sleep(RETRY_WAITS[attempt])
                continue
            r.raise_for_status()
            return r.content
        except Exception:
            if attempt < len(RETRY_WAITS):
                time.sleep(RETRY_WAITS[attempt])
                continue
            return None
    return None


def warm():
    urls = wikimedia_photo_urls()
    focus = api._load_focus_cache()
    detected = null = err = skipped = 0
    for i, url in enumerate(urls, 1):
        # Already computed (a real {x,y} or a real no-face null) -- leave it,
        # so a re-run only does newly-added photos and doesn't re-hammer
        # Wikimedia for the ones already done. Clear the cache entry to force.
        if url in focus:
            skipped += 1
            continue
        content = fetch(url)
        if content is None:
            # A fetch that never succeeded tells us nothing about the face, so
            # don't poison the cache with a null -- drop any stale entry and
            # leave it for a later warm / lazy detection.
            focus.pop(url, None)
            err += 1
            print(f"  [{i}/{len(urls)}] fetch failed, left uncomputed")
        else:
            f = api.detect_face_focus(content)
            focus[url] = f
            if f is None:
                null += 1
            else:
                detected += 1
        api._save_focus_cache(focus)  # checkpoint, so a mid-run stop keeps progress
        time.sleep(DELAY)
    print(
        f"\n  {len(urls)} Wikimedia photos: {detected} newly face-focused, "
        f"{null} no-face (top-crop fallback), {err} fetch failures (left uncomputed), "
        f"{skipped} already cached"
    )
    print(f"  Saved -> {os.path.abspath(api.FOCUS_CACHE_PATH)}")


if __name__ == "__main__":
    if api._get_face_net() is None:
        print("Face detector model unavailable (download failed?) -- aborting.")
        sys.exit(1)
    print("=== Warming face-focus cache for Wikimedia fallback photos ===")
    warm()
