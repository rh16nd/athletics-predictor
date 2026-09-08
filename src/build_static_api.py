"""
build_static_api.py -- writes the API's precomputed responses out as static
JSON for the frontend to serve from Vercel's CDN.

Why this exists: the API is hosted on Render's free tier, which spins the
instance down after ~15 minutes idle. Measured 2026-09-06, a cold request took
**32.7s to first byte** (warm: 1.3s; Vercel itself: 1.2s), so the first visitor
after a quiet spell watched skeletons for half a minute. Almost none of that is
our code -- the whole app imports in ~3s -- it is Render's container start.

The fix is architectural rather than a workaround: nearly every endpoint the
first paint needs is a *snapshot*, not a live computation. It only changes when
a data refresh runs and is pushed. Served as static files there is no server in
the critical path at all, so there is no cold start to have.

Responses are produced through Flask's own test client, deliberately: the files
are then byte-identical to what the live API returns, and cannot drift from the
route logic the way a re-implementation would.

NOT snapshotted (genuinely dynamic, still served by Render):
  /api/search        -- depends on the query
  /api/athlete/...   -- one response per athlete, resolved photos and all

Usage:
    python src/build_static_api.py [output_dir]
Default output: ../track-insights-main/public/data
Run it after any data refresh, then commit BOTH repos (see HANDOFF).
"""
import json
import os
import re
import sys
import unicodedata
from urllib.parse import quote

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import api  # noqa: E402

# Windows consoles are cp1252 by default, and one athlete name outside it
# (Yeral NUÑEZ, Adriana VILAGOŠ, anyone with a Č) turned a single SKIP log
# line into a UnicodeEncodeError that killed the run half-written -- leaving
# exactly the partially-stale public/data this script exists to prevent.
# A progress message is never worth losing the snapshot for.
#
# reconfigure() rather than the sys.stdout = TextIOWrapper(sys.stdout.buffer)
# idiom the scrapers use: that one detaches the stream underneath whoever else
# holds it, which is why importing those modules under pytest tears the session
# down (see tests/test_dl_final_results_scraper.py). This keeps the same object.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError, OSError):
        pass

DEFAULT_OUT = os.path.join(
    os.path.dirname(__file__), "..", "..", "track-insights-main", "public", "data"
)


def snapshot_paths():
    """(url path, output file) for every response that is a precomputed
    snapshot. The per-discipline pages are included because they are linked
    straight off the rankings, and are just as static as the rest."""
    pairs = [
        ("/api/predictions", "predictions.json"),
        ("/api/stats", "stats.json"),
        ("/api/ultimate", "ultimate.json"),
        ("/api/world-rankings", "world-rankings.json"),
        ("/api/news", "news.json"),
        ("/api/qualification", "qualification.json"),
        ("/api/countries", "countries.json"),
    ]
    for key in api.DISC_LABELS:
        pairs.append((f"/api/discipline/{key}", f"discipline/{key}.json"))
    return pairs


def athlete_slug(name):
    """A filename-safe key for an athlete name, computed identically here and
    in the frontend (athleteSlug in src/lib/api.ts).

    Athlete names carry spaces, apostrophes and accents, and the request path
    is percent-encoded -- none of which makes a good static file name. So both
    sides fold the name to ASCII-ish lowercase and join on hyphens.

    The two implementations agree on every name in the data (checked below by
    the collision guard), and the failure mode if they ever disagreed is safe:
    the static file simply 404s and apiFetch falls back to the live API."""
    s = unicodedata.normalize("NFD", name)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def athlete_pairs(client):
    """(discipline key, athlete name) for every profile the site can link to:
    the projected field, the near-miss list, and the Final finishers that carry
    a profile page. Read from the predictions response rather than the CSV, so
    it is exactly the set the UI actually offers a link to."""
    res = client.get("/api/predictions")
    if res.status_code != 200:
        return []
    data = res.get_json()
    pairs = []
    seen = set()
    for group in ("trackDisciplines", "fieldDisciplines"):
        for disc in data.get(group) or []:
            key = disc.get("id")
            names = [a.get("name") for a in (disc.get("athletes") or [])]
            names += [a.get("name") for a in (disc.get("nearMiss") or [])]
            names += [r.get("name") for r in ((disc.get("result") or {}).get("rows") or [])
                      if r.get("hasPage")]
            for n in names:
                if n and (key, n) not in seen:
                    seen.add((key, n))
                    pairs.append((key, n))
    return pairs

def build(out_dir):
    client = api.app.test_client()
    profiles = [0]
    countries = [0]
    total = 0
    written = skipped = 0
    for path, name in snapshot_paths():
        res = client.get(path)
        if res.status_code != 200:
            # A 404 here means the pipeline has not produced that data yet.
            # Skip it rather than freezing an error page into a static file --
            # apiFetch falls back to the live API for anything missing.
            print(f"  SKIP {path} -> HTTP {res.status_code}")
            skipped += 1
            continue
        dest = os.path.join(out_dir, name)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        payload = json.dumps(res.get_json(), ensure_ascii=False, separators=(",", ":"))
        with open(dest, "w", encoding="utf-8") as f:
            f.write(payload)
        total += len(payload.encode("utf-8"))
        written += 1

    # Country pages. One per nation that has a ranked athlete this season --
    # the country view is what gives the Ultimate's two mixed relays somewhere
    # to live, since a relay result belongs to a nation rather than an athlete.
    res = client.get("/api/countries")
    codes = [c["code"] for c in (res.get_json() or {}).get("countries", [])] if res.status_code == 200 else []
    for code in codes:
        r = client.get(f"/api/country/{code}")
        if r.status_code != 200:
            print(f"  SKIP /api/country/{code} -> HTTP {r.status_code}")
            skipped += 1
            continue
        dest = os.path.join(out_dir, "country", f"{code}.json")
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        payload = json.dumps(r.get_json(), ensure_ascii=False, separators=(",", ":"))
        with open(dest, "w", encoding="utf-8") as f:
            f.write(payload)
        total += len(payload.encode("utf-8"))
        written += 1
        countries[0] += 1

    # Athlete profiles. Per-athlete rather than one payload, but just as
    # static -- and snapshotting them removes a live World Athletics GraphQL
    # call from every profile view, since load_athlete_photo() resolves the
    # headshot on each request.
    taken = {}
    for i, (key, name) in enumerate(athlete_pairs(client), 1):
        slug = athlete_slug(name)
        if not slug:
            print(f"  SKIP profile {key}/{name} -> no usable slug")
            skipped += 1
            continue
        clash = taken.get((key, slug))
        if clash and clash != name:
            # Two athletes in one discipline folding to the same file would
            # serve one of them the other's career. Write neither; both fall
            # back to the live API, which is correct by construction.
            print(f"  SKIP {key}/{slug} -> slug collision: {clash!r} vs {name!r}")
            skipped += 1
            continue
        res = client.get(f"/api/athlete/{key}/{quote(name)}")
        if res.status_code != 200:
            print(f"  SKIP /api/athlete/{key}/{name} -> HTTP {res.status_code}")
            skipped += 1
            continue
        taken[(key, slug)] = name
        dest = os.path.join(out_dir, "athlete", key, slug + ".json")
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        payload = json.dumps(res.get_json(), ensure_ascii=False, separators=(",", ":"))
        with open(dest, "w", encoding="utf-8") as f:
            f.write(payload)
        total += len(payload.encode("utf-8"))
        written += 1
        profiles[0] += 1
        if i % 50 == 0:
            print(f"    ...{i} profiles")

    return written, skipped, total, profiles[0], countries[0]


if __name__ == "__main__":
    out_dir = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUT)
    print("=== Building static API snapshot for the CDN ===")
    print(f"  -> {out_dir}")
    written, skipped, total, profiles, countries = build(out_dir)
    print(f"\n  {written} files written ({profiles} athlete profiles, {countries} countries), {skipped} skipped, {total/1024:.0f} KB total (uncompressed)")
