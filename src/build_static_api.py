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
  /api/search        -- depends on the query. Its *input* does not, though:
                        /api/search-index is snapshotted below and the browser
                        matches against it, so nothing waits on Render for a
                        search. The route stays as a fallback.

Snapshotted per athlete rather than as one payload, but no less static:
  /api/athlete/...        -- the ~240 projected finalists, full profiles
  /api/athlete-status/... -- everyone else search can reach, down to
                             --profile-depth in each discipline's ranking

Usage:
    python src/build_static_api.py [output_dir] [--profile-depth N]
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
        ("/api/results", "results.json"),
        ("/api/stats", "stats.json"),
        ("/api/ultimate", "ultimate.json"),
        ("/api/world-rankings", "world-rankings.json"),
        ("/api/news", "news.json"),
        ("/api/qualification", "qualification.json"),
        ("/api/countries", "countries.json"),
        ("/api/search-index", "search-index.json"),
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


# How deep into each discipline's world ranking to snapshot the "not in the
# projected field" page. Search reaches all ~4,000 ranked athletes, and only
# ~240 of them are projected finalists with a full profile; everyone else lands
# on athlete-status instead, which was the last thing a visitor could click and
# still wait out Render's cold start.
#
# Measured 2026-09-10, per athlete: 12.4 KB and 1.28s to build, because
# athlete_field_status calls World Athletics for the season history and the
# headshot. That makes DEPTH a real trade rather than a free win, and the cost
# is paid again on every refresh:
#
#     top 20   308 new files   3.7 MB    6.6 min   17% of the index
#     top 50  1277 new files  15.5 MB   27.2 min   41%
#     top 100 2843 new files  34.4 MB   60.7 min   80%
#     all     3643 new files  44.1 MB   77.7 min  100%
#
# 50 is the default because it covers the results a search actually puts in
# front of someone -- the list is sorted by world rank, so the visible rows are
# the ranked ones -- without doubling the length of a refresh. Raise it with
# --profile-depth if a deeper tail matters more than the build time.
DEFAULT_PROFILE_DEPTH = 50


def status_pairs(client, depth, already):
    """(discipline key, athlete name) for the ranked athletes who do NOT have a
    full profile, down to `depth` in each discipline's world ranking.

    Read from the search index rather than the toplist CSVs, so the set can
    never be wider than what search can actually reach."""
    res = client.get("/api/search-index")
    if res.status_code != 200:
        return []
    pairs = []
    seen = set()
    for name, disc_key, _mark, rank in (res.get_json() or {}).get("athletes") or []:
        if rank is None or rank > depth:
            continue
        if (disc_key, name) in already or (disc_key, name) in seen:
            continue
        seen.add((disc_key, name))
        pairs.append((disc_key, name))
    return pairs


def write_snapshots(client, out_dir, subdir, pairs, label, quote_name=True):
    """Write one file per (discipline, athlete), named by the same slug the
    frontend computes. Returns (written, skipped, bytes).

    Two athletes in one discipline whose names fold to the same slug would
    serve one of them the other's page, so neither is written and both fall
    back to the live API -- correct by construction rather than by luck."""
    written = skipped = total = 0
    taken = {}
    for i, (key, name) in enumerate(pairs, 1):
        slug = athlete_slug(name)
        if not slug:
            print(f"  SKIP {label} {key}/{name} -> no usable slug")
            skipped += 1
            continue
        clash = taken.get((key, slug))
        if clash and clash != name:
            print(f"  SKIP {key}/{slug} -> slug collision: {clash!r} vs {name!r}")
            skipped += 1
            continue
        path = f"/api/{subdir}/{key}/{quote(name) if quote_name else name}"
        res = client.get(path)
        if res.status_code != 200:
            print(f"  SKIP {path} -> HTTP {res.status_code}")
            skipped += 1
            continue
        taken[(key, slug)] = name
        dest = os.path.join(out_dir, subdir, key, slug + ".json")
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        payload = json.dumps(res.get_json(), ensure_ascii=False, separators=(",", ":"))
        with open(dest, "w", encoding="utf-8") as f:
            f.write(payload)
        total += len(payload.encode("utf-8"))
        written += 1
        if i % 50 == 0:
            print(f"    ...{i}/{len(pairs)} {label}")
    return written, skipped, total

def build(out_dir, depth=DEFAULT_PROFILE_DEPTH):
    client = api.app.test_client()
    profiles = [0]
    countries = [0]
    statuses = [0]
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
    pairs = athlete_pairs(client)
    w, s, t = write_snapshots(client, out_dir, "athlete", pairs, "profiles")
    written, skipped, total = written + w, skipped + s, total + t
    profiles[0] = w

    # The other 3,700. Search reaches every ranked athlete, but only the
    # projected finalists above have a profile -- everyone else 404s there and
    # the page asks /api/athlete-status why they are not in the field. That
    # follow-up was the last click on the site that could still wait out
    # Render's cold start, so it is snapshotted too, down to `depth`.
    w, s, t = write_snapshots(
        client, out_dir, "athlete-status",
        status_pairs(client, depth, set(pairs)), "status pages",
    )
    written, skipped, total = written + w, skipped + s, total + t
    statuses[0] = w

    return written, skipped, total, profiles[0], countries[0], statuses[0]


if __name__ == "__main__":
    args = [a for a in sys.argv[1:]]
    depth = DEFAULT_PROFILE_DEPTH
    if "--profile-depth" in args:
        i = args.index("--profile-depth")
        depth = int(args[i + 1])
        del args[i:i + 2]
    out_dir = os.path.abspath(args[0] if args else DEFAULT_OUT)
    print("=== Building static API snapshot for the CDN ===")
    print(f"  -> {out_dir}")
    print(f"  status pages down to world rank {depth} per discipline")
    written, skipped, total, profiles, countries, statuses = build(out_dir, depth)
    print(f"\n  {written} files written ({profiles} athlete profiles, {statuses} status pages, "
          f"{countries} countries), {skipped} skipped, {total/1024:.0f} KB total (uncompressed)")
