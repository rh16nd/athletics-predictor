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
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import api  # noqa: E402

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
    ]
    for key in api.DISC_LABELS:
        pairs.append((f"/api/discipline/{key}", f"discipline/{key}.json"))
    return pairs


def build(out_dir):
    client = api.app.test_client()
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
    return written, skipped, total


if __name__ == "__main__":
    out_dir = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUT)
    print("=== Building static API snapshot for the CDN ===")
    print(f"  -> {out_dir}")
    written, skipped, total = build(out_dir)
    print(f"\n  {written} files written, {skipped} skipped, {total/1024:.0f} KB total (uncompressed)")
