"""
venue_geo.py -- resolves the venue strings in data/raw/*.csv to real
coordinates and elevation, cached to data/venues_geo.csv.

Why this exists: the scraped rows carry a venue as free text and nothing
else, so two physically real effects on a mark are invisible to the model --
how high the track is, and what the weather was. Altitude in particular is a
known, signed effect in athletics (thin air helps sprints and jumps, hurts
distance), and the project already corrects marks for wind, so correcting
for altitude is the same idea applied to the other obvious confound.

Two venue formats appear in the data and they need different handling:

  "Hayward Field, Eugene, OR (USA)"   -- toplist rows: place is in the string
  "Athletissima Lausanne"             -- dl_meeting/major_meet rows: a MEETING
                                         NAME, with the city only sometimes
                                         embedded in it

For the second kind this matches against train_model.DL_VENUES (the same
city list competition_weight() already uses) rather than inventing a second
copy of that knowledge.

Everything is cached: this hits the network once per venue, ever. Re-running
only fetches venues not already in the cache, so it is safe to re-run after
new meetings are scraped.

Both APIs are Open-Meteo, free and key-less:
  geocoding-api.open-meteo.com  -- place name -> lat/lon
  api.open-meteo.com/v1/elevation -- lat/lon -> metres (batched 100/request)

Usage:
    python src/venue_geo.py               # top 500 venues by row count
    python src/venue_geo.py --limit 1000  # widen coverage
    python src/venue_geo.py --all         # every venue (slow)
"""
import argparse
import glob
import io
import os
import re
import sys
import time
from collections import Counter

import pandas as pd
import requests

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.dirname(__file__))
from train_model import DL_VENUES  # noqa: E402 -- reuse the existing city list

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
CACHE_PATH = os.path.join(BASE_DIR, "data", "venues_geo.csv")

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
ELEVATION_URL = "https://api.open-meteo.com/v1/elevation"

# Meeting names whose city isn't inferable from the string or DL_VENUES.
# Kept deliberately small and explicit -- everything else is derived.
MEETING_CITY = {
    "athletissima": "Lausanne",
    "bauhaus": "Stockholm",
    "bislett": "Oslo",
    "van damme": "Brussels",
    "weltklasse": "Zurich",
    "herculis": "Monaco",
    "prefontaine": "Eugene",
    "golden spike": "Ostrava",
    "ostrava": "Ostrava",
    "fbk games": "Hengelo",
    "hanzekovic": "Zagreb",
    "hanžekovi": "Zagreb",
    "kusocinski": "Chorzow",
    "kusociński": "Chorzow",
    "szewinska": "Bydgoszcz",
    "szewińska": "Bydgoszcz",
    "janusz": "Chorzow",
    "gyulai": "Szekesfehervar",
    "golden gala": "Rome",
    "meeting de paris": "Paris",
    "british gp": "Birmingham",
    "grand prix gateshead": "Gateshead",
    "gateshead": "Gateshead",
    "anniversary games": "London",
    "london athletics": "London",
    "muller grand prix": "Birmingham",
    "müller grand prix": "Birmingham",
    "silesia": "Chorzow",
    "skolimowska": "Chorzow",
    "mohammed vi": "Rabat",
    "doha": "Doha",
    "shanghai": "Shanghai",
    "suzhou": "Suzhou",
    "shaoxing": "Shaoxing",
    "xiamen": "Xiamen",
    "keqiao": "Shaoxing",
    "ready steady tokyo": "Tokyo",
    "irena": "Bydgoszcz",
}

COUNTRY_RE = re.compile(r"\(([A-Z]{3})\)\s*$")


def venue_to_place(venue):
    """Best-effort (place_name, country_code) for a venue string.

    Returns (None, None) when nothing sensible can be extracted, which the
    caller records as a miss rather than guessing -- a wrong coordinate is
    worse than no coordinate, because it produces a confident wrong altitude
    instead of a visible gap."""
    v = str(venue).strip()
    m = COUNTRY_RE.search(v)
    if m:
        country = m.group(1)
        head = v[: m.start()].strip().rstrip(",")
        parts = [p.strip() for p in head.split(",") if p.strip()]
        if parts:
            # "Hayward Field, Eugene, OR (USA)" -> prefer "Eugene" over the
            # state abbreviation, which geocodes to the wrong thing.
            city = parts[-1]
            if len(city) <= 3 and len(parts) >= 2:
                city = parts[-2]
            return city, country
        return None, country

    low = v.lower()
    for needle, city in MEETING_CITY.items():
        if needle in low:
            return city, None
    for city in DL_VENUES:
        if city in low:
            return city.title(), None
    return None, None


def geocode(place, country=None):
    params = {"name": place, "count": 5, "language": "en", "format": "json"}
    r = requests.get(GEOCODE_URL, params=params, timeout=25)
    r.raise_for_status()
    results = r.json().get("results") or []
    if not results:
        return None
    if country:
        for res in results:
            if res.get("country_code", "").upper() == country[:2].upper():
                return res
        # ISO3 (USA) vs ISO2 (US) -- fall through to the top hit rather than
        # dropping a venue just because the code widths differ.
    return results[0]


def fetch_elevations(coords):
    """Open-Meteo takes up to 100 comma-separated coordinate pairs per call."""
    out = []
    for i in range(0, len(coords), 100):
        chunk = coords[i : i + 100]
        params = {
            "latitude": ",".join(f"{c[0]:.4f}" for c in chunk),
            "longitude": ",".join(f"{c[1]:.4f}" for c in chunk),
        }
        r = requests.get(ELEVATION_URL, params=params, timeout=30)
        r.raise_for_status()
        out.extend(r.json()["elevation"])
        time.sleep(0.4)
    return out


def venue_counts():
    c = Counter()
    for f in glob.glob(os.path.join(RAW_DIR, "*.csv")):
        try:
            d = pd.read_csv(f, usecols=["Venue"])
        except Exception:
            continue
        c.update(d["Venue"].dropna().astype(str))
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=500,
                    help="Resolve the N most common venues (default 500, ~85%% of rows).")
    ap.add_argument("--all", action="store_true", help="Resolve every distinct venue.")
    args = ap.parse_args()

    counts = venue_counts()
    total_rows = sum(counts.values())
    wanted = [v for v, _ in (counts.most_common() if args.all else counts.most_common(args.limit))]

    cached = {}
    if os.path.exists(CACHE_PATH):
        prev = pd.read_csv(CACHE_PATH)
        cached = {r["venue"]: r for _, r in prev.iterrows()}
        print(f"  cache: {len(cached)} venues already resolved")

    todo = [v for v in wanted if v not in cached]
    print(f"  resolving {len(todo)} new venues (of {len(wanted)} wanted, {len(counts)} total)")

    rows = [dict(r) for r in cached.values()]
    pending = []
    for i, venue in enumerate(todo, 1):
        place, country = venue_to_place(venue)
        rec = {"venue": venue, "place": place, "country": country,
               "lat": None, "lon": None, "elevation_m": None, "rows": counts[venue]}
        if place:
            try:
                hit = geocode(place, country)
                if hit:
                    rec["lat"], rec["lon"] = hit["latitude"], hit["longitude"]
            except Exception as e:
                print(f"    {venue[:44]:46} geocode ERROR {str(e)[:50]}")
            time.sleep(0.25)
        if rec["lat"] is not None:
            pending.append(rec)
        rows.append(rec)
        if i % 50 == 0:
            print(f"    {i}/{len(todo)} geocoded")

    if pending:
        print(f"  fetching elevation for {len(pending)} coordinates")
        elevs = fetch_elevations([(r["lat"], r["lon"]) for r in pending])
        for rec, e in zip(pending, elevs):
            rec["elevation_m"] = e

    out = pd.DataFrame(rows).drop_duplicates(subset=["venue"], keep="last")
    out.to_csv(CACHE_PATH, index=False)

    resolved = out["elevation_m"].notna().sum()
    covered = out.loc[out["elevation_m"].notna(), "rows"].sum()
    print(f"\n  {resolved}/{len(out)} venues have an elevation")
    print(f"  covering {covered}/{total_rows} rows = {100 * covered / total_rows:.1f}%")
    print(f"  -> {CACHE_PATH}")


if __name__ == "__main__":
    main()
