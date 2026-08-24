"""
venue_weather.py -- real historical weather for every (venue, date) that
actually appears in data/raw/*.csv, cached to data/venue_weather.csv.

Needs data/venues_geo.csv first (src/venue_geo.py) for coordinates.

Why temperature and humidity and not wind: the scraped rows ALREADY carry
the official trackside wind reading (the WIND column, ±m/s along the track,
which is what the rules actually care about and what apply_wind_adjustment()
already uses). A city-level 10m wind speed from a weather model is a worse
measurement of the same thing, so it is deliberately not fetched. Air
temperature and humidity are genuinely absent from the scraped data and do
have real, opposite-signed effects: warm thin air is quick for sprints and
punishing over 5000m.

Only the dates that appear in the data are kept. One API call per venue
covers that venue's whole span, so this is ~420 requests, not ~3800.

Source: Open-Meteo historical archive (ERA5 reanalysis), free and key-less.

Usage:
    python src/venue_weather.py
"""
import glob
import io
import os
import sys
import time

import pandas as pd
import requests

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
GEO_PATH = os.path.join(BASE_DIR, "data", "venues_geo.csv")
OUT_PATH = os.path.join(BASE_DIR, "data", "venue_weather.csv")

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
DAILY = "temperature_2m_max,temperature_2m_mean,relative_humidity_2m_mean"


def needed_pairs():
    """(venue, date) pairs present in the scraped data, as real dates."""
    frames = []
    for f in glob.glob(os.path.join(RAW_DIR, "*.csv")):
        try:
            d = pd.read_csv(f, usecols=["Venue", "Date"])
        except Exception:
            continue
        frames.append(d.dropna(subset=["Venue", "Date"]))
    if not frames:
        return pd.DataFrame(columns=["Venue", "date"])
    all_rows = pd.concat(frames, ignore_index=True)
    all_rows["date"] = pd.to_datetime(all_rows["Date"], format="%d %b %Y", errors="coerce")
    all_rows = all_rows.dropna(subset=["date"])
    return all_rows[["Venue", "date"]].drop_duplicates()


def _flush(existing, out_rows):
    """Write the cache out, deduped. Called periodically as well as at the
    end so an interrupted run still leaves its progress on disk."""
    combined = pd.concat(existing + [pd.DataFrame(out_rows)], ignore_index=True)
    combined = combined.drop_duplicates(subset=["venue", "date"], keep="last")
    combined = combined.dropna(subset=["temp_mean_c"])
    combined.to_csv(OUT_PATH, index=False)
    return combined


def main():
    if not os.path.exists(GEO_PATH):
        sys.exit("data/venues_geo.csv missing -- run: python src/venue_geo.py")
    geo = pd.read_csv(GEO_PATH).dropna(subset=["lat", "lon"])
    coords = {r["venue"]: (r["lat"], r["lon"]) for _, r in geo.iterrows()}

    pairs = needed_pairs()
    pairs = pairs[pairs["Venue"].astype(str).isin(coords)]
    print(f"  {len(pairs)} (venue, date) pairs at {pairs['Venue'].nunique()} geocoded venues")

    done = set()
    existing = []
    if os.path.exists(OUT_PATH):
        prev = pd.read_csv(OUT_PATH)
        existing.append(prev)
        done = set(zip(prev["venue"].astype(str), prev["date"].astype(str)))
        print(f"  cache: {len(done)} pairs already fetched")

    out_rows = []
    venues = sorted(pairs["Venue"].astype(str).unique())
    for i, venue in enumerate(venues, 1):
        want = pairs[pairs["Venue"].astype(str) == venue]["date"]
        want_str = {d.strftime("%Y-%m-%d") for d in want}
        if want_str <= {d for v, d in done if v == venue}:
            continue
        lat, lon = coords[venue]
        params = {
            "latitude": lat, "longitude": lon,
            "start_date": want.min().strftime("%Y-%m-%d"),
            "end_date": want.max().strftime("%Y-%m-%d"),
            "daily": DAILY, "timezone": "UTC",
        }
        try:
            # Open-Meteo's archive endpoint rate-limits harder than the
            # geocoder; a flat sleep hit 429s partway through the first run.
            # Back off and retry rather than silently losing those venues --
            # a gap here becomes a missing feature value later.
            for attempt in range(5):
                r = requests.get(ARCHIVE_URL, params=params, timeout=45)
                if r.status_code != 429:
                    break
                wait = 5 * (attempt + 1)
                print(f"    rate-limited, waiting {wait}s")
                time.sleep(wait)
            r.raise_for_status()
            daily = r.json().get("daily") or {}
            for d, tmax, tmean, rh in zip(daily.get("time", []),
                                          daily.get("temperature_2m_max", []),
                                          daily.get("temperature_2m_mean", []),
                                          daily.get("relative_humidity_2m_mean", [])):
                if d in want_str:
                    out_rows.append({"venue": venue, "date": d, "temp_max_c": tmax,
                                     "temp_mean_c": tmean, "humidity_pct": rh})
        except Exception as e:
            print(f"    {venue[:44]:46} ERROR {str(e)[:60]}")
        time.sleep(1.1)
        if i % 25 == 0:
            # Flush as we go. Open-Meteo's free daily quota is real and this
            # script WILL get cut off against it; writing only at the end
            # threw away hundreds of successful calls the first time that
            # happened. Every partial run now advances the cache, so
            # re-running after the quota resets genuinely resumes.
            _flush(existing, out_rows)
            print(f"    {i}/{len(venues)} venues (cache flushed)")

    combined = _flush(existing, out_rows)
    print(f"\n  {len(combined)} (venue, date) rows with real weather -> {OUT_PATH}")
    if len(combined):
        print(f"  temp_mean_c range: {combined['temp_mean_c'].min():.1f}..{combined['temp_mean_c'].max():.1f} C")


if __name__ == "__main__":
    main()
