"""
refresh_days_since_last.py -- recomputes just the days_since_last column in
outputs/predictions_latest.csv from the real per-meeting current-season data
(data/raw/{discipline}_{MEETS_YEAR}_meetings.csv, see
current_season_scraper.py), without re-running the full run.py pipeline.

Why this exists: run.py's build_2026_features() previously computed
days_since_last from the live toplist snapshot (data/raw/{discipline}_2026.csv),
which has exactly ONE row per athlete -- their season's single best mark.
That silently meant "days since their BEST mark", not their most recent
race, and was also just stale between full run.py refreshes. Confirmed live,
2026-08-24: Rai Benjamin's toplist row was dated 18 Jul even though he'd
actually raced again on 23 Aug. run.py's build_2026_features() has been
fixed at the source for future runs (prefers the meetings file when
present); this script re-derives the same corrected value for the CURRENT
outputs/predictions_latest.csv without needing a full ~1hr live rescrape.

Deliberately does NOT touch win_probability or any other model-fed column
-- days_since_last is also a trained feature, so a fully consistent refresh
of the model's predictions still needs a real run.py run. This just fixes
the one directly-displayed stat the profile page shows, using real,
already-scraped data (not fabricated), same honesty standard as everywhere
else in this project. Safe to run any time; a no-op for athletes with no
current-season meeting file, or none of their own rows in it.

Usage:
    python src/refresh_days_since_last.py
"""
import os
import sys
from datetime import date

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from api import DISC_LABELS, MEETS_YEAR, RAW_DIR, OUTPUTS_DIR  # noqa: E402


def most_recent_date(disc_key, athlete_name, today):
    path = os.path.join(RAW_DIR, f"{disc_key}_{MEETS_YEAR}_meetings.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    mine = df[df["Competitor"].str.lower() == athlete_name.lower()]
    if mine.empty:
        return None
    dates = pd.to_datetime(mine["Date"], format="%d %b %Y", errors="coerce").dropna()
    if dates.empty:
        return None
    return (today - dates.max()).days


if __name__ == "__main__":
    path = os.path.join(OUTPUTS_DIR, "predictions_latest.csv")
    df = pd.read_csv(path)
    label_to_key = {v: k for k, v in DISC_LABELS.items()}
    today = pd.Timestamp(date.today())

    updated = 0
    for idx, row in df.iterrows():
        disc_key = label_to_key.get(row["discipline"])
        if not disc_key:
            continue
        days = most_recent_date(disc_key, row["athlete_name"], today)
        if days is not None and days != row.get("days_since_last"):
            df.at[idx, "days_since_last"] = days
            updated += 1

    df.to_csv(path, index=False)
    print(f"Updated days_since_last for {updated} athletes using real current-season meeting dates -> {path}")
