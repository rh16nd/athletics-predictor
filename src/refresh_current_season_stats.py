"""
refresh_current_season_stats.py -- recomputes the profile-page stats that
depend on real current-season race data (days_since_last, meets_count) in
outputs/predictions_latest.csv, using data/raw/{discipline}_{MEETS_YEAR}
_meetings.csv (see current_season_scraper.py), without re-running the full
run.py pipeline.

Why this exists: run.py's build_2026_features() previously computed both
these stats from the live toplist snapshot (data/raw/{discipline}_2026.csv),
which has exactly ONE row per athlete -- their season's single best mark,
not a results log:
  - days_since_last silently meant "days since their BEST mark", not their
    most recent race. Confirmed live, 2026-08-24: Rai Benjamin's toplist
    row was dated 18 Jul even though he'd actually raced again on 23 Aug.
  - meets_count was structurally always 1 for every single athlete -- the
    toplist file can never have more than one row per athlete, so counting
    rows always counts to 1 regardless of how many real meetings they
    attended. Confirmed live: Benjamin's chart shows 4 real 2026 races,
    the stat tile showed "1".

run.py's build_2026_features() has been fixed at the source for future
runs (prefers the meetings file for both when present); this script
re-derives the same corrected values for the CURRENT
outputs/predictions_latest.csv without needing a full ~1hr live rescrape.

Deliberately does NOT touch win_probability or any other model-fed column
-- both stats are also trained features, so a fully consistent refresh of
the model's predictions still needs a real run.py run. This just fixes the
directly-displayed stats the profile page shows, using real, already-
scraped data (not fabricated), same honesty standard as everywhere else in
this project. Safe to run any time; a no-op for athletes with no
current-season meeting file, or none of their own rows in it.

Usage:
    python src/refresh_current_season_stats.py
"""
import os
import sys
from datetime import date

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from api import DISC_LABELS, MEETS_YEAR, RAW_DIR, OUTPUTS_DIR  # noqa: E402


def real_current_season_rows(disc_key, athlete_name):
    """Returns the athlete's real current-season rows -- possibly an empty
    DataFrame, which is a real, verified answer (zero confirmed DL-circuit
    meetings so far this season), not the same as having no data source at
    all. Only returns None when the meetings file itself doesn't exist yet
    for this discipline."""
    path = os.path.join(RAW_DIR, f"{disc_key}_{MEETS_YEAR}_meetings.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    return df[df["Competitor"].str.lower() == athlete_name.lower()]


if __name__ == "__main__":
    path = os.path.join(OUTPUTS_DIR, "predictions_latest.csv")
    df = pd.read_csv(path)
    label_to_key = {v: k for k, v in DISC_LABELS.items()}
    today = pd.Timestamp(date.today())

    days_updated = 0
    meets_updated = 0
    for idx, row in df.iterrows():
        disc_key = label_to_key.get(row["discipline"])
        if not disc_key:
            continue
        mine = real_current_season_rows(disc_key, row["athlete_name"])
        if mine is None:
            continue

        # A real, verified meeting count (possibly 0) replaces whatever was
        # there before -- including the old toplist-based "always 1", which
        # was never a real measurement to begin with.
        real_meets = len(mine)
        if real_meets != row.get("meets_count"):
            df.at[idx, "meets_count"] = real_meets
            meets_updated += 1

        if not mine.empty:
            dates = pd.to_datetime(mine["Date"], format="%d %b %Y", errors="coerce").dropna()
            if not dates.empty:
                days = (today - dates.max()).days
                if days != row.get("days_since_last"):
                    df.at[idx, "days_since_last"] = days
                    days_updated += 1

    df.to_csv(path, index=False)
    print(f"Updated days_since_last for {days_updated} athletes, meets_count for {meets_updated} -> {path}")
