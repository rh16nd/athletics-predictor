"""
season_activity.py -- how many times an athlete actually contested a
discipline in a season, as far as anything on disk can show.

WHY THIS EXISTS
The site used to show "DL meetings" next to each athlete's rating, and that
number is true but answers a narrower question than readers take it for. The
Diamond League contests each discipline at only a handful of its meetings --
the men's 400m hurdles at 5 of the 14 in 2026 -- so a 0 there means "did not
run a Diamond League 400mH", not "did not run". Across the 32 model top-20s,
28% of athletes show 0.

Rai Benjamin was the case that made it obvious: 46.67 at Lausanne on 21 August
2026, third in the world on points, and 0 Diamond League meetings, because the
men's 400mH was not on the Lausanne Diamond League programme (the women's was).
Both facts are correct and together they are misleading.

WHAT THIS COUNTS
The union of every source that carries a date and a venue for a performance:

  data/raw/<disc>_<year>.csv          the season toplist -- one row per
                                      athlete, their season best, with the
                                      date and venue it was set at
  data/raw/<disc>_<year>_meetings.csv the per-meeting Diamond League log
  data/worldwide/<disc>.csv           the wider race log, which carries the
                                      meetings the other two do not (European
                                      Championships, CAC Games, ISTAF...)

Counted as distinct (date, venue) pairs, so the same race appearing in two
sources counts once.

THIS IS A FLOOR, NOT A CENSUS
None of the three covers every meeting on earth, and the worldwide log stops
at whatever date it was last scraped. A count of 1 means "we can see one race",
not "they ran once". The UI must say so -- the number is there to stop a rating
built on a single visible race from looking like a rating built on a season,
which is a real thing it can do, not to claim a complete record.
"""
import os

import pandas as pd

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
WORLDWIDE_DIR = os.path.join(BASE_DIR, "data", "worldwide")


def _pairs(df, name_col="Competitor", date_col="Date", venue_col="Venue"):
    """(name, date, venue) for every row that has all three. Anything missing a
    date is dropped rather than counted, since it cannot be de-duplicated
    against the same race seen in another file."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["name", "date", "venue"])
    for col in (name_col, date_col, venue_col):
        if col not in df.columns:
            return pd.DataFrame(columns=["name", "date", "venue"])
    # fillna BEFORE the string ops, not after. Under pandas' string dtype a
    # missing value stays pd.NA through .astype(str).str.strip(), and then
    # every comparison against it evaluates falsey -- so a filter written as
    # `!= "nan"` silently keeps the row it was written to drop. Found by the
    # test below, which counted 2 races for an athlete with one date.
    out = pd.DataFrame({
        "name": df[name_col].astype(str).fillna("").str.upper().str.strip(),
        "date": df[date_col].astype(str).fillna("").str.strip(),
        "venue": df[venue_col].astype(str).fillna("").str.strip(),
    })
    blank = {"", "nan", "nat", "none", "<na>"}
    return out[~out["date"].str.lower().isin(blank)]


def races_on_record(discipline, year):
    """{UPPERCASED NAME: distinct races we can see} for one discipline-season.

    Names are upper-cased rather than run through train_model.normalize_name:
    all three sources here are World Athletics' own exports, which already
    agree on spelling and diacritics. The label files need that stricter fold
    because they join scraped results to a different scrape."""
    frames = []

    toplist = os.path.join(RAW_DIR, f"{discipline}_{year}.csv")
    if os.path.exists(toplist):
        frames.append(_pairs(pd.read_csv(toplist, low_memory=False)))

    meetings = os.path.join(RAW_DIR, f"{discipline}_{year}_meetings.csv")
    if os.path.exists(meetings):
        frames.append(_pairs(pd.read_csv(meetings, low_memory=False)))

    worldwide = os.path.join(WORLDWIDE_DIR, f"{discipline}.csv")
    if os.path.exists(worldwide):
        wide = pd.read_csv(worldwide, low_memory=False)
        if "year" in wide.columns:
            wide = wide[wide["year"] == year]
        frames.append(_pairs(wide))

    if not frames:
        return {}
    allrows = pd.concat(frames, ignore_index=True).drop_duplicates()
    return allrows.groupby("name").size().to_dict()


if __name__ == "__main__":
    import sys

    disc = sys.argv[1] if len(sys.argv) > 1 else "men_400h"
    year = int(sys.argv[2]) if len(sys.argv) > 2 else 2026
    counts = races_on_record(disc, year)
    print(f"=== {disc} {year}: races on record ===")
    print(f"  {len(counts)} athletes")
    ranked = sorted(counts.items(), key=lambda kv: -kv[1])
    for name, n in ranked[:10]:
        print(f"    {name:<32} {n}")
    only_one = sum(1 for n in counts.values() if n == 1)
    print(f"  seen exactly once: {only_one} of {len(counts)}")
