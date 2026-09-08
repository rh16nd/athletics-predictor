"""
final_labels.py -- the pooled ground truth: every final we have a podium for,
not just Diamond League Finals.

WHY THIS EXISTS
Until now the model was trained on one question -- "who podiums at a Diamond
League Final?" -- because that was the only label on disk in usable form: 213
finals, 651 podium rows, 2018-2025. Everything downstream inherited it,
including the model's habit of rating Diamond League regulars above faster
athletes who skipped the circuit. That is not a bug to tune out; it is the
literal thing 213 Diamond League labels teach.

`major_meets_scraper.py` has been pulling other finals all along for
feature-building, and nobody had ever used them as labels. Pooled with the
Diamond League Finals they come to 980 finals across the same seven years and
all 32 disciplines -- 4.6x the training signal.

WHAT COUNTS AS A LABEL
One row per athlete who finished in the top three of a final we have complete
results for, with the competition it was, the tier it sits in, and -- the part
that matters -- THE DATE IT WAS RUN.

THE DATE IS THE WHOLE POINT
train_model.py used to compute every athlete's features as of a hard-coded
1 September, which is fine when every label is a September Diamond League
Final and catastrophic the moment it is not. A World Championships final in
July, scored against a season best that includes August, is not a prediction:
it is the answer read backwards out of the future. Every row here carries the
real date of that final so features can be cut at it.

Championship dates are exact -- all 819 major-meet finals have their three
podium rows on a single date, checked. Diamond League Final dates are not on
disk at all (dl_final_results.csv has no date column), so they are taken as
the date of the LAST Diamond League meeting we hold for that year. That is the
Final itself in 2021-2025 and lands two weeks early in 2018-2019, when the
Final was split across two meetings. Early is the safe direction: a cut-off
before the event loses information, a cut-off after it leaks the answer.

Usage:
    python src/final_labels.py            # writes data/labels/finals.csv
"""
import glob
import os
import re
import unicodedata

import pandas as pd

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DL_RESULTS = os.path.join(BASE_DIR, "data", "dl_final_results.csv")
OUT_PATH = os.path.join(BASE_DIR, "data", "labels", "finals.csv")
FIELDS_PATH = os.path.join(BASE_DIR, "data", "labels", "final_fields.csv")

# Files in data/raw that are per-season scrapes rather than the pooled
# per-discipline history the labels come from.
SEASON_FILE = re.compile(r"_20\d\d(_meetings)?\.csv$")

# How much a final is worth as evidence about a championship. Kept as a column
# rather than baked into the pool so accuracy can be reported per tier -- the
# risk of pooling this widely is that "podium in a final" stops meaning one
# thing, and the only way to see that happening is to measure the tiers apart.
#
# The names changed when the IAAF became World Athletics, and the pre-2018
# seasons added on 2026-09-08 use the older ones. "IAAF World Championships"
# matches neither of the two spellings this list originally carried, and
# "Barcelona European Championships" does not match "european athletics
# championships" -- so the five pre-2018 Worlds and three of the four pre-2018
# Euros would have been silently filed as `tour` and then excluded by the
# --tiers argument the model trains with. Making "Athletics" optional in both
# patterns covers every real name across both eras; checked against all 20
# championship meetings the calendar returns for 2009-2025.
#
# Widening a pattern is how the old NAME_EXCLUDE list went wrong, so note the
# difference: nothing decides here whether a meeting BELONGS in the training
# set. major_meets_scraper has already made that call, and these patterns only
# sort what it returned. A too-wide pattern here mislabels a tier; it cannot
# admit a road race.
TIERS = [
    ("global", re.compile(r"olympic|world (athletics )?championships", re.I)),
    ("continental", re.compile(r"european (athletics )?championships|continental cup", re.I)),
    ("dl_final", re.compile(r"^diamond league final$", re.I)),
    # Everything else major_meets_scraper returns: Continental Tour Gold and
    # the bigger invitationals (Ostrava, Paavo Nurmi, Kip Keino, FBK Games).
    ("tour", re.compile(r".", re.S)),
]


def tier_of(competition):
    for name, pattern in TIERS:
        if pattern.search(str(competition)):
            return name
    return "tour"


def normalize_name(name):
    """MUST stay byte-identical to train_model.normalize_name -- diacritics
    stripped, upper-cased, trimmed.

    Not a stylistic copy: this is the key the labels join to the feature rows
    on, and a silent mismatch does not raise, it just matches nothing and
    labels every athlete a non-medallist. The first version of this file
    lower-cased instead and produced a training set with 2,940 podiums in it
    and zero positive labels. tests/test_final_labels.py pins the two
    together."""
    if not isinstance(name, str):
        return name
    nfkd = unicodedata.normalize("NFKD", name)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).upper().strip()


def _discipline_files():
    for path in sorted(glob.glob(os.path.join(RAW_DIR, "*.csv"))):
        base = os.path.basename(path)
        if SEASON_FILE.search(base):
            continue
        yield base[: -len(".csv")], path


def major_finals():
    """Podiums from every non-Diamond-League final on disk.

    A final is a (discipline, year, competition) group whose podium is
    COMPLETE -- all of first, second and third. An incomplete one is dropped
    rather than padded: a label set that says "these two medalled and we don't
    know the third" teaches the third was not on the podium, which is a
    fabricated negative."""
    rows = []
    for key, path in _discipline_files():
        try:
            df = pd.read_csv(path, low_memory=False)
        except (OSError, pd.errors.ParserError):
            continue
        if "source" not in df.columns:
            continue
        df = df[df["source"] == "major_meet"].copy()
        if df.empty:
            continue
        df["place"] = pd.to_numeric(df["Pos"], errors="coerce")
        df = df[df["place"].between(1, 3)]
        if df.empty:
            continue
        df["discipline"] = key
        df["competition"] = df["Venue"].astype(str)
        df["date"] = pd.to_datetime(df["Date"], format="%d %b %Y", errors="coerce")
        rows.append(df[["discipline", "year", "competition", "date", "Competitor", "place", "Mark"]])

    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out = out.rename(columns={"Competitor": "athlete_name", "Mark": "mark"})
    out = out.dropna(subset=["date"])

    # Complete podiums only, and one athlete per place: a duplicated row would
    # otherwise pass the count test with a hole in it.
    out = out.drop_duplicates(["discipline", "year", "competition", "place"])
    sizes = out.groupby(["discipline", "year", "competition"])["place"].transform("size")
    return out[sizes == 3].copy()


def major_fields():
    """Everyone who CONTESTED each non-Diamond-League final, not just the three
    who medalled.

    The podium is the label; this is the candidate pool. Without it the second
    metric the backtest reports -- "find the podium among the athletes who were
    actually there" -- has no way to know who was there for a World
    Championships, and would fall back to the Diamond League field, which is a
    different set of people at a different meeting on a different day."""
    rows = []
    for key, path in _discipline_files():
        try:
            df = pd.read_csv(path, low_memory=False)
        except (OSError, pd.errors.ParserError):
            continue
        if "source" not in df.columns:
            continue
        df = df[df["source"] == "major_meet"].copy()
        if df.empty:
            continue
        df["place"] = pd.to_numeric(df["Pos"], errors="coerce")
        df = df.dropna(subset=["place"])
        if df.empty:
            continue
        df["discipline"] = key
        df["competition"] = df["Venue"].astype(str)
        rows.append(df[["discipline", "year", "competition", "Competitor"]])
    if not rows:
        return pd.DataFrame(columns=["discipline", "year", "competition", "name_key"])
    out = pd.concat(rows, ignore_index=True).rename(columns={"Competitor": "athlete_name"})
    out["name_key"] = out["athlete_name"].apply(normalize_name)
    return out[["discipline", "year", "competition", "name_key"]].drop_duplicates()


def dl_fields():
    """The same for Diamond League Finals, where dl_final_results.csv already
    holds the whole field including the athletes who finished outside three."""
    df = pd.read_csv(DL_RESULTS)
    df["competition"] = "Diamond League Final"
    df["name_key"] = df["athlete_name"].apply(normalize_name)
    return df[["discipline", "year", "competition", "name_key"]].drop_duplicates()


def build_fields():
    return pd.concat([dl_fields(), major_fields()], ignore_index=True).drop_duplicates()


def dl_final_dates():
    """{year: date} for the Diamond League Final, taken as the last DL meeting
    held that year. See the note at the top on why early is the safe error."""
    dates = {}
    for _, path in _discipline_files():
        try:
            df = pd.read_csv(path, low_memory=False)
        except (OSError, pd.errors.ParserError):
            continue
        if "source" not in df.columns:
            continue
        df = df[df["source"] == "dl_meeting"]
        if df.empty:
            continue
        d = pd.to_datetime(df["Date"], format="%d %b %Y", errors="coerce")
        for year, when in pd.DataFrame({"year": df["year"].values, "d": d.values}).dropna().groupby("year")["d"].max().items():
            year = int(year)
            if year not in dates or when > dates[year]:
                dates[year] = when
    return dates


def dl_finals():
    """Podiums from the Diamond League Finals, the labels the model has always
    had, in the same shape as the rest."""
    df = pd.read_csv(DL_RESULTS)
    df["place"] = pd.to_numeric(df["place"], errors="coerce")
    df = df[df["place"].between(1, 3)].copy()
    df["competition"] = "Diamond League Final"
    dates = dl_final_dates()
    df["date"] = df["year"].map(lambda y: dates.get(int(y)))
    df = df.dropna(subset=["date"])
    df = df.drop_duplicates(["discipline", "year", "competition", "place"])
    sizes = df.groupby(["discipline", "year", "competition"])["place"].transform("size")
    return df[sizes == 3][["discipline", "year", "competition", "date", "athlete_name", "place", "mark"]]


def build():
    pool = pd.concat([dl_finals(), major_finals()], ignore_index=True)
    pool["tier"] = pool["competition"].apply(tier_of)
    pool["place"] = pool["place"].astype(int)
    pool["year"] = pool["year"].astype(int)
    pool["name_key"] = pool["athlete_name"].apply(normalize_name)
    pool = pool.sort_values(["year", "discipline", "competition", "place"])
    return pool.reset_index(drop=True)


if __name__ == "__main__":
    pool = build()
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    pool.to_csv(OUT_PATH, index=False)

    finals = pool.groupby(["discipline", "year", "competition"]).size()
    print("=== Pooled final labels ===")
    print(f"  {len(finals)} finals, {len(pool)} podium rows")
    print(f"  {pool['discipline'].nunique()} disciplines, "
          f"{pool['year'].min()}-{pool['year'].max()}")
    print("\n  by tier:")
    by_tier = pool.groupby("tier").agg(
        podium_rows=("place", "size"),
        finals=("competition", lambda s: len(s) // 3),
    )
    for tier, r in by_tier.iterrows():
        print(f"    {tier:<12} {r['finals']:>4} finals  {r['podium_rows']:>5} rows")
    fields = build_fields()
    fields.to_csv(FIELDS_PATH, index=False)
    print(f"\n  {len(fields)} start-list rows (who contested each final, not just who medalled)")
    print(f"  -> {os.path.abspath(OUT_PATH)}")
    print(f"  -> {os.path.abspath(FIELDS_PATH)}")
