"""
h2h_calculator.py — Calculates head-to-head win rates from meet results
Output: data/h2h/h2h_rates.csv
"""
import pandas as pd
import os
import sys
import io
# Guarded: several modules in src/ do this, and each wraps the SAME
# sys.stdout.buffer. With two of them imported into one process the first
# wrapper to be garbage-collected closes the buffer under the second, and
# every later write dies with "I/O operation on closed file" -- which took
# down the whole pytest run on 2026-08-25. After the first wrap the
# encoding is already utf-8, so this becomes a no-op.
if not (sys.stdout.encoding or "").lower().startswith("utf"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

H2H_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "h2h")
INPUT = os.path.join(H2H_DIR, "meet_results.csv")
OUTPUT = os.path.join(H2H_DIR, "h2h_rates.csv")

# The columns h2h_scraper.py writes to say WHICH RACE a row came from.
# Their absence is meaningful, not an error: a CSV scraped before they
# existed gets the older, deliberately conservative collapse below.
RACE_COLUMNS = ["race", "heat"]


def one_row_per_athlete(race_df):
    """One result per athlete per race: their BEST (lowest) place.

    With `race`/`heat` present this is only a guard against an athlete being
    listed twice in one table. Without them it is load-bearing, and it is
    why it still exists: a meet page carries heats, semis and the final as
    separate rows, and before the round was recorded `dict(zip(...))`
    silently kept whichever row happened to be LAST, so an athlete's heat
    number could stand in for their final placing -- at the 2023 World
    Championships Jakob Ingebrigtsen's rows are [1, 13] and Yared Nuguse's
    are [4, 1, 5], and the comparison actually made was 13 against 5,
    handing Nuguse a championship Ingebrigtsen won.

    Best-place is the right collapse while the round is unknown: for a
    progression-based meet it is the furthest round the athlete reached and
    their standing in it, which is what "who beat whom at this meeting"
    means."""
    return race_df.sort_values("place").drop_duplicates(subset=["athlete"], keep="first")


def calculate_h2h(df):
    """Head-to-head records, counted one race at a time.

    A race -- not a meeting -- is the unit, because a meeting is not a
    contest. Two athletes are only compared when they were on the same
    start list, which needs three things from the scraper: the race label
    (a heat and the final are different races), the heat number within it
    (a championship heats table ranks ALL heats together by time, so rank 1
    in heat 1 "beating" rank 4 in heat 4 is a time comparison, not a
    head-to-head), and the podium places that Wikipedia renders as medal
    icons with no text.

    A pair that met in both a heat and the final counts twice. They did race
    each other twice, and the alternative -- keeping only the deepest round
    -- discards real results to no end.

    When those columns are absent the whole meeting is treated as one group
    and collapsed to each athlete's best place, which is the older, lossier
    behaviour; see `one_row_per_athlete`."""
    rounds_known = all(c in df.columns for c in RACE_COLUMNS)
    group_cols = ["meet"] + (RACE_COLUMNS if rounds_known else [])

    records = []
    for disc in df["discipline"].unique():
        disc_df = df[df["discipline"] == disc].copy()
        disc_df = disc_df.dropna(subset=["place"])
        for col in group_cols:
            disc_df[col] = disc_df[col].fillna("")

        for _, race_df in disc_df.groupby(group_cols, sort=False):
            race_df = one_row_per_athlete(race_df)
            # Iterating this de-duplicated frame is also what stops one race
            # being counted many times: the old loop walked the ROW list, so
            # a 2-row athlete against a 3-row athlete produced up to 6
            # "meetings" -- the same comparison repeated, which is why 64% of
            # all pairs with >=5 meetings came out as perfect sweeps.
            rows = list(zip(race_df["athlete"], race_df["place"]))

            for i, (a, place_a) in enumerate(rows):
                for b, place_b in rows[i + 1:]:
                    if a == b or place_a == place_b:
                        # Within one race, equal places are a dead heat with
                        # no winner. Across an unlabelled meeting they can
                        # only mean the two never actually met (the same
                        # position in different heats). Scoring either would
                        # invent a result.
                        continue
                    a_wins = 1 if place_a < place_b else 0
                    records.append({"discipline": disc, "athlete_a": a, "athlete_b": b,
                                    "a_wins": a_wins, "total": 1})
                    records.append({"discipline": disc, "athlete_a": b, "athlete_b": a,
                                    "a_wins": 1 - a_wins, "total": 1})

    if not records:
        return pd.DataFrame()

    h2h = pd.DataFrame(records)
    h2h = h2h.groupby(["discipline", "athlete_a", "athlete_b"]).agg(
        wins=("a_wins", "sum"),
        meetings=("total", "sum")
    ).reset_index()
    h2h["win_rate"] = h2h["wins"] / h2h["meetings"]
    return h2h

if __name__ == "__main__":
    print("Loading meet results...")
    df = pd.read_csv(INPUT)
    print(f"  {len(df)} rows, {df['discipline'].nunique()} disciplines")

    print("Calculating head-to-head records...")
    h2h = calculate_h2h(df)
    print(f"  {len(h2h)} head-to-head matchup records")

    h2h.to_csv(OUTPUT, index=False)
    print(f"Saved to {OUTPUT}")

    print("\nSample h2h records (min 3 meetings):")
    sample = h2h[h2h["meetings"] >= 3].sort_values("meetings", ascending=False).head(10)
    for _, row in sample.iterrows():
        print(f"  {row['athlete_a']} vs {row['athlete_b']} ({row['discipline']}): {row['wins']}/{row['meetings']} = {row['win_rate']:.0%}")