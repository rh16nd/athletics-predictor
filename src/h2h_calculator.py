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

def one_row_per_athlete(meet_df):
    """One result per athlete per meet: their BEST (lowest) place.

    A meet page carries heats, semis and the final as separate rows, and the
    scraper does not record which round a row came from (see
    h2h_scraper.parse_results_table). Before this, `dict(zip(...))` silently
    kept whichever row happened to be LAST, so an athlete's heat number
    could stand in for their final placing -- at the 2023 World
    Championships Jakob Ingebrigtsen's rows are [1, 13] and Yared Nuguse's
    are [4, 1, 5], and the comparison actually made was 13 against 5, handing
    Nuguse a championship Ingebrigtsen won.

    Best-place is the right collapse while the round is unknown: for a
    progression-based meet it is the furthest round the athlete reached and
    their standing in it, which is what "who beat whom at this meeting"
    means. Capturing the round properly in the scraper would be better still
    and needs a re-scrape."""
    return meet_df.sort_values("place").drop_duplicates(subset=["athlete"], keep="first")


def calculate_h2h(df):
    records = []
    for disc in df["discipline"].unique():
        disc_df = df[df["discipline"] == disc].copy()
        disc_df = disc_df.dropna(subset=["place"])

        for meet in disc_df["meet"].unique():
            meet_df = one_row_per_athlete(disc_df[disc_df["meet"] == meet])
            # Iterating this de-duplicated frame is also what stops one
            # meeting being counted many times: the old loop walked the ROW
            # list, so a 2-row athlete against a 3-row athlete produced up
            # to 6 "meetings" -- the same comparison repeated, which is why
            # 64% of all pairs with >=5 meetings came out as perfect sweeps.
            rows = list(zip(meet_df["athlete"], meet_df["place"]))

            for i, (a, place_a) in enumerate(rows):
                for b, place_b in rows[i + 1:]:
                    if a == b or place_a == place_b:
                        # Equal places can only mean the two never actually
                        # met (same position in different heats). Counting
                        # it either way would invent a result.
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