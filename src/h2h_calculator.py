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

def calculate_h2h(df):
    records = []
    for disc in df["discipline"].unique():
        disc_df = df[df["discipline"] == disc].copy()
        disc_df = disc_df.dropna(subset=["place"])
        meets = disc_df["meet"].unique()

        for meet in meets:
            meet_df = disc_df[disc_df["meet"] == meet].copy()
            athletes = meet_df["athlete"].tolist()
            places = dict(zip(meet_df["athlete"], meet_df["place"]))

            for i, a in enumerate(athletes):
                for b in athletes[i+1:]:
                    if a == b:
                        continue
                    place_a = places.get(a)
                    place_b = places.get(b)
                    if place_a is None or place_b is None:
                        continue
                    a_wins = 1 if place_a < place_b else 0
                    records.append({"discipline": disc, "athlete_a": a, "athlete_b": b, "a_wins": a_wins, "total": 1})
                    records.append({"discipline": disc, "athlete_a": b, "athlete_b": a, "a_wins": 1 - a_wins, "total": 1})

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