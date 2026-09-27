"""Is third place predictable at all, from anything known before the race?

The model names 86.2% of winners, 55.0% of second places and 39.0% of thirds.
All the accuracy that is missing sits in third, so the question worth asking
before trying to win it back is whether it is there to be won.

This measures the ceiling rather than a model. For every final it asks a
simple, model-free question: does the athlete who actually finished third look
any different, beforehand, from the athletes who finished fourth and below?

The measure is the pairwise win rate, which is the area under the ROC curve
written out longhand. Take the athlete who finished third and every athlete in
the same final who finished below the podium, and count how often the third
placer had the higher pre-race number. 50% means the feature cannot tell them
apart at all. The same figure for first place is the calibration: whatever it
reads is what a genuinely predictable placing looks like in this data.

Everything is computed inside a final, so field strength, era and discipline
cancel out and nothing has to be normalised.

Usage:
    python src/third_place_ceiling.py
"""
import json
import os

import pandas as pd

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
FINALS_PATH = os.path.join(BASE_DIR, "data", "field", "finals.csv")
OUT_PATH = os.path.join(BASE_DIR, "outputs", "third_place_ceiling.json")

FEATURES = ["sb_score", "form_score", "recent_score"]
PLACES = [1, 2, 3]
# Comparison bands, from the whole field below the podium down to the single
# athlete who finished immediately behind.
def bands_for(place):
    """Comparison bands relative to the placing: the athlete immediately behind,
    the three behind, and everyone who finished off the podium."""
    return {"vsNextOne": (place + 1, place + 1),
            "vsNextThree": (place + 1, place + 3),
            "vsOffPodium": (4, 99)}


def pairwise_win_rate(finals, place, feature, lo=4, hi=99):
    """How often the athlete who finished `place` had a higher `feature` than an
    athlete in the same final who finished between `lo` and `hi`. Ties count a
    half, as they do in an ROC.

    The band matters more than the headline. Separating third from the back of
    the field is easy and says little; separating third from fourth is the
    decision the model actually has to get right."""
    wins = ties = total = 0
    for _, final in finals:
        rows = final.dropna(subset=[feature, "place"])
        target = rows[rows["place"] == place]
        below = rows[(rows["place"] >= lo) & (rows["place"] <= hi)]
        if len(target) != 1 or below.empty:
            continue
        value = float(target.iloc[0][feature])
        other = below[feature].astype(float)
        wins += int((value > other).sum())
        ties += int((value == other).sum())
        total += len(other)
    if not total:
        return None
    return {"winRate": round(100.0 * (wins + 0.5 * ties) / total, 1),
            "comparisons": total}


def rank_position(finals, place, feature):
    """Where the athlete who finished `place` sat in the pre-race order, and how
    often that put them in the top three and the top six."""
    top3 = top6 = counted = 0
    ranks, field_sizes = [], []
    for _, final in finals:
        rows = final.dropna(subset=[feature, "place"]).copy()
        if len(rows) < 6:
            continue
        target = rows[rows["place"] == place]
        if len(target) != 1:
            continue
        rows["pre_rank"] = rows[feature].rank(ascending=False, method="min")
        rank = float(rows.loc[target.index[0], "pre_rank"])
        ranks.append(rank)
        field_sizes.append(len(rows))
        counted += 1
        top3 += rank <= 3
        top6 += rank <= 6
    if not counted:
        return None
    mean_field = sum(field_sizes) / len(field_sizes)
    return {
        "finals": counted,
        "medianPreRaceRank": float(pd.Series(ranks).median()),
        "inPreRaceTop3": round(100.0 * top3 / counted, 1),
        "inPreRaceTop6": round(100.0 * top6 / counted, 1),
        # What picking at random from the same fields would give.
        "randomTop3": round(100.0 * 3 / mean_field, 1),
        "randomTop6": round(100.0 * 6 / mean_field, 1),
    }


def main():
    df = pd.read_csv(FINALS_PATH, low_memory=False)
    df = df.dropna(subset=["place"])
    finals = list(df.groupby(["competition_id", "year", "discipline"]))

    report = {
        "what": "Whether third place is separable, before the race, from the places "
                "below it. Model-free: everything is computed inside a final.",
        "finals": len(finals),
        "finalists": int(len(df)),
        "medallistsWithNoSeasonScore": round(
            100.0 * df[df["place"] <= 3]["sb_score"].isna().mean(), 2),
        "byPlace": {},
    }
    for place in PLACES:
        entry = {"pairwiseWinRateVsOffPodium": {}, "byBand": {}, "preRaceRank": {}}
        for feature in FEATURES:
            got = pairwise_win_rate(finals, place, feature)
            if got:
                entry["pairwiseWinRateVsOffPodium"][feature] = got
        # How the signal decays as the comparison gets closer.
        for name, (lo, hi) in bands_for(place).items():
            got = pairwise_win_rate(finals, place, "sb_score", lo, hi)
            if got:
                entry["byBand"][name] = got
        got = rank_position(finals, place, "sb_score")
        if got:
            entry["preRaceRank"]["sb_score"] = got
        report["byPlace"][str(place)] = entry

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)

    print(f"  {len(finals)} finals, {len(df)} finalists")
    print(f"  medallists with no season score: {report['medallistsWithNoSeasonScore']}%\n")
    print("  How often the athlete who finished Nth beat an off-podium athlete")
    print("  on a pre-race number, in the same final. 50% is no information.\n")
    print(f"  {'':<8}" + "".join(f"{f:>16}" for f in FEATURES))
    for place in PLACES:
        row = report["byPlace"][str(place)]["pairwiseWinRateVsOffPodium"]
        cells = "".join(f"{row[f]['winRate']:>15.1f}%" if f in row else f"{'-':>16}"
                        for f in FEATURES)
        print(f"  {place:<8}{cells}")
    print("\n  Where they sat in the pre-race order on season best:\n")
    print(f"  {'':<8}{'median rank':>12}{'top 3':>9}{'random':>9}{'top 6':>9}{'random':>9}")
    for place in PLACES:
        r = report["byPlace"][str(place)]["preRaceRank"].get("sb_score")
        if not r:
            continue
        print(f"  {place:<8}{r['medianPreRaceRank']:>12.0f}{r['inPreRaceTop3']:>8.1f}%"
              f"{r['randomTop3']:>8.1f}%{r['inPreRaceTop6']:>8.1f}%{r['randomTop6']:>8.1f}%")
    print("\n  How the signal decays as the comparison gets closer (season best).")
    print("  Bands are relative to the placing: 3rd vs 4th, 3rd vs 4th-6th, 3rd vs all below.\n")
    names = list(bands_for(1))
    print(f"  {'':<8}" + "".join(f"{n:>14}" for n in names))
    for place in PLACES:
        row = report["byPlace"][str(place)]["byBand"]
        cells = "".join(f"{row[n]['winRate']:>13.1f}%" if n in row else f"{'-':>14}"
                        for n in names)
        print(f"  {place:<8}{cells}")
    print(f"\n  -> {OUT_PATH}")


if __name__ == "__main__":
    main()
