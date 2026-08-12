import zipfile
import pandas as pd
import os

RAW_DIR = r"C:\Users\rayen\athletics-predictor\data\raw"

NEW_DISCIPLINES = {
    "women_200m":  "women/200-metres.csv",
    "men_800m":    "men/800-metres.csv",
    "women_800m":  "women/800-metres.csv",
    "men_1500m":   "men/1500-metres.csv",
    "women_1500m": "women/1500-metres.csv",
    "women_PV":    "women/pole-vault.csv",
    "men_LJ":      "men/long-jump.csv",
}

zip_path = os.path.join(RAW_DIR, "archive.zip")

with zipfile.ZipFile(zip_path, "r") as z:
    for key, path in NEW_DISCIPLINES.items():
        with z.open(path) as f:
            df = pd.read_csv(f)
            df["year"] = pd.to_datetime(df["Date"], format="%d %b %Y", errors="coerce").dt.year
            df["discipline"] = key
            out = os.path.join(RAW_DIR, f"{key}.csv")
            df.to_csv(out, index=False)
            print(f"{key}: {len(df)} rows, years {int(df['year'].min())}-{int(df['year'].max())}")