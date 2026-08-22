"""
train_model.py — rebuilds the historical training set (2021-2023) and retrains
the RandomForest used by run.py, backtesting on 2023 the same way the original
notebook (notebooks/01_eda.ipynb, cell 28) did.

Fixes a bug found while adding recency features: the notebook's feature builder
looked for data/raw/{discipline}_{year}.csv (never existed for training years),
so weighted_season_best/wind_adj_season_best silently fell back to a copy of
season_best, and recent_trend/days_since_last always fell back to 0.0/999 for
every single training row — i.e. 4 of the model's intended features carried no
real signal. This reads from the actual historical file (data/raw/{discipline}.csv,
which has a year column) instead.

Usage:
    python src/train_model.py                  # fixed weighted/wind features only
    python src/train_model.py --with-recency    # + recent_trend, days_since_last
"""
import argparse
import os
import pickle
import sys
import io
import unicodedata

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

TRAIN_DISCIPLINES = {
    "men_100m":    "Men 100m",
    "women_100m":  "Women 100m",
    "men_200m":    "Men 200m",
    "men_400h":    "Men 400m Hurdles",
    "women_400h":  "Women 400m Hurdles",
    "men_PV":      "Men Pole Vault",
    "women_200m":  "Women 200m",
    "men_800m":    "Men 800m",
    "women_800m":  "Women 800m",
    "men_1500m":   "Men 1500m",
    "women_1500m": "Women 1500m",
    "women_PV":    "Women Pole Vault",
    "men_LJ":      "Men Long Jump",
    # Added to extend training beyond the original 13 disciplines (HANDOFF.md
    # Next Steps #1) -- historical data rebuilt via
    # `python src/historical_scraper.py --new-only`.
    "men_400m":     "Men 400m",
    "women_400m":   "Women 400m",
    "men_110h":     "Men 110m Hurdles",
    "women_100h":   "Women 100m Hurdles",
    "men_5000m":    "Men 5000m",
    "women_5000m":  "Women 5000m",
    "men_3000sc":   "Men 3000m Steeplechase",
    "women_3000sc": "Women 3000m Steeplechase",
    "men_HJ":       "Men High Jump",
    "women_HJ":     "Women High Jump",
    "men_TJ":       "Men Triple Jump",
    "women_TJ":     "Women Triple Jump",
    "men_SP":       "Men Shot Put",
    "women_SP":     "Women Shot Put",
    "men_DT":       "Men Discus Throw",
    "women_DT":     "Women Discus Throw",
    "men_JT":       "Men Javelin Throw",
    "women_JT":     "Women Javelin Throw",
    "women_LJ":     "Women Long Jump",
}
FIELD_EVENTS = {
    "men_PV", "women_PV", "men_LJ", "women_LJ",
    "men_HJ", "women_HJ", "men_TJ", "women_TJ",
    "men_SP", "women_SP", "men_DT", "women_DT", "men_JT", "women_JT",
}

# Diamond League Finals not held at these venues/dates were not always
# contested in their standard track form -- verified via World Athletics
# results pages + Wikipedia + trackalerts.com/world-track.org cross-checks
# (see conversation history 2026-08-22). Excluding these (discipline, year)
# combos entirely from the labeled dataset (not just from DL_RESULTS) so
# they don't inject spurious all-zero "nobody medaled" training/test rows
# for an event that literally didn't run that year.
NOT_CONTESTED = {
    ("men_5000m", 2022),   # Zurich Final ran a men's 5km road race instead
    ("men_5000m", 2023),   # Eugene Final had no 5000m -- Bowerman Mile + separate 3000m instead
    ("women_5000m", 2022), # Zurich Final ran a women's 5km road race instead
}
WIND_EVENTS = {"men_100m", "women_100m", "men_200m", "women_200m"}
DL_VENUES = [
    "doha", "shanghai", "suzhou", "shaoxing", "rabat", "florence", "paris",
    "oslo", "lausanne", "stockholm", "silesia", "monaco", "london",
    "zurich", "brussels", "eugene", "birmingham", "rome", "xiamen",
]
MAJOR_KEYWORDS = ["olympic", "world championship", "world athletics", "european championship"]

# DL Final winners and top-3 finishers 2021-2023 (unchanged from the notebook)
DL_RESULTS = [
    {"discipline": "men_100m", "year": 2021, "athlete_name": "Lamont Marcell JACOBS", "dl_rank": 1},
    {"discipline": "men_100m", "year": 2021, "athlete_name": "Zharnel HUGHES", "dl_rank": 2},
    {"discipline": "men_100m", "year": 2021, "athlete_name": "Fred KERLEY", "dl_rank": 3},
    {"discipline": "men_100m", "year": 2022, "athlete_name": "Fred KERLEY", "dl_rank": 1},
    {"discipline": "men_100m", "year": 2022, "athlete_name": "Trayvon BROMELL", "dl_rank": 2},
    {"discipline": "men_100m", "year": 2022, "athlete_name": "Oblique SEVILLE", "dl_rank": 3},
    {"discipline": "men_100m", "year": 2023, "athlete_name": "Noah LYLES", "dl_rank": 1},
    {"discipline": "men_100m", "year": 2023, "athlete_name": "Oblique SEVILLE", "dl_rank": 2},
    {"discipline": "men_100m", "year": 2023, "athlete_name": "Zharnel HUGHES", "dl_rank": 3},
    {"discipline": "women_100m", "year": 2021, "athlete_name": "Elaine THOMPSON-HERAH", "dl_rank": 1},
    {"discipline": "women_100m", "year": 2021, "athlete_name": "Shericka JACKSON", "dl_rank": 2},
    {"discipline": "women_100m", "year": 2021, "athlete_name": "Marie-Josee TA LOU", "dl_rank": 3},
    {"discipline": "women_100m", "year": 2022, "athlete_name": "Shericka JACKSON", "dl_rank": 1},
    {"discipline": "women_100m", "year": 2022, "athlete_name": "Elaine THOMPSON-HERAH", "dl_rank": 2},
    {"discipline": "women_100m", "year": 2022, "athlete_name": "Dina ASHER-SMITH", "dl_rank": 3},
    {"discipline": "women_100m", "year": 2023, "athlete_name": "Sha'Carri RICHARDSON", "dl_rank": 1},
    {"discipline": "women_100m", "year": 2023, "athlete_name": "Shericka JACKSON", "dl_rank": 2},
    {"discipline": "women_100m", "year": 2023, "athlete_name": "Elaine THOMPSON-HERAH", "dl_rank": 3},
    {"discipline": "men_200m", "year": 2021, "athlete_name": "Kenneth BEDNAREK", "dl_rank": 1},
    {"discipline": "men_200m", "year": 2021, "athlete_name": "Noah LYLES", "dl_rank": 2},
    {"discipline": "men_200m", "year": 2021, "athlete_name": "Fred KERLEY", "dl_rank": 3},
    {"discipline": "men_200m", "year": 2022, "athlete_name": "Noah LYLES", "dl_rank": 1},
    {"discipline": "men_200m", "year": 2022, "athlete_name": "Kenneth BEDNAREK", "dl_rank": 2},
    {"discipline": "men_200m", "year": 2022, "athlete_name": "Erriyon KNIGHTON", "dl_rank": 3},
    {"discipline": "men_200m", "year": 2023, "athlete_name": "Noah LYLES", "dl_rank": 1},
    {"discipline": "men_200m", "year": 2023, "athlete_name": "Kenneth BEDNAREK", "dl_rank": 2},
    {"discipline": "men_200m", "year": 2023, "athlete_name": "Erriyon KNIGHTON", "dl_rank": 3},
    {"discipline": "men_400h", "year": 2021, "athlete_name": "Karsten WARHOLM", "dl_rank": 1},
    {"discipline": "men_400h", "year": 2021, "athlete_name": "Alison DOS SANTOS", "dl_rank": 2},
    {"discipline": "men_400h", "year": 2021, "athlete_name": "Rai BENJAMIN", "dl_rank": 3},
    {"discipline": "men_400h", "year": 2022, "athlete_name": "Karsten WARHOLM", "dl_rank": 1},
    {"discipline": "men_400h", "year": 2022, "athlete_name": "Alison DOS SANTOS", "dl_rank": 2},
    {"discipline": "men_400h", "year": 2022, "athlete_name": "Rai BENJAMIN", "dl_rank": 3},
    {"discipline": "men_400h", "year": 2023, "athlete_name": "Karsten WARHOLM", "dl_rank": 1},
    {"discipline": "men_400h", "year": 2023, "athlete_name": "Alison DOS SANTOS", "dl_rank": 2},
    {"discipline": "men_400h", "year": 2023, "athlete_name": "Rai BENJAMIN", "dl_rank": 3},
    {"discipline": "women_400h", "year": 2021, "athlete_name": "Sydney MCLAUGHLIN", "dl_rank": 1},
    {"discipline": "women_400h", "year": 2021, "athlete_name": "Femke BOL", "dl_rank": 2},
    {"discipline": "women_400h", "year": 2021, "athlete_name": "Dalilah MUHAMMAD", "dl_rank": 3},
    {"discipline": "women_400h", "year": 2022, "athlete_name": "Sydney MCLAUGHLIN", "dl_rank": 1},
    {"discipline": "women_400h", "year": 2022, "athlete_name": "Femke BOL", "dl_rank": 2},
    {"discipline": "women_400h", "year": 2022, "athlete_name": "Anna COCKRELL", "dl_rank": 3},
    {"discipline": "women_400h", "year": 2023, "athlete_name": "Femke BOL", "dl_rank": 1},
    {"discipline": "women_400h", "year": 2023, "athlete_name": "Shamier LITTLE", "dl_rank": 2},
    {"discipline": "women_400h", "year": 2023, "athlete_name": "Rushell CLAYTON", "dl_rank": 3},
    {"discipline": "men_PV", "year": 2021, "athlete_name": "Armand DUPLANTIS", "dl_rank": 1},
    {"discipline": "men_PV", "year": 2021, "athlete_name": "Christopher NILSEN", "dl_rank": 2},
    {"discipline": "men_PV", "year": 2021, "athlete_name": "Ernest John OBIENA", "dl_rank": 3},
    {"discipline": "men_PV", "year": 2022, "athlete_name": "Armand DUPLANTIS", "dl_rank": 1},
    {"discipline": "men_PV", "year": 2022, "athlete_name": "Christopher NILSEN", "dl_rank": 2},
    {"discipline": "men_PV", "year": 2022, "athlete_name": "Ernest John OBIENA", "dl_rank": 3},
    {"discipline": "men_PV", "year": 2023, "athlete_name": "Armand DUPLANTIS", "dl_rank": 1},
    {"discipline": "men_PV", "year": 2023, "athlete_name": "Christopher NILSEN", "dl_rank": 2},
    {"discipline": "men_PV", "year": 2023, "athlete_name": "Ernest John OBIENA", "dl_rank": 3},
    {"discipline": "women_200m", "year": 2021, "athlete_name": "Gabrielle THOMAS", "dl_rank": 1},
    {"discipline": "women_200m", "year": 2021, "athlete_name": "Christine MBOMA", "dl_rank": 2},
    {"discipline": "women_200m", "year": 2021, "athlete_name": "Blessing OKAGBARE", "dl_rank": 3},
    {"discipline": "women_200m", "year": 2022, "athlete_name": "Shericka JACKSON", "dl_rank": 1},
    {"discipline": "women_200m", "year": 2022, "athlete_name": "Gabrielle THOMAS", "dl_rank": 2},
    {"discipline": "women_200m", "year": 2022, "athlete_name": "Tamara CLARK", "dl_rank": 3},
    {"discipline": "women_200m", "year": 2023, "athlete_name": "Sha'Carri RICHARDSON", "dl_rank": 1},
    {"discipline": "women_200m", "year": 2023, "athlete_name": "Gabrielle THOMAS", "dl_rank": 2},
    {"discipline": "women_200m", "year": 2023, "athlete_name": "Shericka JACKSON", "dl_rank": 3},
    {"discipline": "men_800m", "year": 2021, "athlete_name": "Emmanuel Kipkurui KORIR", "dl_rank": 1},
    {"discipline": "men_800m", "year": 2021, "athlete_name": "Peter BOL", "dl_rank": 2},
    {"discipline": "men_800m", "year": 2021, "athlete_name": "Nijel AMOS", "dl_rank": 3},
    {"discipline": "men_800m", "year": 2022, "athlete_name": "Marco AROP", "dl_rank": 1},
    {"discipline": "men_800m", "year": 2022, "athlete_name": "Emmanuel Kipkurui KORIR", "dl_rank": 2},
    {"discipline": "men_800m", "year": 2022, "athlete_name": "Djamel SEDJATI", "dl_rank": 3},
    {"discipline": "men_800m", "year": 2023, "athlete_name": "Emmanuel WANYONYI", "dl_rank": 1},
    {"discipline": "men_800m", "year": 2023, "athlete_name": "Marco AROP", "dl_rank": 2},
    {"discipline": "men_800m", "year": 2023, "athlete_name": "Djamel SEDJATI", "dl_rank": 3},
    {"discipline": "women_800m", "year": 2021, "athlete_name": "Athing MU", "dl_rank": 1},
    {"discipline": "women_800m", "year": 2021, "athlete_name": "Raevyn ROGERS", "dl_rank": 2},
    {"discipline": "women_800m", "year": 2021, "athlete_name": "Habitam ALEMU", "dl_rank": 3},
    {"discipline": "women_800m", "year": 2022, "athlete_name": "Athing MU", "dl_rank": 1},
    {"discipline": "women_800m", "year": 2022, "athlete_name": "Mary MORAA", "dl_rank": 2},
    {"discipline": "women_800m", "year": 2022, "athlete_name": "Keely HODGKINSON", "dl_rank": 3},
    {"discipline": "women_800m", "year": 2023, "athlete_name": "Mary MORAA", "dl_rank": 1},
    {"discipline": "women_800m", "year": 2023, "athlete_name": "Keely HODGKINSON", "dl_rank": 2},
    {"discipline": "women_800m", "year": 2023, "athlete_name": "Athing MU", "dl_rank": 3},
    {"discipline": "men_1500m", "year": 2021, "athlete_name": "Timothy CHERUIYOT", "dl_rank": 1},
    {"discipline": "men_1500m", "year": 2021, "athlete_name": "Jakob INGEBRIGTSEN", "dl_rank": 2},
    {"discipline": "men_1500m", "year": 2021, "athlete_name": "Josh KERR", "dl_rank": 3},
    {"discipline": "men_1500m", "year": 2022, "athlete_name": "Jakob INGEBRIGTSEN", "dl_rank": 1},
    {"discipline": "men_1500m", "year": 2022, "athlete_name": "Timothy CHERUIYOT", "dl_rank": 2},
    {"discipline": "men_1500m", "year": 2022, "athlete_name": "Josh KERR", "dl_rank": 3},
    {"discipline": "men_1500m", "year": 2023, "athlete_name": "Jakob INGEBRIGTSEN", "dl_rank": 1},
    {"discipline": "men_1500m", "year": 2023, "athlete_name": "Yared NUGUSE", "dl_rank": 2},
    {"discipline": "men_1500m", "year": 2023, "athlete_name": "Cole HOCKER", "dl_rank": 3},
    {"discipline": "women_1500m", "year": 2021, "athlete_name": "Faith Chepngetich KIPYEGON", "dl_rank": 1},
    {"discipline": "women_1500m", "year": 2021, "athlete_name": "Laura MUIR", "dl_rank": 2},
    {"discipline": "women_1500m", "year": 2021, "athlete_name": "Gudaf TSEGAY", "dl_rank": 3},
    {"discipline": "women_1500m", "year": 2022, "athlete_name": "Faith Chepngetich KIPYEGON", "dl_rank": 1},
    {"discipline": "women_1500m", "year": 2022, "athlete_name": "Laura MUIR", "dl_rank": 2},
    {"discipline": "women_1500m", "year": 2022, "athlete_name": "Gudaf TSEGAY", "dl_rank": 3},
    {"discipline": "women_1500m", "year": 2023, "athlete_name": "Faith Chepngetich KIPYEGON", "dl_rank": 1},
    {"discipline": "women_1500m", "year": 2023, "athlete_name": "Laura MUIR", "dl_rank": 2},
    {"discipline": "women_1500m", "year": 2023, "athlete_name": "Diribe WELTEJI", "dl_rank": 3},
    {"discipline": "women_PV", "year": 2021, "athlete_name": "Katie NAGEOTTE", "dl_rank": 1},
    {"discipline": "women_PV", "year": 2021, "athlete_name": "Anzhelika SIDOROVA", "dl_rank": 2},
    {"discipline": "women_PV", "year": 2021, "athlete_name": "Katerina STEFANIDI", "dl_rank": 3},
    {"discipline": "women_PV", "year": 2022, "athlete_name": "Nina KENNEDY", "dl_rank": 1},
    {"discipline": "women_PV", "year": 2022, "athlete_name": "Katie NAGEOTTE", "dl_rank": 2},
    {"discipline": "women_PV", "year": 2022, "athlete_name": "Angelica BENGTSSON", "dl_rank": 3},
    {"discipline": "women_PV", "year": 2023, "athlete_name": "Katie MOON", "dl_rank": 1},
    {"discipline": "women_PV", "year": 2023, "athlete_name": "Tina SUTEJ", "dl_rank": 2},
    {"discipline": "women_PV", "year": 2023, "athlete_name": "Sandi MORRIS", "dl_rank": 3},
    {"discipline": "men_LJ", "year": 2021, "athlete_name": "Miltiadis TENTOGLOU", "dl_rank": 1},
    {"discipline": "men_LJ", "year": 2021, "athlete_name": "Juan Miguel ECHEVARRIA", "dl_rank": 2},
    {"discipline": "men_LJ", "year": 2021, "athlete_name": "Marquise GOODWIN", "dl_rank": 3},
    {"discipline": "men_LJ", "year": 2022, "athlete_name": "Miltiadis TENTOGLOU", "dl_rank": 1},
    {"discipline": "men_LJ", "year": 2022, "athlete_name": "Marquis DENDY", "dl_rank": 2},
    {"discipline": "men_LJ", "year": 2022, "athlete_name": "Maykel MASSO", "dl_rank": 3},
    {"discipline": "men_LJ", "year": 2023, "athlete_name": "Miltiadis TENTOGLOU", "dl_rank": 1},
    {"discipline": "men_LJ", "year": 2023, "athlete_name": "Mattia FURLANI", "dl_rank": 2},
    {"discipline": "men_LJ", "year": 2023, "athlete_name": "Carey McLeod", "dl_rank": 3},

    # ================================================================
    # 19 new disciplines added 2026-08-22 -- researched and cross-verified
    # against World Athletics results pages / Wikipedia / trackalerts.com /
    # world-track.org / letsrun.com (2 independent background research
    # passes, one per gender). men_5000m 2022/2023 and women_5000m 2022 are
    # deliberately excluded (see NOT_CONTESTED) -- those Finals ran a road
    # race / different program instead of the standard track 5000m.
    # ================================================================
    # --- 2021 Zurich (Weltklasse Zürich, Sept 8-9, 2021) ---
    {"discipline": "men_400m", "year": 2021, "athlete_name": "Michael CHERRY", "dl_rank": 1},
    {"discipline": "men_400m", "year": 2021, "athlete_name": "Kirani JAMES", "dl_rank": 2},
    {"discipline": "men_400m", "year": 2021, "athlete_name": "Deon LENDORE", "dl_rank": 3},
    {"discipline": "men_110h", "year": 2021, "athlete_name": "Devon ALLEN", "dl_rank": 1},
    {"discipline": "men_110h", "year": 2021, "athlete_name": "Ronald LEVY", "dl_rank": 2},
    {"discipline": "men_110h", "year": 2021, "athlete_name": "Hansle PARCHMENT", "dl_rank": 3},
    {"discipline": "men_5000m", "year": 2021, "athlete_name": "Berihu AREGAWI", "dl_rank": 1},
    {"discipline": "men_5000m", "year": 2021, "athlete_name": "Birhanu BALEW", "dl_rank": 2},
    {"discipline": "men_5000m", "year": 2021, "athlete_name": "Jacob KROP", "dl_rank": 3},
    {"discipline": "men_3000sc", "year": 2021, "athlete_name": "Benjamin KIGEN", "dl_rank": 1},
    {"discipline": "men_3000sc", "year": 2021, "athlete_name": "Soufiane EL BAKKALI", "dl_rank": 2},
    {"discipline": "men_3000sc", "year": 2021, "athlete_name": "Abraham KIBIWOT", "dl_rank": 3},
    {"discipline": "men_HJ", "year": 2021, "athlete_name": "Gianmarco TAMBERI", "dl_rank": 1},
    {"discipline": "men_HJ", "year": 2021, "athlete_name": "Andriy PROTSENKO", "dl_rank": 2},
    {"discipline": "men_HJ", "year": 2021, "athlete_name": "Ilya IVANYUK", "dl_rank": 3},
    {"discipline": "men_TJ", "year": 2021, "athlete_name": "Pedro PICHARDO", "dl_rank": 1},
    {"discipline": "men_TJ", "year": 2021, "athlete_name": "Hugues Fabrice ZANGO", "dl_rank": 2},
    {"discipline": "men_TJ", "year": 2021, "athlete_name": "Yasser TRIKI", "dl_rank": 3},
    {"discipline": "men_SP", "year": 2021, "athlete_name": "Ryan CROUSER", "dl_rank": 1},
    {"discipline": "men_SP", "year": 2021, "athlete_name": "Joe KOVACS", "dl_rank": 2},
    {"discipline": "men_SP", "year": 2021, "athlete_name": "Armin SINANCEVIC", "dl_rank": 3},
    {"discipline": "men_DT", "year": 2021, "athlete_name": "Daniel STAHL", "dl_rank": 1},
    {"discipline": "men_DT", "year": 2021, "athlete_name": "Kristjan CEH", "dl_rank": 2},
    {"discipline": "men_DT", "year": 2021, "athlete_name": "Fedrick DACRES", "dl_rank": 3},
    {"discipline": "men_JT", "year": 2021, "athlete_name": "Johannes VETTER", "dl_rank": 1},
    {"discipline": "men_JT", "year": 2021, "athlete_name": "Julian WEBER", "dl_rank": 2},
    {"discipline": "men_JT", "year": 2021, "athlete_name": "Jakub VADLEJCH", "dl_rank": 3},
    {"discipline": "women_400m", "year": 2021, "athlete_name": "Quanera HAYES", "dl_rank": 1},
    {"discipline": "women_400m", "year": 2021, "athlete_name": "Marileidy PAULINO", "dl_rank": 2},
    {"discipline": "women_400m", "year": 2021, "athlete_name": "Sada WILLIAMS", "dl_rank": 3},
    {"discipline": "women_100h", "year": 2021, "athlete_name": "Tobi AMUSAN", "dl_rank": 1},
    {"discipline": "women_100h", "year": 2021, "athlete_name": "Nadine VISSER", "dl_rank": 2},
    {"discipline": "women_100h", "year": 2021, "athlete_name": "Megan TAPPER", "dl_rank": 3},
    {"discipline": "women_5000m", "year": 2021, "athlete_name": "Francine NIYONSABA", "dl_rank": 1},
    {"discipline": "women_5000m", "year": 2021, "athlete_name": "Hellen OBIRI", "dl_rank": 2},
    {"discipline": "women_5000m", "year": 2021, "athlete_name": "Ejgayehu TAYE", "dl_rank": 3},
    {"discipline": "women_3000sc", "year": 2021, "athlete_name": "Norah JERUTO", "dl_rank": 1},
    {"discipline": "women_3000sc", "year": 2021, "athlete_name": "Hyvin KIYENG", "dl_rank": 2},
    {"discipline": "women_3000sc", "year": 2021, "athlete_name": "Courtney FRERICHS", "dl_rank": 3},
    {"discipline": "women_HJ", "year": 2021, "athlete_name": "Mariya LASITSKENE", "dl_rank": 1},
    {"discipline": "women_HJ", "year": 2021, "athlete_name": "Yaroslava MAHUCHIKH", "dl_rank": 2},
    {"discipline": "women_HJ", "year": 2021, "athlete_name": "Nicola MCDERMOTT", "dl_rank": 3},
    {"discipline": "women_TJ", "year": 2021, "athlete_name": "Yulimar ROJAS", "dl_rank": 1},
    {"discipline": "women_TJ", "year": 2021, "athlete_name": "Shanieka RICKETTS", "dl_rank": 2},
    {"discipline": "women_TJ", "year": 2021, "athlete_name": "Kimberly WILLIAMS", "dl_rank": 3},
    {"discipline": "women_SP", "year": 2021, "athlete_name": "Maggie EWEN", "dl_rank": 1},
    {"discipline": "women_SP", "year": 2021, "athlete_name": "Auriol DONGMO", "dl_rank": 2},
    {"discipline": "women_SP", "year": 2021, "athlete_name": "Fanny ROOS", "dl_rank": 3},
    {"discipline": "women_DT", "year": 2021, "athlete_name": "Valarie ALLMAN", "dl_rank": 1},
    {"discipline": "women_DT", "year": 2021, "athlete_name": "Sandra PERKOVIC", "dl_rank": 2},
    {"discipline": "women_DT", "year": 2021, "athlete_name": "Yaime PEREZ", "dl_rank": 3},
    {"discipline": "women_JT", "year": 2021, "athlete_name": "Christin HUSSONG", "dl_rank": 1},
    {"discipline": "women_JT", "year": 2021, "athlete_name": "Kelsey-Lee BARBER", "dl_rank": 2},
    {"discipline": "women_JT", "year": 2021, "athlete_name": "Nikola OGRODNIKOVA", "dl_rank": 3},
    {"discipline": "women_LJ", "year": 2021, "athlete_name": "Ivana SPANOVIC", "dl_rank": 1},
    {"discipline": "women_LJ", "year": 2021, "athlete_name": "Khaddi SAGNIA", "dl_rank": 2},
    {"discipline": "women_LJ", "year": 2021, "athlete_name": "Maryna BEKH-ROMANCHUK", "dl_rank": 3},

    # --- 2022 Zurich (Weltklasse Zürich, Sept 7-8, 2022) ---
    {"discipline": "men_400m", "year": 2022, "athlete_name": "Kirani JAMES", "dl_rank": 1},
    {"discipline": "men_400m", "year": 2022, "athlete_name": "Bryce DEADMON", "dl_rank": 2},
    {"discipline": "men_400m", "year": 2022, "athlete_name": "Vernon NORWOOD", "dl_rank": 3},
    {"discipline": "men_110h", "year": 2022, "athlete_name": "Grant HOLLOWAY", "dl_rank": 1},
    {"discipline": "men_110h", "year": 2022, "athlete_name": "Rasheed BROADBELL", "dl_rank": 2},
    {"discipline": "men_110h", "year": 2022, "athlete_name": "Hansle PARCHMENT", "dl_rank": 3},
    {"discipline": "men_3000sc", "year": 2022, "athlete_name": "Soufiane EL BAKKALI", "dl_rank": 1},
    {"discipline": "men_3000sc", "year": 2022, "athlete_name": "Getnet WALE", "dl_rank": 2},
    {"discipline": "men_3000sc", "year": 2022, "athlete_name": "Abraham KIBIWOT", "dl_rank": 3},
    {"discipline": "men_HJ", "year": 2022, "athlete_name": "Gianmarco TAMBERI", "dl_rank": 1},
    {"discipline": "men_HJ", "year": 2022, "athlete_name": "JuVaughn HARRISON", "dl_rank": 2},
    {"discipline": "men_HJ", "year": 2022, "athlete_name": "Django LOVETT", "dl_rank": 3},
    {"discipline": "men_TJ", "year": 2022, "athlete_name": "Andy DIAZ", "dl_rank": 1},
    {"discipline": "men_TJ", "year": 2022, "athlete_name": "Pedro PICHARDO", "dl_rank": 2},
    {"discipline": "men_TJ", "year": 2022, "athlete_name": "Jordan DIAZ", "dl_rank": 3},
    {"discipline": "men_SP", "year": 2022, "athlete_name": "Joe KOVACS", "dl_rank": 1},
    {"discipline": "men_SP", "year": 2022, "athlete_name": "Ryan CROUSER", "dl_rank": 2},
    {"discipline": "men_SP", "year": 2022, "athlete_name": "Tom WALSH", "dl_rank": 3},
    {"discipline": "men_DT", "year": 2022, "athlete_name": "Kristjan CEH", "dl_rank": 1},
    {"discipline": "men_DT", "year": 2022, "athlete_name": "Lukas WEISSHAIDINGER", "dl_rank": 2},
    {"discipline": "men_DT", "year": 2022, "athlete_name": "Andrius GUDZIUS", "dl_rank": 3},
    {"discipline": "men_JT", "year": 2022, "athlete_name": "Neeraj CHOPRA", "dl_rank": 1},
    {"discipline": "men_JT", "year": 2022, "athlete_name": "Jakub VADLEJCH", "dl_rank": 2},
    {"discipline": "men_JT", "year": 2022, "athlete_name": "Julian WEBER", "dl_rank": 3},
    {"discipline": "women_400m", "year": 2022, "athlete_name": "Marileidy PAULINO", "dl_rank": 1},
    {"discipline": "women_400m", "year": 2022, "athlete_name": "Fiordaliza COFIL", "dl_rank": 2},
    {"discipline": "women_400m", "year": 2022, "athlete_name": "Sada WILLIAMS", "dl_rank": 3},
    {"discipline": "women_100h", "year": 2022, "athlete_name": "Tobi AMUSAN", "dl_rank": 1},
    {"discipline": "women_100h", "year": 2022, "athlete_name": "Tia JONES", "dl_rank": 2},
    {"discipline": "women_100h", "year": 2022, "athlete_name": "Britany ANDERSON", "dl_rank": 3},
    {"discipline": "women_3000sc", "year": 2022, "athlete_name": "Werkuha GETACHEW", "dl_rank": 1},
    {"discipline": "women_3000sc", "year": 2022, "athlete_name": "Winfred YAVI", "dl_rank": 2},
    {"discipline": "women_3000sc", "year": 2022, "athlete_name": "Faith CHEROTICH", "dl_rank": 3},
    {"discipline": "women_HJ", "year": 2022, "athlete_name": "Yaroslava MAHUCHIKH", "dl_rank": 1},
    {"discipline": "women_HJ", "year": 2022, "athlete_name": "Iryna GERASHCHENKO", "dl_rank": 2},
    {"discipline": "women_HJ", "year": 2022, "athlete_name": "Nicola OLYSLAGERS", "dl_rank": 3},
    {"discipline": "women_TJ", "year": 2022, "athlete_name": "Yulimar ROJAS", "dl_rank": 1},
    {"discipline": "women_TJ", "year": 2022, "athlete_name": "Maryna BEKH-ROMANCHUK", "dl_rank": 2},
    {"discipline": "women_TJ", "year": 2022, "athlete_name": "Shanieka RICKETTS", "dl_rank": 3},
    {"discipline": "women_SP", "year": 2022, "athlete_name": "Chase EALEY", "dl_rank": 1},
    {"discipline": "women_SP", "year": 2022, "athlete_name": "Sarah MITTON", "dl_rank": 2},
    {"discipline": "women_SP", "year": 2022, "athlete_name": "Auriol DONGMO", "dl_rank": 3},
    {"discipline": "women_DT", "year": 2022, "athlete_name": "Valarie ALLMAN", "dl_rank": 1},
    {"discipline": "women_DT", "year": 2022, "athlete_name": "Sandra PERKOVIC", "dl_rank": 2},
    {"discipline": "women_DT", "year": 2022, "athlete_name": "Liliana CA", "dl_rank": 3},
    {"discipline": "women_JT", "year": 2022, "athlete_name": "Kara WINGER", "dl_rank": 1},
    {"discipline": "women_JT", "year": 2022, "athlete_name": "Kelsey-Lee BARBER", "dl_rank": 2},
    {"discipline": "women_JT", "year": 2022, "athlete_name": "Haruka KITAGUCHI", "dl_rank": 3},
    {"discipline": "women_LJ", "year": 2022, "athlete_name": "Ivana VULETA", "dl_rank": 1},
    {"discipline": "women_LJ", "year": 2022, "athlete_name": "Khaddi SAGNIA", "dl_rank": 2},
    {"discipline": "women_LJ", "year": 2022, "athlete_name": "Quanesha BURKS", "dl_rank": 3},

    # --- 2023 Eugene (Prefontaine Classic / Hayward Field, Sept 16-17, 2023) ---
    {"discipline": "men_400m", "year": 2023, "athlete_name": "Kirani JAMES", "dl_rank": 1},
    {"discipline": "men_400m", "year": 2023, "athlete_name": "Quincy HALL", "dl_rank": 2},
    {"discipline": "men_400m", "year": 2023, "athlete_name": "Vernon NORWOOD", "dl_rank": 3},
    {"discipline": "men_110h", "year": 2023, "athlete_name": "Hansle PARCHMENT", "dl_rank": 1},
    {"discipline": "men_110h", "year": 2023, "athlete_name": "Grant HOLLOWAY", "dl_rank": 2},
    {"discipline": "men_110h", "year": 2023, "athlete_name": "Daniel ROBERTS", "dl_rank": 3},
    {"discipline": "men_3000sc", "year": 2023, "athlete_name": "Simon Kiprop KOECH", "dl_rank": 1},
    {"discipline": "men_3000sc", "year": 2023, "athlete_name": "Samuel FIREWU", "dl_rank": 2},
    {"discipline": "men_3000sc", "year": 2023, "athlete_name": "Geordie BEAMISH", "dl_rank": 3},
    {"discipline": "men_HJ", "year": 2023, "athlete_name": "Sanghyeok WOO", "dl_rank": 1},
    {"discipline": "men_HJ", "year": 2023, "athlete_name": "Norbert KOBIELSKI", "dl_rank": 2},
    {"discipline": "men_HJ", "year": 2023, "athlete_name": "JuVaughn HARRISON", "dl_rank": 3},
    {"discipline": "men_TJ", "year": 2023, "athlete_name": "Andy DIAZ", "dl_rank": 1},
    {"discipline": "men_TJ", "year": 2023, "athlete_name": "Hugues Fabrice ZANGO", "dl_rank": 2},
    {"discipline": "men_TJ", "year": 2023, "athlete_name": "Donald SCOTT", "dl_rank": 3},
    {"discipline": "men_SP", "year": 2023, "athlete_name": "Joe KOVACS", "dl_rank": 1},
    {"discipline": "men_SP", "year": 2023, "athlete_name": "Ryan CROUSER", "dl_rank": 2},
    {"discipline": "men_SP", "year": 2023, "athlete_name": "Tom WALSH", "dl_rank": 3},
    {"discipline": "men_DT", "year": 2023, "athlete_name": "Matthew DENNY", "dl_rank": 1},
    {"discipline": "men_DT", "year": 2023, "athlete_name": "Kristjan CEH", "dl_rank": 2},
    {"discipline": "men_DT", "year": 2023, "athlete_name": "Daniel STAHL", "dl_rank": 3},
    {"discipline": "men_JT", "year": 2023, "athlete_name": "Jakub VADLEJCH", "dl_rank": 1},
    {"discipline": "men_JT", "year": 2023, "athlete_name": "Neeraj CHOPRA", "dl_rank": 2},
    {"discipline": "men_JT", "year": 2023, "athlete_name": "Oliver HELANDER", "dl_rank": 3},
    {"discipline": "women_400m", "year": 2023, "athlete_name": "Marileidy PAULINO", "dl_rank": 1},
    {"discipline": "women_400m", "year": 2023, "athlete_name": "Natalia KACZMAREK", "dl_rank": 2},
    {"discipline": "women_400m", "year": 2023, "athlete_name": "Lieke KLAVER", "dl_rank": 3},
    {"discipline": "women_100h", "year": 2023, "athlete_name": "Tobi AMUSAN", "dl_rank": 1},
    {"discipline": "women_100h", "year": 2023, "athlete_name": "Jasmine CAMACHO-QUINN", "dl_rank": 2},
    {"discipline": "women_100h", "year": 2023, "athlete_name": "Kendra HARRISON", "dl_rank": 3},
    {"discipline": "women_5000m", "year": 2023, "athlete_name": "Gudaf TSEGAY", "dl_rank": 1},
    {"discipline": "women_5000m", "year": 2023, "athlete_name": "Beatrice CHEBET", "dl_rank": 2},
    {"discipline": "women_5000m", "year": 2023, "athlete_name": "Ejgayehu TAYE", "dl_rank": 3},
    {"discipline": "women_3000sc", "year": 2023, "athlete_name": "Winfred YAVI", "dl_rank": 1},
    {"discipline": "women_3000sc", "year": 2023, "athlete_name": "Beatrice CHEPKOECH", "dl_rank": 2},
    {"discipline": "women_3000sc", "year": 2023, "athlete_name": "Faith CHEROTICH", "dl_rank": 3},
    {"discipline": "women_HJ", "year": 2023, "athlete_name": "Yaroslava MAHUCHIKH", "dl_rank": 1},
    {"discipline": "women_HJ", "year": 2023, "athlete_name": "Nicola OLYSLAGERS", "dl_rank": 2},
    {"discipline": "women_HJ", "year": 2023, "athlete_name": "Angelina TOPIC", "dl_rank": 3},
    {"discipline": "women_TJ", "year": 2023, "athlete_name": "Yulimar ROJAS", "dl_rank": 1},
    {"discipline": "women_TJ", "year": 2023, "athlete_name": "Shanieka RICKETTS", "dl_rank": 2},
    {"discipline": "women_TJ", "year": 2023, "athlete_name": "Kimberly WILLIAMS", "dl_rank": 3},
    {"discipline": "women_SP", "year": 2023, "athlete_name": "Chase EALEY", "dl_rank": 1},
    {"discipline": "women_SP", "year": 2023, "athlete_name": "Sarah MITTON", "dl_rank": 2},
    {"discipline": "women_SP", "year": 2023, "athlete_name": "Auriol DONGMO", "dl_rank": 3},
    {"discipline": "women_DT", "year": 2023, "athlete_name": "Valarie ALLMAN", "dl_rank": 1},
    {"discipline": "women_DT", "year": 2023, "athlete_name": "Laulauga TAUSAGA", "dl_rank": 2},
    {"discipline": "women_DT", "year": 2023, "athlete_name": "Sandra PERKOVIC", "dl_rank": 3},
    {"discipline": "women_JT", "year": 2023, "athlete_name": "Haruka KITAGUCHI", "dl_rank": 1},
    {"discipline": "women_JT", "year": 2023, "athlete_name": "Tori PEETERS", "dl_rank": 2},
    {"discipline": "women_JT", "year": 2023, "athlete_name": "Mackenzie LITTLE", "dl_rank": 3},
    {"discipline": "women_LJ", "year": 2023, "athlete_name": "Ivana VULETA", "dl_rank": 1},
    {"discipline": "women_LJ", "year": 2023, "athlete_name": "Ese BRUME", "dl_rank": 2},
    {"discipline": "women_LJ", "year": 2023, "athlete_name": "Quanesha BURKS", "dl_rank": 3},
]
NAME_FIXES = {
    "Marcell JACOBS": "Lamont Marcell JACOBS",
    "Kenny BEDNAREK": "Kenneth BEDNAREK",
    "Mondo DUPLANTIS": "Armand DUPLANTIS",
    # Jordan Alejandro Diaz Fortun's WA toplist entry uses his full name +
    # double surname; the DL Final results credit him under a shortened form.
    "Jordan DIAZ": "Jordan A. Diaz Fortun",
    # Found via the unmatched-DL_RESULTS check added 2026-08-22 -- these
    # silently failed the old exact-string merge (no warning existed before):
    "Andy DIAZ": "Andy Diaz Hernandez",       # WA toplist uses his full surname
    "Chase EALEY": "Chase Ealy",              # WA toplist itself misspells her surname
    "Faith Chepngetich KIPYEGON": "Faith Kipyegon",  # WA toplist omits her middle name
    "Andriy PROTSENKO": "Andrii Protsenko",   # WA's Ukrainian transliteration variant
    "Yasser TRIKI": "Yasser Mohammed Triki",   # WA toplist includes his middle name
    "Simon Kiprop KOECH": "Simon Koech",       # WA toplist omits his middle name
    "Katerina STEFANIDI": "Aikaterini Stefanidi",  # WA toplist uses her full Greek first name
}


def convert_mark_to_seconds(mark_str):
    try:
        mark_str = str(mark_str).strip()
        if ":" in mark_str:
            parts = mark_str.split(":")
            if len(parts) == 2:
                return float(parts[0]) * 60 + float(parts[1])
        return float(mark_str)
    except Exception:
        return None


def clean_discipline(df):
    df = df.drop(columns=["Unnamed: 0", "discipline"], errors="ignore")
    rename_map = {
        "Competitor": "athlete_name", "DOB": "dob", "Nat": "country",
        "Results Score": "results_score", "Pos": "pos", "Venue": "venue", "Date": "date",
    }
    if "WIND" in df.columns:
        rename_map["WIND"] = "wind"
    df = df.rename(columns=rename_map)
    if "wind" not in df.columns:
        df["wind"] = np.nan
    df["date"] = pd.to_datetime(df["date"], format="%d %b %Y", errors="coerce")
    df["dob"] = pd.to_datetime(df["dob"], format="%d %b %Y", errors="coerce")
    df["age"] = ((df["date"] - df["dob"]).dt.days / 365.25).round(1)
    # Deliberately NOT truncating to the labeled years (2021-2023) here --
    # build_features() needs earlier history (2018-2020) to compute a real
    # career_best/yoy_improvement for those years instead of silently
    # defaulting yoy to 0.0 for every 2021 row (there's no "before" left to
    # compare against once truncated). Future years (2024+) are still safe:
    # build_features() only loops over [2021, 2022, 2023] as label years, and
    # career_best/yoy per row are separately bounded to that row's own year.
    return df.dropna(subset=["Mark"]).copy()


def build_features(df, discipline_key):
    records = []
    is_track = discipline_key not in FIELD_EVENTS
    for athlete in df["athlete_name"].unique():
        ath = df[df["athlete_name"] == athlete].copy()
        ath["Mark_num"] = ath["Mark"].apply(convert_mark_to_seconds)
        ath = ath.dropna(subset=["Mark_num"])
        if ath.empty:
            continue
        for year in [2021, 2022, 2023]:
            if (discipline_key, year) in NOT_CONTESTED:
                continue
            season = ath[ath["year"] == year]
            prev = ath[ath["year"] < year]
            # career_best must only see marks up to and including this label's
            # year -- using the full athlete history here would leak future
            # seasons (e.g. 2024/2025) into a 2021/2022 labeled row's features.
            up_to_year = ath[ath["year"] <= year]
            if season.empty:
                continue
            if is_track:
                season_best = season["Mark_num"].min()
                career_best = up_to_year["Mark_num"].min()
            else:
                season_best = season["Mark_num"].max()
                career_best = up_to_year["Mark_num"].max()
            pb_gap = abs(season_best - career_best)
            meets_count = len(season)
            consistency = season["Mark_num"].std() if len(season) > 1 else 0.0
            if not prev.empty:
                prev_best = prev["Mark_num"].min() if is_track else prev["Mark_num"].max()
                yoy = (prev_best - season_best) if is_track else (season_best - prev_best)
            else:
                yoy = 0.0
            age = ath["age"].dropna().median()
            country = ath["country"].iloc[0]
            records.append({
                "athlete_name": athlete, "country": country, "discipline": discipline_key,
                "year": year, "season_best": round(season_best, 4), "career_best": round(career_best, 4),
                "pb_gap": round(pb_gap, 4), "meets_count": meets_count,
                "consistency": round(consistency, 4), "yoy_improvement": round(yoy, 4),
                "age": round(age, 1) if not np.isnan(age) else np.nan,
            })
    return pd.DataFrame(records)


def add_season_rank(df):
    """discipline in FIELD_EVENTS -- not just "men_PV" -- must rank descending
    (higher mark = better = rank 1). Using a hardcoded "men_PV" string here
    previously left women_PV and men_LJ ranked backwards (lowest jump/vault
    getting rank 1) for the entire time they've been trained disciplines --
    caught while adding 10 more field events that would have hit the same bug."""
    all_groups = []
    for (discipline, year), group in df.groupby(["discipline", "year"]):
        group = group.copy()
        if discipline in FIELD_EVENTS:
            group["season_rank"] = group["season_best"].rank(ascending=False)
            group["season_percentile"] = group["season_best"].rank(ascending=True) / len(group)
        else:
            group["season_rank"] = group["season_best"].rank(ascending=True)
            group["season_percentile"] = group["season_best"].rank(ascending=False) / len(group)
        all_groups.append(group)
    return pd.concat(all_groups, ignore_index=True)


def competition_weight(venue):
    if not isinstance(venue, str):
        return 1.0
    v = venue.lower()
    if any(k in v for k in MAJOR_KEYWORDS):
        return 1.3
    if any(dl in v for dl in DL_VENUES):
        return 1.2
    return 1.0


def add_new_features(df):
    """Same idea as the notebook's add_new_features, but reads the real
    historical file (data/raw/{discipline}.csv, filtered by year) instead of
    a per-year file that never existed — that bug is why weighted_season_best/
    wind_adj_season_best were silent duplicates of season_best, and why
    recent_trend/days_since_last were always 0.0/999 for every training row."""
    all_groups = []
    for (discipline, year), group in df.groupby(["discipline", "year"]):
        group = group.copy()
        is_field = discipline in FIELD_EVENTS
        weighted_sb_map, wind_adj_map, trend_map, days_map = {}, {}, {}, {}

        raw_path = os.path.join(RAW_DIR, f"{discipline}.csv")
        if os.path.exists(raw_path):
            raw_full = pd.read_csv(raw_path)
            raw = raw_full[raw_full["year"] == year].copy()
            raw = raw.rename(columns={"Competitor": "athlete_name", "Mark": "mark_str"})
            raw["Mark"] = raw["mark_str"].apply(convert_mark_to_seconds)
            raw = raw.dropna(subset=["Mark"])

            if "Venue" in raw.columns:
                raw["comp_weight"] = raw["Venue"].apply(competition_weight)
                raw["weighted_mark"] = raw["Mark"] * raw["comp_weight"]
                wsb = (raw.groupby("athlete_name")["weighted_mark"].max() if is_field
                       else raw.groupby("athlete_name")["weighted_mark"].min())
                weighted_sb_map = wsb.to_dict()

            if discipline in WIND_EVENTS and "WIND" in raw.columns:
                def wind_adj(row):
                    try:
                        wind = float(str(row["WIND"]).replace("+", "").strip())
                        if wind > 1.0:
                            return row["Mark"] + (wind - 1.0) * 0.01
                        return row["Mark"]
                    except Exception:
                        return row["Mark"]
                raw["wind_adj"] = raw.apply(wind_adj, axis=1)
                wind_adj_map = raw.groupby("athlete_name")["wind_adj"].min().to_dict()

            if "Date" in raw.columns:
                raw["date"] = pd.to_datetime(raw["Date"], format="%d %b %Y", errors="coerce")
                ref_date = pd.Timestamp(f"{year}-09-01")
                for athlete in group["athlete_name"]:
                    ath = raw[raw["athlete_name"] == athlete].sort_values("date", ascending=False)
                    if ath.empty or ath["date"].isna().all():
                        trend_map[athlete] = 0.0
                        days_map[athlete] = 999
                        continue
                    last = ath["date"].dropna().iloc[0]
                    days_map[athlete] = (ref_date - last).days
                    recent = ath.head(3)["Mark"].tolist()
                    if len(recent) >= 2:
                        trend_map[athlete] = recent[0] - recent[-1] if not is_field else recent[-1] - recent[0]
                    else:
                        trend_map[athlete] = 0.0

        group["weighted_season_best"] = group["athlete_name"].map(weighted_sb_map).fillna(group["season_best"])
        group["wind_adj_season_best"] = group["athlete_name"].map(wind_adj_map).fillna(group["season_best"])
        group["recent_trend"] = group["athlete_name"].map(trend_map).fillna(0.0)
        group["days_since_last"] = group["athlete_name"].map(days_map).fillna(999)
        all_groups.append(group)
    return pd.concat(all_groups, ignore_index=True)


def normalize_name(name):
    """Case- and diacritic-insensitive key for matching hand/agent-researched
    DL_RESULTS athlete names against WA toplist Competitor names. Added
    alongside the 19-discipline expansion after finding several accented
    names (Cá, Perkovic/Perković, Ceh/Čeh, Spanovic/Španović...) typed
    without diacritics -- the exact h2h_win_rate failure mode from earlier
    this session (silent join misses defaulting everyone to a neutral/zero
    label) would otherwise repeat here across ~30 more athletes."""
    if not isinstance(name, str):
        return name
    nfkd = unicodedata.normalize("NFKD", name)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).upper().strip()


def build_labeled_dataset():
    dfs = {}
    for key in TRAIN_DISCIPLINES:
        path = os.path.join(RAW_DIR, f"{key}.csv")
        df = pd.read_csv(path)
        dfs[key] = clean_discipline(df)

    all_features = {key: build_features(dfs[key], key) for key in TRAIN_DISCIPLINES}
    master = pd.concat(all_features.values(), ignore_index=True)
    master["_name_key"] = master["athlete_name"].apply(normalize_name)

    dl_df = pd.DataFrame(DL_RESULTS)
    dl_df["athlete_name"] = dl_df["athlete_name"].replace(NAME_FIXES)
    dl_df["dl_winner"] = (dl_df["dl_rank"] == 1).astype(int)
    dl_df["dl_top3"] = (dl_df["dl_rank"] <= 3).astype(int)
    dl_df["_name_key"] = dl_df["athlete_name"].apply(normalize_name)

    labeled = master.merge(
        dl_df[["discipline", "year", "_name_key", "dl_winner", "dl_top3", "dl_rank"]],
        on=["discipline", "year", "_name_key"], how="left",
    )
    labeled["dl_winner"] = labeled["dl_winner"].fillna(0).astype(int)
    labeled["dl_top3"] = labeled["dl_top3"].fillna(0).astype(int)
    labeled["dl_rank"] = labeled["dl_rank"].fillna(0).astype(int)

    matched_keys = set(
        labeled.loc[labeled["dl_top3"] == 1, ["discipline", "year", "_name_key"]]
        .apply(tuple, axis=1)
    )
    unmatched = [
        (r["discipline"], r["year"], r["athlete_name"])
        for _, r in dl_df.iterrows()
        if (r["discipline"], r["year"], r["_name_key"]) not in matched_keys
    ]
    if unmatched:
        print(f"\n  WARNING: {len(unmatched)} DL_RESULTS entries found NO matching "
              f"training row (name mismatch or missing from raw toplist data):")
        for discipline, year, name in unmatched:
            print(f"    {discipline} {year}: {name!r}")

    return labeled.drop(columns=["_name_key"])


def add_h2h_features(df):
    """Adds h2h_win_rate per (athlete, discipline, year) row: average win
    rate against the other athletes in that discipline-year's training pool
    (>=2 meetings required, matching run.py's inference-time threshold).

    data/h2h/h2h_rates.csv uses normal-case names ("Trayvon Bromell") while
    every other data source in this pipeline uses WA's ALL-CAPS-surname
    format ("Trayvon BROMELL") -- an exact-string match between the two
    finds ZERO matches. This silently made h2h_win_rate default to a neutral
    0.5 for every athlete in every prediction ever made by run.py's blend,
    despite 156k real matchup rows sitting unused. Matching case-insensitive
    here (and in run.py) is the actual fix -- confirmed live: 0/8 exact
    matches vs 7/8 case-insensitive matches for a sample discipline.
    """
    h2h_path = os.path.join(os.path.dirname(__file__), "..", "data", "h2h", "h2h_rates.csv")
    h2h_df = pd.read_csv(h2h_path)
    h2h_df["a_lower"] = h2h_df["athlete_a"].str.lower()
    h2h_df["b_lower"] = h2h_df["athlete_b"].str.lower()
    h2h_df = h2h_df[h2h_df["meetings"] >= 2]

    df = df.copy()
    rates = []
    for (discipline, year), group in df.groupby(["discipline", "year"]):
        sub = h2h_df[h2h_df["discipline"] == discipline]
        lookup = {}
        for _, r in sub.iterrows():
            lookup.setdefault(r["a_lower"], {})[r["b_lower"]] = r["win_rate"]
        names_lower = [n.lower() for n in group["athlete_name"]]
        for name_lower in names_lower:
            opp_rates = [
                lookup[name_lower][opp] for opp in names_lower
                if opp != name_lower and name_lower in lookup and opp in lookup[name_lower]
            ]
            rates.append(sum(opp_rates) / len(opp_rates) if opp_rates else 0.5)
    df["h2h_win_rate"] = rates
    return df


def train_and_backtest(feature_cols, label=""):
    labeled = build_labeled_dataset()
    ranked = add_season_rank(labeled)
    full = add_new_features(ranked)
    full = add_h2h_features(full)

    train = full[full["year"].isin([2021, 2022])].dropna(subset=feature_cols)
    test = full[full["year"] == 2023].dropna(subset=feature_cols)

    X_train, y_train = train[feature_cols], train["dl_top3"]
    X_test, y_test = test[feature_cols], test["dl_top3"]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
    model.fit(X_train_scaled, y_train)

    test = test.copy()
    test["win_probability"] = model.predict_proba(X_test_scaled)[:, 1]

    print(f"\n=== {label} — 2023 Backtest ===")
    total_correct = 0
    n_disciplines = test["discipline"].nunique()
    for discipline in test["discipline"].unique():
        disc_df = test[test["discipline"] == discipline].sort_values("win_probability", ascending=False)
        top3_predicted = disc_df.head(3)["athlete_name"].tolist()
        top3_actual = disc_df[disc_df["dl_top3"] == 1]["athlete_name"].tolist()
        hits = len(set(top3_predicted) & set(top3_actual))
        total_correct += hits
        print(f"  {discipline}: {hits}/3")

    accuracy_pct = round(total_correct / (n_disciplines * 3) * 100, 1)
    print(f"  Total: {total_correct}/{n_disciplines * 3} = {accuracy_pct}%")

    print("\n  Feature importances:")
    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    for feat, imp in importances.items():
        print(f"    {feat:24s} {imp:.4f}")

    return model, scaler, accuracy_pct


def save_artifacts(model, scaler, feature_cols, accuracy_pct):
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    with open(os.path.join(OUTPUTS_DIR, "model_rf.pkl"), "wb") as f:
        pickle.dump(model, f)
    with open(os.path.join(OUTPUTS_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(OUTPUTS_DIR, "feature_cols.pkl"), "wb") as f:
        pickle.dump(feature_cols, f)
    with open(os.path.join(OUTPUTS_DIR, "model_accuracy.txt"), "w") as f:
        f.write(str(accuracy_pct))
    print(f"\nSaved model_rf.pkl, scaler.pkl, feature_cols.pkl, model_accuracy.txt ({accuracy_pct}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-recency", action="store_true",
                        help="Add recent_trend/days_since_last to the trained feature set")
    parser.add_argument("--with-h2h", action="store_true",
                        help="Add h2h_win_rate to the trained feature set (requires the "
                             "case-insensitive matching fix -- see add_h2h_features)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Backtest only, don't overwrite outputs/")
    args = parser.parse_args()

    base_cols = [
        "season_best", "career_best", "pb_gap", "meets_count", "consistency",
        "yoy_improvement", "age", "season_rank", "season_percentile",
        "weighted_season_best", "wind_adj_season_best",
    ]
    feature_cols = base_cols.copy()
    label_parts = ["V3-fixed"]
    if args.with_recency:
        feature_cols += ["recent_trend", "days_since_last"]
        label_parts.append("recency")
    if args.with_h2h:
        feature_cols += ["h2h_win_rate"]
        label_parts.append("h2h")
    label = " + ".join(label_parts)

    model, scaler, accuracy_pct = train_and_backtest(feature_cols, label=label)

    if not args.dry_run:
        save_artifacts(model, scaler, feature_cols, accuracy_pct)
    else:
        print("\n[dry run — outputs/ not modified]")
