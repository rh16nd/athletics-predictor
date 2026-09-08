"""
h2h_scraper.py — Scrapes meet results from Wikipedia for head-to-head calculations
"""
import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import time
import re
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
os.makedirs(H2H_DIR, exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0 (athletics-predictor research project)"}

MEET_PAGES = [
    # 2021 Diamond League
    ("DL", "2021", "https://en.wikipedia.org/wiki/2021_Doha_Diamond_League"),
    ("DL", "2021", "https://en.wikipedia.org/wiki/2021_Gateshead_Diamond_League"),
    ("DL", "2021", "https://en.wikipedia.org/wiki/2021_Bislett_Games"),
    ("DL", "2021", "https://en.wikipedia.org/wiki/2021_Prefontaine_Classic"),
    ("DL", "2021", "https://en.wikipedia.org/wiki/2021_Herculis"),
    ("DL", "2021", "https://en.wikipedia.org/wiki/2021_Athletissima"),
    ("DL", "2021", "https://en.wikipedia.org/wiki/2021_Bauhausgalan"),
    ("DL", "2021", "https://en.wikipedia.org/wiki/2021_Weltklasse_Z%C3%BCrich"),
    ("DL", "2021", "https://en.wikipedia.org/wiki/2021_Memorial_Van_Damme"),
    # 2022 Diamond League
    ("DL", "2022", "https://en.wikipedia.org/wiki/2022_Doha_Diamond_League"),
    ("DL", "2022", "https://en.wikipedia.org/wiki/2022_Birmingham_Diamond_League"),
    ("DL", "2022", "https://en.wikipedia.org/wiki/2022_Prefontaine_Classic"),
    ("DL", "2022", "https://en.wikipedia.org/wiki/2022_Bislett_Games"),
    ("DL", "2022", "https://en.wikipedia.org/wiki/2022_Bauhausgalan"),
    ("DL", "2022", "https://en.wikipedia.org/wiki/2022_Herculis"),
    ("DL", "2022", "https://en.wikipedia.org/wiki/2022_Meeting_de_Paris"),
    ("DL", "2022", "https://en.wikipedia.org/wiki/2022_Athletissima"),
    ("DL", "2022", "https://en.wikipedia.org/wiki/2022_Kamila_Skolimowska_Memorial"),
    ("DL", "2022", "https://en.wikipedia.org/wiki/2022_Weltklasse_Z%C3%BCrich"),
    ("DL", "2022", "https://en.wikipedia.org/wiki/2022_Memorial_Van_Damme"),
    # 2023 Diamond League
    ("DL", "2023", "https://en.wikipedia.org/wiki/2023_Doha_Diamond_League"),
    ("DL", "2023", "https://en.wikipedia.org/wiki/2023_Meeting_International_Mohammed_VI_d%27Athl%C3%A9tisme_de_Rabat"),
    ("DL", "2023", "https://en.wikipedia.org/wiki/2023_Meeting_de_Paris"),
    ("DL", "2023", "https://en.wikipedia.org/wiki/2023_Bislett_Games"),
    ("DL", "2023", "https://en.wikipedia.org/wiki/2023_Athletissima"),
    ("DL", "2023", "https://en.wikipedia.org/wiki/2023_Bauhausgalan"),
    ("DL", "2023", "https://en.wikipedia.org/wiki/2023_Kamila_Skolimowska_Memorial"),
    ("DL", "2023", "https://en.wikipedia.org/wiki/2023_Herculis"),
    ("DL", "2023", "https://en.wikipedia.org/wiki/2023_London_Athletics_Meet"),
    ("DL", "2023", "https://en.wikipedia.org/wiki/2023_Weltklasse_Z%C3%BCrich"),
    ("DL", "2023", "https://en.wikipedia.org/wiki/2023_Memorial_Van_Damme"),
    ("DL", "2023", "https://en.wikipedia.org/wiki/2023_Prefontaine_Classic"),
    # 2024 Diamond League
    ("DL", "2024", "https://en.wikipedia.org/wiki/2024_Shanghai_Diamond_League"),
    ("DL", "2024", "https://en.wikipedia.org/wiki/2024_Doha_Diamond_League"),
    ("DL", "2024", "https://en.wikipedia.org/wiki/2024_Prefontaine_Classic"),
    ("DL", "2024", "https://en.wikipedia.org/wiki/2024_Bislett_Games"),
    ("DL", "2024", "https://en.wikipedia.org/wiki/2024_Bauhausgalan"),
    ("DL", "2024", "https://en.wikipedia.org/wiki/2024_Meeting_de_Paris"),
    ("DL", "2024", "https://en.wikipedia.org/wiki/2024_Herculis"),
    ("DL", "2024", "https://en.wikipedia.org/wiki/2024_London_Athletics_Meet"),
    ("DL", "2024", "https://en.wikipedia.org/wiki/2024_Athletissima"),
    ("DL", "2024", "https://en.wikipedia.org/wiki/2024_Kamila_Skolimowska_Memorial"),
    ("DL", "2024", "https://en.wikipedia.org/wiki/2024_Weltklasse_Z%C3%BCrich"),
    ("DL", "2024", "https://en.wikipedia.org/wiki/2024_Memorial_Van_Damme"),
    # 2025 Diamond League
    ("DL", "2025", "https://en.wikipedia.org/wiki/2025_Doha_Diamond_League"),
    ("DL", "2025", "https://en.wikipedia.org/wiki/2025_Shanghai_Diamond_League"),
    ("DL", "2025", "https://en.wikipedia.org/wiki/2025_Meeting_International_Mohammed_VI_d%27Athl%C3%A9tisme_de_Rabat"),
    ("DL", "2025", "https://en.wikipedia.org/wiki/2025_Bislett_Games"),
    ("DL", "2025", "https://en.wikipedia.org/wiki/2025_Bauhausgalan"),
    ("DL", "2025", "https://en.wikipedia.org/wiki/2025_Meeting_de_Paris"),
    ("DL", "2025", "https://en.wikipedia.org/wiki/2025_Prefontaine_Classic"),
    ("DL", "2025", "https://en.wikipedia.org/wiki/2025_London_Athletics_Meet"),
    ("DL", "2025", "https://en.wikipedia.org/wiki/2025_Herculis"),
    ("DL", "2025", "https://en.wikipedia.org/wiki/2025_Athletissima"),
    ("DL", "2025", "https://en.wikipedia.org/wiki/2025_Kamila_Skolimowska_Memorial"),
    ("DL", "2025", "https://en.wikipedia.org/wiki/2025_Weltklasse_Z%C3%BCrich"),
    ("DL", "2025", "https://en.wikipedia.org/wiki/2025_Memorial_Van_Damme"),
    # Olympics
    ("Olympics", "2021", "https://en.wikipedia.org/wiki/Athletics_at_the_2020_Summer_Olympics"),
    ("Olympics", "2024", "https://en.wikipedia.org/wiki/Athletics_at_the_2024_Summer_Olympics"),
    # World Championships
    ("Worlds", "2022", "https://en.wikipedia.org/wiki/2022_World_Athletics_Championships"),
    ("Worlds", "2023", "https://en.wikipedia.org/wiki/2023_World_Athletics_Championships"),
    ("Worlds", "2025", "https://en.wikipedia.org/wiki/2025_World_Athletics_Championships"),
    # European Championships
    ("Europeans", "2022", "https://en.wikipedia.org/wiki/2022_European_Athletics_Championships"),
    ("Europeans", "2024", "https://en.wikipedia.org/wiki/2024_European_Athletics_Championships"),
    ("Europeans", "2026", "https://en.wikipedia.org/wiki/2026_European_Athletics_Championships"),
    # 2022 World Athletics Championships — individual events
    ("Worlds", "2022", "https://en.wikipedia.org/wiki/2022_World_Athletics_Championships_%E2%80%93_Men%27s_100_metres"),
    ("Worlds", "2022", "https://en.wikipedia.org/wiki/2022_World_Athletics_Championships_%E2%80%93_Women%27s_100_metres"),
    ("Worlds", "2022", "https://en.wikipedia.org/wiki/2022_World_Athletics_Championships_%E2%80%93_Men%27s_200_metres"),
    ("Worlds", "2022", "https://en.wikipedia.org/wiki/2022_World_Athletics_Championships_%E2%80%93_Women%27s_200_metres"),
    ("Worlds", "2022", "https://en.wikipedia.org/wiki/2022_World_Athletics_Championships_%E2%80%93_Men%27s_400_metres"),
    ("Worlds", "2022", "https://en.wikipedia.org/wiki/2022_World_Athletics_Championships_%E2%80%93_Women%27s_400_metres"),
    ("Worlds", "2022", "https://en.wikipedia.org/wiki/2022_World_Athletics_Championships_%E2%80%93_Men%27s_110_metres_hurdles"),
    ("Worlds", "2022", "https://en.wikipedia.org/wiki/2022_World_Athletics_Championships_%E2%80%93_Women%27s_100_metres_hurdles"),
    ("Worlds", "2022", "https://en.wikipedia.org/wiki/2022_World_Athletics_Championships_%E2%80%93_Men%27s_400_metres_hurdles"),
    ("Worlds", "2022", "https://en.wikipedia.org/wiki/2022_World_Athletics_Championships_%E2%80%93_Women%27s_400_metres_hurdles"),
    ("Worlds", "2022", "https://en.wikipedia.org/wiki/2022_World_Athletics_Championships_%E2%80%93_Men%27s_800_metres"),
    ("Worlds", "2022", "https://en.wikipedia.org/wiki/2022_World_Athletics_Championships_%E2%80%93_Women%27s_800_metres"),
    ("Worlds", "2022", "https://en.wikipedia.org/wiki/2022_World_Athletics_Championships_%E2%80%93_Men%27s_1500_metres"),
    ("Worlds", "2022", "https://en.wikipedia.org/wiki/2022_World_Athletics_Championships_%E2%80%93_Women%27s_1500_metres"),
    ("Worlds", "2022", "https://en.wikipedia.org/wiki/2022_World_Athletics_Championships_%E2%80%93_Men%27s_5000_metres"),
    ("Worlds", "2022", "https://en.wikipedia.org/wiki/2022_World_Athletics_Championships_%E2%80%93_Women%27s_5000_metres"),
    ("Worlds", "2022", "https://en.wikipedia.org/wiki/2022_World_Athletics_Championships_%E2%80%93_Men%27s_3000_metres_steeplechase"),
    ("Worlds", "2022", "https://en.wikipedia.org/wiki/2022_World_Athletics_Championships_%E2%80%93_Women%27s_3000_metres_steeplechase"),
    ("Worlds", "2022", "https://en.wikipedia.org/wiki/2022_World_Athletics_Championships_%E2%80%93_Men%27s_high_jump"),
    ("Worlds", "2022", "https://en.wikipedia.org/wiki/2022_World_Athletics_Championships_%E2%80%93_Women%27s_high_jump"),
    ("Worlds", "2022", "https://en.wikipedia.org/wiki/2022_World_Athletics_Championships_%E2%80%93_Men%27s_pole_vault"),
    ("Worlds", "2022", "https://en.wikipedia.org/wiki/2022_World_Athletics_Championships_%E2%80%93_Women%27s_pole_vault"),
    ("Worlds", "2022", "https://en.wikipedia.org/wiki/2022_World_Athletics_Championships_%E2%80%93_Men%27s_long_jump"),
    ("Worlds", "2022", "https://en.wikipedia.org/wiki/2022_World_Athletics_Championships_%E2%80%93_Women%27s_long_jump"),
    ("Worlds", "2022", "https://en.wikipedia.org/wiki/2022_World_Athletics_Championships_%E2%80%93_Men%27s_triple_jump"),
    ("Worlds", "2022", "https://en.wikipedia.org/wiki/2022_World_Athletics_Championships_%E2%80%93_Women%27s_triple_jump"),
    ("Worlds", "2022", "https://en.wikipedia.org/wiki/2022_World_Athletics_Championships_%E2%80%93_Men%27s_shot_put"),
    ("Worlds", "2022", "https://en.wikipedia.org/wiki/2022_World_Athletics_Championships_%E2%80%93_Women%27s_shot_put"),
    ("Worlds", "2022", "https://en.wikipedia.org/wiki/2022_World_Athletics_Championships_%E2%80%93_Men%27s_discus_throw"),
    ("Worlds", "2022", "https://en.wikipedia.org/wiki/2022_World_Athletics_Championships_%E2%80%93_Women%27s_discus_throw"),
    ("Worlds", "2022", "https://en.wikipedia.org/wiki/2022_World_Athletics_Championships_%E2%80%93_Men%27s_javelin_throw"),
    ("Worlds", "2022", "https://en.wikipedia.org/wiki/2022_World_Athletics_Championships_%E2%80%93_Women%27s_javelin_throw"),
    # 2023 World Athletics Championships
    ("Worlds", "2023", "https://en.wikipedia.org/wiki/2023_World_Athletics_Championships_%E2%80%93_Men%27s_100_metres"),
    ("Worlds", "2023", "https://en.wikipedia.org/wiki/2023_World_Athletics_Championships_%E2%80%93_Women%27s_100_metres"),
    ("Worlds", "2023", "https://en.wikipedia.org/wiki/2023_World_Athletics_Championships_%E2%80%93_Men%27s_200_metres"),
    ("Worlds", "2023", "https://en.wikipedia.org/wiki/2023_World_Athletics_Championships_%E2%80%93_Women%27s_200_metres"),
    ("Worlds", "2023", "https://en.wikipedia.org/wiki/2023_World_Athletics_Championships_%E2%80%93_Men%27s_400_metres"),
    ("Worlds", "2023", "https://en.wikipedia.org/wiki/2023_World_Athletics_Championships_%E2%80%93_Women%27s_400_metres"),
    ("Worlds", "2023", "https://en.wikipedia.org/wiki/2023_World_Athletics_Championships_%E2%80%93_Men%27s_110_metres_hurdles"),
    ("Worlds", "2023", "https://en.wikipedia.org/wiki/2023_World_Athletics_Championships_%E2%80%93_Women%27s_100_metres_hurdles"),
    ("Worlds", "2023", "https://en.wikipedia.org/wiki/2023_World_Athletics_Championships_%E2%80%93_Men%27s_400_metres_hurdles"),
    ("Worlds", "2023", "https://en.wikipedia.org/wiki/2023_World_Athletics_Championships_%E2%80%93_Women%27s_400_metres_hurdles"),
    ("Worlds", "2023", "https://en.wikipedia.org/wiki/2023_World_Athletics_Championships_%E2%80%93_Men%27s_800_metres"),
    ("Worlds", "2023", "https://en.wikipedia.org/wiki/2023_World_Athletics_Championships_%E2%80%93_Women%27s_800_metres"),
    ("Worlds", "2023", "https://en.wikipedia.org/wiki/2023_World_Athletics_Championships_%E2%80%93_Men%27s_1500_metres"),
    ("Worlds", "2023", "https://en.wikipedia.org/wiki/2023_World_Athletics_Championships_%E2%80%93_Women%27s_1500_metres"),
    ("Worlds", "2023", "https://en.wikipedia.org/wiki/2023_World_Athletics_Championships_%E2%80%93_Men%27s_5000_metres"),
    ("Worlds", "2023", "https://en.wikipedia.org/wiki/2023_World_Athletics_Championships_%E2%80%93_Women%27s_5000_metres"),
    ("Worlds", "2023", "https://en.wikipedia.org/wiki/2023_World_Athletics_Championships_%E2%80%93_Men%27s_3000_metres_steeplechase"),
    ("Worlds", "2023", "https://en.wikipedia.org/wiki/2023_World_Athletics_Championships_%E2%80%93_Women%27s_3000_metres_steeplechase"),
    ("Worlds", "2023", "https://en.wikipedia.org/wiki/2023_World_Athletics_Championships_%E2%80%93_Men%27s_high_jump"),
    ("Worlds", "2023", "https://en.wikipedia.org/wiki/2023_World_Athletics_Championships_%E2%80%93_Women%27s_high_jump"),
    ("Worlds", "2023", "https://en.wikipedia.org/wiki/2023_World_Athletics_Championships_%E2%80%93_Men%27s_pole_vault"),
    ("Worlds", "2023", "https://en.wikipedia.org/wiki/2023_World_Athletics_Championships_%E2%80%93_Women%27s_pole_vault"),
    ("Worlds", "2023", "https://en.wikipedia.org/wiki/2023_World_Athletics_Championships_%E2%80%93_Men%27s_long_jump"),
    ("Worlds", "2023", "https://en.wikipedia.org/wiki/2023_World_Athletics_Championships_%E2%80%93_Women%27s_long_jump"),
    ("Worlds", "2023", "https://en.wikipedia.org/wiki/2023_World_Athletics_Championships_%E2%80%93_Men%27s_triple_jump"),
    ("Worlds", "2023", "https://en.wikipedia.org/wiki/2023_World_Athletics_Championships_%E2%80%93_Women%27s_triple_jump"),
    ("Worlds", "2023", "https://en.wikipedia.org/wiki/2023_World_Athletics_Championships_%E2%80%93_Men%27s_shot_put"),
    ("Worlds", "2023", "https://en.wikipedia.org/wiki/2023_World_Athletics_Championships_%E2%80%93_Women%27s_shot_put"),
    ("Worlds", "2023", "https://en.wikipedia.org/wiki/2023_World_Athletics_Championships_%E2%80%93_Men%27s_discus_throw"),
    ("Worlds", "2023", "https://en.wikipedia.org/wiki/2023_World_Athletics_Championships_%E2%80%93_Women%27s_discus_throw"),
    ("Worlds", "2023", "https://en.wikipedia.org/wiki/2023_World_Athletics_Championships_%E2%80%93_Men%27s_javelin_throw"),
    ("Worlds", "2023", "https://en.wikipedia.org/wiki/2023_World_Athletics_Championships_%E2%80%93_Women%27s_javelin_throw"),
    # 2024 Olympics
    ("Olympics", "2024", "https://en.wikipedia.org/wiki/Athletics_at_the_2024_Summer_Olympics_%E2%80%93_Men%27s_100_metres"),
    ("Olympics", "2024", "https://en.wikipedia.org/wiki/Athletics_at_the_2024_Summer_Olympics_%E2%80%93_Women%27s_100_metres"),
    ("Olympics", "2024", "https://en.wikipedia.org/wiki/Athletics_at_the_2024_Summer_Olympics_%E2%80%93_Men%27s_200_metres"),
    ("Olympics", "2024", "https://en.wikipedia.org/wiki/Athletics_at_the_2024_Summer_Olympics_%E2%80%93_Women%27s_200_metres"),
    ("Olympics", "2024", "https://en.wikipedia.org/wiki/Athletics_at_the_2024_Summer_Olympics_%E2%80%93_Men%27s_400_metres"),
    ("Olympics", "2024", "https://en.wikipedia.org/wiki/Athletics_at_the_2024_Summer_Olympics_%E2%80%93_Women%27s_400_metres"),
    ("Olympics", "2024", "https://en.wikipedia.org/wiki/Athletics_at_the_2024_Summer_Olympics_%E2%80%93_Men%27s_110_metres_hurdles"),
    ("Olympics", "2024", "https://en.wikipedia.org/wiki/Athletics_at_the_2024_Summer_Olympics_%E2%80%93_Women%27s_100_metres_hurdles"),
    ("Olympics", "2024", "https://en.wikipedia.org/wiki/Athletics_at_the_2024_Summer_Olympics_%E2%80%93_Men%27s_400_metres_hurdles"),
    ("Olympics", "2024", "https://en.wikipedia.org/wiki/Athletics_at_the_2024_Summer_Olympics_%E2%80%93_Women%27s_400_metres_hurdles"),
    ("Olympics", "2024", "https://en.wikipedia.org/wiki/Athletics_at_the_2024_Summer_Olympics_%E2%80%93_Men%27s_800_metres"),
    ("Olympics", "2024", "https://en.wikipedia.org/wiki/Athletics_at_the_2024_Summer_Olympics_%E2%80%93_Women%27s_800_metres"),
    ("Olympics", "2024", "https://en.wikipedia.org/wiki/Athletics_at_the_2024_Summer_Olympics_%E2%80%93_Men%27s_1500_metres"),
    ("Olympics", "2024", "https://en.wikipedia.org/wiki/Athletics_at_the_2024_Summer_Olympics_%E2%80%93_Women%27s_1500_metres"),
    ("Olympics", "2024", "https://en.wikipedia.org/wiki/Athletics_at_the_2024_Summer_Olympics_%E2%80%93_Men%27s_5000_metres"),
    ("Olympics", "2024", "https://en.wikipedia.org/wiki/Athletics_at_the_2024_Summer_Olympics_%E2%80%93_Women%27s_5000_metres"),
    ("Olympics", "2024", "https://en.wikipedia.org/wiki/Athletics_at_the_2024_Summer_Olympics_%E2%80%93_Men%27s_3000_metres_steeplechase"),
    ("Olympics", "2024", "https://en.wikipedia.org/wiki/Athletics_at_the_2024_Summer_Olympics_%E2%80%93_Women%27s_3000_metres_steeplechase"),
    ("Olympics", "2024", "https://en.wikipedia.org/wiki/Athletics_at_the_2024_Summer_Olympics_%E2%80%93_Men%27s_high_jump"),
    ("Olympics", "2024", "https://en.wikipedia.org/wiki/Athletics_at_the_2024_Summer_Olympics_%E2%80%93_Women%27s_high_jump"),
    ("Olympics", "2024", "https://en.wikipedia.org/wiki/Athletics_at_the_2024_Summer_Olympics_%E2%80%93_Men%27s_pole_vault"),
    ("Olympics", "2024", "https://en.wikipedia.org/wiki/Athletics_at_the_2024_Summer_Olympics_%E2%80%93_Women%27s_pole_vault"),
    ("Olympics", "2024", "https://en.wikipedia.org/wiki/Athletics_at_the_2024_Summer_Olympics_%E2%80%93_Men%27s_long_jump"),
    ("Olympics", "2024", "https://en.wikipedia.org/wiki/Athletics_at_the_2024_Summer_Olympics_%E2%80%93_Women%27s_long_jump"),
    ("Olympics", "2024", "https://en.wikipedia.org/wiki/Athletics_at_the_2024_Summer_Olympics_%E2%80%93_Men%27s_triple_jump"),
    ("Olympics", "2024", "https://en.wikipedia.org/wiki/Athletics_at_the_2024_Summer_Olympics_%E2%80%93_Women%27s_triple_jump"),
    ("Olympics", "2024", "https://en.wikipedia.org/wiki/Athletics_at_the_2024_Summer_Olympics_%E2%80%93_Men%27s_shot_put"),
    ("Olympics", "2024", "https://en.wikipedia.org/wiki/Athletics_at_the_2024_Summer_Olympics_%E2%80%93_Women%27s_shot_put"),
    ("Olympics", "2024", "https://en.wikipedia.org/wiki/Athletics_at_the_2024_Summer_Olympics_%E2%80%93_Men%27s_discus_throw"),
    ("Olympics", "2024", "https://en.wikipedia.org/wiki/Athletics_at_the_2024_Summer_Olympics_%E2%80%93_Women%27s_discus_throw"),
    ("Olympics", "2024", "https://en.wikipedia.org/wiki/Athletics_at_the_2024_Summer_Olympics_%E2%80%93_Men%27s_javelin_throw"),
    ("Olympics", "2024", "https://en.wikipedia.org/wiki/Athletics_at_the_2024_Summer_Olympics_%E2%80%93_Women%27s_javelin_throw"),
]

# Each discipline is (gender, alternatives), where an alternative is a list
# of substrings that must ALL appear. Hurdles events carry only a distance
# because `looks_like_hurdles` has already decided whether the caption is a
# hurdles race at all -- see below for why that separation is necessary.
DISCIPLINE_KEYWORDS = {
    "men_100m":    ("men", [["100 m"], ["100m"]]),
    "women_100m":  ("women", [["100 m"], ["100m"]]),
    "men_200m":    ("men", [["200 m"], ["200m"]]),
    "women_200m":  ("women", [["200 m"], ["200m"]]),
    "men_400m":    ("men", [["400 m"], ["400m"]]),
    "women_400m":  ("women", [["400 m"], ["400m"]]),
    "men_110h":    ("men", [["110"]]),
    "women_100h":  ("women", [["100"]]),
    "men_400h":    ("men", [["400"]]),
    "women_400h":  ("women", [["400"]]),
    "men_800m":    ("men", [["800 m"], ["800m"]]),
    "women_800m":  ("women", [["800 m"], ["800m"]]),
    "men_1500m":   ("men", [["1500 m"], ["1500m"]]),
    "women_1500m": ("women", [["1500 m"], ["1500m"]]),
    "men_5000m":   ("men", [["5000 m"], ["5000m"]]),
    "women_5000m": ("women", [["5000 m"], ["5000m"]]),
    "men_3000sc":  ("men", [["steeplechase"]]),
    "women_3000sc":("women", [["steeplechase"]]),
    "men_HJ":      ("men", [["high jump"]]),
    "women_HJ":    ("women", [["high jump"]]),
    "men_PV":      ("men", [["pole vault"]]),
    "women_PV":    ("women", [["pole vault"]]),
    "men_LJ":      ("men", [["long jump"]]),
    "women_LJ":    ("women", [["long jump"]]),
    "men_TJ":      ("men", [["triple jump"]]),
    "women_TJ":    ("women", [["triple jump"]]),
    "men_SP":      ("men", [["shot put"]]),
    "women_SP":    ("women", [["shot put"]]),
    "men_DT":      ("men", [["discus"]]),
    "women_DT":    ("women", [["discus"]]),
    "men_JT":      ("men", [["javelin"]]),
    "women_JT":    ("women", [["javelin"]]),
}

HURDLES_EVENTS = {"men_110h", "women_100h", "men_400h", "women_400h"}
FLAT_SPRINTS = {"men_100m", "women_100m", "men_400m", "women_400m"}

# Wikipedia writes a hurdles race two ways and the abbreviation has no
# "hurdle" in it at all: "Women's 100 Metres Hurdles" and "Women's
# 100mH(+0.6m/s)" are the same race. Deciding hurdles-or-not FIRST, from
# either spelling, is what stops the two failures this replaced:
#
#   * matching on ANY keyword meant the word "hurdle" alone satisfied
#     men_110h/women_100h, and they are tested first, so every 400m hurdles
#     race went to the sprint hurdles -- men_400h and women_400h had no rows
#     in the dataset at all, and Warholm, Benjamin, dos Santos, Bol and
#     McLaughlin-Levrone were filed as sprint hurdlers.
#   * the flat-sprint guard tested for the word "hurdle", which "100mH" does
#     not contain, so every abbreviated women's 100m hurdles race on a
#     Diamond League page was recorded as a women's 100m FLAT race.
_HURDLES_ABBREV = re.compile(r"\d+\s*m\s*h\b")


def looks_like_hurdles(caption_lower):
    return "hurdle" in caption_lower or bool(_HURDLES_ABBREV.search(caption_lower))


# WHEN a meeting happened, which meet_results.csv has never recorded.
#
# Without it every head-to-head rate is an all-time figure, and train_model
# reads the whole file for every label year -- so a 2022 final was being
# predicted from a rivalry as it stood in 2025. Measured over 10 paired seeds,
# that lookahead is worth up to 2.06 points of the reported accuracy.
#
# Written to its own file rather than into meet_results.csv, so the results
# themselves stay byte-identical to what the eighteenth session's four scraper
# fixes produced. Fold it in at the next full re-scrape.
MEET_DATES_PATH = os.path.join(H2H_DIR, "meet_dates.csv")

_MONTHS = ("January|February|March|April|May|June|July|August|September|"
           "October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec")
_DATE_TOKEN = re.compile(rf"\b(\d{{1,2}})\s+({_MONTHS})\b\.?(?:\s+(\d{{4}}))?", re.I)


def parse_meet_date(soup, year):
    """The LAST day this meeting competed, from the infobox's Dates row.

    The last day, not the first, for two different reasons that agree. A
    two-day Diamond League meeting runs half its programme on day two, and a
    championship event page reads "15 July (preliminary round & heats) 16 July
    (semi-final & final)" -- the final is the day that matters, and it is
    always last. Erring late is also the safe direction here: this date decides
    which races a LATER final is allowed to see, so a day of slack excludes a
    race rather than leaking one.

    The year is often absent from a championship page (it is in the title
    instead), so the caller's year fills in. Returns None rather than guessing
    when there is no infobox or no parsable date -- an undated meet is left out
    of a cut-off, not silently treated as having happened at some default time.
    """
    box = None
    for table in soup.find_all("table"):
        if "infobox" in " ".join(table.get("class") or []):
            box = table
            break
    if box is None:
        return None

    text = ""
    for row in box.find_all("tr"):
        head, cell = row.find("th"), row.find("td")
        if head and cell and head.get_text(" ", strip=True).lower().startswith("date"):
            text = cell.get_text(" ", strip=True)
            break
    if not text:
        return None

    matches = _DATE_TOKEN.findall(text)
    if not matches:
        return None
    day, month, found_year = matches[-1]
    try:
        return pd.Timestamp(f"{day} {month.rstrip('.')} {found_year or year}")
    except ValueError:
        return None


def fetch_page(url):
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code == 200:
            return BeautifulSoup(r.text, "html.parser")
        print(f"  HTTP {r.status_code}: {url}")
        return None
    except Exception as e:
        print(f"  Error: {e}")
        return None

def classify_table(caption):
    caption_lower = caption.lower()
    # A relay is not an individual event, and "4x100 Metres Relay" contains
    # "100 m". The athlete cell holds four names run together, so these rows
    # could never match a real athlete anyway.
    if "relay" in caption_lower:
        return None
    hurdles = looks_like_hurdles(caption_lower)
    for disc_key, (gender, alternatives) in DISCIPLINE_KEYWORDS.items():
        if gender == "men" and "women" in caption_lower:
            continue
        if gender not in caption_lower:
            continue
        # A hurdles caption is never a flat sprint, and a flat caption is
        # never a hurdles event. No other discipline is affected.
        if disc_key in HURDLES_EVENTS and not hurdles:
            continue
        if disc_key in FLAT_SPRINTS and hurdles:
            continue
        if any(all(term in caption_lower for term in alt) for alt in alternatives):
            return disc_key
    return None

# --- which race a table belongs to ---------------------------------------
#
# Wikipedia lays a meeting out as one table per race, and which race a table
# belongs to is carried entirely by the surrounding headings -- never inside
# the table. Without reading them, heats, semis and the final collapse into
# one bucket per meeting, which is the data defect behind HANDOFF 0o.
#
# Two different races can also share a label, so the label has to be a PATH,
# not the nearest heading: an Olympic sprint page has three separate "Heat
# 1"s (first round, repechage, semi-final), and a Diamond League page runs
# "Men's 1500 Metres" twice -- once as the Diamond Discipline and once as a
# national race, which is how Jakob Ingebrigtsen ended up sharing a "meet"
# with the Swiss national 1500m field.

_HEADING_LEVELS = {"h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}

# Positions 1-3 are rendered as a medal template with NO text in the cell.
_MEDAL_ALT_PLACE = {
    "1st place, gold medalist(s)":   1,
    "2nd place, silver medalist(s)": 2,
    "3rd place, bronze medalist(s)": 3,
}
_MEDAL_SORT_PLACE = {"01": 1, "02": 2, "03": 3}

# Header names for the column holding the actual performance. Read by name
# because the old `athlete_col + 2` offset silently landed on Country for
# every table carrying an Age column -- 1382 rows held a country name where
# a mark belonged, so the DNF/DNS/NM/DQ exclusion below was testing the
# wrong column and never fired on those pages.
_MARK_HEADERS = ("time", "mark", "result", "distance", "performance", "best")

# The column saying which heat/group an athlete was in. A championship heats
# table is one table for ALL heats, ranked across them by time, so without
# this column rank 1 (heat 1) reads as beating rank 4 (heat 4) -- a time
# comparison, not a head-to-head. NOT "round": field-event tables have a
# "Round" column meaning the attempt a mark came on.
_HEAT_HEADERS = ("heat", "group")


def heading_path(table):
    """The section headings above `table`, outermost first.

    Walks backwards and keeps a heading only if it is SHALLOWER than the
    last one kept, which is what makes this a real section path instead of
    "the nearest few headings". The Final table on an Olympic page is
    preceded in document order by `h3 Final` and, before that, by the
    semi-finals' `h4 Heat 3`; taking the nearest h4 would file the final
    under a heat."""
    path = []
    level = 99
    node = table
    while True:
        node = node.find_previous(list(_HEADING_LEVELS))
        if node is None:
            break
        node_level = _HEADING_LEVELS[node.name]
        if node_level < level:
            text = node.get_text(strip=True)
            if text:
                path.append(text)
            level = node_level
    return list(reversed(path))


def race_label(table, caption):
    """A label unique to one race within a meeting page."""
    parts = heading_path(table)
    if caption and caption not in parts:
        parts.append(caption)
    return " / ".join(parts)


def parse_place(cell):
    """The finishing position in a results cell.

    The podium is the reason this is not just `int(cell.text)`: Wikipedia
    renders positions 1-3 as a medal template with no text at all, so
    reading the text dropped every podium in the dataset -- 2825 of 13040
    rows (21.7%), and the most valuable rows there are. The template leaves
    two machine-readable traces: the `data-sort-value` that makes the column
    sort correctly, and the medal image's alt text. Only the three medal
    values are accepted, so a DNF's empty cell stays empty."""
    digits = re.sub(r"[^\d]", "", cell.get_text(strip=True))
    if digits:
        return int(digits)

    sortable = cell.find(attrs={"data-sort-value": True})
    if sortable:
        key = re.sub(r"[^\d]", "", sortable["data-sort-value"])
        if key in _MEDAL_SORT_PLACE:
            return _MEDAL_SORT_PLACE[key]

    img = cell.find("img")
    if img and img.get("alt") in _MEDAL_ALT_PLACE:
        return _MEDAL_ALT_PLACE[img["alt"]]
    return None


def parse_results_table(table, disc_key, meet_name, year, competition_level, race=""):
    rows = []
    trs = table.find_all("tr")
    if not trs:
        return rows

    # Detect column layout from header
    headers = [th.get_text(strip=True).lower() for th in trs[0].find_all(["th", "td"])]

    # Find athlete, place, mark and heat column indices
    athlete_col = None
    place_col = None
    mark_col = None
    heat_col = None
    for i, h in enumerate(headers):
        if h in ["athlete", "name"]:
            athlete_col = i
        if h in ["rank", "place", "#"]:
            place_col = i
        if mark_col is None and h in _MARK_HEADERS:
            mark_col = i
        if heat_col is None and h in _HEAT_HEADERS:
            heat_col = i

    if athlete_col is None:
        return rows
    if mark_col is None:
        mark_col = athlete_col + 2
    # Known and left alone: a field-event final spans "Round" over six
    # attempt columns, so the header row is shorter than the data rows and
    # `mark` lands on the third attempt rather than the result. `mark` is
    # not read by anything downstream, and an attempt cell is a distance or
    # "x" -- never DNF/DNS/NM/DQ -- so the exclusion above cannot misfire
    # on it. Only the stored string is wrong, and only for those tables.

    for tr in trs[1:]:
        cells = tr.find_all(["td", "th"])
        if len(cells) <= athlete_col:
            continue

        athlete_text = cells[athlete_col].get_text(strip=True)
        mark_text = cells[mark_col].get_text(strip=True) if len(cells) > mark_col else ""
        heat_text = ""
        if heat_col is not None and len(cells) > heat_col:
            heat_text = cells[heat_col].get_text(strip=True)

        if any(x in mark_text.upper() for x in ["DNF", "DNS", "NM", "DQ"]):
            continue
        if not athlete_text or athlete_text in ["Athlete", "Name"]:
            continue
        # A field-event table has a SECOND header row numbering the six
        # attempts, and a sprint table ends with a wind footer. Both parse
        # as athletes -- "2" and "3" took first place in 40 field-event
        # tables, beating everyone in them, and 56 rows were athletes
        # called "Wind:(+0.4m/s)". A real name has a letter in it.
        if not re.search(r"[A-Za-z]", athlete_text) or athlete_text.lower().startswith("wind"):
            continue

        place = None
        if place_col is not None and len(cells) > place_col:
            place = parse_place(cells[place_col])

        rows.append({
            "meet":              meet_name,
            "year":              year,
            "competition_level": competition_level,
            "discipline":        disc_key,
            "race":              race,
            "heat":              heat_text,
            "place":             place,
            "athlete":           athlete_text,
            "mark":              mark_text,
        })
    return rows

def scrape_meet(url, competition_level, year):
    soup = fetch_page(url)
    if not soup:
        return []
    return scrape_soup(soup, url, competition_level, year)


def meet_name_of(soup, url):
    """The `meet` value every row of meet_results.csv is keyed by.

    Shared with the date pass so the two files cannot drift apart on a title
    Wikipedia has since edited -- a date keyed to a name the results no longer
    use would join to nothing and read as "this meeting has no date"."""
    h1 = soup.find("h1")
    return h1.get_text(strip=True) if h1 else url


def scrape_soup(soup, url, competition_level, year):
    """The parsing half of scrape_meet, split out so it can be tested
    against saved HTML instead of the live encyclopedia."""
    meet_name = meet_name_of(soup, url)
    results = []

    # For individual event pages (Worlds/Olympics), classify from page title
    page_disc = classify_table(meet_name)

    tables = soup.find_all("table", class_="wikitable")
    for table in tables:
        cap_tag = table.find("caption")
        cap_text = cap_tag.get_text(strip=True) if cap_tag else ""
        caption = cap_text
        if not cap_tag:
            prev = table.find_previous(["h2", "h3", "h4"])
            if prev:
                caption = prev.get_text(strip=True)

        race = race_label(table, cap_text)
        # Use page-level discipline if we have it, otherwise classify from
        # the caption -- falling back to the full section path, which is the
        # only place the gender appears when a meeting page splits into
        # "Men"/"Women" sections and heads each table just "400 metres
        # hurdles". The fallback can only ADD rows: it runs when the caption
        # alone classified as nothing.
        disc_key = page_disc or classify_table(caption) or classify_table(race)
        if not disc_key:
            continue

        rows = parse_results_table(table, disc_key, meet_name, year, competition_level,
                                   race=race)
        results.extend(rows)

    return results

def save_meet_dates(dates):
    """dates: list of (meet, year, competition_level, Timestamp or None)."""
    df = pd.DataFrame(dates, columns=["meet", "year", "competition_level", "date"])
    df = df.dropna(subset=["meet"]).drop_duplicates(subset=["meet"], keep="last")
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df.to_csv(MEET_DATES_PATH, index=False)
    missing = df[df["date"].isna()]
    print(f"\nSaved {len(df)} meets to {MEET_DATES_PATH}, "
          f"{len(df) - len(missing)} dated")
    # Named, never counted. An undated meet is dropped from every cut-off, so
    # a silent one is signal quietly going missing.
    for name in missing["meet"]:
        print(f"  NO DATE: {name}")


if __name__ == "__main__":
    # --dates-only fetches the same pages but records just the meet's date,
    # leaving meet_results.csv untouched. That is the safe way to add the
    # column: a full re-scrape would rewrite results the eighteenth session
    # spent four bug fixes getting right.
    dates_only = "--dates-only" in sys.argv

    all_results, all_dates = [], []
    total = len(MEET_PAGES)

    for i, (competition_level, year, url) in enumerate(MEET_PAGES):
        page_name = url.split("/")[-1].replace("_", " ").replace("%C3%BC", "u")[:50]
        print(f"[{i+1}/{total}] {competition_level} {year} — {page_name}")
        soup = fetch_page(url)
        if soup is not None:
            name = meet_name_of(soup, url)
            when = parse_meet_date(soup, year)
            all_dates.append((name, year, competition_level, when))
            print(f"  -> {when.date() if when is not None else 'NO DATE'}")
            if not dates_only:
                rows = scrape_soup(soup, url, competition_level, year)
                if rows:
                    print(f"  -> {len(rows)} rows")
                all_results.extend(rows)
        time.sleep(0.5)

    if all_dates:
        save_meet_dates(all_dates)

    if dates_only:
        print("\n[--dates-only — meet_results.csv not touched]")
    elif all_results:
        df = pd.DataFrame(all_results)
        out_path = os.path.join(H2H_DIR, "meet_results.csv")
        df.to_csv(out_path, index=False)
        print(f"\nSaved {len(df)} rows to {out_path}")
        print("\nRows per discipline:")
        for disc in sorted(df["discipline"].unique()):
            n = len(df[df["discipline"] == disc])
            meets = df[df["discipline"] == disc]["meet"].nunique()
            print(f"  {disc}: {n} rows across {meets} meets")
    else:
        print("No results scraped.")