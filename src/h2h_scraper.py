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

DISCIPLINE_KEYWORDS = {
    "men_100m":    ("men", ["100 m", "100m"]),
    "women_100m":  ("women", ["100 m", "100m"]),
    "men_200m":    ("men", ["200 m", "200m"]),
    "women_200m":  ("women", ["200 m", "200m"]),
    "men_400m":    ("men", ["400 m", "400m"]),
    "women_400m":  ("women", ["400 m", "400m"]),
    "men_110h":    ("men", ["110", "hurdle"]),
    "women_100h":  ("women", ["100", "hurdle"]),
    "men_400h":    ("men", ["400", "hurdle"]),
    "women_400h":  ("women", ["400", "hurdle"]),
    "men_800m":    ("men", ["800 m", "800m"]),
    "women_800m":  ("women", ["800 m", "800m"]),
    "men_1500m":   ("men", ["1500 m", "1500m"]),
    "women_1500m": ("women", ["1500 m", "1500m"]),
    "men_5000m":   ("men", ["5000 m", "5000m"]),
    "women_5000m": ("women", ["5000 m", "5000m"]),
    "men_3000sc":  ("men", ["steeplechase"]),
    "women_3000sc":("women", ["steeplechase"]),
    "men_HJ":      ("men", ["high jump"]),
    "women_HJ":    ("women", ["high jump"]),
    "men_PV":      ("men", ["pole vault"]),
    "women_PV":    ("women", ["pole vault"]),
    "men_LJ":      ("men", ["long jump"]),
    "women_LJ":    ("women", ["long jump"]),
    "men_TJ":      ("men", ["triple jump"]),
    "women_TJ":    ("women", ["triple jump"]),
    "men_SP":      ("men", ["shot put"]),
    "women_SP":    ("women", ["shot put"]),
    "men_DT":      ("men", ["discus"]),
    "women_DT":    ("women", ["discus"]),
    "men_JT":      ("men", ["javelin"]),
    "women_JT":    ("women", ["javelin"]),
}

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
    for disc_key, (gender, keywords) in DISCIPLINE_KEYWORDS.items():
        if gender == "men" and "women" in caption_lower:
            continue
        if gender not in caption_lower:
            continue
        if any(k in caption_lower for k in keywords):
            if disc_key in ("men_100m", "women_100m", "men_400m", "women_400m") and "hurdle" in caption_lower:
                continue
            return disc_key
    return None

def parse_results_table(table, disc_key, meet_name, year, competition_level):
    rows = []
    trs = table.find_all("tr")
    if not trs:
        return rows

    # Detect column layout from header
    headers = [th.get_text(strip=True).lower() for th in trs[0].find_all(["th", "td"])]
    
    # Find athlete and place column indices
    athlete_col = None
    place_col = None
    for i, h in enumerate(headers):
        if h in ["athlete", "name"]:
            athlete_col = i
        if h in ["rank", "place", "#"]:
            place_col = i
    
    if athlete_col is None:
        return rows

    for tr in trs[1:]:
        cells = tr.find_all(["td", "th"])
        if len(cells) <= athlete_col:
            continue
        
        place_text = cells[place_col].get_text(strip=True) if place_col is not None else ""
        athlete_text = cells[athlete_col].get_text(strip=True)
        mark_col = athlete_col + 2
        mark_text = cells[mark_col].get_text(strip=True) if len(cells) > mark_col else ""

        if any(x in mark_text.upper() for x in ["DNF", "DNS", "NM", "DQ"]):
            continue
        if not athlete_text or athlete_text in ["Athlete", "Name"]:
            continue

        try:
            place = int(re.sub(r"[^\d]", "", place_text)) if place_text.strip() else None
        except:
            place = None

        rows.append({
            "meet":              meet_name,
            "year":              year,
            "competition_level": competition_level,
            "discipline":        disc_key,
            "place":             place,
            "athlete":           athlete_text,
            "mark":              mark_text,
        })
    return rows

def scrape_meet(url, competition_level, year):
    soup = fetch_page(url)
    if not soup:
        return []

    meet_name = soup.find("h1").get_text(strip=True) if soup.find("h1") else url
    results = []
    
    # For individual event pages (Worlds/Olympics), classify from page title
    page_disc = classify_table(meet_name)
    
    tables = soup.find_all("table", class_="wikitable")
    for table in tables:
        caption = ""
        cap_tag = table.find("caption")
        if cap_tag:
            caption = cap_tag.get_text(strip=True)
        else:
            prev = table.find_previous(["h2", "h3", "h4"])
            if prev:
                caption = prev.get_text(strip=True)

        # Use page-level discipline if we have it, otherwise classify from caption
        disc_key = page_disc if page_disc else classify_table(caption)
        if not disc_key:
            continue

        rows = parse_results_table(table, disc_key, meet_name, year, competition_level)
        results.extend(rows)

    return results

if __name__ == "__main__":
    all_results = []
    total = len(MEET_PAGES)

    for i, (competition_level, year, url) in enumerate(MEET_PAGES):
        page_name = url.split("/")[-1].replace("_", " ").replace("%C3%BC", "u")[:50]
        print(f"[{i+1}/{total}] {competition_level} {year} — {page_name}")
        rows = scrape_meet(url, competition_level, year)
        if rows:
            print(f"  -> {len(rows)} rows")
        all_results.extend(rows)
        time.sleep(0.5)

    if all_results:
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