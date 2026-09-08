"""
country_index.py -- builds data/countries.json, the country view of everything
we already scrape.

Why this exists: the site had no country entity at all. Every page was about an
athlete or a discipline, so two things users asked for had nowhere to live --
"show me everyone from Jamaica", and the Ultimate's two mixed relays, which are
national teams rather than individuals.

What it contains, per country:
  * every athlete of that nationality in this season's toplists, with their
    event, mark and World Athletics Results Score
  * their direct places at the Ultimate (Olympic/World champions, Diamond
    League Final winners), read from data/ultimate/event.json
  * that nation's mixed relay teams and whether they qualified

Ordering, deliberately: athletes are sorted by **Results Score**, never by the
model's podium probability. Those probabilities are per-discipline -- they sum
to anywhere between 31 and 320 depending on the event -- so they rank athletes
WITHIN an event and are meaningless across events. A country page is the one
view that inevitably compares a shot putter with a 400m runner, so it uses the
one figure World Athletics designed for exactly that.

Country names come from WA's own getCountries (260 nations, keyed by the same
3-letter code the toplists use), so nothing here is a hand-typed lookup table.

Usage:
    python src/country_index.py
Writes data/countries.json.
"""
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import api  # noqa: E402
import dl_final_results_scraper as dlr  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
OUT_PATH = os.path.join(DATA, "countries.json")
ULTIMATE_PATH = os.path.join(DATA, "ultimate", "event.json")

COUNTRIES_QUERY = """query getCountries {
  getCountries { id countryName areaName countryNameUrlSlug isValid }
}"""


def country_names():
    """{IOC code: {name, area, slug}} from World Athletics' own country list.

    Falls back to an empty map on failure -- a country then shows under its
    3-letter code, which is still usable, rather than the build failing."""
    try:
        rows = dlr.graphql("getCountries", {}, COUNTRIES_QUERY)["getCountries"]
    except Exception as e:
        print(f"  country list fetch failed ({str(e)[:60]}) -- using codes only")
        return {}
    return {
        r["id"]: {
            "name": r["countryName"],
            "area": r["areaName"],
            "slug": r["countryNameUrlSlug"],
        }
        for r in rows
        if r.get("id")
    }


def athletes_by_country():
    """{IOC code: [athlete, ...]} from this season's toplists -- every ranked
    athlete, not just the ones in a projected field."""
    out = {}
    for disc_key, label in api.DISC_LABELS.items():
        path = os.path.join(api.RAW_DIR, f"{disc_key}_{api.MEETS_YEAR}.csv")
        if not os.path.exists(path):
            continue
        raw = pd.read_csv(path).dropna(subset=["Competitor"])
        seen = set()
        for _, r in raw.iterrows():
            name = str(r["Competitor"])
            if name in seen:            # an athlete's best row only
                continue
            seen.add(name)
            # WA's export leaves the nationality header blank; read by position,
            # the same way toplist_meta and live_fetcher do.
            nat = str(r.iloc[5]).strip() if pd.notna(r.iloc[5]) else None
            if not nat:
                continue
            try:
                score = int(r["Results Score"]) if pd.notna(r.get("Results Score")) else None
            except (ValueError, TypeError):
                score = None
            rank = r.get("Rank")
            out.setdefault(nat, []).append({
                "name": name,
                "discKey": disc_key,
                "disc": label,
                "mark": str(r["Mark"]) if pd.notna(r.get("Mark")) else None,
                "score": score,
                "worldRank": int(rank) if pd.notna(rank) else None,
                "profileUrl": str(r["ProfileURL"]) if pd.notna(r.get("ProfileURL")) else None,
            })
    for nat, rows in out.items():
        # Results Score across events; unscored athletes last rather than first.
        rows.sort(key=lambda a: (a["score"] is None, -(a["score"] or 0), a["disc"]))
    return out


def ultimate_by_country():
    """{IOC code: {qualifiers: [...], relays: [...]}} from the Ultimate build."""
    if not os.path.exists(ULTIMATE_PATH):
        return {}
    with open(ULTIMATE_PATH, encoding="utf-8") as f:
        ev = json.load(f)
    out = {}
    for q in ev.get("namedQualifiers") or []:
        nat = q.get("nationality")
        if nat:
            out.setdefault(nat, {}).setdefault("qualifiers", []).append({
                "route": q.get("route"),
                "name": q.get("linkName") or q.get("name"),
                "disciplineLabel": q.get("disciplineLabel"),
                "discKey": q.get("discKey"),
            })
    for q in ev.get("dlFinalQualifiers") or []:
        nat = q.get("nationality")
        if nat:
            out.setdefault(nat, {}).setdefault("qualifiers", []).append({
                "route": "dl",
                "name": q.get("athlete_name"),
                "disciplineLabel": None,
                "discKey": q.get("discipline"),
            })
    for relay in ev.get("relays") or []:
        ran = set()
        for team in relay.get("teams") or []:
            nat = team.get("nationality")
            if not nat:
                continue
            ran.add(nat)
            out.setdefault(nat, {}).setdefault("relays", []).append({
                "event": relay.get("event"),
                "place": team.get("place"),
                "mark": team.get("mark"),
                "qualified": team.get("qualified"),
                "host": False,
                # The athletes this nation actually ran, most-used first.
                "squad": team.get("squad") or [],
            })
        # The host holds a place without having raced the qualifier, so it has
        # no place or time. Without this a host nation shows no relay at all,
        # which is the opposite of true -- Hungary have both.
        for nat in relay.get("wildcards") or []:
            if nat in ran:
                continue
            out.setdefault(nat, {}).setdefault("relays", []).append({
                "event": relay.get("event"),
                "place": None,
                "mark": None,
                "qualified": True,
                "host": True,
                # A host place was not raced for, so there is no squad on record.
                "squad": [],
            })
    return out


def build():
    names = country_names()
    athletes = athletes_by_country()
    ultimate = ultimate_by_country()
    countries = {}
    for code, rows in athletes.items():
        meta = names.get(code) or {}
        extra = ultimate.get(code) or {}
        countries[code] = {
            "code": code,
            "name": meta.get("name") or code,
            "area": meta.get("area"),
            "athleteCount": len(rows),
            "disciplineCount": len({a["discKey"] for a in rows}),
            # The nation's best Results Score, so countries can be ranked
            # against each other without inventing a medal-table style total.
            "topScore": max((a["score"] or 0) for a in rows) if rows else 0,
            "athletes": rows,
            "ultimateQualifiers": extra.get("qualifiers") or [],
            "relays": extra.get("relays") or [],
        }
    return countries


if __name__ == "__main__":
    print("=== Building the country index ===")
    countries = build()
    os.makedirs(DATA, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(countries, f, ensure_ascii=False)
    named = sum(1 for c in countries.values() if c["name"] != c["code"])
    athletes = sum(c["athleteCount"] for c in countries.values())
    withq = sum(1 for c in countries.values() if c["ultimateQualifiers"])
    withr = sum(1 for c in countries.values() if c["relays"])
    print(f"  {len(countries)} countries, {athletes} athletes")
    print(f"  {named} resolved to a real country name; "
          f"{withq} have an Ultimate qualifier; {withr} have a relay team")
    print(f"  Saved -> {os.path.abspath(OUT_PATH)}")
