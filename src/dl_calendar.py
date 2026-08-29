"""
dl_calendar.py -- World Athletics' own Diamond League calendar for a season,
and a check that api.py's hand-maintained MEETS list still matches it.

Why this exists: MEETS is typed by hand and is rendered to readers on the
Schedule page as fact. On 2026-08-29 it had drifted badly without anyone
noticing -- it opened the season with "08 May Doha", a meeting that never
existed (Doha ran 19 Jun; the opener was 16 May in Shaoxing/Keqiao, which
the list called "Shanghai"), and Paris and Eugene were each a day or two
out. Nothing in the pipeline caught it because MEETS only feeds display and
the meetings-remaining arithmetic, and by late season every date is in the
past either way -- the errors were invisible precisely when they no longer
changed a number.

It also answers a structural question that is easy to get wrong: the
Diamond League Final is NOT always a single meeting. In 2018 and 2019 it
was split across Zürich and Brussels, two meetings both carrying ranking
category "DF". `rankingCategory` per meeting is the authority on which
meetings are the Final, so this module reports it rather than assuming the
last meeting of the season is it.

Usage:
    python src/dl_calendar.py                # print WA's calendar + diff vs api.MEETS
    python src/dl_calendar.py --year 2019    # any season
    python src/dl_calendar.py --snapshot     # write the test fixture for --year
"""
import argparse
import io
import json
import os
import sys

# Guarded for the same reason every other module in src/ guards it -- see
# HANDOFF item 0i. After the first wrap the encoding is already utf-8.
if not (sys.stdout.encoding or "").lower().startswith("utf"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.dirname(__file__))
import dl_final_results_scraper as dlr  # noqa: E402 -- reuse graphql()/CALENDAR_QUERY
from season_results_scraper import RESULTS_QUERY  # noqa: E402

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures")

# The Wanda Diamond League competition group, same id season_results_scraper
# and dl_final_results_scraper query.
DL_COMPETITION_GROUP = 627

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def day_month(iso_date):
    """"2026-06-19" -> "19 Jun", the format MEETS entries are written in."""
    if not iso_date:
        return None
    y, m, d = str(iso_date)[:10].split("-")
    return f"{int(d):02d} {MONTHS[int(m) - 1]}"


def city_of(venue):
    """The city out of "Stadio Olimpico, Roma (ITA)". Best-effort only --
    it is recorded for eyeballing a diff, never asserted on, because WA's
    venue strings carry US state codes ("Eugene, OR (USA)") and dual city
    names ("Shaoxing/Keqiao") that no single rule tidies correctly."""
    if not venue:
        return None
    parts = [p.strip() for p in str(venue).split(",")]
    if len(parts) < 2:
        return parts[0]
    city = parts[1]
    return city.split("(")[0].strip()


def fetch_calendar(year, with_days=True):
    """WA's Diamond League meetings for `year`, in calendar order.

    `days` is each day the meeting was actually contested, which is what
    makes a two-day meeting's end date real rather than assumed. It needs
    one extra request per meeting and is only available once a meeting has
    results, so it is skipped for anything still in the future."""
    data = dlr.graphql(
        "getMinisiteCalendarEvents",
        {"season": str(year), "competitionGroupId": DL_COMPETITION_GROUP,
         "competitionSubgroupId": 0},
        dlr.CALENDAR_QUERY,
    )
    meetings = []
    for m in data["getMinisiteCalendarEvents"]["results"]:
        entry = {
            "id": m["id"],
            "name": m["name"],
            "venue": m.get("venue"),
            "city": city_of(m.get("venue")),
            "startDate": str(m["startDate"])[:10] if m.get("startDate") else None,
            "rankingCategory": m.get("rankingCategory"),
            "days": None,
        }
        if with_days:
            entry["days"] = fetch_days(m["id"])
        meetings.append(entry)
    meetings.sort(key=lambda m: m["startDate"] or "")
    return meetings


def fetch_days(competition_id):
    """The dates a meeting was contested on, or None if it has no results
    yet (a future meeting errors rather than returning an empty list)."""
    try:
        probe = dlr.graphql(
            "getCalendarCompetitionResults",
            {"competitionId": competition_id, "day": None, "eventId": None},
            RESULTS_QUERY,
        )["getCalendarCompetitionResults"]
    except Exception:
        return None
    days = (probe.get("options") or {}).get("days") or []
    return [d["date"] for d in days if d.get("date")]


def fixture_path(year):
    return os.path.join(FIXTURE_DIR, f"wa_dl_calendar_{year}.json")


def write_snapshot(year, meetings):
    path = fixture_path(year)
    os.makedirs(FIXTURE_DIR, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"season": year, "meetings": meetings}, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return path


def load_snapshot(year):
    with open(fixture_path(year), encoding="utf-8") as f:
        return json.load(f)["meetings"]


def diff_against_meets(meetings, meets):
    """Every way api.py's MEETS disagrees with WA's calendar, as a list of
    human-readable strings. Empty means they agree."""
    problems = []
    if len(meets) != len(meetings):
        problems.append(f"count: MEETS has {len(meets)}, WA lists {len(meetings)}")

    for i, wa in enumerate(meetings):
        if i >= len(meets):
            problems.append(f"missing from MEETS: {day_month(wa['startDate'])} {wa['name']}")
            continue
        mine = meets[i]
        expected = day_month(wa["startDate"])
        if mine.get("date") != expected:
            problems.append(
                f"#{i + 1} {mine.get('city')}: MEETS says {mine.get('date')!r}, "
                f"WA says {expected!r} ({wa['name']})"
            )
        # A meeting's real last day, when WA has results to say so.
        days = wa.get("days") or []
        expected_end = day_month(_iso_of(days[-1])) if len(days) > 1 else None
        if mine.get("dateEnd") != expected_end:
            problems.append(
                f"#{i + 1} {mine.get('city')}: MEETS dateEnd {mine.get('dateEnd')!r}, "
                f"WA contested {len(days)} day(s) ending {expected_end!r}"
            )
        is_final = bool(mine.get("final"))
        wa_final = wa.get("rankingCategory") == "DF"
        if is_final != wa_final:
            problems.append(
                f"#{i + 1} {mine.get('city')}: MEETS final={is_final}, "
                f"WA rankingCategory={wa.get('rankingCategory')!r}"
            )
    return problems


def _iso_of(wa_date):
    """WA's per-day dates come back as "27 AUG 2026", not ISO."""
    if not wa_date:
        return None
    try:
        d, mon, y = str(wa_date).split()
        return f"{y}-{MONTHS.index(mon.title()) + 1:02d}-{int(d):02d}"
    except (ValueError, IndexError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--snapshot", action="store_true",
                        help="write tests/fixtures/wa_dl_calendar_{year}.json")
    args = parser.parse_args()

    meetings = fetch_calendar(args.year)
    print(f"World Athletics lists {len(meetings)} Diamond League meetings for {args.year}:\n")
    for m in meetings:
        days = m.get("days") or []
        span = f"{len(days)}d" if days else "--"
        print(f"  {m['startDate']}  {m['rankingCategory'] or '?':<3} {span:<3} "
              f"{str(m['city']):<18} {m['name']}")

    finals = [m for m in meetings if m["rankingCategory"] == "DF"]
    print(f"\nFinal: {len(finals)} meeting(s) tagged DF -- "
          + ", ".join(f"{m['city']} ({day_month(m['startDate'])})" for m in finals))
    if len(finals) > 1:
        print("  NOTE: a SPLIT Final. Every DF meeting must carry final=True in api.MEETS,")
        print("  or its first leg is scored as a qualifying meeting.")

    if args.snapshot:
        print(f"\nWrote {write_snapshot(args.year, meetings)}")
        return

    if args.year == 2026:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        import api  # noqa: E402 -- safe, api.py guards app.run() behind __main__
        problems = diff_against_meets(meetings, api.MEETS)
        print(f"\napi.MEETS vs WA: {len(problems)} disagreement(s)")
        for p in problems:
            print(f"  - {p}")


if __name__ == "__main__":
    main()
