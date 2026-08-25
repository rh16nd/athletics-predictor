"""Tests for src/live_fetcher.py's Diamond League standings parsing.

scrape_dl_standings() used to keep only the athlete name from each row and
truncate the table at the qualification cut, which meant the two things a
qualification race is made of -- the points, and everyone still chasing from
below the line -- were scraped and then thrown away on every run (HANDOFF
item 0l). The row parser is split out here so that behaviour is testable
against real WA markup without a live Selenium session; the driver-driven
half of scrape_dl_standings() still isn't exercised.

The markup below is a trimmed copy of the real page's structure: five cells
per row [rank, athlete, country, events, points], a header row of <th>s
whose text renders uppercase via CSS, and the discipline heading in a
sibling div above the table rather than in a <caption>.
"""
import bs4

import live_fetcher as lf


def soup_table(html):
    return bs4.BeautifulSoup(html, "html.parser").find("table")


MEN_100M_HTML = """
<div class="TableCollapsible_tablePanel__2_tAe"><div>100 Metres</div></div>
<div class="TableCollapsible_tableWrap__3_10K"><table>
  <tr><th></th><th>Athlete</th><th>Country</th><th>Events</th><th>Points</th></tr>
  <tr><td>1</td><td><a href="/athletes/athlete=1">Oblique SEVILLE</a></td><td>JAM</td><td>3</td><td>23</td></tr>
  <tr><td>2</td><td>Gift LEOTLELA</td><td>RSA</td><td>4</td><td>23</td></tr>
  <tr><td>3</td><td>Trayvon BROMELL</td><td>USA</td><td>3</td><td>19</td></tr>
  <tr><td>4</td><td>Jordan ANTHONY</td><td>USA</td><td>2</td><td>15</td></tr>
  <tr><td>5</td><td>Ferdinand OMANYALA</td><td>KEN</td><td>4</td><td>14</td></tr>
  <tr><td>6</td><td>Kenneth BEDNAREK</td><td>USA</td><td>2</td><td>12</td></tr>
  <tr><td>7</td><td>Emmanuel ESEME</td><td>CMR</td><td>3</td><td>11</td></tr>
  <tr><td>8</td><td>Akani SIMBINE</td><td>RSA</td><td>2</td><td>10</td></tr>
  <tr><td>9</td><td>Ackeem BLAKE</td><td>JAM</td><td>2</td><td>9</td></tr>
  <tr><td>10</td><td>Christian COLEMAN</td><td>USA</td><td>1</td><td>7</td></tr>
</table></div>
"""


def test_parses_the_points_column():
    rows = lf.parse_standings_table(soup_table(MEN_100M_HTML))
    assert rows[0] == {
        "rank": 1, "name": "Oblique SEVILLE", "country": "JAM",
        "events": 3, "points": 23,
    }


def test_keeps_every_row_not_just_the_qualifying_eight():
    """The whole point of item 0l: rank 9 and 10 are exactly the athletes a
    "can they still qualify?" question is about, and the old scraper cut
    them off at get_qual_limit()."""
    rows = lf.parse_standings_table(soup_table(MEN_100M_HTML))
    assert len(rows) == 10 > lf.get_qual_limit("men_100m")
    assert rows[-1]["name"] == "Christian COLEMAN"


def test_drops_the_header_row():
    rows = lf.parse_standings_table(soup_table(MEN_100M_HTML))
    assert all(r["name"] != "Athlete" for r in rows)
    assert [r["rank"] for r in rows] == list(range(1, 11))


def test_tolerates_a_missing_points_cell():
    """A blank cell must null one field, not drop the athlete: someone with
    no points yet is still in the table and still relevant to the race."""
    rows = lf.parse_standings_table(soup_table("""
    <table><tr><td>4</td><td>Alpha SPEEDY</td><td>USA</td><td></td><td>-</td></tr></table>
    """))
    assert rows == [{"rank": 4, "name": "Alpha SPEEDY", "country": "USA",
                     "events": None, "points": None}]


def test_ignores_rows_that_are_not_standings_rows():
    rows = lf.parse_standings_table(soup_table("""
    <table>
      <tr><td colspan="5">No standings available</td></tr>
      <tr><td>1</td><td>Alpha SPEEDY</td><td>USA</td><td>2</td><td>16</td></tr>
    </table>
    """))
    assert [r["name"] for r in rows] == ["Alpha SPEEDY"]


def test_reads_the_discipline_heading_above_the_table():
    table = bs4.BeautifulSoup(MEN_100M_HTML, "html.parser").find("table")
    assert lf.standings_table_label(table) == "100 Metres"


# ---- label_matches_key: guards the positional table read ----

def test_hurdles_heading_is_not_the_flat_event():
    """"400 Metres" is a substring of "400 Metres Hurdles", so a plain
    containment check would call the hurdles table a flat-400 match -- which
    is the exact shape of the table-order bug this guard exists for."""
    assert not lf.label_matches_key("400 Metres Hurdles", "men_400m")
    assert lf.label_matches_key("400 Metres Hurdles", "men_400h")


def test_wa_lists_the_5000m_as_a_combined_heading():
    assert lf.label_matches_key("3000/5000 Metres", "men_5000m")


def test_a_missing_heading_is_not_treated_as_a_mismatch():
    assert lf.label_matches_key(None, "men_100m")


def test_a_genuinely_wrong_heading_is_caught():
    assert not lf.label_matches_key("Pole Vault", "men_HJ")
    assert lf.label_matches_key("Pole Vault", "men_PV")


# ---- ordering: WA's rank column vs the order the rows render in ----

TIED_HTML = """
<table>
  <tr><th></th><th>Athlete</th><th>Country</th><th>Events</th><th>Points</th></tr>
  <tr><td>1</td><td>Yared NUGUSE</td><td>USA</td><td>5</td><td>37</td></tr>
  <tr><td>10</td><td>Josh KERR</td><td>GBR</td><td>2</td><td>8</td></tr>
  <tr><td>11</td><td>Robert FARKEN</td><td>GER</td><td>3</td><td>8</td></tr>
  <tr><td>9</td><td>Azeddine HABZ</td><td>FRA</td><td>3</td><td>8</td></tr>
</table>
"""


def test_the_rows_do_not_arrive_in_rank_order():
    """Real 2026-08-25 shape from the men's 1500m: three athletes tied on 8
    points render as ranks 10, 11, 9. The parser reports what the page
    says -- the ordering is the caller's job."""
    rows = lf.parse_standings_table(soup_table(TIED_HTML))
    assert [r["rank"] for r in rows] == [1, 10, 11, 9]


def test_truncating_by_row_order_would_keep_the_wrong_athlete():
    """Pins the bug this ordering exists to prevent: taking the first N rows
    as the qualifiers admitted Farken (11th) and dropped Habz (9th) from the
    projected men's 1500m field."""
    rows = lf.parse_standings_table(soup_table(TIED_HTML))
    by_row = [r["name"] for r in rows][:3]
    by_rank = [r["name"] for r in sorted(rows, key=lambda r: r["rank"])][:3]
    assert "Robert FARKEN" in by_row and "Azeddine HABZ" not in by_row
    assert by_rank == ["Yared NUGUSE", "Azeddine HABZ", "Josh KERR"]
