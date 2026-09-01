"""Parsing tests for the head-to-head scraper (HANDOFF 0o's open half).

Item 0o fixed the calculator but left the data defect standing: the scraper
never recorded which race a row came from, so heats, semis and the final
landed in one bucket per meeting and the calculator had to collapse them to
each athlete's best place -- discarding every pair that happened to tie on
best place.

Reading the pages to close that turned up two more defects in the same
function, both silent and both live for the whole life of the file:

  * Wikipedia renders positions 1-3 as a medal template with NO text in the
    cell, so `int(cell.text)` returned nothing for every podium. 2825 of
    13040 scraped rows (21.7%) had no place at all, and they were the
    podiums.
  * the mark was read at a fixed `athlete_col + 2` offset, which lands on
    Country for any table carrying an Age column -- 1382 rows held a country
    name where a mark belonged, so the DNF/DNS/NM/DQ exclusion was testing
    the wrong column and never fired on those pages.
  * `classify_table` mislabelled the hurdles twice over. Matching on ANY
    keyword meant the word "hurdle" alone satisfied men_110h/women_100h,
    which are tested first, so every 400m hurdles race went to the sprint
    hurdles -- men_400h and women_400h had no rows in the dataset at all.
    And the guard keeping a flat sprint from swallowing a hurdles race
    tested for the word "hurdle", which the abbreviation "100mH" does not
    contain, so every abbreviated women's 100m hurdles race on a Diamond
    League page was recorded as a women's 100m FLAT race.

The HTML below is the real shape of each page kind, trimmed. No network.
"""
from bs4 import BeautifulSoup

import h2h_scraper as hs


def soup(html):
    return BeautifulSoup(html, "html.parser")


def medal(place):
    """A podium cell exactly as Wikipedia renders it: no text, just the
    template's sort key and the medal image."""
    alt = {1: "1st place, gold medalist(s)",
           2: "2nd place, silver medalist(s)",
           3: "3rd place, bronze medalist(s)"}[place]
    return f'<span data-sort-value="0{place}\xa0!"><img alt="{alt}" src="/medal.png"></span>'


# A Diamond League meeting: the same event run twice under different
# sections, an Age column between athlete and mark, medal-template podium,
# and a DNF.
DL_PAGE = f"""
<h1>2024 Athletissima</h1>
<h2>Results</h2>
<h3>Diamond Discipline</h3>
<table class="wikitable">
  <caption>Men's 1500 Metres</caption>
  <tr><th>Place</th><th>Athlete</th><th>Age</th><th>Country</th><th>Time</th><th>Points</th></tr>
  <tr><td>{medal(1)}</td><td>Jakob Ingebrigtsen</td><td>23</td><td>Norway</td><td>3:27.83</td><td>8</td></tr>
  <tr><td>{medal(2)}</td><td>Cole Hocker</td><td>23</td><td>United States</td><td>3:29.85</td><td>7</td></tr>
  <tr><td>{medal(3)}</td><td>Hobbs Kessler</td><td>21</td><td>United States</td><td>3:30.47</td><td>6</td></tr>
  <tr><td>4</td><td>Reynold Cheruiyot</td><td>20</td><td>Kenya</td><td>3:30.88</td><td>5</td></tr>
  <tr><td></td><td>Luke McCann</td><td>26</td><td>Ireland</td><td>DNF</td><td></td></tr>
</table>
<h3>National events</h3>
<table class="wikitable">
  <caption>Men's 1500 Metres</caption>
  <tr><th>Place</th><th>Athlete</th><th>Age</th><th>Country</th><th>Time</th></tr>
  <tr><td>{medal(1)}</td><td>Elliot Vermeulen</td><td>24</td><td>Belgium</td><td>3:41.10</td></tr>
  <tr><td>4</td><td>Paul McIntyre</td><td>22</td><td>Switzerland</td><td>3:43.00</td></tr>
</table>
"""

# A championship event page: ONE heats table covering every heat, ranked
# across them by time, with the heat itself in a column.
HEATS_PAGE = """
<h1>2023 World Athletics Championships – Men's 1500 metres</h1>
<h2>Results</h2>
<h3>Heats</h3>
<table class="wikitable">
  <tr><th>Rank</th><th>Heat</th><th>Name</th><th>Nationality</th><th>Time</th><th>Notes</th></tr>
  <tr><td>1</td><td>1</td><td>Jakob Ingebrigtsen</td><td>Norway</td><td>3:33.94</td><td>Q</td></tr>
  <tr><td>2</td><td>1</td><td>Josh Kerr</td><td>Great Britain</td><td>3:34.00</td><td>Q</td></tr>
  <tr><td>3</td><td>4</td><td>Abel Kipsang</td><td>Kenya</td><td>3:34.08</td><td>Q</td></tr>
  <tr><td>4</td><td>4</td><td>Yared Nuguse</td><td>United States</td><td>3:34.16</td><td>Q</td></tr>
</table>
"""

# An Olympic sprint page: three separate rounds each numbering their heats
# from 1, and a Final whose nearest PRECEDING h4 belongs to the semi-finals.
OLYMPIC_PAGE = """
<h1>Athletics at the 2024 Summer Olympics – Men's 100 metres</h1>
<h2>Results</h2>
<h3>First round</h3>
<h4>Heat 1</h4>
<table class="wikitable">
  <tr><th>Rank</th><th>Lane</th><th>Athlete</th><th>Nation</th><th>Time</th><th>Notes</th></tr>
  <tr><td>1</td><td>4</td><td>Alpha Speedy</td><td>Jamaica</td><td>9.95</td><td>Q</td></tr>
  <tr><td>2</td><td>5</td><td>Beta Quick</td><td>Kenya</td><td>10.01</td><td>Q</td></tr>
</table>
<h3>Semi-finals</h3>
<h4>Heat 3</h4>
<table class="wikitable">
  <tr><th>Rank</th><th>Lane</th><th>Athlete</th><th>Nation</th><th>Time</th><th>Notes</th></tr>
  <tr><td>1</td><td>4</td><td>Beta Quick</td><td>Kenya</td><td>9.90</td><td>Q</td></tr>
  <tr><td>2</td><td>5</td><td>Alpha Speedy</td><td>Jamaica</td><td>9.92</td><td>Q</td></tr>
</table>
<h3>Final</h3>
<table class="wikitable">
  <tr><th>Rank</th><th>Lane</th><th>Athlete</th><th>Nation</th><th>Time</th><th>Notes</th></tr>
  <tr><td>1</td><td>4</td><td>Alpha Speedy</td><td>Jamaica</td><td>9.79</td><td></td></tr>
  <tr><td>2</td><td>5</td><td>Beta Quick</td><td>Kenya</td><td>9.81</td><td></td></tr>
</table>
"""


def rows_of(page, level="DL", year="2024"):
    return hs.scrape_soup(soup(page), "http://example.invalid", level, year)


# ---- the podium was missing from every race in the dataset ----

def test_a_medal_icon_is_a_finishing_position():
    cell = soup(f"<td>{medal(2)}</td>").td
    assert hs.parse_place(cell) == 2


def test_a_medal_icon_is_read_even_without_the_sort_key():
    cell = soup('<td><img alt="3rd place, bronze medalist(s)" src="/m.png"></td>').td
    assert hs.parse_place(cell) == 3


def test_an_empty_place_cell_stays_empty():
    """A DNF has no position. Only the three medal values are accepted, so
    an unrelated sort key cannot be mistaken for one."""
    assert hs.parse_place(soup("<td></td>").td) is None
    assert hs.parse_place(soup('<td><span data-sort-value="99 !"></span></td>').td) is None


def test_the_whole_podium_survives_a_scrape():
    places = [r["place"] for r in rows_of(DL_PAGE) if r["race"].endswith("Diamond Discipline / Men's 1500 Metres")]
    assert places == [1, 2, 3, 4]


# ---- the mark column is found by name, not by counting two along ----

def test_the_mark_is_the_mark_and_not_the_country():
    winner = rows_of(DL_PAGE)[0]
    assert winner["mark"] == "3:27.83"


def test_a_dnf_is_excluded_even_when_a_column_sits_between_athlete_and_time():
    """The offset landed on Country here, so "DNF" was never seen and the
    row was kept."""
    assert not [r for r in rows_of(DL_PAGE) if r["athlete"] == "Luke McCann"]


# ---- which race a row came from ----

def test_the_same_event_run_twice_is_two_races():
    """Athletissima runs a Diamond Discipline 1500m and a national 1500m.
    Sharing a caption, they used to share a meeting -- which put Jakob
    Ingebrigtsen on the same start list as the Swiss national field."""
    races = {r["race"] for r in rows_of(DL_PAGE)}
    assert len(races) == 2
    assert all(r.endswith("Men's 1500 Metres") for r in races)


def test_a_heats_table_keeps_the_heat_each_athlete_ran_in():
    """Rank here is across ALL heats by time, so the heat number is the only
    thing separating a real head-to-head from a time comparison."""
    rows = rows_of(HEATS_PAGE, level="Worlds", year="2023")
    heats = {r["athlete"]: r["heat"] for r in rows}
    assert heats["Jakob Ingebrigtsen"] == "1"
    assert heats["Yared Nuguse"] == "4"


def test_rounds_that_number_their_heats_alike_stay_apart():
    races = {r["race"] for r in rows_of(OLYMPIC_PAGE, level="Olympics")}
    assert "Results / First round / Heat 1" in races
    assert "Results / Semi-finals / Heat 3" in races


def test_a_final_is_not_filed_under_the_previous_rounds_heat():
    """The Final table's nearest preceding h4 is the semi-finals' "Heat 3".
    Taking the nearest heading rather than the section path put the final
    inside a heat."""
    finals = [r for r in rows_of(OLYMPIC_PAGE, level="Olympics") if r["race"].endswith("Final")]
    assert len(finals) == 2
    assert hs.heading_path(soup(OLYMPIC_PAGE).find_all("table")[-1]) == ["Results", "Final"]


# ---- two hurdles events, not one ----

def test_the_400m_hurdles_is_its_own_event():
    """Warholm, Benjamin, dos Santos, Bol and McLaughlin-Levrone were all
    filed under the sprint hurdles, and the two 400H disciplines held no
    rows whatsoever."""
    assert hs.classify_table("Men's 400 Metres Hurdles") == "men_400h"
    assert hs.classify_table("Women's 400 Metres Hurdles") == "women_400h"


def test_the_sprint_hurdles_still_classify():
    assert hs.classify_table("Men's 110 Metres Hurdles(-0.3m/s)") == "men_110h"
    assert hs.classify_table("Women's 100 Metres Hurdles") == "women_100h"


def test_a_flat_400_is_not_a_hurdles_race():
    assert hs.classify_table("Men's 400 Metres") == "men_400m"


def test_the_abbreviated_spelling_is_the_same_race():
    """Wikipedia writes it both ways on the same site. The abbreviation has
    no "hurdle" in it, which is how "Women's 100mH(+0.6m/s)" -- Tobi Amusan,
    Danielle Williams, Masai Russell -- was recorded as a FLAT 100m."""
    assert hs.classify_table("Women's 100mH(+0.6m/s)") == "women_100h"
    assert hs.classify_table("Men's 110mH Round 1") == "men_110h"
    assert hs.classify_table("Men's 400mH") == "men_400h"
    assert hs.classify_table("Women's 400mH") == "women_400h"


def test_a_distance_with_no_hurdles_marker_is_still_flat():
    """The abbreviation detector must not fire on ordinary captions."""
    assert hs.classify_table("Men's 800m Heat 1") == "men_800m"
    assert hs.classify_table("Men's 3000 Metres Steeplechase") == "men_3000sc"
    assert hs.classify_table("Men's High Jump") == "men_HJ"


def test_a_hurdles_distance_this_project_does_not_track_is_not_forced_in():
    assert hs.classify_table("Men's 300 metres hurdles") is None


def test_the_gender_can_come_from_the_section_above_the_table():
    """Older meeting pages split into Men/Women sections and head each table
    with the bare event, so the caption alone has no gender to match on."""
    assert hs.classify_table("Results / Men / 400 metres hurdles") == "men_400h"
    assert hs.classify_table("Results / Women / 100 m hurdles(-3.9m/s)") == "women_100h"


def test_an_event_page_title_classifies_the_same_way_as_a_caption():
    """Championship pages carry the discipline in the <h1>, not in a
    caption, so both routes have to agree."""
    title = "2023 World Athletics Championships – Women's 400 metres hurdles"
    assert hs.classify_table(title) == "women_400h"


def test_a_relay_is_not_an_individual_event():
    """"4x100 Metres Relay" contains "100 m". The athlete cell holds four
    names run together, so the rows could never match anyone."""
    assert hs.classify_table("Women's 4x100 Metres Relay") is None
    assert hs.classify_table("Men's 4 x 400 Metres Relay") is None


# ---- rows that are not athletes ----

FIELD_PAGE = """
<h1>2022 World Athletics Championships – Men's long jump</h1>
<h2>Results</h2>
<h3>Final</h3>
<table class="wikitable">
  <tr><th>Rank</th><th>Name</th><th>Nationality</th><th>Round</th><th>Mark</th><th>Notes</th></tr>
  <tr><th>1</th><th>2</th><th>3</th><th>4</th><th>5</th><th>6</th></tr>
  <tr><td>1</td><td>Wang Jianan</td><td>China</td><td>7.94</td><td>x</td><td>8.03</td></tr>
  <tr><td>2</td><td>Miltiadis Tentoglou</td><td>Greece</td><td>x</td><td>8.30</td><td>8.29</td></tr>
</table>
"""

SPRINT_WITH_WIND = """
<h1>2025 Prefontaine Classic</h1>
<h2>Results</h2>
<h3>Promotional events</h3>
<table class="wikitable">
  <caption>Women's 100 metres hurdles</caption>
  <tr><th>Place</th><th>Athlete</th><th>Country</th><th>Time</th></tr>
  <tr><td>1</td><td>Ackera Nugent</td><td>Jamaica</td><td>12.32</td></tr>
  <tr><td>2</td><td>Tobi Amusan</td><td>Nigeria</td><td>12.38</td></tr>
  <tr><td>24</td><td>Wind:(+0.4m/s)</td><td></td><td></td></tr>
</table>
"""


def test_an_attempt_number_header_is_not_an_athlete():
    """A field-event table has a second header row numbering the six
    attempts. It parsed as an athlete named "2" who took first place --
    beating everyone in the competition -- in 40 tables."""
    names = [r["athlete"] for r in rows_of(FIELD_PAGE, level="Worlds", year="2022")]
    assert names == ["Wang Jianan", "Miltiadis Tentoglou"]


def test_a_wind_footer_is_not_an_athlete():
    names = [r["athlete"] for r in rows_of(SPRINT_WITH_WIND)]
    assert names == ["Ackera Nugent", "Tobi Amusan"]
