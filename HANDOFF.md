# 2026 Diamond League Predictor — Handoff

_Last updated: 2026-08-22 (evening session)_

## Goal

A fully automated ML prediction system for the 2026 Diamond League Final (Brussels, Sep 4-5). Two repos:

```
C:\Users\rayen\athletics-predictor\   ← Python ML pipeline
C:\Users\rayen\track-insights-main\  ← React dashboard (TanStack Router + Vite)
```

The pipeline scrapes live World Athletics data, checks for injuries/withdrawals, runs a trained model, and serves predictions to the dashboard via a Flask API. The explicit current priority (per the user): **model accuracy first** — extending training data and fixing the head-to-head signal — with visual/UX polish (landing page, styling) deliberately saved for last since it's "probably not very hard."

## How to Run

```powershell
# Terminal 1 — Flask API
cd C:\Users\rayen\athletics-predictor
venv\Scripts\activate
python api.py   # http://localhost:5000

# Terminal 2 — React dashboard
cd C:\Users\rayen\track-insights-main
npm run dev     # http://localhost:8080 (or 8081 if 8080 is taken)

# Refresh predictions (live scrape, ~15-20 min)
cd C:\Users\rayen\athletics-predictor
venv\Scripts\activate
python run.py

# Retrain the model (fast, seconds — uses already-scraped data/raw/*.csv)
python src/train_model.py --with-recency --with-h2h
```

Both dev servers auto-reload on code changes (Flask debug mode, Vite HMR) and re-read data files fresh on every request — no restart needed after `run.py` or a retrain.

## Current State

**Model**: RandomForestClassifier (200 trees, balanced class weights), 14 features:
`season_best, career_best, pb_gap, meets_count, consistency, yoy_improvement, age, season_rank, season_percentile, weighted_season_best, wind_adj_season_best, recent_trend, days_since_last, h2h_win_rate`

**Backtest accuracy: 62.4%** (58/93, 2023 holdout, train on 2021-2022, 31 of 32 disciplines). Honest progression: 46.2% (fake, bug-inflated) → 43.6% (real baseline) → 53.8% (historical data rework) → 59.0% (h2h fix, 13 disciplines) → 58.1% (31 disciplines) → 60.2% (fixed a wrong `women_400h` 2023 label) → **62.4%** (fixed 5 more wrong DL Final podiums — see below). Every step isolated and verified before deploying.

**Wrong label caught by the user's intuition, not by scraping**: the original notebook's `DL_RESULTS` credited Sydney McLaughlin with 2nd place in the 2023 `women_400h` DL Final. She's absent from that discipline's 2023 raw toplist data — the user asked whether athletes who take time away from competing (their example: McLaughlin's maternity break, which actually came later, in 2025) could explain gaps like this. Checked it directly: she didn't compete in the 400mH *at all* in 2023 — she moved to the flat 400m that whole season (won the US title in it, raced it at every DL meet). The real 2023 DL Final women's 400mH result was Femke Bol (1st, already correct), **Shamier Little** (2nd), **Rushell Clayton** (3rd) — the old data's "Anna Cockrell, 3rd" was also wrong; she actually finished 5th. Fixed in `src/train_model.py`. Isolated impact: `women_400h` backtest went 1/3 → 3/3, moving overall accuracy 58.1% → 60.2%. This is exactly the kind of stale/wrong hand-entered label the `normalize_name()` unmatched-entry warning (added earlier this session) exists to surface — it had been flagging this one all along, just not yet acted on.

**Following up, the user asked to check all 8 remaining "genuinely absent" athletes the same way — this paid off big.** Verified each one's real DL Final result via live web search (not just checking if the name exists in raw data, but actually researching what happened that year):
- **5 more wrong podiums found and fixed**, all pre-existing errors in the original notebook's hand-curated data (predate this whole session): `women_200m` 2022 (real: Jackson 1st/Thomas 2nd/Clark 3rd — old data had Schippers 2nd, Thomas 3rd, no Clark at all), `men_800m` 2023 (real: Wanyonyi 1st/Arop 2nd/Sedjati 3rd — old data was missing Wanyonyi entirely and had Korir 3rd instead), `men_1500m` 2023 / Bowerman Mile (real: Ingebrigtsen 1st/Nuguse 2nd/Hocker 3rd — old data had Kerr and Kejelcha instead), `men_LJ` 2022 (real: Tentoglou 1st/Dendy 2nd/Massó 3rd — old data had Echevarria and Gayle instead), `women_PV` 2023 (real: **Katie Moon** 1st/Šutej 2nd/Morris 3rd — old data had Kennedy 1st and Newman 3rd, both wrong; **Katie Nageotte and Katie Moon are the same person** — she married and changed her competing surname between the 2022 and 2023 seasons, which is also why the original "Nageotte" entry for 2023 didn't match the raw toplist).
- **2 checked and found to be genuinely correct, just genuinely missing from the scraped toplist**: `men_5000m` 2021 (Aregawi 1st/Balew 2nd/Krop 3rd is the real, correct podium — verified against a source with exact race times — but Aregawi and Krop's times are faster than that year's #100 cutoff in the raw data, so their marks are missing from the toplist scrape itself for some other reason, not a labeling error; the 2021 race's unusual temporary-track venue is a plausible cause but unconfirmed).
- **1 left unresolved**: `women_PV` 2022 (Bengtsson, 3rd) — multiple searches and a direct page fetch could not produce a definitive podium for this specific event (search results kept returning World Championships/other-year pole vault results instead). Not fixed, not disproven — left as-is pending a better source.

Isolated impact of the 5 podium fixes: 58.1% → **62.4%** (58/93). Combined with the McLaughlin fix, that's 4.3pp of pure label-correctness gains found by re-verifying "missing" entries instead of assuming they were scraping gaps — **the user's original hunch (that absence might be explainable, e.g. a competitive break) was the right question to ask, even though the actual mechanism turned out to be wrong-label rather than wrong-year-of-absence.**

**Trained on 31 of 32 live-predicted disciplines** (up from 13, this session's evening work). Extended by researching World Athletics DL Final top-3 results for 2021-2023 for the 19 previously-untrained disciplines (two independent background research passes, cross-verified against WA results pages, Wikipedia, trackalerts.com, world-track.org, letsrun.com), then rebuilding `data/raw/{discipline}.csv` for all of them via `python src/historical_scraper.py --new-only`. `men_5000m` is the only discipline still untested against its own event type: 2022's Zurich Final ran a 5km road race and 2023's Eugene Final had no 5000m at all (Bowerman Mile + separate 3000m instead), so there's no valid 2023 test-year label for it — it's excluded from the backtest (see `NOT_CONTESTED` in `src/train_model.py`) but still gets live predictions like before.

**Isolating what moved the number** (same discipline as every other change this session): found and fixed a real bug in `add_season_rank()` — it only special-cased `"men_PV"` for descending rank direction, leaving `women_PV` and `men_LJ` (both already-trained field events) ranked *backwards* (lowest jump/vault getting rank 1) since they were first added. Fixed by checking `FIELD_EVENTS` membership instead of a hardcoded string. Isolated test: this bugfix alone, on the original 13 disciplines, took accuracy from 59.0% → **64.1%** (real improvement — the bug had been suppressing it). Adding the 18 new disciplines then brought the pooled average to 58.1%, because the newly-added field/distance events are inherently harder to predict (smaller fields, more tactical racing) — not a data-quality regression.

**Name-matching found a second latent bug class, same shape as the h2h case-mismatch bug from earlier this session**: the exact-string merge between hand-researched `DL_RESULTS` and WA's scraped athlete names was silently dropping labels on any accent/spelling difference, with no warning. Added `normalize_name()` (case + diacritic insensitive matching) plus a verification print that lists any `DL_RESULTS` entry that still finds zero match after normalizing. Caught and fixed 11 real mismatches (e.g. "Faith Chepngetich KIPYEGON" vs WA's "Faith KIPYEGON", "Chase EALEY" vs WA's own misspelling "Chase EALY", several transliteration variants) — 4 of which were in the *original* 13-discipline dataset and had been silently degrading backtest honesty since before this session. **9 entries remain genuinely unmatched** — confirmed absent from the scraped top-100 seasonal toplist even with substring search, despite being elite/famous athletes (Sydney McLaughlin 400mH 2023, Dafne Schippers 200m 2022, Emmanuel Korir 800m 2023, Yomif Kejelcha 1500m 2023, Angelica Bengtsson/Katie Nageotte PV 2022/2023, Juan Miguel Echevarria LJ 2022, Berihu Aregawi/Jacob Krop 5000m 2021). Not chased further this session — see Known Limitations.

**Injury/withdrawal detection**: scrapes LetsRun/Athletics Weekly/World Athletics news for narrative injury mentions, cross-checks meet-results recaps for bare "DNF" entries, and estimates recovery time per injury type (hamstring/achilles/calf/etc.) to decide whether an athlete should be dropped from predictions vs. just flagged. Deliberately ignores DNS (too many non-injury causes) and DQ (rules violation, unrelated to health). **Was computing correctly but never reaching the user** — `api.py`'s `load_predictions()` never read the `injury_watch` column from `predictions_latest.csv`, and the dashboard had zero UI for it. Fixed: `injuryWatch` now flows through to both the per-discipline tables and the dashboard's top-winners list, rendered as a small "Watch" badge with a tooltip. Verified live: Shericka Jackson (flagged from a real DNF mention in the Aug 21 Lausanne results recap) now shows the badge correctly in Women's 200m.

**Full live end-to-end run done with the 31-discipline model** (`run.py`, ~1hr — longer than the "~15-20 min" estimate above now that it covers all 32 disciplines plus injury-checking): completed cleanly, no errors, predictions generated for every discipline. Two entries that looked like data bugs at first glance turned out to be real 2026 storylines confirmed via web search — "Valarie SION" (women's discus) is Valarie Allman under her married name, and "Rumesh Tharanga PATHIRAGE" at 92.62m (men's javelin) is a genuine world-leading throw by a former cricketer from Sri Lanka. Dashboard reflects the run correctly (32 disciplines, 62% accuracy, 13 days to Brussels).

**Dashboard**: shows live last-updated date and dynamically-computed meet status (done/next/upcoming) on every page, real World Athletics profile links on athlete names, honest labeling on the (still-synthetic) Projections trajectory chart, injury/DNF watch badges (new this session).

**Both repos are pushed and up to date with origin/main** as of this writing. This evening's commits: `0e53ea24` (31-discipline extension), `ecfb6ac3` (doc correction), `3a531c1d` + `2d5220d9` (6 wrong DL Final podium labels fixed, 58.1%→62.4%), `4649bfdb` (athletics-predictor: expose injury_watch via API). track-insights-main: `844203d` (injury watch badge UI).

## Files Changed This Session

### athletics-predictor

**New files:**
- `src/injury_checker.py` — news + meet-results DNF scraping, severity/recovery-time estimation
- `src/train_model.py` — canonical retraining script (supersedes the notebook's buggy cell 28)
- `src/historical_scraper.py` — rebuilds `data/raw/{discipline}.csv` from real World Athletics toplists (2018-2025), replacing a dead Wikipedia scraper and a static, capped-at-2023 zip import

**Modified:**
- `run.py` — injury filtering wired in; dynamic per-event qualification limits (was hardcoded `.head(8)`, silently dropping 2 of 10 qualifiers for every long-distance event); real `yoy_improvement`/`consistency` computation; recency features (`recent_trend`/`days_since_last`) as trained features, manual post-hoc penalty removed; **h2h case-matching bug fixed + h2h as a trained feature, manual 60/40 blend removed**; **probability normalization fixed** (removed an arbitrary `(prob/total)*3` rescale that was causing multiple athletes to pile up at an identical 95% ceiling); real World Athletics profile URLs carried through to the CSV
- `src/live_fetcher.py` — fixed a table-order typo (`men_3000m`/`women_3000m` instead of `men_5000m`/`women_5000m`) that zeroed out both 5000m events' qualifiers on every run; added pagination (`?page=2`, etc.) for DL-qualified athletes ranked outside the world top 100; captures each athlete's real WA profile URL from the toplist table
- `api.py` — meet status computed from `date.today()` instead of hand-maintained; serves real profile URLs
- `requirements.txt` — was missing `selenium`, `webdriver-manager`, `flask`, `flask-cors` (a fresh install would have failed immediately)
- `.gitignore` — added `injury_flags.json`, `.ipynb_checkpoints/`
- `notebooks/01_eda.ipynb` — left the buggy training cell as historical record, added a final cell explaining the bug and pointing to `src/train_model.py`

**Deleted (confirmed dead, zero references anywhere):**
- `src/scraper.py` — tested live, returns 0 rows for every discipline/year it targets (wrong Wikipedia URL pattern)
- `src/extract_new.py` — one-time zip importer, output fully superseded by `historical_scraper.py`
- `src/test_api.py` — disconnected scratch file testing an unused package
- 2 stray `.ipynb_checkpoints/` files that had been accidentally committed to git

**Model artifacts** (`outputs/model_rf.pkl`, `scaler.pkl`, `feature_cols.pkl`, `model_accuracy.txt`) retrained and overwritten multiple times as fixes landed — always via `src/train_model.py`, never hand-edited.

**Modified again this evening (extending to 31 disciplines):**
- `src/historical_scraper.py` — `TRAIN_DISCIPLINES` extended from 13 to 32 keys; added a `--new-only` flag (`NEWLY_ADDED` list) so the 19 new disciplines could be scraped without re-scraping the 13 already-good ones
- `src/train_model.py` — `TRAIN_DISCIPLINES`/`FIELD_EVENTS` extended to all 32 disciplines; added `NOT_CONTESTED` set to exclude `men_5000m` 2022/2023 and `women_5000m` 2022 (DL Final ran a road race or different program those years, so there's no valid label); fixed `add_season_rank()`'s hardcoded `"men_PV"` check to use `FIELD_EVENTS` membership (was ranking `women_PV`/`men_LJ` backwards); added `normalize_name()` (case + diacritic insensitive) for the `DL_RESULTS` merge plus an unmatched-entry warning print, catching 11 real name mismatches (4 in the original 13 disciplines); 162 new verified `DL_RESULTS` entries added (research methodology + sources in git history / this session's transcript)

### track-insights-main

**Modified:**
- `src/components/dl/shell.tsx` + all 5 route files — header now takes `lastUpdated`/`daysToFinal` as props from the live API instead of hardcoded constants
- `src/lib/dl-data.ts` — removed the now-dead `LAST_UPDATED`/`DAYS_TO_FINAL` constants
- `src/components/dl/discipline-table.tsx` — uses the real `waUrl` from the API instead of constructing its own search-query link; fixed a `prob * 1.4` bar-width bug that made any two athletes both ≥71.4% render as identical maxed-out bars; added a guard against crashing on an empty discipline list
- `src/routes/projections.tsx` — same `prob * 1.4` bug fixed; trajectory chart relabeled honestly ("Modeled trajectory" / "Illustrative trend") since it's fabricated interpolation, not real per-meet data
- `package.json` — removed 35 dependencies (all `@radix-ui/*`, `react-hook-form`, `zod`, `date-fns`, `lucide-react`, and others) that were only used by now-deleted dead code; `npm install` dropped 91 packages total once transitive deps came out
- `.gitignore` — was corrupted (a UTF-16-encoded line for `.claude/` inside an otherwise UTF-8 file, likely a PowerShell-redirect artifact), so `.claude/` was never actually ignored — **172 files (4.5MB) of the Claude Code skills bundle had been silently committed to the public GitHub repo.** Rewrote cleanly and untracked them; also added `.output/`/`.wrangler/`

**Deleted:**
- `src/routes/index.tsx.bak` — stale, unreferenced, and already broken JSX
- All 44 files in `src/components/ui/` (unused shadcn kit — confirmed via grep that nothing outside that folder imported any of them) plus `src/hooks/use-mobile.tsx` and `src/lib/utils.ts` (only consumed by the deleted files)

## Failed Attempts (tried, verified, did not help — don't redo without new evidence)

- **Event-specific models** (separate RandomForest per event group: sprints/hurdles, middle-distance, field). Tested three ways — full separate models, and a pooled model with an `event_group` feature added — all three scored **exactly the same 53.8%** as the plain pooled model, with only a meaningless ±1 swing between individual groups (noise at 9-18 test judgments per group). Not deployed. Likely reason: `season_rank`/`season_percentile` are already computed relative to each discipline's own field, so the model doesn't need separate models to normalize per-event.
- **`h2h_meetings` (count) as an additional trained feature** alongside `h2h_win_rate`: made it slightly *worse* (22/39 vs 23/39). Not included.
- **Manual h2h blend at various fixed weights** (0.2 through 1.0): tied with the trained-feature approach at its best (23/39 at weight 0.2-0.6). Chose the trained-feature approach instead, for architectural consistency with how recency was already handled (a manual post-hoc blend was exactly the bug we were fixing — didn't want to reintroduce the same pattern for h2h).
- **`src/scraper.py`**: not a "didn't help" so much as fully non-functional — confirmed live, 0 rows for every discipline/year, ever. Deleted rather than fixed (the correct Wikipedia URL pattern isn't obvious and a working alternative already exists in `historical_scraper.py`).

## Known Limitations (real, understood, not bugs to re-chase)

- `meets_count`, `consistency`, and `recent_trend` are **structurally constant** (zero variance) across the entire training set — the World Athletics toplist source is one-mark-per-athlete-per-season everywhere, training included now. Only `days_since_last` survives with real signal (needs just one date, not multiple marks). Fixing this for real needs a different scrape entirely (full per-athlete meet-by-meet results, not a toplist).
- `men_5000m` is trained on 2021 data only and excluded from the backtest entirely (see `NOT_CONTESTED`) — 2022/2023 DL Finals didn't run the standard track event those years. It still gets live predictions like the other 31 disciplines, just never backtested for its own event type.
- **3 DL_RESULTS entries remain unmatched** after re-verifying all 8 originally-flagged ones (see above): `women_PV` 2022 (Bengtsson, 3rd — genuinely couldn't find a definitive source for this specific podium, not confirmed right or wrong), and `men_5000m` 2021 (Aregawi 1st + Krop 3rd — confirmed correct via a source with exact times, but their marks are missing from the scraped toplist despite being faster than that year's own #100 cutoff; likely a scraping gap specific to that discipline+year, possibly related to the race's unusual temporary-track venue, unconfirmed). None of these 3 are in the 2023 test year, so they no longer distort the backtest score.
- Two specific qualified athletes (Agnes Jebet Ngetich, women's 5000m; Yemisi Mabry, women's shot put) are genuinely absent from World Athletics toplists even at top-500 depth — likely qualified via season points/placings rather than a fast raw mark. Not fixable via more pagination.
- Women's Shot Put / Women's 5000m predictions are consistently short by 1-2 athletes (missing from the top-100/500 world list, or missing precise DOB data).
- Projections page's per-meet trajectory chart is still fabricated interpolation, relabeled honestly but not rebuilt — flagged by the user as something to revisit later, deliberately deprioritized.
- Minor, low-priority, not acted on: `api.py` hardcodes `"qualified": true` for every athlete (harmless no-op today); `lovable-error-reporting.ts` does nothing outside the Lovable editor; React Query is wired into the dashboard but `usePredictions.ts` still does manual `fetch`/`useState` instead of using it.

## Next Steps

1. If ever revisiting label quality again: the `women_PV` 2022 (Bengtsson) entry above is still unresolved — worth another look with a better source. Otherwise the "genuinely absent" list is now down to disciplines/years that don't affect the 2023 test-year backtest score.
2. Lower priority, explicitly saved for last per the user: landing/welcome page, README files for both repos, real per-meet Projections chart (needs the bigger multi-meet-results scrape), React Query refactor, mobile layout.

## Key Files to Know

- `run.py` — master pipeline: scrape → injury check → load model → build features → predict
- `src/train_model.py` — retraining entry point; `--with-recency --with-h2h --dry-run` to test without overwriting `outputs/`
- `src/historical_scraper.py` — rebuilds historical training data from World Athletics
- `src/injury_checker.py` — injury/withdrawal detection + severity estimation
- `api.py` — Flask bridge between predictions and the dashboard
- `src/components/dl/shell.tsx`, `src/lib/dl-data.ts` — dashboard shell + API data contract
