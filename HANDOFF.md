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

# Rebuild real DL Final ground-truth labels (~1 min, hits a public WA API)
python src/dl_final_results_scraper.py

# Retrain the model (fast, seconds — uses already-scraped data/raw/*.csv)
python src/train_model.py --with-recency --with-h2h

# Run the test suite (pytest.ini sets -s so it works with no flags)
python -m pytest
```

Both dev servers auto-reload on code changes (Flask debug mode, Vite HMR) and re-read data files fresh on every request — no restart needed after `run.py` or a retrain.

## Current State

**Model**: RandomForestClassifier (200 trees, balanced class weights), 14 features:
`season_best, career_best, pb_gap, meets_count, consistency, yoy_improvement, age, season_rank, season_percentile, weighted_season_best, wind_adj_season_best, recent_trend, days_since_last, h2h_win_rate`

**Backtest accuracy: 51.3%** (143/279, walk-forward validated across 3 independent test years — 2023, 2024, 2025 — each trained only on strictly earlier years, 31 of 32 disciplines). This is a **methodology change, not a regression**: every earlier number in this doc's history (up to 62.4%) was a single holdout year, tested against ground-truth results that were hand-researched/hand-typed and went through several rounds of "found another wrong podium" fixes. As of this session, ground truth for **2021-2025** comes from `src/dl_final_results_scraper.py`, which hits World Athletics' own public GraphQL API directly — no hand-typing, no research agents, no possibility of a wrong podium making it into training data. Testing on 3 years instead of 1 is also a meaningfully more honest estimate of generalization (93 judgments in a single year has wide error bars; 279 across three years narrows that considerably). The number went down because it's now honest across more/harder cases, not because anything got worse — see below.

**How the model went from 13 disciplines/hand-typed labels to 32 disciplines/real scraped labels, condensed** (full blow-by-blow lives in this session's transcript, not worth keeping in full here since the hand-typed approach it describes no longer exists in the code): the model was first extended from 13 to 31 disciplines using DL Final results hand-researched by background agents. That data had real errors — a user question about why some athletes seemed to be "missing" from the training data (their hunch: could a competitive break explain it?) led to checking every one, which found the hunch's *mechanism* was wrong but the instinct to check was right: **11 wrong podiums** were hand-typing errors, not missing-data gaps (e.g. crediting Sydney McLaughlin with a 2023 400mH placing she never ran — she'd moved to the flat 400m that season; crediting a 2023 men's 1500m/Mile bronze to the wrong runner). Fixing them took accuracy from 58.1% to 62.4% on a single 2023 holdout. A separate real bug was also found and fixed in `add_season_rank()` (only special-cased `"men_PV"` for descending rank, leaving `women_PV`/`men_LJ` ranked backwards since they were first added) — worth knowing if the season-rank feature ever looks wrong for a field event.

**Then the user asked for the real fix directly: "gather real data, don't hardcode things."** Found that World Athletics' own Calendar/Results minisite calls a public GraphQL API (`https://graphql-prod-4881.edge.aws.worldathletics.org/graphql`, authenticated with an `x-api-key` header that's a public AWS AppSync key shipped in every visitor's page load — not a secret, not bypassing any login, the exact mechanism the site's own frontend uses). Introspection is enabled on it, so the schema was read directly rather than guessed. `src/dl_final_results_scraper.py` uses two queries: `getMinisiteCalendarEvents(season)` finds each year's Diamond League Final by filtering for `rankingCategory == "DF"` (not a hardcoded city/date lookup), and `getCalendarCompetitionResults(competitionId, day)` returns every event actually contested that day, tagged `"Diamond Discipline"/"DF"` for the ones that count toward the Diamond Trophy. **Whether a discipline was contested in a given year, and under what name, is read directly from what's present in the response** — no hand-maintained NOT_CONTESTED list anymore. This immediately caught a real research error: the background agent had concluded 2021's men's "5000m" was a genuine track final on an unusual temporary track; the real API data shows it was **also** run as a 5km road race that year (same as 2022), just like the scraper's own auto-detection would have shown for both years if it had existed then. Covers 2021-2025 (2018-2019 split the Final's scoring across two meetings, 2020 was a COVID-era exhibition — different formats, deliberately not forced into this scraper's single-meeting model).

`train_model.py`'s hardcoded `DL_RESULTS` list (500+ lines) and `NAME_FIXES`/`NOT_CONTESTED` dicts are gone entirely, replaced by loading `data/dl_final_results.csv` (rebuild via `python src/dl_final_results_scraper.py`). Zero unmatched-name warnings across all 5 years of real data (every single top-3 finisher's name matched a training row via the existing `normalize_name()`) — a strong signal the real data is far cleaner than the hand-typed version ever was.

**Evaluation methodology also changed**: with 5 real labeled years available instead of 3, `train_and_backtest()` now does walk-forward (expanding-window) validation — train on every year strictly before each test year, one fold per test year (2023, 2024, 2025), instead of one fixed 2021-2022-train/2023-test split. Reports each fold plus the combined total. The deployed model is refit on all 5 years for production (standard practice: cross-validate to get an honest number, ship a model trained on everything). **Honest accuracy: 51.3%** (143/279) — down from 62.4%, but this is not a regression: it's a smaller, more reliable estimate across 3x the test judgments, using ground truth with zero hand-typing errors instead of one lucky/unlucky single year. See "Known Limitations" for what's still not addressed (2018-2020 data, deeper feature history).

**Injury/withdrawal detection**: scrapes LetsRun/Athletics Weekly/World Athletics news for narrative injury mentions, cross-checks meet-results recaps for bare "DNF" entries, and estimates recovery time per injury type (hamstring/achilles/calf/etc.) to decide whether an athlete should be dropped from predictions vs. just flagged. Deliberately ignores DNS (too many non-injury causes) and DQ (rules violation, unrelated to health). **Was computing correctly but never reaching the user** — `api.py`'s `load_predictions()` never read the `injury_watch` column from `predictions_latest.csv`, and the dashboard had zero UI for it. Fixed in two passes: (1) `injuryWatch` now flows through to both the per-discipline tables and the dashboard's top-winners list as a small "Watch" badge; (2) a follow-up audit for other "computed but not surfaced" gaps found the badge's tooltip was still generic text (not the real evidence) and, more importantly, that athletes the injury check filters out *entirely* (`status == "remove"`) only ever got a console print in `run.py` — they'd vanish from the dashboard with zero explanation, indistinguishable from a scraping gap. Fixed: `api.py` now reads `data/injury_flags.json` directly (not just the boolean baked into the CSV), exposing `injuryReason`/`injuryUrl` per athlete and a new `removedAthletes` list; the dashboard shows a "Removed from predictions" panel (only when non-empty) plus a real headline+link in the Watch badge tooltip. Verified live both ways: Shericka Jackson's badge links to the real Lausanne results article; a temporary synthetic "remove" entry (injected into the gitignored `injury_flags.json`, then reverted) confirmed the removed-athletes panel renders correctly.

**Full live end-to-end run done with the 31-discipline model** (`run.py`, ~1hr — longer than the "~15-20 min" estimate above now that it covers all 32 disciplines plus injury-checking): completed cleanly, no errors, predictions generated for every discipline. Two entries that looked like data bugs at first glance turned out to be real 2026 storylines confirmed via web search — "Valarie SION" (women's discus) is Valarie Allman under her married name, and "Rumesh Tharanga PATHIRAGE" at 92.62m (men's javelin) is a genuine world-leading throw by a former cricketer from Sri Lanka. Dashboard reflects the run correctly (32 disciplines, 62% accuracy, 13 days to Brussels).

**Dashboard**: shows live last-updated date and dynamically-computed meet status (done/next/upcoming) on every page, real World Athletics profile links on athlete names, honest labeling on the (still-synthetic) Projections trajectory chart, injury/DNF watch badges (new this session).

**Basic test suite added** (`tests/`, run via `python -m pytest` from the repo root — `pytest.ini` sets `-s` so it works with no flags; needed because `train_model.py`/`api.py` reassign `sys.stdout` at import time for Windows console UTF-8 safety, which otherwise conflicts with pytest's own output capturing). 30 unit tests across `test_train_model.py`, `test_api.py`, `test_dl_final_results_scraper.py` — deliberately scoped to pure, deterministic functions only (no Selenium, no live network, no Flask server, no real files on disk). Two are regression tests for real bugs found and fixed this session: `add_season_rank()` ranking field events backwards, and `build_features()` leaking a later year's improvement into an earlier year's `career_best`. Extracted `strip_gender_prefix()`/`resolve_discipline_key()` out of `dl_final_results_scraper.py`'s `scrape_year()` specifically to make its WA-event-name-to-discipline-key mapping testable (this is the exact kind of lookup that silently broke once already this session — see the "Files Changed" entry for that scraper).

**Repo state as of this writing**: both repos are pushed and up to date — athletics-predictor through `c565cca2`, track-insights-main through `868f002`. The test suite (`tests/`, `pytest.ini`, `requirements.txt` pytest addition, the `dl_final_results_scraper.py` refactor) is made locally and **not yet committed/pushed**.

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
- `src/train_model.py` — `TRAIN_DISCIPLINES`/`FIELD_EVENTS` extended to all 32 disciplines; fixed `add_season_rank()`'s hardcoded `"men_PV"` check to use `FIELD_EVENTS` membership (was ranking `women_PV`/`men_LJ` backwards); added `normalize_name()` (case + diacritic insensitive) for the results merge plus an unmatched-entry warning print

**Rewritten again this evening (real scraped ground truth, replacing the above's hand-typed `DL_RESULTS`):**
- `src/dl_final_results_scraper.py` — **new file.** Scrapes real DL Final results 2021-2025 from World Athletics' own public GraphQL API (see Current State above for how the endpoint/API key were found via introspection, not guessed). No hardcoded competition IDs or results — the Final each year is found by filtering `rankingCategory == "DF"`, and which disciplines were contested is read from what's actually present in the response. Writes `data/dl_final_results.csv` (gitignored-generated-but-currently-tracked the same way `predictions_latest.csv` is — check before assuming it needs a commit).
- `src/train_model.py` — `DL_RESULTS` (500+ lines) and `NAME_FIXES`/`NOT_CONTESTED` removed entirely, replaced by loading the scraper's output; `LABEL_YEARS` extended from `[2021,2022,2023]` to `range(2021,2026)`; `train_and_backtest()` rewritten for walk-forward validation (3 folds: train-then-test on 2023, 2024, 2025 in turn) instead of one fixed split, reporting a combined honest accuracy across all three; the deployed model is refit on all 5 years after validation.
- `api.py`, `src/components/dl/discipline-table.tsx`/`shell.tsx`/`index.tsx`/`routes/projections.tsx` unaffected by this — the injury-watch work from earlier the same evening stands as-is.
- `track-insights-main/src/routes/index.tsx` + `projections.tsx` — "2023 backtest" label updated to "walk-forward '23-'25" / an honest description of the new methodology, since the number no longer means "tested on 2023."

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
- `men_5000m` has real labels for only 1 of the 5 years (2024) — 2021/2022 ran a 5km road race instead, 2023/2025 didn't contest it in any form the scraper recognized (see `src/dl_final_results_scraper.py` docstring). This is now auto-detected from the real data rather than a hand-maintained exclusion list, but the practical effect is the same: thin training signal and no multi-year backtest for this discipline specifically. It still gets live predictions like the other 31.
- **Only 2021-2025 are covered** — 2018/2019 used a different Final format (scoring split across two meetings, Zurich + Brussels) and 2020 was a COVID-era exhibition ("Inspiration Games"), so `dl_final_results_scraper.py` deliberately excludes them rather than forcing a top-3-at-one-meeting label onto a different scoring system. If more training years are wanted, this is the boundary to push on next — would need separate handling for the split-meeting years, not just widening the year range.
- Two specific qualified athletes (Agnes Jebet Ngetich, women's 5000m; Yemisi Mabry, women's shot put) are genuinely absent from World Athletics toplists even at top-500 depth — likely qualified via season points/placings rather than a fast raw mark. Not fixable via more pagination.
- Women's Shot Put / Women's 5000m predictions are consistently short by 1-2 athletes (missing from the top-100/500 world list, or missing precise DOB data).
- Projections page's per-meet trajectory chart is still fabricated interpolation, relabeled honestly but not rebuilt — flagged by the user as something to revisit later, deliberately deprioritized.
- Minor, low-priority, not acted on: `api.py` hardcodes `"qualified": true` for every athlete (harmless no-op today); `lovable-error-reporting.ts` does nothing outside the Lovable editor; React Query is wired into the dashboard but `usePredictions.ts` still does manual `fetch`/`useState` instead of using it.

## Next Steps

1. Commit and push the test suite (see "Repo state" above).
2. If more training years are wanted: handle 2018/2019's split two-meeting Final format separately (different scoring logic, not just a wider year range) and decide whether 2020's COVID-era exhibition is usable at all.
3. If expanding the test suite further: `build_labeled_dataset()`/`train_and_backtest()`/`load_predictions()` aren't covered yet since they need real files on disk (data/raw/*.csv, data/dl_final_results.csv, outputs/predictions_latest.csv) — would need small fixture files under `tests/fixtures/` to test properly rather than hitting the real (large, live-scraped) data.
4. Lower priority, explicitly saved for last per the user: landing/welcome page, README files for both repos, real per-meet Projections chart (needs the bigger multi-meet-results scrape), React Query refactor, mobile layout.

## Key Files to Know

- `run.py` — master pipeline: scrape → injury check → load model → build features → predict
- `src/train_model.py` — retraining entry point; `--with-recency --with-h2h --dry-run` to test without overwriting `outputs/`
- `src/dl_final_results_scraper.py` — rebuilds `data/dl_final_results.csv`, the real DL Final ground-truth labels `train_model.py` trains against
- `src/historical_scraper.py` — rebuilds historical training data (season bests, not DL Final results) from World Athletics
- `tests/` — unit tests for the pure functions in `train_model.py`/`api.py`/`dl_final_results_scraper.py`; run via `python -m pytest`
- `src/injury_checker.py` — injury/withdrawal detection + severity estimation
- `api.py` — Flask bridge between predictions and the dashboard
- `src/components/dl/shell.tsx`, `src/lib/dl-data.ts` — dashboard shell + API data contract
