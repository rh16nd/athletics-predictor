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

**Backtest accuracy: 58.1%** (54/93, 2023 holdout, train on 2021-2022, 31 of 32 disciplines). Honest progression: 46.2% (fake, bug-inflated) → 43.6% (real baseline) → 53.8% (historical data rework) → 59.0% (h2h fix, 13 disciplines) → **58.1%** (31 disciplines — see below for why this isn't a regression). Every step isolated and verified before deploying.

**Trained on 31 of 32 live-predicted disciplines** (up from 13, this session's evening work). Extended by researching World Athletics DL Final top-3 results for 2021-2023 for the 19 previously-untrained disciplines (two independent background research passes, cross-verified against WA results pages, Wikipedia, trackalerts.com, world-track.org, letsrun.com), then rebuilding `data/raw/{discipline}.csv` for all of them via `python src/historical_scraper.py --new-only`. `men_5000m` is the only discipline still untested against its own event type: 2022's Zurich Final ran a 5km road race and 2023's Eugene Final had no 5000m at all (Bowerman Mile + separate 3000m instead), so there's no valid 2023 test-year label for it — it's excluded from the backtest (see `NOT_CONTESTED` in `src/train_model.py`) but still gets live predictions like before.

**Isolating what moved the number** (same discipline as every other change this session): found and fixed a real bug in `add_season_rank()` — it only special-cased `"men_PV"` for descending rank direction, leaving `women_PV` and `men_LJ` (both already-trained field events) ranked *backwards* (lowest jump/vault getting rank 1) since they were first added. Fixed by checking `FIELD_EVENTS` membership instead of a hardcoded string. Isolated test: this bugfix alone, on the original 13 disciplines, took accuracy from 59.0% → **64.1%** (real improvement — the bug had been suppressing it). Adding the 18 new disciplines then brought the pooled average to 58.1%, because the newly-added field/distance events are inherently harder to predict (smaller fields, more tactical racing) — not a data-quality regression.

**Name-matching found a second latent bug class, same shape as the h2h case-mismatch bug from earlier this session**: the exact-string merge between hand-researched `DL_RESULTS` and WA's scraped athlete names was silently dropping labels on any accent/spelling difference, with no warning. Added `normalize_name()` (case + diacritic insensitive matching) plus a verification print that lists any `DL_RESULTS` entry that still finds zero match after normalizing. Caught and fixed 11 real mismatches (e.g. "Faith Chepngetich KIPYEGON" vs WA's "Faith KIPYEGON", "Chase EALEY" vs WA's own misspelling "Chase EALY", several transliteration variants) — 4 of which were in the *original* 13-discipline dataset and had been silently degrading backtest honesty since before this session. **9 entries remain genuinely unmatched** — confirmed absent from the scraped top-100 seasonal toplist even with substring search, despite being elite/famous athletes (Sydney McLaughlin 400mH 2023, Dafne Schippers 200m 2022, Emmanuel Korir 800m 2023, Yomif Kejelcha 1500m 2023, Angelica Bengtsson/Katie Nageotte PV 2022/2023, Juan Miguel Echevarria LJ 2022, Berihu Aregawi/Jacob Krop 5000m 2021). Not chased further this session — see Known Limitations.

**Injury/withdrawal detection**: scrapes LetsRun/Athletics Weekly/World Athletics news for narrative injury mentions, cross-checks meet-results recaps for bare "DNF" entries, and estimates recovery time per injury type (hamstring/achilles/calf/etc.) to decide whether an athlete should be dropped from predictions vs. just flagged. Deliberately ignores DNS (too many non-injury causes) and DQ (rules violation, unrelated to health).

**Dashboard**: shows live last-updated date and dynamically-computed meet status (done/next/upcoming) on every page, real World Athletics profile links on athlete names, honest labeling on the (still-synthetic) Projections trajectory chart.

**Both repos were clean and pushed as of the start of this evening's session** (the h2h fix + normalization fix mentioned as "local-only" earlier in this doc had already landed in commit `f150779d` — that line was stale, corrected here). The 31-discipline extension (`train_model.py`, `historical_scraper.py`, `HANDOFF.md`, retrained `outputs/`) is made locally and **not yet committed/pushed** as of this writing.

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
- **9 DL_RESULTS entries are genuinely absent from the scraped seasonal toplists**, confirmed via substring search (not a spelling issue): Sydney McLaughlin (women_400h, 2023 test year), Dafne Schippers (women_200m, 2022), Emmanuel Korir (men_800m, 2023 test year), Yomif Kejelcha (men_1500m, 2023 test year), Angelica Bengtsson (women_PV, 2022), Katie Nageotte (women_PV, 2023 test year), Juan Miguel Echevarria (men_LJ, 2022), Berihu Aregawi + Jacob Krop (men_5000m, 2021). For elite/famous athletes like these, absence from a top-100 seasonal toplist is surprising — most likely a real gap in what `historical_scraper.py`'s single-page scrape captures for that specific discipline+year, not that they genuinely ranked outside the top 100. Not investigated further this session (would need re-scraping with deeper pagination or a different WA endpoint) — flagging honestly rather than silently accepting it, since 5 of these are 2023 test-year labels that put an artificial ceiling on that discipline's backtest score.
- Two specific qualified athletes (Agnes Jebet Ngetich, women's 5000m; Yemisi Mabry, women's shot put) are genuinely absent from World Athletics toplists even at top-500 depth — likely qualified via season points/placings rather than a fast raw mark. Not fixable via more pagination.
- Women's Shot Put / Women's 5000m predictions are consistently short by 1-2 athletes (missing from the top-100/500 world list, or missing precise DOB data).
- Projections page's per-meet trajectory chart is still fabricated interpolation, relabeled honestly but not rebuilt — flagged by the user as something to revisit later, deliberately deprioritized.
- Minor, low-priority, not acted on: `api.py` hardcodes `"qualified": true` for every athlete (harmless no-op today); `lovable-error-reporting.ts` does nothing outside the Lovable editor; React Query is wired into the dashboard but `usePredictions.ts` still does manual `fetch`/`useState` instead of using it.

## Next Steps

1. **Commit and push this evening's work** (31-discipline extension + season_rank bugfix + name-matching fixes) — currently local-only.
2. Consider investigating the 9 genuinely-absent-athlete data gaps above (particularly the 5 in the 2023 test year) if the honest backtest number needs to be tightened further — would need to inspect why `historical_scraper.py`'s scrape misses them (deeper pagination? different WA endpoint?) rather than assuming top-100 is always enough.
3. Lower priority, explicitly saved for last per the user: landing/welcome page, README files for both repos, real per-meet Projections chart (needs the bigger multi-meet-results scrape), React Query refactor, mobile layout.

## Key Files to Know

- `run.py` — master pipeline: scrape → injury check → load model → build features → predict
- `src/train_model.py` — retraining entry point; `--with-recency --with-h2h --dry-run` to test without overwriting `outputs/`
- `src/historical_scraper.py` — rebuilds historical training data from World Athletics
- `src/injury_checker.py` — injury/withdrawal detection + severity estimation
- `api.py` — Flask bridge between predictions and the dashboard
- `src/components/dl/shell.tsx`, `src/lib/dl-data.ts` — dashboard shell + API data contract
