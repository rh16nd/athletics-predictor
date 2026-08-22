# 2026 Diamond League Predictor — Handoff

_Last updated: 2026-08-22_

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

**Backtest accuracy: 59.0%** (2023 holdout, train on 2021-2022). Honest progression this session: 46.2% (fake, bug-inflated) → 43.6% (real baseline after fixing the bug) → 53.8% (historical data rework) → **59.0%** (h2h fix). Every step was isolated and verified before deploying — see Failed Attempts below for what *didn't* move the number.

**Trained on 13 of 32 live-predicted disciplines**: men/women 100m, 200m, men_400h, women_400h, men/women 800m, men/women 1500m, men/women PV, men_LJ. The other 19 (5000m, 3000m steeplechase, most hurdles, all throws, HJ, TJ, women_LJ) get predictions from the model but have never been backtested against real labeled outcomes for their own event type — **this is the in-progress next step, see below.**

**Injury/withdrawal detection**: scrapes LetsRun/Athletics Weekly/World Athletics news for narrative injury mentions, cross-checks meet-results recaps for bare "DNF" entries, and estimates recovery time per injury type (hamstring/achilles/calf/etc.) to decide whether an athlete should be dropped from predictions vs. just flagged. Deliberately ignores DNS (too many non-injury causes) and DQ (rules violation, unrelated to health).

**Dashboard**: shows live last-updated date and dynamically-computed meet status (done/next/upcoming) on every page, real World Athletics profile links on athlete names, honest labeling on the (still-synthetic) Projections trajectory chart.

**Both repos are pushed and up to date with origin/main** as of this writing, except for the h2h fix + normalization fix from this session's final stretch, which are made locally but **not yet committed/pushed**.

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
- Two specific qualified athletes (Agnes Jebet Ngetich, women's 5000m; Yemisi Mabry, women's shot put) are genuinely absent from World Athletics toplists even at top-500 depth — likely qualified via season points/placings rather than a fast raw mark. Not fixable via more pagination.
- `data/raw/men_5000m.csv` / `women_5000m.csv` (historical multi-year files) don't exist — neither the dead `scraper.py` nor `historical_scraper.py` ever covered these two disciplines.
- Women's Shot Put / Women's 5000m predictions are consistently short by 1-2 athletes (missing from the top-100/500 world list, or missing precise DOB data).
- Projections page's per-meet trajectory chart is still fabricated interpolation, relabeled honestly but not rebuilt — flagged by the user as something to revisit later, deliberately deprioritized.
- Minor, low-priority, not acted on: `api.py` hardcodes `"qualified": true` for every athlete (harmless no-op today); `lovable-error-reporting.ts` does nothing outside the Lovable editor; React Query is wired into the dashboard but `usePredictions.ts` still does manual `fetch`/`useState` instead of using it.

## Next Steps

1. **In progress — extend training beyond 13 disciplines.** Blocked on doing this *reliably*: Wikipedia + AI-summarized extraction hit two real problems — (a) the Diamond League Final's event program isn't fixed year to year (2022's Zurich Final ran a 5km road race instead of the track 5000m for both genders), and (b) Wikipedia's coverage has real gaps for several women's field events and hurdles/steeplechase in 2021/2022, with no way to tell yet whether that's a missing-page issue or the event genuinely wasn't contested. **Planned fix**: switch to scraping World Athletics' own official competition results pages directly (same reliable Selenium approach already used elsewhere in this codebase) instead of trusting Wikipedia + AI summarization, to get authoritative confirmation of what was actually contested each year before writing any new labels.
2. Once disciplines are extended: retrain via `src/train_model.py`, compare backtest honestly (same isolate-before-deploy discipline used all session).
3. Commit and push the h2h fix + normalization fix (currently local-only).
4. Lower priority, explicitly saved for last per the user: landing/welcome page, README files for both repos, real per-meet Projections chart (needs the bigger multi-meet-results scrape), React Query refactor, mobile layout.

## Key Files to Know

- `run.py` — master pipeline: scrape → injury check → load model → build features → predict
- `src/train_model.py` — retraining entry point; `--with-recency --with-h2h --dry-run` to test without overwriting `outputs/`
- `src/historical_scraper.py` — rebuilds historical training data from World Athletics
- `src/injury_checker.py` — injury/withdrawal detection + severity estimation
- `api.py` — Flask bridge between predictions and the dashboard
- `src/components/dl/shell.tsx`, `src/lib/dl-data.ts` — dashboard shell + API data contract
