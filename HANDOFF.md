# 2026 Diamond League Predictor — Handoff

_Last updated: 2026-08-23_

## Goal

A fully automated ML prediction system for the 2026 Diamond League Final (Brussels, Sep 4-5). It scrapes live World Athletics data, checks for injuries/withdrawals, runs a trained model, and serves win-probability predictions to a dashboard. Two repos:

```
C:\Users\rayen\athletics-predictor\   ← Python ML pipeline
C:\Users\rayen\track-insights-main\  ← React dashboard (TanStack Router + Vite)
```

**Priority order, per the user**: model accuracy and data honesty first; visual/UX polish (landing page, styling) deliberately saved for last. When in doubt about what to work on next, prefer real data over hardcoded/hand-typed data, and always isolate a change's effect before trusting a number that moved.

## How to Run

```powershell
# Terminal 1 — Flask API
cd C:\Users\rayen\athletics-predictor
venv\Scripts\activate
python api.py   # http://localhost:5000

# Terminal 2 — React dashboard
cd C:\Users\rayen\track-insights-main
npm run dev     # http://localhost:8080 (or 8081 if 8080 is taken)

# Refresh live predictions (scrapes World Athletics, ~1hr — covers all 32 disciplines + injury check)
cd C:\Users\rayen\athletics-predictor
venv\Scripts\activate
python run.py

# Rebuild real DL Final ground-truth labels (~1 min, hits a public WA API)
python src/dl_final_results_scraper.py

# Enrich training data with real per-meeting season history (~few min)
python src/season_results_scraper.py

# Retrain the model (fast, seconds — uses already-scraped data/raw/*.csv)
python src/train_model.py --with-recency --with-h2h

# Grid-search hyperparameters via walk-forward folds (informational, prints only)
python src/train_model.py --with-recency --with-h2h --tune

# Run the test suite
python -m pytest
```

Both dev servers auto-reload on code changes (Flask debug mode, Vite HMR) and re-read data files fresh on every request — no restart needed after `run.py` or a retrain.

## Current State

**Model**: RandomForestClassifier, `n_estimators=100, max_depth=16, min_samples_leaf=1, class_weight=None` (`DEFAULT_MODEL_PARAMS` in `train_model.py` — walk-forward tuned via `--tune`, re-tune whenever the feature set changes since the winning config shifts each time). 14 features, **all carrying real signal**:
`season_best, career_best, pb_gap, meets_count, consistency, yoy_improvement, age, season_rank, season_percentile, weighted_season_best, wind_adj_season_best, recent_trend, days_since_last, h2h_win_rate`

**Backtest accuracy: 59.1%** (165/279) — walk-forward validated: trained on every year strictly before each test year, scored independently on 2023, 2024, and 2025 (not one fixed holdout). The deployed model is refit on all 5 years after validation. Trained on all 32 live-predicted disciplines (`men_5000m` has thin signal — see Known Limitations).

**Ground truth is real, not hand-typed.** `src/dl_final_results_scraper.py` pulls actual Diamond League Final results (2021-2025) directly from World Athletics' own public GraphQL API — the same API the site's own frontend uses (`x-api-key` is a public key shipped in every page load, not a secret). It finds each year's Final by filtering `rankingCategory == "DF"` and reads which disciplines were contested (and under what name — e.g. "Mile" some years instead of "1500 Metres") directly from what's present in the response, rather than a hand-maintained list. 2018/2019 (split two-meeting scoring format) and 2020 (COVID-era exhibition) are deliberately excluded — different format, not forced to fit.

**Training features go beyond a season-best toplist.** `src/season_results_scraper.py` pulls every *regular-season* Diamond League meeting's results (not just the Final — that would leak the label) for 2021-2025, giving real multiple-marks-per-athlete-per-season data. This is what makes `meets_count`/`consistency`/`recent_trend` real features instead of structural zeros.

**Injury/withdrawal detection is fully wired end-to-end.** `src/injury_checker.py` scrapes news + meet-results recaps for injury/DNF signals, estimates recovery time, and either flags ("watch") or drops ("remove") an athlete from predictions. Both outcomes are visible on the dashboard: a "Watch" badge (linking to the real evidence) on flagged athletes, and a "Removed from predictions" panel (shown only when non-empty) for dropped ones.

**Test suite**: `tests/` — pure-function unit tests for `train_model.py`, `api.py`, `dl_final_results_scraper.py` (`python -m pytest`, no network/Selenium/Flask/real-files needed). Includes regression tests for the specific bugs described below.

**Both repos are pushed and up to date** — track-insights-main `868f002`, athletics-predictor `2930f618`.

## Architecture / Key Files

- `run.py` — master pipeline: scrape live 2026 data → injury check → load trained model → build features → predict → `outputs/predictions_latest.csv`
- `api.py` — Flask bridge serving predictions + injury data to the dashboard (`/api/predictions`)
- `src/train_model.py` — retraining entry point. `--with-recency --with-h2h` for the full feature set; `--dry-run` to backtest without overwriting `outputs/`; `--tune` to grid-search hyperparameters
- `src/dl_final_results_scraper.py` — real DL Final ground-truth labels (`data/dl_final_results.csv`), scraped from WA's GraphQL API
- `src/season_results_scraper.py` — real per-meeting season history, enriches `data/raw/{discipline}.csv`
- `src/historical_scraper.py` — season-best toplists 2018-2025 (`data/raw/{discipline}.csv` base layer)
- `src/live_fetcher.py` — scrapes current-season (2026) standings/toplists for live predictions
- `src/injury_checker.py` — injury/withdrawal detection + severity estimation (`data/injury_flags.json`)
- `src/h2h_calculator.py` / `data/h2h/h2h_rates.csv` — head-to-head win rates, a trained feature
- `tests/` — unit tests, `python -m pytest`
- `src/components/dl/shell.tsx`, `src/lib/dl-data.ts` (track-insights-main) — dashboard shell + API data contract

## Failed Attempts (tried, verified, did not help — don't redo without new evidence)

- **Event-specific models** (separate RandomForest per event group). Scored identically to one pooled model — `season_rank`/`season_percentile` already normalize per-discipline, so separate models add nothing.
- **`h2h_meetings` (count) as an additional feature** alongside `h2h_win_rate`: made results slightly worse. Not included.
- **Manual h2h blend at fixed weights**: tied with the trained-feature approach at best. Trained-feature chosen for architectural consistency (a manual post-hoc blend was the exact pattern that caused the recency-feature bug — didn't want to reintroduce it for h2h).
- **`src/scraper.py` (Wikipedia-based)**: fully non-functional, 0 rows ever. Deleted.
- **Including the DL Final's own results in `season_results_scraper.py`** (`rankingCategory in {"GW","DF"}`): produced a suspiciously large accuracy jump (63.8%) that turned out to be label leakage — the Final's own result was leaking into that year's training features. Fixed by restricting to `"GW"` only. **Lesson: a suspiciously good number is a signal to check harder, not a result to bank.**

## Known Limitations (real, understood, not bugs to re-chase)

- `season_results_scraper.py` only covers the Diamond League circuit, not an athlete's entire season everywhere they raced. An athlete who mostly raced non-DL meets still shows thinner history than reality. Real per-athlete profile scraping would close this fully but is a much bigger undertaking.
- `men_5000m` has real labels for only 1 of 5 years (2024) — other years ran a 5km road race or didn't contest it in a form the scraper recognized. Auto-detected from real data, not hand-flagged, but the practical effect is thin signal for this one discipline.
- Only 2021-2025 are covered for training labels. 2018/2019 used a different two-meeting scoring format; 2020 was a COVID-era exhibition. Extending further back needs separate handling for the different format, not just widening the year range.
- Two specific qualified athletes (Agnes Jebet Ngetich, women's 5000m; Yemisi Mabry, women's shot put) are genuinely absent from World Athletics toplists even at top-500 depth. Not fixable via more pagination.
- Women's Shot Put / Women's 5000m predictions are consistently short by 1-2 athletes.
- Projections page's per-meet trajectory chart is fabricated interpolation, honestly labeled but not rebuilt — deliberately deprioritized by the user.
- Minor, not acted on: `api.py` hardcodes `"qualified": true` for every athlete (harmless no-op); `lovable-error-reporting.ts` does nothing outside the Lovable editor; React Query is wired in but `usePredictions.ts` still does manual `fetch`/`useState`.

## Next Steps

1. **If pushing accuracy further**: extend `season_results_scraper.py` to real per-athlete profile history (not just the DL circuit) for the biggest remaining lever — or handle 2018-2020's different scoring format to add more labeled years.
2. **If expanding test coverage**: `build_labeled_dataset()`/`train_and_backtest()`/`load_predictions()` need small fixture files under `tests/fixtures/` to test without hitting real scraped data.
3. **Explicitly deprioritized polish, saved for last per the user**: landing/welcome page, READMEs for both repos, a real per-meet Projections chart, React Query refactor, mobile layout.
4. **Otherwise**: the system is in good shape 12 days out from the Final. Rerun `run.py` after Zurich (Aug 27) and again closer to Sep 4-5 to pick up final-season data — nothing else is currently broken or blocking.
