# PodiumCall (2026 Diamond League Predictor) — Handoff

_Last updated: 2026-08-23 (fourth session, same day) — both repos committed and pushed_

## Goal

A fully automated ML prediction system for the 2026 Diamond League Final (Brussels, Sep 4-5), branded **PodiumCall**. It scrapes live World Athletics data, checks for injuries/withdrawals, runs a trained model, and serves win-probability predictions to a dashboard. Two repos:

```
C:\Users\rayen\athletics-predictor\   ← Python ML pipeline
C:\Users\rayen\track-insights-main\  ← React dashboard (TanStack Router + Vite), "PodiumCall" front end
```

**Priority order, per the user**: model accuracy and data honesty first; visual/UX polish was deliberately saved for last, then explicitly picked back up 2026-08-23 (see track-insights-main's Current State below — landing page, branding, and a design critique pass are now done). When in doubt about what to work on next, prefer real data over hardcoded/hand-typed data, and always isolate a change's effect before trusting a number that moved.

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

# Add real results from Olympics/Worlds/Continental Tour Gold/Euro Champs (~few min)
python src/major_meets_scraper.py

# Retrain the model (fast, seconds — uses already-scraped data/raw/*.csv)
python src/train_model.py --with-recency --with-h2h

# Grid-search hyperparameters via walk-forward folds (informational, prints only)
python src/train_model.py --with-recency --with-h2h --tune

# Run the test suite
python -m pytest
```

Both dev servers auto-reload on code changes (Flask debug mode, Vite HMR) and re-read data files fresh on every request — no restart needed after `run.py` or a retrain.

## Current State

**Model**: RandomForestClassifier, `n_estimators=200, max_depth=16, min_samples_leaf=1, class_weight=None` (`DEFAULT_MODEL_PARAMS` in `train_model.py` — walk-forward tuned via `--tune`, re-tune whenever the feature set, training-set size, OR data-quality filtering changes, since the winning config shifts each time — it's flipped `class_weight` None↔"balanced" twice across five re-tunes so far; hyperparameter search on a dataset this size (~459 rows) is itself somewhat noisy, so don't read too much into which single config "won" a given round). 14 features, **all carrying real signal**:
`season_best, career_best, pb_gap, meets_count, consistency, yoy_improvement, age, season_rank, season_percentile, weighted_season_best, wind_adj_season_best, recent_trend, days_since_last, h2h_win_rate`

**Backtest accuracy: 60.1%** (276/459) — walk-forward validated: trained on every year strictly before each test year, scored independently on 2021, 2022, 2023, 2024, and 2025 (5 folds, not one fixed holdout). The deployed model is refit on all 7 label years after validation. Trained on all 32 live-predicted disciplines (`men_5000m` has thin signal — see Known Limitations). This is essentially flat vs. the same-day earlier number (60.3%) despite two real additions since (major meets beyond the DL circuit, and a noise filter) — see the Failed Attempts / project memory for why that's an expected outcome of this dataset's size, not a wasted afternoon.

**Ground truth is real, not hand-typed.** `src/dl_final_results_scraper.py` pulls actual Diamond League Final results (2018-2025, excluding 2020) directly from World Athletics' own public GraphQL API — the same API the site's own frontend uses (`x-api-key` is a public key shipped in every page load, not a secret). It finds each year's Final by filtering `rankingCategory == "DF"` and reads which disciplines were contested (and under what name — e.g. "Mile" some years instead of "1500 Metres") directly from what's present in the response, rather than a hand-maintained list. **2018/2019 were initially assumed to need a different scoring format** (the Final was split across two meetings, Zurich + Brussels) and skipped — checking the actual per-meeting data (2026-08-23) showed each of the 32 disciplines' DF group appears at exactly one of the two meetings, never both, so it's a two-city Final, not a split score. The scraper now aggregates across however many DF meetings a year has instead of assuming exactly one. 2020 (COVID-era "Inspiration Games" exhibition) is still deliberately excluded.

**Training features go beyond a season-best toplist.** `src/season_results_scraper.py` pulls every *regular-season* Diamond League meeting's results (not just the Final — that would leak the label) for 2018-2025 (excluding 2020), giving real multiple-marks-per-athlete-per-season data. This is what makes `meets_count`/`consistency`/`recent_trend` real features instead of structural zeros.

**Beyond the DL circuit: `src/major_meets_scraper.py`** (added 2026-08-23, user-requested "meets with the big athletes") adds real per-meeting results from the Olympic Games, the senior outdoor World Athletics Championships, World Athletics Continental Tour **Gold tier only** (`rankingCategory == "A"`, filtered out of a much larger Silver/Bronze/Challenger pool), and the European Athletics Championships (the other five Area Championships — African/Asian/American/Oceania/NACAC — were deliberately left out per the user, quality varies too much by continent). These groups were found via `getCalendarEvents`'s own `options.competitionGroups`/`rankingCategories` introspection, not guessed.

**Both per-meeting scrapers filter to "recognized" athletes only** (`dl_final_results_scraper.load_recognized_names`): a row is kept only if that athlete appears somewhere in the discipline's own toplist (any year, not just that exact year — deliberately lenient about single-year toplist gaps). This exists because a big meeting can still have a weak field in a discipline that isn't its main draw (e.g. a field-filler making a Continental Tour Gold meeting's javelin final) — keeping those marks would add noise (thin, one-off meets_count/consistency signal) rather than real information about the athletes DL Final prediction cares about. Drop rates are meaningful (e.g. women_800m dropped 98 of 254 candidate major-meet rows) — this is doing real work, not a no-op filter.

**Injury/withdrawal detection is fully wired end-to-end.** `src/injury_checker.py` scrapes news + meet-results recaps for injury/DNF signals, estimates recovery time, and either flags ("watch") or drops ("remove") an athlete from predictions. Both outcomes are visible on the dashboard: a "Watch" badge (linking to the real evidence) on flagged athletes, and a "Removed from predictions" panel (shown only when non-empty) for dropped ones.

**Test suite**: `tests/` — pure-function unit tests for `train_model.py`, `api.py`, `dl_final_results_scraper.py` (`python -m pytest`, no network/Selenium/Flask/real-files needed), plus `test_fixtures_integration.py` covering `build_labeled_dataset()`/`train_and_backtest()`/`load_predictions()` end to end against small checked-in fixtures (`tests/fixtures/`) instead of real scraped data. 37 tests total. Includes regression tests for the specific bugs described below.

**Both repos are pushed and up to date** — track-insights-main `1bd80d1`, athletics-predictor `8d1db788`. (The medal→rank API change and the frontend nav/emoji/de-box work described below were committed separately, per repo, then both pushed.)

**track-insights-main (dashboard/frontend), 2026-08-23, first round:** the project was branded **PodiumCall** (domain-checked, `podiumcall.com`/`.io` both appeared unregistered) and a real landing page was built at `/`, with the former dashboard moved to `/dashboard` (sidebar nav updated). The landing page deliberately uses a dark theme structurally inspired by personaai.live (dark bg, gradient CTA, glass-bordered cards, a scrolling live-confidence ticker) but recolored to the dashboard's existing terracotta/gold/track-surface identity, per the user's "keep the track & field feel" direction — see `PRODUCT.md` (new, in that repo) for full product context, and `src/routes/index.tsx`/`src/styles.css`'s `.landing`-scoped tokens for the implementation. Installed several third-party Claude Code skills (`impeccable`, `design-taste-frontend`, Emil Kowalski's animation/design-eng set) via `npx skills add`/`npx impeccable install` — these land in `.agents/`/`.claude/skills/`, gitignored like the rest of the project's agent tooling. Ran `impeccable`'s dual-agent design critique against the new landing page and fixed everything it found: real page metadata (was shipping unedited "Lovable App" scaffold `<title>`/OG tags), a WCAG AA contrast failure on the accent colors against the dark background, zero `:focus-visible` styling anywhere in the project, and a non-technical fallback message for the hero stats when live data isn't reachable. Also designed a free hand-drawn SVG logo mark (`src/components/dl/logo.tsx` — a 2-1-3 podium on a track-lane arc, with a `light` variant for the sidebar's textured background) instead of paying for AI logo generation, wired in as the real favicon (`public/favicon.svg`).

**track-insights-main (dashboard/frontend), 2026-08-23, second round — user feedback: "you didn't actually use the skills, get rid of the emojis, and the dashboard needs a top nav instead of a sidebar, less boxy."** Real, substantive changes this round, not just installing skills:
- **Emoji removed end to end.** The 🥇🥈🥉🏅 markers were coming from `api.py`'s `build_top_winners()` (Python, not just the frontend) — fixed both sides. New `RankBadge` component (`src/components/dl/shell.tsx`) renders a colored circle using the same podium palette as the logo (gold/terracotta/brick for 1st/2nd/3rd, neutral gray for 4-6) instead of an emoji glyph.
- **Sidebar → top nav.** `src/components/dl/sidebar.tsx` deleted; `src/components/dl/topnav.tsx` is a new single horizontal bar (reuses the `track-surface` texture as a strip instead of a full-height panel) with the five nav links as pills. `Shell` (`shell.tsx`) no longer reserves 200px of left padding.
- **De-boxed**, per the (correctly-applied-this-time) `design-taste-frontend` skill's actual rule — "cards only when elevation communicates real hierarchy, otherwise group with spacing/dividers": the 4 separate bordered stat tiles on the dashboard are now one card with internal dividers; `Panel`'s outer border and header divider line are gone (relies on a subtle `bg-card`-vs-`bg-background` tone difference instead); Projections' 4 separate "storyline" boxes are now one card; all pill-style buttons (discipline pickers) are now `rounded-full`, matching the nav/CTA shape language instead of `rounded-md`.
- **Found and fixed 3 real pre-existing responsive bugs** while verifying at 375px (these were latent bugs surfaced by testing, not things introduced this round): the discipline table (6 fixed-width columns) blew out the page width on mobile — now wrapped in its own `overflow-x-auto` — and two multi-column grids (dashboard's winners/season-progress split, Projections' chart/confidence split) never collapsed to one column below `lg:`. All fixed and reverified.
- Verified via `tsc --noEmit` (clean), `eslint --fix` (clean, pre-existing warnings only), full Python test suite (37/37 passing), and live browser checks of all 6 routes at both 375px and desktop widths — the overflow bugs above were caught by actually testing, not assumed fixed from reading the code.

## Architecture / Key Files

- `run.py` — master pipeline: scrape live 2026 data → injury check → load trained model → build features → predict → `outputs/predictions_latest.csv`
- `api.py` — Flask bridge serving predictions + injury data to the dashboard (`/api/predictions`)
- `src/train_model.py` — retraining entry point. `--with-recency --with-h2h` for the full feature set; `--dry-run` to backtest without overwriting `outputs/`; `--tune` to grid-search hyperparameters
- `src/dl_final_results_scraper.py` — real DL Final ground-truth labels (`data/dl_final_results.csv`), scraped from WA's GraphQL API
- `src/season_results_scraper.py` — real per-meeting season history, enriches `data/raw/{discipline}.csv`
- `src/major_meets_scraper.py` — real per-meeting results from Olympics/Worlds/Continental Tour Gold/Euro Champs, also enriches `data/raw/{discipline}.csv`
- `src/historical_scraper.py` — season-best toplists 2018-2025 (`data/raw/{discipline}.csv` base layer)
- `src/live_fetcher.py` — scrapes current-season (2026) standings/toplists for live predictions
- `src/injury_checker.py` — injury/withdrawal detection + severity estimation (`data/injury_flags.json`)
- `src/h2h_calculator.py` / `data/h2h/h2h_rates.csv` — head-to-head win rates, a trained feature
- `tests/` — unit tests, `python -m pytest`
- `src/components/dl/shell.tsx`, `src/lib/dl-data.ts` (track-insights-main) — dashboard shell (page wrapper + `Panel`/`RankBadge`/`WatchBadge` primitives) + API data contract
- `src/components/dl/topnav.tsx` (track-insights-main) — the top navigation bar; replaced `sidebar.tsx` (deleted) 2026-08-23
- `src/routes/index.tsx` (track-insights-main) — the "/" landing page; `src/routes/dashboard.tsx` — the actual dashboard, moved here from "/"
- `src/components/dl/logo.tsx`, `PRODUCT.md` (track-insights-main) — PodiumCall's logo mark + durable product context for the `impeccable` skill

## Failed Attempts (tried, verified, did not help — don't redo without new evidence)

- **Event-specific models** (separate RandomForest per event group). Scored identically to one pooled model — `season_rank`/`season_percentile` already normalize per-discipline, so separate models add nothing.
- **`h2h_meetings` (count) as an additional feature** alongside `h2h_win_rate`: made results slightly worse. Not included.
- **Manual h2h blend at fixed weights**: tied with the trained-feature approach at best. Trained-feature chosen for architectural consistency (a manual post-hoc blend was the exact pattern that caused the recency-feature bug — didn't want to reintroduce it for h2h).
- **`src/scraper.py` (Wikipedia-based)**: fully non-functional, 0 rows ever. Deleted.
- **Including the DL Final's own results in `season_results_scraper.py`** (`rankingCategory in {"GW","DF"}`): produced a suspiciously large accuracy jump (63.8%) that turned out to be label leakage — the Final's own result was leaking into that year's training features. Fixed by restricting to `"GW"` only. **Lesson: a suspiciously good number is a signal to check harder, not a result to bank.**
- **Real per-athlete race-log scraping (all meets worldwide, not just the DL circuit)** (2026-08-23): confirmed via GraphQL introspection that WA's public API schema *has* fields for this (`getSingleCompetitorResultsDiscipline`/`getSingleCompetitor.resultsByDate`, taking a competitor id from `searchCompetitors`) — but the resolvers are dead server-side. Every combination of athlete id, year, and order-by argument tried (including with browser-matching headers) returned `resultsByDate: null`, while sibling fields on the same object (`activeYears`) resolved fine. `getSingleCompetitorSeasonBests` *does* work and confirms athletes' bests across all competitions (not just DL), but it's still one row per discipline per season — no more multi-meet granularity than what `season_results_scraper.py` already provides for the DL circuit specifically. **Don't re-attempt this without new evidence the resolver was fixed** — it's a dead field, not a query-construction mistake.

## Known Limitations (real, understood, not bugs to re-chase)

- `season_results_scraper.py` only covers the Diamond League circuit, not an athlete's entire season everywhere they raced. An athlete who mostly raced non-DL meets still shows thinner history than reality. Real per-athlete profile scraping would close this fully but WA's API doesn't currently expose it (see Failed Attempts above) — the only way to close this now would be scraping every meeting worldwide via the general calendar, a much bigger undertaking with real rate-limit/politeness concerns at that scale.
- `men_5000m` has real labels for only 1 of 5 years (2024) — other years ran a 5km road race or didn't contest it in a form the scraper recognized. Auto-detected from real data, not hand-flagged, but the practical effect is thin signal for this one discipline.
- 2020 alone is excluded from training labels (COVID-era "Inspiration Games" exhibition, not a real qualifying Final). 2018/2019 are now included — see the Ground Truth note above for why they turned out not to need special handling after all.
- Two specific qualified athletes (Agnes Jebet Ngetich, women's 5000m; Yemisi Mabry, women's shot put) are genuinely absent from World Athletics toplists even at top-500 depth. Not fixable via more pagination.
- Women's Shot Put / Women's 5000m predictions are consistently short by 1-2 athletes.
- Projections page's per-meet trajectory chart is fabricated interpolation, honestly labeled but not rebuilt — deliberately deprioritized by the user.
- Minor, not acted on: `api.py` hardcodes `"qualified": true` for every athlete (harmless no-op); `lovable-error-reporting.ts` does nothing outside the Lovable editor; React Query is wired in but `usePredictions.ts` still does manual `fetch`/`useState`.

## Next Steps

1. **If pushing accuracy further**: the easy levers are now applied — 2018/2019 label years, real per-athlete profile scraping (dead end, see Failed Attempts), and major non-DL meets (Olympics/Worlds/Continental Tour Gold/Euro Champs, plus a recognized-athlete noise filter) are all done (2026-08-23). None of it moved the headline number much (59.1%→60.3%→60.1%), which at this dataset size (~459 rows) may just be close to this feature set's ceiling. Remaining honest levers: scraping every meeting worldwide for true full-season per-athlete history (big undertaking, real rate-limit concerns), the other five Area Championships if the European-only scoping turns out too narrow, or finding genuinely new predictive features beyond the current 14.
2. **Test coverage**: done (2026-08-23) — `build_labeled_dataset()`/`train_and_backtest()`/`load_predictions()` are now covered end to end via `tests/fixtures/` + `tests/test_fixtures_integration.py`. If further expanding: `run.py`'s `build_2026_features()` and the injury-checker's scraping logic are still untested (both need Selenium/live network, harder to fixture).
3. **Polish, picked back up 2026-08-23**: landing page (done, round 1), branding/logo (done, round 1), sidebar→top-nav + emoji removal + de-boxing (done, round 2 — see "Uncommitted Right Now"). Mobile layout is now actually checked at 375px across all 6 routes (landing, dashboard, track, field, schedule, projections), not just the landing page. Still deprioritized, not yet started: READMEs for both repos, a real per-meet Projections chart (still fabricated interpolation, see Known Limitations), React Query refactor (`usePredictions.ts` still does manual `fetch`/`useState`). If the user wants another design pass, the installed `impeccable`/`design-taste-frontend`/Emil Kowalski skills are there — actually invoke them and follow their guidance (this session's lesson: installing a skill isn't the same as using it).
4. **Otherwise**: the system is in good shape 12 days out from the Final. Rerun `run.py` after Zurich (Aug 27) and again closer to Sep 4-5 to pick up final-season data — nothing else is currently broken or blocking.
