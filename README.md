# PodiumCall — prediction engine & API

This is the Python half of PodiumCall. It scrapes real athletics results from World Athletics, trains a model on them, works out who's likely to make the podium at the 2026 Wanda Diamond League Final, and serves those predictions as a JSON API.

The website that reads this API lives in its own repo: [track-insights](https://github.com/rh16nd/track-insights), a React frontend. The live site is at **https://podiumcall.vercel.app** and this API at **https://podiumcall.onrender.com**.

---

## What it does

For each of the **32 disciplines** at the Final, the model gives every contender a probability of finishing **on the podium (top 3)**, worked out before any of them race. It's trained on what actually happened at past Finals, not on anyone's ranking or hunch.

## How the model works

- **Target:** `dl_top3`, whether an athlete finished top 3 in the Final. It predicts who makes the podium, not who wins.
- **Training data:** the real Diamond League Final results from 2018 to 2025 (2020 skipped), scraped from World Athletics' own public API. That's the ground truth: who actually medalled.
- **Features (14):** season best, career best, season rank and percentile, consistency, recent form, head-to-head win rate, meets contested, gap variability, age, and a few more. All of them come from real per-meeting results, not a single toplist snapshot.
- **Algorithm:** `RandomForestClassifier` (scikit-learn).
- **Validation:** walk-forward. For any test year the model only sees years before it, gets scored on 2021 through 2025 separately, then refits on every year for production.
- **Accuracy:** **72.8%** at the real task (picking 3 from the ~8–10 who actually contest a Final), and **58.8%** on the harder historical version (picking 3 from a discipline's whole ~101-athlete toplist). Both numbers live in `outputs/model_metrics.json`.

## The pipeline (what produces the predictions)

1. **Scrape.** Selenium plus World Athletics' GraphQL API pull toplists, live DL standings, per-meeting results, athlete profiles, and injury/withdrawal news.
2. **Build features.** `src/feature_builder.py` turns raw marks into the 14 model inputs.
3. **Train.** `src/train_model.py` fits and validates the model.
4. **Predict.** `run.py` scores the current season and writes `outputs/predictions_latest.csv`.
5. **Serve.** `api.py` hands it all out as JSON.

## Running it locally

```bash
python -m venv venv && venv/Scripts/activate      # Windows; use source venv/bin/activate on macOS/Linux
pip install -r requirements.txt

python run.py            # full refresh: scrape → features → model → predictions (~1 hr, needs Chrome)
python run.py --no-scrape   # re-run only the modelling half against existing data (seconds)
python api.py            # serve the API (waitress, http://localhost:5000)
python -m pytest         # 339 tests
```

> One catch: a full `run.py` isn't the *whole* refresh anymore. The per-meeting, worldwide, and athlete-profile pipelines run separately. `HANDOFF.md` → "THE FULL REFRESH, IN ORDER" has the exact commands.

## The API (Flask, 11 endpoints)

`/api/predictions` · `/api/stats` · `/api/qualification` · `/api/depth` ·
`/api/discipline/<key>` · `/api/athlete/<key>/<name>` ·
`/api/projections/<key>` · `/api/news` · `/api/search` ·
`/api/athlete-status/<key>/<name>` · `/api/health`

It's safe to run as-is: `python api.py` starts the **waitress** production server with the debugger off. Everything's configured through env vars: `PODIUMCALL_HOST`, `PODIUMCALL_PORT`, `PODIUMCALL_CORS_ORIGINS`, and `PODIUMCALL_DEBUG=1` (a developer opt-in, never set it in production).

## Deployment & the data-refresh loop

The API runs on **Render** and auto-deploys from `main`. Scraping needs a real browser, so it stays on your machine; Render only *serves* the data you commit.

To refresh the live predictions:

```bash
python run.py                       # (on your machine) scrape + rebuild
git add -f outputs/predictions_latest.csv data/standings.json data/standings_detail.json data/injury_flags.json data/raw/ data/worldwide/
git commit -m "Refresh data" && git push   # Render redeploys with the new data
```

The `push` is the easy step to forget, and without it the site just keeps showing the old numbers.

## Repo map

`api.py` (Flask API) · `run.py` (prediction pipeline) · `src/` (scrapers, feature builder, model, analytics) · `data/` (scraped inputs) · `outputs/` (model + predictions) · `tests/` (pytest) · `HANDOFF.md` (the deep engineering log; start there for anything non-obvious).

*Predictions are built from real data and checked against real ground truth. Not affiliated with World Athletics or the Wanda Diamond League.*
