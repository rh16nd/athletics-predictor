# PodiumCall — Prediction Engine & API

The Python side of **PodiumCall**: it scrapes real athletics results from World
Athletics, trains a machine-learning model, generates podium predictions for the
2026 Wanda Diamond League Final, and serves them as a JSON API.

The website that consumes this API lives in a separate repo:
[**track-insights**](https://github.com/rh16nd/track-insights) (React frontend).
Live site: **https://podiumcall.vercel.app** · this API:
**https://podiumcall.onrender.com**

---

## What it does

For each of the **32 disciplines** contested at the Diamond League Final, the
model estimates every contender's probability of finishing **on the podium
(top 3)** — before a single race is run. It is trained on what actually
happened at past Finals, not on opinion.

## How the model works

- **Target:** `dl_top3` — did this athlete finish in the top 3 of the Final?
  (It predicts *podium membership*, not who wins.)
- **Training data:** real Diamond League Final results **2018–2025** (excl. 2020),
  scraped from World Athletics' own public API — the ground truth of who actually
  medalled.
- **Features (14):** season best, career best, season rank & percentile,
  consistency, recent form, **head-to-head win rate**, meets contested, gap
  variability, age, and more — all computed from real per-meeting results, not a
  single toplist snapshot.
- **Algorithm:** `RandomForestClassifier` (scikit-learn).
- **Validation:** walk-forward — trained only on years *before* each test year,
  scored on 2021–2025 independently, then refit on all years for production.
- **Accuracy:** **72.8%** at the real task (picking 3 from the ~8–10 who actually
  contest a Final) and **58.8%** on the harder historical ruler (picking 3 from a
  discipline's whole ~101-athlete toplist). Both live in
  `outputs/model_metrics.json`.

## The pipeline (what produces the predictions)

1. **Scrape** — Selenium + World Athletics' GraphQL API pull toplists, live DL
   standings, per-meeting results, athlete profiles, and injury/withdrawal news.
2. **Build features** — `src/feature_builder.py` turns raw marks into the 14
   model inputs.
3. **Train** — `src/train_model.py` fits and validates the model.
4. **Predict** — `run.py` scores the current season and writes
   `outputs/predictions_latest.csv`.
5. **Serve** — `api.py` exposes it all as JSON.

## Running it locally

```bash
python -m venv venv && venv/Scripts/activate      # Windows; use source venv/bin/activate on macOS/Linux
pip install -r requirements.txt

python run.py            # full refresh: scrape → features → model → predictions (~1 hr, needs Chrome)
python run.py --no-scrape   # re-run only the modelling half against existing data (seconds)
python api.py            # serve the API (waitress, http://localhost:5000)
python -m pytest         # 339 tests
```

> A full `run.py` refresh is no longer the *whole* refresh — the per-meeting,
> worldwide, and athlete-profile pipelines are separate. See `HANDOFF.md` →
> "THE FULL REFRESH, IN ORDER" for the exact commands.

## The API (Flask, 11 endpoints)

`/api/predictions` · `/api/stats` · `/api/qualification` · `/api/depth` ·
`/api/discipline/<key>` · `/api/athlete/<key>/<name>` ·
`/api/projections/<key>` · `/api/news` · `/api/search` ·
`/api/athlete-status/<key>/<name>` · `/api/health`

Secure-by-default: `python api.py` runs the **waitress** production server with
the debugger off. Config via env vars — `PODIUMCALL_HOST`, `PODIUMCALL_PORT`,
`PODIUMCALL_CORS_ORIGINS`, and `PODIUMCALL_DEBUG=1` (developer opt-in only, never
in production).

## Deployment & the data-refresh loop

The API is hosted on **Render**, auto-deploying from `main`. The scraping needs a
real browser and stays on your machine — Render only *serves* the committed data.

**To refresh the live predictions:**
```bash
python run.py                       # (on your machine) scrape + rebuild
git add -f outputs/predictions_latest.csv data/standings.json data/standings_detail.json data/injury_flags.json data/raw/ data/worldwide/
git commit -m "Refresh data" && git push   # Render redeploys with the new data
```
The **push** is the step that's easy to forget — the site won't update without it.

## Repo map

`api.py` (Flask API) · `run.py` (prediction pipeline) · `src/` (scrapers,
feature builder, model, analytics) · `data/` (scraped inputs) · `outputs/`
(model + predictions) · `tests/` (pytest) · `HANDOFF.md` (the deep engineering
log — start there for anything non-obvious).

*Real-data predictions, validated with working ground truth. Not affiliated with
World Athletics or the Wanda Diamond League.*
