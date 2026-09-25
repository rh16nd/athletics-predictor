# Every command for the website

Two ways to use this. Double-click **`podiumcall.cmd`** in this folder and pick a
number, which is the way that needs no typing and no paths. Or copy a command
from below into a terminal.

`refresh-results.cmd` is still there and still works. It is the same job as menu
item 1, kept because it is the one you run daily during a championship.

## If you are pasting into a terminal

Your terminal is PowerShell. Paths look like `C:\Users\rayen\athletics-predictor`
or `C:/Users/rayen/athletics-predictor`. A path that starts `/c/Users/...` is Git
Bash syntax and PowerShell reads it as a folder literally called `c`, which fails
at `Set-Location` before running anything. That is what happened on 24 September.

Every command below assumes you start with:

```powershell
cd C:\Users\rayen\athletics-predictor
```

The website's own repo is `C:\Users\rayen\track-insights-main`.

## During a championship

Refresh the day's results and put them on the live site. It fetches the results,
rebuilds the pages, lists what changed, and asks before pushing. `--yes` skips the
question. It stops if either repo is off `main` or behind GitHub, and refuses a
scrape that came back with fewer events than the saved file already had.

```powershell
venv\Scripts\python.exe src\refresh_results.py
```

Check for injuries and withdrawals. Needs a real browser window, so do not run it
headless or minimised. About 2 minutes.

```powershell
venv\Scripts\python.exe src\injury_checker.py
```

Both of these act on whichever championship is current in `src/championships.py`.
Today that is the Asian Games.

## Refresh the data

`run.py` on its own is not a full refresh. It updates the toplists, standings,
injury flags and predictions, and leaves the per-meeting file, the worldwide race
log and the athlete profiles untouched, which go stale silently with nothing
broken to notice. Run all of these, in this order. Over an hour in total, and
Chrome has to be installed.

```powershell
venv\Scripts\python.exe run.py
venv\Scripts\python.exe src\current_season_scraper.py
venv\Scripts\python.exe src\refresh_current_season_stats.py
venv\Scripts\python.exe src\worldwide_scraper.py
venv\Scripts\python.exe src\athlete_profile_scraper.py
venv\Scripts\python.exe src\country_index.py
venv\Scripts\python.exe src\build_static_api.py
```

To re-run only the modelling against the data already on disk, which takes seconds
and scrapes nothing:

```powershell
venv\Scripts\python.exe run.py --no-scrape
```

## Rebuild what the site serves

The site reads precomputed JSON from its own repo, not the API. So a refresh is
not finished when this repo is pushed. Skip the rebuild and the API is fresh while
the site shows yesterday's numbers.

Everything, over ten minutes. `country_index.py` has to go first or the country
pages ship a refresh behind.

```powershell
venv\Scripts\python.exe src\country_index.py
venv\Scripts\python.exe src\build_static_api.py
```

The main pages only, in seconds. This is results, the championship and the
predictions, without the long athlete-profile pass.

```powershell
venv\Scripts\python.exe src\build_static_api.py --core-only
```

The sitemap, after a rebuild. The address is required and has no default, because
a guessed one would put a made-up host in front of every crawler.

```powershell
cd C:\Users\rayen\track-insights-main
$env:PODIUMCALL_BASE_URL = "https://www.podiumcall.cc"
C:\Users\rayen\athletics-predictor\venv\Scripts\python.exe scripts\make-sitemap.py
```

Then commit and push both repos. Vercel and Render redeploy on their own.

## Photos

Fill gaps in the athlete photos, then work out where the faces are so the cards
crop to them. Run them in that order, and re-run the first after a data refresh
or when a discipline's favourite changes.

```powershell
venv\Scripts\python.exe src\warm_card_photos.py
venv\Scripts\python.exe src\warm_photo_focus.py
```

## Run it on this computer

```powershell
venv\Scripts\python.exe api.py
```

The API comes up on http://localhost:5000. It does not reload when you edit
backend code unless `PODIUMCALL_DEBUG=1` is set, so restart it before judging a
page. The website, in its own terminal:

```powershell
cd C:\Users\rayen\track-insights-main
npm run dev
```

http://localhost:8080, or 8081 if 8080 is taken. The tests:

```powershell
venv\Scripts\python.exe -m pytest
```

## Deliberately not on the menu

These are all real commands, and all of them can do damage if run at the wrong
moment. Read the note before using one.

Re-scrape a championship's field and rebuild its call. About 20 minutes, then
about 10 more. **Do not run this once the call has been frozen and the
championship has started.** Finals are graded against the call as it stood before
anyone competed, and regenerating it would quietly change what the site claims it
predicted.

```powershell
venv\Scripts\python.exe src\asian_games_scraper.py
venv\Scripts\python.exe src\asian_games_predictions.py --refresh-races
```

Retrain the model. The flags are not optional decoration: without them this trains
a different model on different data and overwrites `outputs/` with it, and nothing
errors.

```powershell
venv\Scripts\python.exe src\train_model.py --with-recency --with-h2h --with-schedule --with-form --pooled --tiers dl_final,global,continental
```

Extend the toplists to older seasons. About 45 minutes, and **one at a time**:
every discipline file is shared, and a second process writing them at the same
time is a torn read waiting to happen.

```powershell
venv\Scripts\python.exe src\historical_scraper.py --years 2008-2017
```

Freeze the call before a championship's first session, which is what the Results
page later grades against.

```powershell
venv\Scripts\python.exe src\freeze_prefinal.py
```

`HANDOFF.md` has the rest, including the one-off caches, the calendar check and
the label rebuilds, under "THE FULL REFRESH, IN ORDER".
