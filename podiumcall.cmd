@echo off
rem Double-click this for every routine PodiumCall job, so no path ever has to be
rem typed. It runs each command from the right folder with this repo's venv Python.
rem The jobs and their order come from HANDOFF.md, "THE FULL REFRESH, IN ORDER".
rem COMMANDS.md has the same list as text, plus the jobs too risky for a menu.
setlocal
title PodiumCall commands
cd /d "%~dp0"
set "PY=%~dp0venv\Scripts\python.exe"
set "FRONTEND=%~dp0..\track-insights-main"
set PYTHONIOENCODING=utf-8

if not exist "%PY%" (
  echo Cannot find the virtual environment at:
  echo   %PY%
  echo.
  echo Create it with:  python -m venv venv
  echo Then:            venv\Scripts\python.exe -m pip install -r requirements.txt
  echo.
  pause
  exit /b 1
)

:menu
cls
echo ==============================================================
echo   PodiumCall
echo ==============================================================
echo.
echo   DURING A CHAMPIONSHIP
echo     1  Refresh the results and put them on the live site
echo        Asks before pushing. Run it after each day's session.
echo     2  Check for injuries and withdrawals
echo        Opens a real browser window. About 2 minutes.
echo.
echo   REFRESH THE DATA
echo     3  Full refresh, every pipeline in order
echo        Over an hour, needs Chrome. Skipping it leaves pages stale.
echo     4  Re-run the modelling only, no scraping
echo        Seconds. Use it to check a change to prediction logic.
echo.
echo   REBUILD WHAT THE SITE SERVES
echo     5  Rebuild all the site's data files
echo        Over ten minutes. Run it after any refresh.
echo     6  Rebuild the main pages only
echo        Seconds. Results, championship and predictions.
echo     7  Rebuild the sitemap
echo.
echo   PHOTOS
echo     8  Fill in missing athlete photos
echo     9  Work out where the faces are in them
echo.
echo   RUN IT ON THIS COMPUTER
echo    10  Start the API on port 5000
echo    11  Start the website on port 8080
echo    12  Run the tests
echo.
echo     Q  Quit, or just press Enter
echo.
set "pick="
set /p "pick=Type a number and press Enter, or just Enter to quit: "
rem An empty answer quits. It also stops this looping forever if the script
rem is ever run with nothing on its input, where set /p leaves pick unset.
if not defined pick goto quit
if "%pick%"=="1"  goto results
if "%pick%"=="2"  goto injuries
if "%pick%"=="3"  goto fullrefresh
if "%pick%"=="4"  goto modelonly
if "%pick%"=="5"  goto rebuild
if "%pick%"=="6"  goto rebuildcore
if "%pick%"=="7"  goto sitemap
if "%pick%"=="8"  goto photos
if "%pick%"=="9"  goto faces
if "%pick%"=="10" goto api
if "%pick%"=="11" goto web
if "%pick%"=="12" goto tests
if /i "%pick%"=="q" goto quit
goto menu

:results
echo.
echo == Refreshing the results
"%PY%" src\refresh_results.py
goto done

:injuries
echo.
echo == Checking for injuries and withdrawals
"%PY%" src\injury_checker.py
goto done

:fullrefresh
echo.
echo This scrapes World Athletics and takes over an hour. Chrome must be installed.
set "sure="
set /p "sure=Start it? [y/N] "
if /i not "%sure%"=="y" goto done
echo.
echo == 1 of 7  Toplists, standings, injury flags and predictions
"%PY%" run.py || goto failed
echo.
echo == 2 of 7  Diamond League per-meeting results
"%PY%" src\current_season_scraper.py || goto failed
echo.
echo == 3 of 7  Days since last competed, meets this season
"%PY%" src\refresh_current_season_stats.py || goto failed
echo.
echo == 4 of 7  Worldwide race log
"%PY%" src\worldwide_scraper.py || goto failed
echo.
echo == 5 of 7  Athlete profiles
"%PY%" src\athlete_profile_scraper.py || goto failed
echo.
echo == 6 of 7  Country pages
"%PY%" src\country_index.py || goto failed
echo.
echo == 7 of 7  The site's data files
"%PY%" src\build_static_api.py || goto failed
echo.
echo Refreshed. Nothing is live yet: commit and push both repos.
goto done

:modelonly
echo.
echo == Re-running the modelling against the data already on disk
"%PY%" run.py --no-scrape
goto done

:rebuild
echo.
echo == 1 of 2  Country pages
"%PY%" src\country_index.py || goto failed
echo.
echo == 2 of 2  The site's data files
"%PY%" src\build_static_api.py || goto failed
echo.
echo Rebuilt. Nothing is live yet: commit and push both repos.
goto done

:rebuildcore
echo.
echo == Rebuilding the main pages
"%PY%" src\build_static_api.py --core-only
goto done

:sitemap
echo.
echo == Rebuilding the sitemap
set "PODIUMCALL_BASE_URL=https://www.podiumcall.cc"
pushd "%FRONTEND%"
"%PY%" scripts\make-sitemap.py
popd
goto done

:photos
echo.
echo == Filling in missing athlete photos
"%PY%" src\warm_card_photos.py
goto done

:faces
echo.
echo == Working out where the faces are
"%PY%" src\warm_photo_focus.py
goto done

:api
echo.
echo == API on http://localhost:5000   Press Ctrl+C to stop it
"%PY%" api.py
goto done

:web
echo.
echo == Website on http://localhost:8080   Press Ctrl+C to stop it
pushd "%FRONTEND%"
call npm run dev
popd
goto done

:tests
echo.
echo == Running the tests
"%PY%" -m pytest
goto done

:failed
echo.
echo ******************************************************
echo  That step failed, so the ones after it did not run.
echo  Fix it before pushing anything.
echo ******************************************************
goto done

:done
echo.
echo Finished. Press any key for the menu.
pause >nul
goto menu

:quit
endlocal
exit /b 0
