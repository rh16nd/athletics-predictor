@echo off
rem Double-click to put the latest championship results on www.podiumcall.cc.
rem Run it after each day's session. It fetches that day's results, rebuilds the
rem site's data, shows what changed and asks before pushing anything.
rem Pass --yes to skip the question. See src\refresh_results.py.
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
"%~dp0venv\Scripts\python.exe" src\refresh_results.py %*
echo.
pause
