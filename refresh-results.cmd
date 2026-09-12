@echo off
rem Double-click to put the latest championship results on podiumcall.vercel.app.
rem It shows what changed and asks before pushing. See src\refresh_results.py.
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
"%~dp0venv\Scripts\python.exe" src\refresh_results.py %*
echo.
pause
