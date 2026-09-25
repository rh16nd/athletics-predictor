@echo off
rem What Windows Task Scheduler runs each evening during a championship. It is the
rem same job as refresh-results.cmd, with three differences that matter when nobody
rem is watching: it never waits for a keypress, it answers the push question with
rem yes, and it writes everything it did to logs\refresh-<date>.log.
rem
rem   scheduled-refresh.cmd            the real thing: fetch, rebuild, push
rem   scheduled-refresh.cmd rehearse   everything except the push, for testing
rem
rem Its input is NUL on purpose. refresh_results.py still asks before pushing if a
rem scrape comes back with fewer events than the saved file had, and reading EOF
rem there means the answer is no, so a suspicious run holds off on its own.
setlocal
cd /d "%~dp0"
set "PY=%~dp0venv\Scripts\python.exe"
set PYTHONIOENCODING=utf-8

set "MODE=--yes"
set "WHAT=fetch, rebuild and push"
if /i "%~1"=="rehearse" (
  set "MODE="
  set "WHAT=rehearsal, nothing will be pushed"
)

if not exist "%PY%" (
  echo Cannot find %PY%
  exit /b 1
)
if not exist "%~dp0logs" mkdir "%~dp0logs"
call :stamp STAMP
set "TODAY=%STAMP:~0,10%"
set "LOG=%~dp0logs\refresh-%TODAY%.log"

>>"%LOG%" echo.
>>"%LOG%" echo ==============================================================
>>"%LOG%" echo %STAMP%  %WHAT%
>>"%LOG%" echo ==============================================================
>>"%LOG%" echo.
>>"%LOG%" echo Local commits a push would carry, backend:
>>"%LOG%" 2>&1 git log --oneline origin/main..HEAD
>>"%LOG%" echo Local commits a push would carry, frontend:
>>"%LOG%" 2>&1 git -C "%~dp0..\track-insights-main" log --oneline origin/main..HEAD
>>"%LOG%" echo.

"%PY%" src\refresh_results.py %MODE% >>"%LOG%" 2>&1 <NUL
set "RC=%ERRORLEVEL%"

call :stamp DONE
>>"%LOG%" echo.
if "%RC%"=="0" (
  >>"%LOG%" echo %DONE%  finished, exit code 0
) else (
  >>"%LOG%" echo %DONE%  FAILED, exit code %RC%. Nothing was pushed unless the log says otherwise.
)
exit /b %RC%

rem Sets the named variable to the local time as 2026-09-25T19:00:00.
:stamp
set "_f=%TEMP%\podiumcall-stamp.txt"
"%PY%" -c "import datetime;print(datetime.datetime.now().isoformat()[:19])" > "%_f%"
set "%~1="
set /p "%~1="<"%_f%"
del "%_f%" >nul 2>&1
goto :eof
