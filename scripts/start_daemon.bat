@echo off
REM ARSI Daemon Startup Script — venv python + single instance
REM API key MUST come from environment or local .env — never hardcode secrets in this file.
set PYTHONPATH=E:\ARSI\src
set PYTHONHOME=
if exist "E:\ARSI\.env" (
  for /f "usebackq tokens=1,* delims==" %%A in ("E:\ARSI\.env") do (
    if /i "%%A"=="ARSI_API_KEY" set "ARSI_API_KEY=%%B"
  )
)
if "%ARSI_API_KEY%"=="" (
  echo [ERROR] ARSI_API_KEY is not set. Set it in the parent shell or E:\ARSI\.env
  exit /b 1
)
set TEMP=E:\ARSI\archive\tmp
set TMP=E:\ARSI\archive\tmp
if not exist "E:\ARSI\archive\tmp" mkdir "E:\ARSI\archive\tmp"

REM single instance: refuse if lock is live
if exist "E:\ARSI\archive\arsi.lock" (
  echo [ERROR] arsi.lock exists — another daemon may be running
  exit /b 1
)

echo Starting ARSI Daemon (venv)...
"E:\ARSI\.venv\Scripts\python.exe" E:\ARSI\scripts\arsi_daemon.py --tick-sleep 300
