@echo off
REM ARSI Daemon Startup Script
REM API key MUST come from environment — never hardcode secrets in this file.
set PYTHONPATH=E:\ARSI\src
if "%ARSI_API_KEY%"=="" (
  echo [ERROR] ARSI_API_KEY is not set. Set it in the parent shell before running.
  echo Example: set ARSI_API_KEY=... then run this script.
  exit /b 1
)
set PYTHONHOME=E:\MIMOdesktop\Xiaomi MiMo\resources\runtimes\win32-x64\python
set TEMP=E:\ARSI\archive\tmp
set TMP=E:\ARSI\archive\tmp
if not exist "E:\ARSI\archive\tmp" mkdir "E:\ARSI\archive\tmp"

echo Starting ARSI Daemon...
echo Python: %PYTHONHOME%\python.exe
echo API Key: present ^(from env, length %ARSI_API_KEY:~0,3%***^)
echo.

"%PYTHONHOME%\python.exe" E:\ARSI\scripts\arsi_daemon.py --tick-sleep 300
