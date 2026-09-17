@echo off
REM ARSI 24/7 Daemon - Start script
REM Set ARSI_API_KEY before running

set PYTHONPATH=E:\ARSI\src
set HTTP_PROXY=http://127.0.0.1:7890
set HTTPS_PROXY=http://127.0.0.1:7890

echo Starting ARSI Daemon (24/7 mode)...
echo Tick sleep: 300s (5 minutes)
echo Press Ctrl+C to stop gracefully.
echo.

python E:\ARSI\scripts\arsi_daemon.py --tick-sleep 300
