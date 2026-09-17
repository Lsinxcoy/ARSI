@echo off
REM ARSI Continuous Loop - Auto-run script
REM Set ARSI_API_KEY before running, or edit the line below

set PYTHONPATH=E:\ARSI\src
REM set ARSI_API_KEY=your-key-here
set HTTP_PROXY=http://127.0.0.1:7890
set HTTPS_PROXY=http://127.0.0.1:7890

echo Starting ARSI Continuous Loop (watch mode, 5 min interval)...
echo Press Ctrl+C to stop.
echo.

python E:\ARSI\scripts\continuous_loop.py --watch --interval 300
