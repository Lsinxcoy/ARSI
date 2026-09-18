@echo off
REM ARSI Daemon Startup Script
set PYTHONPATH=E:\ARSI\src
set ARSI_API_KEY=sk-3801816ed09d2f5872997ba6c66b4b7a1b6da2acfa34ec5092c0057cbf56d441
set PYTHONHOME=E:\MIMOdesktop\Xiaomi MiMo\resources\runtimes\win32-x64\python

echo Starting ARSI Daemon...
echo Python: %PYTHONHOME%\python.exe
echo API Key: %ARSI_API_KEY:~0~20%...
echo.

"%PYTHONHOME%\python.exe" E:\ARSI\scripts\arsi_daemon.py --tick-sleep 300
