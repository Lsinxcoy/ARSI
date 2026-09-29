# Register ARSI production scheduled tasks (run as user with rights)
# Backup daily 02:00 · Watch every 30 min
$ErrorActionPreference = 'Stop'
schtasks /Create /F /TN "ARSI-ops-backup" /SC DAILY /ST 02:00 /TR "powershell -NoProfile -ExecutionPolicy Bypass -File E:\ARSI\scripts\ops_backup.ps1"
schtasks /Create /F /TN "ARSI-ops-watch" /SC MINUTE /MO 30 /TR "powershell -NoProfile -ExecutionPolicy Bypass -File E:\ARSI\scripts\ops_watch.ps1"
schtasks /Query /TN "ARSI-ops-backup"
schtasks /Query /TN "ARSI-ops-watch"
Write-Output 'tasks_registered'
