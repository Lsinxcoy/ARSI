# Start ARSI production daemon (single instance)
$ErrorActionPreference = 'Stop'
Set-Location E:\ARSI
$env:PYTHONPATH = 'E:\ARSI\src'
$env:TEMP = 'E:\ARSI\archive\tmp'
$env:TMP = 'E:\ARSI\archive\tmp'
Remove-Item Env:PYTHONHOME -ErrorAction SilentlyContinue

if (Test-Path E:\ARSI\.env) {
    $line = Get-Content E:\ARSI\.env | Where-Object { $_ -match '^ARSI_API_KEY=' } | Select-Object -First 1
    if ($line) { $env:ARSI_API_KEY = ($line -replace '^ARSI_API_KEY=','').Trim() }
}

$lock = 'E:\ARSI\archive\arsi.lock'
if (Test-Path $lock) {
    $old = Get-Content $lock -ErrorAction SilentlyContinue
    if ($old -and (Get-Process -Id $old -ErrorAction SilentlyContinue)) {
        Write-Output "already_running pid=$old"
        exit 0
    }
    Remove-Item $lock -Force
}

Start-Process -FilePath 'E:\ARSI\.venv\Scripts\python.exe' `
    -ArgumentList 'E:\ARSI\scripts\arsi_daemon.py','--tick-sleep','300' `
    -WorkingDirectory 'E:\ARSI' -WindowStyle Hidden
Start-Sleep -Seconds 6
$procs = Get-CimInstance Win32_Process -Filter "Name like 'python%'" | Where-Object { $_.CommandLine -like '*arsi_daemon*' }
Write-Output ("started " + ($procs.ProcessId -join ','))
