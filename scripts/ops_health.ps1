# ARSI production health probe
$ErrorActionPreference = 'Continue'
$ok = $true
$procs = Get-CimInstance Win32_Process -Filter "Name like 'python%'" | Where-Object { $_.CommandLine -like '*arsi_daemon*' }
if (-not $procs) { Write-Output 'FAIL daemon_not_running'; $ok = $false }
else {
    $real = @($procs) | Where-Object { $_.WorkingSetSize -gt 20MB }
    $pids = @($procs) | ForEach-Object { $_.ProcessId }
    $ws = @($procs) | ForEach-Object { [math]::Round($_.WorkingSetSize / 1MB, 1) }
    if (-not $real) { Write-Output 'WARN only_launcher_seen' }
    Write-Output ("OK daemon pids=" + ($pids -join ',') + " ws=" + ($ws -join ','))
}
$h = 'E:\ARSI\archive\arsi_health.jsonl'
if (Test-Path $h) {
    $last = Get-Content $h -Tail 1 | ConvertFrom-Json
    $age = ((Get-Date) - [datetime]$last.timestamp).TotalMinutes
    Write-Output ("health tick=" + $last.tick + " age_min=" + [math]::Round($age,1) + " pool=" + $last.world_pool_size + " traces=" + $last.trace_count)
    if ($age -gt 15) { Write-Output 'FAIL health_stale'; $ok = $false }
} else { Write-Output 'FAIL no_health'; $ok = $false }
if (Test-Path E:\ARSI\archive\arsi.lock) { Write-Output ('lock=' + (Get-Content E:\ARSI\archive\arsi.lock)) }
if ($ok) { Write-Output 'HEALTH_OK' } else { Write-Output 'HEALTH_FAIL'; exit 1 }
