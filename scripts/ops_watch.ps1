# ARSI ops watch — disk + health freshness alerts (write incident log)
$ErrorActionPreference = 'Continue'
$inc = 'E:\ARSI\archive\ops_incidents.jsonl'
$fail = $false

# health freshness
$h = 'E:\ARSI\archive\arsi_health.jsonl'
if (Test-Path $h) {
    $last = Get-Content $h -Tail 1 | ConvertFrom-Json
    $age = ((Get-Date) - [datetime]$last.timestamp).TotalMinutes
    if ($age -gt 15) {
        $row = @{ ts = (Get-Date).ToString('o'); level = 'warn'; code = 'health_stale'; age_min = [math]::Round($age,1) } | ConvertTo-Json -Compress
        Add-Content $inc $row
        Write-Output 'ALERT health_stale'
        $fail = $true
    } else {
        Write-Output ("health_fresh_min=" + [math]::Round($age,1))
    }
}

# disk free on E:
$d = Get-PSDrive E -ErrorAction SilentlyContinue
if ($d) {
    $freeGB = [math]::Round($d.Free / 1GB, 2)
    Write-Output ("disk_free_gb=" + $freeGB)
    if ($freeGB -lt 5) {
        $row = @{ ts = (Get-Date).ToString('o'); level = 'crit'; code = 'disk_low'; free_gb = $freeGB } | ConvertTo-Json -Compress
        Add-Content $inc $row
        Write-Output 'ALERT disk_low'
        $fail = $true
    }
}

# daemon alive
$procs = Get-CimInstance Win32_Process -Filter "Name like 'python%'" | Where-Object { $_.CommandLine -like '*arsi_daemon*' }
if (-not $procs) {
    $row = @{ ts = (Get-Date).ToString('o'); level = 'crit'; code = 'daemon_down' } | ConvertTo-Json -Compress
    Add-Content $inc $row
    Write-Output 'ALERT daemon_down'
    $fail = $true
} else {
    Write-Output ("daemon_ok=" + (@($procs).Count))
}

if ($fail) { exit 1 } else { Write-Output 'WATCH_OK' }
