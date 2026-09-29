# Backup ARSI production state (small set — not full archive/eval)
$ErrorActionPreference = 'Stop'
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$dest = "E:\ARSI\archive\backups\$stamp"
New-Item -ItemType Directory -Path $dest -Force | Out-Null
$files = @(
    'E:\ARSI\archive\world_pool_snapshot.json',
    'E:\ARSI\archive\arsi_checkpoint.json',
    'E:\ARSI\archive\iwm\siwm_vitals.json',
    'E:\ARSI\archive\harness\candidate_fails.json',
    'E:\ARSI\config\dream_rsi_params.yaml'
)
foreach ($f in $files) {
    if (Test-Path $f) { Copy-Item $f $dest -Force }
}
Get-ChildItem 'E:\ARSI\archive\iwm' -Filter '*.json' -ErrorAction SilentlyContinue | Copy-Item -Destination $dest -Force
Write-Output ("backup_ok " + $dest + " files=" + (Get-ChildItem $dest).Count)
