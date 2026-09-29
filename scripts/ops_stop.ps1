# Stop ARSI daemon
Get-CimInstance Win32_Process -Filter "Name like 'python%'" |
  Where-Object { $_.CommandLine -like '*arsi_daemon*' } |
  ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue; Write-Output ("killed " + $_.ProcessId) }
Start-Sleep -Seconds 1
Remove-Item E:\ARSI\archive\arsi.lock -Force -ErrorAction SilentlyContinue
Write-Output 'stopped'
