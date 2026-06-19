# AgriVision AI — clean start (kills stale server + clears Python cache)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "Stopping old Python processes on port 5000..."
Get-NetTCPConnection -LocalPort 5000 -ErrorAction SilentlyContinue |
    ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }

Get-ChildItem -Path . -Recurse -Directory -Filter "__pycache__" -ErrorAction SilentlyContinue |
    Where-Object { $_.FullName -notmatch '\\.venv\\' } |
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

Write-Host "Running API self-test..."
& .\.venv\Scripts\python.exe test_api.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "Self-test FAILED. Fix errors above before using the web UI." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Starting AgriVision server at http://127.0.0.1:5000" -ForegroundColor Green
& .\.venv\Scripts\python.exe app.py
