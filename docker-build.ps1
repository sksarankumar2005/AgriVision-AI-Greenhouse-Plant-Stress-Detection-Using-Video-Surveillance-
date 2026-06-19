# AgriVision AI — build, test, and optionally push Docker image
param(
    [string]$ImageName = "agrivision-ai",
    [string]$Tag = "latest",
    [string]$Registry = "",   # e.g. "yourdockerhubuser" → pushes yourdockerhubuser/agrivision-ai:latest
    [switch]$SkipTest,
    [switch]$Push
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path "runs\detect\train\weights\best.pt")) {
    Write-Host "ERROR: Missing runs\detect\train\weights\best.pt — train the model first (train.py)." -ForegroundColor Red
    exit 1
}

$LocalImage = "${ImageName}:${Tag}"
$RemoteImage = if ($Registry) { "${Registry}/${ImageName}:${Tag}" } else { $LocalImage }

Write-Host "`n=== 1) Building image (minimal files only) ===" -ForegroundColor Cyan
docker build -t $LocalImage .
if ($LASTEXITCODE -ne 0) { exit 1 }

if ($Registry) {
    Write-Host "`n=== Tagging for registry: $RemoteImage ===" -ForegroundColor Cyan
    docker tag $LocalImage $RemoteImage
}

if (-not $SkipTest) {
    Write-Host "`n=== 2) Testing container ===" -ForegroundColor Cyan
    docker rm -f agrivision-test 2>$null | Out-Null
    docker run -d --name agrivision-test -p 5000:5000 $LocalImage | Out-Null

  Write-Host "Waiting for app to start (up to 3 min on first run)..."
    $ok = $false
    for ($i = 0; $i -lt 36; $i++) {
        Start-Sleep -Seconds 5
        try {
            $r = Invoke-RestMethod -Uri "http://127.0.0.1:5000/health" -TimeoutSec 5
            if ($r.app_version) {
                Write-Host "Health OK — version $($r.app_version), model $($r.model)" -ForegroundColor Green
                $ok = $true
                break
            }
        } catch { }
    }

    docker stop agrivision-test | Out-Null
    docker rm agrivision-test | Out-Null

    if (-not $ok) {
        Write-Host "ERROR: Health check failed. Run: docker logs agrivision-test" -ForegroundColor Red
        exit 1
    }
}

if ($Push) {
    if (-not $Registry) {
        Write-Host "ERROR: Use -Registry yourdockerhubuser -Push" -ForegroundColor Red
        exit 1
    }
    Write-Host "`n=== 3) Pushing $RemoteImage ===" -ForegroundColor Cyan
    Write-Host "Run 'docker login' first if not already logged in."
    docker push $RemoteImage
    if ($LASTEXITCODE -ne 0) { exit 1 }
    Write-Host "Pushed: $RemoteImage" -ForegroundColor Green
}

Write-Host "`nDone." -ForegroundColor Green
Write-Host "  Run locally:  docker compose up"
Write-Host "  Or:           docker run -p 5000:5000 $LocalImage"
if ($Registry -and -not $Push) {
    Write-Host "  Push:         .\docker-build.ps1 -Registry $Registry -Push"
}
