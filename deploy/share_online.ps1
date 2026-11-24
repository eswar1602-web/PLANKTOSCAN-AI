# Share the app on a public link from this PC (free Cloudflare quick tunnel, no account needed).
#   powershell -ExecutionPolicy Bypass -File deploy\share_online.ps1
# Keep this window open: closing it (or putting the PC to sleep) takes the link offline.
# The link changes every time you run this script.

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

# A terminal opened before the install does not have cloudflared on its PATH yet, so also look where winget puts it.
$cf = (Get-Command cloudflared -ErrorAction SilentlyContinue).Source
if (-not $cf) {
    $cf = @("$env:ProgramFiles\cloudflared\cloudflared.exe", "${env:ProgramFiles(x86)}\cloudflared\cloudflared.exe") |
        Where-Object { Test-Path $_ } | Select-Object -First 1
}
if (-not $cf) {
    Write-Host "cloudflared is not installed. Run once:  winget install --id Cloudflare.cloudflared" -ForegroundColor Red
    exit 1
}

# Port 8000 must be free, otherwise the health check below could be answered by some other server.
if (Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue) {
    Write-Host "Port 8000 is already in use (is the app already running in another window?). Close it and run this again." -ForegroundColor Red
    exit 1
}
if (-not (Test-Path "frontend\dist\index.html")) {
    Write-Host "Building the frontend first..."
    Push-Location frontend; npm run build; Pop-Location
}

# Start the server (models load in about 20 s) in the background.
$server = Start-Process -FilePath ".venv\Scripts\python.exe" -ArgumentList "-m", "api", "--port", "8000" `
    -PassThru -WindowStyle Minimized
try {
    Write-Host "Waiting for the models to load..."
    $ok = $false
    for ($i = 0; $i -lt 120; $i++) {
        try {
            $h = Invoke-RestMethod "http://127.0.0.1:8000/api/health" -TimeoutSec 2
            if ($h.models_loaded) { $ok = $true; break }
        } catch { }
        Start-Sleep -Seconds 1
    }
    if (-not $ok) { throw "The server did not start. Run  .venv\Scripts\python.exe -m api  to see the error." }
    Write-Host "Server ready (device: $($h.device)). Starting the tunnel: look for the https://....trycloudflare.com link below." -ForegroundColor Green
    & $cf tunnel --url http://127.0.0.1:8000
} finally {
    if ($server -and -not $server.HasExited) { Stop-Process -Id $server.Id -Force }
}
