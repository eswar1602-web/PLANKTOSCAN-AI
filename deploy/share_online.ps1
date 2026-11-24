# Share the app on a public link from this PC (free Cloudflare quick tunnel, no account needed).
#   powershell -ExecutionPolicy Bypass -File deploy\share_online.ps1
# Keep this window open: closing it (or putting the PC to sleep) takes the link offline.
# The link changes every time you run this script.

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (-not (Get-Command cloudflared -ErrorAction SilentlyContinue)) {
    Write-Host "cloudflared is not installed. Run once:  winget install --id Cloudflare.cloudflared" -ForegroundColor Red
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
    cloudflared tunnel --url http://127.0.0.1:8000
} finally {
    if ($server -and -not $server.HasExited) { Stop-Process -Id $server.Id -Force }
}
