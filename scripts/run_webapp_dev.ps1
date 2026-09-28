# Quick local demo of the web dashboard: opens the backend and frontend each in their own
# PowerShell window and opens the dashboard in the browser. Close either window to stop it.
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

if (-not (Test-Path .env)) {
    Write-Host "No .env found. Copying .env.example to .env now."
    Copy-Item .env.example .env
}

$secretKey = (Select-String -Path .env -Pattern '^TW_WEBAPP_SECRET_KEY=(.*)$').Matches.Groups[1].Value

if (-not $secretKey -or $secretKey -eq "changeme-generate-a-real-one") {
    Write-Host "No admin account is set up yet. Run this first:"
    Write-Host ""
    Write-Host "    python scripts\set_admin_password.py"
    Write-Host ""
    Write-Host "Then paste the three lines it prints into your .env file and run this script again."
    exit 1
}

if (-not (Get-Command uvicorn -ErrorAction SilentlyContinue)) {
    Write-Host "uvicorn not found. With your virtual environment active, run:"
    Write-Host ""
    Write-Host "    pip install -r requirements-webapp.txt"
    exit 1
}

if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
    Write-Host "npm not found. Install Node.js 20+ (see docs\08_SETUP_AND_PORTABILITY.md section 2)."
    exit 1
}

if (-not (Test-Path webapp\frontend\node_modules)) {
    Write-Host "Installing frontend dependencies (first run only)..."
    Push-Location webapp\frontend
    npm install
    Pop-Location
}

Write-Host "Starting backend on http://127.0.0.1:8000 in a new window ..."
Start-Process powershell -ArgumentList "-NoExit", "-Command", "uvicorn webapp.backend.main:app --env-file .env --port 8000"

Write-Host "Starting frontend on http://localhost:5173 in a new window ..."
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd webapp\frontend; npm run dev"

Start-Sleep -Seconds 3
Start-Process "http://localhost:5173"

Write-Host ""
Write-Host "Log in with the account from set_admin_password.py. Close the two new windows to stop."
