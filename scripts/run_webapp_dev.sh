#!/usr/bin/env bash
# Quick local demo of the web dashboard: starts the backend and frontend together and stops both
# on Ctrl+C. Linux, macOS, or Windows Git Bash. Run from anywhere; it finds the repo root itself.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -f .env ]; then
    echo "No .env found. Copying .env.example to .env now."
    cp .env.example .env
fi

# Loaded line by line, never `source`d: a bcrypt hash from set_admin_password.py contains
# literal `$2b$12$...`, which `source`/`set -a` would parse as shell parameter expansion
# (e.g. `$2`), corrupting the value or aborting under `set -u`.
while IFS='=' read -r key value; do
    [[ -z "$key" || "$key" == \#* ]] && continue
    export "$key=$value"
done < <(grep -v '^[[:space:]]*#' .env | grep -v '^[[:space:]]*$')

if [ -z "${TW_WEBAPP_SECRET_KEY:-}" ] || [ "${TW_WEBAPP_SECRET_KEY}" = "changeme-generate-a-real-one" ]; then
    echo "No admin account is set up yet. Run this first:"
    echo ""
    echo "    python scripts/set_admin_password.py"
    echo ""
    echo "Then paste the three lines it prints into your .env file and run this script again."
    exit 1
fi

if ! command -v uvicorn > /dev/null 2>&1; then
    echo "uvicorn not found. With your virtual environment active, run:"
    echo ""
    echo "    pip install -r requirements-webapp.txt"
    exit 1
fi

if ! command -v npm > /dev/null 2>&1; then
    echo "npm not found. Install Node.js 20+ (see docs/08_SETUP_AND_PORTABILITY.md section 2)."
    exit 1
fi

if [ ! -d webapp/frontend/node_modules ]; then
    echo "Installing frontend dependencies (first run only)..."
    (cd webapp/frontend && npm install)
fi

echo "Starting backend on http://127.0.0.1:8000 ..."
uvicorn webapp.backend.main:app --env-file .env --port 8000 &
BACKEND_PID=$!

echo "Starting frontend on http://localhost:5173 ..."
(cd webapp/frontend && npm run dev) &
FRONTEND_PID=$!

cleanup() {
    echo ""
    echo "Stopping..."
    kill "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null || true
}
trap cleanup INT TERM EXIT

echo ""
echo "Open http://localhost:5173 in your browser and log in with the account from set_admin_password.py."
echo "Press Ctrl+C here to stop both servers."
wait
