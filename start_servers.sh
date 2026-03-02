#!/usr/bin/env bash
# Start the smolvla-inspect backend (FastAPI) and frontend (Vite dev) servers.
# Usage: ./start_servers.sh [--base-dir ./outputs]

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"

# --- Load nvm for Node.js 22+ ---
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"
nvm use 22 2>/dev/null || true
BACKEND_PORT=8080
FRONTEND_PORT=5173
BASE_DIR="${1:---base-dir}"
BASE_DIR_VAL="${2:-./outputs}"

# --- Parse optional --base-dir flag ---
if [[ "${BASE_DIR}" == "--base-dir" ]]; then
    BASE_DIR_VAL="${BASE_DIR_VAL}"
else
    BASE_DIR_VAL="./outputs"
fi

# --- Find free port starting from a default ---
find_free_port() {
    local port=$1
    while ss -tlnp 2>/dev/null | grep -q ":${port} " || lsof -i ":${port}" &>/dev/null; do
        echo "  Port ${port} in use, trying $((port + 1))..." >&2
        port=$((port + 1))
    done
    echo "${port}"
}

BACKEND_PORT=$(find_free_port $BACKEND_PORT)
FRONTEND_PORT=$(find_free_port $FRONTEND_PORT)

echo "========================================"
echo " smolvla-inspect dev servers"
echo "========================================"
echo "  Backend:  http://localhost:${BACKEND_PORT}"
echo "  Frontend: http://localhost:${FRONTEND_PORT}"
echo "  Base dir: ${BASE_DIR_VAL}"
echo "========================================"
echo ""

# --- Cleanup on exit ---
cleanup() {
    echo ""
    echo "Shutting down servers..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null || true
    wait $BACKEND_PID $FRONTEND_PID 2>/dev/null || true
    echo "Done."
}
trap cleanup EXIT INT TERM

# --- Install frontend deps if needed ---
if [ ! -d "${PROJECT_ROOT}/web/frontend/node_modules" ]; then
    echo "Installing frontend dependencies..."
    (cd "${PROJECT_ROOT}/web/frontend" && npm install)
    echo ""
fi

# --- Start backend ---
echo "Starting backend on port ${BACKEND_PORT}..."
SMOLVLA_BASE_DIR="${BASE_DIR_VAL}" \
    "${PROJECT_ROOT}/.venv/bin/python" -m uvicorn web.backend.main:app \
    --host 0.0.0.0 \
    --port "${BACKEND_PORT}" \
    --reload \
    --reload-dir "${PROJECT_ROOT}/web/backend" &
BACKEND_PID=$!

# --- Start frontend ---
echo "Starting frontend on port ${FRONTEND_PORT}..."
(cd "${PROJECT_ROOT}/web/frontend" && \
    VITE_API_URL="http://localhost:${BACKEND_PORT}" \
    npx vite --port "${FRONTEND_PORT}" --host) &
FRONTEND_PID=$!

echo ""
echo "Press Ctrl+C to stop both servers."
wait
