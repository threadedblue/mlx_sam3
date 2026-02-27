#!/bin/bash

# SAM3 Segmentation Studio - Development Server Launcher
# Starts both backend (FastAPI) and frontend (Next.js) servers

# Don't exit on error in cleanup - we want to clean up even if something fails
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BACKEND_DIR="$SCRIPT_DIR/backend"
FRONTEND_DIR="$SCRIPT_DIR/frontend"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   SAM3 Segmentation Studio Launcher    ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════╝${NC}"
echo ""

# Array to store process PIDs
PIDS=()

# Function to cleanup background processes on exit
cleanup() {
    echo ""
    echo -e "${YELLOW}Shutting down servers...${NC}"
    
    # Send SIGTERM to all processes for graceful shutdown
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -TERM "$pid" 2>/dev/null || true
        fi
    done
    
    # Wait up to 5 seconds for graceful shutdown
    sleep 2
    
    # Force kill any remaining processes
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo -e "${YELLOW}Force killing process $pid...${NC}"
            kill -KILL "$pid" 2>/dev/null || true
        fi
    done
    
    # Also kill any remaining jobs in this shell (macOS compatible)
    local remaining_jobs
    remaining_jobs=$(jobs -p 2>/dev/null)
    if [ -n "$remaining_jobs" ]; then
        echo "$remaining_jobs" | xargs kill -TERM 2>/dev/null || true
        sleep 1
        remaining_jobs=$(jobs -p 2>/dev/null)
        if [ -n "$remaining_jobs" ]; then
            echo "$remaining_jobs" | xargs kill -KILL 2>/dev/null || true
        fi
    fi
    
    echo -e "${GREEN}All servers stopped.${NC}"
    exit 0
}

trap cleanup SIGINT SIGTERM EXIT

# Install backend dependencies using uv (into the project's venv)
echo -e "${YELLOW}Ensuring backend dependencies...${NC}"
cd "$PROJECT_ROOT"
uv pip install -r "$BACKEND_DIR/requirements.txt" --quiet

# Check if frontend dependencies are installed
if [ ! -d "$FRONTEND_DIR/node_modules" ]; then
    echo -e "${YELLOW}Installing frontend dependencies...${NC}"
    cd "$FRONTEND_DIR" && npm install
    cd "$SCRIPT_DIR"
fi

echo -e "${GREEN}Starting Backend (FastAPI) on http://localhost:8000${NC}"
cd "$PROJECT_ROOT"
uv run python "$BACKEND_DIR/main.py" &
BACKEND_PID=$!
PIDS+=($BACKEND_PID)

# Wait a moment for backend to start
sleep 2

echo -e "${GREEN}Starting Frontend (Next.js) on http://localhost:3000${NC}"
cd "$FRONTEND_DIR"
npm run dev &
FRONTEND_PID=$!
PIDS+=($FRONTEND_PID)

echo ""
echo -e "${GREEN}════════════════════════════════════════${NC}"
echo -e "${GREEN}  Servers are running!${NC}"
echo -e "${GREEN}  Frontend: http://localhost:3000${NC}"
echo -e "${GREEN}  Backend:  http://localhost:8000${NC}"
echo -e "${GREEN}  API Docs: http://localhost:8000/docs${NC}"
echo -e "${GREEN}════════════════════════════════════════${NC}"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop all servers${NC}"
echo ""

# Wait for all processes (will exit when any process exits or on Ctrl+C)
# The EXIT trap will ensure cleanup happens
set +e  # Temporarily disable exit on error for wait
wait "${PIDS[@]}" 2>/dev/null
set -e  # Re-enable exit on error

