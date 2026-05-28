#!/bin/bash

# SAM3 Segmentation Studio - Launcher
# Starts the FastAPI backend and the Flutter macOS desktop app.
# Usage: ./run.sh [--clear-storage] [--no-build]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BACKEND_DIR="$SCRIPT_DIR/app/backend"
FRONTEND_DIR="$SCRIPT_DIR/app/frontend"
APP_BUNDLE="$FRONTEND_DIR/build/macos/Build/Products/Debug/frontend.app"

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}╔════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   SAM3 Segmentation Studio Launcher    ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════╝${NC}"
echo ""

for arg in "$@"; do
  if [ "$arg" == "--clear-storage" ]; then
    echo -e "${YELLOW}Clearing storage ($SCRIPT_DIR/storage)...${NC}"
    rm -rf "$SCRIPT_DIR/storage"
    echo -e "${GREEN}Storage cleared.${NC}"
  fi
done

PIDS=()

cleanup() {
    echo ""
    echo -e "${YELLOW}Shutting down...${NC}"
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -TERM "$pid" 2>/dev/null || true
        fi
    done
    pkill -f "frontend.app/Contents/MacOS/frontend" 2>/dev/null || true
    sleep 2
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -KILL "$pid" 2>/dev/null || true
        fi
    done
    echo -e "${GREEN}All processes stopped.${NC}"
    exit 0
}

trap cleanup SIGINT SIGTERM EXIT

# Backend dependencies
echo -e "${YELLOW}Ensuring backend dependencies...${NC}"
cd "$PROJECT_ROOT"
uv pip install -r "$BACKEND_DIR/requirements.txt" --quiet

# Build Flutter macOS app (skip with --no-build)
if [[ " $* " == *" --no-build "* ]]; then
    echo -e "${YELLOW}Skipping Flutter build (--no-build).${NC}"
else
    echo -e "${YELLOW}Building Flutter macOS app...${NC}"
    cd "$FRONTEND_DIR"
    flutter build macos --debug
    cd "$SCRIPT_DIR"
fi

# Start backend
echo -e "${GREEN}Starting backend on http://localhost:8000 ...${NC}"
cd "$BACKEND_DIR"
uv run uvicorn main:app --reload &
BACKEND_PID=$!
PIDS+=($BACKEND_PID)
cd "$SCRIPT_DIR"

# Launch desktop app
echo -e "${GREEN}Launching SAM3 Studio...${NC}"
open "$APP_BUNDLE"

echo ""
echo -e "${GREEN}════════════════════════════════════════${NC}"
echo -e "${GREEN}  Backend:  http://localhost:8000        ${NC}"
echo -e "${GREEN}  API Docs: http://localhost:8000/docs   ${NC}"
echo -e "${GREEN}════════════════════════════════════════${NC}"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop.${NC}"
echo -e "${YELLOW}Tip: use --no-build to skip the Flutter build step.${NC}"
echo ""

set +e
wait "${PIDS[@]}" 2>/dev/null
set -e
