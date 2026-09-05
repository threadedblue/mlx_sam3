#!/bin/bash
# Usage:
#   ./run.sh BE [--clear-storage]   — start FastAPI backend
#   ./run.sh FE                     — start Flutter macOS app (hot reload available)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
BACKEND_DIR="$SCRIPT_DIR/backend"
FRONTEND_DIR="$SCRIPT_DIR/frontend"

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

MODE=$1

case "$MODE" in

  BE)
    for arg in "$@"; do
      if [ "$arg" == "--clear-storage" ]; then
        echo -e "${YELLOW}Clearing storage ($SCRIPT_DIR/storage)...${NC}"
        rm -rf "$SCRIPT_DIR/storage"
        echo -e "${GREEN}Storage cleared.${NC}"
      fi
    done

    echo -e "${BLUE}Starting backend on http://localhost:8000${NC}"
    echo -e "${YELLOW}API docs: http://localhost:8000/docs${NC}"
    echo ""
    cd "$PROJECT_ROOT"
    source "$PROJECT_ROOT/.venv/bin/activate"
    pip install -q -r "$BACKEND_DIR/requirements.txt" 2>&1 | grep -v "already satisfied"
    cd "$BACKEND_DIR"
    exec uvicorn main:app --reload
    ;;

  FE)
    echo -e "${BLUE}Starting Flutter macOS app${NC}"
    echo -e "${YELLOW}Hot reload: r   Hot restart: R   Quit: q${NC}"
    echo ""
    cd "$FRONTEND_DIR"
    flutter clean
    flutter pub get
    dart run flutter_launcher_icons 2>/dev/null || true
    exec flutter run -d macos
    ;;

  *)
    echo -e "${RED}Usage:${NC}"
    echo "  ./run.sh BE [--clear-storage]   start FastAPI backend"
    echo "  ./run.sh FE                     start Flutter macOS app"
    exit 1
    ;;

esac
