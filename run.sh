#!/bin/bash
# Usage:
#   ./run.sh BE [--clear-storage]   — start FastAPI backend
#   ./run.sh FE [--url=URL] [--session=ID]
#                                   — start Flutter macOS app (hot reload available)
#
# The frontend takes the image URL and session id as launch-time inputs. Supply
# them with the flags above, or export SEGFORGE_IMAGE_URL / SEGFORGE_SESSION_ID.
# The session id is optional: the backend allocates one on the first upload.

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
    # Session id and image URL are launch-time inputs to the app. Take them from
    # the environment, overridable with --url= / --session= flags.
    IMAGE_URL="${SEGFORGE_IMAGE_URL:-}"
    SESSION_ID="${SEGFORGE_SESSION_ID:-}"
    for arg in "$@"; do
      case "$arg" in
        --url=*)     IMAGE_URL="${arg#--url=}" ;;
        --session=*) SESSION_ID="${arg#--session=}" ;;
      esac
    done

    DEFINES=()
    if [ -n "$IMAGE_URL" ]; then
      DEFINES+=(--dart-define=SEGFORGE_IMAGE_URL="$IMAGE_URL")
    fi
    if [ -n "$SESSION_ID" ]; then
      DEFINES+=(--dart-define=SEGFORGE_SESSION_ID="$SESSION_ID")
    fi

    echo -e "${BLUE}Starting Flutter macOS app${NC}"
    if [ -n "$IMAGE_URL" ]; then
      echo -e "${GREEN}Image URL: $IMAGE_URL${NC}"
    else
      echo -e "${YELLOW}No image URL: pass --url=... or export SEGFORGE_IMAGE_URL${NC}"
    fi
    if [ -n "$SESSION_ID" ]; then
      echo -e "${GREEN}Session:   $SESSION_ID${NC}"
    else
      echo -e "${YELLOW}No session id: the backend will allocate one on upload${NC}"
    fi
    echo -e "${YELLOW}Hot reload: r   Hot restart: R   Quit: q${NC}"
    echo ""
    cd "$FRONTEND_DIR"
    flutter clean
    flutter pub get
    dart run flutter_launcher_icons 2>/dev/null || true
    exec flutter run -d macos "${DEFINES[@]}"
    ;;

  *)
    echo -e "${RED}Usage:${NC}"
    echo "  ./run.sh BE [--clear-storage]              start FastAPI backend"
    echo "  ./run.sh FE [--url=URL] [--session=ID]     start Flutter macOS app"
    echo ""
    echo "  --url / --session may also be given as SEGFORGE_IMAGE_URL /"
    echo "  SEGFORGE_SESSION_ID environment variables."
    exit 1
    ;;

esac
