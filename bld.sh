#!/usr/bin/env zsh

# Exit immediately if a command fails
set -e

# Get the absolute path of the project root
ROOT_DIR=$(pwd)

echo "🚀 Moving to frontend directory..."
cd "$ROOT_DIR/frontend"

echo "🧹 Cleaning and fetching packages..."
flutter clean
flutter pub get

echo "🏗️ Building for Web (CanvasKit)..."

# Launch-time inputs are baked in at build time for web (see lib/launch_config.dart).
DEFINES=()
if [ -n "${SEGFORGE_IMAGE_URL:-}" ]; then
  DEFINES+=(--dart-define=SEGFORGE_IMAGE_URL="$SEGFORGE_IMAGE_URL")
fi
if [ -n "${SEGFORGE_SESSION_ID:-}" ]; then
  DEFINES+=(--dart-define=SEGFORGE_SESSION_ID="$SEGFORGE_SESSION_ID")
fi

flutter build web --debug --base-href /web/ "${DEFINES[@]}"

echo "✅ Build complete! Files are in frontend/build/web"
cd "$ROOT_DIR"