#!/usr/bin/env zsh

# Exit immediately if a command fails
set -e

# Get the absolute path of the project root
ROOT_DIR=$(pwd)

echo "🚀 Moving to frontend directory..."
cd "$ROOT_DIR/app/frontend"

echo "🧹 Cleaning and fetching packages..."
flutter clean
flutter pub get

echo "🏗️ Building for Web (CanvasKit)..."

flutter build web --debug --base-href /web/

echo "✅ Build complete! Files are in app/frontend/build/web"
cd "$ROOT_DIR"