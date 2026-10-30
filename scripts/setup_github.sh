#!/usr/bin/env bash
# Usage: ./scripts/setup_github.sh [repo-name]   (needs git; uses gh CLI if installed and logged in)
set -e
NAME="${1:-kavachx-edge}"
git init -q 2>/dev/null || true
git add -A
git commit -qm "KavachX edge detection: MOG2 + YOLOv8n fire/smoke + person pipeline" || true
git branch -M main
if command -v gh >/dev/null && gh auth status >/dev/null 2>&1; then
  gh repo create "$NAME" --private --source=. --remote=origin --push
else
  echo "gh CLI not logged in. Create an empty repo named $NAME on GitHub, then run:"
  echo "  git remote add origin git@github.com:<your-username>/$NAME.git && git push -u origin main"
fi
