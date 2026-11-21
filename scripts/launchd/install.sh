#!/bin/bash
# Installs and loads the three launchd jobs for the current user.
# Uninstall: ./install.sh --uninstall
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
AGENTS="$HOME/Library/LaunchAgents"
JOBS=(picks closing settle)
DOMAIN="gui/$(id -u)"

for job in "${JOBS[@]}"; do
  label="com.bigdance.edges-$job"
  launchctl bootout "$DOMAIN/$label" 2>/dev/null || true
  if [ "${1:-}" = "--uninstall" ]; then
    rm -f "$AGENTS/$label.plist"
    echo "removed $label"
    continue
  fi
  mkdir -p "$AGENTS" "$HOME/bigdance-bets/logs"
  sed -e "s|__REPO__|$REPO|g" -e "s|__HOME__|$HOME|g" "$HERE/$label.plist" > "$AGENTS/$label.plist"
  launchctl bootstrap "$DOMAIN" "$AGENTS/$label.plist"
  echo "loaded $label"
done
