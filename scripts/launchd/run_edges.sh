#!/bin/bash
# Wrapper for scheduled `bigdance edges` runs. Usage: run_edges.sh picks|closing|settle
#
# launchd doesn't read ~/.zshrc, so this pulls just the ODDS_API_KEY line out of it (unless the
# variable is already set). Override BIGDANCE_BIN / BETS_DIR via the environment if needed.
set -euo pipefail

MODE="${1:?usage: run_edges.sh picks|closing|settle}"
BIGDANCE_BIN="${BIGDANCE_BIN:-$HOME/bigdance/.venv/bin/bigdance}"
BETS_DIR="${BETS_DIR:-$HOME/bigdance-bets}"
mkdir -p "$BETS_DIR"

# Off-season (May-Oct): nothing to do
case "$(date +%-m)" in 5 | 6 | 7 | 8 | 9 | 10) exit 0 ;; esac

if [ -z "${ODDS_API_KEY:-}" ] && [ -f "$HOME/.zshrc" ]; then
  line="$(grep -E '^[[:space:]]*(export[[:space:]]+)?ODDS_API_KEY=' "$HOME/.zshrc" | tail -1 || true)"
  if [ -n "$line" ]; then eval "$line"; fi
fi
export ODDS_API_KEY="${ODDS_API_KEY:-}"

BETS="$BETS_DIR/bets.csv"
case "$MODE" in
  picks)
    "$BIGDANCE_BIN" edges --date "$(date +%Y-%m-%d)" --log "$BETS" \
      --snapshot "$BETS_DIR/snapshots.csv" ;;
  closing)
    # Costs an API request only when a logged bet tips off within the window (> 2x run interval, so one failed run isn't fatal)
    "$BIGDANCE_BIN" edges --update-closing "$BETS" --closing-window 35 \
      --snapshot "$BETS_DIR/snapshots.csv" ;;
  settle)
    "$BIGDANCE_BIN" edges --settle "$BETS" --snapshot "$BETS_DIR/snapshots.csv" ;;
  *)
    echo "unknown mode: $MODE" >&2
    exit 2 ;;
esac
