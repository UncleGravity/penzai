#!/usr/bin/env bash
# restore-starter.sh - switch the board back to the stock starter-kit app.
set -euo pipefail
cd "$(dirname "$0")"

# shellcheck disable=SC1091
[[ -f config.env ]] || { echo "ERROR: missing config.env" >&2; exit 1; }
set -a
source config.env
set +a

: "${BOARD:?config.env must set BOARD}"
: "${STARTER_APP:?config.env must set STARTER_APP}"

ssh "$BOARD" "STARTER_APP='$STARTER_APP' bash -s" <<'REMOTE'
set -euo pipefail
sudo xmutil unloadapp 2>/dev/null || true
sudo xmutil loadapp "$STARTER_APP"
sudo xmutil listapps
REMOTE
