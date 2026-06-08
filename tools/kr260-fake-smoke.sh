#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONFIG="${PENZAI_CONFIG:-}"
if [[ -z "$CONFIG" ]]; then
  if [[ -f config.env ]]; then
    CONFIG=config.env
  elif [[ -f experiments/kr260-q1a8-matmul-bringup/config.env ]]; then
    CONFIG=experiments/kr260-q1a8-matmul-bringup/config.env
  fi
fi
if [[ -n "$CONFIG" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$CONFIG"
  set +a
fi

: "${BOARD:=ubuntu@kria}"
: "${BOARD_TMP:=/tmp/penzai}"
: "${PENZAI_PORT:=29090}"
: "${PENZAI_MEM:=fake}"
: "${PENZAI_HEAP_MIB:=16}"
: "${PENZAI_ROWS:=8}"
: "${PENZAI_COLS:=1}"
: "${PENZAI_K:=128}"
: "${PENZAI_MAX_REQUESTS:=11}"

BOARD_HOST="${BOARD_HOST:-${BOARD#*@}}"
BOARD_HOST="${BOARD_HOST%%:*}"
REMOTE_BIN="$BOARD_TMP/penzaid"
REMOTE_DEVICE="tcp:0.0.0.0:$PENZAI_PORT"
HOST_DEVICE="tcp:$BOARD_HOST:$PENZAI_PORT"
LOG="$(mktemp -t penzai-kr260-smoke.XXXXXX.log)"
HOST_LOG="$(mktemp -t penzai-kr260-host.XXXXXX.log)"

cleanup() {
  if [[ -n "${SSH_PID:-}" ]]; then
    kill "$SSH_PID" 2>/dev/null || true
    wait "$SSH_PID" 2>/dev/null || true
  fi
  ssh "$BOARD" "pkill -f '$REMOTE_BIN serve --device $REMOTE_DEVICE' 2>/dev/null || true" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "== build host + KR260 daemon =="
zig build

echo "== copy penzaid -> $BOARD:$REMOTE_BIN =="
ssh "$BOARD" "mkdir -p '$BOARD_TMP'"
scp zig-out/bin/penzaid "$BOARD:$REMOTE_BIN"
ssh "$BOARD" "chmod +x '$REMOTE_BIN'"

echo "== start penzaid on $BOARD ($REMOTE_DEVICE, mem=$PENZAI_MEM, max_requests=$PENZAI_MAX_REQUESTS) =="
ssh "$BOARD" "'$REMOTE_BIN' serve --device '$REMOTE_DEVICE' --mem '$PENZAI_MEM' --heap-mib '$PENZAI_HEAP_MIB' --max-requests '$PENZAI_MAX_REQUESTS'" >"$LOG" 2>&1 &
SSH_PID=$!

echo "== run host smoke -> $HOST_DEVICE =="
ok=0
for _ in $(seq 1 30); do
  if ./zig-out/bin/penzai matmul \
    --device "$HOST_DEVICE" \
    --rows "$PENZAI_ROWS" \
    --cols "$PENZAI_COLS" \
    --k "$PENZAI_K" >"$HOST_LOG" 2>&1; then
    cat "$HOST_LOG"
    ok=1
    break
  fi
  sleep 0.2
done

if [[ "$ok" != 1 ]]; then
  kill "$SSH_PID" 2>/dev/null || true
  wait "$SSH_PID" 2>/dev/null || true
  unset SSH_PID
  echo "ERROR: host smoke failed; host log follows" >&2
  cat "$HOST_LOG" >&2
  echo "ERROR: host smoke failed; daemon log follows" >&2
  cat "$LOG" >&2
  exit 1
fi

if ! wait "$SSH_PID"; then
  unset SSH_PID
  echo "ERROR: remote daemon failed; daemon log follows" >&2
  cat "$LOG" >&2
  exit 1
fi
unset SSH_PID

echo "== daemon log =="
cat "$LOG"
echo "== KR260 $PENZAI_MEM-memory TCP smoke passed =="
