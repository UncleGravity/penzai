#!/usr/bin/env bash
# test.sh - compile and run the XRT BO probe and DMA loopback on the KR260.
set -euo pipefail
cd "$(dirname "$0")"

# shellcheck disable=SC1091
[[ -f config.env ]] || { echo "ERROR: missing config.env" >&2; exit 1; }
set -a
source config.env
set +a

: "${BOARD:?config.env must set BOARD}"
: "${BOARD_TMP:?config.env must set BOARD_TMP}"

echo "== [1/2] sync board tests -> $BOARD:$BOARD_TMP =="
ssh "$BOARD" "mkdir -p '$BOARD_TMP'"
scp board/test_bo.c board/dma_loopback.c "$BOARD:$BOARD_TMP/"

echo "== [2/2] build and run tests =="
ssh "$BOARD" "BOARD_TMP='$BOARD_TMP' bash -s" <<'REMOTE'
set -euo pipefail
cd "$BOARD_TMP"

gcc test_bo.c -o test_bo -lxrt_coreutil
gcc dma_loopback.c -o dma_loopback -lxrt_coreutil

echo "== XRT device =="
xrt-smi examine

echo
echo "== BO probe =="
./test_bo

echo
echo "== DMA loopback =="
sudo ./dma_loopback

echo
echo "== fan PWM =="
for h in /sys/class/hwmon/hwmon*; do
    [ -e "$h/name" ] || continue
    if [ "$(cat "$h/name")" = "pwmfan" ]; then
        for f in "$h"/pwm* "$h"/fan*_input; do
            [ -e "$f" ] && printf '%s=' "$(basename "$f")" && cat "$f"
        done
    fi
done
REMOTE
