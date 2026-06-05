#!/usr/bin/env bash
# status.sh - show the loaded app, XRT device, fan PWM, and app package.
set -euo pipefail
cd "$(dirname "$0")"

# shellcheck disable=SC1091
[[ -f config.env ]] || { echo "ERROR: missing config.env" >&2; exit 1; }
set -a
source config.env
set +a

: "${BOARD:?config.env must set BOARD}"
: "${APP:?config.env must set APP}"

ssh "$BOARD" "APP='$APP' bash -s" <<'REMOTE'
set -euo pipefail

echo "== xmutil apps =="
sudo xmutil listapps

echo
echo "== XRT =="
xrt-smi examine || true

echo
echo "== fan / sensors =="
for h in /sys/class/hwmon/hwmon*; do
    [ -e "$h/name" ] || continue
    n=$(cat "$h/name")
    case "$n" in
        pwmfan|ams|ina260*)
            echo "$h name=$n"
            for f in "$h"/pwm* "$h"/fan*_input "$h"/temp*_input "$h"/in*_input; do
                [ -e "$f" ] && printf '  %s=' "$(basename "$f")" && cat "$f"
            done
            ;;
    esac
done

echo
echo "== firmware package =="
sudo ls -l "/lib/firmware/xilinx/$APP" || true
REMOTE
