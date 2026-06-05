#!/usr/bin/env bash
# deploy.sh - install and load the XRT/zocl loopback app on the KR260.
set -euo pipefail
cd "$(dirname "$0")"

# shellcheck disable=SC1091
[[ -f config.env ]] || { echo "ERROR: missing config.env" >&2; exit 1; }
set -a
source config.env
set +a

: "${BOARD:?config.env must set BOARD}"
: "${APP:?config.env must set APP}"
: "${BOARD_TMP:?config.env must set BOARD_TMP}"

BIT="fpga/out/loopback.bit.bin"
DTS="overlay/$APP.dts"

case "$BOARD_TMP" in
    /tmp/*) ;;
    *)
        echo "ERROR: BOARD_TMP must be under /tmp, got: $BOARD_TMP" >&2
        exit 1
        ;;
esac

if [[ ! -f "$BIT" ]]; then
    echo "ERROR: missing $BIT. Run ./build.sh first." >&2
    exit 1
fi
if [[ ! -f "$DTS" ]]; then
    echo "ERROR: missing $DTS." >&2
    exit 1
fi

echo "== [1/3] copy app inputs -> $BOARD:$BOARD_TMP =="
ssh "$BOARD" "rm -rf '$BOARD_TMP' && mkdir -p '$BOARD_TMP'"
scp "$BIT" "$DTS" board/test_bo.c board/dma_loopback.c "$BOARD:$BOARD_TMP/"

echo "== [2/3] package firmware app =="
ssh "$BOARD" "APP='$APP' BOARD_TMP='$BOARD_TMP' bash -s" <<'REMOTE'
set -euo pipefail
FW="/lib/firmware/xilinx/$APP"
cd "$BOARD_TMP"

dtc -@ -O dtb -o "$APP.dtbo" "$APP.dts"
sudo mkdir -p "$FW"
sudo rm -f "$FW"/*.bit.bin "$FW"/*.dtbo "$FW"/shell.json
sudo cp "$APP.dtbo" "$FW/$APP.dtbo"
sudo cp loopback.bit.bin "$FW/$APP.bit.bin"
printf '{"shell_type":"XRT_FLAT","num_slots":"1"}\n' | sudo tee "$FW/shell.json" >/dev/null
sudo ls -l "$FW"
REMOTE

echo "== [3/3] load $APP =="
ssh "$BOARD" "APP='$APP' bash -s" <<'REMOTE'
set -euo pipefail
sudo xmutil unloadapp 2>/dev/null || true
sudo xmutil loadapp "$APP"
sudo xmutil listapps
REMOTE

cat <<EOF

Loaded $APP.
Next:
  ./test.sh
EOF
