#!/usr/bin/env bash
# deploy.sh - load the built flash bitstream onto the KR260 via xmutil.
#
# Packages the .bit.bin + device-tree overlay into /lib/firmware/xilinx/<APP> and
# runs `xmutil loadapp`. penzaid is deployed separately. After this, penzaid's PL
# init should report the flash kernel ready (ID 0xF1A54A00).
#
#   ./deploy.sh           # uses VARIANT from config.env
#   ./deploy.sh f100

set -euo pipefail
cd "$(dirname "$0")"

[[ -f config.env ]] || { echo "ERROR: missing config.env; copy config.env.example first" >&2; exit 1; }
set -a; source config.env; set +a

: "${BOARD:?config.env must set BOARD (e.g. ubuntu@kria)}"
: "${BOARD_TMP:?config.env must set BOARD_TMP (e.g. /tmp/penzai-flash)}"
APP="${APP:-penzai-flash-v1}"
VARIANT="${1:-${VARIANT:-f100}}"
BIT_PREFIX="penzai-flash-v1"
case "$BOARD_TMP" in /tmp/*) ;; *) echo "ERROR: BOARD_TMP must be under /tmp" >&2; exit 1 ;; esac

BIT_NAME="$BIT_PREFIX-$VARIANT.bit.bin"
BIT="out/$BIT_NAME"
DTS_TEMPLATE="overlay/penzai-flash-v1.dts"
[[ -f "$BIT" ]] || { echo "ERROR: missing $BIT. Run ./build.sh $VARIANT first." >&2; exit 1; }
[[ -f "$DTS_TEMPLATE" ]] || { echo "ERROR: missing $DTS_TEMPLATE." >&2; exit 1; }

echo "== copy app inputs -> $BOARD:$BOARD_TMP =="
ssh "$BOARD" "rm -rf '$BOARD_TMP' && mkdir -p '$BOARD_TMP'"
scp "$BIT" "$DTS_TEMPLATE" "$BOARD:$BOARD_TMP/"

echo "== package + load app $APP (variant=$VARIANT) =="
ssh "$BOARD" "APP='$APP' BOARD_TMP='$BOARD_TMP' BIT_NAME='$BIT_NAME' bash -s" <<'REMOTE'
set -euo pipefail
FW="/lib/firmware/xilinx/$APP"
cd "$BOARD_TMP"
GENERATED_DTS="$APP.generated.dts"
sed "s/@FIRMWARE_NAME@/$BIT_NAME/g" penzai-flash-v1.dts > "$GENERATED_DTS"
dtc -@ -O dtb -o "$APP.dtbo" "$GENERATED_DTS"
sudo mkdir -p "$FW"
sudo rm -f "$FW"/*.bit.bin "$FW"/*.dtbo "$FW"/shell.json
sudo cp "$APP.dtbo" "$FW/$APP.dtbo"
sudo cp "$BIT_NAME" "$FW/$BIT_NAME"
printf '{"shell_type":"XRT_FLAT","num_slots":"1"}\n' | sudo tee "$FW/shell.json" >/dev/null
sudo xmutil unloadapp 2>/dev/null || true
sudo xmutil loadapp "$APP"
sudo xmutil listapps
REMOTE
echo "Done. Restart penzaid (as root) and check its PL init reports the flash kernel ready."
