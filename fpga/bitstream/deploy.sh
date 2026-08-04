#!/usr/bin/env bash
# deploy.sh - load the built COMBINED bitstream onto the KR260 via xmutil.
#
# Packages one exact successful run bundle into /lib/firmware/xilinx/<APP>, records
# its hash-verified deployment receipt, and runs `xmutil loadapp`. penzaid is
# deployed separately.
#
#   ./deploy.sh                       # uses VARIANT from config.env
#   ./deploy.sh w512-p4-f300

set -euo pipefail
cd "$(dirname "$0")"

[[ -f config.env ]] || { echo "ERROR: missing config.env; copy config.env.example first" >&2; exit 1; }
set -a; source config.env; set +a

: "${BOARD:?config.env must set BOARD (e.g. ubuntu@kria)}"
: "${BOARD_TMP:?config.env must set BOARD_TMP (e.g. /tmp/penzai-combined)}"
APP="${APP:-penzai-combined-v1}"
VARIANT="${1:-${VARIANT:-w512-p4-f300}}"
BIT_PREFIX="penzai-combined-v1"
case "$BOARD_TMP" in /tmp/*) ;; *) echo "ERROR: BOARD_TMP must be under /tmp" >&2; exit 1 ;; esac

SSH_ARGS=()
if [[ -n "${PENZAI_SSH_IDENTITY:-}" ]]; then
  SSH_ARGS=(-o IdentityAgent=none -o IdentitiesOnly=yes -i "$PENZAI_SSH_IDENTITY")
fi

BIT_NAME="$BIT_PREFIX-$VARIANT.bit.bin"
BIT="out/$BIT_NAME"
DTS_TEMPLATE="overlay/penzai-combined-v1.dts"
RUN_ID="${PENZAI_BITSTREAM_RUN_ID:-}"
RECEIPT="out/deployment_receipt.tsv"
[[ -f "$BIT" ]] || { echo "ERROR: missing $BIT. Run ./build.sh $VARIANT first." >&2; exit 1; }
[[ -f "$DTS_TEMPLATE" ]] || { echo "ERROR: missing $DTS_TEMPLATE." >&2; exit 1; }

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

tsv_value() {
  awk -F '\t' -v wanted="$2" '
    $1 == wanted { value = $2; count += 1 }
    END { if (count != 1) exit 2; print value }
  ' "$1"
}

if [[ -n "$RUN_ID" ]]; then
  [[ "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "ERROR: invalid PENZAI_BITSTREAM_RUN_ID" >&2; exit 1; }
  RUN_DIR="out/runs/$RUN_ID"
else
  [[ -L out/latest ]] || { echo "ERROR: out/latest does not identify a bitstream run bundle" >&2; exit 1; }
  RUN_DIR="$(cd out/latest && pwd -P)"
fi
MANIFEST="$RUN_DIR/manifest.tsv"
DRIVER_STATUS="$RUN_DIR/driver_status.tsv"
RUN_BIT="$RUN_DIR/$BIT_NAME"
[[ -f "$MANIFEST" && -f "$DRIVER_STATUS" && -f "$RUN_BIT" ]] || {
  echo "ERROR: incomplete run bundle: $RUN_DIR" >&2
  exit 1
}

RUN_ID="$(tsv_value "$MANIFEST" run_id)"
RUN_VARIANT="$(tsv_value "$MANIFEST" variant)"
GIT_COMMIT="$(tsv_value "$MANIFEST" git_commit)"
GIT_DIRTY="$(tsv_value "$MANIFEST" git_dirty)"
MANIFEST_SCHEMA="$(tsv_value "$MANIFEST" schema_version)"
SOURCE_HASH="$(tsv_value "$MANIFEST" source_sha256)"
BUILD_STATUS="$(tsv_value "$MANIFEST" status)"
HOST_STATUS="$(tsv_value "$DRIVER_STATUS" host_status)"
[[ "$RUN_VARIANT" == "$VARIANT" ]] || { echo "ERROR: bundle variant $RUN_VARIANT does not match $VARIANT" >&2; exit 1; }
[[ "$BUILD_STATUS" == "vivado_pass" && "$HOST_STATUS" == "complete" ]] || {
  echo "ERROR: run $RUN_ID did not complete successfully" >&2
  exit 1
}

BIT_HASH="$(sha256_file "$RUN_BIT")"
STABLE_BIT_HASH="$(sha256_file "$BIT")"
[[ "$BIT_HASH" == "$STABLE_BIT_HASH" ]] || {
  echo "ERROR: promoted $BIT does not match run bundle $RUN_ID" >&2
  exit 1
}
MANIFEST_HASH="$(sha256_file "$MANIFEST")"

RECEIPT_TMP="$RECEIPT.tmp.$$"
trap 'rm -f "$RECEIPT_TMP"' EXIT
{
  printf 'key\tvalue\n'
  printf 'receipt_schema_version\t1\n'
  printf 'manifest_schema_version\t%s\n' "$MANIFEST_SCHEMA"
  printf 'run_id\t%s\n' "$RUN_ID"
  printf 'variant\t%s\n' "$RUN_VARIANT"
  printf 'git_commit\t%s\n' "$GIT_COMMIT"
  printf 'git_dirty\t%s\n' "$GIT_DIRTY"
  printf 'source_sha256\t%s\n' "$SOURCE_HASH"
  printf 'manifest_sha256\t%s\n' "$MANIFEST_HASH"
  printf 'bitstream_sha256\t%s\n' "$BIT_HASH"
  printf 'bitstream_hash_verified\t1\n'
  printf 'deployed_utc\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'app\t%s\n' "$APP"
} > "$RECEIPT_TMP"
mv "$RECEIPT_TMP" "$RECEIPT"
trap - EXIT

echo "== copy run $RUN_ID -> $BOARD:$BOARD_TMP =="
ssh "${SSH_ARGS[@]}" "$BOARD" "rm -rf '$BOARD_TMP' && mkdir -p '$BOARD_TMP'"
scp "${SSH_ARGS[@]}" "$RUN_BIT" "$DTS_TEMPLATE" "$RECEIPT" "$BOARD:$BOARD_TMP/"

echo "== package + load app $APP (variant=$VARIANT) =="
ssh "${SSH_ARGS[@]}" "$BOARD" "APP='$APP' BOARD_TMP='$BOARD_TMP' BIT_NAME='$BIT_NAME' EXPECTED_BIT_HASH='$BIT_HASH' bash -s" <<'REMOTE'
set -euo pipefail
FW="/lib/firmware/xilinx/$APP"
cd "$BOARD_TMP"
ACTUAL_BIT_HASH="$(sha256sum "$BIT_NAME" | awk '{print $1}')"
[[ "$ACTUAL_BIT_HASH" == "$EXPECTED_BIT_HASH" ]] || {
  echo "ERROR: copied bitstream hash mismatch" >&2
  exit 1
}
GENERATED_DTS="$APP.generated.dts"
sed "s/@FIRMWARE_NAME@/$BIT_NAME/g" penzai-combined-v1.dts > "$GENERATED_DTS"
dtc -@ -O dtb -o "$APP.dtbo" "$GENERATED_DTS"
sudo mkdir -p "$FW"
sudo rm -f "$FW"/*.bit.bin "$FW"/*.dtbo "$FW"/shell.json "$FW"/deployment_receipt.tsv
sudo cp "$APP.dtbo" "$FW/$APP.dtbo"
sudo cp "$BIT_NAME" "$FW/$BIT_NAME"
printf '{"shell_type":"XRT_FLAT","num_slots":"1"}\n' | sudo tee "$FW/shell.json" >/dev/null
sudo xmutil unloadapp 2>/dev/null || true
sudo xmutil loadapp "$APP"
sudo cp deployment_receipt.tsv "$FW/deployment_receipt.tsv"
sudo xmutil listapps
REMOTE
echo "Done. Deployed run=$RUN_ID bitstream_sha256=$BIT_HASH"
echo "Restart penzaid with PENZAI_PL_OPS=all; query identity with: penzai capabilities --device tcp:<board>:29092"
