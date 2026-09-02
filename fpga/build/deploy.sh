#!/usr/bin/env bash
# deploy.sh - load the production Penzai bitstream onto the KR260 via xmutil.
#
# Packages one exact successful run bundle into /lib/firmware/xilinx/<APP>, records
# its hash-verified deployment receipt, and runs `xmutil loadapp`. penzaid is
# deployed separately.
#
#   ./deploy.sh                       # uses VARIANT from config.env
#   ./deploy.sh f225

set -euo pipefail
cd "$(dirname "$0")"

[[ -f config.env ]] || { echo "ERROR: missing config.env; copy config.env.example first" >&2; exit 1; }
set -a; source config.env; set +a

: "${BOARD:?config.env must set BOARD (e.g. ubuntu@kria)}"
: "${BOARD_TMP:?config.env must set BOARD_TMP (e.g. /tmp/penzai)}"
APP="${APP:-penzai}"
QUALIFIED_VARIANT=f225
VARIANT="${1:-${VARIANT:-f225}}"
BIT_PREFIX="penzai"
repo="$(cd ../.. && pwd)"
OUT_ROOT="${PENZAI_FPGA_OUT:-$repo/.zig-cache/fpga-build}"
[[ "$VARIANT" =~ ^f[0-9]+$ ]] || { echo "ERROR: invalid variant '$VARIANT'; expected f<MHz>" >&2; exit 1; }
[[ "$VARIANT" == "$QUALIFIED_VARIANT" ]] || {
  echo "ERROR: only the qualified $QUALIFIED_VARIANT target may be deployed" >&2
  exit 1
}
case "$BOARD_TMP" in /tmp/*) ;; *) echo "ERROR: BOARD_TMP must be under /tmp" >&2; exit 1 ;; esac

SSH_ARGS=(-o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=10)
if [[ -n "${PENZAI_SSH_IDENTITY:-}" ]]; then
  [[ -r "$PENZAI_SSH_IDENTITY" ]] || {
    echo "ERROR: PENZAI_SSH_IDENTITY is not readable: $PENZAI_SSH_IDENTITY" >&2
    exit 1
  }
  SSH_ARGS+=(-o IdentityAgent=none -o IdentitiesOnly=yes -i "$PENZAI_SSH_IDENTITY")
fi

BIT_NAME="$BIT_PREFIX-$VARIANT.bit.bin"
BIT="$OUT_ROOT/$BIT_NAME"
DTS_TEMPLATE="overlay/penzai.dts"
RUN_ID="${PENZAI_BITSTREAM_RUN_ID:-}"
RECEIPT="$OUT_ROOT/deployment_receipt.tsv"
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
  RUN_DIR="$OUT_ROOT/runs/$RUN_ID"
else
  [[ -L "$OUT_ROOT/latest" ]] || { echo "ERROR: $OUT_ROOT/latest does not identify a bitstream run bundle" >&2; exit 1; }
  RUN_DIR="$(cd "$OUT_ROOT/latest" && pwd -P)"
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
RUN_BUILD_MODE="$(tsv_value "$MANIFEST" build_mode)"
GIT_COMMIT="$(tsv_value "$MANIFEST" git_commit)"
GIT_DIRTY="$(tsv_value "$MANIFEST" git_dirty)"
MANIFEST_SCHEMA="$(tsv_value "$MANIFEST" schema_version)"
SOURCE_HASH="$(tsv_value "$MANIFEST" source_sha256)"
BUILD_STATUS="$(tsv_value "$MANIFEST" status)"
HOST_STATUS="$(tsv_value "$DRIVER_STATUS" host_status)"
[[ "$RUN_VARIANT" == "$VARIANT" ]] || { echo "ERROR: bundle variant $RUN_VARIANT does not match $VARIANT" >&2; exit 1; }
[[ "$RUN_BUILD_MODE" == clean ]] || {
  echo "ERROR: bundle $RUN_ID uses build mode $RUN_BUILD_MODE; deployment requires clean" >&2
  exit 1
}
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
  printf 'build_mode\t%s\n' "$RUN_BUILD_MODE"
  printf 'git_commit\t%s\n' "$GIT_COMMIT"
  printf 'git_dirty\t%s\n' "$GIT_DIRTY"
  printf 'source_sha256\t%s\n' "$SOURCE_HASH"
  printf 'manifest_sha256\t%s\n' "$MANIFEST_HASH"
  printf 'bitstream_sha256\t%s\n' "$BIT_HASH"
  printf 'bitstream_hash_verified\t1\n'
  printf 'deployed_utc\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'app\t%s\n' "$APP"
} > "$RECEIPT_TMP"

echo "== copy run $RUN_ID -> $BOARD:$BOARD_TMP =="
ssh "${SSH_ARGS[@]}" "$BOARD" "rm -rf '$BOARD_TMP' && mkdir -p '$BOARD_TMP'"
scp "${SSH_ARGS[@]}" "$RUN_BIT" "$DTS_TEMPLATE" "$BOARD:$BOARD_TMP/"
scp "${SSH_ARGS[@]}" "$RECEIPT_TMP" "$BOARD:$BOARD_TMP/deployment_receipt.tsv"

echo "== package + load app $APP (variant=$VARIANT) =="
rm -f "$RECEIPT"
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
sed "s/@FIRMWARE_NAME@/$BIT_NAME/g" penzai.dts > "$GENERATED_DTS"
dtc -@ -O dtb -o "$APP.dtbo" "$GENERATED_DTS"
sudo mkdir -p "$FW"
sudo rm -f "$FW"/*.bit.bin "$FW"/*.dtbo "$FW"/shell.json "$FW"/deployment_receipt.tsv
sudo cp "$APP.dtbo" "$FW/$APP.dtbo"
sudo cp "$BIT_NAME" "$FW/$BIT_NAME"
printf '{"shell_type":"XRT_FLAT","num_slots":"1"}\n' | sudo tee "$FW/shell.json" >/dev/null

# `xmutil unloadapp` requires a slot argument.  Calling it without one returns an
# error, while a following load of the same app name can appear successful without
# replacing the resident image.  This XRT_FLAT package declares one slot; unload
# that occupied slot and verify it is empty before trusting the new receipt.
listapps_before="$(sudo xmutil listapps)"
while read -r slot; do
  [[ -n "$slot" ]] || continue
  sudo xmutil unloadapp "$slot"
done < <(printf '%s\n' "$listapps_before" | awk '
  $NF ~ /^[0-9]+->[0-9]+,?$/ { split($NF, mapping, "->"); print mapping[1] }
')
listapps_after_unload="$(sudo xmutil listapps)"
if printf '%s\n' "$listapps_after_unload" | awk '
  $NF ~ /^[0-9]+->[0-9]+,?$/ { found = 1 }
  END { exit found ? 0 : 1 }
'; then
  echo "ERROR: an accelerator slot remained loaded after unload" >&2
  exit 1
fi
sudo xmutil loadapp "$APP"
listapps_after_load="$(sudo xmutil listapps)"
if ! printf '%s\n' "$listapps_after_load" | awk -v app="$APP" '
  $1 == app && $NF ~ /^[0-9]+->[0-9]+,?$/ { found = 1 }
  END { exit found ? 0 : 1 }
'; then
  echo "ERROR: $APP did not acquire an accelerator slot" >&2
  exit 1
fi
sudo cp deployment_receipt.tsv "$FW/deployment_receipt.tsv"
printf '%s\n' "$listapps_after_load"
REMOTE
mv "$RECEIPT_TMP" "$RECEIPT"
trap - EXIT
echo "Done. Deployed run=$RUN_ID bitstream_sha256=$BIT_HASH"
echo "Restart penzaid and inspect with: penzai inspect device --device tcp:<board>:29092"
