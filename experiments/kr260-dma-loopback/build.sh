#!/usr/bin/env bash
# build.sh — run from the Mac. One command per iteration:
#   1. sync the FPGA build files to the Windows VM
#   2. run the Vivado/bootgen build there (the ~15-30 min step)
#   3. pull loopback.bit.bin back into fpga/out/
#
#   ./build.sh
#
# Override host/dirs via env if they ever change:
#   VM=10.211.55.3  VM_DIR=kr260  ./build.sh
set -euo pipefail
cd "$(dirname "$0")"

VM="${VM:-10.211.55.3}"
VM_DIR="${VM_DIR:-kr260}"

echo "== [1/3] sync build files -> $VM:$VM_DIR =="
ssh "$VM" "if not exist $VM_DIR mkdir $VM_DIR"
scp fpga/build.tcl fpga/build.bat "$VM:$VM_DIR/"

echo "== [2/3] Vivado build on $VM (~15-30 min; streaming) =="
ssh "$VM" "cd $VM_DIR && build.bat"

echo "== [3/3] fetch outputs -> fpga/out/ =="
mkdir -p fpga/out
if ! scp "$VM:$VM_DIR/out/loopback.bit.bin" "$VM:$VM_DIR/out/loopback.bit" fpga/out/ 2>/dev/null; then
    echo "WARN: outputs not found on VM — build likely failed (scroll up for the error)."
    exit 1
fi

echo
echo "Done:"
ls -la fpga/out/loopback.bit.bin
cat <<'EOF'

Next — deploy to the board (Stage A):
  scp fpga/out/loopback.bit.bin overlay/loopback.dts board/deploy.sh ubuntu@kria:/tmp/
  ssh -t ubuntu@kria 'DMA_BASE=0xA0000000 bash /tmp/deploy.sh'
EOF
