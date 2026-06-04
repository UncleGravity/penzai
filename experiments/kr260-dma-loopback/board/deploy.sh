#!/usr/bin/env bash
# deploy.sh - run ON the KR260 (ubuntu@kria). Compiles the overlay, packages
# the app, loads it, and does Stage-A validation: confirm the PL came up and
# the AXI-DMA control registers are reachable over /dev/mem (no XRT yet).
#
#   scp fpga/out/loopback.bit.bin overlay/loopback.dts board/deploy.sh ubuntu@kria:/tmp/
#   ssh -t ubuntu@kria 'bash /tmp/deploy.sh'
#
# Set DMA_BASE from build.tcl's "DIAG: assigned addresses" output (the
# axi_dma S_AXI_LITE segment; on M_AXI_HPM0_FPD it is typically 0xA0000000).
set -euo pipefail
APP=penzai-loopback
SRC=/tmp
DMA_BASE="${DMA_BASE:-0xA0000000}"

echo "== compile overlay -> dtbo (dtc $(dtc --version 2>&1 | head -1)) =="
dtc -@ -O dtb -o "$SRC/$APP.dtbo" "$SRC/loopback.dts"

echo "== assemble app dir /lib/firmware/xilinx/$APP =="
sudo mkdir -p "/lib/firmware/xilinx/$APP"
sudo cp "$SRC/loopback.bit.bin" "/lib/firmware/xilinx/$APP/$APP.bit.bin"
sudo cp "$SRC/$APP.dtbo"         "/lib/firmware/xilinx/$APP/$APP.dtbo"
printf '{\n    "shell_type": "XRT_FLAT",\n    "num_slots": "1"\n}\n' \
    | sudo tee "/lib/firmware/xilinx/$APP/shell.json" >/dev/null

echo "== load =="
sudo xmutil unloadapp 2>/dev/null || true
sudo xmutil loadapp "$APP"

echo "== fpga_manager state =="; cat /sys/class/fpga_manager/fpga0/state
echo "== fpga_bridge(s) now present =="
for b in /sys/class/fpga_bridge/*/; do [ -e "$b/name" ] && echo "  $(cat "$b/name") state=$(cat "$b/state" 2>/dev/null)"; done
echo "== recent fpga/dma kernel msgs =="; sudo dmesg | tail -15 | grep -iE "fpga|dma|pl|overlay|axi" || true

echo "== Stage-A check: read AXI-DMA MM2S_DMASR @ $DMA_BASE + 0x04 via /dev/mem =="
sudo python3 - "$DMA_BASE" <<'PY'
import sys, os, mmap, struct
base = int(sys.argv[1], 0)
page = base & ~0xFFF
off  = base - page
fd = os.open("/dev/mem", os.O_RDWR | os.O_SYNC)
m = mmap.mmap(fd, 0x1000, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=page)
def rd(o): return struct.unpack("<I", m[off+o:off+o+4])[0]
dmacr = rd(0x00); dmasr = rd(0x04)
print(f"  MM2S_DMACR(0x00)=0x{dmacr:08x}  MM2S_DMASR(0x04)=0x{dmasr:08x}")
# After load+reset the MM2S channel should be Halted(bit0=1), not 0xffffffff
# (bus error / unclocked) and not all-zero.
if dmasr in (0x0, 0xffffffff):
    print("  RESULT: SUSPECT — 0x0 or 0xffffffff means the IP is unclocked/unreachable")
else:
    print("  RESULT: OK — DMA register page is live (PL clocked + AXI-Lite reachable)")
PY

echo "== (info) does XRT see a device yet? (expected: still 0 — no zocl node) =="
xrt-smi examine 2>/dev/null | sed -n '/Device(s) Present/,$p' || true
