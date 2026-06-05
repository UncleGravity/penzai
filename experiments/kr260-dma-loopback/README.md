# kr260-dma-loopback

## Objective

Validate the full KR260 (ZynqMP) bring-up on real hardware: build an AXI-DMA
loopback bitstream → load it → prove **XRT-for-memory + MMIO-poke control**
(the penzai split) by moving data through `MM2S→FIFO→S2MM` and checking
`src == dst`. Pure Xilinx IP, no custom RTL.

```
zynq_ultra_ps_e ─M_AXI_HPM0_FPD─▶ axi_dma.S_AXI_LITE       (control via /dev/mem)
axi_dma.M_AXIS_MM2S ─▶ axis_data_fifo ─▶ axi_dma.S_AXIS_S2MM   (stream loopback)
axi_dma.{M_AXI_MM2S,M_AXI_S2MM} ─▶ ps.S_AXI_HP0_FPD        (DDR via HP, 40-bit)
```

## Instructions

**1. Build bitstream** (Windows VM, Vivado/Vitis 2025.2 at `C:\AMDDesignTools\2025.2.1`):
```sh
./build.sh                 # from Host: syncs to VM, runs vivado+bootgen, pulls bit.bin back
```

**2. zocl driver** (once per board; bare image has none, apt's is stale 2.8.0):
```sh
git clone --depth 1 -b 2024.2 https://github.com/Xilinx/XRT
cd XRT/src/runtime_src/core/edge/drm/zocl && make
sudo make modules_install && sudo depmod -a && sudo modprobe zocl
sudo usermod -aG render,video ubuntu        # then re-login (else XRT sees 0 devices)
```

**3. Load + test** (from host):
```sh
scp fpga/out/loopback.bit.bin overlay/loopback_zocl.dts board/{test_bo.c,dma_loopback.c} ubuntu@kria:/tmp/
ssh ubuntu@kria 'cd /tmp && dtc -@ -O dtb -o penzai-loopback.dtbo loopback_zocl.dts \
  && sudo mkdir -p /lib/firmware/xilinx/penzai-loopback \
  && sudo rm -f /lib/firmware/xilinx/penzai-loopback/*.bit.bin \
  && sudo cp penzai-loopback.dtbo /lib/firmware/xilinx/penzai-loopback/ \
  && sudo cp loopback.bit.bin /lib/firmware/xilinx/penzai-loopback/penzai-loopback.bit.bin \
  && printf "{\"shell_type\":\"XRT_FLAT\",\"num_slots\":\"1\"}\n" | sudo tee /lib/firmware/xilinx/penzai-loopback/shell.json \
  && { sudo xmutil unloadapp 2>/dev/null || true; } \
  && sudo xmutil loadapp penzai-loopback'
ssh ubuntu@kria 'cd /tmp && gcc dma_loopback.c -o dma_loopback -lxrt_coreutil && sudo ./dma_loopback'
# expect: PASS: DMA loopback src==dst
```

## Files

```sh
build.sh           Mac driver: sync → build on VM → fetch bit.bin
fpga/build.tcl     Vivado BD (ps + axi_dma + axis_data_fifo) → loopback.bit + .xsa
fpga/build.bat     Windows entry: settings64 → vivado → bootgen
overlay/loopback.dts        Stage A: bitstream-only overlay (clone of starter-kit dtbo)
overlay/loopback_zocl.dts   Stage B: + zocl node inside the fpga-region fragment
board/test_bo.c             XRT BO alloc/map/addr/sync probe
board/dma_loopback.c        full XRT-BO + MMIO-DMA loopback, src==dst
fpga/gen_dtbo.py            unused alt (Vitis DTG); hand dts won instead
```

## Findings (validated 2026-06-03)

- Whole chain works: 2025.2 bitstream loads on the 2024.1 board, zocl up,
  `xrtBOAlloc/Map/Address/Sync` OK, DMA loopback `src==dst` (4 KB).
- **BOs land in high DDR (>4 GB)** → DMA needs `c_addr_width 40` and the SA/DA
  **MSB** registers (0x1C / 0x4C). A 32-bit DMA cannot reach XRT buffers.
- **No xclbin needed** for BO alloc (`grp=0`) once the zocl node is up.
- DMA control base = `0xA0000000`.

## Notes

- zocl node must be **inside** the fpga-region fragment; `xmutil`/dfx-mgr drops
  any other fragment. (Same recipe captured in memory `kr260-xrt-bringup`.)
- `xrt-smi`/XRT showing "0 devices" = missing `render` group, not a real fault.
- `/dev/mem` (MMIO control) needs root; UIO is the future non-root path.
- AXI wired by hand (proc_sys_reset + 2 SmartConnects) — 2025.2's `axi4`
  automation rule fails for the dma-master→PS-slave direction.
- Kria-PYNQ is not an option (rejects Ubuntu 24.04).
