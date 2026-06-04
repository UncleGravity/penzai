# kr260-dma-loopback

A minimal AXI-DMA loopback overlay for the KR260, to validate the ZynqMP
bring-up path end to end: build a bitstream → package it as a Kria app
(`bit.bin` + `dtbo` + `shell.json`) → `xmutil loadapp` → confirm the PL comes up
and the DMA is reachable → (Stage B) drive a real loopback with XRT memory.

**Zero custom RTL** — pure Xilinx IP:

```
zynq_ultra_ps_e ──M_AXI_HPM0_FPD──▶ axi_dma.S_AXI_LITE        (control: MMIO poke)
axi_dma.M_AXIS_MM2S ─▶ axis_data_fifo ─▶ axi_dma.S_AXIS_S2MM  (stream loopback)
axi_dma.{M_AXI_MM2S,M_AXI_S2MM} ──▶ ps.S_AXI_HP0_FPD          (DDR via HP port)
```

## Environment (confirmed on the actual machines)

| | value |
|---|---|
| Board | KR260, Ubuntu 24.04, kernel `-v2024.1-`, XRT 2.18.0, `dtc` 1.7.0 present |
| board_part | `xilinx.com:kr260_som:part0:1.1` |
| Build VM | Windows, `C:\AMDDesignTools\2025.2.1\{Vivado,Vitis}\`, `cmd` only (no bash/WSL) |
| Tooling note | 2025.2 has **no classic `xsct`** → DTG is now the Vitis **Python client/server** flow (`vitis -s script.py`). See `fpga/gen_dtbo.py`. |

**Two ways to make the dtbo, one per stage:**
- **Stage A** — hand-authored `overlay/loopback.dts`, a clone of the board's
  own known-good `k26-starter-kits.dtbo` (tiny: `firmware-name` + 4 FPD resets,
  targeting `fpga_full`), compiled with on-board `dtc`. No XSA, no Vitis.
- **Stage B** — `fpga/gen_dtbo.py` drives the Vitis 2025.2 DTG
  (`create_platform_component`, `cpu=psu_cortexa53`, `dt_overlay="1"`,
  `user_dtsi=zocl.dtsi`) to emit the full, vendor-correct overlay with a zocl
  node. Input is the `.xsa` from `build.tcl`.

## Files

```
fpga/build.tcl     Vivado: zynq_ultra_ps_e + axi_dma + axis_data_fifo → loopback.bit + .xsa
fpga/build.bat     Windows cmd entry: call settings64.bat → vivado → bootgen
fpga/gen_dtbo.py   Stage B: Vitis 2025.2 Python DTG → full zocl overlay (vitis -s)
overlay/loopback.dts   Stage A PL overlay (clone of starter-kit template)
board/deploy.sh    on-board: dtc → package app → xmutil loadapp → Stage-A check
```

## Flow

**1. Windows VM (cmd):** set the `VIVADO_SETTINGS` path at the top of
`build.bat` to your 2025.2 `settings64.bat`, then:
```
cd fpga
build.bat
```
→ `fpga\out\loopback.bit` + `loopback.bit.bin`. Note the **DMA control base**
printed under `DIAG: assigned addresses` (likely `0xA0000000`).

**2. Mac → board:**
```
scp fpga/out/loopback.bit.bin overlay/loopback.dts board/deploy.sh ubuntu@kria:/tmp/
ssh -t ubuntu@kria 'DMA_BASE=0xA0000000 bash /tmp/deploy.sh'   # use base from step 1
```

## Staged validation (so we learn one thing at a time)

- **Stage A — this package.** Load the overlay, confirm `fpga_manager` →
  `operating`, bridges appear, and reading `MM2S_DMASR` over `/dev/mem` returns
  a sane value (not `0x0`/`0xffffffff`). That proves: 2025.2 bitstream builds,
  loads on the 2024.1 board, PL is clocked, and AXI-Lite control is reachable —
  i.e. the **MMIO-poke control plane works**. No XRT needed; `xrt-smi` will
  still show 0 devices (the overlay has no zocl node, by design).
- **Stage B — DONE (validated 2026-06-03).** XRT-for-memory + DMA loopback work
  end-to-end on the KR260. The recipe (see also memory `kr260-xrt-bringup`):
  1. **Build `zocl` from source** — the bare image has no driver and apt's
     `xrt-dkms` is a stale 2.8.0. `git clone --depth 1 -b 2024.2 .../XRT`
     (2.18=2024.2), `cd src/runtime_src/core/edge/drm/zocl && make`,
     `sudo make modules_install && sudo depmod -a && sudo modprobe zocl`.
  2. **Load `overlay/loopback_zocl.dts`** — same as `loopback.dts` plus a
     `zyxclmm_drm`/`xlnx,zocl` node **inside the fpga-region fragment** (a
     separate fragment is dropped by dfx-mgr). Gives `/dev/dri/renderD128`.
  3. **`sudo usermod -aG render,video ubuntu`** (fresh login) or non-root XRT
     enumerates 0 devices (also the real reason `xrt-smi` showed 0).
  4. `board/test_bo.c` → XRT BO alloc/map/addr/sync OK (no xclbin needed,
     `grp=0`). `board/dma_loopback.c` → full `src==dst` loopback. BOs land in
     high DDR (>4 GB), so the 40-bit DMA + SA/DA MSB regs are mandatory.
     `/dev/mem` (MMIO control) still needs root; UIO is the future non-root path.

## Known second-pass risks

- AXI is wired **by hand** (proc_sys_reset + two SmartConnects), not via
  `apply_bd_automation` — 2025.2's axi4 rule fails to extract options for the
  axi_dma-master → PS-slave direction. Manual pin names fail fast (~30s) in BD
  build if any are wrong, so iteration is cheap.
- 2025.2-generated bitstream on a 2024.1 kernel: the fabric load is
  version-independent, but watch `dmesg` on load for any overlay/compatible
  complaints (Stage A's `dmesg` tail surfaces them).
