# kr260-xrt-loopback

Minimal KR260 experiment that proves the path this project needs:

```text
Zig -> XRT BO allocation -> AXI DMA MM2S -> AXIS FIFO -> AXI DMA S2MM -> XRT BO
```

The FPGA design is a small Xilinx-IP-only loopback fixture. The verifier is a
single Zig binary that loads XRT with `dlopen`, allocates BOs, maps `/dev/mem`,
programs the DMA registers, and checks that `src == dst`.

## Prerequisites

Host:

- SSH/SCP access to the Windows Vivado VM configured in `config.env`.
- SSH/SCP access to the KR260 configured in `config.env`.
- Zig 0.16.0, or `nix develop`.

Windows VM:

- Vivado/Vitis 2025.2 installed at the path in `fpga/build.bat`.

KR260:

- Xilinx Ubuntu with `dtc`, `xmutil`, and XRT userspace.
- `zocl` installed and loadable.
- Passwordless sudo, or an SSH session that can satisfy sudo prompts.

## Run

`config.env` is intentionally committed for this local setup. If it is missing,
the hardware steps fail immediately.

```sh
zig build all
```

That runs:

```sh
zig build bitstream  # build fpga/out/loopback.bit.bin on the Vivado VM
zig build deploy     # install/load the penzai-loopback XRT app on the KR260
zig build verify     # copy and run the Zig verifier as root on the KR260
```

Expected final output:

```text
xrt: device open OK
bo: alloc/map/address/sync PASS phys=0x...
dma: src=0x... dst=0x... n=4096 base=0xa0000000
dma: MM2S idle PASS status=0x...
dma: S2MM idle PASS status=0x...
verify: src == dst PASS
ALL PASS
```

## Files

```text
build.zig                  Zig build, deploy, and verify command surface
config.env                 Required local VM/board settings
src/main.zig               End-to-end proof sequence
src/xrt.zig                Runtime binding to libxrt_coreutil.so.2
src/dma.zig                /dev/mem AXI DMA register driver
src/config.zig             Hardware constants used by the verifier
fpga/build.tcl             Vivado block design for the DMA loopback fixture
fpga/build.bat             Windows Vivado/bootgen entry point
overlay/penzai-loopback.dts XRT/zocl app overlay
```

## What This Proves

- XRT can open the zocl device for the loaded app.
- Zig can allocate, map, address, and sync XRT BOs through the native C API.
- XRT BO physical addresses are reachable from the PL DMA, including high DDR
  addresses that require the AXI DMA MSB registers.
- Zig can control the DMA through MMIO and verify real PL data movement.

This does not prove a production control-plane API. It intentionally uses
`/dev/mem` and root because the experiment is only the verified hardware/software
assumption record.
