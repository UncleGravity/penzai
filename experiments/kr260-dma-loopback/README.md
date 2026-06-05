# kr260-dma-loopback

Build and load a KR260 AXI-DMA loopback app, then prove the full path:

```text
XRT BO memory -> AXI DMA MM2S -> AXIS FIFO -> AXI DMA S2MM -> XRT BO memory
```

The PL design uses Xilinx IP only. XRT owns buffer allocation; the test program
uses `/dev/mem` MMIO writes to control the AXI DMA registers.

## Quick Start

Edit `config.env` if your VM or board address differs, then run:

```sh
./run.sh
```

That runs:

```sh
./build.sh     # Vivado/bootgen on the Windows VM, fetches fpga/out/loopback.bit.bin
./deploy.sh    # installs and loads the penzai-loopback XRT_FLAT app on the KR260
./test.sh      # compiles and runs the BO probe + DMA loopback on the KR260
```

Expected final test line:

```text
PASS: DMA loopback src==dst (4096 bytes through MM2S->FIFO->S2MM)
```

## Prerequisites

Host:

- SSH/SCP access to the Windows Vivado VM (`VM` in `config.env`).
- SSH/SCP access to the board configured as `BOARD`.

Windows VM:

- Vivado/Vitis 2025.2 installed at the path configured in `fpga/build.bat`.

KR260:

- Xilinx Ubuntu with `dtc`, `gcc`, `xmutil`, and XRT user-space libraries.
- `zocl` loaded and matching XRT (`xrt-smi examine` should run).
- User in `render` and `video` groups, then re-login.

## Useful Commands

```sh
./build.sh             # build only
./deploy.sh            # deploy/load current fpga/out/loopback.bit.bin
./test.sh              # run XRT BO probe and DMA loopback
./status.sh            # inspect xmutil, XRT, fan PWM, sensors, firmware package
./restore-starter.sh   # reload k26-starter-kits
```

## Files

```text
build.sh                    Host-side VM build driver
deploy.sh                   Host-side KR260 app installer/loader
test.sh                     Host-side test runner
run.sh                      build + deploy + test
status.sh                   Board state snapshot
restore-starter.sh          Reload stock starter-kit app
config.env                  Required local machine config

fpga/build.tcl              Vivado BD: PS + AXI DMA + AXIS FIFO + fan PWM route
fpga/build.bat              Windows Vivado/bootgen entry point
fpga/out/                   Generated bitstream outputs

overlay/penzai-loopback.dts XRT/zocl app overlay
board/test_bo.c             XRT BO alloc/map/addr/sync probe
board/dma_loopback.c        Full XRT BO + MMIO DMA loopback test
```

## Design Notes

- DMA control base is `0xA0000000`.
- XRT BOs land in high DDR, so the AXI DMA uses 40-bit addressing and the
  SA/DA MSB registers.
- The KR260 fan gate is a PL pin. The bitstream routes Linux's TTC0 PWM
  waveout channel to `fan_en_b` so replacing the PL image does not force the
  fan to full speed.
- The app directory must contain exactly one `*.bit.bin`; `deploy.sh` removes
  stale bitstreams before installing the new one.
