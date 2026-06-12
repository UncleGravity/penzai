# q1a8-w256-mc — KR260 Q1A8 matmul bitstream (v6, multi-column)

The deployable gateware for the PL Q1A8 matmul: `q1a8_kernel_mc` with a 256-bit weight stream, fed by two AXI DMAs (128→256 weight
upsizer). One weight stream is MAC'd against up to 8 columns/run, so decode (cols=1) and
the same proven block design as the bring-up's v4, plus the kernel_top
performance **counter bank** (`W/A/R_STALL`, `W/A/R_BEATS`). Those are AXI-Lite
registers, so the block design and address map are unchanged.

Building this and loading it is what makes `util%` real in `penzai --prof`:
`penzaid`'s PL init starts reporting `version 5`, and `hasCounters()` turns on,
so the per-format matmul detail fills in stall/beat-derived utilization instead
of the placeholder `100%` the v4 bitstream gives.

## Prerequisites

- A Windows host with Vivado (the `VM` in `config.env`), reachable over ssh.
- The KR260 board (`BOARD`) reachable over ssh with passwordless `sudo`.
- The generated register header in place:
  ```
  (cd ../../.. && zig build regmap)   # writes ../../rtl/q1a8/q1a8_regs.vh
  ```

## Build

```sh
cp config.env.example config.env   # then edit VM / BOARD paths
./build.sh                          # variant from config.env (w256-f125)
```

`build.sh` syncs the v5 narrow RTL set + `q1a8_regs.vh` + the TCL/BAT to the VM,
runs `vivado -mode batch -source build.tcl`, refuses to emit a bitstream if
routing isn't timing-clean, and fetches `out/penzai-q1a8-mc-w256-f125.bit(.bin)`.

The unpipelined fp32 reducer closes around ~137 MHz, so 100 MHz is the safe
target. Higher clocks need reducer pipelining (a later build).

## Deploy

```sh
./deploy.sh                         # loads the .bit.bin via xmutil
```

Then restart `penzaid` as root and confirm the PL init line reads
`version 5, counters true`:

```sh
ssh ubuntu@kria 'cd /tmp/penzai && sudo ./penzaid serve \
  --device tcp:0.0.0.0:29092 --mem xrt --heap-mib 768'
```

A `penzai run ... --prof` then shows real `util%` in the `matmul detail` block.

## Notes

- **Same app name as v4.** This replaces `penzai-q1a8-mc` in the board's
  firmware slot; `penzaid` distinguishes v4/v5 at runtime via the VERSION
  register, so no host change is needed.
- **Address map is a contract.** `dma_w=0xA000_0000`, `dma_a=0xA001_0000`,
  `kernel=0xA002_0000` must match `device/pl/matmul.zig`. Don't change one
  without the other.
- **`.bit`/`.bit.bin` are build artifacts** (gitignored here; track with git-lfs
  if you want them in the repo per the plan).
- The **wide** kernel (4× throughput) is a separate, larger build: it needs a
  256-bit weight-stream `kernel_top` and a multi-HP DMA feed the bring-up never
  built on silicon. This directory is the narrow-first step.
