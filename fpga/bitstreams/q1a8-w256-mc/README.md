# q1a8-w256-mc — KR260 Q1A8 matmul bitstream (v7 vertical slice)

The deployable v7 vertical slice for the PL Q1A8 matmul: `q1a8_kernel_mc` with
ROWS=16, a 512-bit internal weight stream, and two 128-bit HP weight lanes at
`wclk` feeding synchronized 256-bit AXIS ports. One weight stream is MAC'd
against up to 8 columns/run, so decode (cols=1) and prefill share the same
kernel.

Building this and loading it is what makes `util%` real in `penzai --prof`:
`penzaid`'s PL init should report `version 7`, and `hasCounters()` turns on, so
the per-format matmul detail fills in stall/beat-derived utilization.

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
./build.sh                          # variant from config.env (w512-p2-f125-wc250)
```

`build.sh` syncs the v7 RTL set + `q1a8_regs.vh` + the TCL/BAT to the VM,
runs `vivado -mode batch -source build.tcl`, refuses to emit a bitstream if
routing isn't timing-clean, and fetches
`out/penzai-q1a8-mc-w512-p2-f125-wc250.bit(.bin)`.

The unpipelined fp32 reducer closes around ~137 MHz, so 100 MHz is the safe
target. Higher clocks need reducer pipelining (a later build).

## Deploy

```sh
./deploy.sh                         # loads the .bit.bin via xmutil
```

Then restart `penzaid` as root and confirm the PL init line reads
`version 7, counters true`:

```sh
ssh ubuntu@kria 'cd /tmp/penzai && sudo ./penzaid serve \
  --device tcp:0.0.0.0:29092 --mem xrt --heap-mib 768'
```

A `penzai run ... --prof` then shows real `util%` in the `matmul detail` block.

## Notes

- **Same app name.** This replaces `penzai-q1a8-mc` in the board's firmware
  slot; `penzaid` distinguishes compatible gateware via VERSION and ROWS.
- **Address map is a contract.** `dma_w0=0xA000_0000`, `dma_w1=0xA001_0000`,
  `dma_a=0xA002_0000`, `kernel=0xA003_0000` must match
  `device/pl/matmul.zig`. Don't change one without the other.
- **`.bit`/`.bit.bin` are build artifacts** (gitignored here; track with git-lfs
  if you want them in the repo per the plan).
