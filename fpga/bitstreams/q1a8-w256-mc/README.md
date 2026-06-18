# q1a8-w256-mc — KR260 matmul bitstream (v8 four-port slice)

The deployable v8 vertical slice for the PL matmul: `matmul_kernel` with
ROWS=16, a 512-bit internal weight stream, and four 128-bit HP weight lanes at
`wclk` feeding synchronized 128-bit AXIS ports. One weight stream is MAC'd
against up to 8 columns/run, so decode (cols=1) and prefill share the same
kernel.

Building this and loading it is what makes `util%` real in `penzai --prof`:
`penzaid`'s PL init should report `version 8`, and `hasCounters()` turns on, so
the per-format matmul detail fills in stall/beat-derived utilization.

## Prerequisites

- A Windows host with Vivado (the `VM` in `config.env`), reachable over ssh.
- The KR260 board (`BOARD`) reachable over ssh with passwordless `sudo`.
- The generated register header in place:
  ```
  (cd ../../.. && zig build regmap)   # writes ../../rtl/matmul/matmul_regs.vh
  ```

## Build

```sh
cp config.env.example config.env   # then edit VM / BOARD paths
./build.sh                          # variant from config.env (w512-p4-f125-wc250)
```

`build.sh` syncs the v8 RTL set + `matmul_regs.vh` + the TCL/BAT to the VM,
runs `vivado -mode batch -source build.tcl`, refuses to emit a bitstream if
routing isn't timing-clean, and fetches
`out/penzai-q1a8-mc-w512-p4-f125-wc250.bit(.bin)`.

The unpipelined fp32 reducer closes around ~137 MHz, so 100 MHz is the safe
target. Higher clocks need reducer pipelining (a later build).

## Deploy

```sh
./deploy.sh                         # loads the .bit.bin via xmutil
```

Then restart `penzaid` as root and confirm the PL init line reads
`version 8, counters true`:

```sh
ssh ubuntu@kria 'cd /tmp/penzai && sudo ./penzaid serve \
  --device tcp:0.0.0.0:29092 --mem xrt --heap-mib 768'
```

A `penzai run ... --prof` then shows real `util%` in the `matmul detail` block.

## Notes

- **Same app name.** This replaces `penzai-q1a8-mc` in the board's firmware
  slot; `penzaid` distinguishes compatible gateware via VERSION and ROWS.
- **Address map is generated, not hand-matched.** `dma_w0..dma_w3=0xA000_0000..0xA003_0000`,
  `dma_a=0xA004_0000`, `kernel=0xA005_0000` and the kernel `COLS_MAX` all live in
  the one manifest `fpga/regmap/matmul.zig` (`addr` / `caps`). `zig build regmap`
  emits `address_map.tcl` (sourced by `build.tcl`) and `MATMUL_COLS_MAX` (in
  `matmul_regs.vh`); `device/pl` reads the same constants. Change them in the
  manifest and rerun `zig build regmap` — never edit the address map by hand.
- **`.bit`/`.bit.bin` are build artifacts** (gitignored here; track with git-lfs
  if you want them in the repo per the plan).
