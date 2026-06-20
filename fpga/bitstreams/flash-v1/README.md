# penzai flash-attention bitstream (v1)

One `flash_top` fed by five AXI DMAs on a single fabric clock — the device-side of
flash attention on PL. The kernel and wrapper are cosim-verified
(`zig build test-rtl-flash-kernel test-rtl-flash-top`); this packages them into a
KR260 bitstream. v1 is correctness-first (the kernel is sequential, not yet
bandwidth-optimized), so a single clock domain and the timing-safe `f100` variant.

## Layout
- `tcl/build.tcl` — Vivado block design: PS + clk_wiz + 5 DMAs + width converters +
  `flash_top`. Single clock; AXIS widths bridged by `axis_dwidth_converter`
  (Q 128→256, mask 128→16, O 256→128; K/V direct 128).
- `tcl/address_map.tcl` — **generated** (`zig build regmap`, single source
  `fpga/regmap/flash_attn.zig`). The host (`device/pl/flash_attn.zig`) maps the same
  addresses, so they cannot drift.
- `build.sh` / `build.bat` — sync RTL+TCL to the Vivado VM and build.
- `deploy.sh` + `overlay/` — load the bitstream on the board via `xmutil`.

## Build & deploy
```sh
cp config.env.example config.env      # edit VM / BOARD
(cd ../../.. && zig build regmap)      # ensure flash_regs.vh + address_map.tcl are current
./build.sh f100                        # Vivado synth/impl/bitstream (timing-gated)
./deploy.sh f100                       # load onto the KR260
# then (re)deploy + serve the daemon, telling it ONLY flash is on PL:
(cd ../../.. && nix run .#deploy-penzaid)
PENZAI_PL_OPS=flash nix run .#serve-penzaid
# PL init should report "pl: flash kernel ready" (ID 0xF1A54A00); matmul runs on PS.
```

**`PENZAI_PL_OPS=flash` is required.** The KR260 holds one bitstream at a time, so this
bitstream has no matmul IP. The daemon probes PL ops by reading their AXI-Lite registers
over `/dev/mem`; a read of an *unmapped* address is a bus fault (SError → SIGBUS), not a
catchable error, so probing the absent matmul IP would kill the daemon silently (exit
255, banner only). `PENZAI_PL_OPS` gates which backends are probed (`matmul` default,
`flash`, `matmul,flash`/`all`, `none`) — set it to match the resident bitstream. A
combined matmul+flash bitstream is the production fix that removes this constraint.

## Validate on silicon
1. `PENZAI_PL_VERIFY=1` — the runtime runs the PS oracle into scratch and compares
   against the PL flash result; logs any drift beyond the fp tolerance.
2. Token stability — `penzai run --device tcp:kria:NNNNN` vs `--device fake` should
   produce the same tokens.
3. Scoreboard — watch `flash_ms_tok` (PS baseline ≈ 30.9). v1 is sequential, so the
   first silicon number is a correctness checkpoint, not the perf target; the
   head-interleave + GQA-reuse + pipelined-walk pass comes next.

## Notes / unknowns to confirm in Vivado (no sim here)
- The mask path (`axis_dwidth_converter` 128→16) and the O path (256→128) widths are
  by inspection; `validate_bd_design` (pre-synth) flags miswiring fast.
- `flash_top` requires `head_dim` a multiple of 8; the tenant falls back to PS otherwise.
- v1 materializes the GQA-replicated K/V streams host-side, so K/V staging (4 MiB each)
  bounds decode to n_kv ≲ 1024 at the 1.7B shape; larger shapes fall back to PS.
