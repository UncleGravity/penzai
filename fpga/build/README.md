# FPGA build

This directory contains the production KR260 bitstream flow. The closed input
set is `sources.f`; it contains 55 Verilog sources and four headers. Nothing
outside that manifest may enter the production project.

The qualified target is `f225`. Build artifacts are written under
`.zig-cache/fpga-build/`, and the promoted image is
`.zig-cache/fpga-build/penzai-f225.bit.bin`.

```sh
zig build regmap
cd fpga/build
cp config.env.example config.env
./build.sh f225
./deploy.sh f225
```

`build.sh --incremental` may reuse the last timing-clean checkpoint for the
same frequency. Experimental frequencies and incremental runs remain in their
run bundles, but only a clean `f225` build can replace the promoted image and
`latest` link. `deploy.sh` accepts only a completed, clean, hash-matched `f225`
run bundle and installs the app as `/lib/firmware/xilinx/penzai`.

Before Vivado starts, the driver resets the remote RTL directory and verifies
the name and SHA-256 digest of every copied source against the local closed
input set. Vivado independently checks that the copied directory exactly
matches `sources.f`. Synthesis and routing must each contain exactly 695
DSP48E2 and eight URAM288 primitives.

Routed methodology errors and new critical warnings fail the build. The two
clock-wizard findings observed on the qualified design, `TIMING-2` and
`TIMING-4`, are allowed at most once each and are recorded explicitly. The
known `TIMING-28` warnings remain recorded rather than treated as signoff
failures.

The build preserves the deployed MMIO identity and layout contract:

- ID `0xB05A4000`
- interface version `0x00010007`
- layout hash `0xC255C7A52FC14A79`
- control base `0xA0000000`

Run `zig build verify-rtl` before a production build. The local structural gate
requires exactly 695 DSP48E2 cells and eight URAM288 cells.
