# kr260-xrt-ddr-bandwidth-singleport

## TLDR

This experiment measures one KR260 PS-to-PL DDR port:

```text
XRT BO in DDR
PS S_AXI_HP0_FPD
128-bit AXI DMA path
PL test clock at 100/200/250/300 MHz
write-only and read-only correctness checks
```

Measured result:

```text
100 MHz: write 1525 MiB/s, read 1525 MiB/s
300 MHz: write 4574 MiB/s, read 4507 MiB/s
```

One `128-bit @ 300 MHz` port is therefore very close to its ideal per-port
ceiling of `4.8 GB/s`, or about `4578 MiB/s`. This is a single-port test, not a
full-board DDR ceiling test.

## Question

How much bandwidth does one KR260 HP DDR port deliver when the whole path is
configured as `128-bit` and clocked at useful PL frequencies?

The experiment uses:

```text
one 768 MiB XRT BO
one AXI DMA with MM2S and S2MM
one 128-bit PL pattern generator
one 128-bit PL pattern checker
one HP0 DDR port
```

Correctness is part of the measurement:

- Write test: PL writes a deterministic pattern into DDR, then CPU validates the
  BO.
- Read test: CPU fills DDR with the same pattern, then PL validates the read
  stream.

## Test Design

Write path:

```text
pattern generator -> AXI DMA S2MM -> HP0 -> DDR/XRT BO
```

Read path:

```text
DDR/XRT BO -> HP0 -> AXI DMA MM2S -> pattern checker
```

Pattern:

```text
byte[i] = (i * 7 + seed) & 0xff
```

The default benchmark transfers `384 MiB` through a `768 MiB` BO. Transfers are
chunked into `32 MiB` DMA operations because AXI DMA simple mode uses a `26-bit`
length register. The PL receives the base byte index for each chunk, so the
pattern remains position-dependent across the full transfer.

Variant clocks are generated in PL:

```text
PS pl_clk0 runtime clock -> clk_wiz -> 100/200/250/300 MHz fabric clock
```

The runtime prints `inferred_clk_mhz_x100` from `pl_cycles / elapsed`, so each
result shows the measured PL clock.

## Run

First-time setup:

```sh
cp config.env.example config.env
$EDITOR config.env
```

Required settings:

```sh
VM=...
VM_DIR=kr260-ddr-bandwidth-singleport
BOARD=ubuntu@kria
APP=penzai-ddr-bandwidth-singleport
BOARD_TMP=/tmp/penzai-ddr-bandwidth-singleport
```

Build, deploy, and run:

```sh
zig build all -Dvariant=w128-f100
zig build all -Dvariant=w128-f300
```

Separate steps for debugging:

```sh
zig build bitstream -Dvariant=w128-f300
zig build deploy    -Dvariant=w128-f300
zig build run       -Dvariant=w128-f300
```

Small smoke test:

```sh
zig build run -Dvariant=w128-f300 -Dboard-args="verify"
```

RTL lint before Vivado:

```sh
nix develop -c verilator --lint-only -Wall fpga/rtl/*.v
```

Supported variants:

```text
w128-f100
w128-f200
w128-f250
w128-f300
```

The board binary accepts optional size controls:

```text
--bo SIZE       backing BO size, default 768MiB for run
--size SIZE     transfer size, default 384MiB for run
--chunk SIZE    DMA chunk size, default 32MiB
--seed N        byte pattern seed, default 1
```

`SIZE` accepts `B`, `KiB`, `MiB`, and `GiB`.

## Output

Output is line-oriented `key=value` text:

```text
case=info variant=w128-f300 xrt_device_open=1 dma_open=1 regs_open=1 bo_bytes=805306368 bo_phys=0x...
case=write variant=w128-f300 size_bytes=402653184 size_mib=384 chunks=12 ok=1 elapsed_us=83939 mib_s=4574 pl_cycles=25165824 inferred_clk_mhz_x100=29980
case=read variant=w128-f300 size_bytes=402653184 size_mib=384 chunks=12 ok=1 elapsed_us=85195 mib_s=4507 pl_cycles=25548310 inferred_clk_mhz_x100=29987
case=summary variant=w128-f300 ok=1
```

Failures print the failing test, byte index, expected value, actual value, and
DMA/checker status when available.

## Results

```text
variant    test   elapsed_us  mib_s  inferred_mhz  pl_cycles
w128-f100  write  251765      1525    99.9         25165824
w128-f100  read   251729      1525    99.9         25167296
w128-f300  write   83939      4574   299.8         25165824
w128-f300  read    85195      4507   299.9         25548310
```

Ideal one-port ceilings:

```text
128-bit @ 100 MHz -> 1.6 GB/s = about 1526 MiB/s
128-bit @ 200 MHz -> 3.2 GB/s = about 3052 MiB/s
128-bit @ 300 MHz -> 4.8 GB/s = about 4578 MiB/s
```

Interpretation:

- `w128-f100` matches the ideal one-port ceiling.
- `w128-f300` confirms a real `300 MHz` fabric clock.
- Write at `300 MHz` is essentially ideal.
- Read at `300 MHz` is slightly lower, but still near the one-port ceiling.

## Files

```text
build.zig                         build/deploy/run steps
config.env.example                VM/board settings template
src/main.zig                      benchmark runtime
src/xrt.zig                       XRT dlopen binding
src/dma.zig                       AXI DMA register driver
src/regs.zig                      generator/checker register driver
src/config.zig                    addresses and default sizes
src/sizes.zig                     size parser
fpga/build.tcl                    Vivado block design
fpga/build.bat                    Windows Vivado entry point
fpga/rtl/axis_pattern_gen.v       128-bit AXIS source
fpga/rtl/axis_pattern_check.v     128-bit AXIS checker
fpga/rtl/bandwidth_regs.v         AXI-Lite control/status wrapper
overlay/penzai-ddr-bandwidth-singleport.dts
```
