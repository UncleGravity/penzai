# kr260-xrt-ddr-bandwidth-multiport

## TLDR

This experiment measures how close the KR260 can get to the board DDR
theoretical ceiling from PL:

```text
one XRT BO in DDR
PS S_AXI_HP0_FPD through S_AXI_HP3_FPD
four independent 128-bit AXI DMA lanes
AXI DMA burst length 16
300 MHz fabric clock from clk_wiz
write-only and read-only correctness checks
```

The target offered bandwidth is:

```text
4 ports * 128 bits * 300 MHz = 19.2 GB/s = about 17881 MiB/s
```

The benchmark runs each HP port alone first, then runs all selected ports
simultaneously. Correctness is required for every reported bandwidth number.

## Result

Run on 2026-06-06 with `w128-f300-p4`, four `128-bit @ 300 MHz` HP lanes,
AXI DMA burst length 16, one 768 MiB BO, 192 MiB per lane, 32 MiB chunks.
The routed design was timing-clean with WNS = 0.131 ns:

```text
single hp0 write = 4572 MiB/s, read = 4377 MiB/s
single hp1 write = 4573 MiB/s, read = 4377 MiB/s
single hp2 write = 4569 MiB/s, read = 4377 MiB/s
single hp3 write = 4573 MiB/s, read = 4377 MiB/s

four-port write aggregate = 11727 MiB/s = 12.30 GB/s
four-port read  aggregate = 11535 MiB/s = 12.10 GB/s
```

Against the theoretical board ceiling:

```text
write: 11727 / 17881 MiB/s = 65.6%
read:  11535 / 17881 MiB/s = 64.5%
```

Against the sum of individually measured ports:

```text
write: 11727 / 18287 MiB/s = 64.1%
read:  11535 / 17508 MiB/s = 65.9%
```

Changing the AXI DMA burst length from 256 to 16 improved the four-port
aggregate result from 11162 to 11727 MiB/s for writes and from 10449 to
11535 MiB/s for reads. Single-port reads decreased from about 4505 to
4377 MiB/s, so burst length 16 is better for this full-board pressure test but
not universally better for every isolated-port case.

All reported cases passed correctness. In aggregate runs, the per-lane
`inferred_clk_mhz_x100` field uses the aggregate elapsed time, so lanes that
finish early print a lower apparent clock. Use single-port runs for fabric clock
sanity; use aggregate `mib_s` for the DDR bandwidth result.

## Decode Bandwidth Interpretation

For read-heavy decode, the important number from this fixture is the four-port
read aggregate:

```text
11535 MiB/s = 12.10 GB/s
```

This is the upper bound this DMA fixture demonstrated for streaming weights from
DDR into PL. A model that must read `weight_bytes_per_token` from DDR has a
memory-only decode ceiling of:

```text
tok/s <= 12.10e9 / weight_bytes_per_token
```

At 300 MHz, the measured read aggregate is about 40 bytes per fabric cycle
across all four HP lanes. The ideal offered width is 64 bytes per fabric cycle.
Reaching 80% of the raw 17881 MiB/s ceiling would require about 14305 MiB/s, or
roughly 24% more read bandwidth than this run.

The four-port read run is imbalanced. HP1 and HP2 are the long-running lanes;
HP0 and HP3 finish their stream work about 11 ms earlier over the 66.6 ms
aggregate read case, then sit idle at the per-chunk synchronization barrier.
Using only HP0 and HP3 would likely improve per-lane efficiency, but it would
not improve total weight bandwidth unless the two-port pair can exceed the
current four-port aggregate, which is unlikely with 128-bit 300 MHz lanes.

Read-only lane-pair sweep on the same bitstream confirmed that two-lane
efficiency is good but not enough to beat the four-lane aggregate:

```text
hp0,hp3: 8136, 8118, 8062 MiB/s  avg 8105 MiB/s
hp1,hp2: 7441, 7431, 7383 MiB/s  avg 7418 MiB/s
hp0,hp1: 8125, 8098, 8092 MiB/s  avg 8105 MiB/s
hp2,hp3: 8095, 8073, 8051 MiB/s  avg 8073 MiB/s
```

Offset permutation sweeps within the same 768 MiB BO did not reveal a hidden
address placement win. HP1 and HP2 stayed the long-running lanes:

```text
0,192,384,576 MiB: 11517, 11486, 11499 MiB/s  avg 11501 MiB/s
576,384,192,0 MiB: 11550, 11544, 11561 MiB/s  avg 11552 MiB/s
192,384,576,0 MiB: 11518, 11526, 11524 MiB/s  avg 11523 MiB/s
```

## Test Design

Each lane is independent:

```text
lane0: engine0 <-> dma0 <-> S_AXI_HP0_FPD <-> DDR
lane1: engine1 <-> dma1 <-> S_AXI_HP1_FPD <-> DDR
lane2: engine2 <-> dma2 <-> S_AXI_HP2_FPD <-> DDR
lane3: engine3 <-> dma3 <-> S_AXI_HP3_FPD <-> DDR
```

Shared infrastructure:

```text
PS M_AXI_HPM0_FPD -> AXI-Lite root SmartConnect -> per-lane control SmartConnects
PS pl_clk0 runtime ~100 MHz -> clk_wiz -> 300 MHz fabric clock
one proc_sys_reset
one 768 MiB XRT BO split into per-lane windows
```

Default windows:

```text
lane0 offset =   0 MiB, seed = 1
lane1 offset = 192 MiB, seed = 17
lane2 offset = 384 MiB, seed = 33
lane3 offset = 576 MiB, seed = 49
```

Pattern:

```text
byte[i] = (i * 7 + seed) & 0xff
```

## Run

First-time setup:

```sh
cp config.env.example config.env
$EDITOR config.env
```

Required settings:

```sh
VM=...
VM_DIR=kr260-ddr-bandwidth-multiport
BOARD=ubuntu@kria
APP=penzai-ddr-bandwidth-multiport
BOARD_TMP=/tmp/penzai-ddr-bandwidth-multiport
```

Build, deploy, and run:

```sh
zig build all -Dvariant=w128-f300-p4
```

Separate steps for debugging:

```sh
zig build bitstream -Dvariant=w128-f300-p4
zig build deploy    -Dvariant=w128-f300-p4
zig build run       -Dvariant=w128-f300-p4
```

Small smoke test:

```sh
zig build run -Dvariant=w128-f300-p4 -Dboard-args="verify"
```

RTL lint before Vivado:

```sh
nix develop -c verilator --lint-only -Wall fpga/rtl/*.v
```

Supported variant:

```text
w128-f300-p4
```

Runtime options:

```text
--ports N       active HP ports from hp0 upward, default 4
--active LIST   comma-separated lanes, e.g. hp0,hp3
--offsets LIST  comma-separated BO offsets for active lanes, e.g. 0MiB,384MiB
--bo SIZE       backing BO size, default 768MiB for run
--size SIZE     per-port transfer size, default 192MiB for run
--chunk SIZE    DMA chunk size, default 32MiB
--read-only     run only DDR -> PL checker cases
--write-only    run only PL generator -> DDR cases
--repeat N      repeat selected benchmark cases, default 1
```

`SIZE` accepts `B`, `KiB`, `MiB`, and `GiB`.

Read-only lane-pair sweep for weight-streaming investigations:

```sh
zig build run -Dvariant=w128-f300-p4 -Dboard-args="run --read-only --active hp0,hp3 --repeat 3"
zig build run -Dvariant=w128-f300-p4 -Dboard-args="run --read-only --active hp1,hp2 --repeat 3"
zig build run -Dvariant=w128-f300-p4 -Dboard-args="run --read-only --active hp0,hp1 --repeat 3"
zig build run -Dvariant=w128-f300-p4 -Dboard-args="run --read-only --active hp2,hp3 --repeat 3"
```

Offset sweep examples. Offsets map positionally to `--active`; use a larger BO
when the last offset plus per-lane size exceeds 768 MiB:

```sh
zig build run -Dvariant=w128-f300-p4 -Dboard-args="run --read-only --active hp0,hp1,hp2,hp3 --offsets 0MiB,192MiB,384MiB,576MiB --repeat 3"
zig build run -Dvariant=w128-f300-p4 -Dboard-args="run --read-only --active hp0,hp1,hp2,hp3 --offsets 576MiB,384MiB,192MiB,0MiB --repeat 3"
zig build run -Dvariant=w128-f300-p4 -Dboard-args="run --read-only --active hp0,hp1,hp2,hp3 --offsets 192MiB,384MiB,576MiB,0MiB --repeat 3"
```

## Output

Output is line-oriented `key=value` text:

```text
case=info variant=w128-f300-p4 ports=4 active=hp0,hp1,hp2,hp3 mode=both repeat=1 xrt_device_open=1 lanes_open=1 bo_bytes=805306368 bo_phys=0x... offsets=hp0:0MiB,hp1:192MiB,hp2:384MiB,hp3:576MiB
case=write variant=w128-f300-p4 ports=1 active=hp0 offsets=hp0:0MiB size_bytes_each=201326592 ok=1 elapsed_us=... aggregate_mib_s=... lane0_mib_s=...
case=read  variant=w128-f300-p4 ports=1 active=hp0 offsets=hp0:0MiB size_bytes_each=201326592 ok=1 elapsed_us=... aggregate_mib_s=... lane0_mib_s=...
case=write variant=w128-f300-p4 ports=4 active=hp0,hp1,hp2,hp3 offsets=hp0:0MiB,hp1:192MiB,hp2:384MiB,hp3:576MiB size_bytes_each=201326592 ok=1 elapsed_us=... aggregate_mib_s=...
case=read  variant=w128-f300-p4 ports=4 active=hp0,hp1,hp2,hp3 offsets=hp0:0MiB,hp1:192MiB,hp2:384MiB,hp3:576MiB size_bytes_each=201326592 ok=1 elapsed_us=... aggregate_mib_s=...
case=summary variant=w128-f300-p4 ports=4 active=hp0,hp1,hp2,hp3 mode=both repeat=1 ok=1
```

Each result includes per-lane chunk count, PL cycles, inferred clock, and checked
byte count. Failures print lane name, mismatch index, expected/actual byte, DMA
status, and engine status when available.

## Files

```text
build.zig                         build/deploy/run steps
config.env.example                VM/board settings template
src/main.zig                      four-lane benchmark runtime
src/xrt.zig                       XRT dlopen binding
src/dma.zig                       AXI DMA register driver
src/regs.zig                      generator/checker register driver
src/config.zig                    lane addresses, seeds, defaults
src/sizes.zig                     size parser
fpga/build.tcl                    Vivado four-HP-port block design
fpga/build.bat                    Windows Vivado entry point
fpga/rtl/axis_pattern_gen.v       128-bit AXIS source
fpga/rtl/axis_pattern_check.v     128-bit AXIS checker
fpga/rtl/bandwidth_regs.v         AXI-Lite control/status wrapper
overlay/penzai-ddr-bandwidth-multiport.dts
```
