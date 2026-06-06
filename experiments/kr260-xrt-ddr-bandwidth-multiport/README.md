# kr260-xrt-ddr-bandwidth-multiport

## TLDR

This experiment measures how close the KR260 can get to the board DDR
theoretical ceiling from PL:

```text
one XRT BO in DDR
PS S_AXI_HP0_FPD through S_AXI_HP3_FPD
four independent 128-bit AXI DMA lanes
300 MHz fabric clock from clk_wiz
write-only and read-only correctness checks
```

The target offered bandwidth is:

```text
4 ports * 128 bits * 300 MHz = 19.2 GB/s = about 17881 MiB/s
```

The benchmark runs each HP port alone first, then runs all selected ports
simultaneously. Correctness is required for every reported bandwidth number.

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
--bo SIZE       backing BO size, default 768MiB for run
--size SIZE     per-port transfer size, default 192MiB for run
--chunk SIZE    DMA chunk size, default 32MiB
```

`SIZE` accepts `B`, `KiB`, `MiB`, and `GiB`.

## Output

Output is line-oriented `key=value` text:

```text
case=info variant=w128-f300-p4 ports=4 xrt_device_open=1 lanes_open=1 bo_bytes=805306368 bo_phys=0x...
case=write variant=w128-f300-p4 ports=1 active=hp0 size_bytes_each=201326592 ok=1 elapsed_us=... aggregate_mib_s=... lane0_mib_s=...
case=read  variant=w128-f300-p4 ports=1 active=hp0 size_bytes_each=201326592 ok=1 elapsed_us=... aggregate_mib_s=... lane0_mib_s=...
case=write variant=w128-f300-p4 ports=4 active=hp0,hp1,hp2,hp3 size_bytes_each=201326592 ok=1 elapsed_us=... aggregate_mib_s=...
case=read  variant=w128-f300-p4 ports=4 active=hp0,hp1,hp2,hp3 size_bytes_each=201326592 ok=1 elapsed_us=... aggregate_mib_s=...
case=summary variant=w128-f300-p4 ports=4 ok=1
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
