# kr260-xrt-dma-loopback

KR260 experiment for the basic PL DMA path:

```text
XRT BO in DDR -> AXI DMA MM2S -> AXIS FIFO -> AXI DMA S2MM -> same XRT BO
```

## TLDR

On the KR260, a `768 MiB` XRT BO can be used as the DMA backing store, split into
source and destination halves. The loopback path is correct through a `384 MiB`
payload, including chunked transfers. Large transfers settle at about
`378 MiB/s` payload bandwidth, or about `756 MiB/s` of DDR traffic because the
loopback reads and writes each byte.

## What This Tests

Can the KR260 move bytes correctly between XRT BO-backed DDR and FPGA fabric,
and how fast is that path?

It tests:

- XRT device open
- XRT BO allocation and mapping
- physical BO addresses passed to PL DMA
- AXI DMA MM2S and S2MM register control through `/dev/mem`
- correctness by comparing `src == dst`
- sustained DMA bandwidth over realistic transfer sizes

It does not test Q1A8 math, a production daemon protocol, interrupts, cache
policy design, or a final accelerator memory layout.

## Current Results

Measured with XRT userspace `2.18.0` and the `penzai-dma-loopback` app loaded
through `xmutil`.

```text
verify:
  4 KiB transfer in 8 MiB BO: PASS

sweep --bo 768MiB --max 384MiB --chunk 32MiB:
  4 KiB:   PASS, ~48 us
  64 KiB:  PASS, ~323 MiB/s payload
  1 MiB:   PASS, ~374 MiB/s payload
  8 MiB+:  PASS, ~377-378 MiB/s payload
  384 MiB: PASS, 12 chunks, ~378 MiB/s payload
```

The large BO mapped at physical base `0x43600000` in the measured run.

## Run

`config.env` is intentionally local to this experiment and must define the
Vivado VM and KR260 SSH targets.

Build the bitstream, deploy the app, and run the verifier:

```sh
zig build all
```

Run only the board verifier:

```sh
zig build verify
```

Useful verifier commands:

```sh
zig build verify -Dboard-args="verify"
zig build verify -Dboard-args="sweep --bo 768MiB --max 384MiB --chunk 32MiB"
zig build verify -Dboard-args="stress --size 256MiB --iters 20 --chunk 32MiB"
```

Deploy only the app:

```sh
zig build deploy
```

## Commands

- `verify`: small smoke test. Defaults to a `4 KiB` transfer in an `8 MiB` BO.
- `sweep`: runs correctness and bandwidth over fixed sizes from `4 KiB` up to
  `--max`.
- `stress`: repeats one transfer size for stability checks.

Common options:

- `--bo SIZE`: backing BO size. Default for sweep/stress is `768MiB`.
- `--size SIZE`: transfer size for `verify` or `stress`.
- `--max SIZE`: largest transfer for `sweep`. Default is `384MiB`.
- `--chunk SIZE`: max DMA transfer chunk. Default is `32MiB`.
- `--iters N`: stress iteration count. Default is `10`.

`SIZE` accepts `B`, `KiB`, `MiB`, and `GiB`.

## Output

Example:

```text
xrt: device open OK
dma: open status MM2S=0x00001002 S2MM=0x00001002
bo: alloc/map PASS phys=0x43600000 size_bytes=805306368 size_mib=768
case=sweep size_bytes=33554432 size_kib=32768 size_mib=32 chunk_bytes=33554432 chunk_mib=32 chunks=1 ok=1 dma_us=84614 payload_mib_s=378 ddr_mib_s=756
ALL PASS
```

`payload_mib_s` is the logical copy rate from source to destination.
`ddr_mib_s` counts both DDR reads and DDR writes.

## Files

```text
build.zig                         build, deploy, verify steps
config.env                        local VM/board settings
src/main.zig                      verify/sweep/stress CLI
src/xrt.zig                       runtime XRT binding
src/dma.zig                       AXI DMA register driver
src/config.zig                    hardware constants
src/sizes.zig                     CLI size parser
fpga/build.tcl                    Vivado block design
fpga/build.bat                    Windows Vivado entry point
overlay/penzai-dma-loopback.dts   XRT/zocl app overlay
```

## Rewrite Takeaway

For the first `penzai` runtime, it is reasonable to assume one large XRT BO,
hand physical offsets to PL DMA, and measure accelerator bandwidth with real DMA
transfers rather than `xrtBOSync`.
