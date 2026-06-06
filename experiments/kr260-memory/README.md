# kr260-memory

KR260 XRT buffer-object memory probe for the `penzai` rewrite.

This experiment measures the memory substrate the future board daemon will use:
XRT BO allocation capacity, chunked allocation behavior, sparse fragmentation,
and `xrtBOSync` cost. It does not require a custom bitstream or `/dev/mem`.

## Current findings

Measured on KR260 with XRT userspace `2.18.0` and `CmaTotal=838860800`
(`800 MiB`).

- XRT opens cleanly from the normal `ubuntu` user when group permissions are
  configured.
- Typical `CmaFree` before probing was about `812-829 MiB`.
- Single BO allocation succeeded through `788 MiB` and failed at `792 MiB`.
  Treat `768 MiB` as the reliable large-BO target for this board state.
- `single-sweep` repeatedly reused the same physical base address
  (`0x43600000`), confirming that allocate/free/reallocate of one large BO is
  not itself the fragmentation problem.
- `chunked --chunk 32MiB` allocated `24` chunks (`768 MiB` total). Physical
  addresses were contiguous and sequential in `32 MiB` steps.
- The fragmentation probe allocated mixed `8/16/32/64 MiB` BOs, freed
  alternating handles, and then retried large allocations. Even after freeing
  `608 MiB` total, only `64 MiB` could be allocated; `128/256/384/512 MiB`
  failed. Aggregate free CMA is therefore not enough: one BO still needs one
  contiguous physical hole.
- `fragment_after_close` matched `fragment_after`, so closing the XRT device
  handle did not release additional CMA beyond freeing BO handles.
- The `xrtBOSync` timing results are too fast to represent DDR or PL bandwidth.
  In this setup they should be treated as cheap/no-op cache-maintenance or XRT
  bookkeeping numbers. Real transfer bandwidth must be measured with the DMA
  loopback experiment.

## Design conclusion

For v1, do not implement multi-extent tensors just in case.

The KR260 runtime should require one large contiguous XRT BO at daemon startup,
then suballocate inside it:

```text
XRT/CMA:
  one large BO, preferably 768 MiB

penzaid:
  one mapped device heap inside that BO
  handles are offsets/ranges in that heap
  no per-tensor XRT BO allocation during normal runtime
```

Recommended v1 policy:

- Default target: one `768 MiB` BO, configurable down to a known-good lower
  bound such as `500 MiB`.
- If the large BO cannot be allocated, fail with clear CMA diagnostics rather
  than silently entering a complex degraded mode.
- Keep the public memory API compatible with future multi-extent support, but do
  not implement multi-extent tensors in v1.
- Use simple internal allocation inside the BO: upload-time weight bump
  allocation, fixed or arena-backed KV/cache regions, and graph scratch arenas
  reset between decode steps.
- A systemd service may start `penzaid` at boot to reserve memory early, but it
  is not required for correctness. Restarting the daemon should usually recover
  the same large region as long as `penzaid` owns all XRT BOs and frees them on
  exit.
- After daemon restart, host-side remote handles are invalid; the host must
  reconnect and re-upload board-resident state.

## Prerequisites

KR260:

- Xilinx Ubuntu with XRT userspace.
- `zocl` loaded and an XRT app/shell available.
- User in the `render` and `video` groups, or run the binary manually with the
  privileges your image requires for XRT.

Host:

- Zig 0.16.0.
- SSH access to the KR260.

## Run

Copy the local config template once:

```sh
cp config.env.example config.env
```

Then run all probes on the board:

```sh
zig build run-board
```

Pass command arguments with `-Dboard-args`:

```sh
zig build run-board -Dboard-args="info"
zig build run-board -Dboard-args="single-sweep --max 800MiB --step 32MiB"
zig build run-board -Dboard-args="chunked --chunk 32MiB --max-chunks 40"
zig build run-board -Dboard-args="sync-sweep --max 768MiB --iters 5"
zig build run-board -Dboard-args="fragment --chunk 32MiB --large 256MiB"
```

The binary can also be copied and run manually:

```sh
zig build
scp zig-out/bin/kr260-memory ubuntu@kria:/tmp/
ssh ubuntu@kria /tmp/kr260-memory all
```

## Commands

- `info`: prints `/proc/meminfo` and verifies XRT device open.
- `single-sweep`: allocates one BO at a time from `--min` to `--max`.
- `chunked`: allocates repeated `--chunk` BOs until allocation fails or
  `--max-chunks` is reached.
- `sync-sweep`: times repeated `xrtBOSync` to/from-device over fixed sizes.
- `fragment`: allocates mixed-size BOs, frees alternating handles, then retries
  `64/128/256/512 MiB` allocations plus `--large` when it differs.
- `all`: runs every command.

Output is line-oriented `key=value` text:

```text
case=meminfo label=info ok=1 mem_total_bytes=... cma_total_bytes=...
case=single size_bytes=33554432 ok=1 phys=0x... alloc_us=...
case=chunked event=summary chunks=24 total_mib=768
case=sync direction=to_device size_mib=128 iters=5 mib_s=...
```

The important values for the rewrite are max single BO size, stable chunked
capacity, physical address distribution, and sync bandwidth/cost.
