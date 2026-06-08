Repo guidance for agents and contributors working on `penzai`.

This project is a Zig rewrite of a host/device/FPGA accelerator stack. The
engineering goal is explicit systems code: fixed contracts, visible ownership,
narrow errors, target-specific modules, and testable reference paths.

## Zig Version

- Pin and document one exact Zig version. Keep `build.zig`, CI, and local
  tooling on the same version.
- Do not assume source or build API compatibility across Zig releases.
- Avoid `master` behavior unless the repo explicitly tracks `master`.
- If targeting Zig 0.16.x, design new code around `std.Io` and
  `std.process.Init` from the start.

## Target Support

- Treat supported targets as a project contract, not an assumption.
- For PYNQ-Z1, verify the exact ARM Linux triple, libc, CPU features, and
  minimum kernel version in `build.zig`.
- Zig 0.16.0 lists `arm-linux`, `armeb-linux`, `thumb-linux`, and
  `thumbeb-linux` as Tier 2 targets. Keep board code within APIs that work
  there unless an isolated path is explicitly tested.
- Keep endian, alignment, pointer-width, and page-size assumptions explicit.
  These are protocol and DMA corruption risks.

## I/O And Main

- In Zig 0.16.x, filesystem, networking, process, time, entropy, and many sync
  APIs route through `std.Io`.
- Application entry points should prefer:

```zig
pub fn main(init: std.process.Init) !void
```

- Pull `gpa`, arenas, `io`, args, and environment from `std.process.Init`
  instead of global process APIs.
- Code that may block, use time, use randomness, touch files, spawn processes,
  or use networking should accept `io: std.Io` or store it on a context.
- Construct I/O at the application boundary. Do not create ad hoc `Io.Threaded`
  instances deep in library code.
- Use `std.testing.io` in I/O tests and `std.Io` sync primitives where
  synchronized code must cooperate with the selected I/O implementation.
- Propagate `error.Canceled` unless the current function initiated cancellation
  and can account for completed side effects.
- CPU-bound host loops may add cancellation points; device hot paths should not
  grow I/O dependencies.

## Directory Boundaries

- `shared/` must stay OS-agnostic and frontend-agnostic: no llama, ggml,
  sockets, filesystems, threads, or platform APIs. Use fixed-width types in
  protocols and persistent data.
- `host/` owns CLI, llama.cpp, ggml backend registration, lowering, transport
  clients, and report formatting.
- `device/` owns runtime dispatch, scheduling, memory slabs, ARM kernels, and PL
  driver integration. It must not import or depend on llama/ggml.
- `fpga/` owns RTL, simulations, bitstream scripts, and generated regmap
  consumers.
- If device code needs a ggml detail, the host lowerer or wire contract is
  missing information.

## Allocation And Ownership

- Allocation must be explicit. Pass an allocator to code that allocates.
- Prefer preallocated buffers, fixed buffers, slabs, and reusable arenas on hot
  paths. Avoid general-purpose allocation in the device steady state.
- Use arenas for short-lived host setup and per-command temporary data.
- Use `std.testing.allocator` in allocation tests.
- Prefer unmanaged standard-library containers: store the container separately
  and pass the allocator to methods that allocate.
- Do not introduce `std.heap.ThreadSafeAllocator`; it was removed in Zig 0.16.0.
  Choose an allocator with the required concurrency properties.
- Owned resources should expose `init`, `deinit`, and when useful `reset`. Use
  `defer` and `errdefer` during initialization.
- Do not store slices with ambiguous ownership. If a field owns memory,
  `deinit` frees it; if it borrows memory, the API must make the lifetime clear.

## Error Handling

- Keep public error sets narrow, especially in `shared/` and `device/`.
- Avoid exposing `anyerror` from protocol, wire, kernel, allocator, and runtime
  APIs.
- Define named public error sets such as `DecodeError = error{ BadMagic,
  UnsupportedVersion, Truncated, InvalidTag }`.
- Use `try` for propagation and `switch` when policy depends on the error.
- Do not collapse distinct protocol or runtime failures into one generic error.

## Wire Format And Dispatch

- Treat all incoming wire bytes as hostile.
- Never leak `usize`, pointers, native-endian values, or platform-specific layout
  into the wire contract.
- Use fixed-width integer types and explicit backing types for wire enums, for
  example `pub const OpTag = enum(u16) { matmul_q1a8 = 1, rmsnorm = 2 };`.
- Prefer explicit binary `encode`/`decode` helpers over pointer-casting byte
  buffers. Specify endianness at every integer read/write.
- Do not rely on `packed struct` layout as a portable protocol.
- Validate lengths, tags, versions, feature bits, alignment requirements, and
  handle ranges before dispatch.
- Keep `shared/protocol/wire.zig` as the host/device contract source of truth.
- If a `packed struct`, `packed union`, or `enum` crosses an extern, MMIO, or
  generated-code boundary, give it an explicit backing integer type.
- Do not put pointers in packed types. Store integer addresses only at low-level
  boundaries and convert with care.
- The host produces explicit command buffers. Device scheduling follows declared
  dependencies; it does not infer graph shape.
- Prefer closed tagged unions and exhaustive `switch` dispatch for device ops.
  Do not introduce runtime kernel registries for the closed device op set.
- Add a protocol version or feature bit when changing compatibility.

## Comptime, Generics, And Seams

- Use `comptime` for real compile-time structure: schema tables, op metadata,
  quant specialization, regmap generation, build configuration, and
  target-specific composition.
- Keep comptime code auditable. Prefer small tables and validation over clever
  pseudo-reflection.
- Use `anytype` sparingly. Prefer concrete types for ordinary functions.
- Good generic or runtime seams are `Link(Transport)`, quant packers, fake vs
  real memory, PL matmul vs scalar reference, and thin C-library boundaries.
- Prefer concrete structs, tagged unions, and exhaustive switches elsewhere.
- Avoid APIs whose compiler errors are harder to understand than handwritten
  code.
- Do not use deprecated `@Type` construction in new code. Use Zig 0.16 builtins
  such as `@Int`, `@Struct`, `@Union`, `@Enum`, `@Pointer`, `@Fn`, and `@Tuple`.
- Declare error sets explicitly with `error{ ... }`; do not try to reify error
  sets at comptime.

## Integer And Numeric Code

- Develop and test kernels in safe modes first.
- Use explicit wrapping, saturating, or checked arithmetic when overflow behavior
  matters. Do not rely on accidental overflow.
- Integer matmul paths must be bit-exact against the scalar reference.
- Floating glue ops should use documented tolerances.
- Keep reference kernels simple, slow if necessary, and obviously correct.
- Only use unchecked operations in tiny, benchmarked sections with tests proving
  equivalence.
- Runtime indexing into vectors is forbidden in Zig 0.16. Coerce vectors to
  arrays before dynamic iteration, or keep indexes comptime-known.
- Do not pointer-cast between array memory and vector memory. Use value
  coercions for intended vector/array conversion.
- Keep integer-to-float conversions explicit when precision may be lost.

## C Interop

- Isolate llama.cpp and ggml C interop in `host/llama.zig` and nearby backend
  wrapper code.
- Prefer build-system C translation over `@cImport` for new code. Put includes
  in a small C header, translate it in `build.zig`, and import the generated
  module from Zig.
- Convert C pointers, nullability, integer widths, and ownership rules at the
  boundary. Do not let raw C pointers spread through host logic.
- Wrap C-owned resources in Zig structs with `init`/`deinit`.
- Cast integer sizes close to the C call site.
- Do not put enums or packed types with implicit backing types into `extern`
  APIs.

## MMIO, DMA, And FPGA Contracts

- Keep MMIO register layout generated from the regmap source of truth.
- Do not duplicate register constants by hand across Zig and RTL.
- Use volatile MMIO access only in the low-level driver layer.
- Keep DMA buffer ownership and cache maintenance explicit.
- Do not mix physical addresses, virtual addresses, and handles without distinct
  types or clearly named wrappers.
- Datatype-specific behavior belongs in the weight decoder and quant packers,
  not throughout the runtime.

## Logging, Trace, And Profiling

- Keep tracing and profiling comptime-gated where possible.
- Device profiling records should be fixed-size binary spans.
- Host code formats and aggregates profiling output.
- Avoid formatting strings or JSON on device hot paths.
- Keep scoped profiling calls close to the work being measured.
- For Zig 0.16 time APIs, prefer `std.Io.Timestamp`, `std.Io.Duration`, and
  clock/time facilities reachable through `std.Io`.
- Do not rely on the removed `{D}` duration formatter. Format
  `std.Io.Duration` values explicitly.

## Testing Expectations

- Add colocated `test` blocks for pure modules.
- Use `std.testing.io` for I/O tests and `std.testing.allocator` for allocation
  tests.
- Use `zig build test --test-timeout ...` for suites where hangs are plausible:
  protocol, transport, and in-process integration.
- Focus tests on wire encode/decode, malformed frames, invalid tags, q1a8
  pack/unpack/merge, slab extents, scalar matmul, lowerer mapping, and in-process
  host/device command execution.
- Use golden vectors to de-risk numerics.
- Use the scalar matmul reference as the oracle for software and RTL simulation.
- Keep hardware tests minimal and differential: real path vs reference path.

## Build System

- Keep `zig build` fast and hardware-free.
- Put proprietary or slow Vivado bitstream builds outside the normal Zig build.
- Add custom steps for local/free tooling: tests, golden tests, Verilator
  simulation, and benchmarks.
- Keep generated files reproducible and tied to their source schema.
- Do not add environment-variable or `dlopen` requirements for the llama backend.
- In Zig 0.16.x, fetched packages may appear in project-local `zig-pkg/`. Do not
  commit it unless the repo deliberately vendors dependencies.
- Keep package fingerprints current in `build.zig.zon`.
- Use build-system C translation for llama/ggml headers instead of new
  `@cImport` blocks.
- Use `--fork=[path]` for temporary local dependency overrides.
- Use `--error-style minimal` when concise build logs help CI.

## Style

- Use `const Self = @This();` inside structs.
- Use `init`, `deinit`, and `reset` consistently.
- Use `encode`/`decode` for binary wire data, `parse` for CLI or text data, and
  `read`/`write` for byte-stream operations.
- Use `[]const u8` for borrowed immutable bytes and `[]u8` for mutable buffers.
- Use many-item pointers only at C, MMIO, or low-level buffer boundaries.
- Keep comments short and useful; explain non-obvious invariants, not syntax.
- Prefer ASCII unless the surrounding file already uses non-ASCII for a reason.
- Prefer current `std.mem` names, such as `find` instead of older "index of"
  naming.
- Prefer `std.Io.Reader` and `std.Io.Writer` over removed `std.io` helpers such
  as `GenericReader`, `AnyReader`, and `FixedBufferStream`.

## Footguns To Avoid

- Floating Zig versions or pre-0.16 global process/network/time APIs.
- Hidden allocation, ambiguous slice ownership, or `anyerror` in public
  contracts.
- Native-endian wire encoding, pointer-casting untrusted bytes, `usize` in
  protocols/files/MMIO, or packed structs as portable wire formats.
- Pointers inside packed types, or implicit backing types in extern/MMIO/generated
  boundaries.
- Runtime registries for closed op sets, overused `anytype`, or comptime
  frameworks that obscure simple control flow.
- Deprecated `@Type` reification or new `@cImport` blocks.
- Runtime vector indexing or unchecked arithmetic without reference tests.
- Raw C pointers escaping wrappers.
- Allocation or string formatting on device hot paths.
- Duplicated register constants between Zig and RTL.
