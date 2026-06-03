Repo guidance for agents and contributors working on `penzai`.

This project is a clean Zig rewrite of a host/device/FPGA accelerator stack. The
main engineering goal is explicit systems code: fixed contracts, visible
ownership, narrow errors, target-specific modules, and testable reference paths.

## Zig Version

- Pin and document one exact Zig version for the repo.
- Do not assume source or build API compatibility across Zig releases.
- Keep `build.zig`, CI, and local development tooling on the same version.
- Avoid depending on `master` behavior unless the repo explicitly tracks master.
- If targeting Zig 0.16.x, design new code around `std.Io` and
  `std.process.Init` from the start.

## Target Support

- Treat supported targets as an explicit project contract, not an assumption.
- For the PYNQ-Z1 device side, verify the exact ARM Linux target triple,
  libc choice, CPU features, and minimum kernel version in `build.zig`.
- Zig 0.16.0 lists `arm-linux`, `armeb-linux`, `thumb-linux`, and
  `thumbeb-linux` as Tier 2 targets, so the board runtime should stay within
  standard-library APIs that work for those targets.
- Avoid depending on target behavior that is only Tier 3, Tier 4, or unknown
  unless the code path is isolated and tested.
- Big-endian and unusual page-size bugs are exactly the kind of failures that
  corrupt protocols and DMA code. Keep endian, alignment, and page-size
  assumptions explicit.

## I/O And Main

- In Zig 0.16.x, filesystem, networking, process, time, entropy, and many sync
  APIs are routed through `std.Io`.
- Application entry points should prefer:

```zig
pub fn main(init: std.process.Init) !void
```

- Pull `gpa`, `arena`, `io`, args, and environment from `std.process.Init`
  instead of using global process APIs.
- Functions that may block, use time, use randomness, touch files, spawn
  processes, or use networking should accept `io: std.Io` or store it on a
  context struct.
- Use `std.testing.io` in tests that exercise I/O.
- Do not create ad hoc `Io.Threaded` instances deep in library code. Construct
  I/O at the application boundary and pass it down.
- Use `std.Io` sync primitives when synchronized code must cooperate with the
  selected I/O implementation.
- Propagate `error.Canceled` unless the current function initiated the
  cancelation and can account for any completed side effects.
- CPU-bound long-running host tasks may add explicit cancelation points, but
  device hot paths should not grow I/O dependencies.

## Directory Boundaries

- `shared/` must stay OS-agnostic and frontend-agnostic.
  - No llama, ggml, sockets, filesystems, threads, or platform APIs.
  - Use fixed-width types in protocols and persistent data.
- `host/` owns CLI, llama.cpp, ggml backend registration, lowering, transport
  clients, and report formatting.
- `device/` owns runtime dispatch, scheduling, memory slabs, ARM kernels, and PL
  driver integration.
  - It must not import or depend on llama/ggml.
- `fpga/` owns RTL, simulations, bitstream scripts, and generated register-map
  consumers.
- If device code needs to know a ggml detail, the host lowerer or wire contract is
  missing information.

## Allocation And Ownership

- Allocation must be explicit. Pass an allocator to code that allocates.
- Prefer preallocated buffers, fixed buffers, slabs, and reusable arenas on hot
  paths.
- Avoid general-purpose allocation in the device steady state.
- Use arena allocation for short-lived host setup and per-command temporary data
  that can be freed together.
- Use `std.testing.allocator` in tests that allocate.
- Prefer unmanaged standard-library containers where applicable: store the
  container separately and pass the allocator to methods that allocate.
- Do not introduce `std.heap.ThreadSafeAllocator`; it was removed in Zig 0.16.0.
  Choose an allocator whose implementation has the required concurrency
  properties.
- Owned resources should follow the normal pattern:

```zig
pub fn init(allocator: Allocator, ...) !Self
pub fn deinit(self: *Self) void
pub fn reset(self: *Self) void
```

- Use `defer` and `errdefer` during initialization.
- Do not store slices whose ownership is ambiguous. If a field owns memory,
  `deinit` frees it. If it borrows memory, the lifetime must be clear from the
  API.

## Error Handling

- Keep public error sets narrow, especially in `shared/` and `device/`.
- Avoid exposing `anyerror` from protocol, wire, kernel, allocator, and runtime
  APIs.
- Define named error sets for public contracts:

```zig
pub const DecodeError = error{
    BadMagic,
    UnsupportedVersion,
    Truncated,
    InvalidTag,
};
```

- Use `try` for propagation and `switch` when policy depends on the error.
- Do not collapse distinct protocol or runtime failures into one generic error.

## Wire Format

- Treat all incoming wire bytes as hostile.
- Never leak `usize`, pointers, native-endian values, or platform-specific layout
  into the wire contract.
- Use fixed-width integer types: `u8`, `u16`, `u32`, `u64`, `i32`, etc.
- Give wire enums explicit backing types:

```zig
pub const OpTag = enum(u16) {
    matmul_q1a8 = 1,
    rmsnorm = 2,
};
```

- Prefer explicit binary encode/decode helpers over pointer-casting byte buffers.
- Do not rely on packed struct layout as a portable protocol.
- Specify endianness at every integer read/write.
- Validate lengths, tags, versions, alignment requirements, and handle ranges
  before dispatch.
- Keep `shared/protocol/wire.zig` as the host/device contract source of truth.
- If a `packed struct`, `packed union`, or `enum` crosses an extern, MMIO, or
  generated-code boundary, give it an explicit backing integer type.
- Do not put pointers in `packed struct` or `packed union` types. Store integer
  addresses only at low-level boundaries and convert with care.
- Avoid packed unions with fields of different bit sizes unless the unused bits
  and backing integer are explicit.

## Protocol And Dispatch

- The host produces explicit command buffers. The device consumes them.
- Device scheduling follows declared dependencies; it does not infer graph shape.
- Prefer closed tagged unions and exhaustive `switch` dispatch for device ops:

```zig
pub const Command = union(OpTag) {
    matmul_q1a8: MatmulQ1A8,
    rmsnorm: RmsNorm,
};
```

- Do not introduce kernel registries for the closed device op set.
- Add a protocol version or feature bit when changing host/device compatibility.

## Comptime And Generics

- Use `comptime` for structure that is actually compile-time:
  - schema tables
  - op metadata
  - quant format specialization
  - register-map generation
  - build configuration
  - target-specific composition
- Keep comptime code auditable. Prefer small tables and validation over clever
  pseudo-reflection.
- Use `anytype` sparingly. Prefer concrete types for ordinary functions.
- Good generic boundaries include `Link(Transport)`, quant packers, and runtime
  composition over fake vs real kernels.
- Avoid APIs whose compiler errors become harder to understand than handwritten
  code.
- Do not use deprecated `@Type`-based type construction in new code. Use the
  Zig 0.16 builtins such as `@Int`, `@Struct`, `@Union`, `@Enum`, `@Pointer`,
  `@Fn`, and `@Tuple`.
- Declare error sets explicitly with `error{ ... }`; do not try to reify error
  sets at comptime.
- Be careful with default field values and type metadata in generated schemas;
  Zig 0.16 has stricter and clearer dependency-loop rules.

## Runtime Polymorphism

- Prefer concrete structs, tagged unions, and exhaustive switches.
- Use vtables or runtime interfaces only for seams that buy testability or target
  substitution:
  - transport byte-pipes
  - slab/CMA vs fake memory
  - PL matmul vs scalar reference matmul
  - thin C-library boundaries
- Do not make every op or kernel a trait-like object.

## Integer And Numeric Code

- Develop and test kernels in safe modes first.
- Use explicit wrapping, saturating, or checked arithmetic when overflow behavior
  matters.
- Do not rely on accidental overflow.
- Integer matmul paths must be bit-exact against the scalar reference.
- Floating glue ops should use documented tolerances.
- Keep reference kernels simple, slow if necessary, and obviously correct.
- Only use unchecked operations in tiny, benchmarked sections with tests proving
  equivalence.
- Runtime indexing into vectors is forbidden in Zig 0.16. Coerce vectors to
  arrays before iterating dynamically, or keep indexes comptime-known.
- Do not pointer-cast between array memory and vector memory. Use value
  coercions where vector/array conversion is intended.
- Keep integer-to-float conversions explicit when precision may be lost.

## C Interop

- Isolate llama.cpp and ggml C interop in `host/llama.zig` and nearby backend
  wrapper code.
- Prefer build-system C translation over `@cImport` for new code. Put includes
  in a small C header, translate it in `build.zig`, and import the generated
  module from Zig.
- Convert C pointers, nullability, integer widths, and ownership rules at the
  boundary.
- Do not let raw C pointers spread through host logic.
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
- Datatype-specific behavior belongs in the weight decoder and quant packers, not
  throughout the runtime.

## Logging, Trace, And Profiling

- Keep tracing and profiling comptime-gated where possible.
- Device profiling records should be fixed-size binary spans.
- Host code formats and aggregates profiling output.
- Avoid formatting strings or JSON on device hot paths.
- Keep scoped profiling calls close to the work being measured.
- For Zig 0.16 time APIs, prefer `std.Io.Timestamp`, `std.Io.Duration`, and the
  clock/time facilities reachable through `std.Io`.
- Do not rely on the removed `{D}` duration formatter. Format
  `std.Io.Duration` values explicitly.

## Testing Expectations

- Add colocated `test` blocks for pure modules.
- Use `std.testing.io` for I/O tests and `std.testing.allocator` for allocation
  tests.
- Use `zig build test --test-timeout ...` for suites where hangs are plausible,
  especially protocol, transport, and inproc integration tests.
- Use focused tests for:
  - wire encode/decode round trips
  - malformed frames and invalid tags
  - q1a8 pack/unpack/merge behavior
  - slab extent allocation
  - scalar matmul reference correctness
  - lowerer op mapping
  - in-process host/device command execution
- Use golden vectors to de-risk numerics.
- Use the scalar matmul reference as the oracle for software and RTL simulation.
- Keep hardware tests minimal and differential: real path vs reference path.

## Build System

- Keep `zig build` fast and hardware-free.
- Put proprietary or slow Vivado bitstream builds outside the normal Zig build.
- Add custom build steps for local/free tooling such as tests, golden tests,
  Verilator simulation, and benchmarks.
- Keep generated files reproducible and tied to their source schema.
- Do not add environment-variable or `dlopen` requirements for the llama backend.
- In Zig 0.16.x, expect fetched packages to appear in project-local `zig-pkg/`.
  Do not commit it unless the repo deliberately vendors dependencies.
- Keep package fingerprints current in `build.zig.zon`.
- Use build-system C translation for llama/ggml headers instead of adding new
  `@cImport` blocks.
- Use `--fork=[path]` for temporary local dependency overrides instead of
  editing fetched packages directly as a permanent workflow.
- Use `--error-style minimal` when a concise build log is useful in CI.

## Style

- Use `const Self = @This();` inside structs.
- Use `init`, `deinit`, and `reset` consistently.
- Use `encode`/`decode` for binary wire data.
- Use `parse` for CLI or text data.
- Use `read`/`write` for byte-stream operations.
- Use `[]const u8` for borrowed immutable bytes and `[]u8` for mutable buffers.
- Use many-item pointers only at C, MMIO, or low-level buffer boundaries.
- Keep comments short and useful; explain non-obvious invariants, not syntax.
- Prefer ASCII unless the surrounding file already uses non-ASCII for a reason.
- Prefer current `std.mem` names in new code, such as `find` instead of older
  "index of" naming.
- Prefer `std.Io.Reader` and `std.Io.Writer` APIs over removed `std.io`
  helpers such as `GenericReader`, `AnyReader`, and `FixedBufferStream`.

## Footguns To Avoid

- Floating Zig versions.
- Writing new code against pre-0.16 global filesystem, networking, process,
  time, entropy, or argument APIs.
- Creating I/O implementations deep inside reusable modules.
- Hidden allocation.
- Ambiguous slice ownership.
- `anyerror` in public contracts.
- Overusing `anytype`.
- Native-endian wire encoding.
- Pointer-casting untrusted byte buffers.
- Using `packed struct` as a cross-platform protocol.
- Pointers inside packed structs or packed unions.
- Implicit enum or packed backing types in extern/MMIO/generated boundaries.
- Letting `usize` into wire formats, files, or MMIO registers.
- Runtime registries for closed op sets.
- Comptime frameworks that obscure simple control flow.
- Deprecated `@Type` reification in new metaprogramming.
- New `@cImport` blocks instead of build-system C translation.
- Runtime vector indexing.
- Raw C pointers escaping wrapper modules.
- Allocation or string formatting on device hot paths.
- Duplicated register constants between Zig and RTL.
- Unchecked arithmetic without a reference test.
