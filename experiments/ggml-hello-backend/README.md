# ggml hello-backend (Zig)

Proves the riskiest seam in the [penzai plan](../../plan.md): implementing
ggml's **backend interface from Zig**, in-process, and letting llama.cpp's
scheduler place ops on it. This is the part `plan.md` draws as a one-liner
(`backend.zig — ggml vtables; registered in-process`) but where most of the
real integration difficulty lives.

No CMake, no `.so`, no system toolchain: ggml-base is compiled straight into
the Zig artifact with Zig's own bundled clang — the "one build" penzai wants.

## Run

```sh
# from repo root; uses the zig pinned in ../../flake.nix
cd experiments/ggml-hello-backend
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git vendor/llama.cpp   # if not present
nix develop ../.. -c zig build run
```

Expected output:

```
Stage 1: custom Zig backend, direct graph_compute dispatch
  [PASS] zig backend mul_mat+add matches reference
Stage 2: scheduler splits graph (matmul→penzai, add→cpu-fallback)
  [PASS] scheduler placed mul_mat on penzai
  [PASS] scheduler placed add on cpu-fallback
  [PASS] scheduled split-graph result matches reference
ALL STAGES PASSED — the Zig<->ggml backend seam works.
```

## What it does

- **`src/main.zig`** implements the full `ggml-backend-impl.h` vtable set in Zig
  — backend, device, reg, and buffer-type — parameterized by a `Dev` struct so
  one implementation backs two registered backends.
  - **Stage 1** builds a `mul_mat`+`add` graph and dispatches it straight
    through our `graph_compute`. Proves ggml calls back into Zig fn-pointers and
    we compute bit-exact results.
  - **Stage 2** registers two backends — a `penzai` accelerator (GPU-type,
    MUL_MAT only, like the real PYNQ board) and a CPU-type fallback — and lets
    `ggml_backend_sched` split the graph. The scheduler routes the matmul to
    `penzai` and the `add` glue op to the CPU fallback, purely from `supports_op`
    + device type. **This is the exact shape of the real system** (matmul
    offloaded to the board, glue ops on the CPU/llama fallback).

The CPU-type fallback is a stand-in for llama.cpp's genuine CPU backend; a
second Zig backend keeps the experiment self-contained (no need to build ggml's
large arch-specific `ggml-cpu/` tree). The ABI risk being de-risked is identical.

## Findings that feed back into `plan.md`

Concrete things penzai's `build.zig` / `host/backend.zig` will have to handle,
discovered here:

1. **`@cImport` is gone in Zig 0.17.** C interop now goes through the build
   system's `translate-c` step (`b.addTranslateC` → a module). See `build.zig`.
2. **The vtable interface ports cleanly.** ggml's `ggml_backend_i` /
   `_device_i` / `_reg_i` / `_buffer_type_i` are plain C structs of function
   pointers; Zig `callconv(.c)` functions drop straight in. Struct layout and
   ABI match with zero shims. `GGML_BACKEND_API_VERSION == 2`.
3. **translate-c name collisions** to know about:
   - `enum ggml_backend_dev_type` collides with the function of the same name →
     the type is `c.enum_ggml_backend_dev_type` (bare name is the getter).
   - `type` is a Zig keyword → tensor/props fields are `t.@"type"`.
4. **`ggml_cgraph` is opaque** in the public headers (defined in `ggml-impl.h`),
   so a backend must walk it via `ggml_graph_n_nodes` / `ggml_graph_node`, not
   field access. Relevant to penzai's `lower.zig`.
5. **ggml does NULL-pointer arithmetic** (`incr_ptr_aligned`, used to size
   graphs) which is UB in C and trips Zig's UB sanitizer in Debug. Compile
   ggml's own TUs with `-fno-sanitize=undefined` (our Zig keeps full safety).
6. **ggml-base builds with zig cc, no CMake.** Needs `-D_DARWIN_C_SOURCE`
   (macOS) and `-DGGML_VERSION=...` / `-DGGML_COMMIT=...` (normally cmake-set).
   CMake itself was a dead end here — its compiler probe hangs under nix on
   macOS; Zig's hermetic clang sidesteps it entirely.
7. **The scheduler asserts the last backend is a CPU-type device**
   (`ggml-backend.cpp`). For penzai in-process this is satisfied for free by
   llama.cpp's real CPU backend, which is always present as the fallback.

## Not covered here (next seams)

- A real (non-host) buffer type backed by CMA/RPC handles instead of `malloc`
  — i.e. the actual `host↔device` boundary. This experiment uses host memory.
- Linking the genuine `ggml-cpu` + `llama` and running a `llama_decode` on a
  real GGUF. The vtable/ABI risk is now retired; that step is build-wiring plus
  the lowering surface (`lower.zig`).
- Cross-compiling the device daemon to armv7 (Zynq) — separate experiment.

Pinned upstream: `ggml-org/llama.cpp` (see `vendor/llama.cpp` after cloning;
ggml version 0.13.x, backend API v2).
