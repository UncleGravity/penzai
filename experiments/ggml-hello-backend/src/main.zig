//! ggml hello-backend: proving the riskiest penzai seam from Zig.
//!
//! penzai's host side must implement ggml's backend interface (the unstable
//! `ggml-backend-impl.h` vtables: backend / device / reg / buffer-type) from
//! Zig, register it in-process, and let llama.cpp's scheduler place ops on it.
//! This program proves that whole path against real ggml, with no C++ glue:
//!
//!   Stage 1 — implement the backend vtable in Zig and dispatch a graph
//!             straight through `graph_compute` (proves @cImport + link +
//!             ggml calling back into Zig + we compute correct results).
//!   Stage 2 — register a full device/reg/buffer-type and let
//!             `ggml_backend_sched` split a graph across TWO Zig backends:
//!             a "penzai" accelerator (GPU-type, MUL_MAT only — like the real
//!             PYNQ board) and a CPU-type fallback that takes the glue op.
//!             This is the exact shape of the real system (matmul offloaded,
//!             glue on the CPU fallback), and it exercises supports_op,
//!             buffer-type placement, and the multi-backend split.
//!
//! The CPU-type fallback here is a stand-in for llama.cpp's genuine CPU
//! backend; using a second Zig backend keeps the experiment self-contained
//! (no need to build ggml's large arch-specific ggml-cpu tree). The ABI risk
//! being de-risked — fn-pointer vtables, struct layout, scheduler interplay —
//! is identical either way.
//!
//! Verification is bit-exact: inputs are small integers, exact in f32.

const std = @import("std");
const c = @import("c");

const GGML_F32 = c.GGML_TYPE_F32;

// ── tiny f32 tensor helpers ───────────────────────────────────────────────

fn f32ptr(t: ?*c.ggml_tensor) [*]f32 {
    return @ptrCast(@alignCast(t.?.*.data.?));
}

fn nelem(t: *const c.ggml_tensor) usize {
    return @intCast(c.ggml_nelements(t));
}

// Reference compute (ggml semantics) used to verify every stage.
//   mul_mat: a[k,n] · b[k,m] -> dst[n,m], dst[i,j] = Σ_l a[l,i]*b[l,j]
fn refMulMat(a: []const f32, b: []const f32, dst: []f32, k: usize, n: usize, m: usize) void {
    for (0..m) |j| {
        for (0..n) |i| {
            var acc: f32 = 0;
            for (0..k) |l| acc += a[i * k + l] * b[j * k + l];
            dst[j * n + i] = acc;
        }
    }
}

fn fill(slice: []f32, seed: usize) void {
    for (slice, 0..) |*v, idx| {
        // small integers → exact in f32, so == comparisons are valid
        v.* = @floatFromInt(@as(i64, @intCast((idx + seed) % 7)) - 3);
    }
}

fn exactEq(got: []const f32, want: []const f32) bool {
    if (got.len != want.len) return false;
    for (got, want) |g, w| if (g != w) return false;
    return true;
}

// ════════════════════════════════════════════════════════════════════════
// Custom backend, parameterized by a `Dev` (one set of ggml plumbing each).
//
// A host-memory backend: buffers are plain malloc, tensors are CPU-addressable
// (is_host = true). That mirrors the old project's v0, where board "tensors"
// were host memory the matmul read directly. Swapping malloc for a CMA/RPC
// buffer type is the next layer — not this one.
// ════════════════════════════════════════════════════════════════════════

const Dev = struct {
    name: [*:0]const u8,
    desc: [*:0]const u8,
    // translate-c renames `enum ggml_backend_dev_type` because a function of
    // the same name exists; the bare name `c.ggml_backend_dev_type` is the getter.
    dev_type: c.enum_ggml_backend_dev_type,
    support_mul_mat: bool,
    support_add: bool,

    // ggml plumbing — addresses are stable (these live in globals), so the
    // vtables recover the owning Dev from each struct's `context` pointer.
    backend: c.ggml_backend = undefined,
    device: c.ggml_backend_device = undefined,
    reg: c.ggml_backend_reg = undefined,
    buft: c.ggml_backend_buffer_type = undefined,

    fn wire(self: *Dev) void {
        self.reg = .{ .api_version = c.GGML_BACKEND_API_VERSION, .iface = reg_iface, .context = self };
        self.device = .{ .iface = device_iface, .reg = &self.reg, .context = self };
        self.buft = .{ .iface = buft_iface, .device = &self.device, .context = self };
        self.backend = .{ .guid = null, .iface = backend_iface, .device = &self.device, .context = self };
    }
};

fn devOf(ctx: ?*anyopaque) *Dev {
    return @ptrCast(@alignCast(ctx.?));
}

// -- buffer (operates on buffer.context = the malloc base) -----------------

fn bufGetBase(buf: c.ggml_backend_buffer_t) callconv(.c) ?*anyopaque {
    return buf.?.*.context;
}
fn bufFree(buf: c.ggml_backend_buffer_t) callconv(.c) void {
    std.c.free(buf.?.*.context);
}
fn bufSetTensor(buf: c.ggml_backend_buffer_t, tensor: ?*c.ggml_tensor, data: ?*const anyopaque, offset: usize, size: usize) callconv(.c) void {
    _ = buf;
    const dst: [*]u8 = @ptrCast(@alignCast(tensor.?.*.data.?));
    const src: [*]const u8 = @ptrCast(data.?);
    @memcpy(dst[offset .. offset + size], src[0..size]);
}
fn bufGetTensor(buf: c.ggml_backend_buffer_t, tensor: ?*const c.ggml_tensor, data: ?*anyopaque, offset: usize, size: usize) callconv(.c) void {
    _ = buf;
    const src: [*]const u8 = @ptrCast(@alignCast(tensor.?.*.data.?));
    const dst: [*]u8 = @ptrCast(data.?);
    @memcpy(dst[0..size], src[offset .. offset + size]);
}
fn bufClear(buf: c.ggml_backend_buffer_t, value: u8) callconv(.c) void {
    const base: [*]u8 = @ptrCast(@alignCast(buf.?.*.context.?));
    @memset(base[0..buf.?.*.size], value);
}

const buffer_iface = c.ggml_backend_buffer_i{
    .free_buffer = &bufFree,
    .get_base = &bufGetBase,
    .init_tensor = null,
    .memset_tensor = null,
    .set_tensor = &bufSetTensor,
    .get_tensor = &bufGetTensor,
    .set_tensor_2d = null,
    .get_tensor_2d = null,
    .cpy_tensor = null,
    .clear = &bufClear,
    .reset = null,
};

// -- buffer type -----------------------------------------------------------

fn buftGetName(buft: c.ggml_backend_buffer_type_t) callconv(.c) [*c]const u8 {
    return devOf(buft.?.*.context).name;
}
fn buftAllocBuffer(buft: c.ggml_backend_buffer_type_t, size: usize) callconv(.c) c.ggml_backend_buffer_t {
    const mem = std.c.malloc(if (size == 0) 1 else size) orelse return null;
    return c.ggml_backend_buffer_init(buft, buffer_iface, mem, size);
}
fn buftGetAlignment(buft: c.ggml_backend_buffer_type_t) callconv(.c) usize {
    _ = buft;
    return 32;
}
fn buftIsHost(buft: c.ggml_backend_buffer_type_t) callconv(.c) bool {
    _ = buft;
    return true;
}

const buft_iface = c.ggml_backend_buffer_type_i{
    .get_name = &buftGetName,
    .alloc_buffer = &buftAllocBuffer,
    .get_alignment = &buftGetAlignment,
    .get_max_size = null,
    .get_alloc_size = null,
    .is_host = &buftIsHost,
};

// -- compute (the actual work) --------------------------------------------

fn computeNode(node: *c.ggml_tensor) void {
    switch (node.op) {
        c.GGML_OP_NONE, c.GGML_OP_RESHAPE, c.GGML_OP_VIEW, c.GGML_OP_PERMUTE, c.GGML_OP_TRANSPOSE => {},
        c.GGML_OP_MUL_MAT => {
            const a = node.src[0];
            const b = node.src[1];
            const k: usize = @intCast(a[0].ne[0]);
            const n: usize = @intCast(a[0].ne[1]);
            const m: usize = @intCast(b[0].ne[1]);
            refMulMat(f32ptr(a)[0 .. k * n], f32ptr(b)[0 .. k * m], f32ptr(node)[0 .. n * m], k, n, m);
        },
        c.GGML_OP_ADD => {
            const a = f32ptr(node.src[0])[0..nelem(node)];
            const b = f32ptr(node.src[1])[0..nelem(node)];
            const d = f32ptr(node)[0..nelem(node)];
            for (d, a, b) |*o, x, y| o.* = x + y;
        },
        else => std.debug.print("  WARNING: unsupported op {d} skipped\n", .{node.op}),
    }
}

fn backendGraphCompute(backend: c.ggml_backend_t, cgraph: ?*c.ggml_cgraph) callconv(.c) c.ggml_status {
    _ = backend;
    const g = cgraph.?;
    // ggml_cgraph is opaque in the public headers — use the accessors.
    const n = c.ggml_graph_n_nodes(g);
    var i: c_int = 0;
    while (i < n) : (i += 1) computeNode(c.ggml_graph_node(g, i).?);
    return c.GGML_STATUS_SUCCESS;
}
fn backendGetName(backend: c.ggml_backend_t) callconv(.c) [*c]const u8 {
    return devOf(backend.?.*.context).name;
}
fn backendFree(backend: c.ggml_backend_t) callconv(.c) void {
    _ = backend;
}

const backend_iface = c.ggml_backend_i{
    .get_name = &backendGetName,
    .free = &backendFree,
    .set_tensor_async = null,
    .get_tensor_async = null,
    .set_tensor_2d_async = null,
    .get_tensor_2d_async = null,
    .cpy_tensor_async = null,
    .synchronize = null,
    .graph_plan_create = null,
    .graph_plan_free = null,
    .graph_plan_update = null,
    .graph_plan_compute = null,
    .graph_compute = &backendGraphCompute,
    .event_record = null,
    .event_wait = null,
    .graph_optimize = null,
};

// -- device ----------------------------------------------------------------

fn devGetName(dev: c.ggml_backend_dev_t) callconv(.c) [*c]const u8 {
    return devOf(dev.?.*.context).name;
}
fn devGetDescription(dev: c.ggml_backend_dev_t) callconv(.c) [*c]const u8 {
    return devOf(dev.?.*.context).desc;
}
fn devGetMemory(dev: c.ggml_backend_dev_t, free: ?*usize, total: ?*usize) callconv(.c) void {
    _ = dev;
    free.?.* = 256 * 1024 * 1024;
    total.?.* = 256 * 1024 * 1024;
}
fn devGetType(dev: c.ggml_backend_dev_t) callconv(.c) c.enum_ggml_backend_dev_type {
    return devOf(dev.?.*.context).dev_type;
}
fn devGetProps(dev: c.ggml_backend_dev_t, props: [*c]c.ggml_backend_dev_props) callconv(.c) void {
    props.*.name = devGetName(dev);
    props.*.description = devGetDescription(dev);
    props.*.@"type" = devGetType(dev);
    devGetMemory(dev, &props.*.memory_free, &props.*.memory_total);
    props.*.caps = .{ .@"async" = false, .host_buffer = true, .buffer_from_host_ptr = false, .events = false };
}
fn devInitBackend(dev: c.ggml_backend_dev_t, params: [*c]const u8) callconv(.c) c.ggml_backend_t {
    _ = params;
    return &devOf(dev.?.*.context).backend;
}
fn devGetBufferType(dev: c.ggml_backend_dev_t) callconv(.c) c.ggml_backend_buffer_type_t {
    return &devOf(dev.?.*.context).buft;
}
fn isF32(t: ?*const c.ggml_tensor) bool {
    return t != null and t.?.*.@"type" == GGML_F32;
}
fn devSupportsOp(dev: c.ggml_backend_dev_t, op: ?*const c.ggml_tensor) callconv(.c) bool {
    const d = devOf(dev.?.*.context);
    const o = op orelse return false;
    return switch (o.op) {
        c.GGML_OP_MUL_MAT => d.support_mul_mat and isF32(o) and isF32(o.src[0]) and isF32(o.src[1]),
        c.GGML_OP_ADD => d.support_add and isF32(o) and isF32(o.src[0]) and isF32(o.src[1]),
        else => false,
    };
}
fn devSupportsBuft(dev: c.ggml_backend_dev_t, buft: c.ggml_backend_buffer_type_t) callconv(.c) bool {
    _ = dev;
    const b = buft orelse return false;
    // accept our own buft and any host buffer type (our kernels read host ptrs)
    if (b.*.iface.is_host) |is_host| return is_host(buft);
    return false;
}

const device_iface = c.ggml_backend_device_i{
    .get_name = &devGetName,
    .get_description = &devGetDescription,
    .get_memory = &devGetMemory,
    .get_type = &devGetType,
    .get_props = &devGetProps,
    .init_backend = &devInitBackend,
    .get_buffer_type = &devGetBufferType,
    .get_host_buffer_type = null,
    .buffer_from_host_ptr = null,
    .supports_op = &devSupportsOp,
    .supports_buft = &devSupportsBuft,
    .offload_op = null,
    .event_new = null,
    .event_free = null,
    .event_synchronize = null,
};

// -- reg -------------------------------------------------------------------

fn regGetName(reg: c.ggml_backend_reg_t) callconv(.c) [*c]const u8 {
    return devOf(reg.?.*.context).name;
}
fn regDevCount(reg: c.ggml_backend_reg_t) callconv(.c) usize {
    _ = reg;
    return 1;
}
fn regDevGet(reg: c.ggml_backend_reg_t, index: usize) callconv(.c) c.ggml_backend_dev_t {
    _ = index;
    return &devOf(reg.?.*.context).device;
}

const reg_iface = c.ggml_backend_reg_i{
    .get_name = &regGetName,
    .get_device_count = &regDevCount,
    .get_device = &regDevGet,
    .get_proc_address = null,
};

// ════════════════════════════════════════════════════════════════════════
// The two backends.
// ════════════════════════════════════════════════════════════════════════

var accel = Dev{
    .name = "penzai",
    .desc = "penzai PYNQ-Z1 accelerator (hello experiment, host-memory)",
    .dev_type = c.GGML_BACKEND_DEVICE_TYPE_GPU, // an offload target, not the CPU
    .support_mul_mat = true,
    .support_add = false, // matmul only, like the real board
};

var cpu_fallback = Dev{
    .name = "cpu-fallback",
    .desc = "CPU-type fallback (stands in for llama.cpp's CPU backend)",
    .dev_type = c.GGML_BACKEND_DEVICE_TYPE_CPU,
    .support_mul_mat = true,
    .support_add = true,
};

// ════════════════════════════════════════════════════════════════════════
// Stages
// ════════════════════════════════════════════════════════════════════════

var failures: usize = 0;

fn report(name: []const u8, ok: bool) void {
    std.debug.print("  [{s}] {s}\n", .{ if (ok) "PASS" else "FAIL", name });
    if (!ok) failures += 1;
}

const K = 64;
const N = 8;
const M = 4;

/// Stage 1 — implement the vtable in Zig, dispatch a graph straight through it.
fn stage1() void {
    std.debug.print("\nStage 1: custom Zig backend, direct graph_compute dispatch\n", .{});
    const ctx = c.ggml_init(.{ .mem_size = 16 * 1024 * 1024, .mem_buffer = null, .no_alloc = false }).?;
    defer c.ggml_free(ctx);

    const a = c.ggml_new_tensor_2d(ctx, GGML_F32, K, N);
    const b = c.ggml_new_tensor_2d(ctx, GGML_F32, K, M);
    const bias = c.ggml_new_tensor_2d(ctx, GGML_F32, N, M);
    fill(f32ptr(a)[0 .. K * N], 1);
    fill(f32ptr(b)[0 .. K * M], 5);
    fill(f32ptr(bias)[0 .. N * M], 2);
    const out = c.ggml_add(ctx, c.ggml_mul_mat(ctx, a, b), bias);

    const gf = c.ggml_new_graph(ctx);
    c.ggml_build_forward_expand(gf, out);

    // Dispatch straight into OUR vtable.
    _ = backendGraphCompute(&accel.backend, gf);

    var mmw: [N * M]f32 = undefined;
    refMulMat(f32ptr(a)[0 .. K * N], f32ptr(b)[0 .. K * M], &mmw, K, N, M);
    var want: [N * M]f32 = undefined;
    for (&want, mmw, f32ptr(bias)[0 .. N * M]) |*o, x, y| o.* = x + y;
    report("zig backend mul_mat+add matches reference", exactEq(f32ptr(out)[0 .. N * M], &want));
}

/// Stage 2 — scheduler splits a graph across the two Zig backends.
fn stage2() void {
    std.debug.print("\nStage 2: scheduler splits graph (matmul→penzai, add→cpu-fallback)\n", .{});

    const ctx = c.ggml_init(.{ .mem_size = 16 * 1024 * 1024, .mem_buffer = null, .no_alloc = true }).?;
    defer c.ggml_free(ctx);

    const a = c.ggml_new_tensor_2d(ctx, GGML_F32, K, N);
    _ = c.ggml_set_name(a, "a");
    c.ggml_set_input(a);
    const b = c.ggml_new_tensor_2d(ctx, GGML_F32, K, M);
    _ = c.ggml_set_name(b, "b");
    c.ggml_set_input(b);
    const bias = c.ggml_new_tensor_2d(ctx, GGML_F32, N, M);
    _ = c.ggml_set_name(bias, "bias");
    c.ggml_set_input(bias);
    const mm = c.ggml_mul_mat(ctx, a, b);
    _ = c.ggml_set_name(mm, "mm");
    const out = c.ggml_add(ctx, mm, bias);
    _ = c.ggml_set_name(out, "out");
    c.ggml_set_output(out);

    const gf = c.ggml_new_graph(ctx);
    c.ggml_build_forward_expand(gf, out);

    // penzai first (priority for supported ops), CPU-type fallback last (the
    // scheduler asserts the final backend is a CPU device).
    var backends = [_]c.ggml_backend_t{ &accel.backend, &cpu_fallback.backend };
    const sched = c.ggml_backend_sched_new(&backends, null, backends.len, @intCast(c.ggml_graph_size(gf)), false, false).?;
    defer c.ggml_backend_sched_free(sched);

    if (!c.ggml_backend_sched_alloc_graph(sched, gf)) {
        report("sched alloc_graph", false);
        return;
    }

    var adata: [K * N]f32 = undefined;
    var bdata: [K * M]f32 = undefined;
    var biasdata: [N * M]f32 = undefined;
    fill(&adata, 1);
    fill(&bdata, 5);
    fill(&biasdata, 2);
    c.ggml_backend_tensor_set(a, &adata, 0, @sizeOf(@TypeOf(adata)));
    c.ggml_backend_tensor_set(b, &bdata, 0, @sizeOf(@TypeOf(bdata)));
    c.ggml_backend_tensor_set(bias, &biasdata, 0, @sizeOf(@TypeOf(biasdata)));

    _ = c.ggml_backend_sched_graph_compute(sched, gf);

    const mm_on = c.ggml_backend_sched_get_tensor_backend(sched, mm);
    const add_on = c.ggml_backend_sched_get_tensor_backend(sched, out);
    report("scheduler placed mul_mat on penzai", mm_on == @as(c.ggml_backend_t, &accel.backend));
    report("scheduler placed add on cpu-fallback", add_on == @as(c.ggml_backend_t, &cpu_fallback.backend));

    var got: [N * M]f32 = undefined;
    c.ggml_backend_tensor_get(out, &got, 0, @sizeOf(@TypeOf(got)));
    var mmw: [N * M]f32 = undefined;
    refMulMat(&adata, &bdata, &mmw, K, N, M);
    var want: [N * M]f32 = undefined;
    for (&want, mmw, biasdata) |*o, x, y| o.* = x + y;
    report("scheduled split-graph result matches reference", exactEq(&got, &want));
}

pub fn main() void {
    accel.wire();
    cpu_fallback.wire();

    std.debug.print("penzai ggml hello-backend\nggml backend API version: {d}\n", .{c.GGML_BACKEND_API_VERSION});

    stage1();
    stage2();

    std.debug.print("\n{d} failure(s)\n", .{failures});
    if (failures != 0) std.process.exit(1);
    std.debug.print("ALL STAGES PASSED — the Zig<->ggml backend seam works.\n", .{});
}
