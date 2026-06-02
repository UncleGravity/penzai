const std = @import("std");
const c = @import("c");

const K = 4;
const N = 3;
const M = 2;

const SmokeError = error{
    InitFailed,
    TensorAllocFailed,
    GraphAllocFailed,
    BackendInitFailed,
    ComputeFailed,
    OutputMismatch,
};

pub fn main() !void {
    c.llama_backend_init();
    defer c.llama_backend_free();

    const model_params = c.llama_model_default_params();
    std.debug.print(
        "llama ok: default n_gpu_layers={d}\n",
        .{model_params.n_gpu_layers},
    );
    std.debug.print(
        "ggml ok: backend api={d}, backend_i_size={d}\n",
        .{ c.GGML_BACKEND_API_VERSION, @sizeOf(c.ggml_backend_i) },
    );

    try runGgmlCpuGraph();
    std.debug.print("smoke passed\n", .{});
}

fn runGgmlCpuGraph() !void {
    const params = c.ggml_init_params{
        .mem_size = 1024 * 1024,
        .mem_buffer = null,
        .no_alloc = false,
    };
    const ctx = c.ggml_init(params) orelse return SmokeError.InitFailed;
    defer c.ggml_free(ctx);

    const a = c.ggml_new_tensor_2d(ctx, c.GGML_TYPE_F32, K, N) orelse return SmokeError.TensorAllocFailed;
    const b = c.ggml_new_tensor_2d(ctx, c.GGML_TYPE_F32, K, M) orelse return SmokeError.TensorAllocFailed;
    const out = c.ggml_mul_mat(ctx, a, b) orelse return SmokeError.TensorAllocFailed;

    const a_values = tensorF32(a, K * N);
    const b_values = tensorF32(b, K * M);
    fill(a_values, 1);
    fill(b_values, 11);

    const graph = c.ggml_new_graph(ctx) orelse return SmokeError.GraphAllocFailed;
    c.ggml_build_forward_expand(graph, out);

    const backend = c.ggml_backend_cpu_init() orelse return SmokeError.BackendInitFailed;
    defer c.ggml_backend_free(backend);

    const status = c.ggml_backend_graph_compute(backend, graph);
    if (status != c.GGML_STATUS_SUCCESS) {
        std.debug.print("ggml compute failed: status={d}\n", .{status});
        return SmokeError.ComputeFailed;
    }

    var expected: [N * M]f32 = undefined;
    refMulMat(a_values, b_values, expected[0..], K, N, M);

    const got = tensorF32(out, N * M);
    if (!same(got, expected[0..])) {
        std.debug.print("output mismatch\n  got:  {any}\n  want: {any}\n", .{ got, expected });
        return SmokeError.OutputMismatch;
    }

    std.debug.print("ggml graph ok: f32 mul_mat [{d}x{d}] * [{d}x{d}]\n", .{ N, K, M, K });
}

fn tensorF32(tensor: *c.ggml_tensor, len: usize) []f32 {
    const ptr: [*]f32 = @ptrCast(@alignCast(tensor.data.?));
    return ptr[0..len];
}

fn fill(values: []f32, seed: usize) void {
    for (values, 0..) |*value, index| {
        const item: i32 = @intCast(((index + seed) % 7));
        value.* = @floatFromInt(item - 3);
    }
}

fn refMulMat(a: []const f32, b: []const f32, dst: []f32, k: usize, n: usize, m: usize) void {
    for (0..m) |col| {
        for (0..n) |row| {
            var acc: f32 = 0;
            for (0..k) |i| {
                acc += a[row * k + i] * b[col * k + i];
            }
            dst[col * n + row] = acc;
        }
    }
}

fn same(got: []const f32, expected: []const f32) bool {
    if (got.len != expected.len) return false;
    for (got, expected) |g, e| {
        if (g != e) return false;
    }
    return true;
}
