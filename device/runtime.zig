const std = @import("std");
const wire = @import("wire");
const heap_mod = @import("heap");
const ps_activations = @import("ps_activations");
const ps_elemwise = @import("ps_elemwise");
const ps_matmul_q1a8 = @import("ps_matmul_q1a8");
const ps_rmsnorm = @import("ps_rmsnorm");
const ps_rows = @import("ps_rows");
const ps_rope = @import("ps_rope");
const ps_softmax = @import("ps_softmax");

pub const RuntimeError = error{
    InvalidRequest,
    OutOfMemory,
    UnknownHandle,
    OutOfBounds,
    UnsupportedOp,
    BackendFailure,
};

pub const Runtime = RuntimeFor(heap_mod.Heap);

pub fn RuntimeFor(comptime Heap: type) type {
    return struct {
        const Self = @This();

        allocator: std.mem.Allocator,
        heap: Heap,

        pub fn init(allocator: std.mem.Allocator, heap_size: usize) !Self {
            return .{
                .allocator = allocator,
                .heap = try Heap.init(allocator, heap_size),
            };
        }

        pub fn deinit(self: *Self) void {
            self.heap.deinit();
            self.* = undefined;
        }

        pub fn dispatch(self: *Self, request: wire.Request) RuntimeError!DispatchResult {
            return switch (request) {
                .hello => |request_id| .{ .meta = ok(request_id) },
                .alloc => |req| blk: {
                    const range = self.heap.allocate(req.nbytes, req.alignment) catch |err| switch (err) {
                        error.OutOfMemory => return error.OutOfMemory,
                        error.InvalidAlignment => return error.InvalidRequest,
                        else => return mapHeapError(err),
                    };
                    break :blk .{ .meta = .{
                        .request_id = req.request_id,
                        .status = .ok,
                        .handle = range.handle,
                        .nbytes = range.nbytes,
                    } };
                },
                .free => |req| blk: {
                    self.heap.free(req.handle) catch |err| return mapHeapError(err);
                    break :blk .{ .meta = ok(req.request_id) };
                },
                .upload => |req| blk: {
                    self.heap.write(req.range, req.bytes) catch |err| return mapHeapError(err);
                    break :blk .{ .meta = .{
                        .request_id = req.request_id,
                        .status = .ok,
                        .nbytes = req.bytes.len,
                    } };
                },
                .download => |req| blk: {
                    const bytes = self.heap.read(req.range) catch |err| return mapHeapError(err);
                    break :blk .{ .meta = .{
                        .request_id = req.request_id,
                        .status = .ok,
                        .nbytes = bytes.len,
                    }, .payload = bytes };
                },
                .run_graph => |req| blk: {
                    const commands = wire.decodeCommandBuffer(self.allocator, req.command_bytes) catch |err| switch (err) {
                        error.OutOfMemory => return error.OutOfMemory,
                        else => return error.InvalidRequest,
                    };
                    defer self.allocator.free(commands);
                    for (commands) |command| try self.execute(command);
                    break :blk .{ .meta = .{
                        .request_id = req.request_id,
                        .status = .ok,
                        .value0 = commands.len,
                    } };
                },
            };
        }

        fn execute(self: *Self, command: wire.Command) RuntimeError!void {
            switch (command) {
                .copy => |copy| {
                    const src = self.heap.read(copy.src) catch |err| return mapHeapError(err);
                    self.heap.write(copy.dst, src) catch |err| return mapHeapError(err);
                },
                .matmul_q1a8 => |matmul| {
                    const weights = self.heap.read(matmul.weights) catch |err| return mapHeapError(err);
                    const acts = self.heap.read(matmul.acts) catch |err| return mapHeapError(err);
                    const dst = self.heap.bytes(matmul.dst) catch |err| return mapHeapError(err);
                    ps_matmul_q1a8.runQ1A8(self.allocator, weights, acts, dst, matmul.rows, matmul.cols, matmul.k) catch |err| switch (err) {
                        error.OutOfMemory => return error.OutOfMemory,
                        else => return error.InvalidRequest,
                    };
                },
                .rmsnorm => |rmsnorm| {
                    const input = self.heap.read(rmsnorm.input) catch |err| return mapHeapError(err);
                    const dst = self.heap.bytes(rmsnorm.dst) catch |err| return mapHeapError(err);
                    ps_rmsnorm.runBytes(input, dst, rmsnorm.rows, rmsnorm.cols, rmsnorm.eps) catch |err| return mapKernelError(err);
                },
                .rope => |rope| {
                    const input = self.heap.read(rope.input) catch |err| return mapHeapError(err);
                    const positions = self.heap.read(rope.positions) catch |err| return mapHeapError(err);
                    const dst = self.heap.bytes(rope.dst) catch |err| return mapHeapError(err);
                    ps_rope.applyBytes(input, positions, dst, .{
                        .head_dim = rope.head_dim,
                        .n_heads = rope.n_heads,
                        .n_tokens = rope.n_tokens,
                        .n_dims = rope.n_dims,
                        .mode = ropeMode(rope.mode),
                        .n_ctx_orig = rope.n_ctx_orig,
                        .freq_base = rope.freq_base,
                        .freq_scale = rope.freq_scale,
                        .ext_factor = rope.ext_factor,
                        .attn_factor = rope.attn_factor,
                        .beta_fast = rope.beta_fast,
                        .beta_slow = rope.beta_slow,
                    }) catch |err| return mapKernelError(err);
                },
                .softmax => |softmax| {
                    const src = self.heap.read(softmax.src) catch |err| return mapHeapError(err);
                    const dst = self.heap.bytes(softmax.dst) catch |err| return mapHeapError(err);
                    ps_softmax.runBytes(src, dst) catch |err| return mapKernelError(err);
                },
                .silu => |silu| {
                    const src = self.heap.read(silu.src) catch |err| return mapHeapError(err);
                    const dst = self.heap.bytes(silu.dst) catch |err| return mapHeapError(err);
                    ps_activations.siluBytes(src, dst) catch |err| return mapKernelError(err);
                },
                .swiglu => |swiglu| {
                    const gate = self.heap.read(swiglu.lhs) catch |err| return mapHeapError(err);
                    const up = self.heap.read(swiglu.rhs) catch |err| return mapHeapError(err);
                    const dst = self.heap.bytes(swiglu.dst) catch |err| return mapHeapError(err);
                    ps_activations.swigluBytes(gate, up, dst) catch |err| return mapKernelError(err);
                },
                .add_f32 => |add| {
                    const lhs = self.heap.read(add.lhs) catch |err| return mapHeapError(err);
                    const rhs = self.heap.read(add.rhs) catch |err| return mapHeapError(err);
                    const dst = self.heap.bytes(add.dst) catch |err| return mapHeapError(err);
                    ps_elemwise.add2dBytes(lhs, rhs, dst, add.rows, add.cols, rhsRowBroadcast(add.mode)) catch |err| return mapKernelError(err);
                },
                .mul_f32 => |mul| {
                    const lhs = self.heap.read(mul.lhs) catch |err| return mapHeapError(err);
                    const rhs = self.heap.read(mul.rhs) catch |err| return mapHeapError(err);
                    const dst = self.heap.bytes(mul.dst) catch |err| return mapHeapError(err);
                    ps_elemwise.mul2dBytes(lhs, rhs, dst, mul.rows, mul.cols, rhsRowBroadcast(mul.mode)) catch |err| return mapKernelError(err);
                },
                .scale_f32 => |scale| {
                    const src = self.heap.read(scale.src) catch |err| return mapHeapError(err);
                    const dst = self.heap.bytes(scale.dst) catch |err| return mapHeapError(err);
                    ps_elemwise.scaleBytes(src, scale.scale, dst) catch |err| return mapKernelError(err);
                },
                .add_scaled_f32 => |add_scaled| {
                    const lhs = self.heap.read(add_scaled.lhs) catch |err| return mapHeapError(err);
                    const rhs = self.heap.read(add_scaled.rhs) catch |err| return mapHeapError(err);
                    const dst = self.heap.bytes(add_scaled.dst) catch |err| return mapHeapError(err);
                    ps_elemwise.addScaledBytes(lhs, rhs, add_scaled.rhs_scale, dst) catch |err| return mapKernelError(err);
                },
                .set_rows => |set_rows| {
                    const src = self.heap.read(set_rows.src) catch |err| return mapHeapError(err);
                    const indices = self.heap.read(set_rows.indices) catch |err| return mapHeapError(err);
                    const dst = self.heap.bytes(set_rows.dst) catch |err| return mapHeapError(err);
                    ps_rows.setRowsF32ToF16Bytes(set_rows, src, indices, dst) catch |err| return mapKernelError(err);
                },
                .get_rows => |get_rows| {
                    const src = self.heap.read(get_rows.src) catch |err| return mapHeapError(err);
                    const indices = self.heap.read(get_rows.indices) catch |err| return mapHeapError(err);
                    const dst = self.heap.bytes(get_rows.dst) catch |err| return mapHeapError(err);
                    ps_rows.getRowsF32Bytes(get_rows, src, indices, dst) catch |err| return mapKernelError(err);
                },
            }
        }
    };
}

pub const DispatchResult = struct {
    meta: wire.ResponseMeta,
    payload: []const u8 = &.{},
};

pub fn errorCode(err: RuntimeError) wire.ErrorCode {
    return switch (err) {
        error.InvalidRequest => .invalid_request,
        error.OutOfMemory => .out_of_memory,
        error.UnknownHandle => .unknown_handle,
        error.OutOfBounds => .out_of_bounds,
        error.UnsupportedOp => .unsupported_op,
        error.BackendFailure => .backend_failure,
    };
}

fn ok(request_id: u64) wire.ResponseMeta {
    return .{ .request_id = request_id, .status = .ok };
}

fn rhsRowBroadcast(mode: wire.BinaryF32Mode) bool {
    return switch (mode) {
        .same_shape => false,
        .rhs_row_broadcast => true,
    };
}

fn ropeMode(mode: wire.RopeMode) ps_rope.Mode {
    return switch (mode) {
        .normal => .normal,
        .neox => .neox,
    };
}

fn mapHeapError(err: anyerror) RuntimeError {
    return switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        error.UnknownHandle => error.UnknownHandle,
        error.OutOfBounds => error.OutOfBounds,
        error.InvalidAlignment => error.InvalidRequest,
        error.BackendFailure => error.BackendFailure,
        else => error.BackendFailure,
    };
}

fn mapKernelError(err: anyerror) RuntimeError {
    return switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        else => error.InvalidRequest,
    };
}

test "runtime dispatches ps f32 command variants" {
    var runtime = try Runtime.init(std.testing.allocator, 4096);
    defer runtime.deinit();

    const a = try tensor(&runtime, 2);
    const b = try tensor(&runtime, 2);
    const get_rows_src = try tensor(&runtime, 4);
    const out_rms = try tensor(&runtime, 2);
    const out_rope = try tensor(&runtime, 2);
    const out_softmax = try tensor(&runtime, 2);
    const out_silu = try tensor(&runtime, 2);
    const out_swiglu = try tensor(&runtime, 2);
    const out_add = try tensor(&runtime, 2);
    const out_mul = try tensor(&runtime, 2);
    const out_scale = try tensor(&runtime, 2);
    const out_add_scaled = try tensor(&runtime, 2);
    const indices = try rawTensor(&runtime, @sizeOf(i32), @alignOf(i32));
    const out_rows = try rawTensor(&runtime, 4 * @sizeOf(f16), @alignOf(f16));
    const out_get_rows = try tensor(&runtime, 2);

    try writeTensor(&runtime, a, &.{ 3, 4 });
    try writeTensor(&runtime, b, &.{ 1, 2 });
    try writeTensor(&runtime, get_rows_src, &.{ 1, 2, 3, 4 });
    try writeI32Tensor(&runtime, indices, &.{1});

    const commands = [_]wire.Command{
        .{ .rmsnorm = .{ .input = a, .dst = out_rms, .rows = 2, .cols = 1, .eps = 0 } },
        .{ .rope = .{
            .input = b,
            .positions = indices,
            .dst = out_rope,
            .head_dim = 2,
            .n_heads = 1,
            .n_tokens = 1,
            .n_dims = 2,
            .mode = .normal,
            .n_ctx_orig = 0,
            .freq_base = 10000,
            .freq_scale = 1,
            .ext_factor = 0,
            .attn_factor = 1,
            .beta_fast = 0,
            .beta_slow = 0,
        } },
        .{ .softmax = .{ .src = b, .dst = out_softmax } },
        .{ .silu = .{ .src = b, .dst = out_silu } },
        .{ .swiglu = .{ .lhs = b, .rhs = a, .dst = out_swiglu } },
        .{ .add_f32 = .{ .lhs = a, .rhs = b, .dst = out_add, .rows = 2, .cols = 1, .mode = .same_shape } },
        .{ .mul_f32 = .{ .lhs = a, .rhs = b, .dst = out_mul, .rows = 2, .cols = 1, .mode = .same_shape } },
        .{ .scale_f32 = .{ .src = b, .dst = out_scale, .scale = 0.5 } },
        .{ .add_scaled_f32 = .{ .lhs = a, .rhs = b, .dst = out_add_scaled, .rhs_scale = 0.5 } },
        .{ .set_rows = .{
            .src = a,
            .indices = indices,
            .dst = out_rows,
            .index_type = .i32,
            .head_dim = 2,
            .ne01 = 1,
            .ne02 = 1,
            .ne03 = 1,
            .ne11 = 1,
            .ne12 = 1,
            .src_nb1 = 2 * @sizeOf(f32),
            .src_nb2 = 2 * @sizeOf(f32),
            .src_nb3 = 2 * @sizeOf(f32),
            .indices_nb1 = @sizeOf(i32),
            .indices_nb2 = @sizeOf(i32),
            .dst_nb1 = 2 * @sizeOf(f16),
            .dst_nb2 = 4 * @sizeOf(f16),
            .dst_nb3 = 4 * @sizeOf(f16),
        } },
        .{ .get_rows = .{
            .src = get_rows_src,
            .indices = indices,
            .dst = out_get_rows,
            .src_type = .f32,
            .row_width = 2,
            .src_rows = 2,
            .ne10 = 1,
            .ne11 = 1,
            .ne12 = 1,
            .src_nb1 = 2 * @sizeOf(f32),
            .src_nb2 = 2 * @sizeOf(f32),
            .src_nb3 = 2 * @sizeOf(f32),
            .indices_nb1 = @sizeOf(i32),
            .indices_nb2 = @sizeOf(i32),
            .dst_nb1 = 2 * @sizeOf(f32),
            .dst_nb2 = 2 * @sizeOf(f32),
            .dst_nb3 = 2 * @sizeOf(f32),
        } },
    };
    var command_bytes: [2048]u8 = undefined;
    const command_len = try wire.encodeCommandBuffer(&commands, &command_bytes);

    const result = try runtime.dispatch(.{ .run_graph = .{
        .request_id = 99,
        .command_bytes = command_bytes[0..command_len],
    } });
    try std.testing.expectEqual(wire.Status.ok, result.meta.status);
    try std.testing.expectEqual(@as(u64, commands.len), result.meta.value0);

    try expectTensor(&runtime, out_rms, &.{ 0.84852815, 1.1313709 });
    try expectTensor(&runtime, out_rope, &.{ -1.1426396, 1.9220756 });
    try expectTensor(&runtime, out_softmax, &.{ 0.26894143, 0.7310586 });
    try expectTensor(&runtime, out_silu, &.{ 0.7310586, 1.761594 });
    try expectTensor(&runtime, out_swiglu, &.{ 2.1931758, 7.046376 });
    try expectTensor(&runtime, out_add, &.{ 4, 6 });
    try expectTensor(&runtime, out_mul, &.{ 3, 8 });
    try expectTensor(&runtime, out_scale, &.{ 0.5, 1 });
    try expectTensor(&runtime, out_add_scaled, &.{ 3.5, 5 });
    try expectF16TensorAt(&runtime, out_rows, 2, &.{ 3, 4 });
    try expectTensor(&runtime, out_get_rows, &.{ 3, 4 });
}

fn tensor(runtime: *Runtime, len: usize) !wire.TensorRange {
    return runtime.heap.allocate(len * @sizeOf(f32), @alignOf(f32));
}

fn rawTensor(runtime: *Runtime, len: usize, tensor_alignment: u32) !wire.TensorRange {
    return runtime.heap.allocate(len, tensor_alignment);
}

fn writeTensor(runtime: *Runtime, range: wire.TensorRange, values: []const f32) !void {
    const bytes = try runtime.heap.bytes(range);
    try std.testing.expectEqual(values.len * @sizeOf(f32), bytes.len);
    for (values, 0..) |value, i| {
        writeF32(bytes, i, value);
    }
}

fn writeI32Tensor(runtime: *Runtime, range: wire.TensorRange, values: []const i32) !void {
    const bytes = try runtime.heap.bytes(range);
    try std.testing.expectEqual(values.len * @sizeOf(i32), bytes.len);
    for (values, 0..) |value, i| {
        std.mem.writeInt(i32, bytes[i * @sizeOf(i32) ..][0..4], value, .little);
    }
}

fn expectTensor(runtime: *Runtime, range: wire.TensorRange, expected: []const f32) !void {
    const bytes = try runtime.heap.read(range);
    try std.testing.expectEqual(expected.len * @sizeOf(f32), bytes.len);
    for (expected, 0..) |value, i| {
        try std.testing.expect(@abs(value - readF32(bytes, i)) <= 0.000001);
    }
}

fn expectF16TensorAt(runtime: *Runtime, range: wire.TensorRange, start_index: usize, expected: []const f32) !void {
    const bytes = try runtime.heap.read(range);
    for (expected, 0..) |value, i| {
        const index = start_index + i;
        const half: f16 = @bitCast(std.mem.readInt(u16, bytes[index * @sizeOf(f16) ..][0..2], .little));
        try std.testing.expect(@abs(value - @as(f32, @floatCast(half))) <= 0.000001);
    }
}

fn writeF32(bytes: []u8, index: usize, value: f32) void {
    std.mem.writeInt(u32, bytes[index * @sizeOf(f32) ..][0..4], @bitCast(value), .little);
}

fn readF32(bytes: []const u8, index: usize) f32 {
    return @bitCast(std.mem.readInt(u32, bytes[index * @sizeOf(f32) ..][0..4], .little));
}
