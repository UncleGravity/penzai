const std = @import("std");
const wire = @import("wire");
const heap_mod = @import("heap");
const matmul_ref = @import("matmul_ref");

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
                    matmul_ref.runQ1A8(self.allocator, weights, acts, dst, matmul.rows, matmul.cols, matmul.k) catch |err| switch (err) {
                        error.OutOfMemory => return error.OutOfMemory,
                        else => return error.InvalidRequest,
                    };
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
