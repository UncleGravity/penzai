//! Persistent request runtime for the resident inference engine.

const std = @import("std");
const engine = @import("engine");
const engine_regmap = @import("engine_regmap");
const memory = @import("memory");
const shared = @import("shared");

const scheduler_mod = @import("scheduler.zig");

const capabilities = shared.capabilities;
const metrics = shared.engine.metrics;
const rpc = shared.engine.rpc;
const wire = shared.wire;

pub const RuntimeError = error{
    InvalidRequest,
    OutOfMemory,
    UnknownHandle,
    OutOfBounds,
    UnsupportedOp,
    BackendFailure,
};

pub const Runtime = RuntimeFor(memory.fake.Heap, void);
pub const XrtRuntime = RuntimeFor(memory.xrt.Heap, engine.driver.Driver);

/// The void-driver specialization exists only for protocol and memory tests. It
/// never advertises or accepts inference requests.
pub fn RuntimeFor(comptime Heap: type, comptime HardwareDriver: type) type {
    const has_hardware = HardwareDriver != void;
    const Scheduler = scheduler_mod.SchedulerFor(HardwareDriver);

    return struct {
        const Self = @This();

        allocator: std.mem.Allocator,
        heap: Heap,
        scheduler: Scheduler,
        receipt: capabilities.Receipt = .{},

        pub fn init(allocator: std.mem.Allocator, heap_size: usize) !Self {
            return initWithReceipt(allocator, heap_size, .{});
        }

        pub fn initWithReceipt(
            allocator: std.mem.Allocator,
            heap_size: usize,
            receipt: capabilities.Receipt,
        ) !Self {
            var heap = try Heap.init(allocator, heap_size);
            errdefer heap.deinit();

            if (comptime has_hardware) {
                const hardware = HardwareDriver.open(
                    engine_regmap.addr.kernel,
                ) catch return error.EngineOpenFailed;
                std.debug.print(
                    "inference engine ready (interface {x}, {d} Hz)\n",
                    .{ hardware.version, hardware.clk_hz },
                );
                return .{
                    .allocator = allocator,
                    .heap = heap,
                    .scheduler = Scheduler.init(hardware),
                    .receipt = receipt,
                };
            }

            return .{
                .allocator = allocator,
                .heap = heap,
                .scheduler = Scheduler.init({}),
                .receipt = receipt,
            };
        }

        pub fn deinit(self: *Self) void {
            self.scheduler.deinit();
            self.heap.deinit();
            self.* = undefined;
        }

        pub fn dispatch(
            self: *Self,
            request: wire.Request,
            io: ?std.Io,
        ) RuntimeError!DispatchResult {
            return self.dispatchMeasured(request, io, 0);
        }

        pub fn dispatchMeasured(
            self: *Self,
            request: wire.Request,
            io: ?std.Io,
            request_decode_ns: u64,
        ) RuntimeError!DispatchResult {
            _ = io;
            _ = request_decode_ns;
            return switch (request) {
                .hello => |request_id| .{ .meta = ok(request_id) },
                .capabilities => |request_id| self.inspect(request_id),
                .alloc => |allocation| self.allocate(allocation),
                .free => |release_request| self.release(release_request),
                .upload => |upload_request| self.upload(upload_request),
                .inference => |inference| self.dispatchInference(inference),
            };
        }

        fn inspect(self: *const Self, request_id: u64) RuntimeError!DispatchResult {
            const payload = self.allocator.alloc(
                u8,
                capabilities.encoded_len,
            ) catch return error.OutOfMemory;
            errdefer self.allocator.free(payload);
            _ = capabilities.encode(self.capabilityReport(), payload) catch
                unreachable;
            return .{
                .meta = .{
                    .request_id = request_id,
                    .status = .ok,
                    .nbytes = payload.len,
                },
                .payload = payload,
                .owns_payload = true,
            };
        }

        fn allocate(
            self: *Self,
            request: wire.AllocRequest,
        ) RuntimeError!DispatchResult {
            const range = self.heap.allocate(
                request.nbytes,
                request.alignment,
            ) catch |err| switch (err) {
                error.OutOfMemory => return error.OutOfMemory,
                error.InvalidAlignment => return error.InvalidRequest,
                else => return mapHeapError(err),
            };
            return .{ .meta = .{
                .request_id = request.request_id,
                .status = .ok,
                .handle = range.handle,
                .nbytes = range.nbytes,
            } };
        }

        fn release(
            self: *Self,
            request: wire.FreeRequest,
        ) RuntimeError!DispatchResult {
            if (self.scheduler.ownsHandle(request.handle))
                return error.InvalidRequest;
            self.heap.free(request.handle) catch |err| return mapHeapError(err);
            return .{ .meta = ok(request.request_id) };
        }

        fn upload(
            self: *Self,
            request: wire.TransferRequest,
        ) RuntimeError!DispatchResult {
            self.heap.write(request.range, request.bytes) catch |err|
                return mapHeapError(err);
            return .{ .meta = .{
                .request_id = request.request_id,
                .status = .ok,
                .nbytes = request.bytes.len,
            } };
        }

        fn dispatchInference(
            self: *Self,
            request: wire.InferenceRequest,
        ) RuntimeError!DispatchResult {
            if (comptime !has_hardware) return error.UnsupportedOp;

            const payload = rpc.decode(request.action, request.bytes) catch
                return error.InvalidRequest;
            return switch (payload) {
                .install_model => |install| blk: {
                    self.scheduler.installModel(
                        &self.heap,
                        install,
                    ) catch |err| return mapSchedulerError(err);
                    break :blk .{ .meta = ok(request.request_id) };
                },
                .uninstall_model => |uninstall| blk: {
                    self.scheduler.uninstallModel(uninstall.slot) catch |err|
                        return mapSchedulerError(err);
                    break :blk .{ .meta = ok(request.request_id) };
                },
                .open_session => |open| blk: {
                    self.scheduler.openSession(&self.heap, open) catch |err|
                        return mapSchedulerError(err);
                    break :blk .{ .meta = ok(request.request_id) };
                },
                .close_session => |close| blk: {
                    self.scheduler.closeSession(
                        &self.heap,
                        close.slot,
                    ) catch |err| return mapSchedulerError(err);
                    break :blk .{ .meta = ok(request.request_id) };
                },
                .reset_session => |reset| blk: {
                    self.scheduler.resetSession(reset) catch |err|
                        return mapSchedulerError(err);
                    break :blk .{ .meta = ok(request.request_id) };
                },
                .execute => |execute_request| self.execute(
                    request.request_id,
                    execute_request,
                ),
            };
        }

        fn execute(
            self: *Self,
            request_id: u64,
            request: shared.engine.command.ExecuteTile,
        ) RuntimeError!DispatchResult {
            if (request.request_id != request_id) return error.InvalidRequest;

            // Reserve the response before the scheduler acquires a session lease.
            // Once hardware commits, response allocation can no longer fail.
            const result_bytes = self.allocator.alloc(
                u8,
                rpc.resultLen(request.metrics_level),
            ) catch return error.OutOfMemory;
            errdefer self.allocator.free(result_bytes);

            const result = self.scheduler.execute(request) catch |err|
                return mapSchedulerError(err);
            _ = rpc.encodeResult(result, result_bytes) catch unreachable;
            return .{
                .meta = .{
                    .request_id = request_id,
                    .status = .ok,
                    .nbytes = result_bytes.len,
                },
                .payload = result_bytes,
                .owns_payload = true,
            };
        }

        fn capabilityReport(self: *const Self) capabilities.Report {
            var report: capabilities.Report = .{
                .wire_abi = wire.version,
                .metrics_schema = metrics.schema_version,
            };
            report.applyReceipt(self.receipt);
            self.scheduler.addCapabilities(&report);
            return report;
        }
    };
}

pub const DispatchResult = struct {
    meta: wire.ResponseMeta,
    payload: []const u8 = &.{},
    owns_payload: bool = false,

    pub fn deinit(self: *DispatchResult, allocator: std.mem.Allocator) void {
        if (self.owns_payload) allocator.free(@constCast(self.payload));
        self.* = undefined;
    }
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
        else => error.BackendFailure,
    };
}

fn mapSchedulerError(err: anyerror) RuntimeError {
    return switch (err) {
        error.UnsupportedOp => error.UnsupportedOp,
        error.OutOfMemory => error.OutOfMemory,
        error.HeapFailure, error.BackendFailure => error.BackendFailure,
        else => error.InvalidRequest,
    };
}

test "runtime keeps engine-owned allocations out of generic release" {
    var runtime = try Runtime.init(std.testing.allocator, 4096);
    defer runtime.deinit();

    const range = try runtime.heap.allocate(64, 64);
    runtime.scheduler.manager.model_ranges[0] = range;
    try std.testing.expectError(error.InvalidRequest, runtime.dispatch(.{
        .free = .{ .request_id = 7, .handle = range.handle },
    }, null));

    runtime.scheduler.manager.model_ranges[0] = null;
    var result = try runtime.dispatch(.{
        .free = .{ .request_id = 8, .handle = range.handle },
    }, null);
    defer result.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(u64, 8), result.meta.request_id);
}

test "test runtime reports no accelerator and rejects inference" {
    var runtime = try Runtime.init(std.testing.allocator, 4096);
    defer runtime.deinit();

    var inspect = try runtime.dispatch(.{ .capabilities = 1 }, null);
    defer inspect.deinit(std.testing.allocator);
    const report = try capabilities.decode(inspect.payload);
    try std.testing.expectEqual(@as(u32, 0), report.feature_mask);

    try std.testing.expectError(error.UnsupportedOp, runtime.dispatch(.{
        .inference = .{
            .request_id = 2,
            .action = .close_session,
            .bytes = &[_]u8{0} ** rpc.slot_len,
        },
    }, null));
}
