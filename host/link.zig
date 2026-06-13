const std = @import("std");
const shared = @import("shared");
const runtime_mod = @import("runtime");
const server = @import("server");
const host_tcp = @import("transport/tcp.zig");

const framing = shared.framing;
const protocol_transport = shared.protocol_transport;
const wire = shared.wire;
const profiling = shared.profiling;

pub const LinkError = error{
    OutOfMemory,
    Protocol,
    RemoteFailed,
    Transport,
};

/// Device-reported service time for one link op (from `wire.ResponseMeta`).
/// The host pairs it with its own measured round trip to split transport from
/// device work. Zero when the device was built without profiling.
pub const OpTiming = struct {
    device_service_ns: u64 = 0,
};

/// `alloc` result: the granted range plus the op's service timing.
pub const AllocResult = struct {
    range: wire.TensorRange,
    timing: OpTiming = .{},
};

pub const Client = struct {
    const Self = @This();

    ctx: *anyopaque,
    vtable: *const VTable,

    const VTable = struct {
        hello: *const fn (*anyopaque) LinkError!void,
        alloc: *const fn (*anyopaque, u64, u32) LinkError!AllocResult,
        free: *const fn (*anyopaque, wire.TensorRange) LinkError!OpTiming,
        upload: *const fn (*anyopaque, wire.TensorRange, []const u8) LinkError!OpTiming,
        fill: *const fn (*anyopaque, wire.TensorRange, u8) LinkError!OpTiming,
        download: *const fn (*anyopaque, wire.TensorRange, []u8) LinkError!OpTiming,
        run_graph: *const fn (*anyopaque, []const u8, []const wire.Command) LinkError!void,
        run_graph_profile: *const fn (*anyopaque, []const u8, []const wire.Command, wire.ProfileTier) LinkError!ProfiledRunGraph,
    };

    pub fn init(link: anytype) Self {
        const Ptr = @TypeOf(link);
        const info = @typeInfo(Ptr);
        if (info != .pointer) @compileError("Link.Client.init expects a pointer");
        const Child = info.pointer.child;

        const adapter = struct {
            fn ptr(ctx: *anyopaque) *Child {
                return @ptrCast(@alignCast(ctx));
            }

            fn callHello(ctx: *anyopaque) LinkError!void {
                return ptr(ctx).hello();
            }

            fn callAlloc(ctx: *anyopaque, nbytes: u64, alignment: u32) LinkError!AllocResult {
                return ptr(ctx).alloc(nbytes, alignment);
            }

            fn callFree(ctx: *anyopaque, range: wire.TensorRange) LinkError!OpTiming {
                return ptr(ctx).free(range);
            }

            fn callUpload(ctx: *anyopaque, range: wire.TensorRange, bytes: []const u8) LinkError!OpTiming {
                return ptr(ctx).upload(range, bytes);
            }

            fn callFill(ctx: *anyopaque, range: wire.TensorRange, value: u8) LinkError!OpTiming {
                return ptr(ctx).fill(range, value);
            }

            fn callDownload(ctx: *anyopaque, range: wire.TensorRange, out: []u8) LinkError!OpTiming {
                return ptr(ctx).download(range, out);
            }

            fn callRunGraph(ctx: *anyopaque, preload_bytes: []const u8, commands: []const wire.Command) LinkError!void {
                return ptr(ctx).runGraphPreload(preload_bytes, commands);
            }

            fn callRunGraphProfile(ctx: *anyopaque, preload_bytes: []const u8, commands: []const wire.Command, tier: wire.ProfileTier) LinkError!ProfiledRunGraph {
                return ptr(ctx).runGraphProfilePreload(preload_bytes, commands, tier);
            }

            const vtable = VTable{
                .hello = callHello,
                .alloc = callAlloc,
                .free = callFree,
                .upload = callUpload,
                .fill = callFill,
                .download = callDownload,
                .run_graph = callRunGraph,
                .run_graph_profile = callRunGraphProfile,
            };
        };

        return .{
            .ctx = @ptrCast(link),
            .vtable = &adapter.vtable,
        };
    }

    pub fn hello(self: Self) LinkError!void {
        return self.vtable.hello(self.ctx);
    }

    pub fn alloc(self: Self, nbytes: u64, alignment: u32) LinkError!AllocResult {
        return self.vtable.alloc(self.ctx, nbytes, alignment);
    }

    pub fn free(self: Self, range: wire.TensorRange) LinkError!OpTiming {
        return self.vtable.free(self.ctx, range);
    }

    pub fn upload(self: Self, range: wire.TensorRange, bytes: []const u8) LinkError!OpTiming {
        return self.vtable.upload(self.ctx, range, bytes);
    }

    pub fn fill(self: Self, range: wire.TensorRange, value: u8) LinkError!OpTiming {
        return self.vtable.fill(self.ctx, range, value);
    }

    pub fn download(self: Self, range: wire.TensorRange, out: []u8) LinkError!OpTiming {
        return self.vtable.download(self.ctx, range, out);
    }

    pub fn runGraph(self: Self, commands: []const wire.Command) LinkError!void {
        return self.runGraphPreload(&.{}, commands);
    }

    pub fn runGraphPreload(self: Self, preload_bytes: []const u8, commands: []const wire.Command) LinkError!void {
        return self.vtable.run_graph(self.ctx, preload_bytes, commands);
    }

    pub fn runGraphProfile(self: Self, commands: []const wire.Command, tier: wire.ProfileTier) LinkError!ProfiledRunGraph {
        return self.runGraphProfilePreload(&.{}, commands, tier);
    }

    pub fn runGraphProfilePreload(self: Self, preload_bytes: []const u8, commands: []const wire.Command, tier: wire.ProfileTier) LinkError!ProfiledRunGraph {
        return self.vtable.run_graph_profile(self.ctx, preload_bytes, commands, tier);
    }
};

pub const RpcTiming = struct {
    request_bytes: u64 = 0,
    response_bytes: u64 = 0,
    round_trip_ns: u64 = 0,
};

pub const ProfiledRunGraph = struct {
    allocator: std.mem.Allocator,
    rpc: RpcTiming,
    report: profiling.Report,

    pub fn deinit(self: *ProfiledRunGraph) void {
        self.report.deinit(self.allocator);
        self.* = undefined;
    }
};

pub const FakeLink = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    runtime: *runtime_mod.Runtime,
    io: ?std.Io = null,
    next_request_id: u64 = 1,

    pub fn init(allocator: std.mem.Allocator, runtime: *runtime_mod.Runtime) Self {
        return .{ .allocator = allocator, .runtime = runtime };
    }

    pub fn initWithIo(allocator: std.mem.Allocator, runtime: *runtime_mod.Runtime, io: std.Io) Self {
        return .{ .allocator = allocator, .runtime = runtime, .io = io };
    }

    pub fn hello(self: *Self) LinkError!void {
        var meta: [32]u8 = undefined;
        const id = self.nextId();
        const meta_len = wire.encodeHello(&meta, id) catch return error.Protocol;
        var response = try self.call(meta[0..meta_len], "");
        defer response.deinit(self.allocator);
        try response.expectOk(id);
    }

    pub fn alloc(self: *Self, nbytes: u64, alignment: u32) LinkError!AllocResult {
        var meta: [64]u8 = undefined;
        const id = self.nextId();
        const meta_len = wire.encodeAlloc(&meta, id, nbytes, alignment) catch return error.Protocol;
        var response = try self.call(meta[0..meta_len], "");
        defer response.deinit(self.allocator);
        try response.expectOk(id);
        return .{
            .range = .{ .handle = response.meta.handle, .offset = 0, .nbytes = response.meta.nbytes },
            .timing = .{ .device_service_ns = response.meta.device_service_ns },
        };
    }

    pub fn free(self: *Self, range: wire.TensorRange) LinkError!OpTiming {
        var meta: [32]u8 = undefined;
        const id = self.nextId();
        const meta_len = wire.encodeFree(&meta, id, range.handle) catch return error.Protocol;
        var response = try self.call(meta[0..meta_len], "");
        defer response.deinit(self.allocator);
        try response.expectOk(id);
        return .{ .device_service_ns = response.meta.device_service_ns };
    }

    pub fn upload(self: *Self, range: wire.TensorRange, bytes: []const u8) LinkError!OpTiming {
        var meta: [64]u8 = undefined;
        const id = self.nextId();
        const meta_len = wire.encodeUpload(&meta, id, range) catch return error.Protocol;
        var response = try self.call(meta[0..meta_len], bytes);
        defer response.deinit(self.allocator);
        try response.expectOk(id);
        return .{ .device_service_ns = response.meta.device_service_ns };
    }

    pub fn fill(self: *Self, range: wire.TensorRange, value: u8) LinkError!OpTiming {
        var meta: [64]u8 = undefined;
        const id = self.nextId();
        const meta_len = wire.encodeFill(&meta, id, range, value) catch return error.Protocol;
        var response = try self.call(meta[0..meta_len], "");
        defer response.deinit(self.allocator);
        try response.expectOk(id);
        return .{ .device_service_ns = response.meta.device_service_ns };
    }

    pub fn download(self: *Self, range: wire.TensorRange, out: []u8) LinkError!OpTiming {
        var meta: [64]u8 = undefined;
        const id = self.nextId();
        const meta_len = wire.encodeDownload(&meta, id, range) catch return error.Protocol;
        var response = try self.call(meta[0..meta_len], "");
        defer response.deinit(self.allocator);
        try response.expectOk(id);
        if (response.payload.len != out.len) return error.Protocol;
        @memcpy(out, response.payload);
        return .{ .device_service_ns = response.meta.device_service_ns };
    }

    pub fn runGraph(self: *Self, commands: []const wire.Command) LinkError!void {
        return self.runGraphPreload(&.{}, commands);
    }

    pub fn runGraphPreload(self: *Self, preload_bytes: []const u8, commands: []const wire.Command) LinkError!void {
        return runGraphImpl(self, preload_bytes, commands);
    }

    pub fn runGraphProfile(self: *Self, commands: []const wire.Command, tier: wire.ProfileTier) LinkError!ProfiledRunGraph {
        return self.runGraphProfilePreload(&.{}, commands, tier);
    }

    pub fn runGraphProfilePreload(self: *Self, preload_bytes: []const u8, commands: []const wire.Command, tier: wire.ProfileTier) LinkError!ProfiledRunGraph {
        return runGraphProfileImpl(self, self.io, preload_bytes, commands, tier);
    }

    fn call(self: *Self, metadata: []const u8, payload: []const u8) LinkError!Response {
        const request_len = framing.encodedLen(metadata.len, payload.len) catch return error.Protocol;
        const request_frame = try self.allocator.alloc(u8, request_len);
        defer self.allocator.free(request_frame);
        _ = framing.encode(metadata, payload, request_frame) catch return error.Protocol;

        const response_frame = server.handleFrame(self.io, self.allocator, self.runtime, request_frame) catch return error.Protocol;
        errdefer self.allocator.free(response_frame);
        const frame = framing.decode(response_frame) catch return error.Protocol;
        const meta = wire.decodeResponseMeta(frame.metadata) catch return error.Protocol;
        return .{ .frame = response_frame, .meta = meta, .payload = frame.payload };
    }

    fn nextId(self: *Self) u64 {
        const id = self.next_request_id;
        self.next_request_id += 1;
        return id;
    }
};

pub const TcpLink = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    endpoint: host_tcp.Endpoint,
    next_request_id: u64 = 1,

    pub fn connect(allocator: std.mem.Allocator, io: std.Io, spec: protocol_transport.TcpSpec) LinkError!Self {
        const endpoint = host_tcp.Endpoint.connect(io, spec) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            error.Protocol => return error.Protocol,
            error.InvalidAddress, error.Transport => return error.Transport,
        };
        return .{
            .allocator = allocator,
            .endpoint = endpoint,
        };
    }

    pub fn deinit(self: *Self) void {
        self.endpoint.deinit();
        self.* = undefined;
    }

    pub fn hello(self: *Self) LinkError!void {
        var meta: [32]u8 = undefined;
        const id = self.nextId();
        const meta_len = wire.encodeHello(&meta, id) catch return error.Protocol;
        var response = try self.call(meta[0..meta_len], "");
        defer response.deinit(self.allocator);
        try response.expectOk(id);
    }

    pub fn alloc(self: *Self, nbytes: u64, alignment: u32) LinkError!AllocResult {
        var meta: [64]u8 = undefined;
        const id = self.nextId();
        const meta_len = wire.encodeAlloc(&meta, id, nbytes, alignment) catch return error.Protocol;
        var response = try self.call(meta[0..meta_len], "");
        defer response.deinit(self.allocator);
        try response.expectOk(id);
        return .{
            .range = .{ .handle = response.meta.handle, .offset = 0, .nbytes = response.meta.nbytes },
            .timing = .{ .device_service_ns = response.meta.device_service_ns },
        };
    }

    pub fn free(self: *Self, range: wire.TensorRange) LinkError!OpTiming {
        var meta: [32]u8 = undefined;
        const id = self.nextId();
        const meta_len = wire.encodeFree(&meta, id, range.handle) catch return error.Protocol;
        var response = try self.call(meta[0..meta_len], "");
        defer response.deinit(self.allocator);
        try response.expectOk(id);
        return .{ .device_service_ns = response.meta.device_service_ns };
    }

    pub fn upload(self: *Self, range: wire.TensorRange, bytes: []const u8) LinkError!OpTiming {
        var meta: [64]u8 = undefined;
        const id = self.nextId();
        const meta_len = wire.encodeUpload(&meta, id, range) catch return error.Protocol;
        var response = try self.call(meta[0..meta_len], bytes);
        defer response.deinit(self.allocator);
        try response.expectOk(id);
        return .{ .device_service_ns = response.meta.device_service_ns };
    }

    pub fn fill(self: *Self, range: wire.TensorRange, value: u8) LinkError!OpTiming {
        var meta: [64]u8 = undefined;
        const id = self.nextId();
        const meta_len = wire.encodeFill(&meta, id, range, value) catch return error.Protocol;
        var response = try self.call(meta[0..meta_len], "");
        defer response.deinit(self.allocator);
        try response.expectOk(id);
        return .{ .device_service_ns = response.meta.device_service_ns };
    }

    pub fn download(self: *Self, range: wire.TensorRange, out: []u8) LinkError!OpTiming {
        var meta: [64]u8 = undefined;
        const id = self.nextId();
        const meta_len = wire.encodeDownload(&meta, id, range) catch return error.Protocol;
        const request_len = framing.encodedLen(meta_len, 0) catch return error.Protocol;
        const request_frame = try self.allocator.alloc(u8, request_len);
        defer self.allocator.free(request_frame);
        _ = framing.encode(meta[0..meta_len], "", request_frame) catch return error.Protocol;

        var response = self.endpoint.callInto(self.allocator, request_frame, out) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            error.Protocol => return error.Protocol,
            error.InvalidAddress, error.Transport => return error.Transport,
        };
        defer response.deinit(self.allocator);

        const response_meta = wire.decodeResponseMeta(response.metadata) catch return error.Protocol;
        if (response_meta.request_id != id) return error.Protocol;
        if (response_meta.status != .ok) return error.RemoteFailed;
        if (!response.payload_copied or response.payload_len != out.len) return error.Protocol;
        return .{ .device_service_ns = response_meta.device_service_ns };
    }

    pub fn runGraph(self: *Self, commands: []const wire.Command) LinkError!void {
        return self.runGraphPreload(&.{}, commands);
    }

    pub fn runGraphPreload(self: *Self, preload_bytes: []const u8, commands: []const wire.Command) LinkError!void {
        return runGraphImpl(self, preload_bytes, commands);
    }

    pub fn runGraphProfile(self: *Self, commands: []const wire.Command, tier: wire.ProfileTier) LinkError!ProfiledRunGraph {
        return self.runGraphProfilePreload(&.{}, commands, tier);
    }

    pub fn runGraphProfilePreload(self: *Self, preload_bytes: []const u8, commands: []const wire.Command, tier: wire.ProfileTier) LinkError!ProfiledRunGraph {
        return runGraphProfileImpl(self, self.endpoint.io, preload_bytes, commands, tier);
    }

    fn call(self: *Self, metadata: []const u8, payload: []const u8) LinkError!Response {
        const request_len = framing.encodedLen(metadata.len, payload.len) catch return error.Protocol;
        const request_frame = try self.allocator.alloc(u8, request_len);
        defer self.allocator.free(request_frame);
        _ = framing.encode(metadata, payload, request_frame) catch return error.Protocol;

        const response_frame = self.endpoint.call(self.allocator, request_frame) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            error.Protocol => return error.Protocol,
            error.InvalidAddress, error.Transport => return error.Transport,
        };
        errdefer self.allocator.free(response_frame);
        const frame = framing.decode(response_frame) catch return error.Protocol;
        const meta = wire.decodeResponseMeta(frame.metadata) catch return error.Protocol;
        return .{ .frame = response_frame, .meta = meta, .payload = frame.payload };
    }

    fn nextId(self: *Self) u64 {
        const id = self.next_request_id;
        self.next_request_id += 1;
        return id;
    }
};

// Shared run_graph bodies for FakeLink and TcpLink. They differ only in their
// transport (`self.call`) and io source, so the encode/RPC/decode logic lives here.
fn encodeCommands(allocator: std.mem.Allocator, commands: []const wire.Command) LinkError![]u8 {
    const command_len = wire.commandBufferLen(commands) catch return error.Protocol;
    const command_bytes = try allocator.alloc(u8, command_len);
    errdefer allocator.free(command_bytes);
    _ = wire.encodeCommandBuffer(commands, command_bytes) catch return error.Protocol;
    return command_bytes;
}

fn buildRunGraphPayload(allocator: std.mem.Allocator, preload_bytes: []const u8, commands: []const wire.Command) LinkError![]u8 {
    const command_bytes = try encodeCommands(allocator, commands);
    defer allocator.free(command_bytes);
    const payload_len = wire.runGraphPayloadLen(preload_bytes.len, command_bytes.len) catch return error.Protocol;
    const payload = try allocator.alloc(u8, payload_len);
    errdefer allocator.free(payload);
    _ = wire.encodeRunGraphPayload(payload, preload_bytes, command_bytes) catch return error.Protocol;
    return payload;
}

fn runGraphImpl(self: anytype, preload_bytes: []const u8, commands: []const wire.Command) LinkError!void {
    const payload = try buildRunGraphPayload(self.allocator, preload_bytes, commands);
    defer self.allocator.free(payload);

    var meta: [32]u8 = undefined;
    const id = self.nextId();
    const meta_len = wire.encodeRunGraph(&meta, id, .off) catch return error.Protocol;
    var response = try self.call(meta[0..meta_len], payload);
    defer response.deinit(self.allocator);
    try response.expectOk(id);
}

fn runGraphProfileImpl(self: anytype, io: ?std.Io, preload_bytes: []const u8, commands: []const wire.Command, tier: wire.ProfileTier) LinkError!ProfiledRunGraph {
    const payload = try buildRunGraphPayload(self.allocator, preload_bytes, commands);
    defer self.allocator.free(payload);

    var meta: [32]u8 = undefined;
    const id = self.nextId();
    const meta_len = wire.encodeRunGraph(&meta, id, tier) catch return error.Protocol;
    const request_len = framing.encodedLen(meta_len, payload.len) catch return error.Protocol;
    const start_ns = profiling.nowNs(io);
    var response = try self.call(meta[0..meta_len], payload);
    const end_ns = profiling.nowNs(io);
    defer response.deinit(self.allocator);
    try response.expectOk(id);
    var report = profiling.decodeAlloc(self.allocator, response.payload) catch return error.Protocol;
    errdefer report.deinit(self.allocator);
    return .{
        .allocator = self.allocator,
        .rpc = .{
            .request_bytes = @intCast(request_len),
            .response_bytes = @intCast(response.frame.len),
            .round_trip_ns = profiling.elapsed(start_ns, end_ns),
        },
        .report = report,
    };
}

const Response = struct {
    frame: []u8,
    meta: wire.ResponseMeta,
    payload: []const u8,

    fn deinit(self: *Response, allocator: std.mem.Allocator) void {
        allocator.free(self.frame);
        self.* = undefined;
    }

    fn expectOk(self: Response, request_id: u64) LinkError!void {
        if (self.meta.request_id != request_id) return error.Protocol;
        if (self.meta.status != .ok) return error.RemoteFailed;
    }
};
