const std = @import("std");
const framing = @import("framing");
const protocol_transport = @import("protocol_transport");
const wire = @import("wire");
const runtime_mod = @import("runtime");
const server = @import("server");
const host_tcp = @import("host_tcp");

pub const LinkError = error{
    OutOfMemory,
    Protocol,
    RemoteFailed,
    Transport,
};

pub const Client = struct {
    const Self = @This();

    ctx: *anyopaque,
    vtable: *const VTable,

    const VTable = struct {
        hello: *const fn (*anyopaque) LinkError!void,
        alloc: *const fn (*anyopaque, u64, u32) LinkError!wire.TensorRange,
        free: *const fn (*anyopaque, wire.TensorRange) LinkError!void,
        upload: *const fn (*anyopaque, wire.TensorRange, []const u8) LinkError!void,
        download: *const fn (*anyopaque, wire.TensorRange, []u8) LinkError!void,
        run_graph: *const fn (*anyopaque, []const wire.Command) LinkError!void,
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

            fn callAlloc(ctx: *anyopaque, nbytes: u64, alignment: u32) LinkError!wire.TensorRange {
                return ptr(ctx).alloc(nbytes, alignment);
            }

            fn callFree(ctx: *anyopaque, range: wire.TensorRange) LinkError!void {
                return ptr(ctx).free(range);
            }

            fn callUpload(ctx: *anyopaque, range: wire.TensorRange, bytes: []const u8) LinkError!void {
                return ptr(ctx).upload(range, bytes);
            }

            fn callDownload(ctx: *anyopaque, range: wire.TensorRange, out: []u8) LinkError!void {
                return ptr(ctx).download(range, out);
            }

            fn callRunGraph(ctx: *anyopaque, commands: []const wire.Command) LinkError!void {
                return ptr(ctx).runGraph(commands);
            }

            const vtable = VTable{
                .hello = callHello,
                .alloc = callAlloc,
                .free = callFree,
                .upload = callUpload,
                .download = callDownload,
                .run_graph = callRunGraph,
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

    pub fn alloc(self: Self, nbytes: u64, alignment: u32) LinkError!wire.TensorRange {
        return self.vtable.alloc(self.ctx, nbytes, alignment);
    }

    pub fn free(self: Self, range: wire.TensorRange) LinkError!void {
        return self.vtable.free(self.ctx, range);
    }

    pub fn upload(self: Self, range: wire.TensorRange, bytes: []const u8) LinkError!void {
        return self.vtable.upload(self.ctx, range, bytes);
    }

    pub fn download(self: Self, range: wire.TensorRange, out: []u8) LinkError!void {
        return self.vtable.download(self.ctx, range, out);
    }

    pub fn runGraph(self: Self, commands: []const wire.Command) LinkError!void {
        return self.vtable.run_graph(self.ctx, commands);
    }
};

pub const FakeLink = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    runtime: *runtime_mod.Runtime,
    next_request_id: u64 = 1,

    pub fn init(allocator: std.mem.Allocator, runtime: *runtime_mod.Runtime) Self {
        return .{ .allocator = allocator, .runtime = runtime };
    }

    pub fn hello(self: *Self) LinkError!void {
        var meta: [32]u8 = undefined;
        const id = self.nextId();
        const meta_len = wire.encodeHello(&meta, id) catch return error.Protocol;
        var response = try self.call(meta[0..meta_len], "");
        defer response.deinit(self.allocator);
        try response.expectOk(id);
    }

    pub fn alloc(self: *Self, nbytes: u64, alignment: u32) LinkError!wire.TensorRange {
        var meta: [64]u8 = undefined;
        const id = self.nextId();
        const meta_len = wire.encodeAlloc(&meta, id, nbytes, alignment) catch return error.Protocol;
        var response = try self.call(meta[0..meta_len], "");
        defer response.deinit(self.allocator);
        try response.expectOk(id);
        return .{ .handle = response.meta.handle, .offset = 0, .nbytes = response.meta.nbytes };
    }

    pub fn free(self: *Self, range: wire.TensorRange) LinkError!void {
        var meta: [32]u8 = undefined;
        const id = self.nextId();
        const meta_len = wire.encodeFree(&meta, id, range.handle) catch return error.Protocol;
        var response = try self.call(meta[0..meta_len], "");
        defer response.deinit(self.allocator);
        try response.expectOk(id);
    }

    pub fn upload(self: *Self, range: wire.TensorRange, bytes: []const u8) LinkError!void {
        var meta: [64]u8 = undefined;
        const id = self.nextId();
        const meta_len = wire.encodeUpload(&meta, id, range) catch return error.Protocol;
        var response = try self.call(meta[0..meta_len], bytes);
        defer response.deinit(self.allocator);
        try response.expectOk(id);
    }

    pub fn download(self: *Self, range: wire.TensorRange, out: []u8) LinkError!void {
        var meta: [64]u8 = undefined;
        const id = self.nextId();
        const meta_len = wire.encodeDownload(&meta, id, range) catch return error.Protocol;
        var response = try self.call(meta[0..meta_len], "");
        defer response.deinit(self.allocator);
        try response.expectOk(id);
        if (response.payload.len != out.len) return error.Protocol;
        @memcpy(out, response.payload);
    }

    pub fn runGraph(self: *Self, commands: []const wire.Command) LinkError!void {
        const command_len = wire.commandBufferLen(commands) catch return error.Protocol;
        const command_bytes = try self.allocator.alloc(u8, command_len);
        defer self.allocator.free(command_bytes);
        _ = wire.encodeCommandBuffer(commands, command_bytes) catch return error.Protocol;

        var meta: [32]u8 = undefined;
        const id = self.nextId();
        const meta_len = wire.encodeRunGraph(&meta, id) catch return error.Protocol;
        var response = try self.call(meta[0..meta_len], command_bytes);
        defer response.deinit(self.allocator);
        try response.expectOk(id);
    }

    fn call(self: *Self, metadata: []const u8, payload: []const u8) LinkError!Response {
        const request_len = framing.encodedLen(metadata.len, payload.len) catch return error.Protocol;
        const request_frame = try self.allocator.alloc(u8, request_len);
        defer self.allocator.free(request_frame);
        _ = framing.encode(metadata, payload, request_frame) catch return error.Protocol;

        const response_frame = server.handleFrame(self.allocator, self.runtime, request_frame) catch return error.Protocol;
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

    pub fn alloc(self: *Self, nbytes: u64, alignment: u32) LinkError!wire.TensorRange {
        var meta: [64]u8 = undefined;
        const id = self.nextId();
        const meta_len = wire.encodeAlloc(&meta, id, nbytes, alignment) catch return error.Protocol;
        var response = try self.call(meta[0..meta_len], "");
        defer response.deinit(self.allocator);
        try response.expectOk(id);
        return .{ .handle = response.meta.handle, .offset = 0, .nbytes = response.meta.nbytes };
    }

    pub fn free(self: *Self, range: wire.TensorRange) LinkError!void {
        var meta: [32]u8 = undefined;
        const id = self.nextId();
        const meta_len = wire.encodeFree(&meta, id, range.handle) catch return error.Protocol;
        var response = try self.call(meta[0..meta_len], "");
        defer response.deinit(self.allocator);
        try response.expectOk(id);
    }

    pub fn upload(self: *Self, range: wire.TensorRange, bytes: []const u8) LinkError!void {
        var meta: [64]u8 = undefined;
        const id = self.nextId();
        const meta_len = wire.encodeUpload(&meta, id, range) catch return error.Protocol;
        var response = try self.call(meta[0..meta_len], bytes);
        defer response.deinit(self.allocator);
        try response.expectOk(id);
    }

    pub fn download(self: *Self, range: wire.TensorRange, out: []u8) LinkError!void {
        var meta: [64]u8 = undefined;
        const id = self.nextId();
        const meta_len = wire.encodeDownload(&meta, id, range) catch return error.Protocol;
        var response = try self.call(meta[0..meta_len], "");
        defer response.deinit(self.allocator);
        try response.expectOk(id);
        if (response.payload.len != out.len) return error.Protocol;
        @memcpy(out, response.payload);
    }

    pub fn runGraph(self: *Self, commands: []const wire.Command) LinkError!void {
        const command_len = wire.commandBufferLen(commands) catch return error.Protocol;
        const command_bytes = try self.allocator.alloc(u8, command_len);
        defer self.allocator.free(command_bytes);
        _ = wire.encodeCommandBuffer(commands, command_bytes) catch return error.Protocol;

        var meta: [32]u8 = undefined;
        const id = self.nextId();
        const meta_len = wire.encodeRunGraph(&meta, id) catch return error.Protocol;
        var response = try self.call(meta[0..meta_len], command_bytes);
        defer response.deinit(self.allocator);
        try response.expectOk(id);
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
