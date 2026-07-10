//! Host↔device link — the erased `Client` (backend.zig's stable transport field),
//! one generic `Link(Transport)` that writes every wire op exactly once, and the
//! `FakeLink`/`TcpLink` factory namespaces. This file is transport-agnostic: it
//! encodes a request, hands the bytes to `transport.call`/`callInto`, and decodes
//! the response. The two transports live in their own files — `fake.zig` (the
//! in-process device server) and `tcp.zig` (the wire) — so only `fake.zig` pulls
//! in the device simulator. The §5 transport seam.
const std = @import("std");
const shared = @import("shared");
const fake = @import("fake.zig");
const tcp = @import("tcp.zig");

const framing = shared.framing;
const protocol_transport = shared.protocol_transport;
const wire = shared.wire;
const profiling = shared.profiling;

const max_transfer_payload: usize = @intCast(framing.max_payload_len);

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

/// Device-side time for the budget = request servicing (device_service_ns) + the
/// response encode/copy (device_encode_ns), both stamped on the wire ResponseMeta.
/// Folding them together here keeps the host's transport bucket pure wire time.
fn opTimingFrom(meta: wire.ResponseMeta) OpTiming {
    return .{ .device_service_ns = meta.device_service_ns +| meta.device_encode_ns };
}

/// `alloc` result: the granted range plus the op's service timing.
pub const AllocResult = struct {
    range: wire.TensorRange,
    timing: OpTiming = .{},
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

// ===========================  Client — the erased link  ===========================
// backend.zig holds this stable, type-erased handle; one `Link(Transport)` lives
// behind it. The vtable is the seam between the backend and any transport.

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

// ===========================  Link(Transport) — every op, written once  ===========================
// A `Transport` is any type exposing `call(allocator, request_frame) -> response_frame`,
// `callInto(allocator, request_frame, payload_out) -> FrameIntoResult` (zero-copy
// payload), and `io() -> ?std.Io`. `Link` owns the wire: encode → transport → decode
// → timing. FakeLink/TcpLink used to duplicate every op below; now they share it.

pub fn Link(comptime Transport: type) type {
    return struct {
        const Self = @This();

        allocator: std.mem.Allocator,
        transport: Transport,
        next_request_id: u64 = 1,

        /// Wrap an already-built transport.
        pub fn from(allocator: std.mem.Allocator, transport: Transport) Self {
            return .{ .allocator = allocator, .transport = transport };
        }

        // Transport-specific convenience constructors. Each compiles only for the
        // transport that provides the matching factory (FakeTransport.init /
        // TcpTransport.connect); the mismatched ones are never instantiated, so
        // their bodies are never analyzed. Callers spell FakeLink.* / TcpLink.connect.
        pub fn init(allocator: std.mem.Allocator, runtime: anytype) Self {
            return from(allocator, Transport.init(runtime, null));
        }

        pub fn initWithIo(allocator: std.mem.Allocator, runtime: anytype, io: std.Io) Self {
            return from(allocator, Transport.init(runtime, io));
        }

        pub fn connect(allocator: std.mem.Allocator, io: std.Io, spec: protocol_transport.TcpSpec) LinkError!Self {
            return from(allocator, try Transport.connect(io, spec));
        }

        pub fn deinit(self: *Self) void {
            if (@hasDecl(Transport, "deinit")) self.transport.deinit();
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
                .timing = opTimingFrom(response.meta),
            };
        }

        pub fn free(self: *Self, range: wire.TensorRange) LinkError!OpTiming {
            var meta: [32]u8 = undefined;
            const id = self.nextId();
            const meta_len = wire.encodeFree(&meta, id, range.handle) catch return error.Protocol;
            var response = try self.call(meta[0..meta_len], "");
            defer response.deinit(self.allocator);
            try response.expectOk(id);
            return opTimingFrom(response.meta);
        }

        /// Upload one logical tensor range, splitting it across wire frames when
        /// it is larger than the protocol payload limit.
        pub fn upload(self: *Self, range: wire.TensorRange, bytes: []const u8) LinkError!OpTiming {
            return self.uploadChunked(range, bytes, max_transfer_payload);
        }

        fn uploadChunked(
            self: *Self,
            range: wire.TensorRange,
            bytes: []const u8,
            chunk_limit: usize,
        ) LinkError!OpTiming {
            if (!validTransfer(range, bytes.len) or !validChunkLimit(chunk_limit)) return error.Protocol;
            if (bytes.len == 0) return self.uploadOne(range, bytes);

            var total: OpTiming = .{};
            var offset: usize = 0;
            while (offset < bytes.len) {
                const chunk_len = @min(chunk_limit, bytes.len - offset);
                const chunk_range = try transferChunkRange(range, offset, chunk_len);
                const timing = try self.uploadOne(chunk_range, bytes[offset..][0..chunk_len]);
                total.device_service_ns = total.device_service_ns +| timing.device_service_ns;
                offset += chunk_len;
            }
            return total;
        }

        fn uploadOne(self: *Self, range: wire.TensorRange, bytes: []const u8) LinkError!OpTiming {
            var meta: [64]u8 = undefined;
            const id = self.nextId();
            const meta_len = wire.encodeUpload(&meta, id, range) catch return error.Protocol;
            var response = try self.call(meta[0..meta_len], bytes);
            defer response.deinit(self.allocator);
            try response.expectOk(id);
            return opTimingFrom(response.meta);
        }

        pub fn fill(self: *Self, range: wire.TensorRange, value: u8) LinkError!OpTiming {
            var meta: [64]u8 = undefined;
            const id = self.nextId();
            const meta_len = wire.encodeFill(&meta, id, range, value) catch return error.Protocol;
            var response = try self.call(meta[0..meta_len], "");
            defer response.deinit(self.allocator);
            try response.expectOk(id);
            return opTimingFrom(response.meta);
        }

        /// Download one logical tensor range, splitting it across wire frames when
        /// it is larger than the protocol payload limit.
        pub fn download(self: *Self, range: wire.TensorRange, out: []u8) LinkError!OpTiming {
            return self.downloadChunked(range, out, max_transfer_payload);
        }

        fn downloadChunked(
            self: *Self,
            range: wire.TensorRange,
            out: []u8,
            chunk_limit: usize,
        ) LinkError!OpTiming {
            if (!validTransfer(range, out.len) or !validChunkLimit(chunk_limit)) return error.Protocol;
            if (out.len == 0) return self.downloadOne(range, out);

            var total: OpTiming = .{};
            var offset: usize = 0;
            while (offset < out.len) {
                const chunk_len = @min(chunk_limit, out.len - offset);
                const chunk_range = try transferChunkRange(range, offset, chunk_len);
                const timing = try self.downloadOne(chunk_range, out[offset..][0..chunk_len]);
                total.device_service_ns = total.device_service_ns +| timing.device_service_ns;
                offset += chunk_len;
            }
            return total;
        }

        /// Each download chunk routes through `transport.callInto` so the wire
        /// transport reads the payload straight into its final destination.
        fn downloadOne(self: *Self, range: wire.TensorRange, out: []u8) LinkError!OpTiming {
            var meta: [64]u8 = undefined;
            const id = self.nextId();
            const meta_len = wire.encodeDownload(&meta, id, range) catch return error.Protocol;
            const request_len = framing.encodedLen(meta_len, 0) catch return error.Protocol;
            const request_frame = try self.allocator.alloc(u8, request_len);
            defer self.allocator.free(request_frame);
            _ = framing.encode(meta[0..meta_len], "", request_frame) catch return error.Protocol;

            var response = try self.transport.callInto(self.allocator, request_frame, out);
            defer response.deinit(self.allocator);
            const response_meta = wire.decodeResponseMeta(response.metadata) catch return error.Protocol;
            if (response_meta.request_id != id) return error.Protocol;
            if (response_meta.status != .ok) return error.RemoteFailed;
            if (!response.payload_copied or response.payload_len != out.len) return error.Protocol;
            return opTimingFrom(response_meta);
        }

        pub fn runGraph(self: *Self, commands: []const wire.Command) LinkError!void {
            return self.runGraphPreload(&.{}, commands);
        }

        pub fn runGraphPreload(self: *Self, preload_bytes: []const u8, commands: []const wire.Command) LinkError!void {
            const payload = try buildRunGraphPayload(self.allocator, preload_bytes, commands);
            defer self.allocator.free(payload);

            var meta: [32]u8 = undefined;
            const id = self.nextId();
            const meta_len = wire.encodeRunGraph(&meta, id, .off) catch return error.Protocol;
            var response = try self.call(meta[0..meta_len], payload);
            defer response.deinit(self.allocator);
            try response.expectOk(id);
        }

        pub fn runGraphProfile(self: *Self, commands: []const wire.Command, tier: wire.ProfileTier) LinkError!ProfiledRunGraph {
            return self.runGraphProfilePreload(&.{}, commands, tier);
        }

        pub fn runGraphProfilePreload(
            self: *Self,
            preload_bytes: []const u8,
            commands: []const wire.Command,
            tier: wire.ProfileTier,
        ) LinkError!ProfiledRunGraph {
            const payload = try buildRunGraphPayload(self.allocator, preload_bytes, commands);
            defer self.allocator.free(payload);

            var meta: [32]u8 = undefined;
            const id = self.nextId();
            const meta_len = wire.encodeRunGraph(&meta, id, tier) catch return error.Protocol;
            const request_len = framing.encodedLen(meta_len, payload.len) catch return error.Protocol;
            const io = self.transport.ioHandle();
            const start_ns = profiling.nowNs(io);
            var response = try self.call(meta[0..meta_len], payload);
            const end_ns = profiling.nowNs(io);
            defer response.deinit(self.allocator);
            try response.expectOk(id);
            var report = profiling.decodeAlloc(self.allocator, response.payload) catch return error.Protocol;
            errdefer report.deinit(self.allocator);
            // Fold the device's response-encode time (framing + the profile payload
            // copy) into device_total so the host's transport bucket stays pure wire.
            report.summary.device_total_ns +|= response.meta.device_encode_ns;
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

        fn call(self: *Self, metadata: []const u8, payload: []const u8) LinkError!Response {
            const request_len = framing.encodedLen(metadata.len, payload.len) catch return error.Protocol;
            const request_frame = try self.allocator.alloc(u8, request_len);
            defer self.allocator.free(request_frame);
            _ = framing.encode(metadata, payload, request_frame) catch return error.Protocol;

            const response_frame = try self.transport.call(self.allocator, request_frame);
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
}

// ===========================  FakeLink / TcpLink — the two link types  ===========================
// Aliases for the two transports. Constructors live on `Link` above
// (`FakeLink.init`/`initWithIo`, `TcpLink.connect`), so each name is both the type
// callers spell (`*FakeLink`) and the namespace they construct through.

pub const FakeLink = Link(fake.FakeTransport);
pub const TcpLink = Link(tcp.TcpTransport);

// ===========================  Wire helpers + Response  ===========================

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

fn validTransfer(range: wire.TensorRange, len: usize) bool {
    const len_u64 = std.math.cast(u64, len) orelse return false;
    if (range.nbytes != len_u64) return false;
    _ = std.math.add(u64, range.offset, range.nbytes) catch return false;
    return true;
}

fn validChunkLimit(chunk_limit: usize) bool {
    return chunk_limit != 0 and chunk_limit <= max_transfer_payload;
}

fn transferChunkRange(range: wire.TensorRange, offset: usize, len: usize) LinkError!wire.TensorRange {
    const offset_u64 = std.math.cast(u64, offset) orelse return error.Protocol;
    const len_u64 = std.math.cast(u64, len) orelse return error.Protocol;
    return .{
        .handle = range.handle,
        .offset = std.math.add(u64, range.offset, offset_u64) catch return error.Protocol,
        .nbytes = len_u64,
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

test "chunked upload and download preserve ranges, bytes, and timing" {
    const TestTransport = struct {
        const timing: wire.ResponseMeta = .{
            .request_id = 0,
            .status = .ok,
            .device_service_ns = 3,
            .device_encode_ns = 4,
        };

        ranges: [10]wire.TensorRange = undefined,
        range_count: usize = 0,
        uploaded: [16]u8 = undefined,
        uploaded_len: usize = 0,

        pub fn ioHandle(_: *@This()) ?std.Io {
            return null;
        }

        pub fn call(
            self: *@This(),
            allocator: std.mem.Allocator,
            request_frame: []const u8,
        ) LinkError![]u8 {
            const frame = framing.decode(request_frame) catch return error.Protocol;
            const request = wire.decodeRequest(frame.metadata, frame.payload) catch return error.Protocol;
            const upload = switch (request) {
                .upload => |value| value,
                else => return error.Protocol,
            };
            try self.record(upload.range);
            if (self.uploaded.len - self.uploaded_len < upload.bytes.len) return error.Protocol;
            @memcpy(self.uploaded[self.uploaded_len..][0..upload.bytes.len], upload.bytes);
            self.uploaded_len += upload.bytes.len;
            return encodeTestResponse(allocator, upload.request_id);
        }

        pub fn callInto(
            self: *@This(),
            allocator: std.mem.Allocator,
            request_frame: []const u8,
            payload_out: []u8,
        ) LinkError!protocol_transport.FrameIntoResult {
            const frame = framing.decode(request_frame) catch return error.Protocol;
            const request = wire.decodeRequest(frame.metadata, frame.payload) catch return error.Protocol;
            const download = switch (request) {
                .download => |value| value,
                else => return error.Protocol,
            };
            try self.record(download.range);
            if (!validTransfer(download.range, payload_out.len)) return error.Protocol;
            for (payload_out, 0..) |*byte, i| {
                byte.* = @truncate(download.range.offset + @as(u64, @intCast(i)));
            }

            const metadata = try allocator.alloc(u8, wire.response_meta_len);
            errdefer allocator.free(metadata);
            var response_meta = timing;
            response_meta.request_id = download.request_id;
            _ = wire.encodeResponseMeta(metadata, response_meta) catch return error.Protocol;
            return .{
                .metadata = metadata,
                .payload_len = payload_out.len,
                .payload_copied = true,
            };
        }

        fn record(self: *@This(), range: wire.TensorRange) LinkError!void {
            if (self.range_count == self.ranges.len) return error.Protocol;
            self.ranges[self.range_count] = range;
            self.range_count += 1;
        }

        fn encodeTestResponse(allocator: std.mem.Allocator, request_id: u64) LinkError![]u8 {
            var response_meta = timing;
            response_meta.request_id = request_id;
            var metadata: [wire.response_meta_len]u8 = undefined;
            const metadata_len = wire.encodeResponseMeta(&metadata, response_meta) catch return error.Protocol;
            const frame_len = framing.encodedLen(metadata_len, 0) catch return error.Protocol;
            const frame = try allocator.alloc(u8, frame_len);
            errdefer allocator.free(frame);
            _ = framing.encode(metadata[0..metadata_len], "", frame) catch return error.Protocol;
            return frame;
        }
    };

    const transport: TestTransport = .{};
    var link = Link(TestTransport).from(std.testing.allocator, transport);

    const range: wire.TensorRange = .{ .handle = 9, .offset = 10, .nbytes = 7 };
    const upload_timing = try link.uploadChunked(range, "abcdefg", 3);
    try std.testing.expectEqual(@as(u64, 21), upload_timing.device_service_ns);
    try std.testing.expectEqualSlices(u8, "abcdefg", link.transport.uploaded[0..link.transport.uploaded_len]);
    try expectRange(.{ .handle = 9, .offset = 10, .nbytes = 3 }, link.transport.ranges[0]);
    try expectRange(.{ .handle = 9, .offset = 13, .nbytes = 3 }, link.transport.ranges[1]);
    try expectRange(.{ .handle = 9, .offset = 16, .nbytes = 1 }, link.transport.ranges[2]);

    var downloaded: [7]u8 = undefined;
    const download_timing = try link.downloadChunked(range, &downloaded, 3);
    try std.testing.expectEqual(@as(u64, 21), download_timing.device_service_ns);
    try std.testing.expectEqualSlices(u8, &.{ 10, 11, 12, 13, 14, 15, 16 }, &downloaded);
    try expectRange(.{ .handle = 9, .offset = 10, .nbytes = 3 }, link.transport.ranges[3]);
    try expectRange(.{ .handle = 9, .offset = 13, .nbytes = 3 }, link.transport.ranges[4]);
    try expectRange(.{ .handle = 9, .offset = 16, .nbytes = 1 }, link.transport.ranges[5]);

    const empty_range: wire.TensorRange = .{ .handle = 9, .offset = 17, .nbytes = 0 };
    const empty_upload_timing = try link.uploadChunked(empty_range, "", 3);
    try std.testing.expectEqual(@as(u64, 7), empty_upload_timing.device_service_ns);
    var empty_download: [0]u8 = .{};
    const empty_download_timing = try link.downloadChunked(empty_range, &empty_download, 3);
    try std.testing.expectEqual(@as(u64, 7), empty_download_timing.device_service_ns);
    try expectRange(empty_range, link.transport.ranges[6]);
    try expectRange(empty_range, link.transport.ranges[7]);

    const calls_before_mismatch = link.transport.range_count;
    const wrong_range: wire.TensorRange = .{ .handle = 9, .offset = 10, .nbytes = 6 };
    try std.testing.expectError(error.Protocol, link.uploadChunked(wrong_range, "abcdefg", 3));
    try std.testing.expectError(error.Protocol, link.downloadChunked(wrong_range, &downloaded, 3));
    try std.testing.expectError(error.Protocol, link.uploadChunked(range, "abcdefg", 0));
    const overflowing_range: wire.TensorRange = .{ .handle = 9, .offset = std.math.maxInt(u64), .nbytes = 1 };
    try std.testing.expectError(error.Protocol, link.uploadChunked(overflowing_range, "x", 3));
    try std.testing.expectEqual(calls_before_mismatch, link.transport.range_count);
}

fn expectRange(expected: wire.TensorRange, actual: wire.TensorRange) !void {
    try std.testing.expectEqual(expected.handle, actual.handle);
    try std.testing.expectEqual(expected.offset, actual.offset);
    try std.testing.expectEqual(expected.nbytes, actual.nbytes);
}
