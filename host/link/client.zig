//! Typed host client for the resident inference daemon.
const std = @import("std");
const shared = @import("shared");
const tcp = @import("tcp.zig");

const capabilities_schema = shared.capabilities;
const framing = shared.framing;
const inference_rpc = shared.engine.rpc;
const protocol_transport = shared.protocol_transport;
const wire = shared.wire;

const max_transfer_payload: usize = @intCast(framing.max_payload_len);

pub const LinkError = error{
    OutOfMemory,
    Protocol,
    RemoteFailed,
    RemoteInvalidRequest,
    RemoteOutOfMemory,
    RemoteUnknownHandle,
    RemoteOutOfBounds,
    RemoteUnsupportedOp,
    RemoteBackendFailure,
    Transport,
};

pub const OpTiming = struct {
    device_rpc_ns: u64 = 0,
};

pub const AllocResult = struct {
    range: wire.TensorRange,
    timing: OpTiming = .{},
};

pub const InferenceReply = struct {
    timing: OpTiming = .{},
    result: ?inference_rpc.ExecuteResult = null,
};

/// Stable type-erased link held by the llama executor registry.
pub const Client = struct {
    ctx: *anyopaque,
    vtable: *const VTable,

    const VTable = struct {
        hello: *const fn (*anyopaque) LinkError!void,
        capabilities: *const fn (*anyopaque) LinkError!capabilities_schema.Report,
        alloc: *const fn (*anyopaque, u64, u32) LinkError!AllocResult,
        free: *const fn (*anyopaque, wire.TensorRange) LinkError!OpTiming,
        upload: *const fn (
            *anyopaque,
            wire.TensorRange,
            []const u8,
        ) LinkError!OpTiming,
        inference: *const fn (
            *anyopaque,
            inference_rpc.Payload,
        ) LinkError!InferenceReply,
    };

    pub fn init(link: anytype) Client {
        const Ptr = @TypeOf(link);
        const info = @typeInfo(Ptr);
        if (info != .pointer)
            @compileError("Client.init expects a pointer");
        const Child = info.pointer.child;

        const adapter = struct {
            fn ptr(ctx: *anyopaque) *Child {
                return @ptrCast(@alignCast(ctx));
            }

            fn callHello(ctx: *anyopaque) LinkError!void {
                return ptr(ctx).hello();
            }

            fn callCapabilities(
                ctx: *anyopaque,
            ) LinkError!capabilities_schema.Report {
                return ptr(ctx).capabilities();
            }

            fn callAlloc(
                ctx: *anyopaque,
                nbytes: u64,
                alignment: u32,
            ) LinkError!AllocResult {
                return ptr(ctx).alloc(nbytes, alignment);
            }

            fn callFree(
                ctx: *anyopaque,
                range: wire.TensorRange,
            ) LinkError!OpTiming {
                return ptr(ctx).free(range);
            }

            fn callUpload(
                ctx: *anyopaque,
                range: wire.TensorRange,
                bytes: []const u8,
            ) LinkError!OpTiming {
                return ptr(ctx).upload(range, bytes);
            }

            fn callInference(
                ctx: *anyopaque,
                payload: inference_rpc.Payload,
            ) LinkError!InferenceReply {
                return ptr(ctx).inference(payload);
            }

            const vtable: VTable = .{
                .hello = callHello,
                .capabilities = callCapabilities,
                .alloc = callAlloc,
                .free = callFree,
                .upload = callUpload,
                .inference = callInference,
            };
        };

        return .{ .ctx = @ptrCast(link), .vtable = &adapter.vtable };
    }

    pub fn hello(self: Client) LinkError!void {
        return self.vtable.hello(self.ctx);
    }

    pub fn capabilities(self: Client) LinkError!capabilities_schema.Report {
        return self.vtable.capabilities(self.ctx);
    }

    pub fn alloc(
        self: Client,
        nbytes: u64,
        alignment: u32,
    ) LinkError!AllocResult {
        return self.vtable.alloc(self.ctx, nbytes, alignment);
    }

    pub fn free(self: Client, range: wire.TensorRange) LinkError!OpTiming {
        return self.vtable.free(self.ctx, range);
    }

    pub fn upload(
        self: Client,
        range: wire.TensorRange,
        bytes: []const u8,
    ) LinkError!OpTiming {
        return self.vtable.upload(self.ctx, range, bytes);
    }

    pub fn inference(
        self: Client,
        payload: inference_rpc.Payload,
    ) LinkError!InferenceReply {
        return self.vtable.inference(self.ctx, payload);
    }
};

pub fn Link(comptime Transport: type) type {
    return struct {
        const Self = @This();

        allocator: std.mem.Allocator,
        transport: Transport,
        next_request_id: u64 = 1,

        pub fn from(
            allocator: std.mem.Allocator,
            transport: Transport,
        ) Self {
            return .{ .allocator = allocator, .transport = transport };
        }

        pub fn connect(
            allocator: std.mem.Allocator,
            io: std.Io,
            spec: protocol_transport.TcpSpec,
        ) LinkError!Self {
            return from(allocator, try Transport.connect(io, spec));
        }

        pub fn deinit(self: *Self) void {
            if (@hasDecl(Transport, "deinit")) self.transport.deinit();
            self.* = undefined;
        }

        pub fn hello(self: *Self) LinkError!void {
            var metadata: [16]u8 = undefined;
            const request_id = self.nextId();
            const metadata_len = wire.encodeHello(
                &metadata,
                request_id,
            ) catch return error.Protocol;
            var response = try self.call(metadata[0..metadata_len], "");
            defer response.deinit(self.allocator);
            try response.expectOk(request_id);
        }

        pub fn capabilities(
            self: *Self,
        ) LinkError!capabilities_schema.Report {
            var metadata: [16]u8 = undefined;
            const request_id = self.nextId();
            const metadata_len = wire.encodeCapabilities(
                &metadata,
                request_id,
            ) catch return error.Protocol;
            var response = try self.call(metadata[0..metadata_len], "");
            defer response.deinit(self.allocator);
            try response.expectOk(request_id);
            if (response.meta.nbytes != response.payload.len)
                return error.Protocol;
            return capabilities_schema.decode(response.payload) catch
                return error.Protocol;
        }

        pub fn alloc(
            self: *Self,
            nbytes: u64,
            alignment: u32,
        ) LinkError!AllocResult {
            var metadata: [32]u8 = undefined;
            const request_id = self.nextId();
            const metadata_len = wire.encodeAlloc(
                &metadata,
                request_id,
                nbytes,
                alignment,
            ) catch return error.Protocol;
            var response = try self.call(metadata[0..metadata_len], "");
            defer response.deinit(self.allocator);
            try response.expectOk(request_id);
            return .{
                .range = .{
                    .handle = response.meta.handle,
                    .offset = 0,
                    .nbytes = response.meta.nbytes,
                },
                .timing = opTiming(response.meta),
            };
        }

        pub fn free(
            self: *Self,
            range: wire.TensorRange,
        ) LinkError!OpTiming {
            var metadata: [24]u8 = undefined;
            const request_id = self.nextId();
            const metadata_len = wire.encodeFree(
                &metadata,
                request_id,
                range.handle,
            ) catch return error.Protocol;
            var response = try self.call(metadata[0..metadata_len], "");
            defer response.deinit(self.allocator);
            try response.expectOk(request_id);
            return opTiming(response.meta);
        }

        pub fn upload(
            self: *Self,
            range: wire.TensorRange,
            bytes: []const u8,
        ) LinkError!OpTiming {
            return self.uploadChunked(range, bytes, max_transfer_payload);
        }

        fn uploadChunked(
            self: *Self,
            range: wire.TensorRange,
            bytes: []const u8,
            chunk_limit: usize,
        ) LinkError!OpTiming {
            if (!validTransfer(range, bytes.len) or
                chunk_limit == 0 or
                chunk_limit > max_transfer_payload)
            {
                return error.Protocol;
            }
            if (bytes.len == 0) return self.uploadOne(range, bytes);

            var total: OpTiming = .{};
            var offset: usize = 0;
            while (offset < bytes.len) {
                const chunk_len = @min(chunk_limit, bytes.len - offset);
                const chunk = try transferChunkRange(
                    range,
                    offset,
                    chunk_len,
                );
                const timing = try self.uploadOne(
                    chunk,
                    bytes[offset..][0..chunk_len],
                );
                total.device_rpc_ns +|= timing.device_rpc_ns;
                offset += chunk_len;
            }
            return total;
        }

        fn uploadOne(
            self: *Self,
            range: wire.TensorRange,
            bytes: []const u8,
        ) LinkError!OpTiming {
            var metadata: [40]u8 = undefined;
            const request_id = self.nextId();
            const metadata_len = wire.encodeUpload(
                &metadata,
                request_id,
                range,
            ) catch return error.Protocol;
            var response = try self.call(metadata[0..metadata_len], bytes);
            defer response.deinit(self.allocator);
            try response.expectOk(request_id);
            return opTiming(response.meta);
        }

        pub fn inference(
            self: *Self,
            payload_in: inference_rpc.Payload,
        ) LinkError!InferenceReply {
            const request_id = self.nextId();
            var payload = payload_in;
            if (payload == .execute)
                payload.execute.request_id = request_id;

            var payload_bytes: [64]u8 = undefined;
            const payload_bytes_encoded = inference_rpc.encode(
                payload,
                &payload_bytes,
            ) catch return error.Protocol;
            var metadata: [24]u8 = undefined;
            const metadata_len = wire.encodeInference(
                &metadata,
                request_id,
                std.meta.activeTag(payload),
            ) catch return error.Protocol;
            var response = try self.call(
                metadata[0..metadata_len],
                payload_bytes_encoded,
            );
            defer response.deinit(self.allocator);
            try response.expectOk(request_id);

            const result = if (payload == .execute) blk: {
                const decoded = inference_rpc.decodeResult(response.payload) catch
                    return error.Protocol;
                if (decoded.metrics_level != payload.execute.metrics_level)
                    return error.Protocol;
                break :blk decoded;
            } else blk: {
                if (response.payload.len != 0) return error.Protocol;
                break :blk null;
            };
            return .{ .timing = opTiming(response.meta), .result = result };
        }

        fn call(
            self: *Self,
            metadata: []const u8,
            payload: []const u8,
        ) LinkError!Response {
            const request_len = framing.encodedLen(
                metadata.len,
                payload.len,
            ) catch return error.Protocol;
            const request = try self.allocator.alloc(u8, request_len);
            defer self.allocator.free(request);
            _ = framing.encode(metadata, payload, request) catch
                return error.Protocol;

            const response_bytes = try self.transport.call(
                self.allocator,
                request,
            );
            errdefer self.allocator.free(response_bytes);
            const frame = framing.decode(response_bytes) catch
                return error.Protocol;
            const metadata_out = wire.decodeResponseMeta(frame.metadata) catch
                return error.Protocol;
            return .{
                .frame = response_bytes,
                .meta = metadata_out,
                .payload = frame.payload,
            };
        }

        fn nextId(self: *Self) u64 {
            const request_id = self.next_request_id;
            self.next_request_id += 1;
            return request_id;
        }
    };
}

pub const TcpLink = Link(tcp.TcpTransport);

fn opTiming(meta: wire.ResponseMeta) OpTiming {
    return .{
        .device_rpc_ns = meta.device_service_ns +| meta.device_encode_ns,
    };
}

fn validTransfer(range: wire.TensorRange, len: usize) bool {
    const nbytes = std.math.cast(u64, len) orelse return false;
    if (range.nbytes != nbytes) return false;
    _ = std.math.add(u64, range.offset, range.nbytes) catch return false;
    return true;
}

fn transferChunkRange(
    range: wire.TensorRange,
    offset: usize,
    len: usize,
) LinkError!wire.TensorRange {
    const offset_u64 = std.math.cast(u64, offset) orelse
        return error.Protocol;
    const len_u64 = std.math.cast(u64, len) orelse return error.Protocol;
    return .{
        .handle = range.handle,
        .offset = std.math.add(u64, range.offset, offset_u64) catch
            return error.Protocol,
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
        return expectResponseOk(self.meta, request_id);
    }
};

fn expectResponseOk(meta: wire.ResponseMeta, request_id: u64) LinkError!void {
    if (meta.request_id != request_id) return error.Protocol;
    if (meta.status == .ok) return;
    return switch (meta.error_code) {
        .none => error.RemoteFailed,
        .invalid_request => error.RemoteInvalidRequest,
        .out_of_memory => error.RemoteOutOfMemory,
        .unknown_handle => error.RemoteUnknownHandle,
        .out_of_bounds => error.RemoteOutOfBounds,
        .unsupported_op => error.RemoteUnsupportedOp,
        .backend_failure => error.RemoteBackendFailure,
    };
}

test "remote response errors retain their wire cause" {
    const failed: wire.ResponseMeta = .{
        .request_id = 7,
        .status = .failed,
        .error_code = .backend_failure,
    };
    try std.testing.expectError(
        error.RemoteBackendFailure,
        expectResponseOk(failed, 7),
    );
    try std.testing.expectError(error.Protocol, expectResponseOk(failed, 8));
}

test "chunked upload preserves ranges, bytes, and timing" {
    const TestTransport = struct {
        ranges: [4]wire.TensorRange = undefined,
        range_count: usize = 0,
        uploaded: [8]u8 = undefined,
        uploaded_len: usize = 0,

        pub fn call(
            self: *@This(),
            allocator: std.mem.Allocator,
            request_bytes: []const u8,
        ) LinkError![]u8 {
            const frame = framing.decode(request_bytes) catch
                return error.Protocol;
            const request = wire.decodeRequest(
                frame.metadata,
                frame.payload,
            ) catch return error.Protocol;
            const upload = switch (request) {
                .upload => |value| value,
                else => return error.Protocol,
            };
            self.ranges[self.range_count] = upload.range;
            self.range_count += 1;
            @memcpy(
                self.uploaded[self.uploaded_len..][0..upload.bytes.len],
                upload.bytes,
            );
            self.uploaded_len += upload.bytes.len;

            var metadata: [wire.response_meta_len]u8 = undefined;
            const metadata_len = wire.encodeResponseMeta(&metadata, .{
                .request_id = upload.request_id,
                .status = .ok,
                .device_service_ns = 3,
                .device_encode_ns = 4,
            }) catch return error.Protocol;
            const response_len = framing.encodedLen(
                metadata_len,
                0,
            ) catch return error.Protocol;
            const response = try allocator.alloc(u8, response_len);
            errdefer allocator.free(response);
            _ = framing.encode(
                metadata[0..metadata_len],
                "",
                response,
            ) catch return error.Protocol;
            return response;
        }
    };

    var link = Link(TestTransport).from(std.testing.allocator, .{});
    const range: wire.TensorRange = .{
        .handle = 9,
        .offset = 10,
        .nbytes = 7,
    };
    const timing = try link.uploadChunked(range, "abcdefg", 3);
    try std.testing.expectEqual(@as(u64, 21), timing.device_rpc_ns);
    try std.testing.expectEqualSlices(
        u8,
        "abcdefg",
        link.transport.uploaded[0..link.transport.uploaded_len],
    );
    try std.testing.expectEqual(
        wire.TensorRange{ .handle = 9, .offset = 10, .nbytes = 3 },
        link.transport.ranges[0],
    );
    try std.testing.expectEqual(
        wire.TensorRange{ .handle = 9, .offset = 13, .nbytes = 3 },
        link.transport.ranges[1],
    );
    try std.testing.expectEqual(
        wire.TensorRange{ .handle = 9, .offset = 16, .nbytes = 1 },
        link.transport.ranges[2],
    );

    try std.testing.expectError(
        error.Protocol,
        link.uploadChunked(
            .{ .handle = 9, .offset = 10, .nbytes = 6 },
            "abcdefg",
            3,
        ),
    );
}
