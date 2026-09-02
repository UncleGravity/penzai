const std = @import("std");
const shared = @import("shared");
const regions_mod = @import("regions.zig");

const wire = shared.wire;

pub const HeapError = error{
    OutOfMemory,
    UnknownHandle,
    OutOfBounds,
    InvalidAlignment,
    BackendFailure,
};

/// In-memory heap used for protocol and allocator tests without a board.
/// Offset bookkeeping is delegated to `Regions`; this type only owns the bytes.
pub const Heap = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    data: []u8,
    regions: regions_mod.Regions,

    pub fn init(allocator: std.mem.Allocator, size: usize) !Self {
        const data = try allocator.alloc(u8, size);
        errdefer allocator.free(data);
        @memset(data, 0);
        return .{
            .allocator = allocator,
            .data = data,
            .regions = try regions_mod.Regions.init(allocator, size),
        };
    }

    pub fn deinit(self: *Self) void {
        self.regions.deinit();
        self.allocator.free(self.data);
        self.* = undefined;
    }

    pub fn allocate(self: *Self, nbytes: u64, alignment: u32) HeapError!wire.TensorRange {
        const a = try self.regions.allocate(nbytes, alignment);
        return .{ .handle = a.handle, .offset = 0, .nbytes = nbytes };
    }

    pub fn free(self: *Self, handle: u64) HeapError!void {
        return self.regions.free(handle);
    }

    pub fn read(self: *Self, range: wire.TensorRange) HeapError![]const u8 {
        return self.bytes(range);
    }

    pub fn write(self: *Self, range: wire.TensorRange, src: []const u8) HeapError!void {
        if (src.len != try checkedUsize(range.nbytes)) return error.OutOfBounds;
        @memcpy((try self.bytes(range))[0..src.len], src);
    }

    pub fn fill(self: *Self, range: wire.TensorRange, value: u8) HeapError!void {
        @memset(try self.bytes(range), value);
    }

    pub fn bytes(self: *Self, range: wire.TensorRange) HeapError![]u8 {
        const abs = try self.regions.absolute(range.handle, range.offset, range.nbytes);
        return self.data[@intCast(abs)..][0..try checkedUsize(range.nbytes)];
    }
};

fn checkedUsize(value: u64) HeapError!usize {
    if (value > std.math.maxInt(usize)) return error.OutOfBounds;
    return @intCast(value);
}

test "heap alloc write read" {
    var heap = try Heap.init(std.testing.allocator, 1024);
    defer heap.deinit();

    const range = try heap.allocate(16, 64);
    try heap.write(range, "abcdefghijklmnop");
    try std.testing.expectEqualSlices(u8, "abcdefghijklmnop", try heap.read(range));
}

test "heap rejects out of bounds range" {
    var heap = try Heap.init(std.testing.allocator, 64);
    defer heap.deinit();

    const range = try heap.allocate(8, 8);
    try std.testing.expectError(error.OutOfBounds, heap.read(.{
        .handle = range.handle,
        .offset = 4,
        .nbytes = 8,
    }));
}

test "heap reuses freed space" {
    var heap = try Heap.init(std.testing.allocator, 64);
    defer heap.deinit();

    const a = try heap.allocate(64, 1); // whole heap
    try heap.free(a.handle);
    const b = try heap.allocate(64, 1); // fits again only because free reclaims
    try heap.write(b, &[_]u8{0xAB} ** 64);
    try std.testing.expectEqual(@as(u8, 0xAB), (try heap.read(b))[63]);
}
