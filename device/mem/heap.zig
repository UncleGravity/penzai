const std = @import("std");
const wire = @import("wire");

pub const HeapError = error{
    OutOfMemory,
    UnknownHandle,
    OutOfBounds,
    InvalidAlignment,
    BackendFailure,
};

const Record = struct {
    handle: u64,
    offset: u64,
    nbytes: u64,
    alive: bool,
};

pub const Heap = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    data: []u8,
    records: std.ArrayList(Record),
    cursor: u64,
    next_handle: u64,

    pub fn init(allocator: std.mem.Allocator, size: usize) !Self {
        const data = try allocator.alloc(u8, size);
        @memset(data, 0);
        return .{
            .allocator = allocator,
            .data = data,
            .records = .empty,
            .cursor = 0,
            .next_handle = 1,
        };
    }

    pub fn deinit(self: *Self) void {
        self.records.deinit(self.allocator);
        self.allocator.free(self.data);
        self.* = undefined;
    }

    pub fn allocate(self: *Self, nbytes: u64, alignment: u32) HeapError!wire.TensorRange {
        if (alignment == 0 or !std.math.isPowerOfTwo(alignment)) return error.InvalidAlignment;
        const aligned = alignForward(self.cursor, alignment);
        if (aligned > self.data.len or nbytes > self.data.len - aligned) return error.OutOfMemory;

        const handle = self.next_handle;
        self.next_handle += 1;
        self.records.append(self.allocator, .{
            .handle = handle,
            .offset = aligned,
            .nbytes = nbytes,
            .alive = true,
        }) catch return error.OutOfMemory;
        self.cursor = aligned + nbytes;
        return .{ .handle = handle, .offset = 0, .nbytes = nbytes };
    }

    pub fn free(self: *Self, handle: u64) HeapError!void {
        const rec = try self.findRecord(handle);
        rec.alive = false;
    }

    pub fn read(self: *Self, range: wire.TensorRange) HeapError![]const u8 {
        const abs = try self.absolute(range);
        return self.data[abs..][0..@intCast(range.nbytes)];
    }

    pub fn write(self: *Self, range: wire.TensorRange, src: []const u8) HeapError!void {
        if (src.len != try checkedUsize(range.nbytes)) return error.OutOfBounds;
        const abs = try self.absolute(range);
        @memcpy(self.data[abs..][0..src.len], src);
    }

    pub fn fill(self: *Self, range: wire.TensorRange, value: u8) HeapError!void {
        const abs = try self.absolute(range);
        @memset(self.data[abs..][0..try checkedUsize(range.nbytes)], value);
    }

    pub fn bytes(self: *Self, range: wire.TensorRange) HeapError![]u8 {
        const abs = try self.absolute(range);
        return self.data[abs..][0..@intCast(range.nbytes)];
    }

    fn absolute(self: *Self, range: wire.TensorRange) HeapError!usize {
        const rec = try self.findRecord(range.handle);
        if (range.offset > rec.nbytes or range.nbytes > rec.nbytes - range.offset) return error.OutOfBounds;
        return @intCast(rec.offset + range.offset);
    }

    fn findRecord(self: *Self, handle: u64) HeapError!*Record {
        for (self.records.items) |*rec| {
            if (rec.handle == handle and rec.alive) return rec;
        }
        return error.UnknownHandle;
    }
};

fn checkedUsize(value: u64) HeapError!usize {
    if (value > std.math.maxInt(usize)) return error.OutOfBounds;
    return @intCast(value);
}

fn alignForward(value: u64, alignment: u32) u64 {
    const mask = @as(u64, alignment) - 1;
    return (value + mask) & ~mask;
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
