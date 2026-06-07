const std = @import("std");
const wire = @import("wire");
const xrt = @import("xrt");

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
    x: xrt.Xrt,
    dev: xrt.DeviceHandle,
    bo: xrt.BufferHandle,
    data: []u8,
    phys_base: u64,
    records: std.ArrayList(Record),
    cursor: u64,
    next_handle: u64,

    pub fn init(allocator: std.mem.Allocator, size: usize) !Self {
        var x = try xrt.Xrt.open();
        errdefer x.close();

        const dev = x.deviceOpen(0);
        if (dev == null) return error.DeviceOpen;
        errdefer _ = x.deviceClose(dev);

        const bo = x.boAlloc(dev, size, xrt.flags_normal, xrt.group_default);
        if (bo == null) return error.OutOfMemory;
        errdefer _ = x.boFree(bo);

        const mapped = x.boMap(bo) orelse return error.BOMapFailed;
        const data = @as([*]u8, @ptrCast(mapped))[0..size];
        @memset(data, 0);
        if (x.boSync(bo, xrt.sync_to_device, size, 0) != 0) return error.BOSyncFailed;

        return .{
            .allocator = allocator,
            .x = x,
            .dev = dev,
            .bo = bo,
            .data = data,
            .phys_base = x.boAddress(bo),
            .records = .empty,
            .cursor = 0,
            .next_handle = 1,
        };
    }

    pub fn deinit(self: *Self) void {
        self.records.deinit(self.allocator);
        _ = self.x.boFree(self.bo);
        _ = self.x.deviceClose(self.dev);
        self.x.close();
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
        return self.data[abs..][0..try checkedUsize(range.nbytes)];
    }

    pub fn write(self: *Self, range: wire.TensorRange, src: []const u8) HeapError!void {
        if (src.len != try checkedUsize(range.nbytes)) return error.OutOfBounds;
        const abs = try self.absolute(range);
        @memcpy(self.data[abs..][0..src.len], src);
        try self.syncToDevice(range);
    }

    pub fn bytes(self: *Self, range: wire.TensorRange) HeapError![]u8 {
        const abs = try self.absolute(range);
        return self.data[abs..][0..try checkedUsize(range.nbytes)];
    }

    pub fn deviceAddress(self: *Self, range: wire.TensorRange) HeapError!u64 {
        const abs = try self.absolute(range);
        return self.phys_base + @as(u64, @intCast(abs));
    }

    pub fn syncToDevice(self: *Self, range: wire.TensorRange) HeapError!void {
        const abs = try self.absolute(range);
        const len = try checkedUsize(range.nbytes);
        if (self.x.boSync(self.bo, xrt.sync_to_device, len, abs) != 0) return error.BackendFailure;
    }

    pub fn syncFromDevice(self: *Self, range: wire.TensorRange) HeapError!void {
        const abs = try self.absolute(range);
        const len = try checkedUsize(range.nbytes);
        if (self.x.boSync(self.bo, xrt.sync_from_device, len, abs) != 0) return error.BackendFailure;
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
