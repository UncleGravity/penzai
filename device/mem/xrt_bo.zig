const std = @import("std");
const shared = @import("shared");
const xrt = @import("../xrt.zig");
const regions_mod = @import("regions.zig");

const wire = shared.wire;

pub const HeapError = error{
    OutOfMemory,
    UnknownHandle,
    OutOfBounds,
    InvalidAlignment,
    BackendFailure,
};

pub const InitError = error{
    XrtOpenFailed,
    XrtSymbolMissing,
    XrtDeviceOpenFailed,
    XrtBOAllocFailed,
    XrtBOMapFailed,
    XrtBOSyncFailed,
};

/// Board heap: a single contiguous XRT buffer object mapped into the daemon and
/// physically addressable by the PL.  Offset bookkeeping is delegated to
/// `Regions`; this type owns the BO, the CPU mapping, and the cache syncs.
pub const Heap = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    x: xrt.Xrt,
    dev: xrt.DeviceHandle,
    bo: xrt.BufferHandle,
    data: []u8,
    phys_base: u64,
    regions: regions_mod.Regions,

    pub fn init(allocator: std.mem.Allocator, size: usize) InitError!Self {
        var x = xrt.Xrt.open() catch |err| switch (err) {
            error.XrtSymbolMissing => return error.XrtSymbolMissing,
            else => return error.XrtOpenFailed,
        };
        errdefer x.close();

        const dev = x.deviceOpen(0);
        if (dev == null) return error.XrtDeviceOpenFailed;
        errdefer _ = x.deviceClose(dev);

        const bo = x.boAlloc(dev, size, xrt.flags_normal, xrt.group_default);
        if (bo == null) return error.XrtBOAllocFailed;
        errdefer _ = x.boFree(bo);

        const mapped = x.boMap(bo) orelse return error.XrtBOMapFailed;
        const data = @as([*]u8, @ptrCast(mapped))[0..size];
        @memset(data, 0);
        if (x.boSync(bo, xrt.sync_to_device, size, 0) != 0) return error.XrtBOSyncFailed;

        const regions = regions_mod.Regions.init(allocator, size) catch return error.XrtBOAllocFailed;

        return .{
            .allocator = allocator,
            .x = x,
            .dev = dev,
            .bo = bo,
            .data = data,
            .phys_base = x.boAddress(bo),
            .regions = regions,
        };
    }

    pub fn deinit(self: *Self) void {
        self.regions.deinit();
        _ = self.x.boFree(self.bo);
        _ = self.x.deviceClose(self.dev);
        self.x.close();
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
        try self.syncToDevice(range);
    }

    pub fn fill(self: *Self, range: wire.TensorRange, value: u8) HeapError!void {
        @memset(try self.bytes(range), value);
        try self.syncToDevice(range);
    }

    pub fn bytes(self: *Self, range: wire.TensorRange) HeapError![]u8 {
        const abs = try self.regions.absolute(range.handle, range.offset, range.nbytes);
        return self.data[@intCast(abs)..][0..try checkedUsize(range.nbytes)];
    }

    pub fn deviceAddress(self: *Self, range: wire.TensorRange) HeapError!u64 {
        const abs = try self.regions.absolute(range.handle, range.offset, range.nbytes);
        return self.phys_base + abs;
    }

    pub fn syncToDevice(self: *Self, range: wire.TensorRange) HeapError!void {
        const abs = try self.regions.absolute(range.handle, range.offset, range.nbytes);
        const len = try checkedUsize(range.nbytes);
        if (self.x.boSync(self.bo, xrt.sync_to_device, len, @intCast(abs)) != 0) return error.BackendFailure;
    }

    pub fn syncFromDevice(self: *Self, range: wire.TensorRange) HeapError!void {
        const abs = try self.regions.absolute(range.handle, range.offset, range.nbytes);
        const len = try checkedUsize(range.nbytes);
        if (self.x.boSync(self.bo, xrt.sync_from_device, len, @intCast(abs)) != 0) return error.BackendFailure;
    }
};

fn checkedUsize(value: u64) HeapError!usize {
    if (value > std.math.maxInt(usize)) return error.OutOfBounds;
    return @intCast(value);
}
