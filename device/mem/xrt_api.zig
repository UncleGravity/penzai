//! Minimal dlopen binding to the XRT native C API.

const std = @import("std");

pub const DeviceHandle = ?*anyopaque;
pub const BufferHandle = ?*anyopaque;

pub const sync_to_device: c_int = 0;
pub const sync_from_device: c_int = 1;
pub const flags_normal: u64 = 0;
pub const group_default: u32 = 0;

const DeviceOpenFn = *const fn (c_uint) callconv(.c) DeviceHandle;
const DeviceCloseFn = *const fn (DeviceHandle) callconv(.c) c_int;
const BOAllocFn = *const fn (DeviceHandle, usize, u64, u32) callconv(.c) BufferHandle;
const BOFreeFn = *const fn (BufferHandle) callconv(.c) c_int;
const BOMapFn = *const fn (BufferHandle) callconv(.c) ?*anyopaque;
const BOAddressFn = *const fn (BufferHandle) callconv(.c) u64;
const BOSyncFn = *const fn (BufferHandle, c_int, usize, usize) callconv(.c) c_int;

pub const Xrt = struct {
    const Self = @This();

    lib: std.DynLib,
    deviceOpen: DeviceOpenFn,
    deviceClose: DeviceCloseFn,
    boAlloc: BOAllocFn,
    boFree: BOFreeFn,
    boMap: BOMapFn,
    boAddress: BOAddressFn,
    boSync: BOSyncFn,

    pub fn open() !Self {
        var lib = try std.DynLib.open("libxrt_coreutil.so.2");
        errdefer lib.close();

        return .{
            .lib = lib,
            .deviceOpen = lib.lookup(DeviceOpenFn, "xrtDeviceOpen") orelse return error.XrtSymbolMissing,
            .deviceClose = lib.lookup(DeviceCloseFn, "xrtDeviceClose") orelse return error.XrtSymbolMissing,
            .boAlloc = lib.lookup(BOAllocFn, "xrtBOAlloc") orelse return error.XrtSymbolMissing,
            .boFree = lib.lookup(BOFreeFn, "xrtBOFree") orelse return error.XrtSymbolMissing,
            .boMap = lib.lookup(BOMapFn, "xrtBOMap") orelse return error.XrtSymbolMissing,
            .boAddress = lib.lookup(BOAddressFn, "xrtBOAddress") orelse return error.XrtSymbolMissing,
            .boSync = lib.lookup(BOSyncFn, "xrtBOSync") orelse return error.XrtSymbolMissing,
        };
    }

    pub fn close(self: *Self) void {
        self.lib.close();
        self.* = undefined;
    }
};
