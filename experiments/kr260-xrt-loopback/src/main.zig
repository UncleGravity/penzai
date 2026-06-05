//! KR260 XRT + AXI DMA loopback verifier.

const std = @import("std");
const config = @import("config.zig");
const dma = @import("dma.zig");
const xrt = @import("xrt.zig");

pub fn main() !void {
    var x = xrt.Xrt.open() catch |err| {
        std.debug.print("xrt: failed to load libxrt_coreutil.so.2: {s}\n", .{@errorName(err)});
        return err;
    };
    defer x.close();

    const dev = x.deviceOpen(0);
    if (dev == null) {
        std.debug.print("xrt: device open FAIL (zocl/app loaded? render group?)\n", .{});
        return error.DeviceOpen;
    }
    defer _ = x.deviceClose(dev);
    std.debug.print("xrt: device open OK\n", .{});

    try verifyBo(&x, dev);
    try verifyLoopback(&x, dev);

    std.debug.print("ALL PASS\n", .{});
}

fn verifyBo(x: *xrt.Xrt, dev: xrt.DeviceHandle) !void {
    const bo = x.boAlloc(dev, config.transfer_size, xrt.flags_normal, xrt.group_default) orelse {
        std.debug.print("bo: alloc FAIL\n", .{});
        return error.BoAlloc;
    };
    defer _ = x.boFree(bo);

    const phys = x.boAddress(bo);
    const mem: [*]u8 = @ptrCast(x.boMap(bo) orelse {
        std.debug.print("bo: map FAIL\n", .{});
        return error.BoMap;
    });

    for (0..config.transfer_size) |i| mem[i] = pattern(i);
    if (x.boSync(bo, xrt.sync_to_device, config.transfer_size, 0) != 0) return error.BoSync;
    if (x.boSync(bo, xrt.sync_from_device, config.transfer_size, 0) != 0) return error.BoSync;

    for (0..config.transfer_size) |i| {
        if (mem[i] != pattern(i)) {
            std.debug.print("bo: mismatch at {d}: got=0x{x} expected=0x{x}\n", .{ i, mem[i], pattern(i) });
            return error.BoData;
        }
    }

    std.debug.print("bo: alloc/map/address/sync PASS phys=0x{x} n={d}\n", .{ phys, config.transfer_size });
}

fn verifyLoopback(x: *xrt.Xrt, dev: xrt.DeviceHandle) !void {
    const src = x.boAlloc(dev, config.transfer_size, xrt.flags_normal, xrt.group_default) orelse return error.BoAlloc;
    defer _ = x.boFree(src);
    const dst = x.boAlloc(dev, config.transfer_size, xrt.flags_normal, xrt.group_default) orelse return error.BoAlloc;
    defer _ = x.boFree(dst);

    const src_phys = x.boAddress(src);
    const dst_phys = x.boAddress(dst);
    const src_mem: [*]u8 = @ptrCast(x.boMap(src) orelse return error.BoMap);
    const dst_mem: [*]u8 = @ptrCast(x.boMap(dst) orelse return error.BoMap);

    for (0..config.transfer_size) |i| {
        src_mem[i] = pattern(i);
        dst_mem[i] = 0;
    }

    if (x.boSync(src, xrt.sync_to_device, config.transfer_size, 0) != 0) return error.BoSync;
    if (x.boSync(dst, xrt.sync_to_device, config.transfer_size, 0) != 0) return error.BoSync;

    std.debug.print(
        "dma: src=0x{x} dst=0x{x} n={d} base=0x{x}\n",
        .{ src_phys, dst_phys, config.transfer_size, config.dma_base },
    );

    var d = dma.Dma.openDevice() catch |err| {
        std.debug.print("dma: /dev/mem map FAIL (run as root?): {s}\n", .{@errorName(err)});
        return err;
    };
    defer d.closeDevice();

    try d.loopback(src_phys, dst_phys, config.transfer_size);
    if (x.boSync(dst, xrt.sync_from_device, config.transfer_size, 0) != 0) return error.BoSync;

    for (0..config.transfer_size) |i| {
        if (src_mem[i] != dst_mem[i]) {
            std.debug.print("verify: mismatch at {d}: src=0x{x} dst=0x{x}\n", .{ i, src_mem[i], dst_mem[i] });
            return error.Mismatch;
        }
    }

    std.debug.print("verify: src == dst PASS\n", .{});
}

inline fn pattern(i: usize) u8 {
    return @truncate(i *% 7 +% 1);
}
