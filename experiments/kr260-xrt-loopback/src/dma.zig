//! AXI DMA direct-register driver for the KR260 loopback fixture.

const std = @import("std");
const config = @import("config.zig");

const O_RDWR: c_int = 0x2;
const O_SYNC: c_int = 0o4010000;
const PROT_READ: c_int = 0x1;
const PROT_WRITE: c_int = 0x2;
const MAP_SHARED: c_int = 0x1;
const MAP_FAILED = std.math.maxInt(usize);

extern "c" fn open(path: [*:0]const u8, flags: c_int, mode: c_uint) c_int;
extern "c" fn close(fd: c_int) c_int;
extern "c" fn mmap(addr: ?*anyopaque, len: usize, prot: c_int, flags: c_int, fd: c_int, offset: i64) ?*anyopaque;
extern "c" fn munmap(addr: ?*anyopaque, len: usize) c_int;

const MM2S_DMACR = 0x00 / 4;
const MM2S_DMASR = 0x04 / 4;
const MM2S_SA = 0x18 / 4;
const MM2S_SA_MSB = 0x1C / 4;
const MM2S_LENGTH = 0x28 / 4;
const S2MM_DMACR = 0x30 / 4;
const S2MM_DMASR = 0x34 / 4;
const S2MM_DA = 0x48 / 4;
const S2MM_DA_MSB = 0x4C / 4;
const S2MM_LENGTH = 0x58 / 4;

const RS: u32 = 1 << 0;
const IDLE: u32 = 1 << 1;
const RESET: u32 = 1 << 2;
const ERR_MASK: u32 = 0x70;

pub const Dma = struct {
    fd: c_int,
    map: *anyopaque,
    regs: [*]volatile u32,

    pub fn openDevice() !Dma {
        const fd = open("/dev/mem", O_RDWR | O_SYNC, 0);
        if (fd < 0) return error.OpenDevMem;
        errdefer _ = close(fd);

        const maybe_map = mmap(null, config.dma_span, PROT_READ | PROT_WRITE, MAP_SHARED, fd, config.dma_base);
        if (maybe_map == null or @intFromPtr(maybe_map.?) == MAP_FAILED) return error.MapDma;

        const map = maybe_map.?;
        return .{
            .fd = fd,
            .map = map,
            .regs = @ptrCast(@alignCast(map)),
        };
    }

    pub fn closeDevice(self: *Dma) void {
        _ = munmap(self.map, config.dma_span);
        _ = close(self.fd);
    }

    pub fn loopback(self: *Dma, src_phys: u64, dst_phys: u64, len: usize) !void {
        const regs = self.regs;

        std.debug.print("dma: mmio map OK\n", .{});
        self.dumpStatus("before reset");

        regs[MM2S_DMACR] = RESET;
        regs[S2MM_DMACR] = RESET;
        try self.waitResetClear(MM2S_DMACR, "MM2S");
        try self.waitResetClear(S2MM_DMACR, "S2MM");
        self.dumpStatus("after reset");

        regs[MM2S_DMACR] = RS;
        regs[S2MM_DMACR] = RS;

        regs[S2MM_DA] = @truncate(dst_phys & 0xffff_ffff);
        regs[S2MM_DA_MSB] = @truncate(dst_phys >> 32);
        regs[S2MM_LENGTH] = @intCast(len);

        regs[MM2S_SA] = @truncate(src_phys & 0xffff_ffff);
        regs[MM2S_SA_MSB] = @truncate(src_phys >> 32);
        regs[MM2S_LENGTH] = @intCast(len);

        self.dumpStatus("started");
        try self.waitIdle(MM2S_DMASR, "MM2S");
        try self.waitIdle(S2MM_DMASR, "S2MM");
    }

    fn dumpStatus(self: *Dma, label: []const u8) void {
        std.debug.print(
            "dma: {s} status MM2S=0x{x:0>8} S2MM=0x{x:0>8}\n",
            .{ label, self.regs[MM2S_DMASR], self.regs[S2MM_DMASR] },
        );
    }

    fn waitResetClear(self: *Dma, cr: usize, name: []const u8) !void {
        var i: usize = 0;
        while (i < config.wait_limit) : (i += 1) {
            const control = self.regs[cr];
            if (control & RESET == 0) return;
        }

        std.debug.print("dma: {s} reset TIMEOUT control=0x{x:0>8}\n", .{ name, self.regs[cr] });
        return error.DmaResetTimeout;
    }

    fn waitIdle(self: *Dma, sr: usize, name: []const u8) !void {
        var i: usize = 0;
        while (i < config.wait_limit) : (i += 1) {
            const status = self.regs[sr];
            if (status & ERR_MASK != 0) {
                std.debug.print("dma: {s} ERROR status=0x{x:0>8}\n", .{ name, status });
                return error.DmaError;
            }
            if (status & IDLE != 0) {
                std.debug.print("dma: {s} idle PASS status=0x{x:0>8}\n", .{ name, status });
                return;
            }
        }

        std.debug.print("dma: {s} TIMEOUT status=0x{x:0>8}\n", .{ name, self.regs[sr] });
        return error.DmaTimeout;
    }
};
