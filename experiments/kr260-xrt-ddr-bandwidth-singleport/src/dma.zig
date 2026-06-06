//! AXI DMA direct-register driver for the KR260 single-port bandwidth fixture.

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
const HALTED: u32 = 1 << 0;
const IDLE: u32 = 1 << 1;
const RESET: u32 = 1 << 2;
const DMA_INT_ERR: u32 = 1 << 4;
const DMA_SLV_ERR: u32 = 1 << 5;
const DMA_DEC_ERR: u32 = 1 << 6;
const ERR_MASK: u32 = 0x70;

pub const Dma = struct {
    fd: c_int,
    map: *anyopaque,
    regs: [*]volatile u32,

    pub fn openDevice() !Dma {
        const fd = open("/dev/mem", O_RDWR | O_SYNC, 0);
        if (fd < 0) return error.OpenDevMem;
        errdefer _ = close(fd);

        const maybe_map = mmap(null, config.mmio_span, PROT_READ | PROT_WRITE, MAP_SHARED, fd, config.dma_base);
        if (maybe_map == null or @intFromPtr(maybe_map.?) == MAP_FAILED) return error.MapDma;

        const map = maybe_map.?;
        return .{
            .fd = fd,
            .map = map,
            .regs = @ptrCast(@alignCast(map)),
        };
    }

    pub fn closeDevice(self: *Dma) void {
        _ = munmap(self.map, config.mmio_span);
        _ = close(self.fd);
    }

    pub fn reset(self: *Dma) !void {
        self.regs[MM2S_DMACR] = RESET;
        self.regs[S2MM_DMACR] = RESET;
        try self.waitResetClear(MM2S_DMACR, "MM2S");
        try self.waitResetClear(S2MM_DMACR, "S2MM");
    }

    pub fn startWriteToDdr(self: *Dma, dst_phys: u64, len: usize) !void {
        if (len == 0 or len > std.math.maxInt(u32)) return error.InvalidTransfer;
        self.regs[S2MM_DMACR] = RS;
        self.regs[S2MM_DA] = @truncate(dst_phys & 0xffff_ffff);
        self.regs[S2MM_DA_MSB] = @truncate(dst_phys >> 32);
        self.regs[S2MM_LENGTH] = @intCast(len);
    }

    pub fn startReadFromDdr(self: *Dma, src_phys: u64, len: usize) !void {
        if (len == 0 or len > std.math.maxInt(u32)) return error.InvalidTransfer;
        self.regs[MM2S_DMACR] = RS;
        self.regs[MM2S_SA] = @truncate(src_phys & 0xffff_ffff);
        self.regs[MM2S_SA_MSB] = @truncate(src_phys >> 32);
        self.regs[MM2S_LENGTH] = @intCast(len);
    }

    pub fn waitWriteDone(self: *Dma) !void {
        try self.waitIdle(S2MM_DMASR, "S2MM");
    }

    pub fn waitReadDone(self: *Dma) !void {
        try self.waitIdle(MM2S_DMASR, "MM2S");
    }

    pub fn dumpStatus(self: *Dma, label: []const u8) void {
        std.debug.print(
            "case=dma_status label={s} mm2s=0x{x:0>8} s2mm=0x{x:0>8}\n",
            .{ label, self.regs[MM2S_DMASR], self.regs[S2MM_DMASR] },
        );
    }

    fn waitResetClear(self: *Dma, cr: usize, name: []const u8) !void {
        var i: usize = 0;
        while (i < config.wait_limit) : (i += 1) {
            const control = self.regs[cr];
            if (control & RESET == 0) return;
        }

        std.debug.print("case=dma_reset ok=0 channel={s} control=0x{x:0>8}\n", .{ name, self.regs[cr] });
        return error.DmaResetTimeout;
    }

    fn waitIdle(self: *Dma, sr: usize, name: []const u8) !void {
        var i: usize = 0;
        while (i < config.wait_limit) : (i += 1) {
            const status = self.regs[sr];
            if (status & ERR_MASK != 0) {
                std.debug.print(
                    "case=dma_wait ok=0 channel={s} status=0x{x:0>8} halted={d} idle={d} dma_int_err={d} dma_slv_err={d} dma_dec_err={d}\n",
                    .{
                        name,
                        status,
                        bit(status, HALTED),
                        bit(status, IDLE),
                        bit(status, DMA_INT_ERR),
                        bit(status, DMA_SLV_ERR),
                        bit(status, DMA_DEC_ERR),
                    },
                );
                return error.DmaError;
            }
            if (status & IDLE != 0) return;
        }

        std.debug.print("case=dma_wait ok=0 channel={s} timeout_status=0x{x:0>8}\n", .{ name, self.regs[sr] });
        return error.DmaTimeout;
    }
};

fn bit(value: u32, mask: u32) u1 {
    return if (value & mask != 0) 1 else 0;
}
