//! Op-agnostic AXI DMA simple-mode driver (Xilinx standard), built on the PL
//! register window. One Dma per channel; a weights DMA has MM2S + S2MM, a feed DMA
//! only MM2S. Extracted from the former mmio.zig as part of the PL substrate so any
//! op (matmul, flash) drives DDR↔fabric transfers the same way.

const std = @import("std");
const regwin = @import("regwin.zig");

pub const Error = regwin.Error || error{ InvalidTransfer, DmaResetTimeout, DmaError, DmaTimeout };

// ---- AXI DMA simple-mode register offsets (Xilinx standard) ----
const MM2S_DMACR: u32 = 0x00;
const MM2S_DMASR: u32 = 0x04;
const MM2S_SA: u32 = 0x18;
const MM2S_SA_MSB: u32 = 0x1C;
const MM2S_LENGTH: u32 = 0x28;
const S2MM_DMACR: u32 = 0x30;
const S2MM_DMASR: u32 = 0x34;
const S2MM_DA: u32 = 0x48;
const S2MM_DA_MSB: u32 = 0x4C;
const S2MM_LENGTH: u32 = 0x58;

const RS: u32 = 1 << 0;
const IDLE: u32 = 1 << 1;
const RESET: u32 = 1 << 2;
const ERR_MASK: u32 = 0x70; // int/slv/dec error bits

pub const Dma = struct {
    win: regwin.RegWindow,

    pub fn open(base: i64) Error!Dma {
        return .{ .win = try regwin.RegWindow.mapWindow(base) };
    }

    pub fn deinit(self: *Dma) void {
        self.win.deinit();
    }

    /// Reset both channels (weights DMA has MM2S + S2MM).
    pub fn reset(self: *Dma) Error!void {
        self.win.wr(MM2S_DMACR, RESET);
        self.win.wr(S2MM_DMACR, RESET);
        try self.waitResetClear(MM2S_DMACR);
        try self.waitResetClear(S2MM_DMACR);
    }

    /// Reset only MM2S — for a DMA built without S2MM.
    pub fn resetMm2s(self: *Dma) Error!void {
        self.win.wr(MM2S_DMACR, RESET);
        try self.waitResetClear(MM2S_DMACR);
    }

    /// Reset only S2MM — for a DMA built without MM2S (e.g. an output-only channel).
    pub fn resetS2mm(self: *Dma) Error!void {
        self.win.wr(S2MM_DMACR, RESET);
        try self.waitResetClear(S2MM_DMACR);
    }

    pub fn startReadFromDdr(self: *Dma, src_phys: u64, len: usize) Error!void {
        if (len == 0 or len > std.math.maxInt(u32)) return error.InvalidTransfer;
        self.win.wr(MM2S_DMACR, RS);
        self.win.wr(MM2S_SA, @truncate(src_phys & 0xffff_ffff));
        self.win.wr(MM2S_SA_MSB, @truncate(src_phys >> 32));
        self.win.wr(MM2S_LENGTH, @intCast(len));
    }

    pub fn startWriteToDdr(self: *Dma, dst_phys: u64, len: usize) Error!void {
        if (len == 0 or len > std.math.maxInt(u32)) return error.InvalidTransfer;
        self.win.wr(S2MM_DMACR, RS);
        self.win.wr(S2MM_DA, @truncate(dst_phys & 0xffff_ffff));
        self.win.wr(S2MM_DA_MSB, @truncate(dst_phys >> 32));
        self.win.wr(S2MM_LENGTH, @intCast(len));
    }

    pub fn waitReadDone(self: *Dma) Error!void {
        try self.waitIdle(MM2S_DMASR);
    }

    pub fn waitWriteDone(self: *Dma) Error!void {
        try self.waitIdle(S2MM_DMASR);
    }

    fn waitResetClear(self: *Dma, cr: u32) Error!void {
        var i: usize = 0;
        while (i < regwin.wait_limit) : (i += 1) {
            if (self.win.rd(cr) & RESET == 0) return;
        }
        return error.DmaResetTimeout;
    }

    fn waitIdle(self: *Dma, sr: u32) Error!void {
        var i: usize = 0;
        while (i < regwin.wait_limit) : (i += 1) {
            const status = self.win.rd(sr);
            if (status & ERR_MASK != 0) return error.DmaError;
            if (status & IDLE != 0) return;
        }
        return error.DmaTimeout;
    }
};
