//! Op-agnostic AXI DMA simple-mode driver (Xilinx standard), built on the PL control bus.
//! One Dma per channel; a weights DMA has MM2S + S2MM, a feed DMA only MM2S. Extracted from
//! the former mmio.zig as part of the PL substrate so any op (matmul, flash) drives DDR↔fabric
//! transfers the same way.
//!
//! The driver talks to a `bus.Bus`, which is either the real MMIO window OR a record-mode
//! Recorder (seq.v descriptor). So the SAME reset/start/wait code that pokes hardware today is
//! the descriptor builder when handed a record bus — one source of truth (docs/plan-seq-impl.md).

const std = @import("std");
const regwin = @import("regwin.zig");
const bus_mod = @import("bus.zig");
const seq = @import("seq.zig");

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
    bus: bus_mod.Bus,

    /// MMIO-backed (silicon): map this DMA's AXI-Lite window at `base`.
    pub fn open(base: i64) Error!Dma {
        return .{ .bus = try bus_mod.Bus.mmio(@intCast(base)) };
    }

    /// Record-backed: programming this DMA appends seq.v descriptor entries (sc_ctrl-space
    /// addresses = base + reg offset) to `rec` instead of touching hardware.
    pub fn openRecord(base: u32, rec: *seq.Recorder) Dma {
        return .{ .bus = bus_mod.Bus.record(base, rec) };
    }

    pub fn deinit(self: *Dma) void {
        self.bus.deinit();
    }

    /// Reset both channels (weights DMA has MM2S + S2MM).
    pub fn reset(self: *Dma) Error!void {
        self.bus.wr(MM2S_DMACR, RESET);
        self.bus.wr(S2MM_DMACR, RESET);
        try self.waitResetClear(MM2S_DMACR);
        try self.waitResetClear(S2MM_DMACR);
    }

    /// Reset only MM2S — for a DMA built without S2MM.
    pub fn resetMm2s(self: *Dma) Error!void {
        self.bus.wr(MM2S_DMACR, RESET);
        try self.waitResetClear(MM2S_DMACR);
    }

    /// Reset only S2MM — for a DMA built without MM2S (e.g. an output-only channel).
    pub fn resetS2mm(self: *Dma) Error!void {
        self.bus.wr(S2MM_DMACR, RESET);
        try self.waitResetClear(S2MM_DMACR);
    }

    pub fn startReadFromDdr(self: *Dma, src_phys: u64, len: usize) Error!void {
        if (len == 0 or len > std.math.maxInt(u32)) return error.InvalidTransfer;
        self.bus.wr(MM2S_DMACR, RS);
        self.bus.wr(MM2S_SA, @truncate(src_phys & 0xffff_ffff));
        self.bus.wr(MM2S_SA_MSB, @truncate(src_phys >> 32));
        self.bus.wr(MM2S_LENGTH, @intCast(len));
    }

    pub fn startWriteToDdr(self: *Dma, dst_phys: u64, len: usize) Error!void {
        if (len == 0 or len > std.math.maxInt(u32)) return error.InvalidTransfer;
        self.bus.wr(S2MM_DMACR, RS);
        self.bus.wr(S2MM_DA, @truncate(dst_phys & 0xffff_ffff));
        self.bus.wr(S2MM_DA_MSB, @truncate(dst_phys >> 32));
        self.bus.wr(S2MM_LENGTH, @intCast(len));
    }

    pub fn waitReadDone(self: *Dma) Error!void {
        try self.waitIdle(MM2S_DMASR);
    }

    pub fn waitWriteDone(self: *Dma) Error!void {
        try self.waitIdle(S2MM_DMASR);
    }

    fn waitResetClear(self: *Dma, cr: u32) Error!void {
        if (self.bus.isRecording()) {
            self.bus.recordWait(cr, RESET, 0); // poll until RESET bit clears
            return;
        }
        var i: usize = 0;
        while (i < regwin.wait_limit) : (i += 1) {
            if (self.bus.rd(cr) & RESET == 0) return;
        }
        return error.DmaResetTimeout;
    }

    fn waitIdle(self: *Dma, sr: u32) Error!void {
        if (self.bus.isRecording()) {
            self.bus.recordWait(sr, IDLE, IDLE); // poll until IDLE bit set
            return;
        }
        var i: usize = 0;
        while (i < regwin.wait_limit) : (i += 1) {
            const status = self.bus.rd(sr);
            if (status & ERR_MASK != 0) return error.DmaError;
            if (status & IDLE != 0) return;
        }
        return error.DmaTimeout;
    }
};

test "record mode: a weights-DMA op emits the exact PS register stream as a descriptor" {
    const E = seq.Entry;
    const W = struct {
        fn wr(addr: u32, val: u32) E {
            return .{ .tag = 0, .addr = addr, .a = val, .b = 0 };
        }
        fn wait(addr: u32, mask: u32, exp: u32) E {
            return .{ .tag = 1, .addr = addr, .a = mask, .b = exp };
        }
    };

    var buf: [32]E = undefined;
    var rec = seq.Recorder.init(&buf);
    const base: u32 = 0xA000_0000; // dma_w0
    var dma = Dma.openRecord(base, &rec);

    // Mirror the matmul per-op DMA dance (matmul.zig tryMatmul): reset, arm the S2MM result,
    // start the MM2S weight read, then wait both. Addresses/lengths are illustrative.
    try dma.reset();
    try dma.startWriteToDdr(0x1_2345_6789, 256); // result -> DDR
    try dma.startReadFromDdr(0x8_8000_0000, 512); // weights <- DDR
    try dma.waitReadDone();
    try dma.waitWriteDone();

    try std.testing.expect(!rec.overflow);
    const want = [_]E{
        // reset(): both channels reset, then wait both clear
        W.wr(base + 0x00, RESET), W.wr(base + 0x30, RESET),
        W.wait(base + 0x00, RESET, 0), W.wait(base + 0x30, RESET, 0),
        // startWriteToDdr(0x1_2345_6789, 256)
        W.wr(base + 0x30, RS), W.wr(base + 0x48, 0x2345_6789), W.wr(base + 0x4C, 0x1), W.wr(base + 0x58, 256),
        // startReadFromDdr(0x8_8000_0000, 512)
        W.wr(base + 0x00, RS), W.wr(base + 0x18, 0x8000_0000), W.wr(base + 0x1C, 0x8), W.wr(base + 0x28, 512),
        // waitReadDone / waitWriteDone
        W.wait(base + 0x04, IDLE, IDLE), W.wait(base + 0x34, IDLE, IDLE),
    };
    try std.testing.expectEqualSlices(E, &want, rec.entries());
}
