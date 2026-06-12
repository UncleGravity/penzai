//! Low-level PL access: a /dev/mem register window, the AXI DMA simple-mode
//! driver, and the Q1A8 kernel AXI-Lite driver. Register offsets come from the
//! generated `regmap` module — no hand-duplicated constants. Ported from
//! experiments/kr260-q1a8-matmul-bringup/src/{dma,mmio}.zig; this is the only
//! place volatile MMIO and /dev/mem effects live (plan-long §9).

const std = @import("std");
const regmap = @import("regmap");

pub const Error = error{
    OpenDevMem,
    MapFailed,
    InvalidTransfer,
    DmaResetTimeout,
    DmaError,
    DmaTimeout,
    KernelTimeout,
    BadId,
    BadVersion,
};

/// Bytes mapped per AXI-Lite block. All our blocks fit in one 64 KiB page span.
pub const mmio_span: usize = 0x10000;
/// Poll budget for DMA/kernel completion. Matches the bring-up's value; a single
/// matmul completes far under this, so hitting it means a real hang.
pub const wait_limit: usize = 500_000_000;

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

/// A mapped AXI-Lite register window. `byte_off` arguments are register byte
/// offsets (as in the regmap); indexing converts to the u32 word index.
pub const RegWindow = struct {
    fd: c_int,
    map: *anyopaque,

    pub fn mapWindow(base: i64) Error!RegWindow {
        const fd = open(@as([*:0]const u8, "/dev/mem"), O_RDWR | O_SYNC, 0);
        if (fd < 0) return error.OpenDevMem;
        errdefer _ = close(fd);
        const m = mmap(null, mmio_span, PROT_READ | PROT_WRITE, MAP_SHARED, fd, base);
        if (m == null or @intFromPtr(m.?) == MAP_FAILED) return error.MapFailed;
        return .{ .fd = fd, .map = m.? };
    }

    pub fn deinit(self: *RegWindow) void {
        _ = munmap(self.map, mmio_span);
        _ = close(self.fd);
        self.* = undefined;
    }

    inline fn regs(self: RegWindow) [*]volatile u32 {
        return @ptrCast(@alignCast(self.map));
    }

    pub inline fn rd(self: RegWindow, byte_off: u32) u32 {
        return self.regs()[byte_off / 4];
    }

    pub inline fn wr(self: RegWindow, byte_off: u32, value: u32) void {
        self.regs()[byte_off / 4] = value;
    }
};

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
const HALTED: u32 = 1 << 0;
const IDLE: u32 = 1 << 1;
const RESET: u32 = 1 << 2;
const ERR_MASK: u32 = 0x70; // int/slv/dec error bits

pub const Dma = struct {
    win: RegWindow,

    pub fn open(base: i64) Error!Dma {
        return .{ .win = try RegWindow.mapWindow(base) };
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

    /// Reset only MM2S — for a DMA built without S2MM (the acts lane).
    pub fn resetMm2s(self: *Dma) Error!void {
        self.win.wr(MM2S_DMACR, RESET);
        try self.waitResetClear(MM2S_DMACR);
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
        while (i < wait_limit) : (i += 1) {
            if (self.win.rd(cr) & RESET == 0) return;
        }
        return error.DmaResetTimeout;
    }

    fn waitIdle(self: *Dma, sr: u32) Error!void {
        var i: usize = 0;
        while (i < wait_limit) : (i += 1) {
            const status = self.win.rd(sr);
            if (status & ERR_MASK != 0) return error.DmaError;
            if (status & IDLE != 0) return;
        }
        return error.DmaTimeout;
    }
};

const CTRL_START: u32 = 1 << 0;
const STATUS_BUSY: u32 = 1 << 0;
const STATUS_DONE: u32 = 1 << 1;

/// Minimum kernel VERSION the driver accepts. v6 is the multi-column kernel that
/// reads the wide weight layout; v4/v5 read the old narrow layout and are
/// incompatible with the resident wide weights, so the PL path requires v6.
pub const min_version: u32 = 6;
pub const version_with_counters: u32 = 5;
pub const expected_id: u32 = 0xB05A2000;

pub const Kernel = struct {
    win: RegWindow,
    version: u32,

    pub fn open(base: i64) Error!Kernel {
        var win = try RegWindow.mapWindow(base);
        errdefer win.deinit();
        const id = win.rd(regmap.offsetOf("ID"));
        if (id != expected_id) return error.BadId;
        const version = win.rd(regmap.offsetOf("VERSION"));
        if (version < min_version) return error.BadVersion;
        return .{ .win = win, .version = version };
    }

    pub fn deinit(self: *Kernel) void {
        self.win.deinit();
    }

    pub fn hasCounters(self: Kernel) bool {
        return self.version >= version_with_counters;
    }

    /// Set dims then strobe start. done_latched clears on the strobe.
    pub fn run(self: *Kernel, num_q1_blocks: u32, num_rowblocks: u32, num_cols: u32) void {
        self.win.wr(regmap.offsetOf("NUM_Q1_BLOCKS"), num_q1_blocks);
        self.win.wr(regmap.offsetOf("NUM_ROWBLOCKS"), num_rowblocks);
        self.win.wr(regmap.offsetOf("NUM_COLS"), num_cols);
        self.win.wr(regmap.offsetOf("CTRL"), CTRL_START);
    }

    pub fn waitDone(self: *Kernel) Error!void {
        var i: usize = 0;
        while (i < wait_limit) : (i += 1) {
            if (self.win.rd(regmap.offsetOf("STATUS")) & STATUS_DONE != 0) return;
        }
        return error.KernelTimeout;
    }

    pub fn cycles(self: Kernel) u32 {
        return self.win.rd(regmap.offsetOf("CYCLES"));
    }

    pub fn wStall(self: Kernel) u32 {
        return self.win.rd(regmap.offsetOf("W_STALL"));
    }
    pub fn aStall(self: Kernel) u32 {
        return self.win.rd(regmap.offsetOf("A_STALL"));
    }
    pub fn rStall(self: Kernel) u32 {
        return self.win.rd(regmap.offsetOf("R_STALL"));
    }
    pub fn wBeats(self: Kernel) u32 {
        return self.win.rd(regmap.offsetOf("W_BEATS"));
    }
    pub fn aBeats(self: Kernel) u32 {
        return self.win.rd(regmap.offsetOf("A_BEATS"));
    }
    pub fn rBeats(self: Kernel) u32 {
        return self.win.rd(regmap.offsetOf("R_BEATS"));
    }
};
