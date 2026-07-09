//! seq.zig - the host side of the seq.v command contract (docs/plan-seq-impl-v2.md, v2.1).
//!
//! A run = a flat array of 16-byte entries living in seq_top's command BRAM (written through
//! its control slave — no DRAM, no cache coherency, by construction). The executor knows two
//! records only:
//!   WRITE(0): reg.wr(addr, a)                          (b unused)
//!   WAIT (1): poll until (reg.rd(addr) & a) == b       (a=mask, b=expected)
//! RUN_COUNT bounds every run; there is no END record. `addr` is an absolute sc_ctrl-space
//! byte address (the same addresses the PS pokes over MMIO today).
//!
//! Builder accumulates a run; validate() is the pre-submit safety gate: it refuses any stream
//! that would program a DMA transfer outside the caller's known buffer ranges — the failure
//! class that bricked the board under v1 (a rogue S2MM over kernel memory).

const std = @import("std");

pub const Tag = enum(u32) { write = 0, wait = 1 };

/// seq_top control-slave register map — host mirror of fpga/rtl/seq/seq_top.v's OFF_*
/// localparams and STATUS/CTRL bits. One-address contract with the BD (build.tcl assigns
/// seq_top S_AXI at `base`).
pub const ctrl = struct {
    pub const base: i64 = 0xA020_0000;

    pub const RUN_START: u32 = 0x00; // RW: entry index the run begins at
    pub const RUN_COUNT: u32 = 0x04; // RW: entries to execute
    pub const CTRL: u32 = 0x08; // W : bit0 go, bit1 abort
    pub const STATUS: u32 = 0x0C; // RO: {err_watchdog[3], err_timeout[2], done[1], busy[0]}
    pub const ERR_INDEX: u32 = 0x10; // RO: entry index at the fault (debug)

    pub const CTRL_GO: u32 = 1 << 0;
    pub const CTRL_ABORT: u32 = 1 << 1;
    pub const STATUS_BUSY: u32 = 1 << 0;
    pub const STATUS_DONE: u32 = 1 << 1;
    pub const STATUS_ERR_TIMEOUT: u32 = 1 << 2;
    pub const STATUS_ERR_WATCHDOG: u32 = 1 << 3;

    /// Command window: entry i = 4 little-endian u32 words at CMD_OFF + 16*i. At 0x8000 the
    /// RTL decode is one address bit and 2048 entries fill the 64 KiB window exactly.
    pub const CMD_OFF: u32 = 0x8000;
    /// Entries the seq_top CMD BRAM holds (CMD_DEPTH_LOG2 = 11). The resident-program phase
    /// widens the BD window and this together.
    pub const cmd_capacity: u32 = 2048;
};

/// One command entry; `extern` pins the layout to the wire format seq_core.v decodes.
pub const Entry = extern struct {
    tag: u32,
    addr: u32,
    a: u32,
    b: u32,
};

comptime {
    if (@sizeOf(Entry) != 16) @compileError("Entry must be 16 bytes (4 CMD-window words)");
}

/// Fixed-capacity entry accumulator over a caller-provided backing store. Pushes are
/// infallible — an over-cap run trips `overflow` (checked once at submit) rather than
/// failing each push on the dispatch path.
pub const Builder = struct {
    buf: []Entry,
    n: usize = 0,
    overflow: bool = false,

    pub fn init(buf: []Entry) Builder {
        return .{ .buf = buf };
    }

    pub fn reset(self: *Builder) void {
        self.n = 0;
        self.overflow = false;
    }

    fn push(self: *Builder, e: Entry) void {
        if (self.n < self.buf.len) {
            self.buf[self.n] = e;
            self.n += 1;
        } else {
            self.overflow = true;
        }
    }

    pub fn write(self: *Builder, addr: u32, val: u32) void {
        self.push(.{ .tag = @intFromEnum(Tag.write), .addr = addr, .a = val, .b = 0 });
    }
    pub fn wait(self: *Builder, addr: u32, mask: u32, expected: u32) void {
        self.push(.{ .tag = @intFromEnum(Tag.wait), .addr = addr, .a = mask, .b = expected });
    }

    pub fn count(self: *const Builder) u32 {
        return @intCast(self.n);
    }
    pub fn entries(self: *const Builder) []const Entry {
        return self.buf[0..self.n];
    }
};

// ---- pre-submit validation ------------------------------------------------------------------

/// AXI-DMA register offsets the validator interprets (fixed Xilinx map; mirrors dma.zig — kept
/// in sync by the pl_tests stream test, which builds through dma.zig's own constants).
const MM2S_SA: u32 = 0x18;
const MM2S_SA_MSB: u32 = 0x1C;
const MM2S_LENGTH: u32 = 0x28;
const S2MM_DA: u32 = 0x48;
const S2MM_DA_MSB: u32 = 0x4C;
const S2MM_LENGTH: u32 = 0x58;

/// Half-open device-address range [lo, hi).
pub const Range = struct { lo: u64, hi: u64 };

pub const ValidateError = error{
    /// A DMA LENGTH write with no prior SA/DA in the same stream — the transfer target is
    /// unknowable, refuse it.
    LengthBeforeAddress,
    /// An MM2S (read-from-DDR) transfer outside every allowed range.
    Mm2sOutOfRange,
    /// An S2MM (write-to-DDR) transfer outside every allowed range — the brick class.
    S2mmOutOfRange,
    /// More distinct DMA windows in the stream than the validator tracks.
    TooManyDmas,
};

fn inAllowed(allowed: []const Range, lo: u64, len: u64) bool {
    for (allowed) |r| {
        if (lo >= r.lo and lo + len <= r.hi) return true;
    }
    return false;
}

/// Refuse any stream that programs a DMA transfer outside `allowed`. `dma_bases` lists the
/// AXI-Lite window bases (sc_ctrl space) of every DMA the stream may touch; writes to other
/// windows (kernels, seq itself) pass through unchecked — they cannot master memory. LENGTH
/// triggers the transfer (Xilinx simple mode), so the address pair is required to have been
/// written earlier in the SAME stream: the stream must be self-contained, never relying on
/// leftover hardware state.
pub fn validate(items: []const Entry, dma_bases: []const u32, allowed: []const Range) ValidateError!void {
    const max_dmas = 16;
    if (dma_bases.len > max_dmas) return error.TooManyDmas;
    var sa: [max_dmas]?u64 = @splat(null);
    var sa_hi: [max_dmas]u32 = @splat(0);
    var da: [max_dmas]?u64 = @splat(null);
    var da_hi: [max_dmas]u32 = @splat(0);

    for (items) |e| {
        if (e.tag != @intFromEnum(Tag.write)) continue;
        const slot = for (dma_bases, 0..) |b, i| {
            if (e.addr >= b and e.addr - b < 0x60) break i;
        } else continue;
        switch (e.addr - dma_bases[slot]) {
            MM2S_SA => sa[slot] = (@as(u64, sa_hi[slot]) << 32) | e.a,
            MM2S_SA_MSB => {
                sa_hi[slot] = e.a;
                if (sa[slot]) |v| sa[slot] = (@as(u64, e.a) << 32) | (v & 0xffff_ffff);
            },
            S2MM_DA => da[slot] = (@as(u64, da_hi[slot]) << 32) | e.a,
            S2MM_DA_MSB => {
                da_hi[slot] = e.a;
                if (da[slot]) |v| da[slot] = (@as(u64, e.a) << 32) | (v & 0xffff_ffff);
            },
            MM2S_LENGTH => {
                const src = sa[slot] orelse return error.LengthBeforeAddress;
                if (!inAllowed(allowed, src, e.a)) return error.Mm2sOutOfRange;
            },
            S2MM_LENGTH => {
                const dst = da[slot] orelse return error.LengthBeforeAddress;
                if (!inAllowed(allowed, dst, e.a)) return error.S2mmOutOfRange;
            },
            else => {},
        }
    }
}

// ---- tests -----------------------------------------------------------------------------------

test "builder encodes the wire layout" {
    var buf: [4]Entry = undefined;
    var b = Builder.init(&buf);
    b.write(0xA000_0018, 0xdead_beef);
    b.wait(0xA000_0004, 0x2, 0x2);
    try std.testing.expectEqual(@as(u32, 2), b.count());
    try std.testing.expectEqual(Entry{ .tag = 0, .addr = 0xA000_0018, .a = 0xdead_beef, .b = 0 }, b.entries()[0]);
    try std.testing.expectEqual(Entry{ .tag = 1, .addr = 0xA000_0004, .a = 0x2, .b = 0x2 }, b.entries()[1]);
    // In-memory layout must be the 4-word little-endian CMD image.
    const bs = std.mem.sliceAsBytes(b.entries());
    try std.testing.expectEqual(@as(usize, 32), bs.len);
    try std.testing.expectEqual(@as(u32, 0xA000_0018), std.mem.readInt(u32, bs[4..8], .little));

    b.reset();
    try std.testing.expectEqual(@as(u32, 0), b.count());
}

test "builder flags overflow instead of trapping" {
    var buf: [1]Entry = undefined;
    var b = Builder.init(&buf);
    b.write(0x10, 0x1);
    b.write(0x14, 0x2); // over capacity
    try std.testing.expect(b.overflow);
    try std.testing.expectEqual(@as(u32, 1), b.count());
}

test "validate: in-range transfers pass, S2MM outside every range is refused" {
    const base: u32 = 0xA000_0000;
    const allowed = [_]Range{.{ .lo = 0x8_0000_0000, .hi = 0x8_0010_0000 }};
    var buf: [16]Entry = undefined;
    var b = Builder.init(&buf);
    // MM2S read inside the range.
    b.write(base + MM2S_SA, 0x0000_1000);
    b.write(base + MM2S_SA_MSB, 0x8);
    b.write(base + MM2S_LENGTH, 0x100);
    // S2MM write inside the range.
    b.write(base + S2MM_DA, 0x0002_0000);
    b.write(base + S2MM_DA_MSB, 0x8);
    b.write(base + S2MM_LENGTH, 0x40);
    try validate(b.entries(), &.{base}, &allowed);

    // Same stream, S2MM retargeted below the range (e.g. kernel memory) → refused.
    b.write(base + S2MM_DA, 0x1000_0000);
    b.write(base + S2MM_DA_MSB, 0x0);
    b.write(base + S2MM_LENGTH, 0x40);
    try std.testing.expectError(error.S2mmOutOfRange, validate(b.entries(), &.{base}, &allowed));
}

test "validate: transfer straddling a range edge is refused" {
    const base: u32 = 0xA001_0000;
    const allowed = [_]Range{.{ .lo = 0x1000, .hi = 0x2000 }};
    var buf: [8]Entry = undefined;
    var b = Builder.init(&buf);
    b.write(base + S2MM_DA, 0x1F00);
    b.write(base + S2MM_DA_MSB, 0);
    b.write(base + S2MM_LENGTH, 0x200); // [0x1F00, 0x2100) leaks past hi
    try std.testing.expectError(error.S2mmOutOfRange, validate(b.entries(), &.{base}, &allowed));
}

test "validate: LENGTH with no in-stream address is refused" {
    const base: u32 = 0xA000_0000;
    var buf: [4]Entry = undefined;
    var b = Builder.init(&buf);
    b.write(base + S2MM_LENGTH, 0x40); // relies on leftover DA state → refuse
    try std.testing.expectError(error.LengthBeforeAddress, validate(b.entries(), &.{base}, &.{.{ .lo = 0, .hi = 1 << 40 }}));
}

test "validate: non-DMA writes and waits pass through unchecked" {
    var buf: [4]Entry = undefined;
    var b = Builder.init(&buf);
    b.write(0xA005_0010, 123); // kernel config — not a DMA window
    b.wait(0xA005_0000, 0x2, 0x2);
    try validate(b.entries(), &.{0xA000_0000}, &.{});
}
