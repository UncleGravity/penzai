//! seq.zig - the host side of the seq.v descriptor contract (docs/plan-seq-impl.md).
//!
//! A Recorder accumulates a run of {WRITE|WAIT|END} entries that seq_core.v replays. The
//! point: the SAME dispatch code that pokes MMIO today (dma.zig, the op kernels) instead
//! appends entries here when handed a record-mode Bus (bus.zig) — so the descriptor builder
//! IS the dispatch, one source of truth, no second encoding to drift.
//!
//! Entry = 16 B, little-endian u32x4 {tag, addr, a, b} = one 128-bit DRAM beat, matching
//! seq_core.v's decode exactly:
//!   WRITE(0): reg.wr(addr, a)                          (b unused)
//!   WAIT (1): poll until (reg.rd(addr) & a) == b       (a=mask, b=expected)
//!   END  (2): stop a run early

const std = @import("std");

pub const Tag = enum(u32) { write = 0, wait = 1, end = 2 };

/// seq_top control-slave register map — the host mirror of fpga/rtl/seq/seq_top.v's OFF_*
/// localparams + STATUS/CTRL bits. The PS programs the descriptor region + count, strobes go, and
/// polls STATUS once per run.
pub const ctrl = struct {
    /// seq_top control-slave AXI-Lite base. MUST match the BD's assign_bd_address in
    /// fpga/bitstreams/combined-v1/tcl/build.tcl (a one-address contract; not yet in the regmap
    /// codegen). Above the matmul (0xA00x) / flash (0xA01x) windows.
    pub const base: i64 = 0xA020_0000;

    pub const DESC_BASE_LO: u32 = 0x00; // RW
    pub const DESC_BASE_HI: u32 = 0x04; // RW
    pub const DESC_COUNT: u32 = 0x08; // RW
    pub const CTRL: u32 = 0x0C; // W: bit0 = go
    pub const STATUS: u32 = 0x10; // RO: {err[2], done[1], busy[0]}
    pub const ERR_INDEX: u32 = 0x14; // RO

    pub const CTRL_GO: u32 = 1 << 0;
    pub const STATUS_BUSY: u32 = 1 << 0;
    pub const STATUS_DONE: u32 = 1 << 1;
    pub const STATUS_ERR: u32 = 1 << 2;
};

/// One descriptor entry; `extern` pins the layout to the wire format seq_core.v fetches.
pub const Entry = extern struct {
    tag: u32,
    addr: u32,
    a: u32,
    b: u32,
};

comptime {
    if (@sizeOf(Entry) != 16) @compileError("Entry must be 16 bytes (one 128-bit beat)");
}

/// Fixed-capacity entry accumulator over a caller-provided backing store (in integration the
/// store is the DRAM descriptor region). Pushes are infallible — an over-cap run trips
/// `overflow` (checked once at the end) rather than failing each `wr` on the dispatch path.
pub const Recorder = struct {
    buf: []Entry,
    n: usize = 0,
    overflow: bool = false,

    pub fn init(buf: []Entry) Recorder {
        return .{ .buf = buf };
    }

    pub fn reset(self: *Recorder) void {
        self.n = 0;
        self.overflow = false;
    }

    fn push(self: *Recorder, e: Entry) void {
        if (self.n < self.buf.len) {
            self.buf[self.n] = e;
            self.n += 1;
        } else {
            self.overflow = true;
        }
    }

    pub fn write(self: *Recorder, addr: u32, val: u32) void {
        self.push(.{ .tag = @intFromEnum(Tag.write), .addr = addr, .a = val, .b = 0 });
    }
    pub fn wait(self: *Recorder, addr: u32, mask: u32, expected: u32) void {
        self.push(.{ .tag = @intFromEnum(Tag.wait), .addr = addr, .a = mask, .b = expected });
    }
    pub fn end(self: *Recorder) void {
        self.push(.{ .tag = @intFromEnum(Tag.end), .addr = 0, .a = 0, .b = 0 });
    }

    pub fn count(self: *const Recorder) u32 {
        return @intCast(self.n);
    }
    pub fn entries(self: *const Recorder) []const Entry {
        return self.buf[0..self.n];
    }
    /// The descriptor run as raw bytes — what the host DMAs to the seq.v descriptor region.
    pub fn bytes(self: *const Recorder) []const u8 {
        return std.mem.sliceAsBytes(self.buf[0..self.n]);
    }
};

test "recorder encodes the wire layout and tags" {
    var buf: [4]Entry = undefined;
    var r = Recorder.init(&buf);
    r.write(0xA000_0018, 0xdead_beef);
    r.wait(0xA000_0004, 0x2, 0x2);
    r.end();
    try std.testing.expectEqual(@as(u32, 3), r.count());
    try std.testing.expectEqual(Entry{ .tag = 0, .addr = 0xA000_0018, .a = 0xdead_beef, .b = 0 }, r.entries()[0]);
    try std.testing.expectEqual(Entry{ .tag = 1, .addr = 0xA000_0004, .a = 0x2, .b = 0x2 }, r.entries()[1]);
    try std.testing.expectEqual(@as(u32, 2), r.entries()[2].tag);
    // bytes() must be 16 B/entry, little-endian {tag,addr,a,b}.
    const bs = r.bytes();
    try std.testing.expectEqual(@as(usize, 48), bs.len);
    try std.testing.expectEqual(@as(u32, 0xA000_0018), std.mem.readInt(u32, bs[4..8], .little));

    r.reset();
    try std.testing.expectEqual(@as(u32, 0), r.count());
}

test "recorder flags overflow instead of trapping" {
    var buf: [1]Entry = undefined;
    var r = Recorder.init(&buf);
    r.write(0x10, 0x1);
    r.write(0x14, 0x2); // over capacity
    try std.testing.expect(r.overflow);
    try std.testing.expectEqual(@as(u32, 1), r.count());
}
