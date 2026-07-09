//! seq_smoke.zig - gate A of the seq.v silicon bring-up (docs/plan-seq-impl-v2.md increment 4):
//! prove command delivery + replay + error paths with ZERO DMA involvement before any
//! memory-mastering record is ever submitted. The whole run is WRITE/WAIT against the matmul
//! kernel's idle-safe config registers (NUM_Q1_BLOCKS / NUM_ROWBLOCKS — only sampled on
//! CTRL.start, which this never strobes).
//!
//! Run with penzaid STOPPED (both map the same kernel window; a concurrent dispatch would race
//! these registers). Refuses to touch the seq window unless the kernel self-describes
//! version_with_seq — on an older bitstream that window is an unmapped sc_ctrl hole.
//!
//! Gates proven here: CMD BRAM delivery over S_AXI, WRITE replay onto sc_ctrl, WAIT read+match,
//! RUN_START segment kicks (two resident runs), err_timeout on a never-matching WAIT (with
//! run-relative ERR_INDEX), ABORT back to idle, and executor reuse after both.

const std = @import("std");
const regmap = @import("regmap");
const regwin = @import("regwin.zig");
const seq = @import("seq.zig");
const seq_ctrl_mod = @import("seq_ctrl.zig");
const matmul = @import("matmul.zig");

pub const Error = regwin.Error || seq_ctrl_mod.Error || std.Io.Writer.Error ||
    error{ BadId, NoSeqInBitstream, SmokeMismatch, ExpectedTimeout, AbortDidNotIdle };

fn expectEq(out: *std.Io.Writer, what: []const u8, got: u32, want: u32) Error!void {
    if (got != want) {
        try out.print("  FAIL {s}: got 0x{X}, want 0x{X}\n", .{ what, got, want });
        return error.SmokeMismatch;
    }
    try out.print("  ok   {s} = 0x{X}\n", .{ what, got });
}

pub fn run(out: *std.Io.Writer) Error!void {
    try out.print("seq-smoke: gate A (no DMA). Run with penzaid STOPPED.\n", .{});

    var kwin = try regwin.RegWindow.mapWindow(regmap.addr.kernel);
    defer kwin.deinit();
    const id = kwin.rd(regmap.offsetOf("ID"));
    const version = kwin.rd(regmap.offsetOf("VERSION"));
    try out.print("  kernel ID=0x{X} VERSION={d}\n", .{ id, version });
    if (id != matmul.expected_id) return error.BadId;
    if (version < matmul.version_with_seq) {
        try out.print("  REFUSED: bitstream v{d} < v{d} carries no seq_top; its window is an unmapped hole.\n", .{ version, matmul.version_with_seq });
        return error.NoSeqInBitstream;
    }

    var sc = try seq_ctrl_mod.SeqCtrl.open(seq.ctrl.base);
    defer sc.deinit();

    const kb: u32 = @intCast(regmap.addr.kernel);
    const nq1_off = regmap.offsetOf("NUM_Q1_BLOCKS");
    const nrb_off = regmap.offsetOf("NUM_ROWBLOCKS");
    const nq1 = kb + nq1_off;
    const nrb = kb + nrb_off;

    // Three resident segments. A@0: write+wait on NUM_Q1_BLOCKS. B@8: same on NUM_ROWBLOCKS
    // (proves RUN_START offsetting). C@16: a WAIT that can never match (bit31 of a value we
    // control) — the err_timeout + abort path.
    var buf_a: [2]seq.Entry = undefined;
    var a = seq.Builder.init(&buf_a);
    a.write(nq1, 0xAB);
    a.wait(nq1, 0xFFFF_FFFF, 0xAB);
    try sc.load(0, a.entries());

    var buf_b: [2]seq.Entry = undefined;
    var b = seq.Builder.init(&buf_b);
    b.write(nrb, 0x5C);
    b.wait(nrb, 0xFFFF_FFFF, 0x5C);
    try sc.load(8, b.entries());

    var buf_c: [1]seq.Entry = undefined;
    var c = seq.Builder.init(&buf_c);
    c.wait(nq1, 0x8000_0000, 0x8000_0000);
    try sc.load(16, c.entries());

    // Segment A: replay writes a config reg, WAIT reads it back, PS verifies over MMIO.
    try sc.run(0, a.count());
    try sc.waitDone();
    try expectEq(out, "segment A replayed (NUM_Q1_BLOCKS)", kwin.rd(nq1_off), 0xAB);

    // Segment B by RUN_START: same executor, different resident run.
    try sc.run(8, b.count());
    try sc.waitDone();
    try expectEq(out, "segment B replayed (RUN_START=8)", kwin.rd(nrb_off), 0x5C);

    // Segment C: the WAIT never matches -> executor must err out (bounded), not hang.
    try sc.run(16, c.count());
    sc.waitDone() catch |err| switch (err) {
        error.SeqPollTimeout => {
            try expectEq(out, "err_timeout at run-relative entry", sc.errIndex(), 0);
            sc.abort();
            const status = sc.status();
            if (status & (seq.ctrl.STATUS_BUSY | seq.ctrl.STATUS_DONE) != 0) {
                try out.print("  FAIL abort: STATUS=0x{X}, want idle\n", .{status});
                return error.AbortDidNotIdle;
            }
            try out.print("  ok   abort -> idle (STATUS=0x{X})\n", .{status});
        },
        else => return err,
    };

    // Reuse after abort: segment A again, fresh value via a reload.
    var buf_d: [2]seq.Entry = undefined;
    var d = seq.Builder.init(&buf_d);
    d.write(nq1, 0x77);
    d.wait(nq1, 0xFFFF_FFFF, 0x77);
    try sc.load(0, d.entries());
    try sc.run(0, d.count());
    try sc.waitDone();
    try expectEq(out, "post-abort reuse", kwin.rd(nq1_off), 0x77);

    // Leave the kernel config as we found it (reset values are 0).
    kwin.wr(nq1_off, 0);
    kwin.wr(nrb_off, 0);

    try out.print("seq-smoke: ALL GATES PASSED — delivery, replay, segments, timeout, abort, reuse.\n", .{});
}
