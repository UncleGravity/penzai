//! End-to-end cosim for seq_top v2.1: control AXI-Lite slave + command BRAM + seq_core +
//! seq_reg_master, all wired. The tb drives S_AXI as the PS (fill the CMD window, program
//! RUN_START/RUN_COUNT, strobe go/abort, poll STATUS) and models M_AXI_REG's far side as the
//! sc_ctrl target (writelog + a status register that flips done after K reads — the WAIT case).
//!
//! Covers the v2.1 contract:
//!  1. two runs RESIDENT at different CMD indices, kicked separately by RUN_START — the
//!     segmented replay the resident-program design relies on, exact write order per run
//!  2. RUN_COUNT=0 finishes immediately
//!  3. ABORT mid-WAIT recovers to IDLE, and the executor + BRAM are reusable right after
//!     (reload same indices, clean replay) — the reclaim-over-SSH property v1 lacked
//!  4. rewriting RUN_START while busy does not redirect the active command segment

const std = @import("std");
const c = @cImport(@cInclude("shim.h"));

// Host mirror of seq_top's control map (device/pl/seq.zig `ctrl`).
const OFF_RUN_START: u32 = 0x00;
const OFF_RUN_COUNT: u32 = 0x04;
const OFF_CTRL: u32 = 0x08;
const OFF_STATUS: u32 = 0x0C;
const OFF_ERR_INDEX: u32 = 0x10;
const CMD_OFF: u32 = 0x8000;
const CTRL_GO: u32 = 1 << 0;
const CTRL_ABORT: u32 = 1 << 1;
const ST_BUSY: u32 = 1 << 0;
const ST_DONE: u32 = 1 << 1;
const ST_ERR_TIMEOUT: u32 = 1 << 2;
const ST_ERR_WATCHDOG: u32 = 1 << 3;

const Entry = [4]u32;
fn wr(addr: u32, val: u32) Entry {
    return .{ 0, addr, val, 0 };
}
fn wait(addr: u32, mask: u32, exp: u32) Entry {
    return .{ 1, addr, mask, exp };
}

const Write = struct { addr: u32, val: u32 };

const Tb = struct {
    h: *c.Dut,
    // register target model (M_AXI_REG far side)
    writelog: [32]Write = undefined,
    wl_n: usize = 0,
    flip_addr: u32 = 0xFFFF_FFFF,
    flip_after: u32 = 0,
    flip_val: u32 = 0,
    flip_reads: u32 = 0,
    aw_addr: u32 = 0,
    w_data: u32 = 0,
    ar_addr: u32 = 0,
    aw_seen: bool = false,
    w_seen: bool = false,
    b_pending: bool = false,
    ar_seen: bool = false,
    r_pending: bool = false,

    fn init() Tb {
        return .{ .h = c.dut_new().? };
    }
    fn deinit(self: *Tb) void {
        c.dut_free(self.h);
    }

    fn regRead(self: *Tb, addr: u32) u32 {
        if (addr == self.flip_addr) {
            self.flip_reads += 1;
            return if (self.flip_reads > self.flip_after) self.flip_val else 0;
        }
        return 0;
    }

    // ---- M_AXI_REG slave (sc_ctrl target: writelog + flipping status read) ----
    fn respond(self: *Tb) void {
        const h = self.h;
        if (c.dut_reg_awvalid(h) != 0) {
            c.dut_set_reg_awready(h, 1);
            self.aw_addr = c.dut_reg_awaddr(h);
            self.aw_seen = true;
        } else c.dut_set_reg_awready(h, 0);
        if (c.dut_reg_wvalid(h) != 0) {
            c.dut_set_reg_wready(h, 1);
            self.w_data = c.dut_reg_wdata(h);
            self.w_seen = true;
        } else c.dut_set_reg_wready(h, 0);
        if (self.aw_seen and self.w_seen and !self.b_pending) {
            if (self.wl_n < self.writelog.len) {
                self.writelog[self.wl_n] = .{ .addr = self.aw_addr, .val = self.w_data };
                self.wl_n += 1;
            }
            self.b_pending = true;
        }
        c.dut_set_reg_bvalid(h, if (self.b_pending) 1 else 0);
        if (self.b_pending and c.dut_reg_bready(h) != 0) {
            self.b_pending = false;
            self.aw_seen = false;
            self.w_seen = false;
        }
        if (c.dut_reg_arvalid(h) != 0) {
            c.dut_set_reg_arready(h, 1);
            self.ar_addr = c.dut_reg_araddr(h);
            self.ar_seen = true;
        } else c.dut_set_reg_arready(h, 0);
        if (self.ar_seen and !self.r_pending) {
            self.r_pending = true;
            c.dut_set_reg_rdata(h, self.regRead(self.ar_addr));
        }
        c.dut_set_reg_rvalid(h, if (self.r_pending) 1 else 0);
        if (self.r_pending and c.dut_reg_rready(h) != 0) {
            self.r_pending = false;
            self.ar_seen = false;
        }
    }

    fn step(self: *Tb) void {
        self.respond();
        c.dut_set_clk(self.h, 1);
        c.dut_eval(self.h);
        c.dut_set_clk(self.h, 0);
        c.dut_eval(self.h);
    }

    fn reset(self: *Tb) void {
        c.dut_set_s_awvalid(self.h, 0);
        c.dut_set_s_wvalid(self.h, 0);
        c.dut_set_s_bready(self.h, 0);
        c.dut_set_s_arvalid(self.h, 0);
        c.dut_set_s_rready(self.h, 0);
        c.dut_set_rst_n(self.h, 0);
        for (0..4) |_| self.step();
        c.dut_set_rst_n(self.h, 1);
    }

    fn clearLog(self: *Tb) void {
        self.wl_n = 0;
        self.flip_reads = 0;
    }

    // ---- S_AXI master driver (the PS) ----
    fn writeReg(self: *Tb, off: u32, val: u32) !void {
        const h = self.h;
        c.dut_set_s_awaddr(h, off);
        c.dut_set_s_awvalid(h, 1);
        c.dut_set_s_wdata(h, val);
        c.dut_set_s_wvalid(h, 1);
        c.dut_set_s_bready(h, 1);
        var cyc: usize = 0;
        while (cyc < 64) : (cyc += 1) {
            self.step();
            if (c.dut_s_bvalid(h) != 0) break;
        } else return error.NoBvalid;
        c.dut_set_s_awvalid(h, 0);
        c.dut_set_s_wvalid(h, 0);
        self.step(); // bvalid clears (bready high)
        c.dut_set_s_bready(h, 0);
    }

    fn readReg(self: *Tb, off: u32) !u32 {
        const h = self.h;
        c.dut_set_s_araddr(h, off);
        c.dut_set_s_arvalid(h, 1);
        c.dut_set_s_rready(h, 1);
        var cyc: usize = 0;
        var val: u32 = 0;
        while (cyc < 64) : (cyc += 1) {
            self.step();
            if (c.dut_s_rvalid(h) != 0) {
                val = c.dut_s_rdata(h);
                break;
            }
        } else return error.NoRvalid;
        c.dut_set_s_arvalid(h, 0);
        self.step(); // rvalid clears (rready high)
        c.dut_set_s_rready(h, 0);
        return val;
    }

    /// Load entries into the CMD window at entry index `at` — what seq_ctrl.load does.
    fn load(self: *Tb, at: u32, entries: []const Entry) !void {
        for (entries, 0..) |e, i| {
            const base = CMD_OFF + (at + @as(u32, @intCast(i))) * 16;
            for (e, 0..) |word, lane| {
                try self.writeReg(base + @as(u32, @intCast(lane)) * 4, word);
            }
        }
    }

    fn kick(self: *Tb, start: u32, count: u32) !void {
        try self.writeReg(OFF_RUN_START, start);
        try self.writeReg(OFF_RUN_COUNT, count);
        try self.writeReg(OFF_CTRL, CTRL_GO);
    }

    /// Poll STATUS.done like the PS does; each read advances the sim.
    fn waitDone(self: *Tb, max_polls: usize) !u32 {
        var i: usize = 0;
        while (i < max_polls) : (i += 1) {
            const status = try self.readReg(OFF_STATUS);
            if (status & ST_DONE != 0) return status;
        }
        return error.NeverDone;
    }
};

fn expectWrites(tb: *Tb, want: []const Write) !void {
    try std.testing.expectEqualSlices(Write, want, tb.writelog[0..tb.wl_n]);
}

pub fn main() !void {
    var tb = Tb.init();
    defer tb.deinit();
    tb.reset();

    // ---- Scenario 1: two runs resident at different indices, kicked by RUN_START ----
    {
        // Run A at entry 0: two kernel-config writes. Run B at entry 8: write, WAIT, write.
        const run_a = [_]Entry{ wr(0xA005_0010, 0x11), wr(0xA005_0014, 0x22) };
        const run_b = [_]Entry{
            wr(0xA000_0018, 0x33),
            wait(0xA005_0000, 0x2, 0x2), // kernel STATUS.done
            wr(0xA000_001C, 0x44),
        };
        try tb.load(0, &run_a);
        try tb.load(8, &run_b);

        tb.clearLog();
        try tb.kick(0, 2);
        var status = try tb.waitDone(400);
        if (status & (ST_ERR_TIMEOUT | ST_ERR_WATCHDOG) != 0) return error.UnexpectedErr;
        try expectWrites(&tb, &.{ .{ .addr = 0xA005_0010, .val = 0x11 }, .{ .addr = 0xA005_0014, .val = 0x22 } });

        tb.clearLog();
        tb.flip_addr = 0xA005_0000;
        tb.flip_after = 3; // reads 1..3 -> 0, read 4 -> 0x2
        tb.flip_val = 0x2;
        try tb.kick(8, 3);
        status = try tb.waitDone(400);
        if (status & (ST_ERR_TIMEOUT | ST_ERR_WATCHDOG) != 0) return error.UnexpectedErr;
        try expectWrites(&tb, &.{ .{ .addr = 0xA000_0018, .val = 0x33 }, .{ .addr = 0xA000_001C, .val = 0x44 } });
        if (tb.flip_reads < 4) return error.WaitUnderPolled;
        std.debug.print("  scenario 1 (two resident runs): segment replay by RUN_START, WAIT polled {d}x\n", .{tb.flip_reads});
    }

    // ---- Scenario 2: RUN_COUNT=0 finishes immediately, no writes ----
    {
        tb.clearLog();
        tb.flip_addr = 0xFFFF_FFFF;
        try tb.kick(0, 0);
        _ = try tb.waitDone(50);
        try expectWrites(&tb, &.{});
        std.debug.print("  scenario 2 (empty run): immediate done\n", .{});
    }

    // ---- Scenario 3: ABORT mid-WAIT -> IDLE, then reload + clean reuse ----
    {
        // A WAIT that never matches: the executor loops polling (slave responsive, POLL_TIMEOUT
        // is the silicon-scale default and never fires in this horizon).
        const stuck = [_]Entry{ wr(0xA005_0010, 0x55), wait(0xA005_0000, 0x2, 0x2) };
        try tb.load(0, &stuck);
        tb.clearLog();
        tb.flip_addr = 0xDEAD; // the polled addr always reads 0
        try tb.kick(0, 2);
        for (0..200) |_| tb.step(); // let it get stuck in the poll loop
        var status = try tb.readReg(OFF_STATUS);
        if (status & ST_BUSY == 0) return error.ExpectedBusy;

        try tb.writeReg(OFF_CTRL, CTRL_ABORT);
        status = try tb.readReg(OFF_STATUS);
        if (status & (ST_BUSY | ST_DONE) != 0) {
            std.debug.print("  FAIL: post-abort STATUS=0x{x}, want idle\n", .{status});
            return error.AbortDidNotIdle;
        }

        // Reload the same indices with a clean run and go again — executor + BRAM reusable.
        const clean = [_]Entry{ wr(0xA005_0020, 0x66), wr(0xA005_0024, 0x77) };
        try tb.load(0, &clean);
        tb.clearLog();
        try tb.kick(0, 2);
        status = try tb.waitDone(400);
        if (status & (ST_ERR_TIMEOUT | ST_ERR_WATCHDOG) != 0) return error.UnexpectedErr;
        try expectWrites(&tb, &.{ .{ .addr = 0xA005_0020, .val = 0x66 }, .{ .addr = 0xA005_0024, .val = 0x77 } });
        std.debug.print("  scenario 3 (abort mid-WAIT): recovered to idle, clean reuse after reload\n", .{});
    }

    // ---- Scenario 4: RUN_START is a snapshot for the active run ----
    {
        const original = [_]Entry{
            wait(0xA005_0000, 0x2, 0x2),
            wr(0xA005_0030, 0x88),
        };
        const redirected = [_]Entry{wr(0xDEAD_0000, 0xBAD0)};
        try tb.load(0, &original);
        try tb.load(9, &redirected);

        tb.clearLog();
        tb.flip_addr = 0xA005_0000;
        tb.flip_after = std.math.maxInt(u32);
        tb.flip_val = 0x2;
        try tb.kick(0, 2);
        for (0..200) |_| {
            tb.step();
            if (tb.flip_reads != 0) break;
        }
        if (tb.flip_reads == 0) return error.WaitNotStarted;

        try tb.writeReg(OFF_RUN_START, 8);
        tb.flip_after = tb.flip_reads;
        const status = try tb.waitDone(400);
        if (status & (ST_ERR_TIMEOUT | ST_ERR_WATCHDOG) != 0) return error.UnexpectedErr;
        try expectWrites(&tb, &.{.{ .addr = 0xA005_0030, .val = 0x88 }});
        std.debug.print("  scenario 4 (start snapshot): mid-run RUN_START rewrite ignored\n", .{});
    }

    std.debug.print("  all seq_top cosim cases passed\n\n", .{});
}
