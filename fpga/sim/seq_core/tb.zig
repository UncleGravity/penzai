//! Cosim for seq_core: the descriptor-executor core of seq.v (docs/plan-seq-impl.md).
//!
//! seq_core is a BUS MASTER, so the tb models the slaves it drives: a descriptor memory
//! (returns the 128b entry for an index) and a register file (records every WRITE; answers
//! WAIT reads, with one "status" register that flips to its done value after K reads — the
//! kernel-becomes-done case). Both use a 1-cycle req/gnt responder. We then assert seq_core
//! replayed the writes in the EXACT order, polled the WAIT until it matched, honored the END
//! tag, and timed out a never-satisfied WAIT. No software ref — this is pure control logic.

const std = @import("std");
const c = @cImport(@cInclude("shim.h"));

// Entry = u32x4 {tag, addr, a, b}, matching seq_core's 128b little-endian layout.
const TAG_WRITE: u32 = 0;
const TAG_WAIT: u32 = 1;
const TAG_END: u32 = 2;
fn wr(addr: u32, val: u32) [4]u32 {
    return .{ TAG_WRITE, addr, val, 0 };
}
fn wait(addr: u32, mask: u32, exp: u32) [4]u32 {
    return .{ TAG_WAIT, addr, mask, exp };
}
fn end() [4]u32 {
    return .{ TAG_END, 0, 0, 0 };
}

const Write = struct { addr: u32, val: u32 };

const Model = struct {
    h: *c.Dut,
    entries: []const [4]u32 = &.{},
    writelog: [256]Write = undefined,
    wl_n: usize = 0,
    // one status register that reads 0 until it has been read > flip_after times, then flip_val.
    flip_addr: u32 = 0xFFFF_FFFF,
    flip_after: u32 = 0,
    flip_val: u32 = 0,
    flip_reads: u32 = 0,
    // dead-slave mode: the register port never grants (the watchdog's trigger condition).
    stall_reg: bool = false,
    desc_gnt_prev: bool = false,
    reg_gnt_prev: bool = false,

    fn init() Model {
        return .{ .h = c.dut_new().? };
    }
    fn deinit(self: *Model) void {
        c.dut_free(self.h);
    }

    fn modelRead(self: *Model, addr: u32) u32 {
        if (addr == self.flip_addr) {
            self.flip_reads += 1;
            return if (self.flip_reads > self.flip_after) self.flip_val else 0;
        }
        return 0;
    }

    // Drive the 1-cycle req/gnt responses for both ports, based on the DUT's current requests.
    fn respond(self: *Model) void {
        // descriptor port
        if (self.desc_gnt_prev) {
            c.dut_set_desc_gnt(self.h, 0);
            self.desc_gnt_prev = false;
        } else if (c.dut_desc_req(self.h) != 0) {
            const i: usize = @intCast(c.dut_desc_idx(self.h));
            const e = self.entries[i];
            c.dut_set_desc_data(self.h, e[0], e[1], e[2], e[3]);
            c.dut_set_desc_gnt(self.h, 1);
            self.desc_gnt_prev = true;
        }
        // register port
        if (self.stall_reg) {
            // dead slave: leave gnt low forever
        } else if (self.reg_gnt_prev) {
            c.dut_set_reg_gnt(self.h, 0);
            self.reg_gnt_prev = false;
        } else if (c.dut_reg_req(self.h) != 0) {
            const addr = c.dut_reg_addr(self.h);
            if (c.dut_reg_we(self.h) != 0) {
                if (self.wl_n < self.writelog.len) {
                    self.writelog[self.wl_n] = .{ .addr = addr, .val = c.dut_reg_wdata(self.h) };
                    self.wl_n += 1;
                }
            } else {
                c.dut_set_reg_rdata(self.h, self.modelRead(addr));
            }
            c.dut_set_reg_gnt(self.h, 1);
            self.reg_gnt_prev = true;
        }
    }

    fn step(self: *Model) void {
        self.respond();
        c.dut_set_clk(self.h, 1);
        c.dut_eval(self.h);
        c.dut_set_clk(self.h, 0);
        c.dut_eval(self.h);
    }

    fn reset(self: *Model) void {
        c.dut_set_go(self.h, 0);
        c.dut_set_desc_gnt(self.h, 0);
        c.dut_set_reg_gnt(self.h, 0);
        c.dut_set_rst_n(self.h, 0);
        for (0..4) |_| {
            c.dut_set_clk(self.h, 1);
            c.dut_eval(self.h);
            c.dut_set_clk(self.h, 0);
            c.dut_eval(self.h);
        }
        c.dut_set_rst_n(self.h, 1);
        self.wl_n = 0;
        self.flip_reads = 0;
        self.stall_reg = false;
        self.desc_gnt_prev = false;
        self.reg_gnt_prev = false;
    }

    // Start a run of `count` entries; step until done (or maxcyc). Returns cycles spent.
    fn run(self: *Model, count: u32, maxcyc: usize) usize {
        c.dut_set_desc_count(self.h, count);
        c.dut_set_go(self.h, 1);
        self.step(); // posedge samples go
        c.dut_set_go(self.h, 0);
        var cyc: usize = 0;
        while (c.dut_done(self.h) == 0 and cyc < maxcyc) : (cyc += 1) self.step();
        return cyc;
    }
};

fn expectWrites(m: *Model, want: []const Write) !void {
    if (m.wl_n != want.len) {
        std.debug.print("  FAIL: {d} writes, want {d}\n", .{ m.wl_n, want.len });
        return error.WriteCountMismatch;
    }
    for (want, 0..) |w, i| {
        if (m.writelog[i].addr != w.addr or m.writelog[i].val != w.val) {
            std.debug.print("  FAIL: write[{d}] = (0x{x},0x{x}), want (0x{x},0x{x})\n", .{ i, m.writelog[i].addr, m.writelog[i].val, w.addr, w.val });
            return error.WriteMismatch;
        }
    }
}

pub fn main() !void {
    var m = Model.init();
    defer m.deinit();

    // ---- Scenario 1: WRITEs bracketing a WAIT that satisfies after 3 failing polls ----
    {
        m.reset();
        const desc = [_][4]u32{
            wr(0x100, 0xAAAA),
            wr(0x104, 0xBBBB),
            wait(0x200, 0x2, 0x2), // poll STATUS bit1 until set
            wr(0x108, 0xCCCC),
        };
        m.entries = &desc;
        m.flip_addr = 0x200;
        m.flip_after = 3; // reads 1..3 → 0, read 4 → 0x2
        m.flip_val = 0x2;
        const cyc = m.run(4, 2000);
        if (c.dut_done(m.h) == 0) {
            std.debug.print("  FAIL: scenario 1 never finished ({d} cyc)\n", .{cyc});
            return error.NoDone;
        }
        if (c.dut_err_timeout(m.h) != 0) return error.UnexpectedTimeout;
        try expectWrites(&m, &.{
            .{ .addr = 0x100, .val = 0xAAAA },
            .{ .addr = 0x104, .val = 0xBBBB },
            .{ .addr = 0x108, .val = 0xCCCC },
        });
        if (m.flip_reads < 4) {
            std.debug.print("  FAIL: WAIT polled {d}x, expected >=4\n", .{m.flip_reads});
            return error.WaitUnderPolled;
        }
        std.debug.print("  scenario 1 (write/wait/write): 3 writes in order, WAIT polled {d}x, done in {d} cyc\n", .{ m.flip_reads, cyc });
    }

    // ---- Scenario 2: a WAIT that never satisfies → timeout ----
    {
        m.reset();
        const desc = [_][4]u32{wait(0x200, 0x2, 0x2)};
        m.entries = &desc;
        m.flip_addr = 0xDEAD; // 0x200 always reads 0
        const cyc = m.run(1, 50_000);
        if (c.dut_done(m.h) == 0) return error.NoDone;
        if (c.dut_err_timeout(m.h) == 0) {
            std.debug.print("  FAIL: scenario 2 expected timeout, got none\n", .{});
            return error.NoTimeout;
        }
        if (c.dut_err_index(m.h) != 0) return error.WrongErrIndex;
        try expectWrites(&m, &.{}); // no writes
        std.debug.print("  scenario 2 (never-satisfied WAIT): timed out at entry {d} in {d} cyc\n", .{ c.dut_err_index(m.h), cyc });
    }

    // ---- Scenario 3: END tag stops a run early (3rd entry must NOT execute) ----
    {
        m.reset();
        const desc = [_][4]u32{ wr(0x10, 0x55), end(), wr(0x20, 0x66) };
        m.entries = &desc;
        m.flip_addr = 0xFFFF_FFFF;
        const cyc = m.run(3, 500);
        if (c.dut_done(m.h) == 0) return error.NoDone;
        if (c.dut_err_timeout(m.h) != 0) return error.UnexpectedTimeout;
        try expectWrites(&m, &.{.{ .addr = 0x10, .val = 0x55 }});
        std.debug.print("  scenario 3 (END mid-run): only the pre-END write replayed, done in {d} cyc\n", .{cyc});
    }

    // ---- Scenario 4: empty run (count 0) finishes immediately, no writes ----
    {
        m.reset();
        const desc = [_][4]u32{wr(0x10, 0x55)};
        m.entries = &desc;
        const cyc = m.run(0, 100);
        if (c.dut_done(m.h) == 0) return error.NoDone;
        try expectWrites(&m, &.{});
        std.debug.print("  scenario 4 (empty run): finished in {d} cyc, no writes\n", .{cyc});
    }

    // ---- Scenario 5: dead register slave (no gnt, ever) → global watchdog, not a hang ----
    {
        m.reset();
        const desc = [_][4]u32{ wr(0x10, 0x55), wr(0x20, 0x66) };
        m.entries = &desc;
        m.stall_reg = true;
        const cyc = m.run(2, 50_000); // cosim WATCHDOG_TIMEOUT=4096 << 50k budget
        if (c.dut_done(m.h) == 0) {
            std.debug.print("  FAIL: scenario 5 hung instead of watchdogging\n", .{});
            return error.NoDone;
        }
        if (c.dut_err_watchdog(m.h) == 0) return error.NoWatchdog;
        if (c.dut_err_timeout(m.h) != 0) return error.UnexpectedTimeout;
        if (c.dut_err_index(m.h) != 0) return error.WrongErrIndex; // stuck on entry 0
        try expectWrites(&m, &.{}); // the stuck write never granted
        std.debug.print("  scenario 5 (dead slave): watchdog fired at entry {d} in {d} cyc\n", .{ c.dut_err_index(m.h), cyc });

        // ...and the core is reusable after a reset (what seq_top's ABORT does).
        m.reset();
        m.entries = &desc;
        _ = m.run(2, 2000);
        if (c.dut_done(m.h) == 0 or c.dut_err_watchdog(m.h) != 0) return error.NotReusableAfterAbort;
        try expectWrites(&m, &.{ .{ .addr = 0x10, .val = 0x55 }, .{ .addr = 0x20, .val = 0x66 } });
        std.debug.print("  scenario 5b (post-abort reuse): clean replay after reset\n", .{});
    }

    // ---- Scenario 6: desc_count is sampled with go, not observed live mid-run ----
    {
        m.reset();
        const desc = [_][4]u32{ wr(0x30, 0x88), wr(0x34, 0x99) };
        m.entries = &desc;
        c.dut_set_desc_count(m.h, 2);
        c.dut_set_go(m.h, 1);
        m.step();
        c.dut_set_go(m.h, 0);
        c.dut_set_desc_count(m.h, 1); // must not shorten the already-started run
        var cyc: usize = 0;
        while (c.dut_done(m.h) == 0 and cyc < 2000) : (cyc += 1) m.step();
        if (c.dut_done(m.h) == 0) return error.NoDone;
        try expectWrites(&m, &.{ .{ .addr = 0x30, .val = 0x88 }, .{ .addr = 0x34, .val = 0x99 } });
        std.debug.print("  scenario 6 (count snapshot): mid-run desc_count rewrite ignored\n", .{});
    }

    std.debug.print("  all seq_core cosim cases passed\n\n", .{});
}
