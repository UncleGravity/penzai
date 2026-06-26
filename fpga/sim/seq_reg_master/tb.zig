//! Cosim for seq_reg_master: seq_core's register req/gnt port -> AXI-Lite MASTER. The tb models
//! the AXI-Lite SLAVE on the far side (captures writes, answers reads) and drives transactions on
//! the req/gnt side, asserting each retires with the right AXI write/read and gnt/rdata.

const std = @import("std");
const c = @cImport(@cInclude("shim.h"));

const Write = struct { addr: u32, val: u32 };

const Tb = struct {
    h: *c.Dut,
    writelog: [16]Write = undefined,
    wl_n: usize = 0,
    read_ret: u32 = 0,
    // AXI-Lite slave model state
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

    // The modeled AXI-Lite slave: accept AW/W (ready when valid), retire via B; accept AR, answer R.
    fn respond(self: *Tb) void {
        const h = self.h;
        // write address
        if (c.dut_awvalid(h) != 0) {
            c.dut_set_awready(h, 1);
            self.aw_addr = c.dut_awaddr(h);
            self.aw_seen = true;
        } else c.dut_set_awready(h, 0);
        // write data
        if (c.dut_wvalid(h) != 0) {
            c.dut_set_wready(h, 1);
            self.w_data = c.dut_wdata_m(h);
            self.w_seen = true;
        } else c.dut_set_wready(h, 0);
        // write response
        if (self.aw_seen and self.w_seen and !self.b_pending) {
            if (self.wl_n < self.writelog.len) {
                self.writelog[self.wl_n] = .{ .addr = self.aw_addr, .val = self.w_data };
                self.wl_n += 1;
            }
            self.b_pending = true;
        }
        c.dut_set_bvalid(h, if (self.b_pending) 1 else 0);
        c.dut_set_bresp(h, 0);
        if (self.b_pending and c.dut_bready(h) != 0) {
            self.b_pending = false;
            self.aw_seen = false;
            self.w_seen = false;
        }
        // read address
        if (c.dut_arvalid(h) != 0) {
            c.dut_set_arready(h, 1);
            self.ar_addr = c.dut_araddr(h);
            self.ar_seen = true;
        } else c.dut_set_arready(h, 0);
        // read data
        if (self.ar_seen and !self.r_pending) self.r_pending = true;
        c.dut_set_rvalid(h, if (self.r_pending) 1 else 0);
        c.dut_set_rdata_m(h, self.read_ret);
        c.dut_set_rresp(h, 0);
        if (self.r_pending and c.dut_rready(h) != 0) {
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
        c.dut_set_req(self.h, 0);
        c.dut_set_rst_n(self.h, 0);
        for (0..4) |_| {
            c.dut_set_clk(self.h, 1);
            c.dut_eval(self.h);
            c.dut_set_clk(self.h, 0);
            c.dut_eval(self.h);
        }
        c.dut_set_rst_n(self.h, 1);
        self.wl_n = 0;
        self.aw_seen = false;
        self.w_seen = false;
        self.b_pending = false;
        self.ar_seen = false;
        self.r_pending = false;
    }

    // Drive one transaction; hold req until gnt, then drop it (so S_IDLE doesn't re-trigger).
    fn xact(self: *Tb, we: bool, addr: u32, wdata: u32) !void {
        c.dut_set_req(self.h, 1);
        c.dut_set_we(self.h, if (we) 1 else 0);
        c.dut_set_addr(self.h, addr);
        c.dut_set_wdata(self.h, wdata);
        var cyc: usize = 0;
        while (cyc < 64) : (cyc += 1) {
            self.step();
            if (c.dut_gnt(self.h) != 0) {
                c.dut_set_req(self.h, 0);
                self.step(); // a req-low cycle, so the adapter re-arms (matches seq_core)
                return;
            }
        }
        return error.NoGnt;
    }
};

pub fn main() !void {
    var tb = Tb.init();
    defer tb.deinit();

    // ---- write: req/we -> AW+W+B, captured by the slave ----
    {
        tb.reset();
        try tb.xact(true, 0xA000_0018, 0xDEAD_BEEF);
        try std.testing.expectEqual(@as(usize, 1), tb.wl_n);
        try std.testing.expectEqual(Write{ .addr = 0xA000_0018, .val = 0xDEAD_BEEF }, tb.writelog[0]);
        std.debug.print("  write: AW/W/B retired, slave captured (0x{x}, 0x{x})\n", .{ tb.writelog[0].addr, tb.writelog[0].val });
    }

    // ---- read: req/!we -> AR+R, rdata returned ----
    {
        tb.reset();
        tb.read_ret = 0x0000_0002; // e.g. STATUS.done
        try tb.xact(false, 0xA000_0004, 0);
        try std.testing.expectEqual(@as(u32, 0x0000_0002), c.dut_rdata(tb.h));
        try std.testing.expectEqual(@as(usize, 0), tb.wl_n); // a read writes nothing
        std.debug.print("  read: AR/R retired, rdata=0x{x}\n", .{c.dut_rdata(tb.h)});
    }

    // ---- back-to-back: write then read, one master, sequential ----
    {
        tb.reset();
        try tb.xact(true, 0xA005_0000, 0x1);
        tb.read_ret = 0x55;
        try tb.xact(false, 0xA005_0008, 0);
        try std.testing.expectEqual(@as(u32, 0x55), c.dut_rdata(tb.h));
        try std.testing.expectEqual(@as(usize, 1), tb.wl_n);
        try std.testing.expectEqual(Write{ .addr = 0xA005_0000, .val = 0x1 }, tb.writelog[0]);
        std.debug.print("  back-to-back write+read: ok\n", .{});
    }

    std.debug.print("  all seq_reg_master cosim cases passed\n\n", .{});
}
