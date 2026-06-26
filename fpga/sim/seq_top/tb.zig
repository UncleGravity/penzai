//! End-to-end cosim for seq_top: control AXI-Lite slave + seq_core + the two master adapters,
//! all wired. The tb drives S_AXI as the PS (write DESC_BASE/COUNT, strobe go, poll STATUS) and
//! models the far-side slaves: M_AXI_DESC as a descriptor memory and M_AXI_REG as the sc_ctrl
//! target (writelog + a status register that flips done after K reads — the WAIT case). One run
//! of {WRITE, WRITE, WAIT, WRITE} must replay the 3 writes in order, satisfy the WAIT, and finish.

const std = @import("std");
const c = @cImport(@cInclude("shim.h"));

const OFF_DESC_BASE_LO: u32 = 0x00;
const OFF_DESC_BASE_HI: u32 = 0x04;
const OFF_DESC_COUNT: u32 = 0x08;
const OFF_CTRL: u32 = 0x0C;
const OFF_STATUS: u32 = 0x10;
const ST_DONE: u32 = 0x2;
const ST_ERR: u32 = 0x4;

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
    // descriptor memory (M_AXI_DESC)
    mem: []const Entry = &.{},
    desc_base: u64 = 0,
    d_addr: u64 = 0,
    d_ar_seen: bool = false,
    d_r_pending: bool = false,
    // register target (M_AXI_REG)
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

    fn respond(self: *Tb) void {
        const h = self.h;
        // ---- M_AXI_DESC slave (descriptor memory) ----
        if (c.dut_desc_arvalid(h) != 0) {
            c.dut_set_desc_arready(h, 1);
            self.d_addr = c.dut_desc_araddr(h);
            self.d_ar_seen = true;
        } else c.dut_set_desc_arready(h, 0);
        if (self.d_ar_seen and !self.d_r_pending) self.d_r_pending = true;
        if (self.d_r_pending) {
            const idx: usize = @intCast((self.d_addr - self.desc_base) / 16);
            const e = self.mem[idx];
            c.dut_set_desc_rdata(h, e[0], e[1], e[2], e[3]);
            c.dut_set_desc_rvalid(h, 1);
            c.dut_set_desc_rlast(h, 1);
        } else c.dut_set_desc_rvalid(h, 0);
        if (self.d_r_pending and c.dut_desc_rready(h) != 0) {
            self.d_r_pending = false;
            self.d_ar_seen = false;
        }

        // ---- M_AXI_REG slave (sc_ctrl target: writelog + flipping status read) ----
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

    // ---- S_AXI master driver ----
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
};

pub fn main() !void {
    var tb = Tb.init();
    defer tb.deinit();
    tb.reset();

    const desc = [_]Entry{
        wr(0xA000_0010, 0x11),
        wr(0xA000_0014, 0x22),
        wait(0xA005_0000, 0x2, 0x2), // kernel STATUS.done
        wr(0xA000_0018, 0x33),
    };
    tb.mem = &desc;
    tb.desc_base = 0x8_0000_0000;
    tb.flip_addr = 0xA005_0000;
    tb.flip_after = 3; // reads 1..3 -> 0, read 4 -> 0x2
    tb.flip_val = 0x2;

    // Program the run over S_AXI, then go.
    try tb.writeReg(OFF_DESC_BASE_LO, @truncate(tb.desc_base & 0xffff_ffff));
    try tb.writeReg(OFF_DESC_BASE_HI, @truncate(tb.desc_base >> 32));
    try tb.writeReg(OFF_DESC_COUNT, 4);
    try tb.writeReg(OFF_CTRL, 1);

    // Poll STATUS.done (each read advances the sim while the core runs autonomously).
    var status: u32 = 0;
    var i: usize = 0;
    while (i < 400) : (i += 1) {
        status = try tb.readReg(OFF_STATUS);
        if (status & ST_DONE != 0) break;
    } else return error.NeverDone;

    if (status & ST_ERR != 0) {
        std.debug.print("  FAIL: err_timeout set\n", .{});
        return error.Timeout;
    }
    const want = [_]Write{
        .{ .addr = 0xA000_0010, .val = 0x11 },
        .{ .addr = 0xA000_0014, .val = 0x22 },
        .{ .addr = 0xA000_0018, .val = 0x33 },
    };
    try std.testing.expectEqualSlices(Write, &want, tb.writelog[0..tb.wl_n]);
    if (tb.flip_reads < 4) {
        std.debug.print("  FAIL: WAIT polled {d}x, expected >=4\n", .{tb.flip_reads});
        return error.WaitUnderPolled;
    }
    std.debug.print("  end-to-end: 3 writes replayed in order via M_AXI_REG, WAIT polled {d}x, STATUS.done\n", .{tb.flip_reads});
    std.debug.print("  all seq_top cosim cases passed\n\n", .{});
}
