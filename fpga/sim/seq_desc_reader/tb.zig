//! Cosim for seq_desc_reader: seq_core's descriptor req/gnt port -> AXI4 read MASTER. The tb models
//! the AXI4 read SLAVE as a descriptor memory (returns the 128-bit entry at desc_base + idx*16) and
//! checks the reader fetches the right entry for each index, with a correct AR address.

const std = @import("std");
const c = @cImport(@cInclude("shim.h"));

const Entry = [4]u32;

const Tb = struct {
    h: *c.Dut,
    mem: []const Entry = &.{},
    desc_base: u64 = 0,
    ar_addr: u64 = 0,
    ar_seen: bool = false,
    r_pending: bool = false,

    fn init() Tb {
        return .{ .h = c.dut_new().? };
    }
    fn deinit(self: *Tb) void {
        c.dut_free(self.h);
    }

    fn respond(self: *Tb) void {
        const h = self.h;
        if (c.dut_arvalid(h) != 0) {
            c.dut_set_arready(h, 1);
            self.ar_addr = c.dut_araddr(h);
            self.ar_seen = true;
        } else c.dut_set_arready(h, 0);

        if (self.ar_seen and !self.r_pending) self.r_pending = true;
        if (self.r_pending) {
            const idx: usize = @intCast((self.ar_addr - self.desc_base) / 16);
            const e = self.mem[idx];
            c.dut_set_rdata(h, e[0], e[1], e[2], e[3]);
            c.dut_set_rvalid(h, 1);
            c.dut_set_rlast(h, 1);
            c.dut_set_rresp(h, 0);
        } else {
            c.dut_set_rvalid(h, 0);
        }
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
        c.dut_set_desc_req(self.h, 0);
        c.dut_set_rst_n(self.h, 0);
        for (0..4) |_| {
            c.dut_set_clk(self.h, 1);
            c.dut_eval(self.h);
            c.dut_set_clk(self.h, 0);
            c.dut_eval(self.h);
        }
        c.dut_set_rst_n(self.h, 1);
        self.ar_seen = false;
        self.r_pending = false;
    }

    fn fetch(self: *Tb, idx: u32) !Entry {
        c.dut_set_desc_req(self.h, 1);
        c.dut_set_desc_idx(self.h, idx);
        var cyc: usize = 0;
        while (cyc < 64) : (cyc += 1) {
            self.step();
            if (c.dut_desc_gnt(self.h) != 0) {
                const got = Entry{
                    c.dut_desc_data(self.h, 0), c.dut_desc_data(self.h, 1),
                    c.dut_desc_data(self.h, 2), c.dut_desc_data(self.h, 3),
                };
                c.dut_set_desc_req(self.h, 0);
                self.step(); // a req-low cycle, so the adapter re-arms (matches seq_core)
                return got;
            }
        }
        return error.NoGnt;
    }
};

pub fn main() !void {
    var tb = Tb.init();
    defer tb.deinit();

    const mem = [_]Entry{
        .{ 0x1000, 0x2000, 0x3000, 0x4000 },
        .{ 0x1001, 0x2001, 0x3001, 0x4001 },
        .{ 0x1002, 0x2002, 0x3002, 0x4002 },
        .{ 0xAAAA, 0xBBBB, 0xCCCC, 0xDDDD },
    };
    tb.mem = &mem;
    tb.desc_base = 0x8_0000_0000;

    tb.reset();
    c.dut_set_desc_base(tb.h, tb.desc_base);

    // Fetch entries out of order; each must return mem[idx] (AR addr = base + idx*16).
    for ([_]u32{ 0, 3, 1, 2, 3 }) |idx| {
        const got = try tb.fetch(idx);
        try std.testing.expectEqual(mem[idx], got);
    }
    std.debug.print("  fetched entries 0,3,1,2,3 — each matched mem[idx]\n", .{});
    std.debug.print("  all seq_desc_reader cosim cases passed\n\n", .{});
}
