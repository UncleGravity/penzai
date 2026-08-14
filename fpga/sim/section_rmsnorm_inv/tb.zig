//! Bit-exact control and numeric cosim for `section_rmsnorm_inv`.

const std = @import("std");
const section = @import("shared_section");
const c = @cImport(@cInclude("shim.h"));

const eps_1e_6: u32 = 0x3586_37bd;

const Record = struct {
    max_exp: u8,
    sum_sq: u48,
};

const Dut = struct {
    handle: *c.Dut,

    fn init() Dut {
        return .{ .handle = c.dut_new().? };
    }

    fn deinit(self: *Dut) void {
        c.dut_free(self.handle);
    }

    fn eval(self: *Dut) void {
        c.dut_eval(self.handle);
    }

    fn step(self: *Dut) void {
        c.dut_set_clk(self.handle, 1);
        self.eval();
        c.dut_set_clk(self.handle, 0);
        self.eval();
    }

    fn reset(self: *Dut) void {
        c.dut_set_clk(self.handle, 0);
        c.dut_set_rst_n(self.handle, 0);
        c.dut_set_config(self.handle, 0, 0, 0, 0);
        c.dut_set_abort(self.handle, 0);
        c.dut_set_record(self.handle, 0, 0, 0, 0, 0, 0);
        c.dut_set_result_ready(self.handle, 0);
        self.eval();
        for (0..5) |_| self.step();
        c.dut_set_rst_n(self.handle, 1);
        self.step();
    }
};

fn configure(dut: *Dut, rows: u32, tokens: u3, eps: u32) !void {
    try std.testing.expect(c.dut_config_ready(dut.handle) != 0);
    c.dut_set_config(dut.handle, 1, @intCast(rows), tokens, eps);
    dut.eval();
    try std.testing.expect(c.dut_config_ready(dut.handle) != 0);
    dut.step();
    c.dut_set_config(dut.handle, 0, 0, 0, 0);
    dut.eval();
}

fn runSuccess(
    dut: *Dut,
    records: []const Record,
    rows: u32,
    eps: u32,
    random_stalls: bool,
    seed: u64,
) !usize {
    try configure(dut, rows, @intCast(records.len), eps);
    try std.testing.expect(c.dut_busy(dut.handle) != 0);

    var expected: [4]u32 = undefined;
    for (records, 0..) |record, token| {
        expected[token] = (try section.rmsNormInvFixed(
            record.sum_sq,
            rows,
            record.max_exp,
            eps,
        )).inv_rms_bits;
    }

    var prng = std.Random.DefaultPrng.init(seed);
    const rnd = prng.random();
    var sent: usize = 0;
    var received: usize = 0;
    var presenting = false;
    var held_bits: ?u32 = null;
    var cycles: usize = 0;
    while (received < records.len) : (cycles += 1) {
        if (cycles > records.len * 160 + 128) return error.StreamTimeout;
        if (!presenting and sent < records.len)
            presenting = !random_stalls or rnd.uintLessThan(u8, 4) != 0;
        const record = if (sent < records.len) records[sent] else records[0];
        c.dut_set_record(
            dut.handle,
            @intFromBool(presenting),
            @intCast(sent),
            record.max_exp,
            record.sum_sq,
            @intCast(rows),
            @intFromBool(sent + 1 == records.len),
        );
        const ready = !random_stalls or rnd.uintLessThan(u8, 3) != 0;
        c.dut_set_result_ready(dut.handle, @intFromBool(ready));
        dut.eval();

        const input_fire = presenting and c.dut_record_ready(dut.handle) != 0;
        const output_valid = c.dut_result_valid(dut.handle) != 0;
        const output_fire = output_valid and ready;
        if (output_valid and !ready) {
            const bits = c.dut_result_inv_rms(dut.handle);
            if (held_bits) |prior| try std.testing.expectEqual(prior, bits);
            held_bits = bits;
        } else {
            held_bits = null;
        }
        if (output_fire) {
            try std.testing.expectEqual(@as(u8, @intCast(received)), c.dut_result_token(dut.handle));
            try std.testing.expectEqual(expected[received], c.dut_result_inv_rms(dut.handle));
            try std.testing.expectEqual(received + 1 == records.len, c.dut_result_final(dut.handle) != 0);
            received += 1;
        }
        if (input_fire) {
            sent += 1;
            presenting = false;
        }

        dut.step();
        try std.testing.expect(c.dut_error(dut.handle) == 0);
        if (output_fire and received == records.len) {
            try std.testing.expect(c.dut_done(dut.handle) != 0);
            try std.testing.expect(c.dut_busy(dut.handle) == 0);
        }
    }
    try std.testing.expectEqual(records.len, sent);
    return cycles;
}

fn expectConfigFault(dut: *Dut, rows: u32, tokens: u3, eps: u32) !void {
    try configure(dut, rows, tokens, eps);
    try std.testing.expect(c.dut_done(dut.handle) != 0);
    try std.testing.expect(c.dut_error(dut.handle) != 0);
    try std.testing.expectEqual(section.RmsNormInvStatus.bad_cfg, @as(u4, @truncate(c.dut_status(dut.handle))));
    try std.testing.expect(c.dut_busy(dut.handle) == 0);
    dut.step();
}

fn sendFaultRecord(
    dut: *Dut,
    token: u2,
    max_exp: u8,
    sum_sq: u48,
    rows: u32,
    final: bool,
    expected_status: u4,
) !void {
    c.dut_set_record(
        dut.handle,
        1,
        token,
        max_exp,
        sum_sq,
        @intCast(rows),
        @intFromBool(final),
    );
    c.dut_set_result_ready(dut.handle, 1);
    dut.eval();
    try std.testing.expect(c.dut_record_ready(dut.handle) != 0);
    dut.step();
    c.dut_set_record(dut.handle, 0, 0, 0, 0, 0, 0);
    dut.eval();
    var cycles: usize = 0;
    while (c.dut_done(dut.handle) == 0) : (cycles += 1) {
        if (cycles > 96) return error.FaultTimeout;
        try std.testing.expect(c.dut_result_valid(dut.handle) == 0);
        dut.step();
    }
    try std.testing.expect(c.dut_error(dut.handle) != 0);
    try std.testing.expectEqual(expected_status, @as(u4, @truncate(c.dut_status(dut.handle))));
    try std.testing.expect(c.dut_busy(dut.handle) == 0);
    dut.step();
}

fn verifyFaults(dut: *Dut) !void {
    try expectConfigFault(dut, 24, 1, eps_1e_6);
    try expectConfigFault(dut, 2048, 0, eps_1e_6);
    try expectConfigFault(dut, 2048, 5, eps_1e_6);
    try expectConfigFault(dut, 2048, 1, 0);
    try expectConfigFault(dut, 2048, 1, 0xbf80_0000);

    const one_sum: u48 = @as(u48, 2048) << 34;
    try configure(dut, 2048, 1, eps_1e_6);
    try sendFaultRecord(dut, 1, 127, one_sum, 2048, true, section.RmsNormInvStatus.frame);
    try configure(dut, 2048, 1, eps_1e_6);
    try sendFaultRecord(dut, 0, 127, one_sum, 1024, true, section.RmsNormInvStatus.frame);
    try configure(dut, 2048, 1, eps_1e_6);
    try sendFaultRecord(dut, 0, 0, one_sum, 2048, true, section.RmsNormInvStatus.frame);

    try configure(dut, 8, 1, eps_1e_6);
    try sendFaultRecord(dut, 0, 254, 0xffff_ffff_ffff, 8, true, section.RmsNormInvStatus.arithmetic);
    try configure(dut, 8, 1, 0x0080_0000);
    try sendFaultRecord(dut, 0, 0, 0, 8, true, section.RmsNormInvStatus.arithmetic);
}

fn verifyAbortRestart(dut: *Dut) !void {
    const one = Record{ .max_exp = 127, .sum_sq = @as(u48, 2048) << 34 };
    const abort_delays = [_]usize{ 0, 2, 6, 14, 28 };
    for (abort_delays, 0..) |delay, index| {
        try configure(dut, 2048, 1, eps_1e_6);
        c.dut_set_record(dut.handle, 1, 0, one.max_exp, one.sum_sq, 2048, 1);
        c.dut_set_result_ready(dut.handle, 1);
        dut.eval();
        try std.testing.expect(c.dut_record_ready(dut.handle) != 0);
        dut.step();
        c.dut_set_record(dut.handle, 0, 0, 0, 0, 0, 0);
        for (0..delay) |_| dut.step();
        c.dut_set_abort(dut.handle, 1);
        dut.step();
        c.dut_set_abort(dut.handle, 0);
        dut.eval();
        try std.testing.expect(c.dut_busy(dut.handle) == 0);
        try std.testing.expect(c.dut_result_valid(dut.handle) == 0);
        try std.testing.expect(c.dut_error(dut.handle) == 0);
        try std.testing.expectEqual(@as(u8, 0), c.dut_status(dut.handle));

        _ = try runSuccess(dut, &.{one}, 2048, eps_1e_6, true, 0xa55a_1000 + index);
    }
}

pub fn main() !void {
    var dut = Dut.init();
    defer dut.deinit();
    dut.reset();

    const base: u48 = @as(u48, 2048) << 34;
    const records = [_]Record{
        .{ .max_exp = 127, .sum_sq = base },
        .{ .max_exp = 128, .sum_sq = base },
        .{ .max_exp = 126, .sum_sq = base },
        .{ .max_exp = 0, .sum_sq = 0 },
    };
    const cycles = try runSuccess(&dut, &records, 2048, eps_1e_6, true, 0x1357_2468);
    try std.testing.expect(cycles < 320);

    var prng = std.Random.DefaultPrng.init(0x19b3_7221);
    const rnd = prng.random();
    var random_records: [4]Record = undefined;
    for (&random_records, 0..) |*record, token| {
        record.* = .{
            .max_exp = @intCast(110 + token * 7),
            .sum_sq = rnd.intRangeAtMost(u48, 1 << 40, 0xefff_ffff_ffff),
        };
    }
    _ = try runSuccess(&dut, &random_records, 4096, eps_1e_6, true, 0x8844_2211);

    try verifyFaults(&dut);
    try verifyAbortRestart(&dut);

    std.debug.print(
        "section RMSNorm inverse scalar: four-token run={d} cycles; " ++
            "bit-exact oracle, stalls, faults, and pipeline abort/restart passed\n",
        .{cycles},
    );
}
