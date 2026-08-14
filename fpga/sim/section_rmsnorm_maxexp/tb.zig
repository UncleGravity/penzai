//! Bit-exact control and datapath cosim for `section_rmsnorm_maxexp`.

const std = @import("std");
const section = @import("shared_section");
const c = @cImport(@cInclude("shim.h"));

const zero_group = [_]u32{0} ** 8;

const Result = struct {
    token: u2,
    max_exp: u8,
    rows: u14,
    subnormal_warning: bool,
    final: bool,
};

const RunStats = struct {
    cycles: usize,
    group_stalls: usize,
    result_stalls: usize,
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
        c.dut_set_config(self.handle, 0, 0, 0);
        c.dut_set_abort(self.handle, 0);
        c.dut_set_group(self.handle, 0, &zero_group, 0, 0);
        c.dut_set_result_ready(self.handle, 0);
        self.eval();
        for (0..4) |_| self.step();
        c.dut_set_rst_n(self.handle, 1);
        self.step();
    }
};

fn configure(dut: *Dut, rows: u32, tokens: u3) !void {
    try std.testing.expect(c.dut_config_ready(dut.handle) != 0);
    c.dut_set_config(dut.handle, 1, @intCast(rows), tokens);
    dut.eval();
    try std.testing.expect(c.dut_config_ready(dut.handle) != 0);
    dut.step();
    c.dut_set_config(dut.handle, 0, 0, 0);
    dut.eval();
}

fn readResult(dut: *Dut) Result {
    return .{
        .token = @truncate(c.dut_result_token(dut.handle)),
        .max_exp = c.dut_result_max_exp(dut.handle),
        .rows = @truncate(c.dut_result_rows(dut.handle)),
        .subnormal_warning = c.dut_result_subnormal_warning(dut.handle) != 0,
        .final = c.dut_result_final(dut.handle) != 0,
    };
}

fn groupAt(values: []const u32, group_index: usize) [8]u32 {
    var lanes: [8]u32 = undefined;
    @memcpy(&lanes, values[group_index * 8 ..][0..8]);
    return lanes;
}

fn runSuccess(
    dut: *Dut,
    values: []const u32,
    rows: u32,
    tokens: u3,
    random_stalls: bool,
    seed: u64,
) !RunStats {
    const token_count: usize = tokens;
    try std.testing.expectEqual(@as(usize, rows) * token_count, values.len);

    var expected: [4]section.RmsNormMaxExpResult = undefined;
    for (0..token_count) |token| {
        expected[token] = try section.rmsNormMaxExp(
            values[token * rows ..][0..rows],
            rows,
        );
    }

    try configure(dut, rows, tokens);
    try std.testing.expect(c.dut_busy(dut.handle) != 0);

    var prng = std.Random.DefaultPrng.init(seed);
    const rnd = prng.random();
    const total_groups = values.len / 8;
    var sent_groups: usize = 0;
    var received: usize = 0;
    var presenting = false;
    var held: ?Result = null;
    var cycle: usize = 0;
    var group_stalls: usize = 0;
    var result_stalls: usize = 0;

    while (received < token_count) : (cycle += 1) {
        if (cycle > total_groups * 6 + 1024) return error.StreamTimeout;

        if (!presenting and sent_groups < total_groups)
            presenting = !random_stalls or rnd.uintLessThan(u8, 4) != 0;
        const lanes = if (presenting) groupAt(values, sent_groups) else zero_group;
        c.dut_set_group(
            dut.handle,
            @intFromBool(presenting),
            &lanes,
            0,
            @intFromBool(presenting and sent_groups + 1 == total_groups),
        );
        var result_ready = !random_stalls or rnd.uintLessThan(u8, 3) != 0;
        c.dut_set_result_ready(dut.handle, @intFromBool(result_ready));
        dut.eval();
        if (random_stalls and c.dut_result_valid(dut.handle) != 0 and result_stalls == 0) {
            result_ready = false;
            c.dut_set_result_ready(dut.handle, 0);
            dut.eval();
        }

        const group_fire = presenting and c.dut_group_ready(dut.handle) != 0;
        const result_valid = c.dut_result_valid(dut.handle) != 0;
        const result_fire = result_valid and result_ready;
        if (presenting and !group_fire) group_stalls += 1;
        if (result_valid and !result_ready) {
            result_stalls += 1;
            const payload = readResult(dut);
            if (held) |prior| try std.testing.expectEqual(prior, payload);
            held = payload;
        } else {
            held = null;
        }

        if (result_fire) {
            const got = readResult(dut);
            try std.testing.expectEqual(@as(u2, @intCast(received)), got.token);
            try std.testing.expectEqual(expected[received].max_exp, got.max_exp);
            try std.testing.expectEqual(@as(u14, @intCast(rows)), got.rows);
            try std.testing.expectEqual(
                expected[received].subnormal_warning,
                got.subnormal_warning,
            );
            try std.testing.expectEqual(received + 1 == token_count, got.final);
            received += 1;
        }
        if (group_fire) {
            sent_groups += 1;
            presenting = false;
        }

        dut.step();
        try std.testing.expect(c.dut_error(dut.handle) == 0);
        if (result_fire and received == token_count) {
            try std.testing.expect(c.dut_done(dut.handle) != 0);
            try std.testing.expect(c.dut_busy(dut.handle) == 0);
        } else {
            try std.testing.expect(c.dut_done(dut.handle) == 0);
        }
    }

    try std.testing.expectEqual(total_groups, sent_groups);
    return .{
        .cycles = cycle,
        .group_stalls = group_stalls,
        .result_stalls = result_stalls,
    };
}

fn expectImmediateConfigFault(dut: *Dut, rows: u32, tokens: u3) !void {
    try configure(dut, rows, tokens);
    try std.testing.expect(c.dut_done(dut.handle) != 0);
    try std.testing.expect(c.dut_error(dut.handle) != 0);
    try std.testing.expectEqual(
        section.RmsNormMaxExpStatus.bad_cfg,
        @as(u6, @truncate(c.dut_status(dut.handle))),
    );
    try std.testing.expect(c.dut_busy(dut.handle) == 0);
}

fn expectFirstGroupFault(
    dut: *Dut,
    lanes: [8]u32,
    scratch_error: bool,
    last: bool,
    expected_status: u6,
) !void {
    try configure(dut, 8, 1);
    c.dut_set_group(
        dut.handle,
        1,
        &lanes,
        @intFromBool(scratch_error),
        @intFromBool(last),
    );
    c.dut_set_result_ready(dut.handle, 1);
    dut.eval();
    try std.testing.expect(c.dut_group_ready(dut.handle) != 0);
    dut.step();
    c.dut_set_group(dut.handle, 0, &zero_group, 0, 0);
    dut.eval();
    try std.testing.expect(c.dut_done(dut.handle) == 0);
    dut.step();
    try std.testing.expect(c.dut_done(dut.handle) != 0);
    try std.testing.expect(c.dut_error(dut.handle) != 0);
    try std.testing.expectEqual(
        expected_status,
        @as(u6, @truncate(c.dut_status(dut.handle))),
    );
    try std.testing.expect(c.dut_result_valid(dut.handle) == 0);
}

fn verifyFaultsAndRestart(dut: *Dut) !void {
    try expectImmediateConfigFault(dut, 7, 1);
    dut.step();
    try expectImmediateConfigFault(dut, 4104, 1);
    dut.step();
    try expectImmediateConfigFault(dut, 8, 0);
    dut.step();
    try expectImmediateConfigFault(dut, 8, 5);
    dut.step();

    const ones = [_]u32{0x3f80_0000} ** 8;
    try expectFirstGroupFault(
        dut,
        ones,
        true,
        true,
        section.RmsNormMaxExpStatus.scratch,
    );
    dut.step();
    try expectFirstGroupFault(
        dut,
        ones,
        false,
        false,
        section.RmsNormMaxExpStatus.frame,
    );
    dut.step();
    var nonfinite = ones;
    nonfinite[6] = 0x7fc0_0001;
    try expectFirstGroupFault(
        dut,
        nonfinite,
        false,
        true,
        section.RmsNormMaxExpStatus.nonfinite,
    );
    dut.step();

    // Early TLAST is rejected before the token result can be published.
    try configure(dut, 16, 1);
    c.dut_set_group(dut.handle, 1, &ones, 0, 1);
    c.dut_set_result_ready(dut.handle, 1);
    dut.eval();
    dut.step();
    c.dut_set_group(dut.handle, 0, &zero_group, 0, 0);
    dut.eval();
    dut.step();
    try std.testing.expect(c.dut_error(dut.handle) != 0);
    try std.testing.expectEqual(
        section.RmsNormMaxExpStatus.frame,
        @as(u6, @truncate(c.dut_status(dut.handle))),
    );
    dut.step();

    // A later-token error cannot retract an earlier tentative result.
    try configure(dut, 8, 2);
    c.dut_set_group(dut.handle, 1, &ones, 0, 0);
    c.dut_set_result_ready(dut.handle, 1);
    dut.eval();
    dut.step();
    c.dut_set_group(dut.handle, 0, &zero_group, 0, 0);
    dut.eval();
    dut.step();
    try std.testing.expect(c.dut_result_valid(dut.handle) != 0);
    try std.testing.expectEqual(@as(u8, 127), c.dut_result_max_exp(dut.handle));
    dut.step();
    c.dut_set_group(dut.handle, 1, &nonfinite, 0, 1);
    dut.eval();
    try std.testing.expect(c.dut_group_ready(dut.handle) != 0);
    dut.step();
    c.dut_set_group(dut.handle, 0, &zero_group, 0, 0);
    dut.eval();
    dut.step();
    try std.testing.expect(c.dut_error(dut.handle) != 0);
    try std.testing.expect(c.dut_result_valid(dut.handle) == 0);

    // Abort from input and from a stalled result, then restart without reset.
    dut.step();
    try configure(dut, 16, 1);
    c.dut_set_abort(dut.handle, 1);
    dut.step();
    c.dut_set_abort(dut.handle, 0);
    dut.eval();
    try std.testing.expect(c.dut_config_ready(dut.handle) != 0);
    try std.testing.expect(c.dut_error(dut.handle) == 0);

    try configure(dut, 8, 1);
    c.dut_set_group(dut.handle, 1, &ones, 0, 1);
    c.dut_set_result_ready(dut.handle, 0);
    dut.eval();
    dut.step();
    c.dut_set_group(dut.handle, 0, &zero_group, 0, 0);
    dut.eval();
    dut.step();
    try std.testing.expect(c.dut_result_valid(dut.handle) != 0);
    c.dut_set_abort(dut.handle, 1);
    dut.step();
    c.dut_set_abort(dut.handle, 0);
    dut.eval();
    try std.testing.expect(c.dut_config_ready(dut.handle) != 0);

    const stats = try runSuccess(dut, &ones, 8, 1, false, 0);
    try std.testing.expectEqual(@as(usize, 3), stats.cycles);
}

fn fillRandom(values: []u32, rows: usize, tokens: usize, seed: u64) void {
    var prng = std.Random.DefaultPrng.init(seed);
    const rnd = prng.random();
    for (values) |*value| {
        const exponent = rnd.intRangeAtMost(u8, 1, 254);
        const sign = @as(u32, rnd.int(u1)) << 31;
        value.* = sign | (@as(u32, exponent) << 23) |
            (@as(u32, rnd.int(u23)) & 0x007f_ffff);
    }
    for (0..tokens) |token| {
        values[token * rows] = (@as(u32, 254) << 23) | 0x007f_ffff;
        values[token * rows + 1] = 1;
        values[token * rows + 2] = 0x8000_0000;
    }
}

pub fn main() !void {
    var dut = Dut.init();
    defer dut.deinit();
    dut.reset();

    const directed = [_]u32{
        0x0000_0000,
        0x8000_0000,
        0x0000_0001,
        0x3f80_0000,
        0xc120_0000,
        0x4080_0000,
        0x007f_ffff,
        0xbf00_0000,
    };
    const directed_stats = try runSuccess(&dut, &directed, 8, 1, true, 0x51ca_0001);
    try std.testing.expect(directed_stats.group_stalls > 0 or directed_stats.result_stalls > 0);

    var max_values: [4096 * 4]u32 = undefined;
    fillRandom(&max_values, 4096, 4, 0x4096_0004);
    const max_stats = try runSuccess(&dut, &max_values, 4096, 4, false, 0);
    try std.testing.expectEqual(@as(usize, max_values.len / 8 + 8), max_stats.cycles);

    try verifyFaultsAndRestart(&dut);

    std.debug.print(
        "section RMSNorm max-exp scan: 4096x4 in {d} cycles, " ++
            "one accepted group/cycle within tokens; stalls/faults/abort/restart passed\n",
        .{max_stats.cycles},
    );
}
