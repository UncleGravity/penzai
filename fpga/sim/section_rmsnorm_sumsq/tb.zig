//! Bit-exact control/datapath cosim for `section_rmsnorm_sumsq`.
//!
//! Results are tentative until final successful `done`. The test deliberately
//! consumes an earlier token before injecting a later-token failure.

const std = @import("std");
const section = @import("shared_section");
const c = @cImport(@cInclude("shim.h"));

const zero_group = [_]u32{0} ** 8;

const Result = struct {
    token: u2,
    max_exp: u8,
    sum_sq: u48,
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
        c.dut_set_config(self.handle, 0, 0, 0, 0);
        c.dut_set_abort(self.handle, 0);
        c.dut_set_group(self.handle, 0, &zero_group, 0, 0);
        c.dut_set_result_ready(self.handle, 0);
        self.eval();
        for (0..4) |_| self.step();
        c.dut_set_rst_n(self.handle, 1);
        self.step();
    }
};

fn packMaxExp(exponents: [4]u8) u32 {
    return @as(u32, exponents[0]) |
        (@as(u32, exponents[1]) << 8) |
        (@as(u32, exponents[2]) << 16) |
        (@as(u32, exponents[3]) << 24);
}

fn scanMaxExp(values: []const u32) u8 {
    var result: u8 = 0;
    for (values) |bits| {
        const exponent: u8 = @truncate(bits >> 23);
        if (exponent != 0xff and exponent > result) result = exponent;
    }
    return result;
}

fn readResult(dut: *Dut) Result {
    return .{
        .token = @truncate(c.dut_result_token(dut.handle)),
        .max_exp = c.dut_result_max_exp(dut.handle),
        .sum_sq = @truncate(c.dut_result_sum_sq(dut.handle)),
        .rows = @truncate(c.dut_result_rows(dut.handle)),
        .subnormal_warning = c.dut_result_subnormal_warning(dut.handle) != 0,
        .final = c.dut_result_final(dut.handle) != 0,
    };
}

fn configure(dut: *Dut, rows: u32, tokens: u3, max_exp: [4]u8) !void {
    try std.testing.expect(c.dut_config_ready(dut.handle) != 0);
    c.dut_set_config(
        dut.handle,
        1,
        @intCast(rows),
        tokens,
        packMaxExp(max_exp),
    );
    dut.eval();
    try std.testing.expect(c.dut_config_ready(dut.handle) != 0);
    dut.step();
    c.dut_set_config(dut.handle, 0, 0, 0, 0);
    dut.eval();
}

fn groupAt(values: []const u32, group_index: usize) [8]u32 {
    var lanes: [8]u32 = undefined;
    @memcpy(&lanes, values[group_index * 8 ..][0..8]);
    return lanes;
}

fn expectResult(
    got: Result,
    expected: section.RmsNormSumsqResult,
    token: usize,
    rows: u32,
    max_exp: u8,
    tokens: usize,
) !void {
    try std.testing.expectEqual(@as(u2, @intCast(token)), got.token);
    try std.testing.expectEqual(max_exp, got.max_exp);
    try std.testing.expectEqual(expected.sum_sq, got.sum_sq);
    try std.testing.expectEqual(@as(u14, @intCast(rows)), got.rows);
    try std.testing.expectEqual(expected.subnormal_warning, got.subnormal_warning);
    try std.testing.expectEqual(token + 1 == tokens, got.final);
}

fn runSuccess(
    dut: *Dut,
    values: []const u32,
    rows: u32,
    tokens: u3,
    max_exp: [4]u8,
    random_stalls: bool,
    seed: u64,
) !RunStats {
    const token_count: usize = tokens;
    try std.testing.expectEqual(@as(usize, rows) * token_count, values.len);

    var expected: [4]section.RmsNormSumsqResult = undefined;
    for (0..token_count) |token| {
        expected[token] = try section.rmsNormSumsqFixed(
            values[token * rows ..][0..rows],
            rows,
            max_exp[token],
        );
    }

    try configure(dut, rows, tokens, max_exp);
    try std.testing.expect(c.dut_busy(dut.handle) != 0);
    try std.testing.expect(c.dut_error(dut.handle) == 0);

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
        if (cycle > values.len * 4 + 4096) return error.StreamTimeout;

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
        const result_ready = !random_stalls or rnd.uintLessThan(u8, 3) != 0;
        c.dut_set_result_ready(dut.handle, @intFromBool(result_ready));
        dut.eval();

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
            try expectResult(
                readResult(dut),
                expected[received],
                received,
                rows,
                max_exp[received],
                token_count,
            );
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
    c.dut_set_group(dut.handle, 0, &zero_group, 0, 0);
    c.dut_set_result_ready(dut.handle, 0);
    dut.step();
    return .{ .cycles = cycle + 1, .group_stalls = group_stalls, .result_stalls = result_stalls };
}

fn expectFailure(dut: *Dut, status: u7) !void {
    try std.testing.expect(c.dut_done(dut.handle) != 0);
    try std.testing.expect(c.dut_error(dut.handle) != 0);
    try std.testing.expect(c.dut_busy(dut.handle) == 0);
    try std.testing.expect(c.dut_result_valid(dut.handle) == 0);
    try std.testing.expectEqual(@as(u8, status), c.dut_status(dut.handle));
}

fn sendGroup(
    dut: *Dut,
    lanes: [8]u32,
    input_error: bool,
    last: bool,
) !void {
    var cycles: usize = 0;
    while (true) : (cycles += 1) {
        if (cycles > 64) return error.InputTimeout;
        c.dut_set_group(
            dut.handle,
            1,
            &lanes,
            @intFromBool(input_error),
            @intFromBool(last),
        );
        dut.eval();
        if (c.dut_group_ready(dut.handle) != 0) {
            dut.step();
            c.dut_set_group(dut.handle, 0, &zero_group, 0, 0);
            dut.eval();
            return;
        }
        dut.step();
    }
}

fn waitTerminalFailure(dut: *Dut, expected_status: u7) !void {
    for (0..32) |_| {
        if (c.dut_done(dut.handle) != 0)
            return expectFailure(dut, expected_status);
        dut.step();
    }
    return error.FailureTimeout;
}

fn testCadenceAndOutputStability(dut: *Dut) !void {
    const values = [_]u32{0x3f80_0000} ** 16;
    const exponents = [4]u8{ 127, 0, 0, 0 };
    try configure(dut, 16, 1, exponents);
    c.dut_set_result_ready(dut.handle, 0);

    var accept_cycle: [2]usize = undefined;
    var accepted: usize = 0;
    var result_cycle: ?usize = null;
    var cycle: usize = 0;
    while (result_cycle == null) : (cycle += 1) {
        if (cycle > 64) return error.CadenceTimeout;
        const lanes = if (accepted < 2) groupAt(&values, accepted) else zero_group;
        c.dut_set_group(
            dut.handle,
            @intFromBool(accepted < 2),
            &lanes,
            0,
            @intFromBool(accepted == 1),
        );
        dut.eval();
        if (accepted < 2 and c.dut_group_ready(dut.handle) != 0) {
            accept_cycle[accepted] = cycle;
            accepted += 1;
        }
        if (c.dut_result_valid(dut.handle) != 0) result_cycle = cycle;
        if (result_cycle == null) dut.step();
    }
    try std.testing.expectEqual(@as(usize, 11), accept_cycle[1] - accept_cycle[0]);
    try std.testing.expectEqual(@as(usize, 11), result_cycle.? - accept_cycle[1]);

    const expected = try section.rmsNormSumsqFixed(&values, 16, 127);
    const held = readResult(dut);
    try expectResult(held, expected, 0, 16, 127, 1);
    for (0..9) |_| {
        dut.step();
        try std.testing.expect(c.dut_result_valid(dut.handle) != 0);
        try std.testing.expectEqual(held, readResult(dut));
        try std.testing.expect(c.dut_group_ready(dut.handle) == 0);
    }
    c.dut_set_result_ready(dut.handle, 1);
    dut.step();
    try std.testing.expect(c.dut_done(dut.handle) != 0);
    try std.testing.expect(c.dut_busy(dut.handle) == 0);
    c.dut_set_result_ready(dut.handle, 0);
    dut.step();
}

fn testConfigAndInputFaults(dut: *Dut) !void {
    const good_exp = [4]u8{ 127, 127, 127, 127 };
    const bad_configs = [_]struct { rows: u32, tokens: u3, exp: [4]u8 }{
        .{ .rows = 0, .tokens = 1, .exp = good_exp },
        .{ .rows = 7, .tokens = 1, .exp = good_exp },
        .{ .rows = 9, .tokens = 1, .exp = good_exp },
        .{ .rows = 4104, .tokens = 1, .exp = good_exp },
        .{ .rows = 8, .tokens = 0, .exp = good_exp },
        .{ .rows = 8, .tokens = 5, .exp = good_exp },
        .{ .rows = 8, .tokens = 1, .exp = .{ 0xff, 0, 0, 0 } },
    };
    for (bad_configs) |cfg| {
        try configure(dut, cfg.rows, cfg.tokens, cfg.exp);
        try expectFailure(dut, section.RmsNormSumsqStatus.bad_cfg);
        dut.step();
    }

    const ones = [_]u32{0x3f80_0000} ** 8;
    try configure(dut, 8, 1, good_exp);
    try sendGroup(dut, ones, true, true);
    try expectFailure(dut, section.RmsNormSumsqStatus.scratch);
    dut.step();

    try configure(dut, 16, 1, good_exp);
    try sendGroup(dut, ones, false, true);
    try expectFailure(dut, section.RmsNormSumsqStatus.frame);
    dut.step();

    try configure(dut, 8, 1, good_exp);
    try sendGroup(dut, ones, false, false);
    try expectFailure(dut, section.RmsNormSumsqStatus.frame);
    dut.step();

    var nonfinite = ones;
    nonfinite[5] = 0x7f80_0000;
    try configure(dut, 8, 1, good_exp);
    try sendGroup(dut, nonfinite, false, true);
    try waitTerminalFailure(dut, section.RmsNormSumsqStatus.nonfinite);
    dut.step();

    var above_max = ones;
    above_max[2] = 0x4000_0000;
    try configure(dut, 8, 1, good_exp);
    try sendGroup(dut, above_max, false, true);
    try waitTerminalFailure(dut, section.RmsNormSumsqStatus.max_mismatch);
    dut.step();

    try configure(dut, 8, 1, .{ 128, 0, 0, 0 });
    try sendGroup(dut, ones, false, true);
    try waitTerminalFailure(dut, section.RmsNormSumsqStatus.max_mismatch);
    dut.step();

    try configure(dut, 8, 1, .{ 0, 0, 0, 0 });
    try sendGroup(dut, ones, false, true);
    try waitTerminalFailure(dut, section.RmsNormSumsqStatus.max_mismatch);
    dut.step();
}

fn testSubnormalWarning(dut: *Dut) !void {
    const values = [_]u32{ 0, 0x8000_0000, 1, 0x007f_ffff, 0, 0, 0, 0 };
    _ = try runSuccess(dut, &values, 8, 1, .{ 0, 0, 0, 0 }, true, 0x5ab0);
    try std.testing.expect(c.dut_error(dut.handle) == 0);
    try std.testing.expectEqual(
        @as(u8, section.RmsNormSumsqStatus.subnormal_warning),
        c.dut_status(dut.handle),
    );
}

fn testTentativeThenLaterFailure(dut: *Dut) !void {
    const ones = [_]u32{0x3f80_0000} ** 8;
    try configure(dut, 8, 2, .{ 127, 128, 0, 0 });
    c.dut_set_result_ready(dut.handle, 1);
    try sendGroup(dut, ones, false, false);
    for (0..32) |_| {
        dut.eval();
        if (c.dut_result_valid(dut.handle) != 0) break;
        dut.step();
    }
    try std.testing.expect(c.dut_result_valid(dut.handle) != 0);
    const expected = try section.rmsNormSumsqFixed(&ones, 8, 127);
    try expectResult(readResult(dut), expected, 0, 8, 127, 2);
    dut.step();
    try std.testing.expect(c.dut_done(dut.handle) == 0);

    try sendGroup(dut, ones, false, true);
    try waitTerminalFailure(dut, section.RmsNormSumsqStatus.max_mismatch);
    c.dut_set_result_ready(dut.handle, 0);
    dut.step();
}

fn abortAndRestart(dut: *Dut) !void {
    const values = [_]u32{0x3f80_0000} ** 8;
    const exp = [4]u8{ 127, 0, 0, 0 };

    try configure(dut, 8, 1, exp);
    c.dut_set_abort(dut.handle, 1);
    dut.eval();
    try std.testing.expect(c.dut_group_ready(dut.handle) == 0);
    dut.step();
    c.dut_set_abort(dut.handle, 0);
    dut.eval();
    try std.testing.expect(c.dut_config_ready(dut.handle) != 0);
    try std.testing.expect(c.dut_error(dut.handle) == 0);

    try configure(dut, 8, 1, exp);
    try sendGroup(dut, values, false, true);
    for (0..3) |_| dut.step();
    c.dut_set_abort(dut.handle, 1);
    dut.eval();
    try std.testing.expect(c.dut_result_valid(dut.handle) == 0);
    dut.step();
    c.dut_set_abort(dut.handle, 0);
    dut.eval();
    try std.testing.expect(c.dut_config_ready(dut.handle) != 0);

    // sendGroup's acceptance edge enters ST_LANES. Eight further edges issue
    // lanes zero through seven, leaving the explicit final-product drain.
    try configure(dut, 8, 1, exp);
    try sendGroup(dut, values, false, true);
    for (0..8) |_| dut.step();
    try std.testing.expect(c.dut_group_ready(dut.handle) == 0);
    try std.testing.expect(c.dut_result_valid(dut.handle) == 0);
    c.dut_set_abort(dut.handle, 1);
    dut.eval();
    try std.testing.expect(c.dut_result_valid(dut.handle) == 0);
    dut.step();
    c.dut_set_abort(dut.handle, 0);
    dut.eval();
    try std.testing.expect(c.dut_config_ready(dut.handle) != 0);

    try configure(dut, 8, 1, exp);
    c.dut_set_result_ready(dut.handle, 0);
    try sendGroup(dut, values, false, true);
    for (0..32) |_| {
        if (c.dut_result_valid(dut.handle) != 0) break;
        dut.step();
    }
    try std.testing.expect(c.dut_result_valid(dut.handle) != 0);
    c.dut_set_abort(dut.handle, 1);
    dut.eval();
    try std.testing.expect(c.dut_result_valid(dut.handle) == 0);
    dut.step();
    c.dut_set_abort(dut.handle, 0);
    c.dut_set_result_ready(dut.handle, 0);
    dut.eval();
    try std.testing.expect(c.dut_config_ready(dut.handle) != 0);

    _ = try runSuccess(dut, &values, 8, 1, exp, false, 0);
}

fn fillRandomValues(
    values: []u32,
    rows: usize,
    tokens: usize,
    seed: u64,
    min_exp: u8,
    max_exp: u8,
) [4]u8 {
    var prng = std.Random.DefaultPrng.init(seed);
    const rnd = prng.random();
    var exponents = [4]u8{ 0, 0, 0, 0 };
    for (0..tokens) |token| {
        const token_values = values[token * rows ..][0..rows];
        for (token_values) |*bits| {
            const exponent = min_exp + rnd.uintLessThan(u8, max_exp - min_exp + 1);
            const sign = @as(u32, rnd.int(u1)) << 31;
            bits.* = sign | (@as(u32, exponent) << 23) |
                (rnd.int(u32) & 0x007f_ffff);
        }
        // Make the scanner maximum explicit and exercise the maximum mantissa.
        token_values[token % rows] = (@as(u32, max_exp) << 23) | 0x007f_ffff;
        exponents[token] = scanMaxExp(token_values);
    }
    return exponents;
}

fn characterizeApproximation(allocator: std.mem.Allocator) !struct { f64, f64 } {
    var max_adversarial: f64 = 0;
    var max_synthetic_model_shaped: f64 = 0;
    const row_cases = [_]usize{ 8, 128, 2048, 4096 };
    for (row_cases, 0..) |rows, row_case| {
        const values = try allocator.alloc(u32, rows);
        defer allocator.free(values);
        // Deliberately populate every retained alignment d=0..17, then add a
        // minimum-normal tail which truncates completely at this E. Mantissas
        // end in ones to maximize the discarded quantization remainder.
        for (values, 0..) |*bits, index| {
            const exponent: u8 = if (index % 19 == 18)
                1
            else
                @intCast(145 - index % 19);
            bits.* = (@as(u32, exponent) << 23) | 0x007f_ffff;
        }
        values[0] = (@as(u32, 145) << 23) | 0x007f_ffff;
        const constructed_fixed = try section.rmsNormSumsqFixed(values, @intCast(rows), 145);
        const constructed_approximate = section.rmsNormSumsqMeanF64(
            constructed_fixed.sum_sq,
            @intCast(rows),
            145,
        );
        const constructed_exact = section.rmsNormSumsqExactMeanF64(values);
        const constructed_relative = @abs(constructed_approximate - constructed_exact) /
            constructed_exact;
        max_adversarial = @max(max_adversarial, constructed_relative);
        try std.testing.expect(constructed_relative <=
            section.rmsNormSumsqRelativeErrorBound(@intCast(rows)));

        for (0..24) |iteration| {
            const exponents = fillRandomValues(
                values,
                rows,
                1,
                0xa66e_0000 + row_case * 101 + iteration,
                80,
                145,
            );
            const fixed = try section.rmsNormSumsqFixed(values, @intCast(rows), exponents[0]);
            const approximate = section.rmsNormSumsqMeanF64(fixed.sum_sq, @intCast(rows), exponents[0]);
            const exact = section.rmsNormSumsqExactMeanF64(values);
            const relative = @abs(approximate - exact) / exact;
            max_adversarial = @max(max_adversarial, relative);
            try std.testing.expect(relative <= section.rmsNormSumsqRelativeErrorBound(@intCast(rows)));
        }
    }

    const model_rows: usize = 2048;
    const model = try allocator.alloc(u32, model_rows);
    defer allocator.free(model);
    for (0..64) |iteration| {
        const exponents = fillRandomValues(
            model,
            model_rows,
            1,
            0x90de_1000 + iteration,
            115,
            132,
        );
        const fixed = try section.rmsNormSumsqFixed(model, model_rows, exponents[0]);
        const approximate = section.rmsNormSumsqMeanF64(fixed.sum_sq, model_rows, exponents[0]);
        const exact = section.rmsNormSumsqExactMeanF64(model);
        const relative = @abs(approximate - exact) / exact;
        max_synthetic_model_shaped = @max(max_synthetic_model_shaped, relative);
        try std.testing.expect(relative <=
            section.rmsnorm_sumsq_synthetic_model_relative_limit);
    }
    return .{ max_adversarial, max_synthetic_model_shaped };
}

pub fn main() !void {
    var dut = Dut.init();
    defer dut.deinit();
    dut.reset();

    try testCadenceAndOutputStability(&dut);
    try testConfigAndInputFaults(&dut);
    try testSubnormalWarning(&dut);
    try testTentativeThenLaterFailure(&dut);
    try abortAndRestart(&dut);

    const allocator = std.heap.page_allocator;
    const shift_values = try allocator.alloc(u32, 20 * 8);
    defer allocator.free(shift_values);
    for (0..20) |delta| {
        const base = delta * 8;
        @memset(shift_values[base..][0..8], 0);
        shift_values[base] = (@as(u32, 140) << 23) | 0x007f_ffff;
        shift_values[base + 1] = (@as(u32, @intCast(140 - delta)) << 23) |
            0x007f_ffff;
    }
    // Treat each alignment case as one token in a separate run.
    for (0..20) |delta| {
        _ = try runSuccess(
            &dut,
            shift_values[delta * 8 ..][0..8],
            8,
            1,
            .{ 140, 0, 0, 0 },
            true,
            0xd17a + delta,
        );
    }

    const max_values = try allocator.alloc(u32, 4096 * 4);
    defer allocator.free(max_values);
    const max_exp = fillRandomValues(max_values, 4096, 4, 0x4096_4004, 96, 150);
    const max_stats = try runSuccess(&dut, max_values, 4096, 4, max_exp, true, 0x4eed_4004);
    try std.testing.expect(max_stats.group_stalls != 0);
    try std.testing.expect(max_stats.result_stalls != 0);

    const max_finite = try allocator.alloc(u32, 4096);
    defer allocator.free(max_finite);
    @memset(max_finite, 0x7f7f_ffff);
    const max_finite_expected: u48 = @as(u48, 4096) *
        (@as(u48, (1 << 18) - 1) * @as(u48, (1 << 18) - 1));
    try std.testing.expectEqual(
        max_finite_expected,
        (try section.rmsNormSumsqFixed(max_finite, 4096, 254)).sum_sq,
    );
    _ = try runSuccess(&dut, max_finite, 4096, 1, .{ 254, 0, 0, 0 }, false, 0);

    const characterization = try characterizeApproximation(allocator);
    std.debug.print(
        "section RMSNorm sumsq: exact fixed oracle, shifts d=0..19, " ++
            "max 4096x4 completed in {d} cycles\n" ++
            "  cadence: 8 scalar issue cycles/group at within-group II=1, " ++
            "group accept interval 11 cycles (capture/two-drain bubbles)\n" ++
            "  max relative error adversarial={e:.4}, synthetic model-shaped={e:.4}\n",
        .{ max_stats.cycles, characterization[0], characterization[1] },
    );
}
