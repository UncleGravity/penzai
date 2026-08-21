//! Exact finite binary32 RNE residual-adder cosim.

const std = @import("std");
const section = @import("shared_section");
const c = @cImport(@cInclude("shim.h"));

const expected_latency: usize = 14;

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
        c.dut_set_abort(self.handle, 0);
        c.dut_set_input(self.handle, 0, 0, 0);
        c.dut_set_result_ready(self.handle, 0);
        self.eval();
        for (0..4) |_| self.step();
        c.dut_set_rst_n(self.handle, 1);
        self.step();
    }
};

const Vector = struct {
    a: u32,
    b: u32,
    bits: u32,
    status: u2 = 0,
};

const State = struct {
    const align16: u8 = 1;
    const align8: u8 = 2;
    const align4: u8 = 3;
    const align2: u8 = 4;
    const align1: u8 = 5;
    const add: u8 = 6;
    const norm16: u8 = 7;
    const norm8: u8 = 8;
    const norm4: u8 = 9;
    const norm2: u8 = 10;
    const norm1: u8 = 11;
    const round: u8 = 12;
    const final: u8 = 13;
    const result: u8 = 14;
};

fn effectiveExponent(bits: u32) u8 {
    const exponent: u8 = @truncate(bits >> 23);
    return if (exponent == 0) 1 else exponent;
}

fn significand(bits: u32) u32 {
    const exponent: u8 = @truncate(bits >> 23);
    const hidden: u32 = if (exponent == 0) 0 else 0x0080_0000;
    return (bits & 0x007f_ffff) | hidden;
}

fn shiftRightJamStep28(value: u32, amount: u5) u32 {
    const discarded_mask = (@as(u32, 1) << amount) - 1;
    return (value >> amount) | @intFromBool((value & discarded_mask) != 0);
}

fn applyNormalizeStep(magnitude: *u32, exponent: *u16, amount: u5) void {
    if (amount == 16 and (magnitude.* & 0x0800_0000) != 0) {
        magnitude.* = shiftRightJamStep28(magnitude.*, 1);
        exponent.* += 1;
        return;
    }

    const top_shift: u5 = @intCast(27 - @as(u6, amount));
    const top_mask = (@as(u32, 1) << amount) - 1;
    const top_zero = ((magnitude.* >> top_shift) & top_mask) == 0;
    if (top_zero and exponent.* > amount) {
        magnitude.* = (magnitude.* << amount) & 0x0fff_ffff;
        exponent.* -= amount;
    }
}

const Coverage = struct {
    zero: usize = 0,
    subnormal: usize = 0,
    normal: usize = 0,
    overflow: usize = 0,

    fn add(self: *Coverage, result: section.ResidualAddResult) void {
        if (result.status == section.ResidualAddArithmeticStatus.overflow) {
            self.overflow += 1;
        } else if ((result.bits & 0x7fff_ffff) == 0) {
            self.zero += 1;
        } else if ((result.bits & 0x7f80_0000) == 0) {
            self.subnormal += 1;
        } else {
            self.normal += 1;
        }
    }
};

fn nativeFiniteAdd(a: u32, b: u32) section.ResidualAddResult {
    const a_float: f32 = @bitCast(a);
    const b_float: f32 = @bitCast(b);
    const bits: u32 = @bitCast(a_float + b_float);
    return .{
        .bits = bits,
        .status = if ((bits & 0x7f80_0000) == 0x7f80_0000)
            section.ResidualAddArithmeticStatus.overflow
        else
            0,
    };
}

fn runCase(
    dut: *Dut,
    a: u32,
    b: u32,
    hold_cycles: usize,
    present_busy_noise: bool,
) !section.ResidualAddResult {
    const expected = section.residualAddRneBits(a, b);
    c.dut_set_abort(dut.handle, 0);
    c.dut_set_result_ready(dut.handle, 0);
    c.dut_set_input(dut.handle, 1, a, b);
    dut.eval();
    try std.testing.expect(c.dut_busy(dut.handle) == 0);
    try std.testing.expect(c.dut_input_ready(dut.handle) != 0);
    try std.testing.expect(c.dut_result_valid(dut.handle) == 0);

    dut.step();
    var elapsed: usize = 1;
    c.dut_set_input(
        dut.handle,
        @intFromBool(present_busy_noise),
        a ^ 0xa5a5_5a5a,
        b ^ 0x5a5a_a5a5,
    );
    dut.eval();

    while (c.dut_result_valid(dut.handle) == 0) {
        if (elapsed >= expected_latency) return error.BadResultLatency;
        try std.testing.expect(c.dut_busy(dut.handle) != 0);
        try std.testing.expect(c.dut_input_ready(dut.handle) == 0);
        dut.step();
        elapsed += 1;
    }

    try std.testing.expectEqual(expected_latency, elapsed);
    try std.testing.expectEqual(expected.bits, c.dut_result_data(dut.handle));
    try std.testing.expectEqual(
        expected.status,
        @as(u2, @truncate(c.dut_result_status(dut.handle))),
    );

    for (0..hold_cycles) |_| {
        c.dut_set_result_ready(dut.handle, 0);
        dut.eval();
        try std.testing.expect(c.dut_result_valid(dut.handle) != 0);
        try std.testing.expectEqual(expected.bits, c.dut_result_data(dut.handle));
        try std.testing.expectEqual(
            expected.status,
            @as(u2, @truncate(c.dut_result_status(dut.handle))),
        );
        dut.step();
    }

    c.dut_set_result_ready(dut.handle, 1);
    dut.eval();
    try std.testing.expect(c.dut_result_valid(dut.handle) != 0);
    dut.step();
    try std.testing.expect(c.dut_busy(dut.handle) == 0);
    try std.testing.expect(c.dut_result_valid(dut.handle) == 0);
    try std.testing.expectEqual(@as(u32, 0), c.dut_result_data(dut.handle));
    try std.testing.expectEqual(@as(u8, 0), c.dut_result_status(dut.handle));
    c.dut_set_input(dut.handle, 0, 0, 0);
    c.dut_set_result_ready(dut.handle, 0);
    dut.eval();
    try std.testing.expect(c.dut_input_ready(dut.handle) != 0);
    return expected;
}

fn verifyDirected(dut: *Dut) !usize {
    const vectors = [_]Vector{
        .{ .a = 0x3f80_0000, .b = 0x3f80_0000, .bits = 0x4000_0000 },
        .{ .a = 0xc000_0000, .b = 0x3f00_0000, .bits = 0xbfc0_0000 },
        .{ .a = 0, .b = 0xbf80_0000, .bits = 0xbf80_0000 },
        .{ .a = 0x8000_0000, .b = 0x8000_0000, .bits = 0x8000_0000 },
        .{ .a = 0, .b = 0x8000_0000, .bits = 0 },
        .{ .a = 0x3f80_0000, .b = 0xbf80_0000, .bits = 0 },
        // Halfway at 1.0: an even low bit stays, an odd low bit rounds up.
        .{ .a = 0x3f80_0000, .b = 0x3380_0000, .bits = 0x3f80_0000 },
        .{ .a = 0x3f80_0001, .b = 0x3380_0000, .bits = 0x3f80_0002 },
        .{ .a = 0x3f80_0000, .b = 0x3380_0001, .bits = 0x3f80_0001 },
        .{ .a = 0x7f7f_ffff, .b = 0x3f80_0000, .bits = 0x7f7f_ffff },
        .{
            .a = 0x7f7f_ffff,
            .b = 0x7f7f_ffff,
            .bits = 0x7f80_0000,
            .status = section.ResidualAddArithmeticStatus.overflow,
        },
        .{ .a = 0xff7f_ffff, .b = 0xff7f_ffff, .bits = 0xff80_0000, .status = section.ResidualAddArithmeticStatus.overflow },
        .{ .a = 0x7f7f_ffff, .b = 0xff7f_ffff, .bits = 0 },
        .{ .a = 0x0080_0000, .b = 0x807f_ffff, .bits = 0x0000_0001 },
        .{ .a = 0x0000_0001, .b = 0x0000_0001, .bits = 0x0000_0002 },
        .{ .a = 0x0000_0001, .b = 0x8000_0001, .bits = 0 },
        .{ .a = 0x007f_ffff, .b = 0x0000_0001, .bits = 0x0080_0000 },
        .{ .a = 0x007f_ffff, .b = 0x007f_ffff, .bits = 0x00ff_fffe },
        .{ .a = 0x3f80_0000, .b = 0xb300_0000, .bits = 0x3f80_0000 },
        .{ .a = 0x3f80_0000, .b = 0xb300_0001, .bits = 0x3f7f_ffff },
        // Three low bits are required when subtraction renormalizes a jammed
        // tail. A 24+2 implementation incorrectly returns 0x81e0_0000.
        .{ .a = 0x8200_0000, .b = 0x0080_0003, .bits = 0x81df_ffff },
        .{
            .a = 0x7f80_0000,
            .b = 0x3f80_0000,
            .bits = 0,
            .status = section.ResidualAddArithmeticStatus.nonfinite_input,
        },
        .{
            .a = 0x7fc1_2345,
            .b = 0xff80_0000,
            .bits = 0,
            .status = section.ResidualAddArithmeticStatus.nonfinite_input,
        },
    };

    for (vectors, 0..) |vector, index| {
        const oracle = section.residualAddRneBits(vector.a, vector.b);
        try std.testing.expectEqual(vector.bits, oracle.bits);
        try std.testing.expectEqual(vector.status, oracle.status);
        const swapped = section.residualAddRneBits(vector.b, vector.a);
        try std.testing.expectEqual(oracle.bits, swapped.bits);
        try std.testing.expectEqual(oracle.status, swapped.status);
        _ = try runCase(dut, vector.a, vector.b, index % 4, true);
        _ = try runCase(dut, vector.b, vector.a, 0, false);
    }
    return vectors.len * 2;
}

fn verifyExponentPairs(dut: *Dut) !Coverage {
    var coverage = Coverage{};
    coverage.add(try runCase(dut, 0x3f80_0000, 0xbf80_0000, 0, false));
    coverage.add(try runCase(dut, 0x0080_0000, 0x807f_ffff, 0, false));
    coverage.add(try runCase(dut, 0x3f80_0000, 0x3f80_0000, 0, false));
    coverage.add(try runCase(dut, 0x7f7f_ffff, 0x7f7f_ffff, 0, false));
    for (0..255) |exponent_a| {
        for (0..255) |exponent_b| {
            const ea: u32 = @intCast(exponent_a);
            const eb: u32 = @intCast(exponent_b);
            var fraction_a = ((eb *% 0x45d9_f3b) +% (ea *% 0x119d_e1f)) &
                0x007f_ffff;
            var fraction_b = ((ea *% 0x27d4_eb3) +% (eb *% 0x1656_67b)) &
                0x007f_ffff;
            if (ea == 0) fraction_a |= 1;
            if (eb == 0) fraction_b |= 1;
            const a = ((ea ^ eb) & 1) << 31 | (ea << 23) | fraction_a;
            var b = ((ea +% eb) & 1) << 31 | (eb << 23) | fraction_b;
            if (ea == eb and (ea & 63) == 0)
                b = a ^ 0x8000_0000;
            const oracle = section.residualAddRneBits(a, b);
            try std.testing.expectEqual(nativeFiniteAdd(a, b), oracle);
            coverage.add(try runCase(dut, a, b, 0, false));
        }
    }
    try std.testing.expect(coverage.zero != 0);
    try std.testing.expect(coverage.subnormal != 0);
    try std.testing.expect(coverage.normal != 0);
    try std.testing.expect(coverage.overflow != 0);
    return coverage;
}

fn verifyRandomMantissas(dut: *Dut) !usize {
    const count = 65_536;
    var prng = std.Random.DefaultPrng.init(0x6a4d_e316_c81f_0952);
    const rnd = prng.random();
    for (0..count) |index| {
        const exponent_a: u32 = rnd.uintLessThan(u32, 255);
        const exponent_b: u32 = rnd.uintLessThan(u32, 255);
        var a = (rnd.int(u32) & 0x807f_ffff) | (exponent_a << 23);
        var b = (rnd.int(u32) & 0x807f_ffff) | (exponent_b << 23);
        if ((index & 15) == 0) {
            a &= 0x807f_ffff;
            b &= 0x807f_ffff;
            a |= 1;
            b |= 1;
        }
        try std.testing.expectEqual(
            nativeFiniteAdd(a, b),
            section.residualAddRneBits(a, b),
        );
        _ = try runCase(dut, a, b, if ((index & 8191) == 0) 3 else 0, false);
    }
    return count;
}

fn verifyRandomNonfinite(dut: *Dut) !usize {
    const count = 4_096;
    var prng = std.Random.DefaultPrng.init(0xa042_079d_ee91_753b);
    const rnd = prng.random();
    for (0..count) |index| {
        const nonfinite = (rnd.int(u32) & 0x807f_ffff) | 0x7f80_0000;
        const finite_exp: u32 = rnd.uintLessThan(u32, 255);
        const finite = (rnd.int(u32) & 0x807f_ffff) | (finite_exp << 23);
        const a = if ((index & 1) == 0) nonfinite else finite;
        const b = if ((index & 1) == 0) finite else nonfinite;
        const result = try runCase(dut, a, b, if ((index & 1023) == 0) 2 else 0, false);
        try std.testing.expectEqual(@as(u32, 0), result.bits);
        try std.testing.expectEqual(
            section.ResidualAddArithmeticStatus.nonfinite_input,
            result.status,
        );
    }
    return count;
}

fn traceShiftCase(dut: *Dut, a: u32, b: u32) !void {
    const a_is_big = (a & 0x7fff_ffff) >= (b & 0x7fff_ffff);
    const big = if (a_is_big) a else b;
    const small = if (a_is_big) b else a;
    const exponent_difference = effectiveExponent(big) - effectiveExponent(small);
    const distance: u5 = if (exponent_difference >= 27)
        27
    else
        @intCast(exponent_difference);
    var expected_small = significand(small) << 3;

    c.dut_set_abort(dut.handle, 0);
    c.dut_set_result_ready(dut.handle, 0);
    c.dut_set_input(dut.handle, 1, a, b);
    dut.eval();
    try std.testing.expect(c.dut_input_ready(dut.handle) != 0);
    dut.step();
    c.dut_set_input(dut.handle, 0, 0, 0);
    try std.testing.expectEqual(State.align16, c.dut_debug_state(dut.handle));
    try std.testing.expectEqual(distance, c.dut_debug_align_distance(dut.handle));

    const align_states = [_]u8{
        State.align16,
        State.align8,
        State.align4,
        State.align2,
        State.align1,
    };
    const amounts = [_]u5{ 16, 8, 4, 2, 1 };
    for (align_states, amounts) |state, amount| {
        try std.testing.expectEqual(state, c.dut_debug_state(dut.handle));
        try std.testing.expectEqual(expected_small, c.dut_debug_small_ext(dut.handle));
        if ((distance & amount) != 0)
            expected_small = shiftRightJamStep28(expected_small, amount);
        dut.step();
        try std.testing.expectEqual(expected_small, c.dut_debug_small_ext(dut.handle));
    }

    try std.testing.expectEqual(State.add, c.dut_debug_state(dut.handle));
    dut.step();
    try std.testing.expectEqual(State.norm16, c.dut_debug_state(dut.handle));
    var expected_magnitude = c.dut_debug_magnitude(dut.handle);
    var expected_exponent = c.dut_debug_exponent(dut.handle);

    const norm_states = [_]u8{
        State.norm16,
        State.norm8,
        State.norm4,
        State.norm2,
        State.norm1,
    };
    for (norm_states, amounts) |state, amount| {
        try std.testing.expectEqual(state, c.dut_debug_state(dut.handle));
        applyNormalizeStep(&expected_magnitude, &expected_exponent, amount);
        dut.step();
        try std.testing.expectEqual(expected_magnitude, c.dut_debug_magnitude(dut.handle));
        try std.testing.expectEqual(expected_exponent, c.dut_debug_exponent(dut.handle));
    }

    try std.testing.expectEqual(State.round, c.dut_debug_state(dut.handle));
    dut.step();
    try std.testing.expectEqual(State.final, c.dut_debug_state(dut.handle));
    dut.step();
    try std.testing.expectEqual(State.result, c.dut_debug_state(dut.handle));
    const expected = section.residualAddRneBits(a, b);
    try std.testing.expectEqual(expected.bits, c.dut_result_data(dut.handle));
    try std.testing.expectEqual(expected.status, @as(u2, @truncate(c.dut_result_status(dut.handle))));

    c.dut_set_result_ready(dut.handle, 1);
    dut.step();
    c.dut_set_result_ready(dut.handle, 0);
    try std.testing.expect(c.dut_input_ready(dut.handle) != 0);
}

fn verifyShiftStages(dut: *Dut) !usize {
    // d=27 activates ALIGN16/8/2/1 and repeatedly preserves sticky; d=4
    // activates the remaining alignment stage. The cancellation vectors cover
    // NORM16/8/4/2/1, including a long exact left normalization.
    const vectors = [_][2]u32{
        .{ 0x4d00_0000, 0x3f80_0005 },
        .{ 0x4180_0000, 0x3f80_0005 },
        .{ 0x3f80_0001, 0xbf80_0000 },
        .{ 0x3f80_8000, 0xbf80_0000 },
    };
    for (vectors) |vector| try traceShiftCase(dut, vector[0], vector[1]);
    return vectors.len;
}

fn verifyCancellationBoundaries(dut: *Dut) !usize {
    var count: usize = 0;

    // Adjacent same-exponent magnitudes force normalization from every normal
    // exponent down toward the subnormal boundary.
    for (1..255) |exponent| {
        const exponent_bits: u32 = @as(u32, @intCast(exponent)) << 23;
        for (1..9) |delta| {
            const a = exponent_bits | @as(u32, @intCast(delta));
            const b = 0x8000_0000 | exponent_bits;
            _ = try runCase(dut, a, b, 0, false);
            _ = try runCase(dut, b, a, 0, false);
            count += 2;
        }
    }

    // A power-of-two minus a smaller operand at each jammed distance exercises
    // the one-bit renormalization case for which G/R/S must retain round-to-odd.
    const fractions = [_]u32{
        0, 1, 2, 3, 0x003f_ffff, 0x007f_fffd, 0x007f_fffe, 0x007f_ffff,
    };
    for (4..28) |distance| {
        const big = @as(u32, @intCast(127 + distance)) << 23;
        for (fractions) |fraction| {
            const small = 0xbf80_0000 | fraction;
            _ = try runCase(dut, big, small, 0, false);
            _ = try runCase(dut, small, big, 0, false);
            count += 2;
        }
    }
    return count;
}

fn verifySubnormalBoundaries(dut: *Dut) !usize {
    var count: usize = 0;
    for (0..257) |offset_usize| {
        const offset: u32 = @intCast(offset_usize);
        const low = offset;
        const high_subnormal = 0x007f_ffff - offset;
        const low_normal = 0x0080_0000 + offset;
        const vectors = [_][2]u32{
            .{ low, low },
            .{ high_subnormal, low },
            .{ high_subnormal, 0x8000_0000 | low },
            .{ low_normal, 0x8000_0000 | high_subnormal },
            .{ 0x8000_0000 | low_normal, high_subnormal },
            .{ 0x8000_0000 | high_subnormal, 0x8000_0000 | low },
        };
        for (vectors) |vector| {
            _ = try runCase(dut, vector[0], vector[1], 0, false);
            count += 1;
        }
    }
    return count;
}

fn verifyAbortRestart(dut: *Dut) !void {
    c.dut_set_abort(dut.handle, 1);
    dut.eval();
    try std.testing.expect(c.dut_input_ready(dut.handle) == 0);
    try std.testing.expect(c.dut_result_valid(dut.handle) == 0);
    dut.step();
    c.dut_set_abort(dut.handle, 0);
    dut.eval();
    try std.testing.expect(c.dut_input_ready(dut.handle) != 0);

    for (0..expected_latency) |delay| {
        c.dut_set_result_ready(dut.handle, 0);
        c.dut_set_input(dut.handle, 1, 0x3fc0_0000, 0xc000_0000);
        dut.eval();
        try std.testing.expect(c.dut_input_ready(dut.handle) != 0);
        dut.step();
        c.dut_set_input(dut.handle, 0, 0, 0);
        for (0..delay) |_| dut.step();
        try std.testing.expectEqual(
            delay + 1 == expected_latency,
            c.dut_result_valid(dut.handle) != 0,
        );

        c.dut_set_abort(dut.handle, 1);
        dut.eval();
        try std.testing.expect(c.dut_input_ready(dut.handle) == 0);
        try std.testing.expect(c.dut_result_valid(dut.handle) == 0);
        dut.step();
        try std.testing.expect(c.dut_busy(dut.handle) == 0);
        c.dut_set_abort(dut.handle, 0);
        dut.eval();
        try std.testing.expect(c.dut_input_ready(dut.handle) != 0);

        _ = try runCase(dut, 0x3fc0_0000, 0xc000_0000, delay, true);
    }
}

pub fn main() !void {
    var dut = Dut.init();
    defer dut.deinit();
    dut.reset();

    const directed = try verifyDirected(&dut);
    const exponent_coverage = try verifyExponentPairs(&dut);
    const random = try verifyRandomMantissas(&dut);
    const random_nonfinite = try verifyRandomNonfinite(&dut);
    const shift_traces = try verifyShiftStages(&dut);
    const cancellation = try verifyCancellationBoundaries(&dut);
    const subnormal_boundaries = try verifySubnormalBoundaries(&dut);
    try verifyAbortRestart(&dut);

    std.debug.print(
        "section residual RNE adder: {d} directed, 65025 exponent pairs, " ++
            "{d} random mantissas, {d} random nonfinite; " ++
            "{d} stage traces, {d} cancellation and {d} subnormal-boundary; " ++
            "classes z={d} s={d} n={d} o={d}; C0->valid {d} cycles, " ++
            "hold and 14-position abort/restart passed\n",
        .{
            directed,
            random,
            random_nonfinite,
            shift_traces,
            cancellation,
            subnormal_boundaries,
            exponent_coverage.zero,
            exponent_coverage.subnormal,
            exponent_coverage.normal,
            exponent_coverage.overflow,
            expected_latency,
        },
    );
}
