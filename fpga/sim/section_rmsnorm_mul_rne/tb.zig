//! Exact finite binary32 RNE multiplier cosim.

const std = @import("std");
const section = @import("shared_section");
const c = @cImport(@cInclude("shim.h"));

const expected_latency: usize = 6;

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

const Coverage = struct {
    zero: usize = 0,
    subnormal: usize = 0,
    normal: usize = 0,
    overflow: usize = 0,

    fn add(self: *Coverage, result: section.RmsNormMulResult) void {
        if (result.status == section.RmsNormMulStatus.overflow) {
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

fn runCase(
    dut: *Dut,
    a: u32,
    b: u32,
    hold_cycles: usize,
    present_busy_noise: bool,
) !section.RmsNormMulResult {
    const expected = section.rmsNormMulRneBits(a, b);
    c.dut_set_abort(dut.handle, 0);
    c.dut_set_result_ready(dut.handle, 0);
    c.dut_set_input(dut.handle, 1, a, b);
    dut.eval();
    try std.testing.expect(c.dut_busy(dut.handle) == 0);
    try std.testing.expect(c.dut_input_ready(dut.handle) != 0);
    try std.testing.expect(c.dut_result_valid(dut.handle) == 0);

    // C0: the sole request is accepted.
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
    try std.testing.expect(c.dut_busy(dut.handle) != 0);
    try std.testing.expect(c.dut_input_ready(dut.handle) == 0);
    try std.testing.expectEqual(expected.bits, c.dut_result_data(dut.handle));
    try std.testing.expectEqual(
        expected.status,
        @as(u2, @truncate(c.dut_result_status(dut.handle))),
    );

    // A stalled result and its status must remain stable. Keeping a second
    // request asserted also proves that the occupied RESULT state cannot take it.
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
    // An asserted busy-time request is not accepted on the retirement edge.
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
        .{ .a = 0x3f80_0000, .b = 0x3f80_0000, .bits = 0x3f80_0000 },
        .{ .a = 0xc000_0000, .b = 0x3f00_0000, .bits = 0xbf80_0000 },
        .{ .a = 0x0000_0000, .b = 0xbf80_0000, .bits = 0x8000_0000 },
        .{ .a = 0x8000_0000, .b = 0xbf80_0000, .bits = 0x0000_0000 },
        .{ .a = 0x3fc0_0000, .b = 0x3f80_0001, .bits = 0x3fc0_0002 },
        .{ .a = 0x3fc0_0000, .b = 0x3f80_0003, .bits = 0x3fc0_0004 },
        .{ .a = 0x3fff_f830, .b = 0x3f80_03e8, .bits = 0x4000_0000 },
        .{ .a = 0x7f7f_ffff, .b = 0x3f80_0000, .bits = 0x7f7f_ffff },
        .{
            .a = 0x7f7f_ffff,
            .b = 0x3f80_0001,
            .bits = 0x7f80_0000,
            .status = section.RmsNormMulStatus.overflow,
        },
        .{
            .a = 0xff7f_ffff,
            .b = 0x3f80_0001,
            .bits = 0xff80_0000,
            .status = section.RmsNormMulStatus.overflow,
        },
        .{ .a = 0x0080_0000, .b = 0x3f00_0000, .bits = 0x0040_0000 },
        .{ .a = 0x0000_0001, .b = 0x3f00_0000, .bits = 0x0000_0000 },
        .{ .a = 0x8000_0001, .b = 0x3f00_0000, .bits = 0x8000_0000 },
        .{ .a = 0x0000_0003, .b = 0x3f00_0000, .bits = 0x0000_0002 },
        .{ .a = 0x0000_0005, .b = 0x3f00_0000, .bits = 0x0000_0002 },
        .{ .a = 0x0000_0001, .b = 0x3f00_0001, .bits = 0x0000_0001 },
        .{ .a = 0x007f_ffff, .b = 0x3f80_0001, .bits = 0x0080_0000 },
        .{ .a = 0x007f_ffff, .b = 0x4000_0000, .bits = 0x00ff_fffe },
        .{ .a = 0x19ff_ffff, .b = 0x1a7f_ffff, .bits = 0x0000_0001 },
        .{ .a = 0x197f_ffff, .b = 0x1a7f_ffff, .bits = 0x0000_0000 },
        .{ .a = 0x997f_ffff, .b = 0x1a7f_ffff, .bits = 0x8000_0000 },
        .{
            .a = 0x5f61_2000,
            .b = 0x5f91_8e00,
            .bits = 0x7f80_0000,
            .status = section.RmsNormMulStatus.overflow,
        },
        .{
            .a = 0x7f80_0000,
            .b = 0x3f80_0000,
            .bits = 0,
            .status = section.RmsNormMulStatus.nonfinite_input,
        },
        .{
            .a = 0,
            .b = 0x7f80_0000,
            .bits = 0,
            .status = section.RmsNormMulStatus.nonfinite_input,
        },
        .{
            .a = 0x7fc1_2345,
            .b = 0xff80_0000,
            .bits = 0,
            .status = section.RmsNormMulStatus.nonfinite_input,
        },
    };

    for (vectors, 0..) |vector, index| {
        const oracle = section.rmsNormMulRneBits(vector.a, vector.b);
        try std.testing.expectEqual(vector.bits, oracle.bits);
        try std.testing.expectEqual(vector.status, oracle.status);
        const swapped = section.rmsNormMulRneBits(vector.b, vector.a);
        try std.testing.expectEqual(oracle.bits, swapped.bits);
        try std.testing.expectEqual(oracle.status, swapped.status);
        _ = try runCase(dut, vector.a, vector.b, index % 4, true);
        _ = try runCase(dut, vector.b, vector.a, 0, false);
    }
    return vectors.len * 2;
}

fn verifyExponentPairs(dut: *Dut) !Coverage {
    var coverage = Coverage{};
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
            const b = ((ea +% eb) & 1) << 31 | (eb << 23) | fraction_b;
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
    const count = 32_768;
    var prng = std.Random.DefaultPrng.init(0x7d31_995e_d94a_620b);
    const rnd = prng.random();
    for (0..count) |index| {
        const exponent_a: u32 = rnd.uintLessThan(u32, 255);
        const exponent_b: u32 = rnd.uintLessThan(u32, 255);
        var a = (rnd.int(u32) & 0x807f_ffff) | (exponent_a << 23);
        var b = (rnd.int(u32) & 0x807f_ffff) | (exponent_b << 23);
        // Regularly force both operands through the subnormal-input path.
        if ((index & 15) == 0) {
            a &= 0x807f_ffff;
            b &= 0x807f_ffff;
            a |= 1;
            b |= 1;
        }
        _ = try runCase(dut, a, b, if ((index & 4095) == 0) 3 else 0, false);
    }
    return count;
}

fn verifyRandomNonfinite(dut: *Dut) !usize {
    const count = 4_096;
    var prng = std.Random.DefaultPrng.init(0x593a_e20d_127c_b806);
    const rnd = prng.random();
    for (0..count) |index| {
        const payload = rnd.int(u32) & 0x807f_ffff;
        const nonfinite = payload | 0x7f80_0000;
        const finite_exp: u32 = rnd.uintLessThan(u32, 255);
        const finite = (rnd.int(u32) & 0x807f_ffff) | (finite_exp << 23);
        const a = if ((index & 1) == 0) nonfinite else finite;
        const b = if ((index & 1) == 0) finite else nonfinite;
        const result = try runCase(dut, a, b, if ((index & 1023) == 0) 2 else 0, false);
        try std.testing.expectEqual(@as(u32, 0), result.bits);
        try std.testing.expectEqual(
            section.RmsNormMulStatus.nonfinite_input,
            result.status,
        );
    }
    return count;
}

fn verifyAbortRestart(dut: *Dut) !void {
    // Abort while idle suppresses ready and leaves no state behind.
    c.dut_set_abort(dut.handle, 1);
    dut.eval();
    try std.testing.expect(c.dut_input_ready(dut.handle) == 0);
    try std.testing.expect(c.dut_result_valid(dut.handle) == 0);
    dut.step();
    c.dut_set_abort(dut.handle, 0);
    dut.eval();
    try std.testing.expect(c.dut_input_ready(dut.handle) != 0);

    // Zero through five post-accept steps target MUL, PIPE, SCAN, SHIFT, FINAL,
    // and a stalled RESULT respectively.
    for (0..6) |delay| {
        c.dut_set_result_ready(dut.handle, 0);
        c.dut_set_input(dut.handle, 1, 0x3fc0_0000, 0xc000_0000);
        dut.eval();
        try std.testing.expect(c.dut_input_ready(dut.handle) != 0);
        dut.step();
        c.dut_set_input(dut.handle, 0, 0, 0);
        for (0..delay) |_| dut.step();
        try std.testing.expectEqual(delay == 5, c.dut_result_valid(dut.handle) != 0);

        c.dut_set_abort(dut.handle, 1);
        dut.eval();
        try std.testing.expect(c.dut_input_ready(dut.handle) == 0);
        try std.testing.expect(c.dut_result_valid(dut.handle) == 0);
        dut.step();
        try std.testing.expect(c.dut_busy(dut.handle) == 0);
        try std.testing.expect(c.dut_result_valid(dut.handle) == 0);
        try std.testing.expectEqual(@as(u32, 0), c.dut_result_data(dut.handle));
        try std.testing.expectEqual(@as(u8, 0), c.dut_result_status(dut.handle));
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
    try verifyAbortRestart(&dut);

    std.debug.print(
        "section RMSNorm RNE multiplier: {d} directed, 65025 finite exponent " ++
            "pairs, {d} random mantissas, {d} random nonfinite; " ++
            "classes z={d} s={d} n={d} o={d}; " ++
            "C0->valid {d} cycles, hold and abort/restart passed\n",
        .{
            directed,
            random,
            random_nonfinite,
            exponent_coverage.zero,
            exponent_coverage.subnormal,
            exponent_coverage.normal,
            exponent_coverage.overflow,
            expected_latency,
        },
    );
}
