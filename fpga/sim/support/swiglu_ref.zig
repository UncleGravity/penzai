//! Bit model and numerical characterization for `rtl/section_swiglu.v`.

const std = @import("std");
const coeffs = @import("swiglu_coeffs.zig");

pub const Result = struct {
    bits: u32,
    status: u2,
};

const zero_f32: u32 = 0x0000_0000;
const one_f32: u32 = 0x3f80_0000;
const abs_16: u32 = 0x4180_0000;
const max_finite_abs: u32 = 0x7f7f_ffff;

pub fn truncMulBits(a: u32, b: u32) u32 {
    const sign = (a ^ b) & 0x8000_0000;
    const ea: u32 = (a >> 23) & 0xff;
    const eb: u32 = (b >> 23) & 0xff;
    if (ea == 0 or eb == 0) return 0;

    const siga: u64 = 0x80_0000 | (a & 0x7f_ffff);
    const sigb: u64 = 0x80_0000 | (b & 0x7f_ffff);
    const product = siga * sigb;
    const renormalizes = ((product >> 47) & 1) != 0;
    const mantissa: u32 = @truncate(if (renormalizes)
        (product >> 24) & 0x7f_ffff
    else
        (product >> 23) & 0x7f_ffff);
    const exponent = @as(i32, @intCast(ea)) + @as(i32, @intCast(eb)) - 127 +
        @as(i32, @intFromBool(renormalizes));
    if (exponent <= 0) return sign;
    if (exponent >= 255) return sign | max_finite_abs;
    return sign | (@as(u32, @intCast(exponent)) << 23) | mantissa;
}

pub fn truncAddBits(a: u32, b: u32) u32 {
    const sa = a >> 31;
    const sb = b >> 31;
    const ea: u32 = (a >> 23) & 0xff;
    const eb: u32 = (b >> 23) & 0xff;
    const ma = a & 0x7f_ffff;
    const mb = b & 0x7f_ffff;
    const a_zero = ea == 0;
    const b_zero = eb == 0;

    const a_ge_b = ea >= eb;
    const exp_big = if (a_ge_b) ea else eb;
    const exp_diff = if (a_ge_b) ea - eb else eb - ea;
    const mant_big: u32 = 0x80_0000 | (if (a_ge_b) ma else mb);
    const mant_small: u32 = 0x80_0000 | (if (a_ge_b) mb else ma);
    const sign_big = if (a_ge_b) sa else sb;
    const sign_small = if (a_ge_b) sb else sa;
    const mant_small_aligned = if (exp_diff > 24) 0 else mant_small >> @intCast(exp_diff);
    const small_bigger = mant_small_aligned > mant_big;
    const m1 = if (small_bigger) mant_small_aligned else mant_big;
    const m2 = if (small_bigger) mant_big else mant_small_aligned;
    const result_sign = if (small_bigger) sign_small else sign_big;
    const mant_sum: u32 = if (sign_big == sign_small) m1 + m2 else m1 - m2;

    if (a_zero) return b;
    if (b_zero) return a;
    if (mant_sum == 0) return result_sign << 31;

    const lead_pos: u5 = @intCast(31 - @clz(mant_sum));
    const shift_right = lead_pos > 23;
    const right_amount: u5 = if (shift_right) lead_pos - 23 else 0;
    const left_amount: u5 = if (lead_pos < 23) 23 - lead_pos else 0;
    const exponent = if (shift_right)
        @as(i32, @intCast(exp_big)) + @as(i32, right_amount)
    else
        @as(i32, @intCast(exp_big)) - @as(i32, left_amount);
    if (exponent <= 0) return result_sign << 31;
    if (exponent >= 255) return (result_sign << 31) | max_finite_abs;
    const normalized = if (shift_right)
        mant_sum >> right_amount
    else
        mant_sum << left_amount;
    return (result_sign << 31) | (@as(u32, @intCast(exponent)) << 23) |
        (normalized & 0x7f_ffff);
}

pub fn segmentIndexBits(value: u32) u10 {
    const exponent: u8 = @truncate(value >> 23);
    const sig: u24 = @truncate(0x80_0000 | (value & 0x7f_ffff));
    var mag_floor: u9 = 0;
    var has_fraction = (value & 0x7fff_ffff) != 0;
    switch (exponent) {
        122 => {
            mag_floor = @truncate(sig >> 23);
            has_fraction = (sig & 0x7f_ffff) != 0;
        },
        123 => {
            mag_floor = @truncate(sig >> 22);
            has_fraction = (sig & 0x3f_ffff) != 0;
        },
        124 => {
            mag_floor = @truncate(sig >> 21);
            has_fraction = (sig & 0x1f_ffff) != 0;
        },
        125 => {
            mag_floor = @truncate(sig >> 20);
            has_fraction = (sig & 0x0f_ffff) != 0;
        },
        126 => {
            mag_floor = @truncate(sig >> 19);
            has_fraction = (sig & 0x07_ffff) != 0;
        },
        127 => {
            mag_floor = @truncate(sig >> 18);
            has_fraction = (sig & 0x03_ffff) != 0;
        },
        128 => {
            mag_floor = @truncate(sig >> 17);
            has_fraction = (sig & 0x01_ffff) != 0;
        },
        129 => {
            mag_floor = @truncate(sig >> 16);
            has_fraction = (sig & 0x00_ffff) != 0;
        },
        130 => {
            mag_floor = @truncate(sig >> 15);
            has_fraction = (sig & 0x00_7fff) != 0;
        },
        else => {},
    }
    if (value >> 31 == 0) return @intCast(512 + @as(u10, mag_floor));
    const mag_ceil: u10 = @as(u10, mag_floor) + @as(u10, @intFromBool(has_fraction));
    return @intCast(512 - mag_ceil);
}

pub fn modelBits(gate_bits: u32, up_bits: u32) Result {
    const nonfinite = ((gate_bits >> 23) & 0xff) == 0xff or
        ((up_bits >> 23) & 0xff) == 0xff;
    const gate_abs = gate_bits & 0x7fff_ffff;
    const negative_tail = !nonfinite and gate_bits >> 31 != 0 and gate_abs >= abs_16;
    const positive_tail = !nonfinite and gate_bits >> 31 == 0 and gate_abs >= abs_16;

    const gate = if (nonfinite) zero_f32 else gate_bits;
    const up = if (nonfinite) zero_f32 else up_bits;
    var slope = zero_f32;
    var intercept = zero_f32;
    if (positive_tail) {
        slope = one_f32;
    } else if (!nonfinite and !negative_tail) {
        const c = coeffs.coefficient(segmentIndexBits(gate_bits));
        slope = c.slope_bits;
        intercept = c.intercept_bits;
    }

    const gate_mul = truncMulBits(gate, slope);
    const silu = truncAddBits(gate_mul, intercept);
    const result = truncMulBits(silu, up);
    return .{
        .bits = result,
        .status = @as(u2, @intFromBool(nonfinite)) |
            (@as(u2, @intFromBool((result & 0x7fff_ffff) == max_finite_abs)) << 1),
    };
}

pub fn model(gate: f32, up: f32) Result {
    return modelBits(@bitCast(gate), @bitCast(up));
}

pub fn ideal(gate: f32, up: f32) f32 {
    return gate / (1.0 + @exp(-gate)) * up;
}

test "grid index covers all 1024 segments and both sides of zero" {
    for (0..coeffs.segment_count) |index| {
        const midpoint = coeffs.domain_min +
            (@as(f32, @floatFromInt(index)) + 0.5) * coeffs.segment_step;
        try std.testing.expectEqual(@as(u10, @intCast(index)), segmentIndexBits(@bitCast(midpoint)));
    }
    try std.testing.expectEqual(@as(u10, 512), segmentIndexBits(@bitCast(@as(f32, 0.0))));
    try std.testing.expectEqual(@as(u10, 512), segmentIndexBits(@bitCast(@as(f32, -0.0))));
    try std.testing.expectEqual(@as(u10, 511), segmentIndexBits(@bitCast(@as(f32, -0.0001))));
}

test "PWL plus truncating leaves stays within the committed SiLU bound" {
    var max_abs: f64 = 0;
    var max_x: f32 = 0;
    for (0..coeffs.segment_count) |index| {
        for (0..257) |sample| {
            const x = coeffs.domain_min +
                (@as(f32, @floatFromInt(index)) +
                    @as(f32, @floatFromInt(sample)) / 256.0) * coeffs.segment_step;
            const got: f32 = @bitCast(model(x, 1.0).bits);
            const want = ideal(x, 1.0);
            const abs = @abs(@as(f64, got) - @as(f64, want));
            if (abs > max_abs) {
                max_abs = abs;
                max_x = x;
            }
        }
    }
    std.debug.print("section SwiGLU PWL: max_abs={e:.6} at gate={d:.6}\n", .{ max_abs, max_x });
    try std.testing.expect(max_abs < 1.0e-4);
    try std.testing.expect(@abs(@as(f64, ideal(-16.0, 1.0))) < 2.0e-6);
    try std.testing.expect(@abs(@as(f64, ideal(16.0, 1.0)) - 16.0) < 2.0e-6);
}

test "nonfinite inputs fail closed and tails follow the committed guards" {
    try std.testing.expectEqual(Result{ .bits = 0, .status = 1 }, model(std.math.nan(f32), 1));
    try std.testing.expectEqual(Result{ .bits = 0, .status = 1 }, model(1, std.math.inf(f32)));
    try std.testing.expectEqual(@as(u32, 0), model(-16, 1).bits);
    try std.testing.expectEqual(@as(u32, @bitCast(@as(f32, 17))), model(17, 1).bits);
}
