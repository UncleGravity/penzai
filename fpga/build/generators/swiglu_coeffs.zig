const std = @import("std");

pub const segment_count: usize = 1024;
pub const domain_min: f32 = -16.0;
pub const domain_max: f32 = 16.0;
pub const segment_step: f32 = 1.0 / 32.0;

pub const Coeff = struct {
    slope_bits: u32,
    intercept_bits: u32,
};

fn silu64(x: f64) f64 {
    return x / (1.0 + @exp(-x));
}

/// Coefficients are generated from binary32-rounded endpoint values. The slope,
/// slope*x0, and intercept are each rounded to binary32 in this exact order.
pub fn coefficient(index: usize) Coeff {
    std.debug.assert(index < segment_count);
    const x0: f32 = domain_min + @as(f32, @floatFromInt(index)) * segment_step;
    const x1: f32 = x0 + segment_step;
    const f0: f32 = @floatCast(silu64(x0));
    const f1: f32 = @floatCast(silu64(x1));
    const slope: f32 = (f1 - f0) / segment_step;
    const slope_x0: f32 = slope * x0;
    const intercept: f32 = f0 - slope_x0;
    return .{ .slope_bits = @bitCast(slope), .intercept_bits = @bitCast(intercept) };
}

pub fn coefficientHash() u64 {
    var hash: u64 = 0xcbf29ce484222325;
    for (0..segment_count) |index| {
        const c = coefficient(index);
        inline for (.{ c.slope_bits, c.intercept_bits }) |word| {
            inline for (0..4) |byte_index| {
                hash ^= @as(u8, @truncate(word >> @intCast(byte_index * 8)));
                hash *%= 0x100000001b3;
            }
        }
    }
    return hash;
}

test "coefficient table is finite and covers every segment" {
    for (0..segment_count) |index| {
        const c = coefficient(index);
        const slope: f32 = @bitCast(c.slope_bits);
        const intercept: f32 = @bitCast(c.intercept_bits);
        try std.testing.expect(std.math.isFinite(slope));
        try std.testing.expect(std.math.isFinite(intercept));
    }
    try std.testing.expectEqual(@as(u64, 0x3c259838fbe85a1b), coefficientHash());
}
