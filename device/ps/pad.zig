const std = @import("std");

pub const PadError = error{
    InvalidLength,
};

/// Trailing zero-pad of a contiguous f32 tensor (ggml `GGML_OP_PAD`, right side).
///
/// The host only lowers pads where every left pad is zero, dim0 is unchanged, and
/// dims 2/3 are singletons (see `supportsPad` in host/lower.zig). Under those
/// constraints the padded tensor is exactly the source bytes followed by an
/// appended zero region, so the kernel is "copy src to the front of dst, zero the
/// rest". Validating only the byte lengths keeps the device side cheap and total.
pub fn padZeroTailBytes(src: []const u8, dst: []u8) PadError!void {
    if (src.len % @sizeOf(f32) != 0 or dst.len % @sizeOf(f32) != 0) return error.InvalidLength;
    if (dst.len < src.len) return error.InvalidLength;
    @memcpy(dst[0..src.len], src);
    @memset(dst[src.len..], 0);
}

test "pad copies source then zero-fills the tail" {
    var src: [8]u8 = undefined;
    std.mem.writeInt(u32, src[0..4], @bitCast(@as(f32, 1.5)), .little);
    std.mem.writeInt(u32, src[4..8], @bitCast(@as(f32, -2.0)), .little);
    var dst: [16]u8 = undefined;
    @memset(&dst, 0xaa); // poison so we can see the zero-fill happen

    try padZeroTailBytes(&src, &dst);

    try std.testing.expectEqual(@as(f32, 1.5), @as(f32, @bitCast(std.mem.readInt(u32, dst[0..4], .little))));
    try std.testing.expectEqual(@as(f32, -2.0), @as(f32, @bitCast(std.mem.readInt(u32, dst[4..8], .little))));
    try std.testing.expectEqual(@as(u32, 0), std.mem.readInt(u32, dst[8..12], .little));
    try std.testing.expectEqual(@as(u32, 0), std.mem.readInt(u32, dst[12..16], .little));
}

test "pad with no padding is a plain copy" {
    var src: [4]u8 = undefined;
    std.mem.writeInt(u32, src[0..4], @bitCast(@as(f32, 3.25)), .little);
    var dst: [4]u8 = undefined;
    try padZeroTailBytes(&src, &dst);
    try std.testing.expectEqual(@as(f32, 3.25), @as(f32, @bitCast(std.mem.readInt(u32, dst[0..4], .little))));
}

test "pad rejects dst smaller than src or misaligned lengths" {
    var src: [8]u8 = undefined;
    var small: [4]u8 = undefined;
    try std.testing.expectError(error.InvalidLength, padZeroTailBytes(&src, &small));
    var odd: [6]u8 = undefined;
    var dst: [8]u8 = undefined;
    try std.testing.expectError(error.InvalidLength, padZeroTailBytes(&odd, &dst));
}
