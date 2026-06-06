const std = @import("std");

pub const KiB: usize = 1024;
pub const MiB: usize = 1024 * KiB;
pub const GiB: usize = 1024 * MiB;

pub fn parse(text: []const u8) !usize {
    const s = std.mem.trim(u8, text, " \t\r\n");
    if (s.len == 0) return error.InvalidSize;

    var digits_len: usize = 0;
    while (digits_len < s.len and std.ascii.isDigit(s[digits_len])) {
        digits_len += 1;
    }
    if (digits_len == 0) return error.InvalidSize;

    const value = try std.fmt.parseInt(usize, s[0..digits_len], 10);
    const suffix = s[digits_len..];
    const multiplier: usize = if (suffix.len == 0 or std.ascii.eqlIgnoreCase(suffix, "b"))
        1
    else if (std.ascii.eqlIgnoreCase(suffix, "k") or
        std.ascii.eqlIgnoreCase(suffix, "kb") or
        std.ascii.eqlIgnoreCase(suffix, "kib"))
        KiB
    else if (std.ascii.eqlIgnoreCase(suffix, "m") or
        std.ascii.eqlIgnoreCase(suffix, "mb") or
        std.ascii.eqlIgnoreCase(suffix, "mib"))
        MiB
    else if (std.ascii.eqlIgnoreCase(suffix, "g") or
        std.ascii.eqlIgnoreCase(suffix, "gb") or
        std.ascii.eqlIgnoreCase(suffix, "gib"))
        GiB
    else
        return error.InvalidSize;

    return std.math.mul(usize, value, multiplier);
}

pub fn kib(bytes: usize) usize {
    return bytes / KiB;
}

pub fn mib(bytes: usize) usize {
    return bytes / MiB;
}

pub fn ceilMib(bytes: usize) usize {
    return (bytes + MiB - 1) / MiB;
}

test parse {
    try std.testing.expectEqual(@as(usize, 1), try parse("1"));
    try std.testing.expectEqual(@as(usize, 1024), try parse("1KiB"));
    try std.testing.expectEqual(@as(usize, 32 * MiB), try parse("32MiB"));
    try std.testing.expectEqual(@as(usize, 2 * GiB), try parse("2g"));
    try std.testing.expectError(error.InvalidSize, parse(""));
    try std.testing.expectError(error.InvalidSize, parse("4pages"));
}
