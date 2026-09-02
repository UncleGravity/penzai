const std = @import("std");

pub fn nowNs(io: ?std.Io) u64 {
    const active = io orelse return 0;
    const nanoseconds = std.Io.Timestamp.now(active, .awake).nanoseconds;
    return std.math.cast(u64, nanoseconds) orelse 0;
}

pub fn elapsed(start_ns: u64, end_ns: u64) u64 {
    return if (end_ns >= start_ns) end_ns - start_ns else 0;
}

test "elapsed clocks cannot underflow" {
    try std.testing.expectEqual(@as(u64, 7), elapsed(3, 10));
    try std.testing.expectEqual(@as(u64, 0), elapsed(10, 3));
}
