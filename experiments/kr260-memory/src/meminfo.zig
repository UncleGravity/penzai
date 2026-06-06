const std = @import("std");

pub const Info = struct {
    mem_total: ?u64 = null,
    mem_free: ?u64 = null,
    mem_available: ?u64 = null,
    buffers: ?u64 = null,
    cached: ?u64 = null,
    cma_total: ?u64 = null,
    cma_free: ?u64 = null,
};

pub fn read(io: std.Io) !Info {
    const file = try std.Io.Dir.openFileAbsolute(io, "/proc/meminfo", .{ .allow_directory = false });
    defer file.close(io);

    var buf: [64 * 1024]u8 = undefined;
    var used: u64 = 0;
    while (used < buf.len) {
        var dst = [_][]u8{buf[@intCast(used)..]};
        const n = try io.vtable.fileReadPositional(io.userdata, file, &dst, used);
        if (n == 0) break;
        used += n;
    }
    const text = buf[0..@intCast(used)];

    var info: Info = .{};
    var lines = std.mem.splitScalar(u8, text, '\n');
    while (lines.next()) |line| {
        if (std.mem.startsWith(u8, line, "MemTotal:")) {
            info.mem_total = try parseKbLine(line);
        } else if (std.mem.startsWith(u8, line, "MemFree:")) {
            info.mem_free = try parseKbLine(line);
        } else if (std.mem.startsWith(u8, line, "MemAvailable:")) {
            info.mem_available = try parseKbLine(line);
        } else if (std.mem.startsWith(u8, line, "Buffers:")) {
            info.buffers = try parseKbLine(line);
        } else if (std.mem.startsWith(u8, line, "Cached:")) {
            info.cached = try parseKbLine(line);
        } else if (std.mem.startsWith(u8, line, "CmaTotal:")) {
            info.cma_total = try parseKbLine(line);
        } else if (std.mem.startsWith(u8, line, "CmaFree:")) {
            info.cma_free = try parseKbLine(line);
        }
    }
    return info;
}

fn parseKbLine(line: []const u8) !u64 {
    var fields = std.mem.tokenizeAny(u8, line, " \t:");
    _ = fields.next() orelse return error.InvalidMeminfo;
    const value_kib = try std.fmt.parseInt(u64, fields.next() orelse return error.InvalidMeminfo, 10);
    return value_kib * 1024;
}
