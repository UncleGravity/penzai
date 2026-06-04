const std = @import("std");
const builtin = @import("builtin");

const WireProbe = extern struct {
    magic: u32,
    version: u16,
    flags: u16,
};

pub fn main() !void {
    var data = [_]u8{ 0x11, 0x23, 0x35, 0x47, 0x59, 0x6b, 0x7d, 0x8f };
    const sum = checksum(&data);

    if (sum != 0x21efb4e5) {
        std.debug.print("checksum mismatch: got 0x{x}, want 0x21efb4e5\n", .{sum});
        return error.ChecksumMismatch;
    }

    std.debug.print("penzai pynq cross-compile smoke passed\n", .{});
    std.debug.print(
        "target={s}-{s}-{s} cpu={s}\n",
        .{
            @tagName(builtin.cpu.arch),
            @tagName(builtin.os.tag),
            @tagName(builtin.abi),
            builtin.cpu.model.name,
        },
    );
    std.debug.print(
        "sizes: pointer_bits={d} usize={d} wire_probe={d}\n",
        .{
            @bitSizeOf(usize),
            @sizeOf(usize),
            @sizeOf(WireProbe),
        },
    );
    std.debug.print("checksum=0x{x}\n", .{sum});
}

fn checksum(bytes: []const u8) u32 {
    var acc: u32 = 0x811c9dc5;
    for (bytes) |byte| {
        acc ^= byte;
        acc *%= 0x01000193;
    }
    return acc;
}

test checksum {
    const data = [_]u8{ 0x11, 0x23, 0x35, 0x47, 0x59, 0x6b, 0x7d, 0x8f };
    try std.testing.expectEqual(@as(u32, 0x21efb4e5), checksum(&data));
}
