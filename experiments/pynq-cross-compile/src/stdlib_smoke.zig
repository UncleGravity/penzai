const std = @import("std");

const Io = std.Io;
const net = Io.net;

const payload = "penzai-stdlib-smoke\n";

const ServerResult = enum {
    pending,
    ok,
    accept_failed,
    read_failed,
    bad_payload,
    write_failed,
    flush_failed,
};

const ServerContext = struct {
    io: Io,
    server: *net.Server,
    result: ServerResult = .pending,
};

pub fn main(init: std.process.Init) !void {
    const io = init.io;

    try fileRoundTrip(io);
    try clockSmoke(io);
    try threadSmoke();
    try tcpLoopback(io);

    std.debug.print("penzai pynq stdlib smoke passed\n", .{});
    std.debug.print("features=file,clock,thread,tcp-loopback\n", .{});
}

fn fileRoundTrip(io: Io) !void {
    const path = "/tmp/penzai-stdlib-smoke.txt";
    defer Io.Dir.deleteFileAbsolute(io, path) catch {};

    {
        var file = try Io.Dir.createFileAbsolute(io, path, .{ .read = true });
        defer file.close(io);

        var buffer: [128]u8 = undefined;
        var writer = file.writerStreaming(io, &buffer);
        try writer.interface.writeAll(payload);
        try writer.interface.flush();
        try file.sync(io);
    }

    var file = try Io.Dir.openFileAbsolute(io, path, .{});
    defer file.close(io);

    var read_buffer: [128]u8 = undefined;
    var reader = file.readerStreaming(io, &read_buffer);
    var got: [payload.len]u8 = undefined;
    try reader.interface.readSliceAll(&got);

    if (!std.mem.eql(u8, &got, payload)) return error.FilePayloadMismatch;

    const stat = try file.stat(io);
    if (stat.size != payload.len) return error.FileSizeMismatch;
}

fn clockSmoke(io: Io) !void {
    const resolution = try Io.Clock.awake.resolution(io);
    if (resolution.nanoseconds < 0) return error.ClockResolutionInvalid;

    const before = Io.Clock.awake.now(io);
    try Io.sleep(io, .fromMilliseconds(1), .awake);
    const after = Io.Clock.awake.now(io);
    if (after.nanoseconds < before.nanoseconds) return error.ClockWentBackwards;
}

fn threadSmoke() !void {
    var value: u32 = 0;
    const thread = try std.Thread.spawn(.{}, threadMain, .{&value});
    thread.join();
    if (value != 0x5a17c001) return error.ThreadResultMismatch;
}

fn threadMain(value: *u32) void {
    value.* = 0x5a17c001;
}

fn tcpLoopback(io: Io) !void {
    var server_addr = net.IpAddress{ .ip4 = .loopback(0) };
    var server = try server_addr.listen(io, .{ .reuse_address = true });
    defer server.deinit(io);

    var ctx = ServerContext{ .io = io, .server = &server };
    const thread = try std.Thread.spawn(.{}, serverThread, .{&ctx});

    var client = try server.socket.address.connect(io, .{ .mode = .stream });
    defer client.close(io);

    {
        var buffer: [64]u8 = undefined;
        var writer = client.writer(io, &buffer);
        try writer.interface.writeAll("ping");
        try writer.interface.flush();
    }

    {
        var buffer: [64]u8 = undefined;
        var reader = client.reader(io, &buffer);
        var got: [4]u8 = undefined;
        try reader.interface.readSliceAll(&got);
        if (!std.mem.eql(u8, &got, "pong")) return error.TcpPayloadMismatch;
    }

    thread.join();
    if (ctx.result != .ok) return error.TcpServerFailed;
}

fn serverThread(ctx: *ServerContext) void {
    var stream = ctx.server.accept(ctx.io) catch {
        ctx.result = .accept_failed;
        return;
    };
    defer stream.close(ctx.io);

    {
        var buffer: [64]u8 = undefined;
        var reader = stream.reader(ctx.io, &buffer);
        var got: [4]u8 = undefined;
        reader.interface.readSliceAll(&got) catch {
            ctx.result = .read_failed;
            return;
        };
        if (!std.mem.eql(u8, &got, "ping")) {
            ctx.result = .bad_payload;
            return;
        }
    }

    {
        var buffer: [64]u8 = undefined;
        var writer = stream.writer(ctx.io, &buffer);
        writer.interface.writeAll("pong") catch {
            ctx.result = .write_failed;
            return;
        };
        writer.interface.flush() catch {
            ctx.result = .flush_failed;
            return;
        };
    }

    ctx.result = .ok;
}

test "thread smoke value" {
    var value: u32 = 0;
    threadMain(&value);
    try std.testing.expectEqual(@as(u32, 0x5a17c001), value);
}

