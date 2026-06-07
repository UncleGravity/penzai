const std = @import("std");
const protocol_transport = @import("protocol_transport");
const run_mod = @import("run");

const CliError = error{
    InvalidCommand,
    InvalidOption,
    InvalidNumber,
    MissingValue,
    UnsupportedDevice,
} || protocol_transport.ParseError || run_mod.RunError || std.process.Args.Iterator.InitError || std.Io.Writer.Error;

pub fn main(init: std.process.Init) !void {
    var stdout_buf: [4096]u8 = undefined;
    var stderr_buf: [4096]u8 = undefined;
    var stdout = std.Io.File.stdout().writerStreaming(init.io, &stdout_buf);
    var stderr = std.Io.File.stderr().writerStreaming(init.io, &stderr_buf);

    runMain(init, &stdout.interface, &stderr.interface) catch |err| {
        try stderr.interface.print("error: {s}\n\n", .{@errorName(err)});
        try writeUsage(&stderr.interface);
        try stderr.interface.flush();
        std.process.exit(2);
    };

    try stdout.interface.flush();
    try stderr.interface.flush();
}

fn runMain(init: std.process.Init, stdout: *std.Io.Writer, stderr: *std.Io.Writer) CliError!void {
    _ = stderr;
    var args = try std.process.Args.Iterator.initAllocator(init.minimal.args, init.gpa);
    defer args.deinit();
    _ = args.next();

    const command = args.next() orelse {
        try writeUsage(stdout);
        return;
    };

    if (std.mem.eql(u8, command, "help") or std.mem.eql(u8, command, "--help") or std.mem.eql(u8, command, "-h")) {
        try writeUsage(stdout);
        return;
    }

    if (!std.mem.eql(u8, command, "run")) return error.InvalidCommand;

    var options: run_mod.RunOptions = .{};
    var device: []const u8 = "fake";
    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "--device")) {
            device = try requireValue(&args, "--device");
        } else if (std.mem.startsWith(u8, arg, "--device=")) {
            device = arg["--device=".len..];
        } else if (std.mem.eql(u8, arg, "--rows")) {
            options.rows = try parseU32(try requireValue(&args, "--rows"));
        } else if (std.mem.startsWith(u8, arg, "--rows=")) {
            options.rows = try parseU32(arg["--rows=".len..]);
        } else if (std.mem.eql(u8, arg, "--cols")) {
            options.cols = try parseU32(try requireValue(&args, "--cols"));
        } else if (std.mem.startsWith(u8, arg, "--cols=")) {
            options.cols = try parseU32(arg["--cols=".len..]);
        } else if (std.mem.eql(u8, arg, "--k")) {
            options.k = try parseU32(try requireValue(&args, "--k"));
        } else if (std.mem.startsWith(u8, arg, "--k=")) {
            options.k = try parseU32(arg["--k=".len..]);
        } else if (std.mem.eql(u8, arg, "--heap-mib")) {
            options.heap_mib = try parseU32(try requireValue(&args, "--heap-mib"));
        } else if (std.mem.startsWith(u8, arg, "--heap-mib=")) {
            options.heap_mib = try parseU32(arg["--heap-mib=".len..]);
        } else if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            try writeUsage(stdout);
            return;
        } else {
            return error.InvalidOption;
        }
    }

    const device_spec = try protocol_transport.parseDeviceSpec(device);
    const result = switch (device_spec) {
        .fake => try run_mod.runFakeMatmul(init.gpa, options),
        .tcp => |tcp| try run_mod.runTcpMatmul(init.io, init.gpa, tcp, options),
    };
    try stdout.print("penzai run\n", .{});
    switch (device_spec) {
        .fake => try stdout.print("device=fake rows={d} cols={d} k={d} commands={d}\n", .{
            result.rows,
            result.cols,
            result.k,
            result.command_count,
        }),
        .tcp => |tcp| try stdout.print("device=tcp host={s} port={d} rows={d} cols={d} k={d} commands={d}\n", .{
            tcp.host,
            tcp.port,
            result.rows,
            result.cols,
            result.k,
            result.command_count,
        }),
    }
    try stdout.print("bytes weights={d} acts={d} dst={d}\n", .{
        result.weights_nbytes,
        result.acts_nbytes,
        result.dst_nbytes,
    });
    try stdout.print("check=ok expected={d:.3} max_abs_diff={d:.6}\n", .{
        result.expected,
        result.max_abs_diff,
    });
}

fn requireValue(args: *std.process.Args.Iterator, option: []const u8) CliError![]const u8 {
    _ = option;
    return args.next() orelse error.MissingValue;
}

fn parseU32(value: []const u8) CliError!u32 {
    return std.fmt.parseInt(u32, value, 10) catch return error.InvalidNumber;
}

fn writeUsage(writer: *std.Io.Writer) std.Io.Writer.Error!void {
    try writer.writeAll(
        \\usage:
        \\  penzai run --device fake [--rows N] [--cols N] [--k N] [--heap-mib N]
        \\  penzai run --device tcp:HOST:PORT [--rows N] [--cols N] [--k N]
        \\
        \\commands:
        \\  run      execute the Q1A8 smoke path through fake or TCP device
        \\  help     show this help
        \\
    );
}
