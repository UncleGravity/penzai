const std = @import("std");
const protocol_transport = @import("protocol_transport");
const device_tcp = @import("device_tcp");

const CliError = error{
    InvalidCommand,
    InvalidOption,
    InvalidNumber,
    MissingValue,
    UnsupportedDevice,
} || protocol_transport.ParseError || device_tcp.ServeError || std.process.Args.Iterator.InitError || std.Io.Writer.Error;

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

    if (!std.mem.eql(u8, command, "serve")) return error.InvalidCommand;

    var device_text: []const u8 = "tcp:0.0.0.0:9000";
    var options: device_tcp.ServeOptions = .{};
    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "--device")) {
            device_text = try requireValue(&args);
        } else if (std.mem.startsWith(u8, arg, "--device=")) {
            device_text = arg["--device=".len..];
        } else if (std.mem.eql(u8, arg, "--heap-mib")) {
            options.heap_mib = try parseU32(try requireValue(&args));
        } else if (std.mem.startsWith(u8, arg, "--heap-mib=")) {
            options.heap_mib = try parseU32(arg["--heap-mib=".len..]);
        } else if (std.mem.eql(u8, arg, "--max-requests")) {
            options.max_requests = try parseU32(try requireValue(&args));
        } else if (std.mem.startsWith(u8, arg, "--max-requests=")) {
            options.max_requests = try parseU32(arg["--max-requests=".len..]);
        } else if (std.mem.eql(u8, arg, "--mem")) {
            options.memory = try parseMemoryBackend(try requireValue(&args));
        } else if (std.mem.startsWith(u8, arg, "--mem=")) {
            options.memory = try parseMemoryBackend(arg["--mem=".len..]);
        } else if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            try writeUsage(stdout);
            return;
        } else {
            return error.InvalidOption;
        }
    }

    const device = try protocol_transport.parseDeviceSpec(device_text);
    switch (device) {
        .fake => return error.UnsupportedDevice,
        .tcp => |tcp| {
            try stdout.print("penzaid serve device=tcp host={s} port={d} heap_mib={d} mem={s}\n", .{
                tcp.host,
                tcp.port,
                options.heap_mib,
                @tagName(options.memory),
            });
            try stdout.flush();
            try device_tcp.serve(init.io, init.gpa, tcp, options);
        },
    }
}

fn requireValue(args: *std.process.Args.Iterator) CliError![]const u8 {
    return args.next() orelse error.MissingValue;
}

fn parseU32(value: []const u8) CliError!u32 {
    return std.fmt.parseInt(u32, value, 10) catch return error.InvalidNumber;
}

fn parseMemoryBackend(value: []const u8) CliError!device_tcp.MemoryBackend {
    if (std.mem.eql(u8, value, "fake")) return .fake;
    if (std.mem.eql(u8, value, "xrt")) return .xrt;
    return error.InvalidOption;
}

fn writeUsage(writer: *std.Io.Writer) std.Io.Writer.Error!void {
    try writer.writeAll(
        \\usage:
        \\  penzaid serve --device tcp:HOST:PORT [--mem fake|xrt] [--heap-mib N] [--max-requests N]
        \\
        \\commands:
        \\  serve    run the board/device runtime over TCP
        \\  help     show this help
        \\
    );
}
