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

const LlamaMode = enum { generate, census, logits };

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

    if (std.mem.eql(u8, command, "run")) {
        try runLlamaCommand(init, &args, stdout, .generate);
        return;
    }
    if (std.mem.eql(u8, command, "census")) {
        try runLlamaCommand(init, &args, stdout, .census);
        return;
    }
    if (std.mem.eql(u8, command, "logits")) {
        try runLlamaCommand(init, &args, stdout, .logits);
        return;
    }
    if (std.mem.eql(u8, command, "matmul")) {
        try runMatmulCommand(init, &args, stdout);
        return;
    }
    return error.InvalidCommand;
}

fn runLlamaCommand(
    init: std.process.Init,
    args: *std.process.Args.Iterator,
    stdout: *std.Io.Writer,
    mode: LlamaMode,
) CliError!void {
    var options: run_mod.LlamaOptions = .{};
    if (mode == .census) {
        options.max_tokens = 2;
        options.census = true;
    } else if (mode == .logits) {
        options.max_tokens = 1;
    }
    var device: []const u8 = "fake";
    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "--device")) {
            device = try requireValue(args, "--device");
        } else if (std.mem.startsWith(u8, arg, "--device=")) {
            device = arg["--device=".len..];
        } else if (std.mem.eql(u8, arg, "-m") or std.mem.eql(u8, arg, "--model")) {
            options.model_path = try requireValue(args, arg);
        } else if (std.mem.startsWith(u8, arg, "--model=")) {
            options.model_path = arg["--model=".len..];
        } else if (std.mem.eql(u8, arg, "--prompt")) {
            options.prompt = try requireValue(args, "--prompt");
        } else if (std.mem.startsWith(u8, arg, "--prompt=")) {
            options.prompt = arg["--prompt=".len..];
        } else if (std.mem.eql(u8, arg, "--max-tokens")) {
            options.max_tokens = try parseU32(try requireValue(args, "--max-tokens"));
        } else if (std.mem.startsWith(u8, arg, "--max-tokens=")) {
            options.max_tokens = try parseU32(arg["--max-tokens=".len..]);
        } else if (std.mem.eql(u8, arg, "--heap-mib")) {
            options.heap_mib = try parseU32(try requireValue(args, "--heap-mib"));
        } else if (std.mem.startsWith(u8, arg, "--heap-mib=")) {
            options.heap_mib = try parseU32(arg["--heap-mib=".len..]);
        } else if (std.mem.eql(u8, arg, "--tolerance")) {
            options.logits_tolerance = try parseF32(try requireValue(args, "--tolerance"));
        } else if (std.mem.startsWith(u8, arg, "--tolerance=")) {
            options.logits_tolerance = try parseF32(arg["--tolerance=".len..]);
        } else if (std.mem.eql(u8, arg, "--raw-prompt")) {
            options.chat_template = false;
        } else if (std.mem.eql(u8, arg, "--think")) {
            options.enable_thinking = true;
        } else if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            try writeUsage(stdout);
            return;
        } else {
            return error.InvalidOption;
        }
    }

    const device_spec = try protocol_transport.parseDeviceSpec(device);
    switch (mode) {
        .generate, .census => switch (device_spec) {
            .fake => try run_mod.runFakeLlama(init.gpa, stdout, options),
            .tcp => |tcp| try run_mod.runTcpLlama(init.io, init.gpa, stdout, tcp, options),
        },
        .logits => switch (device_spec) {
            .fake => try run_mod.runFakeLogitsCheck(init.gpa, stdout, options),
            .tcp => |tcp| try run_mod.runTcpLogitsCheck(init.io, init.gpa, stdout, tcp, options),
        },
    }
}

fn runMatmulCommand(
    init: std.process.Init,
    args: *std.process.Args.Iterator,
    stdout: *std.Io.Writer,
) CliError!void {
    var options: run_mod.RunOptions = .{};
    var device: []const u8 = "fake";
    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "--device")) {
            device = try requireValue(args, "--device");
        } else if (std.mem.startsWith(u8, arg, "--device=")) {
            device = arg["--device=".len..];
        } else if (std.mem.eql(u8, arg, "--rows")) {
            options.rows = try parseU32(try requireValue(args, "--rows"));
        } else if (std.mem.startsWith(u8, arg, "--rows=")) {
            options.rows = try parseU32(arg["--rows=".len..]);
        } else if (std.mem.eql(u8, arg, "--cols")) {
            options.cols = try parseU32(try requireValue(args, "--cols"));
        } else if (std.mem.startsWith(u8, arg, "--cols=")) {
            options.cols = try parseU32(arg["--cols=".len..]);
        } else if (std.mem.eql(u8, arg, "--k")) {
            options.k = try parseU32(try requireValue(args, "--k"));
        } else if (std.mem.startsWith(u8, arg, "--k=")) {
            options.k = try parseU32(arg["--k=".len..]);
        } else if (std.mem.eql(u8, arg, "--heap-mib")) {
            options.heap_mib = try parseU32(try requireValue(args, "--heap-mib"));
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
    try stdout.print("penzai matmul\n", .{});
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

fn parseF32(value: []const u8) CliError!f32 {
    return std.fmt.parseFloat(f32, value) catch return error.InvalidNumber;
}

fn writeUsage(writer: *std.Io.Writer) std.Io.Writer.Error!void {
    try writer.writeAll(
        \\usage:
        \\  penzai run -m MODEL.gguf --device fake|tcp:HOST:PORT --prompt TEXT [--max-tokens N] [--raw-prompt] [--think]
        \\  penzai census -m MODEL.gguf --device fake|tcp:HOST:PORT --prompt TEXT [--max-tokens N] [--raw-prompt] [--think]
        \\  penzai logits -m MODEL.gguf --device fake|tcp:HOST:PORT --prompt TEXT [--max-tokens N] [--tolerance F] [--raw-prompt] [--think]
        \\  penzai matmul --device fake [--rows N] [--cols N] [--k N] [--heap-mib N]
        \\  penzai matmul --device tcp:HOST:PORT [--rows N] [--cols N] [--k N]
        \\
        \\commands:
        \\  run      generate text through llama.cpp and the penzai backend
        \\  census   report the actual ggml graph_compute op surface
        \\  logits   compare penzai logits against llama.cpp CPU logits
        \\  matmul   execute the Q1A8 smoke path through fake or TCP device
        \\  help     show this help
        \\
    );
}
