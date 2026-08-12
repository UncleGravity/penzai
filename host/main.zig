const std = @import("std");
const shared = @import("shared");
const run_mod = @import("run.zig");
const prof = @import("prof");
const prof_model = prof.model;
const prof_render = prof.render;
const capabilities = shared.capabilities;

const protocol_transport = shared.protocol_transport;

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
    if (std.mem.eql(u8, command, "bench")) {
        try runBenchCommand(init, &args, stdout);
        return;
    }
    if (std.mem.eql(u8, command, "capabilities")) {
        try runCapabilitiesCommand(init, &args, stdout);
        return;
    }
    return error.InvalidCommand;
}

fn runCapabilitiesCommand(
    init: std.process.Init,
    args: *std.process.Args.Iterator,
    stdout: *std.Io.Writer,
) CliError!void {
    var device: []const u8 = "fake";
    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "--device")) {
            device = try requireValue(args, "--device");
        } else if (std.mem.startsWith(u8, arg, "--device=")) {
            device = arg["--device=".len..];
        } else if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            try writeUsage(stdout);
            return;
        } else {
            return error.InvalidOption;
        }
    }

    const device_spec = try protocol_transport.parseDeviceSpec(device);
    const report = switch (device_spec) {
        .fake => try run_mod.fakeCapabilities(init.gpa),
        .tcp => |tcp| try run_mod.tcpCapabilities(init.io, init.gpa, tcp),
    };
    try writeCapabilities(stdout, device, report);
}

fn writeCapabilities(writer: *std.Io.Writer, device: []const u8, report: capabilities.Report) std.Io.Writer.Error!void {
    try writer.print("penzai capabilities\n", .{});
    try writer.print("device={s}\n", .{device});
    try writer.print("capability_schema={d}\n", .{capabilities.version});
    try writer.print("wire_abi={d}\n", .{report.wire_abi});
    try writer.print("profile_abi={d}\n", .{report.profile_abi});
    try writer.print("receipt_status={s}\n", .{@tagName(report.receipt_status)});
    try writer.print("manifest_schema={d}\n", .{report.manifest_schema});
    try writer.print("run_id={s}\n", .{report.run_id.slice()});
    try writer.print("variant={s}\n", .{report.variant.slice()});
    try writer.print("git_commit={s}\n", .{report.git_commit.slice()});
    try writer.print("git_dirty={}\n", .{report.identity_flags & capabilities.IdentityFlag.git_dirty != 0});
    try writer.print("bitstream_hash_verified={}\n", .{report.identity_flags & capabilities.IdentityFlag.bitstream_hash_verified != 0});
    try writer.print("bitstream_sha256={s}\n", .{report.bitstream_sha256.slice()});
    try writer.print("manifest_sha256={s}\n", .{report.manifest_sha256.slice()});
    try writer.print("source_sha256={s}\n", .{report.source_sha256.slice()});
    try writer.print("engine_mask=0x{X:0>8}\n", .{report.engine_mask});
    try writer.print("format_mask=0x{X:0>8}\n", .{report.format_mask});
    try writer.print("matmul.id=0x{X:0>8}\n", .{report.matmul.id});
    try writer.print("matmul.version={d}\n", .{report.matmul.version});
    try writer.print("matmul.clock_hz={d}\n", .{report.matmul.clock_hz});
    try writer.print("matmul.rows={d}\n", .{report.matmul.dim0});
    try writer.print("matmul.weight_ports={d}\n", .{report.matmul.dim1});
    try writer.print("matmul.cols_max={d}\n", .{report.matmul.dim2});
    try writer.print("matmul.k_max={d}\n", .{report.matmul.dim3});
    try writer.print("flash.id=0x{X:0>8}\n", .{report.flash.id});
    try writer.print("flash.version={d}\n", .{report.flash.version});
    try writer.print("flash.clock_hz={d}\n", .{report.flash.clock_hz});
    try writer.print("flash.lanes={d}\n", .{report.flash.dim0});
    try writer.print("flash.head_dim_max={d}\n", .{report.flash.dim1});
    try writer.print("flash.heads_max={d}\n", .{report.flash.dim2});
    try writer.print("flash.kv_heads_max={d}\n", .{report.flash.dim3});
    try writer.print("flash.query_slots={d}\n", .{report.flash.dim4});
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
        } else if (std.mem.eql(u8, arg, "--prompt-tokens")) {
            options.prompt_tokens = try parseU32(try requireValue(args, "--prompt-tokens"));
        } else if (std.mem.startsWith(u8, arg, "--prompt-tokens=")) {
            options.prompt_tokens = try parseU32(arg["--prompt-tokens=".len..]);
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
        } else if (std.mem.eql(u8, arg, "--backend-sampling")) {
            options.backend_sampling = true;
        } else if (std.mem.eql(u8, arg, "--exact-tokens")) {
            options.exact_tokens = true;
        } else if (std.mem.eql(u8, arg, "--context")) {
            options.n_ctx = try parseU32(try requireValue(args, "--context"));
        } else if (std.mem.startsWith(u8, arg, "--context=")) {
            options.n_ctx = try parseU32(arg["--context=".len..]);
        } else if (std.mem.eql(u8, arg, "--batch")) {
            options.n_batch = try parseU32(try requireValue(args, "--batch"));
        } else if (std.mem.startsWith(u8, arg, "--batch=")) {
            options.n_batch = try parseU32(arg["--batch=".len..]);
        } else if (std.mem.eql(u8, arg, "--ubatch")) {
            options.n_ubatch = try parseU32(try requireValue(args, "--ubatch"));
        } else if (std.mem.startsWith(u8, arg, "--ubatch=")) {
            options.n_ubatch = try parseU32(arg["--ubatch=".len..]);
        } else if (std.mem.eql(u8, arg, "--prof")) {
            options.profile = true;
        } else if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            try writeUsage(stdout);
            return;
        } else {
            return error.InvalidOption;
        }
    }
    options.device_label = device;

    const device_spec = try protocol_transport.parseDeviceSpec(device);
    switch (mode) {
        .generate, .census => switch (device_spec) {
            .fake => try run_mod.runFakeLlama(init.io, init.gpa, stdout, options),
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

fn runBenchCommand(
    init: std.process.Init,
    args: *std.process.Args.Iterator,
    stdout: *std.Io.Writer,
) CliError!void {
    const bench_kind = args.next() orelse {
        try writeUsage(stdout);
        return;
    };
    if (std.mem.eql(u8, bench_kind, "--help") or std.mem.eql(u8, bench_kind, "-h")) {
        try writeUsage(stdout);
        return;
    }
    if (!std.mem.eql(u8, bench_kind, "op")) return error.InvalidCommand;
    const op = args.next() orelse return error.MissingValue;
    if (!std.mem.eql(u8, op, "matmul-q1a8")) return error.InvalidCommand;

    var options: run_mod.BenchOptions = .{};
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
        } else if (std.mem.eql(u8, arg, "--warmup")) {
            options.warmup = try parseU32(try requireValue(args, "--warmup"));
        } else if (std.mem.startsWith(u8, arg, "--warmup=")) {
            options.warmup = try parseU32(arg["--warmup=".len..]);
        } else if (std.mem.eql(u8, arg, "--iters")) {
            options.iters = try parseU32(try requireValue(args, "--iters"));
        } else if (std.mem.startsWith(u8, arg, "--iters=")) {
            options.iters = try parseU32(arg["--iters=".len..]);
        } else if (std.mem.eql(u8, arg, "--prof")) {
            options.profile = true;
        } else if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            try writeUsage(stdout);
            return;
        } else {
            return error.InvalidOption;
        }
    }

    const device_spec = try protocol_transport.parseDeviceSpec(device);
    const result = switch (device_spec) {
        .fake => try run_mod.benchFakeMatmulQ1A8(init.io, init.gpa, options),
        .tcp => |tcp| try run_mod.benchTcpMatmulQ1A8(init.io, init.gpa, tcp, options),
    };

    try stdout.print("penzai bench op matmul-q1a8\n", .{});
    switch (device_spec) {
        .fake => try stdout.print("device=fake rows={d} cols={d} k={d} warmup={d} iters={d} prof={}\n", .{
            result.rows,
            result.cols,
            result.k,
            result.warmup,
            result.iters,
            result.profiled,
        }),
        .tcp => |tcp| try stdout.print("device=tcp host={s} port={d} rows={d} cols={d} k={d} warmup={d} iters={d} prof={}\n", .{
            tcp.host,
            tcp.port,
            result.rows,
            result.cols,
            result.k,
            result.warmup,
            result.iters,
            result.profiled,
        }),
    }
    try stdout.print("bytes weights={d} acts={d} dst={d}\n", .{
        result.weights_nbytes,
        result.acts_nbytes,
        result.dst_nbytes,
    });
    try stdout.print("host total_ms={d:.3} avg_ms={d:.3} min_ms={d:.3} max_ms={d:.3} ops_s={d:.3}\n", .{
        prof_model.nsToMs(result.host_total_ns),
        prof_model.avgMs(result.host_total_ns, result.iters),
        prof_model.nsToMs(result.host_min_ns),
        prof_model.nsToMs(result.host_max_ns),
        prof_model.perSecond(result.iters, result.host_total_ns),
    });
    try stdout.print("check=ok expected={d:.3} max_abs_diff={d:.6}\n", .{
        result.expected,
        result.max_abs_diff,
    });
    if (result.profiled) try writeBenchProfile(stdout, result.profile);
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

fn writeBenchProfile(writer: *std.Io.Writer, profile: run_mod.BenchProfile) std.Io.Writer.Error!void {
    try prof_render.writeLinkSection(writer, &profile);
    try writer.writeByte('\n');
    try prof_render.writeOpTable(writer, "device ops", &profile.op_totals, profile.device_execute_ns);
    try prof_render.writeMatmulDetail(writer, "matmul", profile.usedMatmul(), profile.device_fclk_hz);
    try prof_render.writeFlashDetail(writer, "flash", profile.usedFlash(), profile.device_fclk_hz);
}

fn writeUsage(writer: *std.Io.Writer) std.Io.Writer.Error!void {
    try writer.writeAll(
        \\usage:
        \\  penzai run -m MODEL.gguf --device fake|tcp:HOST:PORT --prompt TEXT [--prompt-tokens N] [--max-tokens N] [--context N] [--batch N] [--ubatch N] [--exact-tokens] [--raw-prompt] [--think] [--prof]
        \\  penzai census -m MODEL.gguf --device fake|tcp:HOST:PORT --prompt TEXT [--max-tokens N] [--raw-prompt] [--think]
        \\  penzai logits -m MODEL.gguf --device fake|tcp:HOST:PORT --prompt TEXT [--max-tokens N] [--tolerance F] [--raw-prompt] [--think]
        \\  penzai matmul --device fake [--rows N] [--cols N] [--k N] [--heap-mib N]
        \\  penzai matmul --device tcp:HOST:PORT [--rows N] [--cols N] [--k N]
        \\  penzai bench op matmul-q1a8 --device fake|tcp:HOST:PORT [--rows N] [--cols N] [--k N] [--warmup N] [--iters N] [--prof]
        \\  penzai capabilities --device fake|tcp:HOST:PORT
        \\
        \\commands:
        \\  run      generate text through llama.cpp and the penzai backend
        \\  census   report the actual ggml graph_compute op surface
        \\  logits   compare token choices and report logit drift against llama.cpp CPU
        \\  matmul   execute the Q1A8 smoke path through fake or TCP device
        \\  bench    run resident-buffer microbenchmarks
        \\  capabilities  report daemon and loaded-bitstream identity
        \\  help     show this help
        \\
    );
}
