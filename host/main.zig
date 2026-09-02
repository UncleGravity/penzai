const std = @import("std");
const shared = @import("shared");
const run_mod = @import("run.zig");
const cli = @import("cli/args.zig");

const capabilities = shared.capabilities;
const protocol_transport = shared.protocol_transport;

pub fn main(init: std.process.Init) !void {
    var stdout_buffer: [4096]u8 = undefined;
    var stderr_buffer: [4096]u8 = undefined;
    var stdout = std.Io.File.stdout().writerStreaming(init.io, &stdout_buffer);
    var stderr = std.Io.File.stderr().writerStreaming(init.io, &stderr_buffer);

    runMain(init, &stdout.interface, &stderr.interface) catch |err| {
        try stderr.interface.print("error: {s}\n", .{@errorName(err)});
        try stderr.interface.flush();
        std.process.exit(1);
    };

    try stdout.interface.flush();
    try stderr.interface.flush();
}

fn runMain(init: std.process.Init, stdout: *std.Io.Writer, stderr: *std.Io.Writer) !void {
    var iterator = try std.process.Args.Iterator.initAllocator(init.minimal.args, init.gpa);
    defer iterator.deinit();
    _ = iterator.next();

    var argv: std.ArrayList([]const u8) = .empty;
    defer argv.deinit(init.gpa);
    while (iterator.next()) |arg| try argv.append(init.gpa, arg);

    const command = cli.parse(argv.items) catch |err| {
        try stderr.print("error: {s}\n\n", .{@errorName(err)});
        try writeUsage(stderr);
        try stderr.flush();
        std.process.exit(2);
    };

    switch (command) {
        .help => try writeUsage(stdout),
        .run => |options| try runInference(init, stdout, options, false),
        .serve => |options| try runServer(init, stdout, stderr, options),
        .benchmark_inference => |options| {
            try stdout.print("penzai benchmark inference\nmetrics={s}\n", .{@tagName(options.metrics)});
            try runInference(init, stdout, options, false);
        },
        .benchmark_hardware => |options| {
            try stdout.print("penzai benchmark hardware\nmetrics={s}\n", .{@tagName(options.metrics)});
            try runInference(init, stdout, options, true);
        },
        .verify_logits => |options| try verifyLogits(init, stdout, options),
        .inspect_device => |options| try inspectDevice(init, stdout, options),
    }
}

fn runInference(
    init: std.process.Init,
    writer: *std.Io.Writer,
    parsed: cli.InferenceOptions,
    hardware_metrics: bool,
) !void {
    var options: run_mod.LlamaOptions = .{};
    options.model_path = parsed.model.?;
    options.prompt = parsed.prompt;
    options.prompt_tokens = parsed.prompt_tokens;
    options.max_tokens = parsed.max_tokens;
    options.chat_template = !parsed.raw_prompt;
    options.enable_thinking = parsed.thinking;
    options.metrics_level = @enumFromInt(@intFromEnum(parsed.metrics));
    options.hardware_metrics = hardware_metrics or parsed.metrics == .full;
    options.device_label = parsed.endpoint.label;
    options.exact_tokens = parsed.exact_tokens;
    options.token_ids = parsed.token_ids;
    options.n_ctx = parsed.context;
    options.n_batch = parsed.batch;
    options.n_ubatch = parsed.ubatch;
    try run_mod.runTcpLlama(init.io, init.gpa, writer, tcpSpec(parsed.endpoint), options);
}

fn verifyLogits(init: std.process.Init, writer: *std.Io.Writer, parsed: cli.InferenceOptions) !void {
    var options: run_mod.LlamaOptions = .{};
    options.model_path = parsed.model.?;
    options.prompt = parsed.prompt;
    options.prompt_tokens = parsed.prompt_tokens;
    options.max_tokens = parsed.max_tokens;
    options.logits_tolerance = parsed.tolerance;
    options.chat_template = !parsed.raw_prompt;
    options.enable_thinking = parsed.thinking;
    options.metrics_level = @enumFromInt(@intFromEnum(parsed.metrics));
    options.hardware_metrics = parsed.metrics == .full;
    options.device_label = parsed.endpoint.label;
    options.exact_tokens = parsed.exact_tokens;
    options.n_ctx = parsed.context;
    options.n_batch = parsed.batch;
    options.n_ubatch = parsed.ubatch;
    try run_mod.runTcpLogitsCheck(init.io, init.gpa, writer, tcpSpec(parsed.endpoint), options);
}

fn inspectDevice(init: std.process.Init, writer: *std.Io.Writer, options: cli.InspectOptions) !void {
    const report = try run_mod.tcpCapabilities(init.io, init.gpa, tcpSpec(options.endpoint));
    try writeCapabilities(writer, options.endpoint.label, report);
}

fn writeCapabilities(writer: *std.Io.Writer, device: []const u8, report: capabilities.Report) std.Io.Writer.Error!void {
    try writer.writeAll("penzai inspect device\n");
    try writer.print("device={s}\n", .{device});
    try writer.print("capability_schema={d}\n", .{capabilities.version});
    try writer.print("wire_abi={d}\n", .{report.wire_abi});
    try writer.print("metrics_schema={d}\n", .{report.metrics_schema});
    try writer.print("metrics.summary_supported={}\n", .{
        report.feature_mask & capabilities.Feature.metrics_summary != 0,
    });
    try writer.print("metrics.full_supported={}\n", .{
        report.feature_mask & capabilities.Feature.metrics_full != 0,
    });
    try writer.print("feature_mask=0x{X:0>8}\n", .{report.feature_mask});
    try writer.print("format_mask=0x{X:0>8}\n", .{report.format_mask});
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
    try writer.print("engine.interface_id=0x{X:0>8}\n", .{report.engine.interface_id});
    try writer.print("engine.interface_version=0x{X:0>8}\n", .{report.engine.interface_version});
    try writer.print("engine.clock_hz={d}\n", .{report.engine.clock_hz});
    try writer.print("engine.token_tile_max={d}\n", .{report.engine.token_tile_max});
    try writer.print("engine.token_lanes={d}\n", .{report.engine.token_lanes});
    try writer.print("engine.model_spec_count={d}\n", .{report.engine.model_spec_count});
    try writer.print("engine.context_tokens_max={d}\n", .{report.engine.context_tokens_max});
    try writer.print("engine.address_record_bytes={d}\n", .{report.engine.address_record_bytes});
}

test "device inspection prints the engine interface as canonical hex" {
    var buffer: [2048]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buffer);
    const report: capabilities.Report = .{
        .engine = .{ .interface_version = 0x0001_0007 },
    };

    try writeCapabilities(&writer, "tcp:board:29092", report);
    try std.testing.expect(std.mem.indexOf(
        u8,
        writer.buffered(),
        "engine.interface_version=0x00010007\n",
    ) != null);
}

fn runServer(
    init: std.process.Init,
    stdout: *std.Io.Writer,
    stderr: *std.Io.Writer,
    options: cli.ServeOptions,
) !void {
    const server = init.environ_map.get("PENZAI_LLAMA_SERVER") orelse
        "llama-server";
    if (server.len == 0) return error.InvalidServerPath;

    var environment = try init.environ_map.clone(init.gpa);
    defer environment.deinit();

    const backend_path = init.environ_map.get("PENZAI_GGML_BACKEND_PATH") orelse
        return error.MissingBackendPath;
    try environment.put("GGML_BACKEND_PATH", backend_path);

    var device_port_buffer: [5]u8 = undefined;
    const device_port = try std.fmt.bufPrint(&device_port_buffer, "{d}", .{options.endpoint.port});
    try environment.put("PENZAI_HOST", options.endpoint.host);
    try environment.put("PENZAI_PORT", device_port);
    try environment.put("PENZAI_METRICS", "none");

    var listen_port_buffer: [5]u8 = undefined;
    const listen_port = try std.fmt.bufPrint(&listen_port_buffer, "{d}", .{options.listen_port});
    var context_buffer: [10]u8 = undefined;
    const context = try std.fmt.bufPrint(&context_buffer, "{d}", .{options.context});
    var batch_buffer: [10]u8 = undefined;
    const batch = try std.fmt.bufPrint(&batch_buffer, "{d}", .{options.batch});
    var ubatch_buffer: [10]u8 = undefined;
    const ubatch = try std.fmt.bufPrint(&ubatch_buffer, "{d}", .{options.ubatch});
    const child_argv = [_][]const u8{
        server,
        "--device",
        "penzai",
        "-ngl",
        "999",
        "--no-op-offload",
        "-fa",
        "on",
        "-m",
        options.model.?,
        "--ctx-size",
        context,
        "--batch-size",
        batch,
        "--ubatch-size",
        ubatch,
        "--temp",
        "0",
        "--no-warmup",
        "--host",
        options.listen_host,
        "--port",
        listen_port,
        "--parallel",
        "1",
        "--no-context-shift",
    };

    // Flush buffered handles before replacement; the server inherits the same
    // terminal streams and signal lifecycle.
    try stdout.flush();
    try stderr.flush();
    if (std.process.can_replace)
        return std.process.replace(init.io, .{ .argv = &child_argv, .environ_map = &environment });

    var child = try std.process.spawn(init.io, .{
        .argv = &child_argv,
        .environ_map = &environment,
        .stdin = .inherit,
        .stdout = .inherit,
        .stderr = .inherit,
    });
    const term = try child.wait(init.io);
    switch (term) {
        .exited => |status| if (status != 0) return error.LlamaServerTerminated,
        else => return error.LlamaServerTerminated,
    }
}

fn tcpSpec(endpoint: cli.Endpoint) protocol_transport.TcpSpec {
    return .{ .host = endpoint.host, .port = endpoint.port };
}

fn writeUsage(writer: *std.Io.Writer) std.Io.Writer.Error!void {
    try writer.writeAll(
        \\usage:
        \\  penzai run -m MODEL.gguf [--device tcp:HOST:PORT] [--prompt TEXT] [--max-tokens N] [--metrics none|summary|full]
        \\  penzai serve -m MODEL.gguf [--device tcp:HOST:PORT] [--host ADDRESS] [--port N] [--context N] [--batch N] [--ubatch N]
        \\  penzai benchmark inference -m MODEL.gguf [--device tcp:HOST:PORT] [--prompt TEXT] [--max-tokens N] [--metrics summary|full]
        \\  penzai benchmark hardware -m MODEL.gguf [--device tcp:HOST:PORT] [--prompt TEXT] [--max-tokens N] [--metrics summary|full]
        \\  penzai verify logits -m MODEL.gguf [--device tcp:HOST:PORT] [--prompt TEXT] [--max-tokens N] [--exact-tokens] [--tolerance F]
        \\  penzai inspect device [--device tcp:HOST:PORT]
        \\  penzai help
        \\
        \\The default device is tcp:127.0.0.1:29092.
        \\`serve` launches a persistent server with one active greedy sequence; PENZAI_LLAMA_SERVER overrides the executable.
        \\
    );
}

test "usage advertises exact logits qualification controls" {
    var buffer: [4096]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buffer);

    try writeUsage(&writer);
    try std.testing.expect(std.mem.indexOf(
        u8,
        writer.buffered(),
        "verify logits -m MODEL.gguf",
    ) != null);
    try std.testing.expect(std.mem.indexOf(u8, writer.buffered(), "[--max-tokens N] [--exact-tokens]") != null);
}
