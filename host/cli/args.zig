const std = @import("std");

pub const default_device = "tcp:127.0.0.1:29092";

pub const ParseError = error{
    InvalidCommand,
    InvalidDevice,
    InvalidMetrics,
    InvalidNumber,
    InvalidOption,
    InvalidParallel,
    MissingModel,
    MissingValue,
};

pub const Metrics = enum {
    none,
    summary,
    full,

    fn atLeast(self: Metrics, minimum: Metrics) Metrics {
        return if (@intFromEnum(self) < @intFromEnum(minimum)) minimum else self;
    }
};

pub const Endpoint = struct {
    label: []const u8 = default_device,
    host: []const u8 = "127.0.0.1",
    port: u16 = 29092,
};

pub const InferenceOptions = struct {
    endpoint: Endpoint = .{},
    model: ?[]const u8 = null,
    prompt: []const u8 = "Hello",
    prompt_tokens: ?u32 = null,
    max_tokens: u32 = 16,
    tolerance: f32 = 0.25,
    raw_prompt: bool = false,
    thinking: bool = false,
    exact_tokens: bool = false,
    token_ids: bool = false,
    context: u32 = 4096,
    batch: u32 = 32,
    ubatch: u32 = 16,
    metrics: Metrics = .none,
};

pub const ServeOptions = struct {
    endpoint: Endpoint = .{},
    model: ?[]const u8 = null,
    listen_host: []const u8 = "127.0.0.1",
    listen_port: u16 = 8080,
    context: u32 = 4096,
    batch: u32 = 32,
    ubatch: u32 = 16,
};

pub const InspectOptions = struct {
    endpoint: Endpoint = .{},
};

pub const Command = union(enum) {
    run: InferenceOptions,
    serve: ServeOptions,
    benchmark_inference: InferenceOptions,
    benchmark_hardware: InferenceOptions,
    verify_logits: InferenceOptions,
    inspect_device: InspectOptions,
    help,
};

const InferenceMode = enum { run, benchmark_inference, benchmark_hardware, verify_logits };

pub fn parse(argv: []const []const u8) ParseError!Command {
    if (argv.len == 0) return .help;
    if (argv.len == 1 and isHelp(argv[0])) return .help;

    if (std.mem.eql(u8, argv[0], "run"))
        return if (containsHelp(argv[1..])) .help else .{ .run = try parseInference(argv[1..], .run) };
    if (std.mem.eql(u8, argv[0], "serve"))
        return if (containsHelp(argv[1..])) .help else .{ .serve = try parseServe(argv[1..]) };
    if (std.mem.eql(u8, argv[0], "benchmark")) {
        if (containsHelp(argv[1..])) return .help;
        if (argv.len < 2) return error.MissingValue;
        if (std.mem.eql(u8, argv[1], "inference"))
            return .{ .benchmark_inference = try parseInference(argv[2..], .benchmark_inference) };
        if (std.mem.eql(u8, argv[1], "hardware"))
            return .{ .benchmark_hardware = try parseInference(argv[2..], .benchmark_hardware) };
        return error.InvalidCommand;
    }
    if (std.mem.eql(u8, argv[0], "verify")) {
        if (containsHelp(argv[1..])) return .help;
        if (argv.len < 2) return error.MissingValue;
        if (!std.mem.eql(u8, argv[1], "logits")) return error.InvalidCommand;
        return .{ .verify_logits = try parseInference(argv[2..], .verify_logits) };
    }
    if (std.mem.eql(u8, argv[0], "inspect")) {
        if (containsHelp(argv[1..])) return .help;
        if (argv.len < 2) return error.MissingValue;
        if (!std.mem.eql(u8, argv[1], "device")) return error.InvalidCommand;
        return .{ .inspect_device = try parseInspect(argv[2..]) };
    }
    return error.InvalidCommand;
}

fn parseInference(argv: []const []const u8, mode: InferenceMode) ParseError!InferenceOptions {
    var options: InferenceOptions = .{};
    var i: usize = 0;
    while (i < argv.len) : (i += 1) {
        const arg = argv[i];
        if (isHelp(arg)) return error.InvalidOption;
        if (std.mem.eql(u8, arg, "-m") or std.mem.eql(u8, arg, "--model")) {
            options.model = try nextValue(argv, &i);
        } else if (inlineValue(arg, "--model")) |value| {
            options.model = value;
        } else if (std.mem.eql(u8, arg, "--device")) {
            options.endpoint = try parseEndpoint(try nextValue(argv, &i));
        } else if (inlineValue(arg, "--device")) |value| {
            options.endpoint = try parseEndpoint(value);
        } else if (std.mem.eql(u8, arg, "--prompt")) {
            options.prompt = try nextValue(argv, &i);
        } else if (inlineValue(arg, "--prompt")) |value| {
            options.prompt = value;
        } else if (std.mem.eql(u8, arg, "--prompt-tokens")) {
            options.prompt_tokens = try parseU32(try nextValue(argv, &i));
        } else if (inlineValue(arg, "--prompt-tokens")) |value| {
            options.prompt_tokens = try parseU32(value);
        } else if (std.mem.eql(u8, arg, "--max-tokens")) {
            options.max_tokens = try parseU32(try nextValue(argv, &i));
        } else if (inlineValue(arg, "--max-tokens")) |value| {
            options.max_tokens = try parseU32(value);
        } else if (std.mem.eql(u8, arg, "--context")) {
            options.context = try parseU32(try nextValue(argv, &i));
        } else if (inlineValue(arg, "--context")) |value| {
            options.context = try parseU32(value);
        } else if (std.mem.eql(u8, arg, "--batch")) {
            options.batch = try parseU32(try nextValue(argv, &i));
        } else if (inlineValue(arg, "--batch")) |value| {
            options.batch = try parseU32(value);
        } else if (std.mem.eql(u8, arg, "--ubatch")) {
            options.ubatch = try parseU32(try nextValue(argv, &i));
        } else if (inlineValue(arg, "--ubatch")) |value| {
            options.ubatch = try parseU32(value);
        } else if (std.mem.eql(u8, arg, "--metrics")) {
            options.metrics = try parseMetrics(try nextValue(argv, &i));
        } else if (inlineValue(arg, "--metrics")) |value| {
            options.metrics = try parseMetrics(value);
        } else if (std.mem.eql(u8, arg, "--tolerance")) {
            if (mode != .verify_logits) return error.InvalidOption;
            options.tolerance = try parseF32(try nextValue(argv, &i));
        } else if (inlineValue(arg, "--tolerance")) |value| {
            if (mode != .verify_logits) return error.InvalidOption;
            options.tolerance = try parseF32(value);
        } else if (std.mem.eql(u8, arg, "--raw-prompt")) {
            options.raw_prompt = true;
        } else if (std.mem.eql(u8, arg, "--think")) {
            options.thinking = true;
        } else if (std.mem.eql(u8, arg, "--exact-tokens")) {
            options.exact_tokens = true;
        } else if (std.mem.eql(u8, arg, "--token-ids")) {
            if (mode != .run) return error.InvalidOption;
            options.token_ids = true;
        } else {
            return error.InvalidOption;
        }
    }
    if (options.model == null or options.model.?.len == 0) return error.MissingModel;
    if (options.context == 0 or options.batch == 0 or options.ubatch == 0)
        return error.InvalidNumber;
    switch (mode) {
        .benchmark_inference => options.metrics = options.metrics.atLeast(.summary),
        .benchmark_hardware => options.metrics = options.metrics.atLeast(.summary),
        .run, .verify_logits => {},
    }
    return options;
}

fn parseServe(argv: []const []const u8) ParseError!ServeOptions {
    var options: ServeOptions = .{};
    var i: usize = 0;
    while (i < argv.len) : (i += 1) {
        const arg = argv[i];
        if (isHelp(arg)) return error.InvalidOption;
        if (std.mem.eql(u8, arg, "-m") or std.mem.eql(u8, arg, "--model")) {
            options.model = try nextValue(argv, &i);
        } else if (inlineValue(arg, "--model")) |value| {
            options.model = value;
        } else if (std.mem.eql(u8, arg, "--device")) {
            options.endpoint = try parseEndpoint(try nextValue(argv, &i));
        } else if (inlineValue(arg, "--device")) |value| {
            options.endpoint = try parseEndpoint(value);
        } else if (std.mem.eql(u8, arg, "--host")) {
            options.listen_host = try nextValue(argv, &i);
        } else if (inlineValue(arg, "--host")) |value| {
            options.listen_host = value;
        } else if (std.mem.eql(u8, arg, "--port")) {
            options.listen_port = try parseU16(try nextValue(argv, &i));
        } else if (inlineValue(arg, "--port")) |value| {
            options.listen_port = try parseU16(value);
        } else if (std.mem.eql(u8, arg, "--context")) {
            options.context = try parseU32(try nextValue(argv, &i));
        } else if (inlineValue(arg, "--context")) |value| {
            options.context = try parseU32(value);
        } else if (std.mem.eql(u8, arg, "--batch")) {
            options.batch = try parseU32(try nextValue(argv, &i));
        } else if (inlineValue(arg, "--batch")) |value| {
            options.batch = try parseU32(value);
        } else if (std.mem.eql(u8, arg, "--ubatch")) {
            options.ubatch = try parseU32(try nextValue(argv, &i));
        } else if (inlineValue(arg, "--ubatch")) |value| {
            options.ubatch = try parseU32(value);
        } else if (std.mem.eql(u8, arg, "--parallel")) {
            if (try parseU32(try nextValue(argv, &i)) != 1) return error.InvalidParallel;
        } else if (inlineValue(arg, "--parallel")) |value| {
            if (try parseU32(value) != 1) return error.InvalidParallel;
        } else if (std.mem.eql(u8, arg, "--metrics")) {
            if (try parseMetrics(try nextValue(argv, &i)) != .none)
                return error.InvalidMetrics;
        } else if (inlineValue(arg, "--metrics")) |value| {
            if (try parseMetrics(value) != .none) return error.InvalidMetrics;
        } else {
            return error.InvalidOption;
        }
    }
    if (options.model == null or options.model.?.len == 0) return error.MissingModel;
    if (options.listen_host.len == 0) return error.MissingValue;
    if (options.context == 0 or options.batch == 0 or options.ubatch == 0)
        return error.InvalidNumber;
    return options;
}

fn parseInspect(argv: []const []const u8) ParseError!InspectOptions {
    var options: InspectOptions = .{};
    var i: usize = 0;
    while (i < argv.len) : (i += 1) {
        const arg = argv[i];
        if (isHelp(arg)) return error.InvalidOption;
        if (std.mem.eql(u8, arg, "--device")) {
            options.endpoint = try parseEndpoint(try nextValue(argv, &i));
        } else if (inlineValue(arg, "--device")) |value| {
            options.endpoint = try parseEndpoint(value);
        } else {
            return error.InvalidOption;
        }
    }
    return options;
}

fn parseEndpoint(text: []const u8) ParseError!Endpoint {
    if (!std.mem.startsWith(u8, text, "tcp:")) return error.InvalidDevice;
    const rest = text["tcp:".len..];
    const separator = std.mem.lastIndexOfScalar(u8, rest, ':') orelse return error.InvalidDevice;
    const host = rest[0..separator];
    const port_text = rest[separator + 1 ..];
    if (host.len == 0 or port_text.len == 0) return error.InvalidDevice;
    const port = parseU16(port_text) catch return error.InvalidDevice;
    if (port == 0) return error.InvalidDevice;
    return .{
        .label = text,
        .host = host,
        .port = port,
    };
}

fn nextValue(argv: []const []const u8, index: *usize) ParseError![]const u8 {
    index.* += 1;
    if (index.* >= argv.len or argv[index.*].len == 0) return error.MissingValue;
    return argv[index.*];
}

fn inlineValue(arg: []const u8, name: []const u8) ?[]const u8 {
    if (arg.len <= name.len or arg[name.len] != '=' or !std.mem.eql(u8, arg[0..name.len], name))
        return null;
    return arg[name.len + 1 ..];
}

fn parseMetrics(value: []const u8) ParseError!Metrics {
    return std.meta.stringToEnum(Metrics, value) orelse error.InvalidMetrics;
}

fn parseU16(value: []const u8) ParseError!u16 {
    return std.fmt.parseInt(u16, value, 10) catch error.InvalidNumber;
}

fn parseU32(value: []const u8) ParseError!u32 {
    return std.fmt.parseInt(u32, value, 10) catch error.InvalidNumber;
}

fn parseF32(value: []const u8) ParseError!f32 {
    const parsed = std.fmt.parseFloat(f32, value) catch return error.InvalidNumber;
    if (!std.math.isFinite(parsed) or parsed < 0) return error.InvalidNumber;
    return parsed;
}

fn isHelp(arg: []const u8) bool {
    return std.mem.eql(u8, arg, "help") or std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h");
}

fn containsHelp(argv: []const []const u8) bool {
    for (argv) |arg| {
        if (isHelp(arg)) return true;
    }
    return false;
}

test "public command grammar validates the root and TCP device" {
    try std.testing.expectEqual(.help, try parse(&.{}));
    try std.testing.expectError(error.InvalidCommand, parse(&.{"unknown"}));
    try std.testing.expectError(error.InvalidDevice, parse(&.{ "run", "-m", "model.gguf", "--device", "fake" }));
    try std.testing.expectError(error.MissingModel, parse(&.{ "run", "--model=" }));
    try std.testing.expectError(error.InvalidDevice, parse(&.{ "run", "-m", "model.gguf", "--device", "tcp:board:0" }));
    try std.testing.expectError(error.InvalidNumber, parse(&.{ "run", "-m", "model.gguf", "--context", "0" }));
    try std.testing.expectError(error.InvalidNumber, parse(&.{ "benchmark", "inference", "-m", "model.gguf", "--batch", "0" }));
    try std.testing.expectError(error.InvalidNumber, parse(&.{ "verify", "logits", "-m", "model.gguf", "--ubatch", "0" }));
    try std.testing.expectEqual(.help, try parse(&.{ "run", "--help" }));
    try std.testing.expectEqual(.help, try parse(&.{ "serve", "-h" }));
    try std.testing.expectEqual(.help, try parse(&.{ "benchmark", "hardware", "help" }));
}

test "run parses TCP endpoint and stable metrics level" {
    const command = try parse(&.{
        "run",          "-m", "bonsai.gguf", "--device=tcp:kria:29092", "--prompt",       "hello",
        "--max-tokens", "7",  "--metrics",   "full",                    "--exact-tokens", "--token-ids",
    });
    const options = command.run;
    try std.testing.expectEqualStrings("bonsai.gguf", options.model.?);
    try std.testing.expectEqualStrings("kria", options.endpoint.host);
    try std.testing.expectEqual(@as(u16, 29092), options.endpoint.port);
    try std.testing.expectEqual(@as(u32, 7), options.max_tokens);
    try std.testing.expectEqual(@as(u32, 4096), options.context);
    try std.testing.expectEqual(.full, options.metrics);
    try std.testing.expect(options.exact_tokens);
    try std.testing.expect(options.token_ids);
}

test "benchmark modes enforce their minimum instrumentation" {
    const inference = try parse(&.{ "benchmark", "inference", "-m", "bonsai.gguf", "--metrics", "none" });
    try std.testing.expectEqual(.summary, inference.benchmark_inference.metrics);

    const hardware = try parse(&.{ "benchmark", "hardware", "-m", "bonsai.gguf", "--metrics", "none" });
    try std.testing.expectEqual(.summary, hardware.benchmark_hardware.metrics);
    const diagnostic = try parse(&.{ "benchmark", "hardware", "-m", "bonsai.gguf", "--metrics", "full" });
    try std.testing.expectEqual(.full, diagnostic.benchmark_hardware.metrics);
    try std.testing.expectError(error.InvalidOption, parse(&.{ "benchmark", "hardware", "-m", "bonsai.gguf", "--rows", "32" }));
}

test "serve fixes parallelism at one and separates device from listener" {
    const command = try parse(&.{
        "serve",    "-m",        "bonsai.gguf", "--device", "tcp:board:9000",
        "--host",   "0.0.0.0",   "--port",      "8088",     "--parallel",
        "1",        "--context", "513",         "--batch",  "64",
        "--ubatch", "8",
    });
    const options = command.serve;
    try std.testing.expectEqualStrings("board", options.endpoint.host);
    try std.testing.expectEqual(@as(u16, 9000), options.endpoint.port);
    try std.testing.expectEqualStrings("0.0.0.0", options.listen_host);
    try std.testing.expectEqual(@as(u16, 8088), options.listen_port);
    try std.testing.expectEqual(@as(u32, 513), options.context);
    try std.testing.expectEqual(@as(u32, 64), options.batch);
    try std.testing.expectEqual(@as(u32, 8), options.ubatch);
    _ = try parse(&.{ "serve", "-m", "bonsai.gguf", "--metrics", "none" });
    try std.testing.expectError(
        error.InvalidMetrics,
        parse(&.{ "serve", "-m", "bonsai.gguf", "--metrics", "summary" }),
    );
    try std.testing.expectError(
        error.InvalidMetrics,
        parse(&.{ "serve", "-m", "bonsai.gguf", "--metrics", "full" }),
    );
    try std.testing.expectError(error.InvalidParallel, parse(&.{ "serve", "-m", "bonsai.gguf", "--parallel", "2" }));
    try std.testing.expectError(error.InvalidNumber, parse(&.{ "serve", "-m", "bonsai.gguf", "--context", "0" }));
}

test "serve defaults to the development context" {
    const command = try parse(&.{ "serve", "-m", "bonsai.gguf" });
    try std.testing.expectEqual(@as(u32, 4096), command.serve.context);
}

test "verify and inspect require the documented two-word commands" {
    const verify = try parse(&.{ "verify", "logits", "-m", "bonsai.gguf", "--tolerance", "0.1" });
    try std.testing.expectApproxEqAbs(@as(f32, 0.1), verify.verify_logits.tolerance, 0.0001);
    _ = try parse(&.{ "inspect", "device", "--device", "tcp:kria:29092" });
    try std.testing.expectError(error.InvalidCommand, parse(&.{ "verify", "tokens" }));
    try std.testing.expectError(error.InvalidCommand, parse(&.{ "inspect", "capabilities" }));
}
