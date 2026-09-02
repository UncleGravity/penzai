//! Host command execution over the persistent TCP daemon.
const std = @import("std");
const build_options = @import("build_options");
const shared = @import("shared");
const link_mod = @import("link");
const llama_mod = if (build_options.enable_llama) @import("llama/llama.zig") else struct {
    pub const Error = error{};
};

const protocol_transport = shared.protocol_transport;

pub const RunError = error{LlamaDisabled} || link_mod.LinkError || llama_mod.Error;

pub const LlamaOptions = struct {
    model_path: []const u8 = build_options.default_model_path,
    prompt: []const u8 = "Hello",
    prompt_tokens: ?u32 = null,
    max_tokens: u32 = 16,
    logits_tolerance: f32 = 0.25,
    chat_template: bool = true,
    enable_thinking: bool = false,
    metrics_level: shared.engine.metrics.Level = .none,
    hardware_metrics: bool = false,
    device_label: []const u8 = "tcp:127.0.0.1:29092",
    exact_tokens: bool = false,
    token_ids: bool = false,
    n_ctx: u32 = 4096,
    n_batch: u32 = 32,
    n_ubatch: u32 = 16,
};

pub fn tcpCapabilities(
    io: std.Io,
    allocator: std.mem.Allocator,
    spec: protocol_transport.TcpSpec,
) RunError!shared.capabilities.Report {
    var link = try link_mod.TcpLink.connect(allocator, io, spec);
    defer link.deinit();
    return link.capabilities();
}

pub fn runTcpLlama(
    io: std.Io,
    allocator: std.mem.Allocator,
    writer: *std.Io.Writer,
    spec: protocol_transport.TcpSpec,
    options: LlamaOptions,
) RunError!void {
    if (!build_options.enable_llama) return error.LlamaDisabled;
    var link = try link_mod.TcpLink.connect(allocator, io, spec);
    defer link.deinit();
    return llama_mod.runPrompt(io, allocator, writer, link_mod.Client.init(&link), .{
        .model_path = options.model_path,
        .prompt = options.prompt,
        .prompt_tokens = options.prompt_tokens,
        .max_tokens = options.max_tokens,
        .logits_tolerance = options.logits_tolerance,
        .chat_template = options.chat_template,
        .enable_thinking = options.enable_thinking,
        .metrics = options.metrics_level,
        .hardware_metrics = options.hardware_metrics,
        .device_label = options.device_label,
        .exact_tokens = options.exact_tokens,
        .token_ids = options.token_ids,
        .n_ctx = options.n_ctx,
        .n_batch = options.n_batch,
        .n_ubatch = options.n_ubatch,
    });
}

pub fn runTcpLogitsCheck(
    io: std.Io,
    allocator: std.mem.Allocator,
    writer: *std.Io.Writer,
    spec: protocol_transport.TcpSpec,
    options: LlamaOptions,
) RunError!void {
    if (!build_options.enable_llama) return error.LlamaDisabled;
    var link = try link_mod.TcpLink.connect(allocator, io, spec);
    defer link.deinit();
    return llama_mod.runLogitsCheck(allocator, writer, link_mod.Client.init(&link), .{
        .model_path = options.model_path,
        .prompt = options.prompt,
        .prompt_tokens = options.prompt_tokens,
        .max_tokens = options.max_tokens,
        .logits_tolerance = options.logits_tolerance,
        .chat_template = options.chat_template,
        .enable_thinking = options.enable_thinking,
        .metrics = options.metrics_level,
        .hardware_metrics = options.hardware_metrics,
        .device_label = options.device_label,
        .exact_tokens = options.exact_tokens,
        .token_ids = options.token_ids,
        .n_ctx = options.n_ctx,
        .n_batch = options.n_batch,
        .n_ubatch = options.n_ubatch,
    });
}
