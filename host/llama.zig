const std = @import("std");
const c = @import("c");
const build_options = @import("build_options");
const backend_mod = @import("backend");
const census_mod = @import("census");
const link_mod = @import("link");

pub const Error = error{
    MissingModel,
    ModelPathTooLong,
    ModelLoadFailed,
    ContextInitFailed,
    TokenizeFailed,
    DecodeFailed,
    SamplerInitFailed,
    PieceDecodeFailed,
    BackendHandshakeFailed,
} || std.mem.Allocator.Error || std.Io.Writer.Error;

pub const Options = struct {
    model_path: []const u8 = build_options.default_model_path,
    prompt: []const u8 = "Hello",
    max_tokens: u32 = 16,
    census: bool = false,
    n_ctx: u32 = 128,
    n_batch: u32 = 32,
    n_ubatch: u32 = 16,
    threads: u32 = 1,
};

pub fn runPrompt(
    allocator: std.mem.Allocator,
    writer: *std.Io.Writer,
    link: link_mod.Client,
    options: Options,
) Error!void {
    if (options.model_path.len == 0) return error.MissingModel;

    const device = backend_mod.Device.create(allocator, link) catch return error.BackendHandshakeFailed;
    defer device.destroy();
    var census: census_mod.Census = .{};
    if (options.census) device.census = &census;

    const model_path = try allocator.dupeZ(u8, options.model_path);
    defer allocator.free(model_path);

    c.llama_backend_init();
    defer c.llama_backend_free();
    c.llama_log_set(&quietLog, null);

    var model_params = c.llama_model_default_params();
    model_params.n_gpu_layers = 999;
    model_params.split_mode = c.LLAMA_SPLIT_MODE_LAYER;
    var devices = [_]c.ggml_backend_dev_t{ device.ggmlDevice(), null };
    model_params.devices = &devices;

    const model = c.llama_model_load_from_file(model_path.ptr, model_params) orelse return error.ModelLoadFailed;
    defer c.llama_model_free(model);

    const vocab = c.llama_model_get_vocab(model) orelse return error.ModelLoadFailed;

    var ctx_params = c.llama_context_default_params();
    ctx_params.n_ctx = options.n_ctx;
    ctx_params.n_batch = options.n_batch;
    ctx_params.n_ubatch = options.n_ubatch;
    ctx_params.n_seq_max = 1;
    ctx_params.n_threads = @intCast(options.threads);
    ctx_params.n_threads_batch = @intCast(options.threads);
    ctx_params.flash_attn_type = c.LLAMA_FLASH_ATTN_TYPE_DISABLED;
    ctx_params.op_offload = true;

    const ctx = c.llama_init_from_model(model, ctx_params) orelse return error.ContextInitFailed;
    defer c.llama_free(ctx);

    const sampler = c.llama_sampler_init_greedy() orelse return error.SamplerInitFailed;
    defer c.llama_sampler_free(sampler);

    const max_prompt_tokens = @max(@as(usize, 8), options.prompt.len + 8);
    const tokens = try allocator.alloc(c.llama_token, max_prompt_tokens + options.max_tokens + 1);
    defer allocator.free(tokens);

    const n_prompt_raw = c.llama_tokenize(
        vocab,
        options.prompt.ptr,
        @intCast(options.prompt.len),
        tokens.ptr,
        @intCast(max_prompt_tokens),
        true,
        false,
    );
    if (n_prompt_raw <= 0) return error.TokenizeFailed;
    var n_past: usize = @intCast(n_prompt_raw);

    try decodeTokens(ctx, tokens[0..n_past], 0, true);

    var generated: u32 = 0;
    while (generated < options.max_tokens) : (generated += 1) {
        const token = c.llama_sampler_sample(sampler, ctx, -1);
        c.llama_sampler_accept(sampler, token);
        if (c.llama_vocab_is_eog(vocab, token)) break;

        if (!options.census) try writePiece(writer, vocab, token);
        tokens[n_past] = token;
        n_past += 1;
        try decodeTokens(ctx, tokens[n_past - 1 .. n_past], n_past - 1, true);
    }

    if (options.census) {
        try census.report(writer);
    } else {
        try writer.writeByte('\n');
    }
}

fn decodeTokens(ctx: *c.llama_context, tokens: []c.llama_token, start_pos: usize, want_logits: bool) Error!void {
    var batch = c.llama_batch_init(@intCast(tokens.len), 0, 1);
    defer c.llama_batch_free(batch);
    batch.n_tokens = @intCast(tokens.len);
    for (tokens, 0..) |token, i| {
        batch.token[i] = token;
        batch.pos[i] = @intCast(start_pos + i);
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = if (want_logits and i + 1 == tokens.len) 1 else 0;
    }
    if (c.llama_decode(ctx, batch) != 0) return error.DecodeFailed;
}

fn writePiece(writer: *std.Io.Writer, vocab: *const c.llama_vocab, token: c.llama_token) Error!void {
    var buf: [256]u8 = undefined;
    const n = c.llama_token_to_piece(vocab, token, @ptrCast(buf[0..].ptr), @intCast(buf.len), 0, false);
    if (n < 0) return error.PieceDecodeFailed;
    try writer.writeAll(buf[0..@intCast(n)]);
}

fn quietLog(level: c.enum_ggml_log_level, text: [*c]const u8, user_data: ?*anyopaque) callconv(.c) void {
    _ = level;
    _ = text;
    _ = user_data;
}
