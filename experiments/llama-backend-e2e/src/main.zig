const std = @import("std");
const c = @import("c");
const build_options = @import("build_options");

const allocator = std.heap.page_allocator;

const E2eError = error{
    MissingModel,
    ModelLoadFailed,
    ContextInitFailed,
    TokenizeFailed,
    DecodeFailed,
    LogitsMissing,
    BackendNotInitialized,
    SupportsOpNotCalled,
    NoAcceptedOps,
    NoAcceptedMatmul,
    BackendComputeNotCalled,
    LogitMismatch,
};

const Counters = struct {
    init_backend_calls: usize = 0,
    supports_op_calls: usize = 0,
    accepted_ops: usize = 0,
    accepted_mul_mat: usize = 0,
    graph_compute_calls: usize = 0,
    alloc_buffer_calls: usize = 0,
    set_tensor_calls: usize = 0,
    get_tensor_calls: usize = 0,

    fn reset(self: *Counters) void {
        self.* = .{};
    }
};

var counters = Counters{};

const Dev = struct {
    name: [*:0]const u8,
    desc: [*:0]const u8,
    backend: c.ggml_backend = undefined,
    device: c.ggml_backend_device = undefined,
    reg: c.ggml_backend_reg = undefined,
    buft: c.ggml_backend_buffer_type = undefined,
    cpu_delegate: c.ggml_backend_t = null,

    fn wire(self: *Dev) void {
        self.reg = .{ .api_version = c.GGML_BACKEND_API_VERSION, .iface = reg_iface, .context = self };
        self.device = .{ .iface = device_iface, .reg = &self.reg, .context = self };
        self.buft = .{ .iface = buft_iface, .device = &self.device, .context = self };
        self.backend = .{ .guid = null, .iface = backend_iface, .device = &self.device, .context = self };
    }
};

var panzai = Dev{
    .name = "panzai-e2e",
    .desc = "panzai e2e experiment backend (CPU-delegating)",
};

fn devOf(ctx: ?*anyopaque) *Dev {
    return @ptrCast(@alignCast(ctx.?));
}

fn bufGetBase(buf: c.ggml_backend_buffer_t) callconv(.c) ?*anyopaque {
    return buf.?.*.context;
}

fn bufFree(buf: c.ggml_backend_buffer_t) callconv(.c) void {
    std.c.free(buf.?.*.context);
}

fn bufSetTensor(
    buf: c.ggml_backend_buffer_t,
    tensor: ?*c.ggml_tensor,
    data: ?*const anyopaque,
    offset: usize,
    size: usize,
) callconv(.c) void {
    _ = buf;
    counters.set_tensor_calls += 1;
    const dst: [*]u8 = @ptrCast(@alignCast(tensor.?.*.data.?));
    const src: [*]const u8 = @ptrCast(data.?);
    @memcpy(dst[offset .. offset + size], src[0..size]);
}

fn bufGetTensor(
    buf: c.ggml_backend_buffer_t,
    tensor: ?*const c.ggml_tensor,
    data: ?*anyopaque,
    offset: usize,
    size: usize,
) callconv(.c) void {
    _ = buf;
    counters.get_tensor_calls += 1;
    const src: [*]const u8 = @ptrCast(@alignCast(tensor.?.*.data.?));
    const dst: [*]u8 = @ptrCast(data.?);
    @memcpy(dst[0..size], src[offset .. offset + size]);
}

fn bufClear(buf: c.ggml_backend_buffer_t, value: u8) callconv(.c) void {
    const base: [*]u8 = @ptrCast(@alignCast(buf.?.*.context.?));
    @memset(base[0..buf.?.*.size], value);
}

const buffer_iface = c.ggml_backend_buffer_i{
    .free_buffer = &bufFree,
    .get_base = &bufGetBase,
    .init_tensor = null,
    .memset_tensor = null,
    .set_tensor = &bufSetTensor,
    .get_tensor = &bufGetTensor,
    .set_tensor_2d = null,
    .get_tensor_2d = null,
    .cpy_tensor = null,
    .clear = &bufClear,
    .reset = null,
};

fn buftGetName(buft: c.ggml_backend_buffer_type_t) callconv(.c) [*c]const u8 {
    return devOf(buft.?.*.context).name;
}

fn buftAllocBuffer(buft: c.ggml_backend_buffer_type_t, size: usize) callconv(.c) c.ggml_backend_buffer_t {
    counters.alloc_buffer_calls += 1;
    const mem = std.c.malloc(if (size == 0) 1 else size) orelse return null;
    return c.ggml_backend_buffer_init(buft, buffer_iface, mem, size);
}

fn buftGetAlignment(buft: c.ggml_backend_buffer_type_t) callconv(.c) usize {
    _ = buft;
    return 32;
}

fn buftIsHost(buft: c.ggml_backend_buffer_type_t) callconv(.c) bool {
    _ = buft;
    return true;
}

const buft_iface = c.ggml_backend_buffer_type_i{
    .get_name = &buftGetName,
    .alloc_buffer = &buftAllocBuffer,
    .get_alignment = &buftGetAlignment,
    .get_max_size = null,
    .get_alloc_size = null,
    .is_host = &buftIsHost,
};

fn backendGetName(backend: c.ggml_backend_t) callconv(.c) [*c]const u8 {
    return devOf(backend.?.*.context).name;
}

fn backendFree(backend: c.ggml_backend_t) callconv(.c) void {
    const d = devOf(backend.?.*.context);
    if (d.cpu_delegate) |cpu| {
        c.ggml_backend_free(cpu);
        d.cpu_delegate = null;
    }
}

fn backendGraphCompute(backend: c.ggml_backend_t, cgraph: ?*c.ggml_cgraph) callconv(.c) c.ggml_status {
    counters.graph_compute_calls += 1;
    const d = devOf(backend.?.*.context);
    if (d.cpu_delegate == null) {
        d.cpu_delegate = c.ggml_backend_cpu_init();
    }
    const cpu = d.cpu_delegate orelse return c.GGML_STATUS_FAILED;
    return c.ggml_backend_graph_compute(cpu, cgraph);
}

const backend_iface = c.ggml_backend_i{
    .get_name = &backendGetName,
    .free = &backendFree,
    .set_tensor_async = null,
    .get_tensor_async = null,
    .set_tensor_2d_async = null,
    .get_tensor_2d_async = null,
    .cpy_tensor_async = null,
    .synchronize = null,
    .graph_plan_create = null,
    .graph_plan_free = null,
    .graph_plan_update = null,
    .graph_plan_compute = null,
    .graph_compute = &backendGraphCompute,
    .event_record = null,
    .event_wait = null,
    .graph_optimize = null,
};

fn devGetName(dev: c.ggml_backend_dev_t) callconv(.c) [*c]const u8 {
    return devOf(dev.?.*.context).name;
}

fn devGetDescription(dev: c.ggml_backend_dev_t) callconv(.c) [*c]const u8 {
    return devOf(dev.?.*.context).desc;
}

fn devGetMemory(dev: c.ggml_backend_dev_t, free: ?*usize, total: ?*usize) callconv(.c) void {
    _ = dev;
    free.?.* = 512 * 1024 * 1024;
    total.?.* = 512 * 1024 * 1024;
}

fn devGetType(dev: c.ggml_backend_dev_t) callconv(.c) c.enum_ggml_backend_dev_type {
    _ = dev;
    return c.GGML_BACKEND_DEVICE_TYPE_GPU;
}

fn devGetProps(dev: c.ggml_backend_dev_t, props: [*c]c.ggml_backend_dev_props) callconv(.c) void {
    props.*.name = devGetName(dev);
    props.*.description = devGetDescription(dev);
    props.*.@"type" = devGetType(dev);
    devGetMemory(dev, &props.*.memory_free, &props.*.memory_total);
    props.*.caps = .{ .@"async" = false, .host_buffer = true, .buffer_from_host_ptr = false, .events = false };
}

fn devInitBackend(dev: c.ggml_backend_dev_t, params: [*c]const u8) callconv(.c) c.ggml_backend_t {
    _ = params;
    counters.init_backend_calls += 1;
    return &devOf(dev.?.*.context).backend;
}

fn devGetBufferType(dev: c.ggml_backend_dev_t) callconv(.c) c.ggml_backend_buffer_type_t {
    return &devOf(dev.?.*.context).buft;
}

fn devSupportsOp(dev: c.ggml_backend_dev_t, op: ?*const c.ggml_tensor) callconv(.c) bool {
    _ = dev;
    counters.supports_op_calls += 1;
    const t = op orelse return false;
    const ok = t.op != c.GGML_OP_NONE;
    if (ok) {
        counters.accepted_ops += 1;
        if (t.op == c.GGML_OP_MUL_MAT) counters.accepted_mul_mat += 1;
    }
    return ok;
}

fn devSupportsBuft(dev: c.ggml_backend_dev_t, buft: c.ggml_backend_buffer_type_t) callconv(.c) bool {
    _ = dev;
    const b = buft orelse return false;
    if (b.*.iface.is_host) |is_host| return is_host(buft);
    return false;
}

const device_iface = c.ggml_backend_device_i{
    .get_name = &devGetName,
    .get_description = &devGetDescription,
    .get_memory = &devGetMemory,
    .get_type = &devGetType,
    .get_props = &devGetProps,
    .init_backend = &devInitBackend,
    .get_buffer_type = &devGetBufferType,
    .get_host_buffer_type = null,
    .buffer_from_host_ptr = null,
    .supports_op = &devSupportsOp,
    .supports_buft = &devSupportsBuft,
    .offload_op = null,
    .event_new = null,
    .event_free = null,
    .event_synchronize = null,
};

fn regGetName(reg: c.ggml_backend_reg_t) callconv(.c) [*c]const u8 {
    return devOf(reg.?.*.context).name;
}

fn regDevCount(reg: c.ggml_backend_reg_t) callconv(.c) usize {
    _ = reg;
    return 1;
}

fn regDevGet(reg: c.ggml_backend_reg_t, index: usize) callconv(.c) c.ggml_backend_dev_t {
    _ = index;
    return &devOf(reg.?.*.context).device;
}

const reg_iface = c.ggml_backend_reg_i{
    .get_name = &regGetName,
    .get_device_count = &regDevCount,
    .get_device = &regDevGet,
    .get_proc_address = null,
};

const DecodeResult = struct {
    logits: []f32,
    vocab: usize,

    fn deinit(self: DecodeResult) void {
        allocator.free(self.logits);
    }
};

fn runDecode(model_path: [:0]const u8, use_panzai: bool) !DecodeResult {
    var model_params = c.llama_model_default_params();
    model_params.n_gpu_layers = if (use_panzai) 1 else 0;
    model_params.split_mode = c.LLAMA_SPLIT_MODE_LAYER;

    var devices = [_]c.ggml_backend_dev_t{ &panzai.device, null };
    if (use_panzai) {
        model_params.devices = &devices;
    }

    const model = c.llama_model_load_from_file(model_path.ptr, model_params) orelse return E2eError.ModelLoadFailed;
    defer c.llama_model_free(model);

    const vocab_ptr = c.llama_model_get_vocab(model) orelse return E2eError.ModelLoadFailed;
    const vocab_count: usize = @intCast(c.llama_vocab_n_tokens(vocab_ptr));

    var ctx_params = c.llama_context_default_params();
    ctx_params.n_ctx = 64;
    ctx_params.n_batch = 16;
    ctx_params.n_ubatch = 16;
    ctx_params.n_seq_max = 1;
    ctx_params.n_threads = 1;
    ctx_params.n_threads_batch = 1;
    ctx_params.flash_attn_type = c.LLAMA_FLASH_ATTN_TYPE_DISABLED;
    ctx_params.op_offload = true;

    const ctx = c.llama_init_from_model(model, ctx_params) orelse return E2eError.ContextInitFailed;
    defer c.llama_free(ctx);

    const prompt = "Hello";
    var tokens: [32]c.llama_token = undefined;
    const n_tokens_raw = c.llama_tokenize(
        vocab_ptr,
        prompt.ptr,
        prompt.len,
        &tokens,
        tokens.len,
        true,
        false,
    );
    if (n_tokens_raw <= 0) return E2eError.TokenizeFailed;
    const n_tokens: usize = @intCast(n_tokens_raw);

    var batch = c.llama_batch_init(@intCast(n_tokens), 0, 1);
    defer c.llama_batch_free(batch);
    batch.n_tokens = @intCast(n_tokens);

    for (0..n_tokens) |i| {
        batch.token[i] = tokens[i];
        batch.pos[i] = @intCast(i);
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = if (i + 1 == n_tokens) 1 else 0;
    }

    const rc = c.llama_decode(ctx, batch);
    if (rc != 0) {
        std.debug.print("llama_decode failed: rc={d}\n", .{rc});
        return E2eError.DecodeFailed;
    }

    const logits_ptr = c.llama_get_logits_ith(ctx, -1) orelse return E2eError.LogitsMissing;
    const logits = try allocator.alloc(f32, vocab_count);
    @memcpy(logits, logits_ptr[0..vocab_count]);

    return .{ .logits = logits, .vocab = vocab_count };
}

fn maxAbsDiff(a: []const f32, b: []const f32) f32 {
    var max: f32 = 0;
    for (a, b) |x, y| {
        const diff = @abs(x - y);
        if (diff > max) max = diff;
    }
    return max;
}

fn argModelPath() ![:0]const u8 {
    if (build_options.default_model_path.len == 0) return E2eError.MissingModel;
    return try allocator.dupeSentinel(u8, build_options.default_model_path, 0);
}

pub fn main() !void {
    panzai.wire();

    const model_path = try argModelPath();
    defer allocator.free(model_path);

    c.llama_backend_init();
    defer c.llama_backend_free();

    std.debug.print("llama backend e2e\nmodel: {s}\n", .{model_path});

    const cpu = try runDecode(model_path, false);
    defer cpu.deinit();
    std.debug.print("cpu baseline ok: vocab={d}\n", .{cpu.vocab});

    counters.reset();
    const accelerated = try runDecode(model_path, true);
    defer accelerated.deinit();

    if (cpu.vocab != accelerated.vocab) return E2eError.LogitMismatch;
    const diff = maxAbsDiff(cpu.logits, accelerated.logits);

    std.debug.print(
        "panzai counters: init_backend={d}, supports_op={d}, accepted={d}, accepted_mul_mat={d}, graph_compute={d}, alloc_buffer={d}, set_tensor={d}, get_tensor={d}\n",
        .{
            counters.init_backend_calls,
            counters.supports_op_calls,
            counters.accepted_ops,
            counters.accepted_mul_mat,
            counters.graph_compute_calls,
            counters.alloc_buffer_calls,
            counters.set_tensor_calls,
            counters.get_tensor_calls,
        },
    );
    std.debug.print("logits max_abs_diff={d:.6}\n", .{diff});

    if (counters.init_backend_calls == 0) return E2eError.BackendNotInitialized;
    if (counters.supports_op_calls == 0) return E2eError.SupportsOpNotCalled;
    if (counters.accepted_ops == 0) return E2eError.NoAcceptedOps;
    if (counters.accepted_mul_mat == 0) return E2eError.NoAcceptedMatmul;
    if (counters.graph_compute_calls == 0) return E2eError.BackendComputeNotCalled;
    if (diff > 0.0001) return E2eError.LogitMismatch;

    std.debug.print("PASS: llama_decode used the Zig ggml backend and matched CPU logits\n", .{});
}
