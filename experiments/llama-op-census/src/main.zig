const std = @import("std");
const c = @import("c");
const build_options = @import("build_options");

const allocator = std.heap.page_allocator;

const Error = error{
    MissingModel,
    ModelLoadFailed,
    ContextInitFailed,
    TokenizeFailed,
    DecodeFailed,
    LogitsMissing,
    NoCensus,
    NoPanzaiCompute,
    TooManyPatterns,
};

const MaxPatterns = 256;
const MaxName = 48;
const MaxTypeName = 16;
const MaxOpName = 24;

const Key = struct {
    op: c_int,
    dst_type: c_int,
    src_type: [4]c_int,
    dst_ne: [4]i64,
    src0_ne: [4]i64,
    src1_ne: [4]i64,

    fn eql(a: Key, b: Key) bool {
        return a.op == b.op and
            a.dst_type == b.dst_type and
            std.mem.eql(c_int, &a.src_type, &b.src_type) and
            std.mem.eql(i64, &a.dst_ne, &b.dst_ne) and
            std.mem.eql(i64, &a.src0_ne, &b.src0_ne) and
            std.mem.eql(i64, &a.src1_ne, &b.src1_ne);
    }
};

const Pattern = struct {
    key: Key,
    op_name: [MaxOpName]u8 = undefined,
    op_name_len: usize = 0,
    dst_type_name: [MaxTypeName]u8 = undefined,
    dst_type_name_len: usize = 0,
    src_type_name: [4][MaxTypeName]u8 = undefined,
    src_type_name_len: [4]usize = .{ 0, 0, 0, 0 },
    example_name: [MaxName]u8 = undefined,
    example_name_len: usize = 0,
    support_calls: usize = 0,
    eval_ask_calls: usize = 0,
    eval_observe_calls: usize = 0,
    panzai_compute_calls: usize = 0,
};

const Census = struct {
    patterns: [MaxPatterns]Pattern = undefined,
    len: usize = 0,
    overflow: usize = 0,
    support_calls: usize = 0,
    eval_ask_calls: usize = 0,
    eval_observe_calls: usize = 0,
    panzai_compute_calls: usize = 0,
    graph_compute_calls: usize = 0,
    accepted_ops: usize = 0,
    accepted_mul_mat: usize = 0,
    init_backend_calls: usize = 0,

    fn reset(self: *Census) void {
        self.* = .{};
    }
};

var census = Census{};

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
    .name = "panzai-census",
    .desc = "panzai op census backend (CPU-delegating)",
};

fn quietLog(level: c.enum_ggml_log_level, text: [*c]const u8, user_data: ?*anyopaque) callconv(.c) void {
    _ = level;
    _ = text;
    _ = user_data;
}

fn devOf(ctx: ?*anyopaque) *Dev {
    return @ptrCast(@alignCast(ctx.?));
}

fn copyCStr(comptime N: usize, dst: *[N]u8, src: [*c]const u8) usize {
    if (src == null) return 0;
    const slice = std.mem.span(src);
    const n = @min(slice.len, N);
    @memcpy(dst[0..n], slice[0..n]);
    return n;
}

fn shapeOf(t: anytype) [4]i64 {
    return .{ t.ne[0], t.ne[1], t.ne[2], t.ne[3] };
}

fn keyOf(t: *c.ggml_tensor) Key {
    var src_type: [4]c_int = .{ -1, -1, -1, -1 };
    var src0_ne: [4]i64 = .{ 0, 0, 0, 0 };
    var src1_ne: [4]i64 = .{ 0, 0, 0, 0 };

    for (0..4) |i| {
        const src = t.src[i];
        if (src != null) {
            src_type[i] = @intCast(src[0].@"type");
            if (i == 0) src0_ne = shapeOf(src[0]);
            if (i == 1) src1_ne = shapeOf(src[0]);
        }
    }

    return .{
        .op = @intCast(t.op),
        .dst_type = @intCast(t.@"type"),
        .src_type = src_type,
        .dst_ne = shapeOf(t.*),
        .src0_ne = src0_ne,
        .src1_ne = src1_ne,
    };
}

fn patternFor(t: *c.ggml_tensor) ?*Pattern {
    const key = keyOf(t);
    for (census.patterns[0..census.len]) |*pattern| {
        if (pattern.key.eql(key)) return pattern;
    }
    if (census.len == MaxPatterns) {
        census.overflow += 1;
        return null;
    }

    const index = census.len;
    census.len += 1;
    var pattern = &census.patterns[index];
    pattern.* = .{ .key = key };
    pattern.op_name_len = copyCStr(MaxOpName, &pattern.op_name, c.ggml_op_name(t.op));
    pattern.dst_type_name_len = copyCStr(MaxTypeName, &pattern.dst_type_name, c.ggml_type_name(t.@"type"));
    pattern.example_name_len = copyCStr(MaxName, &pattern.example_name, c.ggml_get_name(t));

    for (0..4) |i| {
        const src = t.src[i];
        if (src != null) {
            pattern.src_type_name_len[i] = copyCStr(MaxTypeName, &pattern.src_type_name[i], c.ggml_type_name(src[0].@"type"));
        } else {
            @memcpy(pattern.src_type_name[i][0..1], "-");
            pattern.src_type_name_len[i] = 1;
        }
    }

    return pattern;
}

const RecordKind = enum { support, eval_ask, eval_observe, panzai_compute };

fn recordNode(t: *c.ggml_tensor, kind: RecordKind) void {
    const pattern = patternFor(t) orelse return;
    switch (kind) {
        .support => {
            census.support_calls += 1;
            pattern.support_calls += 1;
        },
        .eval_ask => {
            census.eval_ask_calls += 1;
            pattern.eval_ask_calls += 1;
        },
        .eval_observe => {
            census.eval_observe_calls += 1;
            pattern.eval_observe_calls += 1;
        },
        .panzai_compute => {
            census.panzai_compute_calls += 1;
            pattern.panzai_compute_calls += 1;
        },
    }
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
    census.graph_compute_calls += 1;
    const g = cgraph.?;
    const n = c.ggml_graph_n_nodes(g);
    var i: c_int = 0;
    while (i < n) : (i += 1) {
        if (c.ggml_graph_node(g, i)) |node| recordNode(node, .panzai_compute);
    }

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
    census.init_backend_calls += 1;
    return &devOf(dev.?.*.context).backend;
}

fn devGetBufferType(dev: c.ggml_backend_dev_t) callconv(.c) c.ggml_backend_buffer_type_t {
    return &devOf(dev.?.*.context).buft;
}

fn devSupportsOp(dev: c.ggml_backend_dev_t, op: ?*const c.ggml_tensor) callconv(.c) bool {
    _ = dev;
    const t_const = op orelse return false;
    const t: *c.ggml_tensor = @constCast(t_const);
    recordNode(t, .support);
    if (t.op != c.GGML_OP_NONE) {
        census.accepted_ops += 1;
        if (t.op == c.GGML_OP_MUL_MAT) census.accepted_mul_mat += 1;
    }
    return true;
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

fn evalCallback(t: ?*c.ggml_tensor, ask: bool, user_data: ?*anyopaque) callconv(.c) bool {
    _ = user_data;
    const node = t orelse return true;
    recordNode(node, if (ask) .eval_ask else .eval_observe);
    return true;
}

fn runDecode(model_path: [:0]const u8) !void {
    var model_params = c.llama_model_default_params();
    model_params.n_gpu_layers = 1;
    model_params.split_mode = c.LLAMA_SPLIT_MODE_LAYER;

    var devices = [_]c.ggml_backend_dev_t{ &panzai.device, null };
    model_params.devices = &devices;

    const model = c.llama_model_load_from_file(model_path.ptr, model_params) orelse return Error.ModelLoadFailed;
    defer c.llama_model_free(model);

    const vocab_ptr = c.llama_model_get_vocab(model) orelse return Error.ModelLoadFailed;

    var ctx_params = c.llama_context_default_params();
    ctx_params.n_ctx = 64;
    ctx_params.n_batch = 16;
    ctx_params.n_ubatch = 16;
    ctx_params.n_seq_max = 1;
    ctx_params.n_threads = 1;
    ctx_params.n_threads_batch = 1;
    ctx_params.flash_attn_type = c.LLAMA_FLASH_ATTN_TYPE_DISABLED;
    ctx_params.op_offload = true;
    ctx_params.cb_eval = &evalCallback;
    ctx_params.cb_eval_user_data = null;

    const ctx = c.llama_init_from_model(model, ctx_params) orelse return Error.ContextInitFailed;
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
    if (n_tokens_raw <= 0) return Error.TokenizeFailed;
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
        return Error.DecodeFailed;
    }

    _ = c.llama_get_logits_ith(ctx, -1) orelse return Error.LogitsMissing;
}

fn argModelPath() ![:0]const u8 {
    if (build_options.default_model_path.len == 0) return Error.MissingModel;
    return try allocator.dupeSentinel(u8, build_options.default_model_path, 0);
}

fn printPattern(index: usize, p: *const Pattern) void {
    std.debug.print(
        "{d: >3} eval={d: >3} panzai={d: >3} support={d: >4} op={s: <12} dst={s: <5} dst_ne={any} src0={s: <5}{any} src1={s: <5}{any} ex={s}\n",
        .{
            index,
            p.eval_observe_calls,
            p.panzai_compute_calls,
            p.support_calls,
            p.op_name[0..p.op_name_len],
            p.dst_type_name[0..p.dst_type_name_len],
            p.key.dst_ne,
            p.src_type_name[0][0..p.src_type_name_len[0]],
            p.key.src0_ne,
            p.src_type_name[1][0..p.src_type_name_len[1]],
            p.key.src1_ne,
            p.example_name[0..p.example_name_len],
        },
    );
}

fn printCensus(model_path: [:0]const u8) void {
    std.debug.print("llama op census\nmodel: {s}\n", .{model_path});
    std.debug.print(
        "summary: unique_patterns={d}, support_calls={d}, eval_nodes={d}, panzai_nodes={d}, graph_compute_calls={d}, accepted_ops={d}, accepted_mul_mat={d}, overflow={d}\n",
        .{
            census.len,
            census.support_calls,
            census.eval_observe_calls,
            census.panzai_compute_calls,
            census.graph_compute_calls,
            census.accepted_ops,
            census.accepted_mul_mat,
            census.overflow,
        },
    );
    std.debug.print("\npatterns:\n", .{});
    for (census.patterns[0..census.len], 0..) |*pattern, i| {
        printPattern(i, pattern);
    }
}

pub fn main() !void {
    panzai.wire();

    const model_path = try argModelPath();
    defer allocator.free(model_path);

    c.llama_backend_init();
    defer c.llama_backend_free();
    c.llama_log_set(&quietLog, null);

    census.reset();
    try runDecode(model_path);
    printCensus(model_path);

    if (census.len == 0 or census.eval_observe_calls == 0) return Error.NoCensus;
    if (census.panzai_compute_calls == 0 or census.graph_compute_calls == 0) return Error.NoPanzaiCompute;
    if (census.overflow != 0) return Error.TooManyPatterns;
}
