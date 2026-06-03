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
    BackendNotInitialized,
    SupportsOpNotCalled,
    NoAcceptedMatmul,
    BackendComputeNotCalled,
    RemoteBufferNotAllocated,
    RemoteUploadMissing,
    RemoteDownloadMissing,
    TensorInitMissing,
    ViewBindingMissing,
    DryRunCommandMissing,
    DryRunBindingMissing,
    UnsupportedComputeOp,
    BindingOverflow,
    LogitMismatch,
};

const MaxBindings = 4096;
const RemoteMagic: u64 = 0x7061_6e7a_6169_6275; // "panzaibu"

const Counters = struct {
    init_backend_calls: usize = 0,
    supports_op_calls: usize = 0,
    accepted_mul_mat: usize = 0,
    accepted_metadata: usize = 0,
    accepted_none: usize = 0,
    rejected_ops: usize = 0,
    offload_op_calls: usize = 0,
    offloaded_mul_mat: usize = 0,
    graph_compute_calls: usize = 0,
    graph_nodes: usize = 0,
    graph_metadata_nodes: usize = 0,
    graph_mul_mat_nodes: usize = 0,
    graph_unexpected_nodes: usize = 0,
    dryrun_commands: usize = 0,
    dryrun_missing_bindings: usize = 0,
    init_tensor_calls: usize = 0,
    normal_bindings: usize = 0,
    view_bindings: usize = 0,
    metadata_bound_nodes: usize = 0,
    binding_overflow: usize = 0,
    alloc_buffer_calls: usize = 0,
    set_tensor_calls: usize = 0,
    get_tensor_calls: usize = 0,
    non_host_buffer_allocs: usize = 0,
    remote_alloc_bytes: usize = 0,
    remote_upload_bytes: usize = 0,
    remote_download_bytes: usize = 0,

    fn reset(self: *Counters) void {
        self.* = .{};
    }
};

var counters = Counters{};
var next_remote_handle: u64 = 1;

const RemoteBinding = struct {
    tensor: *const c.ggml_tensor,
    handle: u64,
    handle_nbytes: usize,
    tensor_nbytes: usize,
    remote_offset: usize,
    is_view: bool,
};

const FakeRemoteBuffer = struct {
    magic: u64,
    handle: u64,
    base: ?*anyopaque,
    size: usize,
    bindings_len: usize,
    bindings: [MaxBindings]RemoteBinding,
};

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
    .name = "panzai-bind-dryrun",
    .desc = "panzai remote binding + dry-run lowerer",
};

fn quietLog(level: c.enum_ggml_log_level, text: [*c]const u8, user_data: ?*anyopaque) callconv(.c) void {
    _ = level;
    _ = text;
    _ = user_data;
}

fn devOf(ctx: ?*anyopaque) *Dev {
    return @ptrCast(@alignCast(ctx.?));
}

fn remoteOf(buf: c.ggml_backend_buffer_t) *FakeRemoteBuffer {
    return @ptrCast(@alignCast(buf.?.*.context.?));
}

fn isMetadataOp(op: anytype) bool {
    return op == c.GGML_OP_NONE or
        op == c.GGML_OP_RESHAPE or
        op == c.GGML_OP_VIEW or
        op == c.GGML_OP_PERMUTE or
        op == c.GGML_OP_TRANSPOSE;
}

fn tensorNbytes(t: *const c.ggml_tensor) usize {
    return c.ggml_nbytes(@constCast(t));
}

fn rangeValid(binding: RemoteBinding, offset: usize, size: usize) bool {
    if (offset > binding.tensor_nbytes) return false;
    if (size > binding.tensor_nbytes - offset) return false;
    if (binding.remote_offset > binding.handle_nbytes) return false;
    const handle_avail = binding.handle_nbytes - binding.remote_offset;
    if (offset > handle_avail) return false;
    return size <= handle_avail - offset;
}

fn appendBinding(remote: *FakeRemoteBuffer, binding: RemoteBinding) bool {
    if (remote.bindings_len == MaxBindings) {
        counters.binding_overflow += 1;
        return false;
    }
    remote.bindings[remote.bindings_len] = binding;
    remote.bindings_len += 1;
    return true;
}

fn findBindingIn(remote: *FakeRemoteBuffer, tensor: *const c.ggml_tensor) ?*const RemoteBinding {
    for (remote.bindings[0..remote.bindings_len]) |*binding| {
        if (binding.tensor == tensor) return binding;
    }
    return null;
}

fn findBinding(tensor: ?*const c.ggml_tensor) ?*const RemoteBinding {
    const t = tensor orelse return null;
    const buf = t.*.buffer orelse return null;
    const remote = remoteOf(buf);
    if (remote.magic != RemoteMagic) return null;
    return findBindingIn(remote, t);
}

fn tensorOffsetInBuffer(remote: *FakeRemoteBuffer, tensor: *const c.ggml_tensor) ?usize {
    if (remote.size == 0) return if (tensorNbytes(tensor) == 0) 0 else null;
    const base = remote.base orelse return null;
    const data = tensor.*.data orelse return null;
    const base_addr = @intFromPtr(base);
    const data_addr = @intFromPtr(data);
    if (data_addr < base_addr) return null;
    const offset = data_addr - base_addr;
    const nbytes = tensorNbytes(tensor);
    if (offset > remote.size) return null;
    if (nbytes > remote.size - offset) return null;
    return offset;
}

fn bufGetBase(buf: c.ggml_backend_buffer_t) callconv(.c) ?*anyopaque {
    return remoteOf(buf).base;
}

fn bufFree(buf: c.ggml_backend_buffer_t) callconv(.c) void {
    const remote = remoteOf(buf);
    std.c.free(remote.base);
    std.c.free(remote);
}

fn bufInitTensor(buf: c.ggml_backend_buffer_t, tensor: ?*c.ggml_tensor) callconv(.c) c.ggml_status {
    counters.init_tensor_calls += 1;
    const t = tensor orelse return c.GGML_STATUS_FAILED;
    const remote = remoteOf(buf);

    if (t.*.view_src) |view_src| {
        const src = findBindingIn(remote, view_src) orelse return c.GGML_STATUS_FAILED;
        var binding = src.*;
        binding.tensor = t;
        binding.tensor_nbytes = tensorNbytes(t);
        binding.remote_offset += t.*.view_offs;
        binding.is_view = true;
        if (!rangeValid(binding, 0, binding.tensor_nbytes)) return c.GGML_STATUS_FAILED;
        if (!appendBinding(remote, binding)) return c.GGML_STATUS_FAILED;
        counters.view_bindings += 1;
        return c.GGML_STATUS_SUCCESS;
    }

    const nbytes = tensorNbytes(t);
    const offset = tensorOffsetInBuffer(remote, t) orelse return c.GGML_STATUS_FAILED;
    const binding = RemoteBinding{
        .tensor = t,
        .handle = remote.handle,
        .handle_nbytes = remote.size,
        .tensor_nbytes = nbytes,
        .remote_offset = offset,
        .is_view = false,
    };
    if (!rangeValid(binding, 0, nbytes)) return c.GGML_STATUS_FAILED;
    if (!appendBinding(remote, binding)) return c.GGML_STATUS_FAILED;
    counters.normal_bindings += 1;
    return c.GGML_STATUS_SUCCESS;
}

fn bufSetTensor(
    buf: c.ggml_backend_buffer_t,
    tensor: ?*c.ggml_tensor,
    data: ?*const anyopaque,
    offset: usize,
    size: usize,
) callconv(.c) void {
    counters.set_tensor_calls += 1;
    counters.remote_upload_bytes += size;
    const remote = remoteOf(buf);
    const binding = findBindingIn(remote, tensor orelse return) orelse return;
    if (!rangeValid(binding.*, offset, size)) return;
    const dst: [*]u8 = @ptrCast(@alignCast(remote.base.?));
    const src: [*]const u8 = @ptrCast(data.?);
    const remote_offset = binding.remote_offset + offset;
    @memcpy(dst[remote_offset .. remote_offset + size], src[0..size]);
}

fn bufGetTensor(
    buf: c.ggml_backend_buffer_t,
    tensor: ?*const c.ggml_tensor,
    data: ?*anyopaque,
    offset: usize,
    size: usize,
) callconv(.c) void {
    counters.get_tensor_calls += 1;
    counters.remote_download_bytes += size;
    const remote = remoteOf(buf);
    const binding = findBindingIn(remote, tensor orelse return) orelse return;
    if (!rangeValid(binding.*, offset, size)) return;
    const src: [*]const u8 = @ptrCast(@alignCast(remote.base.?));
    const dst: [*]u8 = @ptrCast(data.?);
    const remote_offset = binding.remote_offset + offset;
    @memcpy(dst[0..size], src[remote_offset .. remote_offset + size]);
}

fn bufClear(buf: c.ggml_backend_buffer_t, value: u8) callconv(.c) void {
    const remote = remoteOf(buf);
    const base: [*]u8 = @ptrCast(@alignCast(remote.base.?));
    @memset(base[0..remote.size], value);
}

const buffer_iface = c.ggml_backend_buffer_i{
    .free_buffer = &bufFree,
    .get_base = &bufGetBase,
    .init_tensor = &bufInitTensor,
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
    counters.non_host_buffer_allocs += 1;
    counters.remote_alloc_bytes += size;

    const shadow = std.c.malloc(if (size == 0) 1 else size) orelse return null;
    const remote_raw = std.c.malloc(@sizeOf(FakeRemoteBuffer)) orelse {
        std.c.free(shadow);
        return null;
    };
    const remote: *FakeRemoteBuffer = @ptrCast(@alignCast(remote_raw));
    remote.* = .{
        .magic = RemoteMagic,
        .handle = next_remote_handle,
        .base = shadow,
        .size = size,
        .bindings_len = 0,
        .bindings = undefined,
    };
    next_remote_handle += 1;

    return c.ggml_backend_buffer_init(buft, buffer_iface, remote, size);
}

fn buftGetAlignment(buft: c.ggml_backend_buffer_type_t) callconv(.c) usize {
    _ = buft;
    return 32;
}

fn buftIsHost(buft: c.ggml_backend_buffer_type_t) callconv(.c) bool {
    _ = buft;
    return false;
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

fn recordDryRunMatmul(node: *c.ggml_tensor) void {
    counters.graph_mul_mat_nodes += 1;
    const src0 = node.*.src[0];
    const src1 = node.*.src[1];
    const src0_b = findBinding(src0);
    const src1_b = findBinding(src1);
    const dst_b = findBinding(node);
    if (src0 == null or src1 == null or src0_b == null or src1_b == null or dst_b == null) {
        counters.dryrun_missing_bindings += 1;
        return;
    }
    counters.dryrun_commands += 1;
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
    const g = cgraph.?;
    const n = c.ggml_graph_n_nodes(g);
    var i: c_int = 0;
    while (i < n) : (i += 1) {
        const node = c.ggml_graph_node(g, i) orelse continue;
        counters.graph_nodes += 1;
        if (node.*.op == c.GGML_OP_MUL_MAT) {
            recordDryRunMatmul(node);
        } else if (isMetadataOp(node.*.op)) {
            counters.graph_metadata_nodes += 1;
            if (findBinding(node) != null) counters.metadata_bound_nodes += 1;
        } else {
            counters.graph_unexpected_nodes += 1;
            std.debug.print(
                "unexpected panzai graph node: op={s}, name={s}\n",
                .{ std.mem.span(c.ggml_op_name(node.*.op)), std.mem.span(c.ggml_get_name(node)) },
            );
        }
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
    props.*.caps = .{ .@"async" = false, .host_buffer = false, .buffer_from_host_ptr = false, .events = false };
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
    if (t.*.op == c.GGML_OP_MUL_MAT) {
        counters.accepted_mul_mat += 1;
        return true;
    }
    if (isMetadataOp(t.*.op)) {
        counters.accepted_metadata += 1;
        if (t.*.op == c.GGML_OP_NONE) counters.accepted_none += 1;
        return true;
    }
    counters.rejected_ops += 1;
    return false;
}

fn devOffloadOp(dev: c.ggml_backend_dev_t, op: ?*const c.ggml_tensor) callconv(.c) bool {
    _ = dev;
    counters.offload_op_calls += 1;
    const t = op orelse return false;
    if (t.*.op == c.GGML_OP_MUL_MAT) {
        counters.offloaded_mul_mat += 1;
        return true;
    }
    return false;
}

fn devSupportsBuft(dev: c.ggml_backend_dev_t, buft: c.ggml_backend_buffer_type_t) callconv(.c) bool {
    const d = devOf(dev.?.*.context);
    // Match the old pynqz1 backend: real lowerable commands must use panzai
    // buffers so every tensor has a remote binding. Accepting host buffers here
    // lets the scheduler hand graph_compute CPU-resident inputs.
    return buft == &d.buft;
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
    .offload_op = &devOffloadOp,
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

    const model = c.llama_model_load_from_file(model_path.ptr, model_params) orelse return Error.ModelLoadFailed;
    defer c.llama_model_free(model);

    const vocab_ptr = c.llama_model_get_vocab(model) orelse return Error.ModelLoadFailed;
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

    const logits_ptr = c.llama_get_logits_ith(ctx, -1) orelse return Error.LogitsMissing;
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
    if (build_options.default_model_path.len == 0) return Error.MissingModel;
    return try allocator.dupeSentinel(u8, build_options.default_model_path, 0);
}

pub fn main() !void {
    panzai.wire();

    const model_path = try argModelPath();
    defer allocator.free(model_path);

    c.llama_backend_init();
    defer c.llama_backend_free();
    c.llama_log_set(&quietLog, null);

    std.debug.print("llama binding lower dryrun\nmodel: {s}\n", .{model_path});

    const cpu = try runDecode(model_path, false);
    defer cpu.deinit();
    std.debug.print("cpu baseline ok: vocab={d}\n", .{cpu.vocab});

    counters.reset();
    const accelerated = try runDecode(model_path, true);
    defer accelerated.deinit();

    if (cpu.vocab != accelerated.vocab) return Error.LogitMismatch;
    const diff = maxAbsDiff(cpu.logits, accelerated.logits);

    std.debug.print(
        "panzai counters: init_backend={d}, supports_op={d}, accepted_mul_mat={d}, accepted_metadata={d}, accepted_none={d}, rejected={d}, offload_op={d}, offloaded_mul_mat={d}, graph_compute={d}, graph_nodes={d}, graph_metadata={d}, metadata_bound={d}, graph_mul_mat={d}, graph_unexpected={d}, dryrun_commands={d}, dryrun_missing_bindings={d}, init_tensor={d}, normal_bindings={d}, view_bindings={d}, binding_overflow={d}, alloc_buffer={d}, non_host_alloc={d}, set_tensor={d}, get_tensor={d}, remote_alloc_bytes={d}, remote_upload_bytes={d}, remote_download_bytes={d}\n",
        .{
            counters.init_backend_calls,
            counters.supports_op_calls,
            counters.accepted_mul_mat,
            counters.accepted_metadata,
            counters.accepted_none,
            counters.rejected_ops,
            counters.offload_op_calls,
            counters.offloaded_mul_mat,
            counters.graph_compute_calls,
            counters.graph_nodes,
            counters.graph_metadata_nodes,
            counters.metadata_bound_nodes,
            counters.graph_mul_mat_nodes,
            counters.graph_unexpected_nodes,
            counters.dryrun_commands,
            counters.dryrun_missing_bindings,
            counters.init_tensor_calls,
            counters.normal_bindings,
            counters.view_bindings,
            counters.binding_overflow,
            counters.alloc_buffer_calls,
            counters.non_host_buffer_allocs,
            counters.set_tensor_calls,
            counters.get_tensor_calls,
            counters.remote_alloc_bytes,
            counters.remote_upload_bytes,
            counters.remote_download_bytes,
        },
    );
    std.debug.print("logits max_abs_diff={d:.6}\n", .{diff});

    if (counters.init_backend_calls == 0) return Error.BackendNotInitialized;
    if (counters.supports_op_calls == 0) return Error.SupportsOpNotCalled;
    if (counters.accepted_mul_mat == 0) return Error.NoAcceptedMatmul;
    if (counters.graph_compute_calls == 0) return Error.BackendComputeNotCalled;
    if (counters.non_host_buffer_allocs == 0) return Error.RemoteBufferNotAllocated;
    if (counters.remote_upload_bytes == 0) return Error.RemoteUploadMissing;
    if (counters.remote_download_bytes == 0) return Error.RemoteDownloadMissing;
    if (counters.init_tensor_calls == 0 or counters.normal_bindings == 0) return Error.TensorInitMissing;
    if (counters.view_bindings == 0) return Error.ViewBindingMissing;
    if (counters.dryrun_commands == 0) return Error.DryRunCommandMissing;
    if (counters.dryrun_missing_bindings != 0) return Error.DryRunBindingMissing;
    if (counters.graph_unexpected_nodes != 0) return Error.UnsupportedComputeOp;
    if (counters.binding_overflow != 0) return Error.BindingOverflow;
    if (diff > 0.0001) return Error.LogitMismatch;

    std.debug.print("PASS: remote tensor bindings represented metadata and dry-run lowered matmul commands while matching CPU logits\n", .{});
}
