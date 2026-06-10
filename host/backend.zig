const std = @import("std");
const c = @import("c");
const q1a8 = @import("q1a8");
const wire = @import("wire");
const profiling = @import("profiling");
const prof_report = @import("prof_report");
const trace_mod = @import("trace");
const link_mod = @import("link");
const lower = @import("lower");
const census_mod = @import("census");

const RemoteMagic: u64 = 0x7065_6e7a_6169_6275; // "penzaibu"
const MaxBindings = 8192;
const alignment = 64;
const fill_chunk = 4096;

pub const BackendError = error{
    HandshakeFailed,
    OutOfMemory,
};

pub const Counters = struct {
    alloc_buffer: usize = 0,
    init_tensor: usize = 0,
    view_bindings: usize = 0,
    normal_bindings: usize = 0,
    upload_bytes: usize = 0,
    download_bytes: usize = 0,
    graph_compute: usize = 0,
    lowered_commands: usize = 0,
    unsupported_graphs: usize = 0,
};

pub const Profile = struct {
    io: std.Io,
    model_load_ns: u64 = 0,
    context_init_ns: u64 = 0,
    prefill_ns: u64 = 0,
    first_decode_ns: u64 = 0,
    steady_decode_ns: u64 = 0,
    steady_decode_count: u64 = 0,
    generated_tokens: u64 = 0,
    alloc_count: u64 = 0,
    alloc_bytes: u64 = 0,
    alloc_ns: u64 = 0,
    upload_count: u64 = 0,
    upload_bytes: u64 = 0,
    upload_ns: u64 = 0,
    download_count: u64 = 0,
    download_bytes: u64 = 0,
    download_ns: u64 = 0,
    rg: prof_report.RunGraphTotals = .{},

    pub fn init(io: std.Io) Profile {
        return .{ .io = io };
    }

    pub fn now(self: *const Profile) u64 {
        return profiling.nowNs(self.io);
    }

    pub fn elapsedSince(self: *const Profile, start_ns: u64) u64 {
        return profiling.elapsed(start_ns, self.now());
    }

    pub fn recordAlloc(self: *Profile, nbytes: u64, ns: u64) void {
        self.alloc_count += 1;
        self.alloc_bytes += nbytes;
        self.alloc_ns += ns;
    }

    pub fn recordUpload(self: *Profile, nbytes: u64, ns: u64) void {
        self.upload_count += 1;
        self.upload_bytes += nbytes;
        self.upload_ns += ns;
    }

    pub fn recordDownload(self: *Profile, nbytes: u64, ns: u64) void {
        self.download_count += 1;
        self.download_bytes += nbytes;
        self.download_ns += ns;
    }

    pub fn recordRunGraph(self: *Profile, profiled: link_mod.ProfiledRunGraph) void {
        self.rg.record(profiled);
    }

    /// Wall clock as the sum of the phases the pretty report breaks out, so the
    /// phase table's percentages add to 100. alloc/download are excluded — they
    /// overlap setup and are sub-percent; the json format still carries them.
    fn wallNs(self: *const Profile) u64 {
        return self.model_load_ns + self.context_init_ns + self.upload_ns +
            self.prefill_ns + self.first_decode_ns + self.steady_decode_ns;
    }

    fn decodeNs(self: *const Profile) u64 {
        return self.first_decode_ns + self.steady_decode_ns;
    }

    pub fn report(
        self: *const Profile,
        writer: *std.Io.Writer,
        counters: Counters,
        fmt: prof_report.ProfFormat,
        model_path: []const u8,
        device_label: []const u8,
    ) std.Io.Writer.Error!void {
        switch (fmt) {
            .pretty => try self.reportPretty(writer, model_path, device_label),
            .json => try self.reportJson(writer, counters, model_path, device_label),
        }
    }

    fn reportPretty(self: *const Profile, writer: *std.Io.Writer, model_path: []const u8, device_label: []const u8) std.Io.Writer.Error!void {
        const wall_ns = self.wallNs();
        var buf: [32]u8 = undefined;

        try writer.print("penzai profile\n\n", .{});
        try writer.print("  model        {s}\n", .{modelName(model_path)});
        try writer.print("  device       {s}\n", .{device_label});
        try writer.print("  tokens       {d}\n", .{self.generated_tokens});
        try writer.print("  throughput   {d:.2} tok/s\n", .{prof_report.perSecond(self.steady_decode_count, self.steady_decode_ns)});
        try writer.print("  wall         {s}\n\n", .{prof_report.formatDuration(&buf, wall_ns)});

        try writer.print("phases             time    wall%\n", .{});
        try phaseRow(writer, "model load", self.model_load_ns, wall_ns, "");
        try phaseRow(writer, "context init", self.context_init_ns, wall_ns, "");

        var up_bytes_buf: [32]u8 = undefined;
        var up_buf: [96]u8 = undefined;
        const up_note = std.fmt.bufPrint(&up_buf, "{s}, {d} calls, {d:.1} MiB/s", .{
            prof_report.formatBytes(&up_bytes_buf, self.upload_bytes),
            self.upload_count,
            prof_report.mibPerSecond(self.upload_bytes, self.upload_ns),
        }) catch unreachable;
        try phaseRow(writer, "weight upload", self.upload_ns, wall_ns, up_note);

        try phaseRow(writer, "prefill", self.prefill_ns, wall_ns, "");

        var ttft_buf: [32]u8 = undefined;
        const ttft = prof_report.formatDuration(&ttft_buf, self.first_decode_ns);
        var steady_buf: [32]u8 = undefined;
        var dec_buf: [96]u8 = undefined;
        const dec_note = if (self.steady_decode_count == 0)
            std.fmt.bufPrint(&dec_buf, "{d} tok, TTFT {s}", .{ self.generated_tokens, ttft }) catch unreachable
        else
            std.fmt.bufPrint(&dec_buf, "{d} tok, TTFT {s}, then {s}/tok", .{
                self.generated_tokens,
                ttft,
                prof_report.formatDuration(&steady_buf, self.steady_decode_ns / self.steady_decode_count),
            }) catch unreachable;
        try phaseRow(writer, "decode", self.decodeNs(), wall_ns, dec_note);

        var total_buf: [32]u8 = undefined;
        try writer.print("  {s:<13} {s:>8}   100%\n\n", .{ "total", prof_report.formatDuration(&total_buf, wall_ns) });

        try prof_report.writeLinkSection(writer, &self.rg);
        try writer.writeByte('\n');
        try prof_report.writeOpTable(writer, &self.rg.op_totals, self.rg.device_total_ns);
    }

    /// Every counter, machine-readable. Built as a value tree and stringified by
    /// std.json so escaping and comma placement are the library's problem, not
    /// ours. Carries the fields the pretty view drops (alloc, download, the raw
    /// host/device timing triplet).
    fn reportJson(self: *const Profile, writer: *std.Io.Writer, counters: Counters, model_path: []const u8, device_label: []const u8) std.Io.Writer.Error!void {
        const OpRow = struct {
            op: []const u8,
            count: u32,
            total_ms: f64,
            avg_ms: f64,
            dev_pct: f64,
            bytes: u64,
            eff_mib_s: f64,
        };
        var op_rows: [profiling.max_op_tag + 1]OpRow = undefined;
        var op_count: usize = 0;
        for (self.rg.op_totals) |aggregate| {
            if (aggregate.count == 0) continue;
            op_rows[op_count] = .{
                .op = prof_report.opName(aggregate.tag),
                .count = aggregate.count,
                .total_ms = prof_report.nsToMs(aggregate.total_ns),
                .avg_ms = prof_report.avgMs(aggregate.total_ns, aggregate.count),
                .dev_pct = prof_report.percent(aggregate.total_ns, self.rg.device_total_ns),
                .bytes = aggregate.bytes,
                .eff_mib_s = prof_report.mibPerSecond(aggregate.bytes, aggregate.total_ns),
            };
            op_count += 1;
        }

        const payload = .{
            .model = modelName(model_path),
            .device = device_label,
            .tokens = self.generated_tokens,
            .throughput_tok_s = prof_report.perSecond(self.steady_decode_count, self.steady_decode_ns),
            .wall_ms = prof_report.nsToMs(self.wallNs()),
            .phases = .{
                .model_load_ms = prof_report.nsToMs(self.model_load_ns),
                .context_init_ms = prof_report.nsToMs(self.context_init_ns),
                .upload_ms = prof_report.nsToMs(self.upload_ns),
                .prefill_ms = prof_report.nsToMs(self.prefill_ns),
                .first_token_decode_ms = prof_report.nsToMs(self.first_decode_ns),
                .steady_decode_ms = prof_report.nsToMs(self.steady_decode_ns),
                .steady_decode_avg_ms = prof_report.avgMs(self.steady_decode_ns, self.steady_decode_count),
                .decode_ms = prof_report.nsToMs(self.decodeNs()),
            },
            .setup = .{
                .alloc_count = self.alloc_count,
                .alloc_bytes = self.alloc_bytes,
                .alloc_ms = prof_report.nsToMs(self.alloc_ns),
                .upload_count = self.upload_count,
                .upload_bytes = self.upload_bytes,
                .upload_ms = prof_report.nsToMs(self.upload_ns),
                .upload_mib_s = prof_report.mibPerSecond(self.upload_bytes, self.upload_ns),
                .download_count = self.download_count,
                .download_bytes = self.download_bytes,
                .download_ms = prof_report.nsToMs(self.download_ns),
            },
            .link = .{
                .graphs = self.rg.run_graph_count,
                .commands = self.rg.command_count,
                .host_rpc_ms = prof_report.nsToMs(self.rg.host_run_graph_ns),
                .device_total_ms = prof_report.nsToMs(self.rg.device_total_ns),
                .device_op_ms = prof_report.nsToMs(self.rg.opTotalNs()),
                .device_runtime_ms = prof_report.nsToMs(self.rg.deviceRuntimeNs()),
                .rpc_overhead_ms = prof_report.nsToMs(self.rg.rpcOverheadNs()),
                .request_bytes = self.rg.request_bytes,
                .response_bytes = self.rg.response_bytes,
                .spans_dropped = self.rg.span_dropped,
            },
            .backend = .{
                .graph_compute = counters.graph_compute,
                .lowered_commands = counters.lowered_commands,
                .upload_bytes = counters.upload_bytes,
                .download_bytes = counters.download_bytes,
            },
            .ops = op_rows[0..op_count],
        };

        try std.json.Stringify.value(payload, .{}, writer);
        try writer.writeByte('\n');
    }
};

/// One phase-table row: label, auto-unit time, share of wall, optional note.
fn phaseRow(writer: *std.Io.Writer, label: []const u8, ns: u64, wall_ns: u64, note: []const u8) std.Io.Writer.Error!void {
    var buf: [32]u8 = undefined;
    try writer.print("  {s:<13} {s:>8} {d:>5.0}%", .{ label, prof_report.formatDuration(&buf, ns), prof_report.percent(ns, wall_ns) });
    if (note.len > 0) try writer.print("    {s}", .{note});
    try writer.writeByte('\n');
}

/// Basename of the model path with the `.gguf` suffix removed, for the header.
fn modelName(path: []const u8) []const u8 {
    var name = path;
    if (std.mem.lastIndexOfScalar(u8, name, '/')) |i| name = name[i + 1 ..];
    if (std.mem.endsWith(u8, name, ".gguf")) name = name[0 .. name.len - 5];
    return name;
}

pub const Device = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    link: link_mod.Client,
    backend: c.ggml_backend = undefined,
    device: c.ggml_backend_device = undefined,
    reg: c.ggml_backend_reg = undefined,
    buft: c.ggml_backend_buffer_type = undefined,
    counters: Counters = .{},
    census: ?*census_mod.Census = null,
    profile: ?*Profile = null,
    trace: ?*trace_mod.Capture = null,
    name: [*:0]const u8 = "penzai",
    desc: [*:0]const u8 = "penzai remote tensor backend",

    // Device holds ggml vtable structs (reg/device/buft/backend) whose context
    // and cross-link fields point back into itself. It is therefore pinned:
    // heap it and pass it only as *Device; never copy or move it after wiring.
    pub fn create(allocator: std.mem.Allocator, link: link_mod.Client) BackendError!*Self {
        link.hello() catch return error.HandshakeFailed;
        const self = try allocator.create(Self);
        self.* = .{ .allocator = allocator, .link = link };
        self.wire();
        return self;
    }

    pub fn destroy(self: *Self) void {
        self.allocator.destroy(self);
    }

    pub fn ggmlDevice(self: *Self) c.ggml_backend_dev_t {
        return &self.device;
    }

    fn wire(self: *Self) void {
        self.reg = .{ .api_version = c.GGML_BACKEND_API_VERSION, .iface = reg_iface, .context = self };
        self.device = .{ .iface = device_iface, .reg = &self.reg, .context = self };
        self.buft = .{ .iface = buft_iface, .device = &self.device, .context = self };
        self.backend = .{ .guid = null, .iface = backend_iface, .device = &self.device, .context = self };
    }
};

/// A page reservation: virtual address space with no committed pages and no
/// access. ggml needs a stable base pointer for tensor->data arithmetic; the
/// real bytes live on-device under the remote handle.
const Reservation = struct {
    ptr: [*]align(std.heap.page_size_min) u8,
    len: usize,

    fn reserve(len: usize) ?Reservation {
        const n = @max(len, 1);
        const prot_none: std.posix.PROT = @bitCast(@as(u32, 0));
        const m = std.posix.mmap(
            null,
            n,
            prot_none,
            .{ .TYPE = .PRIVATE, .ANONYMOUS = true },
            -1,
            0,
        ) catch return null;
        return .{ .ptr = m.ptr, .len = n };
    }

    fn base(self: Reservation) *anyopaque {
        return self.ptr;
    }

    fn release(self: Reservation) void {
        std.posix.munmap(self.ptr[0..self.len]);
    }
};

const RemoteBinding = struct {
    tensor: *const c.ggml_tensor,
    range: wire.TensorRange,
    handle_nbytes: u64,
    tensor_nbytes: u64,
};

const RemoteBuffer = struct {
    magic: u64,
    dev: *Device,
    reservation: Reservation,
    remote: wire.TensorRange,
    bindings_len: usize,
    bindings: [MaxBindings]RemoteBinding,
};

fn devOf(ctx: ?*anyopaque) *Device {
    return @ptrCast(@alignCast(ctx.?));
}

fn remoteOf(buf: c.ggml_backend_buffer_t) *RemoteBuffer {
    return @ptrCast(@alignCast(buf.?.*.context.?));
}

fn findBindingIn(remote: *RemoteBuffer, tensor: *const c.ggml_tensor) ?RemoteBinding {
    for (remote.bindings[0..remote.bindings_len]) |binding| {
        if (binding.tensor == tensor) return binding;
    }
    return null;
}

fn appendBinding(remote: *RemoteBuffer, binding: RemoteBinding) bool {
    if (remote.bindings_len == MaxBindings) return false;
    remote.bindings[remote.bindings_len] = binding;
    remote.bindings_len += 1;
    return true;
}

fn findTensorBinding(tensor: ?*const c.ggml_tensor) ?RemoteBinding {
    const t = tensor orelse return null;
    const buf = t.*.buffer orelse return null;
    const remote = remoteOf(buf);
    if (remote.magic != RemoteMagic) return null;
    return findBindingIn(remote, t);
}

fn lookupBinding(ctx: *anyopaque, tensor: ?*const c.ggml_tensor) ?lower.Binding {
    _ = ctx;
    const binding = findTensorBinding(tensor) orelse return null;
    return .{ .range = binding.range, .handle_nbytes = binding.handle_nbytes };
}

fn tensorNbytes(tensor: *const c.ggml_tensor) usize {
    const raw = c.ggml_nbytes(@constCast(tensor));
    return @intCast(raw);
}

fn tensorElements(tensor: *const c.ggml_tensor) usize {
    const raw = c.ggml_nelements(@constCast(tensor));
    if (raw <= 0) return 0;
    return @intCast(raw);
}

fn effectiveAllocSize(tensor: *const c.ggml_tensor) usize {
    if (isRepackableQ1_0(tensor)) {
        const rows: usize = @intCast(dim(tensor, 1));
        const k: usize = @intCast(dim(tensor, 0));
        return q1a8.packedWeightBytes(rows, k) catch tensorNbytes(tensor);
    }
    return tensorNbytes(tensor);
}

fn isRepackableQ1_0(tensor: *const c.ggml_tensor) bool {
    return tensor.*.type == c.GGML_TYPE_Q1_0 and
        dim(tensor, 0) > 0 and
        dim(tensor, 1) > 0 and
        dim(tensor, 2) == 1 and
        dim(tensor, 3) == 1 and
        @mod(@as(usize, @intCast(dim(tensor, 0))), q1a8.q1_block) == 0;
}

fn shouldRepackQ1_0(tensor: *const c.ggml_tensor, offset: usize, size: usize) bool {
    return isRepackableQ1_0(tensor) and offset == 0 and size == tensorNbytes(tensor);
}

fn rangeValid(binding: RemoteBinding, offset: usize, size: usize) bool {
    const off: u64 = @intCast(offset);
    const len: u64 = @intCast(size);
    if (off > binding.tensor_nbytes or len > binding.tensor_nbytes - off) return false;
    if (binding.range.offset > binding.handle_nbytes) return false;
    if (off > binding.handle_nbytes - binding.range.offset) return false;
    return len <= binding.handle_nbytes - binding.range.offset - off;
}

fn tensorOffsetInBuffer(remote: *RemoteBuffer, tensor: *const c.ggml_tensor) ?usize {
    if (remote.remote.nbytes == 0) return if (effectiveAllocSize(tensor) == 0) 0 else null;
    const base = remote.reservation.base();
    const data = tensor.*.data orelse return null;
    const base_addr = @intFromPtr(base);
    const data_addr = @intFromPtr(data);
    if (data_addr < base_addr) return null;
    const offset = data_addr - base_addr;
    const size = effectiveAllocSize(tensor);
    if (offset > remote.remote.nbytes or size > remote.remote.nbytes - offset) return null;
    return offset;
}

fn bufGetBase(buf: c.ggml_backend_buffer_t) callconv(.c) ?*anyopaque {
    return remoteOf(buf).reservation.base();
}

fn bufFree(buf: c.ggml_backend_buffer_t) callconv(.c) void {
    const remote = remoteOf(buf);
    if (remote.remote.handle != 0) {
        remote.dev.link.free(remote.remote) catch {};
    }
    remote.reservation.release();
    std.c.free(remote);
}

fn bufInitTensor(buf: c.ggml_backend_buffer_t, tensor: ?*c.ggml_tensor) callconv(.c) c.ggml_status {
    const remote = remoteOf(buf);
    remote.dev.counters.init_tensor += 1;
    const t = tensor orelse return c.GGML_STATUS_FAILED;

    if (t.*.view_src) |view_src| {
        var src = findBindingIn(remote, view_src) orelse return c.GGML_STATUS_FAILED;
        src.tensor = t;
        src.range.offset += @intCast(t.*.view_offs);
        src.tensor_nbytes = tensorNbytes(t);
        src.range.nbytes = src.tensor_nbytes;
        if (!rangeValid(src, 0, tensorNbytes(t))) return c.GGML_STATUS_FAILED;
        if (!appendBinding(remote, src)) return c.GGML_STATUS_FAILED;
        remote.dev.counters.view_bindings += 1;
        return c.GGML_STATUS_SUCCESS;
    }

    const offset = tensorOffsetInBuffer(remote, t) orelse return c.GGML_STATUS_FAILED;
    const alloc_size = effectiveAllocSize(t);
    const binding: RemoteBinding = .{
        .tensor = t,
        .range = .{
            .handle = remote.remote.handle,
            .offset = @intCast(offset),
            .nbytes = @intCast(alloc_size),
        },
        .handle_nbytes = remote.remote.nbytes,
        .tensor_nbytes = @intCast(alloc_size),
    };
    if (!rangeValid(binding, 0, alloc_size)) return c.GGML_STATUS_FAILED;
    if (!appendBinding(remote, binding)) return c.GGML_STATUS_FAILED;
    remote.dev.counters.normal_bindings += 1;
    return c.GGML_STATUS_SUCCESS;
}

fn bufSetTensor(
    buf: c.ggml_backend_buffer_t,
    tensor: ?*c.ggml_tensor,
    data: ?*const anyopaque,
    offset: usize,
    size: usize,
) callconv(.c) void {
    const remote = remoteOf(buf);
    const t = tensor orelse return;
    const binding = findBindingIn(remote, t) orelse return;
    if (!rangeValid(binding, offset, size)) return;
    const src: [*]const u8 = @ptrCast(data orelse return);

    if (shouldRepackQ1_0(t, offset, size)) {
        const rows: usize = @intCast(dim(t, 1));
        const k: usize = @intCast(dim(t, 0));
        const packed_len = q1a8.packedWeightBytes(rows, k) catch return;
        const packed_weights = remote.dev.allocator.alloc(u8, packed_len) catch return;
        defer remote.dev.allocator.free(packed_weights);
        q1a8.packWeightsFromGgmlQ1_0(rows, k, src[0..size], packed_weights) catch return;
        timedUpload(remote.dev, binding.range, packed_weights) catch return;
        remote.dev.counters.upload_bytes += packed_weights.len;
        return;
    }

    const dst_range: wire.TensorRange = .{
        .handle = binding.range.handle,
        .offset = binding.range.offset + @as(u64, @intCast(offset)),
        .nbytes = @intCast(size),
    };
    timedUpload(remote.dev, dst_range, src[0..size]) catch return;
    remote.dev.counters.upload_bytes += size;
}

fn bufGetTensor(
    buf: c.ggml_backend_buffer_t,
    tensor: ?*const c.ggml_tensor,
    data: ?*anyopaque,
    offset: usize,
    size: usize,
) callconv(.c) void {
    const remote = remoteOf(buf);
    const binding = findBindingIn(remote, tensor orelse return) orelse return;
    if (!rangeValid(binding, offset, size)) return;
    const dst: [*]u8 = @ptrCast(data orelse return);
    const src_range: wire.TensorRange = .{
        .handle = binding.range.handle,
        .offset = binding.range.offset + @as(u64, @intCast(offset)),
        .nbytes = @intCast(size),
    };
    timedDownload(remote.dev, src_range, dst[0..size]) catch return;
    remote.dev.counters.download_bytes += size;
}

fn bufMemsetTensor(
    buf: c.ggml_backend_buffer_t,
    tensor: ?*c.ggml_tensor,
    value: u8,
    offset: usize,
    size: usize,
) callconv(.c) void {
    const remote = remoteOf(buf);
    const binding = findBindingIn(remote, tensor orelse return) orelse return;
    if (!rangeValid(binding, offset, size)) return;
    uploadFill(remote.dev, binding.range.handle, binding.range.offset + @as(u64, @intCast(offset)), size, value) catch return;
}

fn bufClear(buf: c.ggml_backend_buffer_t, value: u8) callconv(.c) void {
    const remote = remoteOf(buf);
    if (remote.remote.handle == 0 or remote.remote.nbytes == 0) return;
    uploadFill(remote.dev, remote.remote.handle, 0, @intCast(remote.remote.nbytes), value) catch return;
}

fn bufReset(buf: c.ggml_backend_buffer_t) callconv(.c) void {
    remoteOf(buf).bindings_len = 0;
}

fn uploadFill(dev: *Device, handle: u64, offset: u64, size: usize, value: u8) link_mod.LinkError!void {
    var buf: [fill_chunk]u8 = undefined;
    @memset(&buf, value);
    var done: usize = 0;
    while (done < size) {
        const n = @min(buf.len, size - done);
        try timedUpload(dev, .{
            .handle = handle,
            .offset = offset + @as(u64, @intCast(done)),
            .nbytes = @intCast(n),
        }, buf[0..n]);
        done += n;
    }
    dev.counters.upload_bytes += size;
}

fn timedAlloc(dev: *Device, nbytes: u64, tensor_alignment: u32) link_mod.LinkError!wire.TensorRange {
    const start_ns = if (dev.profile) |profile| profile.now() else 0;
    const range = try dev.link.alloc(nbytes, tensor_alignment);
    if (dev.profile) |profile| profile.recordAlloc(nbytes, profile.elapsedSince(start_ns));
    return range;
}

fn timedUpload(dev: *Device, range: wire.TensorRange, bytes: []const u8) link_mod.LinkError!void {
    const start_ns = if (dev.profile) |profile| profile.now() else 0;
    try dev.link.upload(range, bytes);
    if (dev.profile) |profile| profile.recordUpload(@intCast(bytes.len), profile.elapsedSince(start_ns));
}

fn timedDownload(dev: *Device, range: wire.TensorRange, out: []u8) link_mod.LinkError!void {
    const start_ns = if (dev.profile) |profile| profile.now() else 0;
    try dev.link.download(range, out);
    if (dev.profile) |profile| profile.recordDownload(@intCast(out.len), profile.elapsedSince(start_ns));
}

const buffer_iface = c.ggml_backend_buffer_i{
    .free_buffer = &bufFree,
    .get_base = &bufGetBase,
    .init_tensor = &bufInitTensor,
    .memset_tensor = &bufMemsetTensor,
    .set_tensor = &bufSetTensor,
    .get_tensor = &bufGetTensor,
    .set_tensor_2d = null,
    .get_tensor_2d = null,
    .cpy_tensor = null,
    .clear = &bufClear,
    .reset = &bufReset,
};

fn buftGetName(buft: c.ggml_backend_buffer_type_t) callconv(.c) [*c]const u8 {
    return devOf(buft.?.*.context).name;
}

fn buftAllocBuffer(buft: c.ggml_backend_buffer_type_t, size: usize) callconv(.c) c.ggml_backend_buffer_t {
    const dev = devOf(buft.?.*.context);
    dev.counters.alloc_buffer += 1;

    const reservation = Reservation.reserve(size) orelse return null;
    const remote_raw = std.c.malloc(@sizeOf(RemoteBuffer)) orelse {
        reservation.release();
        return null;
    };
    const remote: *RemoteBuffer = @ptrCast(@alignCast(remote_raw));
    const remote_range: wire.TensorRange = if (size == 0)
        .{ .handle = 0, .offset = 0, .nbytes = 0 }
    else
        timedAlloc(dev, @intCast(size), alignment) catch {
            reservation.release();
            std.c.free(remote_raw);
            return null;
        };
    remote.* = .{
        .magic = RemoteMagic,
        .dev = dev,
        .reservation = reservation,
        .remote = remote_range,
        .bindings_len = 0,
        .bindings = undefined,
    };

    const buffer = c.ggml_backend_buffer_init(buft, buffer_iface, remote, size);
    if (buffer == null) {
        if (remote_range.handle != 0) dev.link.free(remote_range) catch {};
        reservation.release();
        std.c.free(remote_raw);
        return null;
    }
    return buffer;
}

fn buftGetAlignment(buft: c.ggml_backend_buffer_type_t) callconv(.c) usize {
    _ = buft;
    return alignment;
}

fn buftGetAllocSize(buft: c.ggml_backend_buffer_type_t, tensor: ?*const c.ggml_tensor) callconv(.c) usize {
    _ = buft;
    return effectiveAllocSize(tensor orelse return 0);
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
    .get_alloc_size = &buftGetAllocSize,
    .is_host = &buftIsHost,
};

fn backendGetName(backend: c.ggml_backend_t) callconv(.c) [*c]const u8 {
    return devOf(backend.?.*.context).name;
}

fn backendFree(backend: c.ggml_backend_t) callconv(.c) void {
    _ = backend;
}

fn backendGraphCompute(backend: c.ggml_backend_t, graph: ?*c.ggml_cgraph) callconv(.c) c.ggml_status {
    const dev = devOf(backend.?.*.context);
    dev.counters.graph_compute += 1;
    const g = graph orelse return c.GGML_STATUS_FAILED;
    const lookup = lower.Lookup{ .ctx = backend.?.*.context.?, .findFn = lookupBinding };
    if (dev.census) |census| {
        census.recordGraph(g, lookup);
        return c.GGML_STATUS_SUCCESS;
    }

    const commands = lower.lowerGraph(dev.allocator, g, lookup) catch {
        dev.counters.unsupported_graphs += 1;
        return c.GGML_STATUS_FAILED;
    };
    defer dev.allocator.free(commands);
    if (commands.len == 0) return c.GGML_STATUS_SUCCESS;
    if (dev.profile) |profile| {
        // Trace requests per-command spans; otherwise aggregate-only.
        const tier: wire.ProfileTier = if (dev.trace != null) .trace else .aggregate;
        const host_base_ns = profile.now();
        var profiled = dev.link.runGraphProfile(commands, tier) catch return c.GGML_STATUS_FAILED;
        defer profiled.deinit();
        profile.recordRunGraph(profiled);
        if (dev.trace) |cap| cap.append(profiled.report, host_base_ns) catch {};
    } else {
        dev.link.runGraph(commands) catch return c.GGML_STATUS_FAILED;
    }
    dev.counters.lowered_commands += commands.len;
    return c.GGML_STATUS_SUCCESS;
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
    if (free) |p| p.* = 512 * 1024 * 1024;
    if (total) |p| p.* = 512 * 1024 * 1024;
}

fn devGetType(dev: c.ggml_backend_dev_t) callconv(.c) c.enum_ggml_backend_dev_type {
    _ = dev;
    return c.GGML_BACKEND_DEVICE_TYPE_GPU;
}

fn devGetProps(dev: c.ggml_backend_dev_t, props: [*c]c.ggml_backend_dev_props) callconv(.c) void {
    props.*.name = devGetName(dev);
    props.*.description = devGetDescription(dev);
    props.*.type = devGetType(dev);
    props.*.device_id = null;
    devGetMemory(dev, &props.*.memory_free, &props.*.memory_total);
    props.*.caps = .{ .async = false, .host_buffer = false, .buffer_from_host_ptr = false, .events = false };
}

fn devInitBackend(dev: c.ggml_backend_dev_t, params: [*c]const u8) callconv(.c) c.ggml_backend_t {
    _ = params;
    return &devOf(dev.?.*.context).backend;
}

fn devGetBufferType(dev: c.ggml_backend_dev_t) callconv(.c) c.ggml_backend_buffer_type_t {
    return &devOf(dev.?.*.context).buft;
}

fn devSupportsOp(dev: c.ggml_backend_dev_t, op: ?*const c.ggml_tensor) callconv(.c) bool {
    const d = devOf(dev.?.*.context);
    if (d.census != null) return op != null;
    return lower.supportsOp(op);
}

fn devSupportsBuft(dev: c.ggml_backend_dev_t, buft: c.ggml_backend_buffer_type_t) callconv(.c) bool {
    const d = devOf(dev.?.*.context);
    return buft == &d.buft;
}

fn devOffloadOp(dev: c.ggml_backend_dev_t, op: ?*const c.ggml_tensor) callconv(.c) bool {
    const d = devOf(dev.?.*.context);
    if (d.census != null) return op != null;
    const tensor = op orelse return false;
    return !lower.isMetadataOp(tensor.*.op) and lower.supportsOp(op);
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

fn dim(tensor: anytype, index: comptime_int) i64 {
    return tensor.*.ne[index];
}
