const std = @import("std");
const shared = @import("shared");
const link_mod = @import("link");

const profiling = shared.profiling;
const layout = shared.layout;

pub const AccountingStatus = enum {
    ok,
    warn,
    invalid,

    pub fn label(self: AccountingStatus) []const u8 {
        return switch (self) {
            .ok => "ok",
            .warn => "WARN",
            .invalid => "INVALID",
        };
    }

    pub fn combine(a: AccountingStatus, b: AccountingStatus) AccountingStatus {
        return @enumFromInt(@max(@intFromEnum(a), @intFromEnum(b)));
    }
};

const service_tolerance_per_graph_ns: u64 = 50_000;
const stage_tolerance_per_graph_ns: u64 = 50_000;
const execution_tolerance_per_command_ns: u64 = 1_000;

/// Host-side accumulation of run_graph profiles across many calls, shared by the
/// llama decode path and the bench harness. Domain-specific summaries (model load,
/// prefill, warmup/iters, …) live in the consumer; the run_graph + per-op rollup
/// and its rendering live here so there is exactly one of each.
pub const RunGraphTotals = struct {
    run_graph_count: u64 = 0,
    command_count: u64 = 0,
    host_run_graph_ns: u64 = 0,
    request_bytes: u64 = 0,
    response_bytes: u64 = 0,
    device_service_ns: u64 = 0,
    device_encode_ns: u64 = 0,
    /// Authoritative device RPC time: service plus response encoding.
    device_total_ns: u64 = 0,
    profiled_service_ns: u64 = 0,
    request_decode_ns: u64 = 0,
    preload_ns: u64 = 0,
    command_decode_ns: u64 = 0,
    device_execute_ns: u64 = 0,
    profile_encode_ns: u64 = 0,
    preload_bytes: u64 = 0,
    command_bytes: u64 = 0,
    accounting_violations: u32 = 0,
    matmul_bucket_overflow: u32 = 0,
    flash_bucket_overflow: u32 = 0,
    op_totals: [profiling.max_op_tag + 1]profiling.Aggregate = [_]profiling.Aggregate{.{}} ** (profiling.max_op_tag + 1),
    matmul_stats: [profiling.max_matmul_buckets]profiling.MatmulStat = [_]profiling.MatmulStat{.{}} ** profiling.max_matmul_buckets,
    flash_stats: [profiling.max_flash_buckets]profiling.FlashStat = [_]profiling.FlashStat{.{}} ** profiling.max_flash_buckets,
    matmul_count: usize = 0,
    flash_count: usize = 0,
    /// Constant across a run (the PL bitstream's fabric clock); 0 on a PS-only or
    /// fake run. Last-wins, ignoring zeros so a stray PS run_graph can't clear it.
    device_fclk_hz: u32 = 0,

    pub fn record(self: *RunGraphTotals, profiled: link_mod.ProfiledRunGraph) void {
        self.run_graph_count += 1;
        self.command_count += profiled.report.summary.command_count;
        self.host_run_graph_ns += profiled.rpc.round_trip_ns;
        self.request_bytes += profiled.rpc.request_bytes;
        self.response_bytes += profiled.rpc.response_bytes;
        self.device_service_ns +|= profiled.rpc.device_service_ns;
        self.device_encode_ns +|= profiled.rpc.device_encode_ns;
        self.device_total_ns +|= profiled.rpc.deviceRpcNs();
        self.profiled_service_ns +|= profiled.report.summary.profile_span_ns;
        self.request_decode_ns +|= profiled.report.summary.request_decode_ns;
        self.preload_ns +|= profiled.report.summary.preload_ns;
        self.command_decode_ns +|= profiled.report.summary.command_decode_ns;
        self.device_execute_ns += profiled.report.summary.execute_ns;
        self.profile_encode_ns +|= profiled.report.summary.profile_encode_ns;
        self.preload_bytes +|= profiled.report.summary.preload_bytes;
        self.command_bytes +|= profiled.report.summary.command_bytes;
        self.accounting_violations |= profiled.report.summary.accounting_violations;
        self.matmul_bucket_overflow +|= profiled.report.summary.matmul_bucket_overflow;
        self.flash_bucket_overflow +|= profiled.report.summary.flash_bucket_overflow;
        if (profiled.rpc.round_trip_ns < profiled.rpc.deviceRpcNs() or
            profiled.rpc.device_service_ns < profiled.report.summary.profile_span_ns or
            profiled.report.summary.profile_span_ns < profiled.report.summary.stagesNs())
        {
            self.accounting_violations |= profiling.AccountingViolation.rpc_budget;
        }
        for (profiled.report.aggregates) |aggregate| {
            const index: usize = aggregate.tag;
            if (index >= self.op_totals.len) continue;
            var total = &self.op_totals[index];
            total.tag = aggregate.tag;
            total.count += aggregate.count;
            total.total_ns += aggregate.total_ns;
            total.bytes += aggregate.bytes;
        }
        for (profiled.report.matmul_stats) |stat| {
            const total = self.matmulBucket(stat) orelse continue;
            addMatmul(total, stat);
        }
        for (profiled.report.flash_stats) |stat| {
            const total = self.flashBucket(stat) orelse continue;
            addFlash(total, stat);
        }
        if (profiled.report.summary.device_fclk_hz != 0) self.device_fclk_hz = profiled.report.summary.device_fclk_hz;
    }

    pub fn usedMatmul(self: *const RunGraphTotals) []const profiling.MatmulStat {
        return self.matmul_stats[0..self.matmul_count];
    }

    pub fn usedFlash(self: *const RunGraphTotals) []const profiling.FlashStat {
        return self.flash_stats[0..self.flash_count];
    }

    fn matmulBucket(self: *RunGraphTotals, incoming: profiling.MatmulStat) ?*profiling.MatmulStat {
        for (self.matmul_stats[0..self.matmul_count]) |*stat| if (stat.sameKey(incoming)) return stat;
        if (self.matmul_count == self.matmul_stats.len) {
            self.matmul_bucket_overflow +|= incoming.count;
            return null;
        }
        const stat = &self.matmul_stats[self.matmul_count];
        stat.* = .{ .backend = incoming.backend, .path = incoming.path, .fmt = incoming.fmt, .rows = incoming.rows, .cols = incoming.cols, .k = incoming.k };
        self.matmul_count += 1;
        return stat;
    }

    fn flashBucket(self: *RunGraphTotals, incoming: profiling.FlashStat) ?*profiling.FlashStat {
        for (self.flash_stats[0..self.flash_count]) |*stat| if (stat.sameKey(incoming)) return stat;
        if (self.flash_count == self.flash_stats.len) {
            self.flash_bucket_overflow +|= incoming.count;
            return null;
        }
        const stat = &self.flash_stats[self.flash_count];
        stat.* = .{ .backend = incoming.backend, .path = incoming.path, .n_heads = incoming.n_heads, .n_head_kv = incoming.n_head_kv, .head_dim_q = incoming.head_dim_q, .head_dim_v = incoming.head_dim_v, .n_tokens = incoming.n_tokens };
        self.flash_count += 1;
        return stat;
    }

    pub fn opTotalNs(self: *const RunGraphTotals) u64 {
        var total: u64 = 0;
        for (self.op_totals) |aggregate| total += aggregate.total_ns;
        return total;
    }

    pub fn transportNs(self: *const RunGraphTotals) u64 {
        return if (self.host_run_graph_ns >= self.device_total_ns)
            self.host_run_graph_ns - self.device_total_ns
        else
            0;
    }

    pub fn profileStagesNs(self: *const RunGraphTotals) u64 {
        return self.request_decode_ns +| self.preload_ns +| self.command_decode_ns +|
            self.device_execute_ns +| self.profile_encode_ns;
    }

    pub fn structurallyValid(self: *const RunGraphTotals) bool {
        return self.accounting_violations == 0 and
            self.host_run_graph_ns >= self.device_total_ns and
            self.device_service_ns >= self.profiled_service_ns and
            self.profiled_service_ns >= self.profileStagesNs() and
            self.device_execute_ns >= self.opTotalNs();
    }

    pub fn serviceResidualNs(self: *const RunGraphTotals) u64 {
        return checkedResidual(self.device_service_ns, self.profiled_service_ns);
    }

    pub fn stageResidualNs(self: *const RunGraphTotals) u64 {
        return checkedResidual(self.profiled_service_ns, self.profileStagesNs());
    }

    pub fn executionResidualNs(self: *const RunGraphTotals) u64 {
        return checkedResidual(self.device_execute_ns, self.opTotalNs());
    }

    pub fn accountingStatus(self: *const RunGraphTotals) AccountingStatus {
        if (!self.structurallyValid()) return .invalid;
        if (!residualWithinTolerance(self.device_service_ns, self.serviceResidualNs(), service_tolerance_per_graph_ns *| self.run_graph_count) or
            !residualWithinTolerance(self.profiled_service_ns, self.stageResidualNs(), stage_tolerance_per_graph_ns *| self.run_graph_count) or
            !residualWithinTolerance(self.device_execute_ns, self.executionResidualNs(), execution_tolerance_per_command_ns *| self.command_count))
        {
            return .warn;
        }
        return .ok;
    }

    pub fn accountingValid(self: *const RunGraphTotals) bool {
        return self.accountingStatus() == .ok;
    }

    /// Device time the profiler attributed to no op span (dispatch/scheduling).
    pub fn deviceRuntimeNs(self: *const RunGraphTotals) u64 {
        return self.executionResidualNs();
    }
};

fn checkedResidual(parent: u64, child: u64) u64 {
    return if (parent >= child) parent - child else 0;
}

fn residualWithinTolerance(parent: u64, residual: u64, absolute_tolerance_ns: u64) bool {
    return residual <= @max(parent / 100, absolute_tolerance_ns);
}

fn addMatmul(dst: *profiling.MatmulStat, src: profiling.MatmulStat) void {
    inline for (.{ "count", "kernel_runs" }) |field| @field(dst, field) +|= @field(src, field);
    inline for (.{ "macs", "command_ns", "wrapper_ns", "quantize_pack_ns", "sync_to_ns", "setup_ns", "wait_ns", "sync_from_ns", "result_layout_ns", "cycles", "w_stall_cycles", "a_stall_cycles", "r_stall_cycles", "w_beats", "a_beats", "r_beats" }) |field| @field(dst, field) +|= @field(src, field);
}

fn addFlash(dst: *profiling.FlashStat, src: profiling.FlashStat) void {
    inline for (.{ "count", "kernel_runs" }) |field| @field(dst, field) +|= @field(src, field);
    inline for (.{ "command_ns", "wrapper_ns", "prepare_ns", "sync_to_ns", "setup_ns", "wait_ns", "sync_from_ns", "result_layout_ns", "requested_n_kv_sum", "valid_n_kv_sum", "processed_n_kv_sum", "requested_qkv_pairs", "valid_qkv_pairs", "processed_qkv_pairs", "q_bytes", "k_bytes", "v_bytes", "mask_bytes", "o_bytes", "cycles", "q_beats", "k_beats", "k_stall_cycles", "v_beats", "v_stall_cycles", "o_beats", "o_stall_cycles" }) |field| @field(dst, field) +|= @field(src, field);
    dst.requested_n_kv_max = @max(dst.requested_n_kv_max, src.requested_n_kv_max);
    dst.valid_n_kv_max = @max(dst.valid_n_kv_max, src.valid_n_kv_max);
    dst.processed_n_kv_max = @max(dst.processed_n_kv_max, src.processed_n_kv_max);
}

/// A run phase. Every nanosecond of host wall time belongs to exactly one phase,
/// and within a phase the budget closes: wall = device + transport + residual
/// (see `PhaseAccum`). Setup splits into model_load / context_init because their
/// link traffic (weight upload vs KV alloc/fill) has very different shape.
pub const Phase = enum {
    model_load,
    context_init,
    prefill,
    decode,

    pub fn label(self: Phase) []const u8 {
        return switch (self) {
            .model_load => "model load",
            .context_init => "context init",
            .prefill => "prefill",
            .decode => "decode",
        };
    }
};

pub const phase_count = @typeInfo(Phase).@"enum".fields.len;

/// One kind of link op (alloc/upload/fill/download/free) accrued within a phase.
/// `host_ns` is the host-measured round trip; `device_ns` is the device's own
/// service time (from `wire.ResponseMeta`). Their difference is transport.
pub const LinkOp = struct {
    count: u64 = 0,
    bytes: u64 = 0,
    host_ns: u64 = 0,
    device_ns: u64 = 0,
    accounting_valid: bool = true,

    pub fn record(self: *LinkOp, bytes: u64, host_ns: u64, device_ns: u64) void {
        self.count += 1;
        self.bytes += bytes;
        self.host_ns += host_ns;
        self.device_ns += device_ns;
        if (host_ns < device_ns) self.accounting_valid = false;
    }

    pub fn name(kind: Kind) []const u8 {
        return switch (kind) {
            .alloc => "alloc",
            .upload => "upload",
            .fill => "fill",
            .download => "download",
            .free => "free",
        };
    }

    pub const Kind = enum { alloc, upload, fill, download, free };
};

/// Per-phase accounting. The closed identity is the whole point: a phase's wall
/// time is exactly the device work it triggered, plus wire transport, plus the
/// host residual (llama graph-build / sampling / file I/O — everything that is
/// not a link op). No bucket is dropped; `residualNs` is what makes that true.
pub const PhaseAccum = struct {
    wall_ns: u64 = 0,
    alloc: LinkOp = .{},
    upload: LinkOp = .{},
    fill: LinkOp = .{},
    download: LinkOp = .{},
    free: LinkOp = .{},
    rg: RunGraphTotals = .{},

    pub fn op(self: *PhaseAccum, kind: LinkOp.Kind) *LinkOp {
        return switch (kind) {
            .alloc => &self.alloc,
            .upload => &self.upload,
            .fill => &self.fill,
            .download => &self.download,
            .free => &self.free,
        };
    }

    /// Σ host round trip over every link op (transfers + run_graph). Sequential
    /// in this synchronous protocol, so it never exceeds wall.
    pub fn linkHostNs(self: *const PhaseAccum) u64 {
        return self.alloc.host_ns + self.upload.host_ns + self.fill.host_ns +
            self.download.host_ns + self.free.host_ns + self.rg.host_run_graph_ns;
    }

    /// Σ device service over every link op.
    pub fn deviceNs(self: *const PhaseAccum) u64 {
        return self.alloc.device_ns + self.upload.device_ns + self.fill.device_ns +
            self.download.device_ns + self.free.device_ns + self.rg.device_total_ns;
    }

    pub fn transportNs(self: *const PhaseAccum) u64 {
        return if (self.linkHostNs() >= self.deviceNs()) self.linkHostNs() - self.deviceNs() else 0;
    }

    pub fn residualNs(self: *const PhaseAccum) u64 {
        return if (self.wall_ns >= self.linkHostNs()) self.wall_ns - self.linkHostNs() else 0;
    }

    pub fn accountingStatus(self: *const PhaseAccum) AccountingStatus {
        const structurally_valid = self.linkHostNs() >= self.deviceNs() and self.wall_ns >= self.linkHostNs() and
            self.alloc.accounting_valid and self.upload.accounting_valid and self.fill.accounting_valid and
            self.download.accounting_valid and self.free.accounting_valid;
        if (!structurally_valid) return .invalid;
        return self.rg.accountingStatus();
    }

    pub fn accountingValid(self: *const PhaseAccum) bool {
        return self.accountingStatus() == .ok;
    }

    pub fn isEmpty(self: *const PhaseAccum) bool {
        return self.wall_ns == 0 and self.linkHostNs() == 0;
    }
};

/// Residency diagnostic: per-(phase, tensor) upload tally. Answers "what is
/// being uploaded, and in which phase" — e.g. a weight that shows up under the
/// decode phase is being re-sent every token (a residency bug), distinct from
/// the same weight legitimately uploaded once under model load. Tensor names are
/// normalized so per-layer tensors aggregate into one row (see `normalizeName`).
/// ggml-agnostic: it takes a name string, so this module never imports ggml.
pub const upload_census_cap = 48;
const upload_key_max = 40;

pub const UploadBucket = struct {
    phase: u8 = 0,
    key: [upload_key_max]u8 = undefined,
    key_len: u8 = 0,
    count: u64 = 0,
    bytes: u64 = 0,
};

pub const UploadCensus = struct {
    buckets: [upload_census_cap]UploadBucket = [_]UploadBucket{.{}} ** upload_census_cap,
    len: usize = 0,
    other_count: u64 = 0,
    other_bytes: u64 = 0,

    pub fn record(self: *UploadCensus, phase: u8, name: []const u8, nbytes: u64) void {
        var keybuf: [upload_key_max]u8 = undefined;
        const key = normalizeName(name, &keybuf);
        for (self.buckets[0..self.len]) |*b| {
            if (b.phase == phase and std.mem.eql(u8, b.key[0..b.key_len], key)) {
                b.count += 1;
                b.bytes += nbytes;
                return;
            }
        }
        if (self.len < self.buckets.len) {
            const b = &self.buckets[self.len];
            b.phase = phase;
            @memcpy(b.key[0..key.len], key);
            b.key_len = @intCast(key.len);
            b.count = 1;
            b.bytes = nbytes;
            self.len += 1;
        } else {
            self.other_count += 1;
            self.other_bytes += nbytes;
        }
    }
};

/// Collapse digit runs to '#' so per-layer tensors share a row:
/// "blk.12.attn_q.weight" -> "blk.#.attn_q.weight", "cache_k_l3" -> "cache_k_l#".
fn normalizeName(name: []const u8, buf: []u8) []const u8 {
    var n: usize = 0;
    var in_digits = false;
    for (name) |ch| {
        if (n == buf.len) break;
        if (ch >= '0' and ch <= '9') {
            if (!in_digits) {
                buf[n] = '#';
                n += 1;
                in_digits = true;
            }
        } else {
            in_digits = false;
            buf[n] = ch;
            n += 1;
        }
    }
    return buf[0..n];
}

pub fn nsToMs(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / 1_000_000.0;
}

pub fn avgMs(total_ns: u64, count: u64) f64 {
    if (count == 0) return 0;
    return nsToMs(total_ns) / @as(f64, @floatFromInt(count));
}

pub fn perSecond(count: u64, total_ns: u64) f64 {
    if (count == 0 or total_ns == 0) return 0;
    return @as(f64, @floatFromInt(count)) * 1_000_000_000.0 / @as(f64, @floatFromInt(total_ns));
}

pub fn mibPerSecond(bytes: u64, total_ns: u64) f64 {
    if (bytes == 0 or total_ns == 0) return 0;
    return (@as(f64, @floatFromInt(bytes)) / (1024.0 * 1024.0)) * 1_000_000_000.0 / @as(f64, @floatFromInt(total_ns));
}

pub fn percent(part_ns: u64, total_ns: u64) f64 {
    if (part_ns == 0 or total_ns == 0) return 0;
    return @as(f64, @floatFromInt(part_ns)) * 100.0 / @as(f64, @floatFromInt(total_ns));
}

/// Billions per second (decimal), for MAC throughput. Returns 0 on no data.
pub fn giga(count: u64, total_ns: u64) f64 {
    if (count == 0 or total_ns == 0) return 0;
    return @as(f64, @floatFromInt(count)) / @as(f64, @floatFromInt(total_ns));
}

/// Kernel-busy weight bandwidth from PL counters: `W_BEATS·beat_bytes·fclk/cycles`
/// when the bitstream clock is known, else a PL wall-time estimate. Shared by the
/// matmul detail and the scoreboard so they never disagree.
pub fn weightGbps(stat: profiling.MatmulStat, fclk_hz: u32) f64 {
    const weight_bytes = stat.w_beats * layout.weight_beat_bytes;
    const wall = giga(weight_bytes, stat.wrapper_ns);
    if (fclk_hz == 0 or stat.cycles == 0) return wall;
    const exact = @as(f64, @floatFromInt(weight_bytes)) / @as(f64, @floatFromInt(stat.cycles)) *
        @as(f64, @floatFromInt(fclk_hz)) / 1_000_000_000.0;
    // Kernel-busy bandwidth must be >= wall-time bandwidth (busy time <= wall).
    // If the reported clock makes it lower, CLK_HZ is implausibly low (e.g. a
    // bitstream built before the clock self-description was wired) — fall back to
    // the wall-time estimate rather than report an impossible number.
    return if (exact >= wall) exact else wall;
}

/// True when the reported clock is physically impossible against the counters:
/// kernel-busy time (cycles/fclk) exceeds the matmul wall time. Signals a stale
/// or mis-set CLK_HZ (e.g. a bitstream predating the clock self-description).
pub fn clockImplausible(stat: profiling.MatmulStat, fclk_hz: u32) bool {
    if (fclk_hz == 0 or stat.cycles == 0 or stat.wrapper_ns == 0) return false;
    const busy_ns = @as(f64, @floatFromInt(stat.cycles)) / @as(f64, @floatFromInt(fclk_hz)) * 1_000_000_000.0;
    return busy_ns > @as(f64, @floatFromInt(stat.wrapper_ns));
}

test "phase accounting exposes invalid child budgets" {
    var phase: PhaseAccum = .{ .wall_ns = 100 };
    phase.upload.record(8, 90, 80);
    try std.testing.expectEqual(AccountingStatus.ok, phase.accountingStatus());
    try std.testing.expect(phase.accountingValid());
    try std.testing.expectEqual(@as(u64, 10), phase.residualNs());
    try std.testing.expectEqual(@as(u64, 10), phase.transportNs());

    phase.download.record(8, 5, 6);
    try std.testing.expectEqual(AccountingStatus.invalid, phase.accountingStatus());
    try std.testing.expect(!phase.accountingValid());
}

test "run graph accounting warns on residuals over one percent" {
    var totals: RunGraphTotals = .{
        .run_graph_count = 1,
        .command_count = 10,
        .host_run_graph_ns = 12_000_000,
        .device_service_ns = 10_000_000,
        .device_total_ns = 10_000_000,
        .profiled_service_ns = 9_950_000,
        .request_decode_ns = 100_000,
        .preload_ns = 100_000,
        .command_decode_ns = 100_000,
        .device_execute_ns = 5_000_000,
        .profile_encode_ns = 4_600_000,
    };
    totals.op_totals[1] = .{ .tag = 1, .count = 10, .total_ns = 4_950_000 };

    try std.testing.expectEqual(@as(u64, 50_000), totals.serviceResidualNs());
    try std.testing.expectEqual(@as(u64, 50_000), totals.stageResidualNs());
    try std.testing.expectEqual(@as(u64, 50_000), totals.executionResidualNs());
    try std.testing.expectEqual(AccountingStatus.ok, totals.accountingStatus());

    totals.device_service_ns = 12_000_000;
    totals.device_total_ns = 12_000_000;
    try std.testing.expectEqual(AccountingStatus.warn, totals.accountingStatus());
    try std.testing.expect(!totals.accountingValid());

    totals.profiled_service_ns = 13_000_000;
    try std.testing.expectEqual(AccountingStatus.invalid, totals.accountingStatus());
}
