const std = @import("std");
const shared = @import("shared");
const link_mod = @import("link");

const profiling = shared.profiling;
const layout = shared.layout;

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
    device_total_ns: u64 = 0,
    device_decode_ns: u64 = 0,
    device_execute_ns: u64 = 0,
    op_totals: [profiling.max_op_tag + 1]profiling.Aggregate = [_]profiling.Aggregate{.{}} ** (profiling.max_op_tag + 1),
    matmul_stats: [profiling.max_weight_fmt]profiling.MatmulStat = [_]profiling.MatmulStat{.{}} ** profiling.max_weight_fmt,
    flash: profiling.FlashStat = .{},
    /// Constant across a run (the PL bitstream's fabric clock); 0 on a PS-only or
    /// fake run. Last-wins, ignoring zeros so a stray PS run_graph can't clear it.
    device_fclk_hz: u32 = 0,

    pub fn record(self: *RunGraphTotals, profiled: link_mod.ProfiledRunGraph) void {
        self.run_graph_count += 1;
        self.command_count += profiled.report.summary.command_count;
        self.host_run_graph_ns += profiled.rpc.round_trip_ns;
        self.request_bytes += profiled.rpc.request_bytes;
        self.response_bytes += profiled.rpc.response_bytes;
        self.device_total_ns += profiled.report.summary.device_total_ns;
        self.device_decode_ns += profiled.report.summary.decode_ns;
        self.device_execute_ns += profiled.report.summary.execute_ns;
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
            const index: usize = stat.fmt;
            if (index >= self.matmul_stats.len) continue;
            var total = &self.matmul_stats[index];
            total.fmt = stat.fmt;
            total.count += stat.count;
            total.macs += stat.macs;
            total.total_ns += stat.total_ns;
            total.pl_count += stat.pl_count;
            total.pl_macs += stat.pl_macs;
            total.pl_ns += stat.pl_ns;
            total.cycles += stat.cycles;
            total.w_stall_cycles += stat.w_stall_cycles;
            total.a_stall_cycles += stat.a_stall_cycles;
            total.r_stall_cycles += stat.r_stall_cycles;
            total.w_beats += stat.w_beats;
            total.a_beats += stat.a_beats;
            total.r_beats += stat.r_beats;
        }
        for (profiled.report.flash_stats) |stat| {
            self.flash.count += stat.count;
            self.flash.sum_n_kv += stat.sum_n_kv;
            self.flash.max_n_kv = @max(self.flash.max_n_kv, stat.max_n_kv);
            self.flash.total_ns += stat.total_ns;
            self.flash.bytes += stat.bytes;
            if (stat.n_heads != 0) {
                self.flash.n_heads = stat.n_heads;
                self.flash.n_head_kv = stat.n_head_kv;
                self.flash.head_dim_q = stat.head_dim_q;
                self.flash.head_dim_v = stat.head_dim_v;
            }
        }
        if (profiled.report.summary.device_fclk_hz != 0) self.device_fclk_hz = profiled.report.summary.device_fclk_hz;
    }

    pub fn opTotalNs(self: *const RunGraphTotals) u64 {
        var total: u64 = 0;
        for (self.op_totals) |aggregate| total += aggregate.total_ns;
        return total;
    }

    /// Host round-trip minus device-measured work. Device timing stops before
    /// profile serialization and response framing, so this bucket is RPC +
    /// profiler overhead, not pure wire transport.
    pub fn rpcOverheadNs(self: *const RunGraphTotals) u64 {
        return self.host_run_graph_ns -| self.device_total_ns;
    }

    /// Device time the profiler attributed to no op span (dispatch/scheduling).
    pub fn deviceRuntimeNs(self: *const RunGraphTotals) u64 {
        return self.device_total_ns -| self.opTotalNs();
    }
};

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

    pub fn record(self: *LinkOp, bytes: u64, host_ns: u64, device_ns: u64) void {
        self.count += 1;
        self.bytes += bytes;
        self.host_ns += host_ns;
        self.device_ns += device_ns;
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
        return self.linkHostNs() -| self.deviceNs();
    }

    pub fn residualNs(self: *const PhaseAccum) u64 {
        return self.wall_ns -| self.linkHostNs();
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
    const wall = giga(weight_bytes, stat.pl_ns);
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
    if (fclk_hz == 0 or stat.cycles == 0 or stat.pl_ns == 0) return false;
    const busy_ns = @as(f64, @floatFromInt(stat.cycles)) / @as(f64, @floatFromInt(fclk_hz)) * 1_000_000_000.0;
    return busy_ns > @as(f64, @floatFromInt(stat.pl_ns));
}
