const std = @import("std");
const shared = @import("shared");
const link_mod = @import("link");

const wire = shared.wire;
const profiling = shared.profiling;
const q1a8 = shared.q1a8;

/// Selects how the llama `--prof` summary is rendered. `pretty` is the default
/// human-facing layout; `json` emits every counter as a machine-readable object.
pub const ProfFormat = enum { pretty, json };

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
    span_dropped: u64 = 0,
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
        self.span_dropped += profiled.report.summary.span_dropped;
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

    fn isEmpty(self: *const PhaseAccum) bool {
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

const UploadBucket = struct {
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

fn bucketBytesDesc(_: void, a: UploadBucket, b: UploadBucket) bool {
    return a.bytes > b.bytes;
}

/// Render the upload census, sorted by bytes. Empty when nothing was uploaded.
pub fn writeUploadCensus(writer: *std.Io.Writer, census: *const UploadCensus) std.Io.Writer.Error!void {
    if (census.len == 0) return;
    var rows: [upload_census_cap]UploadBucket = undefined;
    @memcpy(rows[0..census.len], census.buckets[0..census.len]);
    std.mem.sort(UploadBucket, rows[0..census.len], {}, bucketBytesDesc);

    try writer.print("  {s:<32} {s:>12} {s:>7} {s:>9} {s:>9}\n", .{ "uploads by tensor", "phase", "calls", "bytes", "avg" });
    for (rows[0..census.len]) |b| {
        var bytes_buf: [16]u8 = undefined;
        var avg_buf: [16]u8 = undefined;
        const avg = if (b.count == 0) 0 else b.bytes / b.count;
        try writer.print("  {s:<32} {s:>12} {d:>7} {s:>9} {s:>9}\n", .{
            b.key[0..b.key_len],
            (@as(Phase, @enumFromInt(b.phase))).label(),
            b.count,
            formatBytes(&bytes_buf, b.bytes),
            formatBytes(&avg_buf, avg),
        });
    }
    if (census.other_count != 0) {
        var bytes_buf: [16]u8 = undefined;
        try writer.print("  {s:<32} {s:>12} {d:>7} {s:>9} {s:>9}\n", .{
            "(other)", "", census.other_count, formatBytes(&bytes_buf, census.other_bytes), "",
        });
    }
}

/// Phase budget: the headline closed-identity table. Each row reconciles
/// wall = device + transport + residual; the total row sums the columns.
pub fn writePhaseBudget(writer: *std.Io.Writer, phases: []const PhaseAccum) std.Io.Writer.Error!void {
    try writer.print("phase            {s:>9} {s:>9} {s:>9} {s:>9}\n", .{ "wall", "device", "transport", "residual" });
    var tot_wall: u64 = 0;
    var tot_dev: u64 = 0;
    var tot_tr: u64 = 0;
    var tot_res: u64 = 0;
    for (phases, 0..) |phase, i| {
        if (phase.isEmpty()) continue;
        const label = @as(Phase, @enumFromInt(i)).label();
        try phaseBudgetRow(writer, label, phase.wall_ns, phase.deviceNs(), phase.transportNs(), phase.residualNs());
        tot_wall += phase.wall_ns;
        tot_dev += phase.deviceNs();
        tot_tr += phase.transportNs();
        tot_res += phase.residualNs();
    }
    try phaseBudgetRow(writer, "total", tot_wall, tot_dev, tot_tr, tot_res);
}

fn phaseBudgetRow(writer: *std.Io.Writer, label: []const u8, wall: u64, device: u64, transport: u64, residual: u64) std.Io.Writer.Error!void {
    var w: [16]u8 = undefined;
    var d: [16]u8 = undefined;
    var t: [16]u8 = undefined;
    var r: [16]u8 = undefined;
    try writer.print("  {s:<13} {s:>9} {s:>9} {s:>9} {s:>9}\n", .{
        label,
        formatDuration(&w, wall),
        formatDuration(&d, device),
        formatDuration(&t, transport),
        formatDuration(&r, residual),
    });
}

/// Per-phase transfer detail. Rows are (phase, op-kind) with nonzero traffic:
/// calls, bytes, host vs device time, and effective host MiB/s. This is where
/// the weight-upload anomaly and per-token decode copies become visible.
pub fn writeTransfers(writer: *std.Io.Writer, phases: []const PhaseAccum) std.Io.Writer.Error!void {
    var any = false;
    for (phases) |phase| {
        inline for (.{ LinkOp.Kind.upload, .download, .fill, .alloc, .free }) |kind| {
            if (@constCast(&phase).op(kind).count != 0) any = true;
        }
    }
    if (!any) return;

    try writer.print("transfers      {s:>12} {s:>7} {s:>9} {s:>8} {s:>8} {s:>10}\n", .{ "phase", "calls", "bytes", "host", "device", "MiB/s" });
    for (phases, 0..) |phase, i| {
        const label = @as(Phase, @enumFromInt(i)).label();
        inline for (.{ LinkOp.Kind.upload, .download, .fill, .alloc, .free }) |kind| {
            const link_op = @constCast(&phase).op(kind);
            if (link_op.count != 0) try transferRow(writer, LinkOp.name(kind), label, link_op.*);
        }
    }
}

fn transferRow(writer: *std.Io.Writer, kind: []const u8, phase_label: []const u8, link_op: LinkOp) std.Io.Writer.Error!void {
    var bytes_buf: [16]u8 = undefined;
    var host_buf: [16]u8 = undefined;
    var dev_buf: [16]u8 = undefined;
    try writer.print("  {s:<10} {s:>12} {d:>7} {s:>9} {s:>8} {s:>8} {d:>10.1}\n", .{
        kind,
        phase_label,
        link_op.count,
        formatBytes(&bytes_buf, link_op.bytes),
        formatDuration(&host_buf, link_op.host_ns),
        formatDuration(&dev_buf, link_op.device_ns),
        mibPerSecond(link_op.bytes, link_op.host_ns),
    });
}

/// The `link` section: run_graph RPC cost vs measured device work. Shared by the
/// llama `--prof` report and the bench harness so there is one rendering of it.
pub fn writeLinkSection(writer: *std.Io.Writer, totals: *const RunGraphTotals) std.Io.Writer.Error!void {
    var ovh_buf: [32]u8 = undefined;
    var req_buf: [32]u8 = undefined;
    var resp_buf: [32]u8 = undefined;
    try writer.print("link\n", .{});
    try writer.print("  graphs           {d}\n", .{totals.run_graph_count});
    try writer.print("  commands         {d}\n", .{totals.command_count});
    try writer.print("  rpc overhead     {s} ({d:.1}%)\n", .{
        formatDuration(&ovh_buf, totals.rpcOverheadNs()),
        percent(totals.rpcOverheadNs(), totals.host_run_graph_ns),
    });
    try writer.print("  request          {s}\n", .{formatBytes(&req_buf, totals.request_bytes)});
    try writer.print("  response         {s}\n", .{formatBytes(&resp_buf, totals.response_bytes)});
    try writer.print("  dropped          {d}\n", .{totals.span_dropped});
}

/// Canonical op name from the wire enum — single source of truth.
pub fn opName(tag: u16) []const u8 {
    const op = std.enums.fromInt(wire.OpTag, tag) orelse return "unknown";
    return @tagName(op);
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

// The format helpers below emit at most ~12 chars, so the 32-byte scratch their
// callers pass always fits — bufPrint cannot overflow, hence `catch unreachable`.

/// Format a duration with an auto-selected unit (s >= 1s, else ms) into `buf`,
/// returning the slice. For right-aligned columns, print the result with `{s:>N}`.
pub fn formatDuration(buf: []u8, ns: u64) []const u8 {
    const ms = nsToMs(ns);
    if (ms >= 1000.0) return std.fmt.bufPrint(buf, "{d:.1} s", .{ms / 1000.0}) catch unreachable;
    if (ms >= 1.0) return std.fmt.bufPrint(buf, "{d:.1} ms", .{ms}) catch unreachable;
    if (ms > 0.0) return std.fmt.bufPrint(buf, "{d:.2} ms", .{ms}) catch unreachable;
    return std.fmt.bufPrint(buf, "0 ms", .{}) catch unreachable;
}

/// Format a byte count with an auto-selected binary unit (GiB/MiB/KiB/B) into `buf`.
pub fn formatBytes(buf: []u8, n: u64) []const u8 {
    const f: f64 = @floatFromInt(n);
    if (n >= 1 << 30) return std.fmt.bufPrint(buf, "{d:.1} GiB", .{f / (1 << 30)}) catch unreachable;
    if (n >= 1 << 20) return std.fmt.bufPrint(buf, "{d:.1} MiB", .{f / (1 << 20)}) catch unreachable;
    if (n >= 1 << 10) return std.fmt.bufPrint(buf, "{d:.1} KiB", .{f / (1 << 10)}) catch unreachable;
    return std.fmt.bufPrint(buf, "{d} B", .{n}) catch unreachable;
}

fn opTotalDesc(_: void, x: profiling.Aggregate, y: profiling.Aggregate) bool {
    return x.total_ns > y.total_ns;
}

/// Canonical weight-format name from the wire enum (stat.fmt is the raw value).
pub fn weightFormatName(fmt: u16) []const u8 {
    const value = std.enums.fromInt(wire.WeightFormat, fmt) orelse return "unknown";
    return @tagName(value);
}

/// Billions per second (decimal), for MAC throughput. Returns 0 on no data.
pub fn giga(count: u64, total_ns: u64) f64 {
    if (count == 0 or total_ns == 0) return 0;
    return @as(f64, @floatFromInt(count)) / @as(f64, @floatFromInt(total_ns));
}

/// Per-(weight format) matmul breakdown. The top line is all matmuls of that
/// format (CPU + PL): calls, GMAC, wall MAC/s. When some ran on the PL, a `pl`
/// sub-line reports the PL-executed subset only: its calls, end-to-end MAC/s
/// (`pl_macs/pl_ns`, includes per-call quantize/DMA/sync overhead), kernel
/// MAC/cycle (`pl_macs/cycles`, the array's intrinsic rate — clock-independent),
/// and array utilization `(cycles - max stall)/cycles`. The gap between the
/// PL MAC/s and MAC/cycle×fclk is the per-call software overhead. The weight
/// stream columns are derived from PL counters: W B/cyc is clock-independent;
/// W GB/s is `W_BEATS·beat_bytes·fclk/cycles` (kernel-busy bandwidth, the figure
/// to compare against the DDR ceiling) when the bitstream reports its clock
/// (`device_fclk_hz`), else it falls back to a PL end-to-end wall-time estimate.
pub fn writeMatmulDetail(
    writer: *std.Io.Writer,
    title: []const u8,
    matmul_stats: []const profiling.MatmulStat,
    fclk_hz: u32,
) std.Io.Writer.Error!void {
    var any = false;
    for (matmul_stats) |stat| {
        if (stat.count != 0) any = true;
    }
    if (!any) return;

    try writer.print("{s}\n", .{title});
    try writer.print("  {s:<10} {s:>7} {s:>10} {s:>12} {s:>9} {s:>7} {s:>8} {s:>8}\n", .{
        "fmt", "calls", "GMAC", "MAC/s", "MAC/cyc", "util%", "W B/cyc", "W GB/s",
    });
    for (matmul_stats) |stat| {
        if (stat.count == 0) continue;
        const gmac = @as(f64, @floatFromInt(stat.macs)) / 1_000_000_000.0;
        var macps_buf: [24]u8 = undefined;
        try writer.print("  {s:<10} {d:>7} {d:>10.1} {s:>12} {s:>9} {s:>7} {s:>8} {s:>8}\n", .{
            weightFormatName(stat.fmt),
            stat.count,
            gmac,
            std.fmt.bufPrint(&macps_buf, "{d:.2} G/s", .{giga(stat.macs, stat.total_ns)}) catch unreachable,
            "-",
            "-",
            "-",
            "-",
        });
        if (stat.pl_count == 0) continue;
        var plps_buf: [24]u8 = undefined;
        const mac_per_cyc = if (stat.cycles == 0) 0 else @as(f64, @floatFromInt(stat.pl_macs)) / @as(f64, @floatFromInt(stat.cycles));
        const max_stall = @max(stat.w_stall_cycles, @max(stat.a_stall_cycles, stat.r_stall_cycles));
        const util = if (stat.cycles == 0) 0 else percent(stat.cycles -| max_stall, stat.cycles);
        const weight_bytes = stat.w_beats * q1a8.weight_beat_bytes;
        const weight_bytes_per_cycle = if (stat.cycles == 0) 0 else @as(f64, @floatFromInt(weight_bytes)) / @as(f64, @floatFromInt(stat.cycles));
        const weight_gb_s_wall = weightGbps(stat, fclk_hz);
        try writer.print("    {s:<8} {d:>7} {s:>10} {s:>12} {d:>9.1} {d:>7.1} {d:>8.2} {d:>8.2}\n", .{
            "pl",
            stat.pl_count,
            "",
            std.fmt.bufPrint(&plps_buf, "{d:.2} G/s", .{giga(stat.pl_macs, stat.pl_ns)}) catch unreachable,
            mac_per_cyc,
            util,
            weight_bytes_per_cycle,
            weight_gb_s_wall,
        });
        if (stat.cycles != 0 or stat.w_beats != 0 or stat.a_beats != 0 or stat.r_beats != 0) {
            try writer.print("      cycles={d} stalls W/A/R={d}/{d}/{d} beats W/A/R={d}/{d}/{d}\n", .{
                stat.cycles,
                stat.w_stall_cycles,
                stat.a_stall_cycles,
                stat.r_stall_cycles,
                stat.w_beats,
                stat.a_beats,
                stat.r_beats,
            });
        }
        if (clockImplausible(stat, fclk_hz)) {
            var busy_buf: [16]u8 = undefined;
            var wall_buf: [16]u8 = undefined;
            const busy_ns: u64 = @intFromFloat(@as(f64, @floatFromInt(stat.cycles)) /
                @as(f64, @floatFromInt(fclk_hz)) * 1_000_000_000.0);
            try writer.print("      note: CLK_HZ={d} implausible (kernel busy {s} > wall {s}) — W GB/s is wall-time; rebuild bitstream\n", .{
                fclk_hz,
                formatDuration(&busy_buf, busy_ns),
                formatDuration(&wall_buf, stat.pl_ns),
            });
        }
    }
}

/// Kernel-busy weight bandwidth from PL counters: `W_BEATS·beat_bytes·fclk/cycles`
/// when the bitstream clock is known, else a PL wall-time estimate. Shared by the
/// matmul detail and the scoreboard so they never disagree.
pub fn weightGbps(stat: profiling.MatmulStat, fclk_hz: u32) f64 {
    const weight_bytes = stat.w_beats * q1a8.weight_beat_bytes;
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

/// Sub-millisecond latency with an auto unit (us >= 1us, else ns). For the
/// flash per-inner-iteration column, which lives in the ns–us range.
pub fn formatLatencyFine(buf: []u8, ns: f64) []const u8 {
    if (ns >= 1000.0) return std.fmt.bufPrint(buf, "{d:.2} us", .{ns / 1000.0}) catch unreachable;
    return std.fmt.bufPrint(buf, "{d:.0} ns", .{ns}) catch unreachable;
}

/// Flash-attention shape detail for a phase (one row — there is one flash op
/// kind). The diagnostic is `ns/inner` = total_ns / (n_heads · Σn_kv), the cost
/// of one (head, kv) inner iteration. Microseconds there at small avg n_kv means
/// the op is overhead/padding-bound (two passes, walking masked entries), not
/// bound on the K/V stream — which decides PS rewrite vs PL move. Read it as a
/// decode metric: decode is always n_tokens=1, so the per-token denominator is
/// exact; prefill packs n_tokens query rows per call, so its `ns/inner`
/// aggregates over them (n_tokens is not in the denominator) and runs lower.
pub fn writeFlashDetail(
    writer: *std.Io.Writer,
    title: []const u8,
    flash: profiling.FlashStat,
) std.Io.Writer.Error!void {
    if (flash.count == 0) return;
    try writer.print("{s}\n", .{title});
    try writer.print("  {s:<8} {s:>7} {s:>11} {s:>7} {s:>9} {s:>10} {s:>10} {s:>10}\n", .{
        "qkv", "calls", "n_kv a/max", "heads", "hdim q/v", "ns/call", "ns/inner", "eff MiB/s",
    });
    const avg_n_kv = @as(f64, @floatFromInt(flash.sum_n_kv)) / @as(f64, @floatFromInt(flash.count));
    const ns_per_call = flash.total_ns / flash.count;
    const inner = @as(f64, @floatFromInt(flash.n_heads)) * @as(f64, @floatFromInt(flash.sum_n_kv));
    const ns_per_inner = if (inner == 0) 0 else @as(f64, @floatFromInt(flash.total_ns)) / inner;
    var nkv_buf: [20]u8 = undefined;
    var heads_buf: [12]u8 = undefined;
    var hdim_buf: [12]u8 = undefined;
    var call_buf: [16]u8 = undefined;
    var inner_buf: [16]u8 = undefined;
    try writer.print("  {s:<8} {d:>7} {s:>11} {s:>7} {s:>9} {s:>10} {s:>10} {d:>10.1}\n", .{
        "f32/f16",
        flash.count,
        std.fmt.bufPrint(&nkv_buf, "{d:.0}/{d}", .{ avg_n_kv, flash.max_n_kv }) catch "?",
        std.fmt.bufPrint(&heads_buf, "{d}/{d}", .{ flash.n_heads, flash.n_head_kv }) catch "?",
        std.fmt.bufPrint(&hdim_buf, "{d}/{d}", .{ flash.head_dim_q, flash.head_dim_v }) catch "?",
        formatDuration(&call_buf, ns_per_call),
        formatLatencyFine(&inner_buf, ns_per_inner),
        mibPerSecond(flash.bytes, flash.total_ns),
    });
}

/// One compact, greppable line per run for cross-variant comparison across
/// clock/port/kernel sweeps. All values are decode-phase and derived from data
/// already in the report — pure rendering, no new measurement. The variant label
/// auto-derives from the resident layout (host constants) and the device clock,
/// so it can't drift from the build the way a hand-typed string can.
pub fn writeScoreboard(
    writer: *std.Io.Writer,
    decode: *const PhaseAccum,
    tokens: u64,
) std.Io.Writer.Error!void {
    if (tokens == 0 or decode.wall_ns == 0) return;
    const rg = &decode.rg;
    const t: f64 = @floatFromInt(tokens);
    const mhz: u64 = (@as(u64, rg.device_fclk_hz) + 500_000) / 1_000_000;
    var vbuf: [32]u8 = undefined;
    const variant = std.fmt.bufPrint(&vbuf, "w{d}-p{d}-f{d}", .{
        q1a8.rows_per_block * 32, q1a8.weight_ports, mhz,
    }) catch "unknown";

    var mm: profiling.MatmulStat = .{};
    for (rg.matmul_stats) |s| {
        if (s.count != 0) {
            mm = s;
            break;
        }
    }
    const mac_cyc = if (mm.cycles == 0) 0 else @as(f64, @floatFromInt(mm.pl_macs)) / @as(f64, @floatFromInt(mm.cycles));
    const flash_ns = rg.op_totals[@intFromEnum(wire.OpTag.flash_attn_f32)].total_ns;
    const swiglu_ns = rg.op_totals[@intFromEnum(wire.OpTag.swiglu)].total_ns;

    try writer.print(
        "scoreboard variant={s} tok_s={d:.2} decode_ms={d:.1} device_ms={d:.1} transport_ms={d:.1}" ++
            " matmul_gmac_s={d:.2} matmul_mac_cyc={d:.1} matmul_w_gbps={d:.2} matmul_w_stall_pct={d:.1}" ++
            " flash_ms_tok={d:.1} swiglu_ms_tok={d:.1}\n",
        .{
            variant,
            perSecond(tokens, decode.wall_ns),
            nsToMs(decode.wall_ns) / t,
            nsToMs(decode.deviceNs()) / t,
            nsToMs(decode.transportNs()) / t,
            giga(mm.macs, mm.total_ns),
            mac_cyc,
            weightGbps(mm, rg.device_fclk_hz),
            percent(mm.w_stall_cycles, mm.cycles),
            nsToMs(flash_ns) / t,
            nsToMs(swiglu_ns) / t,
        },
    );
}

/// Per-op breakdown, sorted by device time descending, with human units. The
/// throughput is operand-bytes/op-time (effective traffic, not measured bus
/// bandwidth) — see commandBytes in device/profile.zig — hence the `eff` label.
pub fn writeOpTable(
    writer: *std.Io.Writer,
    title: []const u8,
    op_totals: []const profiling.Aggregate,
    device_total_ns: u64,
) std.Io.Writer.Error!void {
    var rows: [profiling.max_op_tag + 1]profiling.Aggregate = undefined;
    var n: usize = 0;
    for (op_totals) |aggregate| {
        if (aggregate.count == 0) continue;
        rows[n] = aggregate;
        n += 1;
    }
    std.mem.sort(profiling.Aggregate, rows[0..n], {}, opTotalDesc);

    var total_buf: [32]u8 = undefined;
    try writer.print("{s}  ({s} device)\n", .{ title, formatDuration(&total_buf, device_total_ns) });
    try writer.print("  {s:<16} {s:>7} {s:>10} {s:>9} {s:>6} {s:>10}\n", .{ "op", "count", "total", "avg", "dev%", "eff MiB/s" });
    for (rows[0..n]) |aggregate| {
        const avg_ns = if (aggregate.count == 0) 0 else aggregate.total_ns / aggregate.count;
        var tot_buf: [32]u8 = undefined;
        var avg_buf: [32]u8 = undefined;
        try writer.print("  {s:<16} {d:>7} {s:>10} {s:>9} {d:>6.1} {d:>10.1}\n", .{
            opName(aggregate.tag),
            aggregate.count,
            formatDuration(&tot_buf, aggregate.total_ns),
            formatDuration(&avg_buf, avg_ns),
            percent(aggregate.total_ns, device_total_ns),
            mibPerSecond(aggregate.bytes, aggregate.total_ns),
        });
    }
}
