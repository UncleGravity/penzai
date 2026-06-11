const std = @import("std");
const shared = @import("shared");
const link_mod = @import("link");

const wire = shared.wire;
const profiling = shared.profiling;

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
/// PL MAC/s and MAC/cycle×fclk is the per-call software overhead.
pub fn writeMatmulDetail(
    writer: *std.Io.Writer,
    matmul_stats: []const profiling.MatmulStat,
) std.Io.Writer.Error!void {
    var any = false;
    for (matmul_stats) |stat| {
        if (stat.count != 0) any = true;
    }
    if (!any) return;

    try writer.print("matmul detail\n", .{});
    try writer.print("  {s:<10} {s:>7} {s:>10} {s:>12} {s:>9} {s:>7}\n", .{ "fmt", "calls", "GMAC", "MAC/s", "MAC/cyc", "util%" });
    for (matmul_stats) |stat| {
        if (stat.count == 0) continue;
        const gmac = @as(f64, @floatFromInt(stat.macs)) / 1_000_000_000.0;
        var macps_buf: [24]u8 = undefined;
        try writer.print("  {s:<10} {d:>7} {d:>10.1} {s:>12} {s:>9} {s:>7}\n", .{
            weightFormatName(stat.fmt),
            stat.count,
            gmac,
            std.fmt.bufPrint(&macps_buf, "{d:.2} G/s", .{giga(stat.macs, stat.total_ns)}) catch unreachable,
            "-",
            "-",
        });
        if (stat.pl_count == 0) continue;
        var plps_buf: [24]u8 = undefined;
        const mac_per_cyc = if (stat.cycles == 0) 0 else @as(f64, @floatFromInt(stat.pl_macs)) / @as(f64, @floatFromInt(stat.cycles));
        const max_stall = @max(stat.w_stall_cycles, @max(stat.a_stall_cycles, stat.r_stall_cycles));
        const util = if (stat.cycles == 0) 0 else percent(stat.cycles -| max_stall, stat.cycles);
        try writer.print("    {s:<8} {d:>7} {s:>10} {s:>12} {d:>9.1} {d:>7.1}\n", .{
            "pl",
            stat.pl_count,
            "",
            std.fmt.bufPrint(&plps_buf, "{d:.2} G/s", .{giga(stat.pl_macs, stat.pl_ns)}) catch unreachable,
            mac_per_cyc,
            util,
        });
    }
}

/// Per-op breakdown, sorted by device time descending, with human units. The
/// throughput is operand-bytes/op-time (effective traffic, not measured bus
/// bandwidth) — see commandBytes in device/profile.zig — hence the `eff` label.
pub fn writeOpTable(
    writer: *std.Io.Writer,
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
    try writer.print("device ops{s:>33}\n", .{formatDuration(&total_buf, device_total_ns)});
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
