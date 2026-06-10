const std = @import("std");
const wire = @import("wire");
const profiling = @import("profiling");
const link_mod = @import("link");

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
    }

    pub fn opTotalNs(self: *const RunGraphTotals) u64 {
        var total: u64 = 0;
        for (self.op_totals) |aggregate| total += aggregate.total_ns;
        return total;
    }
};

/// One line summarizing run_graph RPC cost vs measured device work.
pub fn writeRunGraphLine(writer: *std.Io.Writer, totals: *const RunGraphTotals) std.Io.Writer.Error!void {
    // Host round-trip minus device-measured work. Device timing stops before profile
    // serialization and response framing, so this bucket is RPC + profiler overhead,
    // not pure wire transport.
    const rpc_overhead_ns = totals.host_run_graph_ns -| totals.device_total_ns;
    const device_op_ns = totals.opTotalNs();
    const device_runtime_ns = totals.device_total_ns -| device_op_ns;
    try writer.print("run_graph calls={d} commands={d} host_rpc_ms={d:.3} device_total_ms={d:.3} device_op_ms={d:.3} device_runtime_ms={d:.3} rpc_overhead_ms={d:.3} request_bytes={d} response_bytes={d} spans_dropped={d}\n", .{
        totals.run_graph_count,
        totals.command_count,
        nsToMs(totals.host_run_graph_ns),
        nsToMs(totals.device_total_ns),
        nsToMs(device_op_ns),
        nsToMs(device_runtime_ns),
        nsToMs(rpc_overhead_ns),
        totals.request_bytes,
        totals.response_bytes,
        totals.span_dropped,
    });
}

/// Per-op breakdown table. device_total_ns is the denominator for device_%.
pub fn writeOpTable(
    writer: *std.Io.Writer,
    op_totals: []const profiling.Aggregate,
    device_total_ns: u64,
) std.Io.Writer.Error!void {
    try writer.writeAll("op                 count    total_ms      avg_ms   device_%        bytes       GB/s\n");
    for (op_totals) |aggregate| {
        if (aggregate.count == 0) continue;
        try writer.print("{s:<16} {d:>7} {d:>11.3} {d:>11.3} {d:>10.2} {d:>12} {d:>10.3}\n", .{
            opName(aggregate.tag),
            aggregate.count,
            nsToMs(aggregate.total_ns),
            avgMs(aggregate.total_ns, aggregate.count),
            percent(aggregate.total_ns, device_total_ns),
            aggregate.bytes,
            gbPerSecond(aggregate.bytes, aggregate.total_ns),
        });
    }
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

// bytes/ns == GB/s (decimal): 1 byte/ns = 1e9 bytes/s = 1 GB/s.
pub fn gbPerSecond(bytes: u64, total_ns: u64) f64 {
    if (bytes == 0 or total_ns == 0) return 0;
    return @as(f64, @floatFromInt(bytes)) / @as(f64, @floatFromInt(total_ns));
}

pub fn percent(part_ns: u64, total_ns: u64) f64 {
    if (part_ns == 0 or total_ns == 0) return 0;
    return @as(f64, @floatFromInt(part_ns)) * 100.0 / @as(f64, @floatFromInt(total_ns));
}
