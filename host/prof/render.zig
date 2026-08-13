const std = @import("std");
const shared = @import("shared");
const model = @import("model.zig");
const collector_mod = @import("collector.zig");

const wire = shared.wire;
const profiling = shared.profiling;
const layout = shared.layout;

// Aliases so the renderers below reference the model types/helpers unqualified.
const RunGraphTotals = model.RunGraphTotals;
const AccountingStatus = model.AccountingStatus;
const Phase = model.Phase;
const phase_count = model.phase_count;
const LinkOp = model.LinkOp;
const PhaseAccum = model.PhaseAccum;
const UploadCensus = model.UploadCensus;
const UploadBucket = model.UploadBucket;
const upload_census_cap = model.upload_census_cap;
const nsToMs = model.nsToMs;
const perSecond = model.perSecond;
const mibPerSecond = model.mibPerSecond;
const percent = model.percent;
const giga = model.giga;
const weightGbps = model.weightGbps;
const clockImplausible = model.clockImplausible;

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
    try writer.print("phase            {s:>9} {s:>9} {s:>9} {s:>9} {s:>8}\n", .{ "wall", "device", "transport", "residual", "account" });
    var tot_wall: u64 = 0;
    var tot_dev: u64 = 0;
    var tot_tr: u64 = 0;
    var tot_res: u64 = 0;
    for (phases, 0..) |phase, i| {
        if (phase.isEmpty()) continue;
        const label = @as(Phase, @enumFromInt(i)).label();
        try phaseBudgetRow(writer, label, phase.wall_ns, phase.deviceNs(), phase.transportNs(), phase.residualNs(), phase.accountingStatus());
        tot_wall += phase.wall_ns;
        tot_dev += phase.deviceNs();
        tot_tr += phase.transportNs();
        tot_res += phase.residualNs();
    }
    var status: AccountingStatus = .ok;
    for (phases) |phase| status = status.combine(phase.accountingStatus());
    try phaseBudgetRow(writer, "total", tot_wall, tot_dev, tot_tr, tot_res, status);
}

fn phaseBudgetRow(writer: *std.Io.Writer, label: []const u8, wall: u64, device: u64, transport: u64, residual: u64, status: AccountingStatus) std.Io.Writer.Error!void {
    var w: [16]u8 = undefined;
    var d: [16]u8 = undefined;
    var t: [16]u8 = undefined;
    var r: [16]u8 = undefined;
    try writer.print("  {s:<13} {s:>9} {s:>9} {s:>9} {s:>9} {s:>8}\n", .{
        label,
        formatDuration(&w, wall),
        formatDuration(&d, device),
        formatDuration(&t, transport),
        formatDuration(&r, residual),
        status.label(),
    });
}

/// Nested device budgets are diagnostics rather than additive phase columns.
/// Each residual is shown against its own authoritative parent.
pub fn writeDeviceAccounting(writer: *std.Io.Writer, phases: []const PhaseAccum) std.Io.Writer.Error!void {
    var any = false;
    for (phases) |phase| if (phase.rg.run_graph_count != 0) {
        any = true;
    };
    if (!any) return;

    try writer.print("device accounting {s:>10} {s:>7} {s:>10} {s:>7} {s:>10} {s:>7} {s:>8}\n", .{
        "service", "svc%", "stages", "stage%", "execute", "exec%", "account",
    });
    for (phases, 0..) |phase, i| {
        if (phase.rg.run_graph_count == 0) continue;
        const rg = &phase.rg;
        var service_buf: [16]u8 = undefined;
        var stage_buf: [16]u8 = undefined;
        var execution_buf: [16]u8 = undefined;
        try writer.print("  {s:<13} {s:>10} {d:>6.2}% {s:>10} {d:>6.2}% {s:>10} {d:>6.2}% {s:>8}\n", .{
            (@as(Phase, @enumFromInt(i))).label(),
            formatDuration(&service_buf, rg.serviceResidualNs()),
            percent(rg.serviceResidualNs(), rg.device_service_ns),
            formatDuration(&stage_buf, rg.stageResidualNs()),
            percent(rg.stageResidualNs(), rg.profiled_service_ns),
            formatDuration(&execution_buf, rg.executionResidualNs()),
            percent(rg.executionResidualNs(), rg.device_execute_ns),
            rg.accountingStatus().label(),
        });
    }
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
    var transport_buf: [32]u8 = undefined;
    var service_buf: [32]u8 = undefined;
    var response_encode_buf: [32]u8 = undefined;
    var request_decode_buf: [32]u8 = undefined;
    var preload_buf: [32]u8 = undefined;
    var command_decode_buf: [32]u8 = undefined;
    var execute_buf: [32]u8 = undefined;
    var profile_encode_buf: [32]u8 = undefined;
    var residual_buf: [32]u8 = undefined;
    var stage_residual_buf: [32]u8 = undefined;
    var execute_residual_buf: [32]u8 = undefined;
    var req_buf: [32]u8 = undefined;
    var resp_buf: [32]u8 = undefined;
    try writer.print("link\n", .{});
    try writer.print("  graphs           {d}\n", .{totals.run_graph_count});
    try writer.print("  commands         {d}\n", .{totals.command_count});
    try writer.print("  device service   {s}\n", .{formatDuration(&service_buf, totals.device_service_ns)});
    try writer.print("  response encode  {s}\n", .{formatDuration(&response_encode_buf, totals.device_encode_ns)});
    try writer.print("  transport        {s} ({d:.1}%)\n", .{
        formatDuration(&transport_buf, totals.transportNs()),
        percent(totals.transportNs(), totals.host_run_graph_ns),
    });
    try writer.print("  service stages   request={s} preload={s} command-decode={s} execute={s} profile-encode={s}\n", .{
        formatDuration(&request_decode_buf, totals.request_decode_ns),
        formatDuration(&preload_buf, totals.preload_ns),
        formatDuration(&command_decode_buf, totals.command_decode_ns),
        formatDuration(&execute_buf, totals.device_execute_ns),
        formatDuration(&profile_encode_buf, totals.profile_encode_ns),
    });
    try writer.print("  service residual {s} ({d:.2}%)\n", .{ formatDuration(&residual_buf, totals.serviceResidualNs()), percent(totals.serviceResidualNs(), totals.device_service_ns) });
    try writer.print("  stage residual   {s} ({d:.2}%)\n", .{ formatDuration(&stage_residual_buf, totals.stageResidualNs()), percent(totals.stageResidualNs(), totals.profiled_service_ns) });
    try writer.print("  execute residual {s} ({d:.2}%)\n", .{ formatDuration(&execute_residual_buf, totals.executionResidualNs()), percent(totals.executionResidualNs(), totals.device_execute_ns) });
    try writer.print("  request          {s}\n", .{formatBytes(&req_buf, totals.request_bytes)});
    try writer.print("  response         {s}\n", .{formatBytes(&resp_buf, totals.response_bytes)});
    if (totals.matmul_bucket_overflow != 0 or totals.flash_bucket_overflow != 0)
        try writer.print("  bucket overflow  matmul={d} flash={d}\n", .{ totals.matmul_bucket_overflow, totals.flash_bucket_overflow });
    if (totals.accountingStatus() != .ok)
        try writer.print("  accounting       {s} (flags=0x{x})\n", .{ totals.accountingStatus().label(), totals.accounting_violations });
}

/// Canonical op name from the wire enum — single source of truth.
pub fn opName(tag: u16) []const u8 {
    const op = std.enums.fromInt(wire.OpTag, tag) orelse return "unknown";
    return @tagName(op);
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

fn backendName(backend: profiling.Backend) []const u8 {
    return @tagName(backend);
}

fn pathName(path: profiling.ExecutionPath) []const u8 {
    return @tagName(path);
}

/// Shape/path buckets keep PS fallbacks and direct/staged PL executions separate.
/// Wrapper stages are additive; kernel cycles are nested inside wait.
pub fn writeMatmulDetail(
    writer: *std.Io.Writer,
    title: []const u8,
    matmul_stats: []const profiling.MatmulStat,
    fclk_hz: u32,
) std.Io.Writer.Error!void {
    var any = false;
    var saw_axis_counters = false;
    for (matmul_stats) |stat| {
        if (stat.count != 0) any = true;
    }
    if (!any) return;

    try writer.print("{s}\n", .{title});
    try writer.print("  {s:<9} {s:<8} {s:<15} {s:>7} {s:>10} {s:>10} {s:>10} {s:>9}\n", .{
        "backend", "fmt", "shape", "calls", "command", "wrapper", "MAC/s", "MAC/cyc",
    });
    for (matmul_stats) |stat| {
        if (stat.count == 0) continue;
        var shape_buf: [32]u8 = undefined;
        var command_buf: [16]u8 = undefined;
        var wrapper_buf: [16]u8 = undefined;
        var macps_buf: [24]u8 = undefined;
        const mac_per_cyc = if (stat.cycles == 0) 0 else @as(f64, @floatFromInt(stat.macs)) / @as(f64, @floatFromInt(stat.cycles));
        try writer.print("  {s:<2}/{s:<6} {s:<8} {s:<15} {d:>7} {s:>10} {s:>10} {s:>10} {d:>9.1}\n", .{
            backendName(stat.backend),
            pathName(stat.path),
            weightFormatName(stat.fmt),
            std.fmt.bufPrint(&shape_buf, "{d}x{d}x{d}", .{ stat.rows, stat.cols, stat.k }) catch "?",
            stat.count,
            formatDuration(&command_buf, stat.command_ns / stat.count),
            formatDuration(&wrapper_buf, if (stat.count == 0) 0 else stat.wrapper_ns / stat.count),
            std.fmt.bufPrint(&macps_buf, "{d:.2} G/s", .{giga(stat.macs, stat.command_ns)}) catch unreachable,
            mac_per_cyc,
        });
        if (stat.backend != .pl) continue;
        const max_stall = @max(stat.w_stall_cycles, @max(stat.a_stall_cycles, stat.r_stall_cycles));
        const nonstall = if (stat.cycles == 0) 0 else percent(stat.cycles -| max_stall, stat.cycles);
        const weight_bytes = stat.w_beats * layout.weight_beat_bytes;
        const weight_bytes_per_cycle = if (stat.cycles == 0) 0 else @as(f64, @floatFromInt(weight_bytes)) / @as(f64, @floatFromInt(stat.cycles));
        const residual = if (stat.wrapper_ns >= stat.wrapperChildrenNs()) stat.wrapper_ns - stat.wrapperChildrenNs() else 0;
        var quant_buf: [16]u8 = undefined;
        var sync_to_buf: [16]u8 = undefined;
        var setup_buf: [16]u8 = undefined;
        var wait_buf: [16]u8 = undefined;
        var sync_from_buf: [16]u8 = undefined;
        var layout_buf: [16]u8 = undefined;
        var residual_buf: [16]u8 = undefined;
        try writer.print("      stages host-q/pack={s} sync-to={s} setup={s} wait={s} sync-from={s} layout={s} residual={s}\n", .{
            formatDuration(&quant_buf, stat.quantize_pack_ns),
            formatDuration(&sync_to_buf, stat.sync_to_ns),
            formatDuration(&setup_buf, stat.setup_ns),
            formatDuration(&wait_buf, stat.wait_ns),
            formatDuration(&sync_from_buf, stat.sync_from_ns),
            formatDuration(&layout_buf, stat.result_layout_ns),
            formatDuration(&residual_buf, residual),
        });
        if (stat.cycles != 0 or stat.w_beats != 0 or stat.a_beats != 0 or stat.r_beats != 0) {
            try writer.print("      kernel runs={d} cycles={d} nonstall={d:.1}% stalls W/A/R={d}/{d}/{d} axis-beats W/A/R={d}/{d}/{d} W={d:.2} B/cyc {d:.2} GB/s\n", .{
                stat.kernel_runs,
                stat.cycles,
                nonstall,
                stat.w_stall_cycles,
                stat.a_stall_cycles,
                stat.r_stall_cycles,
                stat.w_beats,
                stat.a_beats,
                stat.r_beats,
                weight_bytes_per_cycle,
                weightGbps(stat, fclk_hz),
            });
            saw_axis_counters = true;
        }
        if (clockImplausible(stat, fclk_hz)) {
            var busy_buf: [16]u8 = undefined;
            var wall_buf: [16]u8 = undefined;
            const busy_ns: u64 = @intFromFloat(@as(f64, @floatFromInt(stat.cycles)) /
                @as(f64, @floatFromInt(fclk_hz)) * 1_000_000_000.0);
            try writer.print("      note: CLK_HZ={d} implausible (kernel busy {s} > wall {s}) — W GB/s is wall-time; rebuild bitstream\n", .{
                fclk_hz,
                formatDuration(&busy_buf, busy_ns),
                formatDuration(&wall_buf, stat.wrapper_ns),
            });
        }
    }
    if (saw_axis_counters) {
        try writer.writeAll("  note: A beats are external 64-bit AXIS transfers; primitive loads use packed Q8, grouped loads use raw F32, and reuse consumes zero\n");
    }
}

/// Sub-millisecond latency with an auto unit (us >= 1us, else ns). For the
/// flash per-inner-iteration column, which lives in the ns–us range.
pub fn formatLatencyFine(buf: []u8, ns: f64) []const u8 {
    if (ns >= 1000.0) return std.fmt.bufPrint(buf, "{d:.2} us", .{ns / 1000.0}) catch unreachable;
    return std.fmt.bufPrint(buf, "{d:.0} ns", .{ns}) catch unreachable;
}

fn formatQuotient(buf: []u8, numerator: u64, denominator: u64) []const u8 {
    if (denominator == 0) return "n/a";
    return std.fmt.bufPrint(buf, "{d:.2}", .{
        @as(f64, @floatFromInt(numerator)) / @as(f64, @floatFromInt(denominator)),
    }) catch "?";
}

fn formatDensity(buf: []u8, valid: u64, processed: u64) []const u8 {
    if (processed == 0) return "n/a";
    return std.fmt.bufPrint(buf, "{d:.1}%", .{percent(valid, processed)}) catch "?";
}

fn formatMillionPerSecond(buf: []u8, work: u64, cycles: u64, fclk_hz: u32) []const u8 {
    if (work == 0 or cycles == 0 or fclk_hz == 0) return "n/a";
    const rate = @as(f64, @floatFromInt(work)) * @as(f64, @floatFromInt(fclk_hz)) /
        @as(f64, @floatFromInt(cycles)) / 1_000_000.0;
    return std.fmt.bufPrint(buf, "{d:.2}", .{rate}) catch "?";
}

/// Flash shape/path buckets report the padded request, mask entries not equal to
/// f16 negative infinity, and the backend walk separately. Bytes and counters
/// characterize work; they are not added to the elapsed-time budget.
pub fn writeFlashDetail(
    writer: *std.Io.Writer,
    title: []const u8,
    flash_stats: []const profiling.FlashStat,
    fclk_hz: u32,
) std.Io.Writer.Error!void {
    if (flash_stats.len == 0) return;
    try writer.print("{s}\n", .{title});
    try writer.print("  {s:<9} {s:>6} {s:>7} {s:>13} {s:>13} {s:>13} {s:>10} {s:>10}\n", .{
        "backend", "tokens", "calls", "requested", "valid", "processed", "command", "wrapper",
    });
    for (flash_stats) |flash| {
        if (flash.count == 0) continue;
        const count: f64 = @floatFromInt(flash.count);
        var requested_buf: [20]u8 = undefined;
        var valid_buf: [20]u8 = undefined;
        var processed_buf: [20]u8 = undefined;
        var command_buf: [16]u8 = undefined;
        var wrapper_buf: [16]u8 = undefined;
        try writer.print("  {s:<2}/{s:<6} {d:>6} {d:>7} {s:>13} {s:>13} {s:>13} {s:>10} {s:>10}\n", .{
            backendName(flash.backend),
            pathName(flash.path),
            flash.n_tokens,
            flash.count,
            std.fmt.bufPrint(&requested_buf, "{d:.1}/{d}", .{ @as(f64, @floatFromInt(flash.requested_n_kv_sum)) / count, flash.requested_n_kv_max }) catch "?",
            std.fmt.bufPrint(&valid_buf, "{d:.1}/{d}", .{ @as(f64, @floatFromInt(flash.valid_n_kv_sum)) / count, flash.valid_n_kv_max }) catch "?",
            std.fmt.bufPrint(&processed_buf, "{d:.1}/{d}", .{ @as(f64, @floatFromInt(flash.processed_n_kv_sum)) / count, flash.processed_n_kv_max }) catch "?",
            formatDuration(&command_buf, flash.command_ns / flash.count),
            formatDuration(&wrapper_buf, flash.wrapper_ns / flash.count),
        });
        try writer.print("      query-KV pairs total requested={d} valid={d} processed={d} (per call {d:.1}/{d:.1}/{d:.1})\n", .{
            flash.requested_qkv_pairs,
            flash.valid_qkv_pairs,
            flash.processed_qkv_pairs,
            @as(f64, @floatFromInt(flash.requested_qkv_pairs)) / count,
            @as(f64, @floatFromInt(flash.valid_qkv_pairs)) / count,
            @as(f64, @floatFromInt(flash.processed_qkv_pairs)) / count,
        });
        if (flash.backend != .pl) continue;
        const residual = if (flash.wrapper_ns >= flash.wrapperChildrenNs()) flash.wrapper_ns - flash.wrapperChildrenNs() else 0;
        var prep_buf: [16]u8 = undefined;
        var sync_to_buf: [16]u8 = undefined;
        var setup_buf: [16]u8 = undefined;
        var wait_buf: [16]u8 = undefined;
        var sync_from_buf: [16]u8 = undefined;
        var layout_buf: [16]u8 = undefined;
        var residual_buf: [16]u8 = undefined;
        try writer.print("      stages prepare={s} sync-to={s} setup={s} wait={s} sync-from={s} layout={s} residual={s}\n", .{
            formatDuration(&prep_buf, flash.prepare_ns),
            formatDuration(&sync_to_buf, flash.sync_to_ns),
            formatDuration(&setup_buf, flash.setup_ns),
            formatDuration(&wait_buf, flash.wait_ns),
            formatDuration(&sync_from_buf, flash.sync_from_ns),
            formatDuration(&layout_buf, flash.result_layout_ns),
            formatDuration(&residual_buf, residual),
        });
        try writer.print("      kernel runs={d} cycles={d} beats Q/K/V/O={d}/{d}/{d}/{d} stalls K/V/O={d}/{d}/{d} DMA={d:.1} MiB/s\n", .{
            flash.kernel_runs,
            flash.cycles,
            flash.q_beats,
            flash.k_beats,
            flash.v_beats,
            flash.o_beats,
            flash.k_stall_cycles,
            flash.v_stall_cycles,
            flash.o_stall_cycles,
            mibPerSecond(flash.q_bytes +| flash.k_bytes +| flash.v_bytes +| flash.mask_bytes +| flash.o_bytes, flash.wrapper_ns),
        });
        var cycles_run_buf: [24]u8 = undefined;
        var cycles_valid_buf: [24]u8 = undefined;
        var cycles_processed_buf: [24]u8 = undefined;
        var density_buf: [24]u8 = undefined;
        var rate_buf: [24]u8 = undefined;
        try writer.print("      efficiency cyc/run={s} cyc/valid-qhkv={s} cyc/processed-qhkv={s} valid-density={s} Mvalid-qhkv/s={s}\n", .{
            formatQuotient(&cycles_run_buf, flash.cycles, flash.kernel_runs),
            formatQuotient(&cycles_valid_buf, flash.cycles, flash.validQueryHeadKvUpdates()),
            formatQuotient(&cycles_processed_buf, flash.cycles, flash.processedQueryHeadKvUpdates()),
            formatDensity(&density_buf, flash.validQueryHeadKvUpdates(), flash.processedQueryHeadKvUpdates()),
            formatMillionPerSecond(&rate_buf, flash.validQueryHeadKvUpdates(), flash.cycles, fclk_hz),
        });
    }
}

const AttentionTotals = struct {
    cycles: u64 = 0,
    pl_valid_qhkv_updates: u64 = 0,
};

fn attentionTotals(stats: []const profiling.FlashStat) AttentionTotals {
    var out: AttentionTotals = .{};
    for (stats) |stat| {
        if (stat.backend != .pl) continue;
        out.cycles +|= stat.cycles;
        out.pl_valid_qhkv_updates +|= stat.validQueryHeadKvUpdates();
    }
    return out;
}

/// Stable, integer-only records for benchmark artifacts. Each line preserves an
/// exact backend/path/shape bucket so consumers never compare unlike head shapes
/// accidentally. Derived rates remain a consumer concern.
pub fn writeAttentionResult(
    writer: *std.Io.Writer,
    phase: []const u8,
    stats: []const profiling.FlashStat,
    fclk_hz: u32,
) std.Io.Writer.Error!void {
    for (stats) |stat| {
        try writer.print(
            "attention_result schema=1 phase={s} backend={s} path={s}" ++
                " n_heads={d} n_head_kv={d} head_dim_q={d} head_dim_v={d} n_tokens={d}" ++
                " calls={d} fclk_hz={d} kernel_runs={d} cycles={d}" ++
                " requested_qkv_pairs={d} valid_qkv_pairs={d} processed_qkv_pairs={d}" ++
                " valid_qhkv_updates={d} processed_qhkv_updates={d}" ++
                " q_bytes={d} k_bytes={d} v_bytes={d} mask_bytes={d} o_bytes={d}" ++
                " q_beats={d} k_beats={d} v_beats={d} o_beats={d}" ++
                " k_stall_cycles={d} v_stall_cycles={d} o_stall_cycles={d}\n",
            .{
                phase,
                backendName(stat.backend),
                pathName(stat.path),
                stat.n_heads,
                stat.n_head_kv,
                stat.head_dim_q,
                stat.head_dim_v,
                stat.n_tokens,
                stat.count,
                fclk_hz,
                stat.kernel_runs,
                stat.cycles,
                stat.requested_qkv_pairs,
                stat.valid_qkv_pairs,
                stat.processed_qkv_pairs,
                stat.validQueryHeadKvUpdates(),
                stat.processedQueryHeadKvUpdates(),
                stat.q_bytes,
                stat.k_bytes,
                stat.v_bytes,
                stat.mask_bytes,
                stat.o_bytes,
                stat.q_beats,
                stat.k_beats,
                stat.v_beats,
                stat.o_beats,
                stat.k_stall_cycles,
                stat.v_stall_cycles,
                stat.o_stall_cycles,
            },
        );
    }
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
        layout.rows_per_block * 32, layout.weight_ports, mhz,
    }) catch "unknown";

    var mm: profiling.MatmulStat = .{};
    mm.backend = .pl;
    mm.path = .direct;
    for (rg.usedMatmul()) |s| {
        if (s.backend != .pl) continue;
        mm.count +|= s.count;
        mm.kernel_runs +|= s.kernel_runs;
        mm.macs +|= s.macs;
        mm.command_ns +|= s.command_ns;
        mm.wrapper_ns +|= s.wrapper_ns;
        mm.cycles +|= s.cycles;
        mm.w_stall_cycles +|= s.w_stall_cycles;
        mm.w_beats +|= s.w_beats;
    }
    const mac_cyc = if (mm.cycles == 0) 0 else @as(f64, @floatFromInt(mm.macs)) / @as(f64, @floatFromInt(mm.cycles));
    const flash_ns = rg.op_totals[@intFromEnum(wire.OpTag.flash_attn_f32)].total_ns;
    const swiglu_ns = rg.op_totals[@intFromEnum(wire.OpTag.swiglu)].total_ns;
    const ffn_ns = rg.op_totals[@intFromEnum(wire.OpTag.ffn_section)].total_ns;
    const fa = attentionTotals(rg.usedFlash());
    const flash_cycles_valid = if (fa.pl_valid_qhkv_updates == 0) 0 else @as(f64, @floatFromInt(fa.cycles)) / @as(f64, @floatFromInt(fa.pl_valid_qhkv_updates));
    const flash_mvalid_s = if (fa.cycles == 0 or rg.device_fclk_hz == 0) 0 else @as(f64, @floatFromInt(fa.pl_valid_qhkv_updates)) * @as(f64, @floatFromInt(rg.device_fclk_hz)) /
        @as(f64, @floatFromInt(fa.cycles)) / 1_000_000.0;

    try writer.print(
        "scoreboard variant={s} tok_s={d:.2} decode_ms={d:.1} device_ms={d:.1} transport_ms={d:.1}" ++
            " matmul_gmac_s={d:.2} matmul_mac_cyc={d:.1} matmul_w_gbps={d:.2} matmul_w_stall_pct={d:.1}" ++
            " flash_ms_tok={d:.1} flash_cyc_valid_qhkv={d:.2} flash_mvalid_qhkv_s={d:.2}" ++
            " swiglu_ms_tok={d:.1} ffn_ms_tok={d:.1}\n",
        .{
            variant,
            perSecond(tokens, decode.wall_ns),
            nsToMs(decode.wall_ns) / t,
            nsToMs(decode.deviceNs()) / t,
            nsToMs(decode.transportNs()) / t,
            giga(mm.macs, mm.command_ns),
            mac_cyc,
            weightGbps(mm, rg.device_fclk_hz),
            percent(mm.w_stall_cycles, mm.cycles),
            nsToMs(flash_ns) / t,
            flash_cycles_valid,
            flash_mvalid_s,
            nsToMs(swiglu_ns) / t,
            nsToMs(ffn_ns) / t,
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

/// The full `--prof` report: header + phase budget + transfers + upload census +
/// per-phase op/matmul/flash detail + the decode scoreboard. Reads a Collector;
/// the budget math lives in `model`, the per-section writers above.
pub fn writeProfile(
    writer: *std.Io.Writer,
    c: *const collector_mod.Collector,
    model_path: []const u8,
    device_label: []const u8,
) std.Io.Writer.Error!void {
    var buf: [32]u8 = undefined;
    try writer.print("penzai profile\n\n", .{});
    try writer.print("  model        {s}\n", .{modelName(model_path)});
    try writer.print("  device       {s}\n", .{device_label});
    try writer.print("  prompt tokens {d}\n", .{c.prompt_tokens});
    try writer.print("  tokens       {d}\n", .{c.generated_tokens});
    if (c.generated_tokens > 0) {
        var avg_buf: [32]u8 = undefined;
        var first_buf: [32]u8 = undefined;
        var compute_buf: [32]u8 = undefined;
        var output_buf: [32]u8 = undefined;
        const steady_avg = if (c.steady_decode_count == 0) 0 else c.steady_decode_ns / c.steady_decode_count;
        try writer.print("  decode       {s}/steady-token ({d:.2} tok/s)\n", .{
            formatDuration(&avg_buf, steady_avg),
            perSecond(c.steady_decode_count, c.steady_decode_ns),
        });
        try writer.print("  first step   {s}\n", .{formatDuration(&first_buf, c.first_decode_step_ns)});
        try writer.print("  compute TTFT {s}\n", .{formatDuration(&compute_buf, c.compute_ttft_ns)});
        if (c.output_ttft_ns != 0)
            try writer.print("  output TTFT  {s}\n", .{formatDuration(&output_buf, c.output_ttft_ns)});
    }
    try writer.print("  wall         {s}\n\n", .{formatDuration(&buf, c.wallNs())});

    // The headline: wall = device + transport + residual, per phase.
    try writePhaseBudget(writer, &c.phases);
    try writer.writeByte('\n');
    try writeDeviceAccounting(writer, &c.phases);
    try writer.writeByte('\n');
    try writeTransfers(writer, &c.phases);
    try writer.writeByte('\n');
    try writeUploadCensus(writer, &c.upload_census);

    // Device-side op + matmul detail, separately per phase that ran graphs —
    // prefill (compute-bound) and decode (bandwidth-bound) have distinct
    // rooflines, so a merged op table would be meaningless.
    for (&c.phases, 0..) |*phase, i| {
        if (phase.rg.run_graph_count == 0) continue;
        const label = (@as(Phase, @enumFromInt(i))).label();
        var ops_title: [96]u8 = undefined;
        var mm_title: [48]u8 = undefined;
        try writer.writeByte('\n');
        try writeOpTable(
            writer,
            std.fmt.bufPrint(&ops_title, "ops \u{b7} {s}  {d} graphs, {d} cmds", .{ label, phase.rg.run_graph_count, phase.rg.command_count }) catch "ops",
            &phase.rg.op_totals,
            phase.rg.device_execute_ns,
        );
        try writeMatmulDetail(
            writer,
            std.fmt.bufPrint(&mm_title, "matmul \u{b7} {s}", .{label}) catch "matmul",
            phase.rg.usedMatmul(),
            phase.rg.device_fclk_hz,
        );
        var fa_title: [48]u8 = undefined;
        try writeFlashDetail(
            writer,
            std.fmt.bufPrint(&fa_title, "flash \u{b7} {s}", .{label}) catch "flash",
            phase.rg.usedFlash(),
            phase.rg.device_fclk_hz,
        );
    }

    // One greppable comparison line for the decode phase (the roofline target).
    const decode = &c.phases[@intFromEnum(Phase.decode)];
    if (decode.rg.run_graph_count != 0) {
        try writer.writeByte('\n');
        try writeScoreboard(writer, decode, c.generated_tokens);
    }

    // Raw device-counter records for artifact tooling. These remain valid when
    // wall and transport latency are distorted by a remote tunnel.
    for (&c.phases, 0..) |*phase, i| {
        if (phase.rg.flash_count == 0) continue;
        try writeAttentionResult(writer, (@as(Phase, @enumFromInt(i))).label(), phase.rg.usedFlash(), phase.rg.device_fclk_hz);
    }

    var accounting: AccountingStatus = .ok;
    for (c.phases) |phase| accounting = accounting.combine(phase.accountingStatus());
    const prefill = &c.phases[@intFromEnum(Phase.prefill)];
    try writer.print(
        "benchmark_result schema=1 prompt_tokens={d} generated_tokens={d}" ++
            " prefill_wall_ns={d} first_decode_step_ns={d} compute_ttft_ns={d}" ++
            " output_ttft_ns={d} steady_decode_ns={d} steady_decode_count={d}" ++
            " decode_wall_ns={d} decode_device_ns={d} decode_transport_ns={d}" ++
            " decode_residual_ns={d} accounting={s}\n",
        .{
            c.prompt_tokens,
            c.generated_tokens,
            prefill.wall_ns,
            c.first_decode_step_ns,
            c.compute_ttft_ns,
            c.output_ttft_ns,
            c.steady_decode_ns,
            c.steady_decode_count,
            decode.wall_ns,
            decode.deviceNs(),
            decode.transportNs(),
            decode.residualNs(),
            accounting.label(),
        },
    );
}

/// Basename of the model path with the `.gguf` suffix removed, for the header.
fn modelName(path: []const u8) []const u8 {
    var name = path;
    if (std.mem.lastIndexOfScalar(u8, name, '/')) |i| name = name[i + 1 ..];
    if (std.mem.endsWith(u8, name, ".gguf")) name = name[0 .. name.len - 5];
    return name;
}

test "flash efficiency renders empty valid work as n/a" {
    var out: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer out.deinit();
    const stats = [_]profiling.FlashStat{.{
        .backend = .pl,
        .path = .direct,
        .n_heads = 16,
        .count = 1,
        .kernel_runs = 1,
        .n_tokens = 1,
        .processed_qkv_pairs = 8,
        .cycles = 800,
    }};
    try writeFlashDetail(&out.writer, "flash", &stats, 300_000_000);
    try std.testing.expect(std.mem.indexOf(u8, out.written(), "cyc/valid-qhkv=n/a") != null);
    try std.testing.expect(std.mem.indexOf(u8, out.written(), "cyc/processed-qhkv=6.25") != null);
}

test "attention results retain PS and PL shapes" {
    var out: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer out.deinit();
    const stats = [_]profiling.FlashStat{
        .{ .backend = .ps, .n_heads = 16, .count = 2, .valid_qkv_pairs = 10, .processed_qkv_pairs = 12 },
        .{ .backend = .pl, .n_heads = 16, .count = 3, .kernel_runs = 3, .cycles = 24_000, .valid_qkv_pairs = 100, .processed_qkv_pairs = 125 },
    };
    try writeAttentionResult(&out.writer, "decode", &stats, 300_000_000);
    try std.testing.expectEqual(@as(usize, 2), std.mem.count(u8, out.written(), "attention_result "));
    try std.testing.expect(std.mem.indexOf(u8, out.written(), "backend=ps path=software") != null);
    try std.testing.expect(std.mem.indexOf(u8, out.written(), "backend=pl path=software") != null);
    try std.testing.expect(std.mem.indexOf(u8, out.written(), "valid_qkv_pairs=100 processed_qkv_pairs=125 valid_qhkv_updates=1600") != null);
}
