const std = @import("std");
const shared = @import("shared");

const wire = shared.wire;
const profiling = shared.profiling;
const layout = shared.layout;

/// Hardware counters read from the PL kernel after a matmul run. Cycle/stall/beat
/// counts; all zero for a CPU matmul. The PL backend produces these and the
/// runtime hands them to `Collector.record`.
pub const PlCounters = struct {
    cycles: u64 = 0,
    w_stall: u64 = 0,
    a_stall: u64 = 0,
    r_stall: u64 = 0,
    w_beats: u64 = 0,
    a_beats: u64 = 0,
    r_beats: u64 = 0,
};

/// Device-side per-graph profile collection. The runtime drives a Collector
/// around its execute loop; the Collector owns the aggregate bookkeeping and
/// produces the wire payload. All accumulators are fixed-size, so a Collector is
/// a plain value with no allocation.
pub const Collector = struct {
    aggregates: [profiling.max_op_tag + 1]profiling.Aggregate =
        [_]profiling.Aggregate{.{}} ** (profiling.max_op_tag + 1),
    /// Per-(weight format) matmul rollup, indexed by the wire.WeightFormat value.
    /// macs/total_ns are filled here; the PL counter fields stay zero until the
    /// PL backend reports them (P2).
    matmul_stats: [profiling.max_weight_fmt]profiling.MatmulStat =
        [_]profiling.MatmulStat{.{}} ** profiling.max_weight_fmt,
    /// Flash-attention shape rollup. One flash op kind, so a single accumulator
    /// (emitted as a 0-or-1 element slice on the wire).
    flash_stat: profiling.FlashStat = .{},

    /// Record one executed command. Timestamps are device-clock nanoseconds.
    pub fn record(
        self: *Collector,
        command: wire.Command,
        start_ns: u64,
        end_ns: u64,
        pl: ?PlCounters,
    ) void {
        const raw_tag: u16 = @intFromEnum(commandTag(command));
        const bytes = commandBytes(command);
        const index: usize = raw_tag;
        if (index < self.aggregates.len) {
            var aggregate = &self.aggregates[index];
            aggregate.tag = raw_tag;
            aggregate.count +|= 1;
            aggregate.total_ns +|= profiling.elapsed(start_ns, end_ns);
            aggregate.bytes +|= bytes;
        }
        switch (command) {
            .matmul_q1a8 => |mm| {
                const fmt: u16 = @intCast(@intFromEnum(mm.weight_fmt));
                if (fmt < self.matmul_stats.len) {
                    var stat = &self.matmul_stats[fmt];
                    stat.fmt = fmt;
                    stat.count +|= 1;
                    const macs = mul(mul(mm.rows, mm.cols), mm.k);
                    const ns = profiling.elapsed(start_ns, end_ns);
                    stat.macs +|= macs;
                    stat.total_ns +|= ns;
                    if (pl) |c| {
                        stat.pl_count +|= 1;
                        stat.pl_macs +|= macs;
                        stat.pl_ns +|= ns;
                        stat.cycles +|= c.cycles;
                        stat.w_stall_cycles +|= c.w_stall;
                        stat.a_stall_cycles +|= c.a_stall;
                        stat.r_stall_cycles +|= c.r_stall;
                        stat.w_beats +|= c.w_beats;
                        stat.a_beats +|= c.a_beats;
                        stat.r_beats +|= c.r_beats;
                    }
                }
            },
            .flash_attn_f32 => |fa| {
                // Shape fields are constant per model (last-wins capture); only
                // n_kv varies, so its sum/max describe the per-call distribution.
                var s = &self.flash_stat;
                s.count +|= 1;
                s.n_heads = sat16(fa.n_heads);
                s.n_head_kv = sat16(fa.n_head_kv);
                s.head_dim_q = sat16(fa.head_dim_q);
                s.head_dim_v = sat16(fa.head_dim_v);
                s.sum_n_kv +|= fa.n_kv;
                s.max_n_kv = @max(s.max_n_kv, fa.n_kv);
                s.total_ns +|= profiling.elapsed(start_ns, end_ns);
                s.bytes +|= bytes;
            },
            else => {},
        }
    }

    /// Encode the collected report. `summary` carries the caller's timing and
    /// command_count.
    pub fn encode(
        self: *const Collector,
        allocator: std.mem.Allocator,
        summary: profiling.Summary,
    ) std.mem.Allocator.Error![]u8 {
        var used: [profiling.max_op_tag + 1]profiling.Aggregate = undefined;
        var n: usize = 0;
        for (self.aggregates) |aggregate| {
            if (aggregate.count == 0) continue;
            used[n] = aggregate;
            n += 1;
        }
        var used_stats: [profiling.max_weight_fmt]profiling.MatmulStat = undefined;
        var sn: usize = 0;
        for (self.matmul_stats) |stat| {
            if (stat.count == 0) continue;
            used_stats[sn] = stat;
            sn += 1;
        }
        var flash_buf = [_]profiling.FlashStat{self.flash_stat};
        const flash_used = if (self.flash_stat.count == 0) flash_buf[0..0] else flash_buf[0..1];
        return profiling.encodeAlloc(allocator, .{
            .summary = summary,
            .aggregates = used[0..n],
            .matmul_stats = used_stats[0..sn],
            .flash_stats = flash_used,
        });
    }
};

pub fn commandTag(command: wire.Command) wire.OpTag {
    return switch (command) {
        .copy => .copy,
        .cpy_f32_to_f16 => .cpy_f32_to_f16,
        .matmul_q1a8 => .matmul_q1a8,
        .rmsnorm => .rmsnorm,
        .rope => .rope,
        .softmax => .softmax,
        .silu => .silu,
        .swiglu => .swiglu,
        .add_f32 => .add_f32,
        .mul_f32 => .mul_f32,
        .scale_f32 => .scale_f32,
        .add_scaled_f32 => .add_scaled_f32,
        .set_rows => .set_rows,
        .get_rows => .get_rows,
        .flash_attn_f32 => .flash_attn_f32,
        .argmax => .argmax,
        .pad => .pad,
    };
}

pub fn commandBytes(command: wire.Command) u64 {
    // Saturating sums: commandBytes runs before the kernel validates ranges, so a
    // hostile command with huge nbytes must not overflow/panic here.
    return switch (command) {
        .copy => |op| op.src.nbytes +| op.dst.nbytes,
        .cpy_f32_to_f16 => |op| op.src.nbytes +| op.dst.nbytes,
        // Per-op read traffic: weights are persistent but re-read every decode, so count them each call.
        .matmul_q1a8 => |op| op.weights.nbytes +| op.acts.nbytes +| op.dst.nbytes,
        .rmsnorm => |op| op.input.nbytes +| op.dst.nbytes,
        .rope => |op| op.input.nbytes +| op.positions.nbytes +| op.dst.nbytes,
        .softmax => |op| op.src.nbytes +| op.dst.nbytes,
        .silu => |op| op.src.nbytes +| op.dst.nbytes,
        .swiglu => |op| op.lhs.nbytes +| op.rhs.nbytes +| op.dst.nbytes,
        .add_f32 => |op| op.lhs.nbytes +| op.rhs.nbytes +| op.dst.nbytes,
        .mul_f32 => |op| op.lhs.nbytes +| op.rhs.nbytes +| op.dst.nbytes,
        .scale_f32 => |op| op.src.nbytes +| op.dst.nbytes,
        .add_scaled_f32 => |op| op.lhs.nbytes +| op.rhs.nbytes +| op.dst.nbytes,
        .set_rows => |op| setRowsBytes(op),
        .get_rows => |op| getRowsBytes(op),
        .flash_attn_f32 => |op| op.q.nbytes +| op.k.nbytes +| op.v.nbytes +| (if (op.has_mask) op.mask.nbytes else 0) +| op.dst.nbytes,
        .argmax => |op| op.src.nbytes +| op.dst.nbytes,
        .pad => |op| op.src.nbytes +| op.dst.nbytes,
    };
}

fn setRowsBytes(op: wire.SetRows) u64 {
    const rows = mul(mul(op.ne01, op.ne02), op.ne03);
    const index_size: u64 = switch (op.index_type) {
        .i32 => @sizeOf(i32),
        .i64 => @sizeOf(i64),
    };
    const src_row_bytes = mul(op.head_dim, @sizeOf(f32));
    const dst_row_bytes = mul(op.head_dim, @sizeOf(f16));
    return mul(rows, src_row_bytes +| index_size +| dst_row_bytes);
}

fn getRowsBytes(op: wire.GetRows) u64 {
    const rows = mul(mul(op.ne10, op.ne11), op.ne12);
    const index_bytes: u64 = @sizeOf(i32);
    const dst_row_bytes = mul(op.row_width, @sizeOf(f32));
    const src_row_bytes: u64 = switch (op.src_type) {
        .f32 => mul(op.row_width, @sizeOf(f32)),
        .q1_0 => q1GetRowsSourceBytes(op.row_width),
        .q2_0 => ternaryGetRowsSourceBytes(op.row_width),
    };
    return mul(rows, src_row_bytes +| index_bytes +| dst_row_bytes);
}

fn q1GetRowsSourceBytes(row_width: u32) u64 {
    if (row_width == 0) return 0;
    const blocks = (@as(u64, row_width) + layout.q1_block - 1) / layout.q1_block;
    const bytes_per_block = layout.beat_bytes * (1 + layout.q8_subblocks);
    return mul(blocks, bytes_per_block);
}

fn ternaryGetRowsSourceBytes(row_width: u32) u64 {
    if (row_width == 0) return 0;
    const blocks = (@as(u64, row_width) + layout.q1_block - 1) / layout.q1_block;
    return mul(blocks, layout.ternary_block_bytes);
}

fn mul(a: u64, b: u64) u64 {
    return a *| b;
}

/// Saturating u32→u16 for profile shape fields (post-execute values are sane;
/// this just guards the device ReleaseFast path against a silent wrap).
fn sat16(v: u32) u16 {
    return std.math.cast(u16, v) orelse std.math.maxInt(u16);
}

test "profile byte estimates for row ops count touched rows, not backing spans" {
    const large = wire.TensorRange{ .handle = 1, .offset = 0, .nbytes = 1_000_000 };
    const indices = wire.TensorRange{ .handle = 2, .offset = 0, .nbytes = 4096 };
    const set_rows = wire.Command{ .set_rows = .{
        .src = large,
        .indices = indices,
        .dst = large,
        .index_type = .i32,
        .head_dim = 256,
        .ne01 = 13,
        .ne02 = 1,
        .ne03 = 1,
        .ne11 = 1,
        .ne12 = 1,
        .src_nb1 = 256 * @sizeOf(f32),
        .src_nb2 = 13 * 256 * @sizeOf(f32),
        .src_nb3 = 13 * 256 * @sizeOf(f32),
        .indices_nb1 = 13 * @sizeOf(i32),
        .indices_nb2 = 13 * @sizeOf(i32),
        .dst_nb1 = 256 * @sizeOf(f16),
        .dst_nb2 = 13 * 256 * @sizeOf(f16),
        .dst_nb3 = 13 * 256 * @sizeOf(f16),
    } };
    try std.testing.expectEqual(@as(u64, 13 * (256 * 4 + 4 + 256 * 2)), commandBytes(set_rows));

    const get_rows = wire.Command{ .get_rows = .{
        .src = large,
        .indices = indices,
        .dst = large,
        .src_type = .q1_0,
        .row_width = 256,
        .src_rows = 1024,
        .ne10 = 2,
        .ne11 = 3,
        .ne12 = 1,
        .src_nb1 = 256 * @sizeOf(f32),
        .src_nb2 = 256 * @sizeOf(f32),
        .src_nb3 = 256 * @sizeOf(f32),
        .indices_nb1 = 2 * @sizeOf(i32),
        .indices_nb2 = 6 * @sizeOf(i32),
        .dst_nb1 = 256 * @sizeOf(f32),
        .dst_nb2 = 2 * 256 * @sizeOf(f32),
        .dst_nb3 = 6 * 256 * @sizeOf(f32),
    } };
    const q1_source_bytes = 2 * layout.beat_bytes * (1 + layout.q8_subblocks);
    try std.testing.expectEqual(@as(u64, 6 * (q1_source_bytes + 4 + 256 * 4)), commandBytes(get_rows));

    var ternary_rows = get_rows;
    ternary_rows.get_rows.src_type = .q2_0;
    const ternary_source_bytes = 2 * layout.ternary_block_bytes;
    try std.testing.expectEqual(@as(u64, 6 * (ternary_source_bytes + 4 + 256 * 4)), commandBytes(ternary_rows));
}
