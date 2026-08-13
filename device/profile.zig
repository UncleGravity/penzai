const std = @import("std");
const shared = @import("shared");

const wire = shared.wire;
const profiling = shared.profiling;
const layout = shared.layout;

pub const ProfileContext = struct {
    collector: *Collector,
    io: ?std.Io,
    timer_ns: u64 = 0,

    pub fn begin(self: *ProfileContext) u64 {
        self.timer_ns = profiling.nowNs(self.io);
        return self.timer_ns;
    }

    pub fn lap(self: *ProfileContext, last: *u64) u64 {
        const current_ns = profiling.nowNs(self.io);
        const ns = profiling.elapsed(last.*, current_ns);
        last.* = current_ns;
        self.timer_ns = current_ns;
        return ns;
    }

    pub fn now(self: *ProfileContext) u64 {
        self.timer_ns = profiling.nowNs(self.io);
        return self.timer_ns;
    }
};

pub fn begin(ctx: ?*ProfileContext) u64 {
    return if (ctx) |active| active.begin() else 0;
}

pub fn lap(ctx: ?*ProfileContext, last: *u64) u64 {
    return if (ctx) |active| active.lap(last) else 0;
}

pub const MatmulExecution = struct {
    path: profiling.ExecutionPath,
    kernel_runs: u32 = 0,
    wrapper_ns: u64 = 0,
    quantize_pack_ns: u64 = 0,
    sync_to_ns: u64 = 0,
    setup_ns: u64 = 0,
    wait_ns: u64 = 0,
    sync_from_ns: u64 = 0,
    result_layout_ns: u64 = 0,
    cycles: u64 = 0,
    w_stall_cycles: u64 = 0,
    a_stall_cycles: u64 = 0,
    r_stall_cycles: u64 = 0,
    w_beats: u64 = 0,
    a_beats: u64 = 0,
    r_beats: u64 = 0,
};

/// Nested timings for a named FFN command. The command aggregate remains the
/// authoritative wall time; these fields attribute its internal stages and
/// feed the existing matmul buckets without changing the profiling wire ABI.
pub const FfnExecution = struct {
    gate_up: MatmulExecution = .{ .path = .software },
    down: MatmulExecution = .{ .path = .software },
    gate_up_command_ns: u64 = 0,
    down_command_ns: u64 = 0,
    norm_ns: u64 = 0,
    swiglu_ns: u64 = 0,
    residual_add_ns: u64 = 0,
};

pub const FlashExecution = struct {
    path: profiling.ExecutionPath,
    kernel_runs: u32 = 0,
    wrapper_ns: u64 = 0,
    prepare_ns: u64 = 0,
    sync_to_ns: u64 = 0,
    setup_ns: u64 = 0,
    wait_ns: u64 = 0,
    sync_from_ns: u64 = 0,
    result_layout_ns: u64 = 0,
    requested_n_kv: u32 = 0,
    valid_n_kv: u32 = 0,
    processed_n_kv: u32 = 0,
    requested_qkv_pairs: u64 = 0,
    valid_qkv_pairs: u64 = 0,
    processed_qkv_pairs: u64 = 0,
    q_bytes: u64 = 0,
    k_bytes: u64 = 0,
    v_bytes: u64 = 0,
    mask_bytes: u64 = 0,
    o_bytes: u64 = 0,
    cycles: u64 = 0,
    q_beats: u64 = 0,
    k_beats: u64 = 0,
    k_stall_cycles: u64 = 0,
    v_beats: u64 = 0,
    v_stall_cycles: u64 = 0,
    o_beats: u64 = 0,
    o_stall_cycles: u64 = 0,
};

pub const CommandOutcome = struct {
    backend: profiling.Backend = .ps,
    path: profiling.ExecutionPath = .software,
    detail: Detail = .none,

    pub const Detail = union(enum) {
        none,
        matmul: MatmulExecution,
        flash: FlashExecution,
        ffn: FfnExecution,
    };
};

/// Device-side per-graph profile collection. The runtime drives a Collector
/// around its execute loop; the Collector owns the aggregate bookkeeping and
/// produces the wire payload. All accumulators are fixed-size, so a Collector is
/// a plain value with no allocation.
pub const Collector = struct {
    aggregates: [profiling.max_op_tag + 1]profiling.Aggregate =
        [_]profiling.Aggregate{.{}} ** (profiling.max_op_tag + 1),
    matmul_stats: [profiling.max_matmul_buckets]profiling.MatmulStat =
        [_]profiling.MatmulStat{.{}} ** profiling.max_matmul_buckets,
    flash_stats: [profiling.max_flash_buckets]profiling.FlashStat =
        [_]profiling.FlashStat{.{}} ** profiling.max_flash_buckets,
    matmul_count: usize = 0,
    flash_count: usize = 0,
    matmul_overflow: u32 = 0,
    flash_overflow: u32 = 0,
    accounting_violations: u32 = 0,

    /// Record one executed command. Timestamps are device-clock nanoseconds.
    pub fn record(
        self: *Collector,
        command: wire.Command,
        command_ns: u64,
        outcome: CommandOutcome,
    ) void {
        const raw_tag: u16 = @intFromEnum(commandTag(command));
        const bytes = commandBytes(command);
        const index: usize = raw_tag;
        if (index < self.aggregates.len) {
            var aggregate = &self.aggregates[index];
            aggregate.tag = raw_tag;
            aggregate.count +|= 1;
            aggregate.total_ns +|= command_ns;
            aggregate.bytes +|= bytes;
        }
        switch (command) {
            .matmul_q1a8 => |mm| {
                const detail = switch (outcome.detail) {
                    .matmul => |value| value,
                    else => MatmulExecution{ .path = outcome.path },
                };
                self.recordMatmul(mm.rows, mm.cols, mm.k, mm.weight_fmt, 1, 1, command_ns, outcome, detail);
            },
            .matmul_q1a8_group2 => |group| {
                const group_detail = switch (outcome.detail) {
                    .matmul => |value| value,
                    else => MatmulExecution{ .path = outcome.path },
                };
                // The matcher requires identical projection shapes and formats,
                // so one shape record represents one command and two MAC sets.
                self.recordMatmul(
                    group.projections[0].rows,
                    group.cols,
                    group.k,
                    group.projections[0].weight_fmt,
                    1,
                    group.projections.len,
                    command_ns,
                    outcome,
                    group_detail,
                );
            },
            .ffn_section => |ffn| {
                const detail = switch (outcome.detail) {
                    .ffn => |value| value,
                    else => FfnExecution{
                        .gate_up = .{ .path = outcome.path },
                        .down = .{ .path = outcome.path },
                    },
                };
                self.recordMatmul(
                    ffn.ffn_dim,
                    ffn.token_count,
                    ffn.model_dim,
                    ffn.weight_fmt,
                    1,
                    2,
                    detail.gate_up_command_ns,
                    outcome,
                    detail.gate_up,
                );
                self.recordMatmul(
                    ffn.model_dim,
                    ffn.token_count,
                    ffn.ffn_dim,
                    ffn.weight_fmt,
                    1,
                    1,
                    detail.down_command_ns,
                    outcome,
                    detail.down,
                );
                const stage_ns = detail.norm_ns +| detail.gate_up_command_ns +|
                    detail.swiglu_ns +| detail.down_command_ns +| detail.residual_add_ns;
                if (stage_ns > command_ns)
                    self.accounting_violations |= profiling.AccountingViolation.wrapper_segments;
            },
            .flash_attn_f32 => |fa| {
                const detail = switch (outcome.detail) {
                    .flash => |value| value,
                    else => FlashExecution{
                        .path = outcome.path,
                        .requested_n_kv = fa.n_kv,
                        .valid_n_kv = fa.n_kv,
                        .processed_n_kv = fa.n_kv,
                        .requested_qkv_pairs = mul(fa.n_tokens, fa.n_kv),
                        .valid_qkv_pairs = mul(fa.n_tokens, fa.n_kv),
                        .processed_qkv_pairs = mul(fa.n_tokens, fa.n_kv),
                    },
                };
                const incoming = profiling.FlashStat{
                    .backend = outcome.backend,
                    .path = outcome.path,
                    .n_heads = sat16(fa.n_heads),
                    .n_head_kv = sat16(fa.n_head_kv),
                    .head_dim_q = sat16(fa.head_dim_q),
                    .head_dim_v = sat16(fa.head_dim_v),
                    .n_tokens = fa.n_tokens,
                };
                const s = self.flashBucket(incoming) orelse return;
                s.count +|= 1;
                s.kernel_runs +|= detail.kernel_runs;
                s.command_ns +|= command_ns;
                s.wrapper_ns +|= detail.wrapper_ns;
                s.prepare_ns +|= detail.prepare_ns;
                s.sync_to_ns +|= detail.sync_to_ns;
                s.setup_ns +|= detail.setup_ns;
                s.wait_ns +|= detail.wait_ns;
                s.sync_from_ns +|= detail.sync_from_ns;
                s.result_layout_ns +|= detail.result_layout_ns;
                s.requested_n_kv_sum +|= detail.requested_n_kv;
                s.valid_n_kv_sum +|= detail.valid_n_kv;
                s.processed_n_kv_sum +|= detail.processed_n_kv;
                s.requested_qkv_pairs +|= detail.requested_qkv_pairs;
                s.valid_qkv_pairs +|= detail.valid_qkv_pairs;
                s.processed_qkv_pairs +|= detail.processed_qkv_pairs;
                s.requested_n_kv_max = @max(s.requested_n_kv_max, detail.requested_n_kv);
                s.valid_n_kv_max = @max(s.valid_n_kv_max, detail.valid_n_kv);
                s.processed_n_kv_max = @max(s.processed_n_kv_max, detail.processed_n_kv);
                s.q_bytes +|= detail.q_bytes;
                s.k_bytes +|= detail.k_bytes;
                s.v_bytes +|= detail.v_bytes;
                s.mask_bytes +|= detail.mask_bytes;
                s.o_bytes +|= detail.o_bytes;
                s.cycles +|= detail.cycles;
                s.q_beats +|= detail.q_beats;
                s.k_beats +|= detail.k_beats;
                s.k_stall_cycles +|= detail.k_stall_cycles;
                s.v_beats +|= detail.v_beats;
                s.v_stall_cycles +|= detail.v_stall_cycles;
                s.o_beats +|= detail.o_beats;
                s.o_stall_cycles +|= detail.o_stall_cycles;
                if (detail.wrapper_ns > command_ns or detail.wrapper_ns < wrapperChildrenFlash(detail))
                    self.accounting_violations |= profiling.AccountingViolation.wrapper_segments;
            },
            else => {},
        }
    }

    fn recordMatmul(
        self: *Collector,
        rows: u32,
        cols: u32,
        k: u32,
        weight_fmt: wire.WeightFormat,
        logical_count: usize,
        logical_multiplicity: usize,
        command_ns: u64,
        outcome: CommandOutcome,
        detail: MatmulExecution,
    ) void {
        const incoming = profiling.MatmulStat{
            .backend = outcome.backend,
            .path = outcome.path,
            .fmt = @intCast(@intFromEnum(weight_fmt)),
            .rows = rows,
            .cols = cols,
            .k = k,
        };
        const stat = self.matmulBucket(incoming) orelse return;
        stat.count +|= @intCast(logical_count);
        stat.kernel_runs +|= detail.kernel_runs;
        stat.macs +|= mul(mul(mul(rows, cols), k), logical_multiplicity);
        stat.command_ns +|= command_ns;
        stat.wrapper_ns +|= detail.wrapper_ns;
        stat.quantize_pack_ns +|= detail.quantize_pack_ns;
        stat.sync_to_ns +|= detail.sync_to_ns;
        stat.setup_ns +|= detail.setup_ns;
        stat.wait_ns +|= detail.wait_ns;
        stat.sync_from_ns +|= detail.sync_from_ns;
        stat.result_layout_ns +|= detail.result_layout_ns;
        stat.cycles +|= detail.cycles;
        stat.w_stall_cycles +|= detail.w_stall_cycles;
        stat.a_stall_cycles +|= detail.a_stall_cycles;
        stat.r_stall_cycles +|= detail.r_stall_cycles;
        stat.w_beats +|= detail.w_beats;
        stat.a_beats +|= detail.a_beats;
        stat.r_beats +|= detail.r_beats;
        if (detail.wrapper_ns > command_ns or detail.wrapper_ns < wrapperChildrenMatmul(detail))
            self.accounting_violations |= profiling.AccountingViolation.wrapper_segments;
    }

    fn matmulBucket(self: *Collector, incoming: profiling.MatmulStat) ?*profiling.MatmulStat {
        for (self.matmul_stats[0..self.matmul_count]) |*stat| if (stat.sameKey(incoming)) return stat;
        if (self.matmul_count == self.matmul_stats.len) {
            self.matmul_overflow +|= 1;
            return null;
        }
        const stat = &self.matmul_stats[self.matmul_count];
        stat.* = incoming;
        self.matmul_count += 1;
        return stat;
    }

    fn flashBucket(self: *Collector, incoming: profiling.FlashStat) ?*profiling.FlashStat {
        for (self.flash_stats[0..self.flash_count]) |*stat| if (stat.sameKey(incoming)) return stat;
        if (self.flash_count == self.flash_stats.len) {
            self.flash_overflow +|= 1;
            return null;
        }
        const stat = &self.flash_stats[self.flash_count];
        stat.* = incoming;
        self.flash_count += 1;
        return stat;
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
        var checked = summary;
        checked.matmul_bucket_overflow = self.matmul_overflow;
        checked.flash_bucket_overflow = self.flash_overflow;
        checked.accounting_violations |= self.accounting_violations;
        var command_children: u64 = 0;
        for (used[0..n]) |aggregate| command_children +|= aggregate.total_ns;
        if (command_children > checked.execute_ns)
            checked.accounting_violations |= profiling.AccountingViolation.command_children;
        return profiling.encodeAlloc(allocator, .{
            .summary = checked,
            .aggregates = used[0..n],
            .matmul_stats = self.matmul_stats[0..self.matmul_count],
            .flash_stats = self.flash_stats[0..self.flash_count],
        });
    }
};

fn wrapperChildrenMatmul(s: MatmulExecution) u64 {
    return s.quantize_pack_ns +| s.sync_to_ns +| s.setup_ns +| s.wait_ns +| s.sync_from_ns +| s.result_layout_ns;
}

fn wrapperChildrenFlash(s: FlashExecution) u64 {
    return s.prepare_ns +| s.sync_to_ns +| s.setup_ns +| s.wait_ns +| s.sync_from_ns +| s.result_layout_ns;
}

pub fn commandTag(command: wire.Command) wire.OpTag {
    return switch (command) {
        .copy => .copy,
        .cpy_f32_to_f16 => .cpy_f32_to_f16,
        .matmul_q1a8 => .matmul_q1a8,
        .matmul_q1a8_group2 => .matmul_q1a8_group2,
        .ffn_section => .ffn_section,
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
        .matmul_q1a8_group2 => |op| op.acts.nbytes +|
            op.projections[0].weights.nbytes +| op.projections[0].dst.nbytes +|
            op.projections[1].weights.nbytes +| op.projections[1].dst.nbytes,
        .ffn_section => |op| op.residual.nbytes +| op.norm_weight.nbytes +|
            op.up_weights.nbytes +| op.gate_weights.nbytes +| op.down_weights.nbytes +|
            op.dst.nbytes,
        .rmsnorm => |op| op.input.nbytes +| (if (op.has_weight) op.weight.nbytes else 0) +| op.dst.nbytes,
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

test "profile byte estimate includes optional rmsnorm weight" {
    const input = wire.TensorRange{ .handle = 1, .offset = 0, .nbytes = 96 };
    const weight = wire.TensorRange{ .handle = 2, .offset = 0, .nbytes = 12 };
    const dst = wire.TensorRange{ .handle = 3, .offset = 0, .nbytes = 96 };
    const standalone = wire.Command{ .rmsnorm = .{ .input = input, .dst = dst, .rows = 3, .cols = 8, .eps = 1e-5 } };
    const fused = wire.Command{ .rmsnorm = .{
        .input = input,
        .weight = weight,
        .dst = dst,
        .rows = 3,
        .cols = 8,
        .eps = 1e-5,
        .has_weight = true,
    } };
    try std.testing.expectEqual(@as(u64, 192), commandBytes(standalone));
    try std.testing.expectEqual(@as(u64, 204), commandBytes(fused));
}

test "grouped matmul accounting counts one activation and two logical projections" {
    const acts = wire.TensorRange{ .handle = 1, .offset = 0, .nbytes = 1024 };
    const weights0 = wire.TensorRange{ .handle = 2, .offset = 0, .nbytes = 256 };
    const weights1 = wire.TensorRange{ .handle = 3, .offset = 0, .nbytes = 256 };
    const dst0 = wire.TensorRange{ .handle = 4, .offset = 0, .nbytes = 128 };
    const dst1 = wire.TensorRange{ .handle = 5, .offset = 0, .nbytes = 128 };
    const command = wire.Command{ .matmul_q1a8_group2 = .{
        .acts = acts,
        .projections = .{
            .{ .weights = weights0, .dst = dst0, .rows = 16, .weight_fmt = .w1a8 },
            .{ .weights = weights1, .dst = dst1, .rows = 16, .weight_fmt = .w1a8 },
        },
        .cols = 2,
        .k = 128,
    } };
    try std.testing.expectEqual(@as(u64, 1792), commandBytes(command));

    var collector: Collector = .{};
    collector.record(command, 100, .{
        .backend = .pl,
        .path = .direct,
        .detail = .{ .matmul = .{
            .path = .direct,
            .kernel_runs = 2,
            .wrapper_ns = 90,
            .quantize_pack_ns = 10,
            .wait_ns = 80,
        } },
    });
    const aggregate = collector.aggregates[@intFromEnum(wire.OpTag.matmul_q1a8_group2)];
    try std.testing.expectEqual(@as(u32, 1), aggregate.count);
    try std.testing.expectEqual(@as(u64, 100), aggregate.total_ns);
    try std.testing.expectEqual(@as(usize, 1), collector.matmul_count);
    try std.testing.expectEqual(@as(u32, 1), collector.matmul_stats[0].count);
    try std.testing.expectEqual(@as(u64, 100), collector.matmul_stats[0].command_ns);
    try std.testing.expectEqual(@as(u64, 2 * 16 * 2 * 128), collector.matmul_stats[0].macs);
    try std.testing.expectEqual(@as(u32, 2), collector.matmul_stats[0].kernel_runs);
}

test "ffn section records one aggregate and three logical matmuls" {
    const residual = wire.TensorRange{ .handle = 1, .offset = 0, .nbytes = 1024 };
    const norm_weight = wire.TensorRange{ .handle = 2, .offset = 0, .nbytes = 512 };
    const up_weights = wire.TensorRange{ .handle = 3, .offset = 0, .nbytes = 4096 };
    const gate_weights = wire.TensorRange{ .handle = 4, .offset = 0, .nbytes = 4096 };
    const down_weights = wire.TensorRange{ .handle = 5, .offset = 0, .nbytes = 4096 };
    const dst = wire.TensorRange{ .handle = 6, .offset = 0, .nbytes = 1024 };
    const command = wire.Command{ .ffn_section = .{
        .residual = residual,
        .norm_weight = norm_weight,
        .up_weights = up_weights,
        .gate_weights = gate_weights,
        .down_weights = down_weights,
        .dst = dst,
        .model_dim = 128,
        .ffn_dim = 256,
        .token_count = 2,
        .eps = 1e-6,
        .weight_fmt = .w1a8,
    } };
    try std.testing.expectEqual(@as(u64, 14_848), commandBytes(command));

    var collector: Collector = .{};
    collector.record(command, 100, .{
        .backend = .pl,
        .path = .direct,
        .detail = .{ .ffn = .{
            .gate_up = .{ .path = .direct, .kernel_runs = 2, .wrapper_ns = 35, .wait_ns = 30 },
            .down = .{ .path = .direct, .kernel_runs = 1, .wrapper_ns = 25, .wait_ns = 20 },
            .gate_up_command_ns = 40,
            .down_command_ns = 30,
            .norm_ns = 10,
            .swiglu_ns = 5,
            .residual_add_ns = 5,
        } },
    });
    const aggregate = collector.aggregates[@intFromEnum(wire.OpTag.ffn_section)];
    try std.testing.expectEqual(@as(u32, 1), aggregate.count);
    try std.testing.expectEqual(@as(u64, 100), aggregate.total_ns);
    try std.testing.expectEqual(@as(u64, 14_848), aggregate.bytes);
    try std.testing.expectEqual(@as(usize, 2), collector.matmul_count);
    try std.testing.expectEqual(@as(u32, 1), collector.matmul_stats[0].count);
    try std.testing.expectEqual(@as(u64, 2 * 256 * 2 * 128), collector.matmul_stats[0].macs);
    try std.testing.expectEqual(@as(u64, 40), collector.matmul_stats[0].command_ns);
    try std.testing.expectEqual(@as(u32, 1), collector.matmul_stats[1].count);
    try std.testing.expectEqual(@as(u64, 128 * 2 * 256), collector.matmul_stats[1].macs);
    try std.testing.expectEqual(@as(u64, 30), collector.matmul_stats[1].command_ns);
    try std.testing.expectEqual(@as(u32, 0), collector.accounting_violations);
}

fn testMatmulCommand(rows: u32) wire.Command {
    const range = wire.TensorRange{ .handle = 1, .offset = 0, .nbytes = 64 };
    return .{ .matmul_q1a8 = .{
        .weights = range,
        .acts = range,
        .dst = range,
        .rows = rows,
        .cols = 2,
        .k = 128,
        .weight_fmt = .w1a8,
    } };
}

test "collector separates backend and path and validates wrapper accounting" {
    var collector: Collector = .{};
    const command = testMatmulCommand(16);
    collector.record(command, 100, .{ .backend = .ps, .path = .software, .detail = .{ .matmul = .{ .path = .software } } });
    collector.record(command, 200, .{ .backend = .pl, .path = .staged, .detail = .{ .matmul = .{
        .path = .staged,
        .kernel_runs = 2,
        .wrapper_ns = 180,
        .quantize_pack_ns = 20,
        .sync_to_ns = 10,
        .setup_ns = 20,
        .wait_ns = 100,
        .sync_from_ns = 10,
        .result_layout_ns = 10,
        .cycles = 50,
        .w_beats = 40,
    } } });

    try std.testing.expectEqual(@as(usize, 2), collector.matmul_count);
    try std.testing.expectEqual(profiling.Backend.ps, collector.matmul_stats[0].backend);
    try std.testing.expectEqual(profiling.Backend.pl, collector.matmul_stats[1].backend);
    try std.testing.expectEqual(profiling.ExecutionPath.staged, collector.matmul_stats[1].path);
    try std.testing.expectEqual(@as(u32, 2), collector.matmul_stats[1].kernel_runs);
    try std.testing.expectEqual(@as(u64, 50), collector.matmul_stats[1].cycles);
    try std.testing.expectEqual(@as(u32, 0), collector.accounting_violations);

    collector.record(command, 50, .{ .backend = .pl, .path = .direct, .detail = .{ .matmul = .{
        .path = .direct,
        .wrapper_ns = 40,
        .wait_ns = 41,
    } } });
    try std.testing.expect(collector.accounting_violations & profiling.AccountingViolation.wrapper_segments != 0);
}

test "collector reports bounded bucket overflow" {
    var collector: Collector = .{};
    for (0..profiling.max_matmul_buckets + 3) |i| {
        collector.record(testMatmulCommand(@intCast(i + 1)), 1, .{});
    }
    try std.testing.expectEqual(@as(usize, profiling.max_matmul_buckets), collector.matmul_count);
    try std.testing.expectEqual(@as(u32, 3), collector.matmul_overflow);

    const payload = try collector.encode(std.testing.allocator, .{ .execute_ns = profiling.max_matmul_buckets + 3 });
    defer std.testing.allocator.free(payload);
    var decoded = try profiling.decodeAlloc(std.testing.allocator, payload);
    defer decoded.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(u32, 3), decoded.summary.matmul_bucket_overflow);
    try std.testing.expectEqual(@as(usize, profiling.max_matmul_buckets), decoded.matmul_stats.len);
}

test "collector preserves flash extents and multi-token query-KV work" {
    const range = wire.TensorRange{ .handle = 1, .offset = 0, .nbytes = 64 };
    const command = wire.Command{ .flash_attn_f32 = .{
        .q = range,
        .k = range,
        .v = range,
        .mask = range,
        .dst = range,
        .has_mask = true,
        .head_dim_q = 128,
        .head_dim_v = 64,
        .n_heads = 16,
        .n_head_kv = 8,
        .n_kv = 512,
        .n_tokens = 8,
        .scale = 1,
        .q_nb1 = 512,
        .q_nb2 = 512,
        .k_nb1 = 2048,
        .k_nb2 = 256,
        .v_nb1 = 1024,
        .v_nb2 = 128,
        .mask_nb1 = 1024,
        .dst_nb1 = 256,
        .dst_nb2 = 4096,
    } };
    var collector: Collector = .{};
    collector.record(command, 100, .{ .backend = .pl, .path = .direct, .detail = .{ .flash = .{
        .path = .direct,
        .wrapper_ns = 90,
        .wait_ns = 80,
        .requested_n_kv = 512,
        .valid_n_kv = 8,
        .processed_n_kv = 8,
        .requested_qkv_pairs = 4096,
        .valid_qkv_pairs = 36,
        .processed_qkv_pairs = 36,
        .cycles = 70,
    } } });
    const stat = collector.flash_stats[0];
    try std.testing.expectEqual(@as(u64, 512), stat.requested_n_kv_sum);
    try std.testing.expectEqual(@as(u32, 8), stat.n_tokens);
    try std.testing.expectEqual(@as(u64, 8), stat.valid_n_kv_sum);
    try std.testing.expectEqual(@as(u64, 8), stat.processed_n_kv_sum);
    try std.testing.expectEqual(@as(u64, 4096), stat.requested_qkv_pairs);
    try std.testing.expectEqual(@as(u64, 36), stat.valid_qkv_pairs);
    try std.testing.expectEqual(@as(u64, 36), stat.processed_qkv_pairs);
    try std.testing.expectEqual(profiling.Backend.pl, stat.backend);
}
