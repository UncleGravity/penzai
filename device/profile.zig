const std = @import("std");
const wire = @import("wire");
const profiling = @import("profiling");
const q1a8 = @import("q1a8");

/// Device-side per-graph profile collection. The runtime drives a Collector
/// around its execute loop; the Collector owns the aggregate/span bookkeeping
/// and produces the wire payload. Aggregates are always accrued (fixed size);
/// spans are recorded only for the `trace` tier, into a buffer sized to the
/// command count (no large stack array, no spans on the cheap path).
pub const Collector = struct {
    aggregates: [profiling.max_op_tag + 1]profiling.Aggregate,
    spans: []profiling.Span,
    span_count: u32 = 0,
    span_dropped: u32 = 0,

    pub fn init(
        allocator: std.mem.Allocator,
        tier: wire.ProfileTier,
        command_count: usize,
    ) std.mem.Allocator.Error!Collector {
        const cap: usize = if (tier == .trace) @min(command_count, profiling.max_spans) else 0;
        return .{
            .aggregates = [_]profiling.Aggregate{.{}} ** (profiling.max_op_tag + 1),
            .spans = try allocator.alloc(profiling.Span, cap),
        };
    }

    pub fn deinit(self: *Collector, allocator: std.mem.Allocator) void {
        allocator.free(self.spans);
        self.* = undefined;
    }

    /// Record one executed command. Timestamps are device-clock nanoseconds;
    /// span offsets are stored relative to the request start.
    pub fn record(
        self: *Collector,
        command: wire.Command,
        request_start_ns: u64,
        start_ns: u64,
        end_ns: u64,
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
        if (self.span_count < self.spans.len) {
            self.spans[self.span_count] = .{
                .tag = raw_tag,
                .start_ns = profiling.elapsed(request_start_ns, start_ns),
                .end_ns = profiling.elapsed(request_start_ns, end_ns),
                .bytes = bytes,
            };
            self.span_count += 1;
        } else if (self.spans.len > 0) {
            self.span_dropped += 1;
        }
    }

    /// Encode the collected report. `summary` carries the caller's timing and
    /// command_count; the span counts are filled in here.
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
        var filled = summary;
        filled.span_count = self.span_count;
        filled.span_dropped = self.span_dropped;
        return profiling.encodeAlloc(allocator, .{
            .summary = filled,
            .aggregates = used[0..n],
            .spans = self.spans[0..self.span_count],
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
    };
    return mul(rows, src_row_bytes +| index_bytes +| dst_row_bytes);
}

fn q1GetRowsSourceBytes(row_width: u32) u64 {
    if (row_width == 0) return 0;
    const blocks = (@as(u64, row_width) + q1a8.q1_block - 1) / q1a8.q1_block;
    const bytes_per_block = q1a8.beat_bytes * (1 + q1a8.q8_subblocks);
    return mul(blocks, bytes_per_block);
}

fn mul(a: u64, b: u64) u64 {
    return a *| b;
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
    const q1_source_bytes = 2 * q1a8.beat_bytes * (1 + q1a8.q8_subblocks);
    try std.testing.expectEqual(@as(u64, 6 * (q1_source_bytes + 4 + 256 * 4)), commandBytes(get_rows));
}
