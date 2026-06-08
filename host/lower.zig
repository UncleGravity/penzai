const std = @import("std");
const c = @import("c");
const q1a8 = @import("q1a8");
const wire = @import("wire");

pub const LowerError = error{
    OutOfMemory,
    UnsupportedOp,
    MissingBinding,
    InvalidShape,
};

pub const Binding = struct {
    range: wire.TensorRange,
    handle_nbytes: u64,
};

pub const Lookup = struct {
    ctx: *anyopaque,
    findFn: *const fn (*anyopaque, ?*const c.ggml_tensor) ?Binding,

    pub fn find(self: Lookup, tensor: ?*const c.ggml_tensor) ?Binding {
        return self.findFn(self.ctx, tensor);
    }
};

pub fn isMetadataOp(op: anytype) bool {
    return op == c.GGML_OP_NONE or
        op == c.GGML_OP_RESHAPE or
        op == c.GGML_OP_VIEW or
        op == c.GGML_OP_PERMUTE or
        op == c.GGML_OP_TRANSPOSE;
}

pub fn supportsOp(op: ?*const c.ggml_tensor) bool {
    const tensor = op orelse return false;
    if (isMetadataOp(tensor.*.op)) return true;
    return supportsMatmulQ1A8(tensor) or
        supportsBinaryF32(tensor) or
        supportsRmsNormF32(tensor) or
        supportsRopeF32(tensor) or
        supportsSwigluF32(tensor) or
        supportsSetRows(tensor) or
        supportsGetRows(tensor);
}

pub fn lowerGraph(
    allocator: std.mem.Allocator,
    graph: *c.ggml_cgraph,
    lookup: Lookup,
) LowerError![]wire.Command {
    var commands: std.ArrayList(wire.Command) = .empty;
    errdefer commands.deinit(allocator);

    const n = c.ggml_graph_n_nodes(graph);
    var i: c_int = 0;
    while (i < n) : (i += 1) {
        const node = c.ggml_graph_node(graph, i) orelse continue;
        if (isMetadataOp(node.*.op)) continue;
        if (tensorElements(node) == 0) continue;

        if (supportsMatmulQ1A8(node)) {
            try commands.append(allocator, try lowerMatmulQ1A8(node, lookup));
            continue;
        }
        if (supportsBinaryF32(node)) {
            try commands.append(allocator, try lowerBinaryF32(node, lookup));
            continue;
        }
        if (supportsRmsNormF32(node)) {
            try commands.append(allocator, try lowerRmsNormF32(node, lookup));
            continue;
        }
        if (supportsRopeF32(node)) {
            try commands.append(allocator, try lowerRopeF32(node, lookup));
            continue;
        }
        if (supportsSwigluF32(node)) {
            try commands.append(allocator, try lowerSwigluF32(node, lookup));
            continue;
        }
        if (supportsSetRows(node)) {
            try commands.append(allocator, try lowerSetRows(node, lookup));
            continue;
        }
        if (supportsGetRows(node)) {
            try commands.append(allocator, try lowerGetRows(node, lookup));
            continue;
        }

        return error.UnsupportedOp;
    }

    return try commands.toOwnedSlice(allocator);
}

fn supportsMatmulQ1A8(op: *const c.ggml_tensor) bool {
    if (op.*.op != c.GGML_OP_MUL_MAT or op.*.type != c.GGML_TYPE_F32) return false;
    const weights: *const c.ggml_tensor = op.*.src[0] orelse return false;
    const acts: *const c.ggml_tensor = op.*.src[1] orelse return false;
    if (weights.*.type != c.GGML_TYPE_Q1_0 or acts.*.type != c.GGML_TYPE_F32) return false;
    if (!c.ggml_is_contiguous(weights) or !c.ggml_is_contiguous(acts) or !c.ggml_is_contiguous(op)) return false;
    if (dim(weights, 0) <= 0 or dim(weights, 1) <= 0 or dim(acts, 1) <= 0) return false;
    if (dim(weights, 0) != dim(acts, 0)) return false;
    if (dim(op, 0) != dim(weights, 1) or dim(op, 1) != dim(acts, 1)) return false;
    if (@mod(@as(usize, @intCast(dim(weights, 0))), q1a8.q1_block) != 0) return false;
    for (2..4) |axis| {
        if (dimAt(weights, axis) != 1 or dimAt(acts, axis) != 1 or dimAt(op, axis) != 1) return false;
    }
    return true;
}

const BinaryF32Lowering = struct {
    lhs: *const c.ggml_tensor,
    rhs: *const c.ggml_tensor,
    mode: wire.BinaryF32Mode,
};

fn supportsBinaryF32(op: *const c.ggml_tensor) bool {
    return binaryF32Lowering(op) != null;
}

fn binaryF32Lowering(op: *const c.ggml_tensor) ?BinaryF32Lowering {
    if ((op.*.op != c.GGML_OP_ADD and op.*.op != c.GGML_OP_MUL) or !isContiguousF32(op)) return null;
    const src0: *const c.ggml_tensor = op.*.src[0] orelse return null;
    const src1: *const c.ggml_tensor = op.*.src[1] orelse return null;
    if (!isContiguousF32(src0) or !isContiguousF32(src1)) return null;

    if (sameShape(src0, op) and sameShape(src1, op)) {
        return .{ .lhs = src0, .rhs = src1, .mode = .same_shape };
    }
    if (sameShape(src0, op) and isRowBroadcastFor(src1, op)) {
        return .{ .lhs = src0, .rhs = src1, .mode = .rhs_row_broadcast };
    }
    if (sameShape(src1, op) and isRowBroadcastFor(src0, op)) {
        return .{ .lhs = src1, .rhs = src0, .mode = .rhs_row_broadcast };
    }
    return null;
}

fn supportsRmsNormF32(op: *const c.ggml_tensor) bool {
    if (op.*.op != c.GGML_OP_RMS_NORM or !isContiguousF32(op)) return false;
    const src: *const c.ggml_tensor = op.*.src[0] orelse return false;
    return isContiguousF32(src) and sameShape(src, op);
}

fn supportsRopeF32(op: *const c.ggml_tensor) bool {
    if (op.*.op != c.GGML_OP_ROPE or !isContiguousF32(op)) return false;
    const src: *const c.ggml_tensor = op.*.src[0] orelse return false;
    const positions: *const c.ggml_tensor = op.*.src[1] orelse return false;
    if (op.*.src[2] != null) return false;
    if (!isContiguousF32(src) or positions.*.type != c.GGML_TYPE_I32 or !c.ggml_is_contiguous(positions)) return false;
    if (src.*.nb[0] != @sizeOf(f32) or positions.*.nb[0] != @sizeOf(i32)) return false;
    if (dim(src, 3) != 1 or dim(op, 3) != 1) return false;
    if (dim(src, 0) != dim(op, 0) or dim(src, 1) != dim(op, 1) or dim(src, 2) != dim(op, 2)) return false;
    if (dim(src, 0) <= 0 or dim(src, 1) <= 0 or dim(src, 2) <= 0) return false;
    if (dim(positions, 0) < dim(src, 2)) return false;
    const n_dims = opParamI32(op, 1);
    if (n_dims <= 0 or @mod(n_dims, 2) != 0 or n_dims > dim(src, 0)) return false;
    if (ropeMode(opParamI32(op, 2)) == null) return false;
    const freq_base = opParamF32(op, 5);
    const freq_scale = opParamF32(op, 6);
    const ext_factor = opParamF32(op, 7);
    const attn_factor = opParamF32(op, 8);
    const beta_fast = opParamF32(op, 9);
    const beta_slow = opParamF32(op, 10);
    if (!finitePositive(freq_base) or !finitePositive(freq_scale)) return false;
    if (!std.math.isFinite(ext_factor) or !std.math.isFinite(attn_factor) or !std.math.isFinite(beta_fast) or !std.math.isFinite(beta_slow)) return false;
    if (ext_factor != 0 and (beta_fast <= 0 or beta_slow <= 0)) return false;
    return true;
}

fn supportsSwigluF32(op: *const c.ggml_tensor) bool {
    if (op.*.op != c.GGML_OP_GLU or c.ggml_get_glu_op(op) != c.GGML_GLU_OP_SWIGLU or !isContiguousF32(op)) return false;
    const src0: *const c.ggml_tensor = op.*.src[0] orelse return false;
    const src1: *const c.ggml_tensor = op.*.src[1] orelse return false;
    return isContiguousF32(src0) and isContiguousF32(src1) and sameShape(src0, op) and sameShape(src1, op);
}

fn supportsSetRows(op: *const c.ggml_tensor) bool {
    if (op.*.op != c.GGML_OP_SET_ROWS) return false;
    const src0: *const c.ggml_tensor = op.*.src[0] orelse return false;
    const src1: *const c.ggml_tensor = op.*.src[1] orelse return false;
    if (src0.*.type != c.GGML_TYPE_F32 or op.*.type != c.GGML_TYPE_F16) return false;
    if (src1.*.type != c.GGML_TYPE_I32 and src1.*.type != c.GGML_TYPE_I64) return false;
    if (src0.*.nb[0] != @sizeOf(f32) or op.*.nb[0] != @sizeOf(f16)) return false;
    if (src1.*.type == c.GGML_TYPE_I32 and src1.*.nb[0] != @sizeOf(i32)) return false;
    if (src1.*.type == c.GGML_TYPE_I64 and src1.*.nb[0] != @sizeOf(i64)) return false;
    if (dim(src0, 0) <= 0 or dim(src0, 1) <= 0 or dim(src0, 2) <= 0 or dim(src0, 3) <= 0) return false;
    if (dim(src1, 0) < dim(src0, 1) or dim(src1, 1) <= 0 or dim(src1, 2) <= 0) return false;
    if (dim(op, 0) != dim(src0, 0)) return false;
    if (@mod(dim(op, 2), dim(src1, 1)) != 0) return false;
    if (@mod(dim(op, 3), dim(src1, 2)) != 0) return false;
    return true;
}

fn supportsGetRows(op: *const c.ggml_tensor) bool {
    if (op.*.op != c.GGML_OP_GET_ROWS) return false;
    const src0: *const c.ggml_tensor = op.*.src[0] orelse return false;
    const src1: *const c.ggml_tensor = op.*.src[1] orelse return false;
    if ((src0.*.type != c.GGML_TYPE_F32 and src0.*.type != c.GGML_TYPE_Q1_0) or src1.*.type != c.GGML_TYPE_I32 or op.*.type != c.GGML_TYPE_F32) return false;
    if (dim(src0, 0) <= 0 or dim(src0, 1) <= 0 or dim(src0, 2) <= 0 or dim(src0, 3) <= 0) return false;
    if (dim(src1, 0) <= 0 or dim(src1, 1) <= 0 or dim(src1, 2) <= 0 or dim(src1, 3) != 1) return false;
    if (src0.*.type == c.GGML_TYPE_F32 and src0.*.nb[0] != @sizeOf(f32)) return false;
    if (src0.*.type == c.GGML_TYPE_Q1_0 and (src0.*.nb[0] != q1a8.q1_block_bytes or @mod(@as(usize, @intCast(dim(src0, 0))), q1a8.q1_block) != 0)) return false;
    if (src1.*.nb[0] != @sizeOf(i32) or op.*.nb[0] != @sizeOf(f32)) return false;
    if (src0.*.type == c.GGML_TYPE_Q1_0 and (dim(src0, 2) != 1 or dim(src0, 3) != 1)) return false;
    if (dim(src0, 2) != dim(src1, 1) or dim(src0, 3) != dim(src1, 2)) return false;
    if (dim(op, 0) != dim(src0, 0) or dim(op, 1) != dim(src1, 0) or dim(op, 2) != dim(src1, 1) or dim(op, 3) != dim(src1, 2)) return false;
    return true;
}

fn lowerMatmulQ1A8(node: *const c.ggml_tensor, lookup: Lookup) LowerError!wire.Command {
    const weights: *const c.ggml_tensor = node.*.src[0] orelse return error.InvalidShape;
    const acts: *const c.ggml_tensor = node.*.src[1] orelse return error.InvalidShape;

    const rows = try u32Dim(dim(weights, 1));
    const cols = try u32Dim(dim(acts, 1));
    const k = try u32Dim(dim(weights, 0));

    const weights_binding = lookup.find(weights) orelse return error.MissingBinding;
    const acts_binding = lookup.find(acts) orelse return error.MissingBinding;
    const dst_binding = lookup.find(node) orelse return error.MissingBinding;

    const weights_bytes = q1a8.packedWeightBytes(rows, k) catch return error.InvalidShape;
    const acts_bytes = q1a8.actsF32Bytes(cols, k) catch return error.InvalidShape;
    const dst_bytes = q1a8.outputF32Bytes(rows, cols) catch return error.InvalidShape;

    return .{ .matmul_q1a8 = .{
        .weights = range(weights_binding, weights_bytes),
        .acts = range(acts_binding, acts_bytes),
        .dst = range(dst_binding, dst_bytes),
        .rows = rows,
        .cols = cols,
        .k = k,
    } };
}

fn lowerBinaryF32(node: *const c.ggml_tensor, lookup: Lookup) LowerError!wire.Command {
    const lowering = binaryF32Lowering(node) orelse return error.InvalidShape;
    const rows = try u32Dim(dim(node, 0));
    const cols = try flattenedCols(node, rows);
    const total_bytes = try f32MatrixBytes(rows, cols);
    const rhs_bytes = switch (lowering.mode) {
        .same_shape => total_bytes,
        .rhs_row_broadcast => try f32MatrixBytes(rows, 1),
    };

    const lhs_binding = lookup.find(lowering.lhs) orelse return error.MissingBinding;
    const rhs_binding = lookup.find(lowering.rhs) orelse return error.MissingBinding;
    const dst_binding = lookup.find(node) orelse return error.MissingBinding;
    const command: wire.BinaryBroadcastF32 = .{
        .lhs = try backingRange(lhs_binding, total_bytes),
        .rhs = try backingRange(rhs_binding, rhs_bytes),
        .dst = try backingRange(dst_binding, total_bytes),
        .rows = rows,
        .cols = cols,
        .mode = lowering.mode,
    };

    return switch (node.*.op) {
        c.GGML_OP_ADD => .{ .add_f32 = command },
        c.GGML_OP_MUL => .{ .mul_f32 = command },
        else => error.InvalidShape,
    };
}

fn lowerRmsNormF32(node: *const c.ggml_tensor, lookup: Lookup) LowerError!wire.Command {
    const src: *const c.ggml_tensor = node.*.src[0] orelse return error.InvalidShape;
    const rows = try u32Dim(dim(node, 0));
    const cols = try flattenedCols(node, rows);
    const total_bytes = try f32MatrixBytes(rows, cols);

    const src_binding = lookup.find(src) orelse return error.MissingBinding;
    const dst_binding = lookup.find(node) orelse return error.MissingBinding;

    return .{ .rmsnorm = .{
        .input = try backingRange(src_binding, total_bytes),
        .dst = try backingRange(dst_binding, total_bytes),
        .rows = rows,
        .cols = cols,
        .eps = opParamF32(node, 0),
    } };
}

fn lowerRopeF32(node: *const c.ggml_tensor, lookup: Lookup) LowerError!wire.Command {
    const src: *const c.ggml_tensor = node.*.src[0] orelse return error.InvalidShape;
    const positions: *const c.ggml_tensor = node.*.src[1] orelse return error.InvalidShape;
    const head_dim = try u32Dim(dim(src, 0));
    const n_heads = try u32Dim(dim(src, 1));
    const n_tokens = try u32Dim(dim(src, 2));
    const n_dims = try u32Dim(opParamI32(node, 1));
    const mode = ropeMode(opParamI32(node, 2)) orelse return error.InvalidShape;
    const n_ctx_orig_i32 = opParamI32(node, 4);
    if (n_ctx_orig_i32 < 0) return error.InvalidShape;

    const total_elements = try checkedMul(
        try checkedMul(@as(usize, @intCast(head_dim)), @as(usize, @intCast(n_heads))),
        @as(usize, @intCast(n_tokens)),
    );
    const total_bytes = try checkedMul(total_elements, @sizeOf(f32));
    const positions_bytes = try checkedMul(@as(usize, @intCast(n_tokens)), @sizeOf(i32));

    const src_binding = lookup.find(src) orelse return error.MissingBinding;
    const positions_binding = lookup.find(positions) orelse return error.MissingBinding;
    const dst_binding = lookup.find(node) orelse return error.MissingBinding;

    return .{ .rope = .{
        .input = try backingRange(src_binding, total_bytes),
        .positions = try backingRange(positions_binding, positions_bytes),
        .dst = try backingRange(dst_binding, total_bytes),
        .head_dim = head_dim,
        .n_heads = n_heads,
        .n_tokens = n_tokens,
        .n_dims = n_dims,
        .mode = mode,
        .n_ctx_orig = @intCast(n_ctx_orig_i32),
        .freq_base = opParamF32(node, 5),
        .freq_scale = opParamF32(node, 6),
        .ext_factor = opParamF32(node, 7),
        .attn_factor = opParamF32(node, 8),
        .beta_fast = opParamF32(node, 9),
        .beta_slow = opParamF32(node, 10),
    } };
}

fn lowerSwigluF32(node: *const c.ggml_tensor, lookup: Lookup) LowerError!wire.Command {
    const src0: *const c.ggml_tensor = node.*.src[0] orelse return error.InvalidShape;
    const src1: *const c.ggml_tensor = node.*.src[1] orelse return error.InvalidShape;
    const rows = try u32Dim(dim(node, 0));
    const cols = try flattenedCols(node, rows);
    const total_bytes = try f32MatrixBytes(rows, cols);

    const src0_binding = lookup.find(src0) orelse return error.MissingBinding;
    const src1_binding = lookup.find(src1) orelse return error.MissingBinding;
    const dst_binding = lookup.find(node) orelse return error.MissingBinding;

    return .{ .swiglu = .{
        .lhs = try backingRange(src0_binding, total_bytes),
        .rhs = try backingRange(src1_binding, total_bytes),
        .dst = try backingRange(dst_binding, total_bytes),
    } };
}

fn lowerGetRows(node: *const c.ggml_tensor, lookup: Lookup) LowerError!wire.Command {
    const src0: *const c.ggml_tensor = node.*.src[0] orelse return error.InvalidShape;
    const src1: *const c.ggml_tensor = node.*.src[1] orelse return error.InvalidShape;

    const src0_binding = lookup.find(src0) orelse return error.MissingBinding;
    const src1_binding = lookup.find(src1) orelse return error.MissingBinding;
    const dst_binding = lookup.find(node) orelse return error.MissingBinding;

    const row_width = try u32Dim(dim(src0, 0));
    const src_rows = try u32Dim(dim(src0, 1));
    const ne10 = try u32Dim(dim(src1, 0));
    const ne11 = try u32Dim(dim(src1, 1));
    const ne12 = try u32Dim(dim(src1, 2));
    const src_type: wire.GetRowsSrcType = if (src0.*.type == c.GGML_TYPE_Q1_0) .q1_0 else .f32;

    const src_span = switch (src_type) {
        .f32 => try stridedSpan(
            try checkedMul(@as(usize, @intCast(row_width)), @sizeOf(f32)),
            src_rows,
            try u32Dim(dim(src0, 2)),
            try u32Dim(dim(src0, 3)),
            src0.*.nb[1],
            src0.*.nb[2],
            src0.*.nb[3],
        ),
        .q1_0 => q1a8.packedWeightBytes(src_rows, row_width) catch return error.InvalidShape,
    };
    const indices_span = try stridedSpan(
        try checkedMul(@as(usize, @intCast(ne10)), @sizeOf(i32)),
        1,
        ne11,
        ne12,
        @sizeOf(i32),
        src1.*.nb[1],
        src1.*.nb[2],
    );
    const dst_span = try stridedSpan(
        try checkedMul(@as(usize, @intCast(row_width)), @sizeOf(f32)),
        ne10,
        ne11,
        ne12,
        node.*.nb[1],
        node.*.nb[2],
        node.*.nb[3],
    );

    return .{ .get_rows = .{
        .src = try backingRange(src0_binding, src_span),
        .indices = try backingRange(src1_binding, indices_span),
        .dst = try backingRange(dst_binding, dst_span),
        .src_type = src_type,
        .row_width = row_width,
        .src_rows = src_rows,
        .ne10 = ne10,
        .ne11 = ne11,
        .ne12 = ne12,
        .src_nb1 = @intCast(src0.*.nb[1]),
        .src_nb2 = @intCast(src0.*.nb[2]),
        .src_nb3 = @intCast(src0.*.nb[3]),
        .indices_nb1 = @intCast(src1.*.nb[1]),
        .indices_nb2 = @intCast(src1.*.nb[2]),
        .dst_nb1 = @intCast(node.*.nb[1]),
        .dst_nb2 = @intCast(node.*.nb[2]),
        .dst_nb3 = @intCast(node.*.nb[3]),
    } };
}

fn lowerSetRows(node: *const c.ggml_tensor, lookup: Lookup) LowerError!wire.Command {
    const src0: *const c.ggml_tensor = node.*.src[0] orelse return error.InvalidShape;
    const src1: *const c.ggml_tensor = node.*.src[1] orelse return error.InvalidShape;

    const src0_binding = lookup.find(src0) orelse return error.MissingBinding;
    const src1_binding = lookup.find(src1) orelse return error.MissingBinding;
    const dst_binding = lookup.find(node) orelse return error.MissingBinding;

    const head_dim = try u32Dim(dim(src0, 0));
    const ne01 = try u32Dim(dim(src0, 1));
    const ne02 = try u32Dim(dim(src0, 2));
    const ne03 = try u32Dim(dim(src0, 3));
    const ne11 = try u32Dim(dim(src1, 1));
    const ne12 = try u32Dim(dim(src1, 2));
    const index_type = if (src1.*.type == c.GGML_TYPE_I32) wire.IndexType.i32 else wire.IndexType.i64;
    const index_size: usize = if (index_type == .i32) @sizeOf(i32) else @sizeOf(i64);

    const src_span = try stridedSpan(
        try checkedMul(@as(usize, @intCast(head_dim)), @sizeOf(f32)),
        ne01,
        ne02,
        ne03,
        src0.*.nb[1],
        src0.*.nb[2],
        src0.*.nb[3],
    );
    const indices_span = try stridedSpan(
        try checkedMul(@as(usize, @intCast(ne01)), index_size),
        1,
        ne11,
        ne12,
        index_size,
        src1.*.nb[1],
        src1.*.nb[2],
    );

    return .{ .set_rows = .{
        .src = try backingRange(src0_binding, src_span),
        .indices = try backingRange(src1_binding, indices_span),
        .dst = try backingRemaining(dst_binding),
        .index_type = index_type,
        .head_dim = head_dim,
        .ne01 = ne01,
        .ne02 = ne02,
        .ne03 = ne03,
        .ne11 = ne11,
        .ne12 = ne12,
        .src_nb1 = @intCast(src0.*.nb[1]),
        .src_nb2 = @intCast(src0.*.nb[2]),
        .src_nb3 = @intCast(src0.*.nb[3]),
        .indices_nb1 = @intCast(src1.*.nb[1]),
        .indices_nb2 = @intCast(src1.*.nb[2]),
        .dst_nb1 = @intCast(node.*.nb[1]),
        .dst_nb2 = @intCast(node.*.nb[2]),
        .dst_nb3 = @intCast(node.*.nb[3]),
    } };
}

fn range(binding: Binding, nbytes: usize) wire.TensorRange {
    return .{
        .handle = binding.range.handle,
        .offset = binding.range.offset,
        .nbytes = @intCast(nbytes),
    };
}

fn backingRange(binding: Binding, nbytes: usize) LowerError!wire.TensorRange {
    if (binding.range.offset > binding.handle_nbytes) return error.InvalidShape;
    if (nbytes > binding.handle_nbytes - binding.range.offset) return error.InvalidShape;
    return .{
        .handle = binding.range.handle,
        .offset = binding.range.offset,
        .nbytes = @intCast(nbytes),
    };
}

fn backingRemaining(binding: Binding) LowerError!wire.TensorRange {
    if (binding.range.offset > binding.handle_nbytes) return error.InvalidShape;
    return .{
        .handle = binding.range.handle,
        .offset = binding.range.offset,
        .nbytes = binding.handle_nbytes - binding.range.offset,
    };
}

fn stridedSpan(row_bytes: usize, ne1: u32, ne2: u32, ne3: u32, nb1: usize, nb2: usize, nb3: usize) LowerError!usize {
    if (row_bytes == 0 or ne1 == 0 or ne2 == 0 or ne3 == 0) return error.InvalidShape;
    var span = row_bytes;
    span = try checkedAdd(span, try checkedMul(@as(usize, @intCast(ne1 - 1)), nb1));
    span = try checkedAdd(span, try checkedMul(@as(usize, @intCast(ne2 - 1)), nb2));
    span = try checkedAdd(span, try checkedMul(@as(usize, @intCast(ne3 - 1)), nb3));
    return span;
}

fn tensorElements(tensor: *const c.ggml_tensor) usize {
    const raw = c.ggml_nelements(@constCast(tensor));
    if (raw <= 0) return 0;
    return @intCast(raw);
}

fn isContiguousF32(tensor: *const c.ggml_tensor) bool {
    return tensor.*.type == c.GGML_TYPE_F32 and dim(tensor, 0) > 0 and tensorElements(tensor) > 0 and c.ggml_is_contiguous(tensor);
}

fn sameShape(lhs: *const c.ggml_tensor, rhs: *const c.ggml_tensor) bool {
    for (0..4) |axis| {
        if (dimAt(lhs, axis) != dimAt(rhs, axis)) return false;
    }
    return true;
}

fn isRowBroadcastFor(src: *const c.ggml_tensor, dst: *const c.ggml_tensor) bool {
    if (!isContiguousF32(src) or !isContiguousF32(dst) or dim(src, 0) != dim(dst, 0)) return false;
    for (1..4) |axis| {
        if (dimAt(src, axis) != 1) return false;
    }
    return true;
}

fn flattenedCols(tensor: *const c.ggml_tensor, rows: u32) LowerError!u32 {
    const elements = tensorElements(tensor);
    if (rows == 0 or elements == 0 or @mod(elements, @as(usize, @intCast(rows))) != 0) return error.InvalidShape;
    const cols = elements / @as(usize, @intCast(rows));
    if (cols == 0 or cols > std.math.maxInt(u32)) return error.InvalidShape;
    return @intCast(cols);
}

fn f32MatrixBytes(rows: u32, cols: u32) LowerError!usize {
    const elements = try checkedMul(@as(usize, @intCast(rows)), @as(usize, @intCast(cols)));
    return try checkedMul(elements, @sizeOf(f32));
}

fn opParamF32(tensor: *const c.ggml_tensor, index: usize) f32 {
    const bytes = std.mem.asBytes(&tensor.*.op_params);
    const offset = index * @sizeOf(f32);
    return @bitCast(std.mem.readInt(u32, bytes[offset..][0..4], .little));
}

fn opParamI32(tensor: *const c.ggml_tensor, index: usize) i32 {
    const bytes = std.mem.asBytes(&tensor.*.op_params);
    const offset = index * @sizeOf(i32);
    return std.mem.readInt(i32, bytes[offset..][0..4], .little);
}

fn ropeMode(ggml_mode: i32) ?wire.RopeMode {
    return switch (ggml_mode) {
        0 => .normal,
        2 => .neox,
        else => null,
    };
}

fn finitePositive(value: f32) bool {
    return value > 0 and std.math.isFinite(value);
}

fn dim(tensor: anytype, index: comptime_int) i64 {
    return tensor.*.ne[index];
}

fn dimAt(tensor: anytype, index: usize) i64 {
    return tensor.*.ne[index];
}

fn u32Dim(value: anytype) LowerError!u32 {
    if (value <= 0 or value > std.math.maxInt(u32)) return error.InvalidShape;
    return @intCast(value);
}

fn checkedAdd(a: usize, b: usize) LowerError!usize {
    return std.math.add(usize, a, b) catch return error.InvalidShape;
}

fn checkedMul(a: usize, b: usize) LowerError!usize {
    return std.math.mul(usize, a, b) catch return error.InvalidShape;
}
