//! ggml graph lowering — turns each supported ggml node into a fixed-width
//! `wire.Command` for the device. Two halves: `supportsOp` + the per-op
//! `supportsX` shape predicates (the gate ggml queries), and the `lowerX`
//! emitters that pack operand byte-ranges + dims. The dense shape predicates are
//! domain knowledge, not clutter — tablify the dispatch but keep them intact
//! (plan-host-rebuild.md §4). llama-free.
const std = @import("std");
const c = @import("c_ggml");
const shared = @import("shared");

const layout = shared.layout;
const wire = shared.wire;

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

const max_flash_head_dim = 256;

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
    return supportsCopy(tensor) or
        supportsCopyF32ToF16(tensor) or
        supportsMatmulQ1A8(tensor) or
        supportsBinaryF32(tensor) or
        supportsRmsNormF32(tensor) or
        supportsRopeF32(tensor) or
        supportsFlashAttnF32(tensor) or
        supportsSwigluF32(tensor) or
        supportsSetRows(tensor) or
        supportsGetRows(tensor) or
        supportsArgmax(tensor) or
        supportsPad(tensor);
}

// ===========================  Graph lowering — the dispatch loop  ===========================

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

        if (supportsCopy(node)) {
            try commands.append(allocator, try lowerCopy(node, lookup));
            continue;
        }
        if (supportsCopyF32ToF16(node)) {
            try commands.append(allocator, try lowerCopyF32ToF16(node, lookup));
            continue;
        }
        if (supportsMatmulQ1A8(node)) {
            if (tryLowerMatmulQ1A8Group2(graph, i, lookup)) |group| {
                try commands.append(allocator, group.command);
                i = group.last_index;
                continue;
            }
            try commands.append(allocator, try lowerMatmulQ1A8(node, lookup));
            continue;
        }
        if (supportsBinaryF32(node)) {
            try commands.append(allocator, try lowerBinaryF32(node, lookup));
            continue;
        }
        if (supportsRmsNormF32(node)) {
            if (tryLowerRmsNormMulF32(graph, i, lookup)) |command| {
                try commands.append(allocator, command);
                i += 1;
                continue;
            }
            try commands.append(allocator, try lowerRmsNormF32(node, lookup));
            continue;
        }
        if (supportsRopeF32(node)) {
            try commands.append(allocator, try lowerRopeF32(node, lookup));
            continue;
        }
        if (supportsFlashAttnF32(node)) {
            try commands.append(allocator, try lowerFlashAttnF32(node, lookup));
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
        if (supportsArgmax(node)) {
            try commands.append(allocator, try lowerArgmax(node, lookup));
            continue;
        }
        if (supportsPad(node)) {
            try commands.append(allocator, try lowerPad(node, lookup));
            continue;
        }

        return error.UnsupportedOp;
    }

    return try commands.toOwnedSlice(allocator);
}

// ===========================  supports_op predicates — per-op shape gates  ===========================
// The domain knowledge: exactly which shapes / dtypes / op_params each op
// accepts. Ugly but load-bearing — tablify the dispatch above, never these (§4).

fn supportsCopy(op: *const c.ggml_tensor) bool {
    if (op.*.op != c.GGML_OP_CPY) return false;
    const src: *const c.ggml_tensor = op.*.src[0] orelse return false;
    if (src.*.type != op.*.type) return false;
    if (!c.ggml_is_contiguous(src) or !c.ggml_is_contiguous(op)) return false;
    return tensorNbytes(src) == tensorNbytes(op);
}

fn supportsCopyF32ToF16(op: *const c.ggml_tensor) bool {
    if (op.*.op != c.GGML_OP_CPY) return false;
    const src: *const c.ggml_tensor = op.*.src[0] orelse return false;
    if (src.*.type != c.GGML_TYPE_F32 or op.*.type != c.GGML_TYPE_F16) return false;
    if (!c.ggml_is_contiguous(src) or !c.ggml_is_contiguous(op)) return false;
    return tensorElements(src) == tensorElements(op);
}

fn supportsMatmulQ1A8(op: *const c.ggml_tensor) bool {
    if (op.*.op != c.GGML_OP_MUL_MAT or op.*.type != c.GGML_TYPE_F32) return false;
    const weights: *const c.ggml_tensor = op.*.src[0] orelse return false;
    const acts: *const c.ggml_tensor = op.*.src[1] orelse return false;
    if ((weights.*.type != c.GGML_TYPE_Q1_0 and weights.*.type != c.GGML_TYPE_Q2_0) or acts.*.type != c.GGML_TYPE_F32) return false;
    if (!c.ggml_is_contiguous(weights) or !c.ggml_is_contiguous(acts) or !c.ggml_is_contiguous(op)) return false;
    if (dim(weights, 0) <= 0 or dim(weights, 1) <= 0 or dim(acts, 1) <= 0) return false;
    if (dim(weights, 0) != dim(acts, 0)) return false;
    if (dim(op, 0) != dim(weights, 1) or dim(op, 1) != dim(acts, 1)) return false;
    if (@mod(@as(usize, @intCast(dim(weights, 0))), layout.q1_block) != 0) return false;
    // K past the kernel's acts-BRAM capacity would alias (see layout.max_q1_blocks),
    // so decline here and let it fall back to the CPU backend.
    if (@divTrunc(@as(usize, @intCast(dim(weights, 0))), layout.q1_block) > layout.max_q1_blocks) return false;
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

const RmsNormMulF32Pair = struct {
    rmsnorm: *const c.ggml_tensor,
    input: *const c.ggml_tensor,
    weight: *const c.ggml_tensor,
    mul: *const c.ggml_tensor,
};

fn rmsNormMulF32Pair(graph: *c.ggml_cgraph, rms_index: c_int) ?RmsNormMulF32Pair {
    const n = c.ggml_graph_n_nodes(graph);
    if (rms_index < 0 or rms_index + 1 >= n) return null;

    const rmsnorm: *const c.ggml_tensor = c.ggml_graph_node(graph, rms_index) orelse return null;
    const mul: *const c.ggml_tensor = c.ggml_graph_node(graph, rms_index + 1) orelse return null;
    if (!supportsRmsNormF32(rmsnorm) or mul.*.op != c.GGML_OP_MUL) return null;
    if ((rmsnorm.*.flags & c.GGML_TENSOR_FLAG_COMPUTE) == 0 or (mul.*.flags & c.GGML_TENSOR_FLAG_COMPUTE) == 0) return null;
    if ((rmsnorm.*.flags & c.GGML_TENSOR_FLAG_OUTPUT) != 0 or rmsnorm.*.view_src != null) return null;

    const input: *const c.ggml_tensor = rmsnorm.*.src[0] orelse return null;
    const weight: *const c.ggml_tensor = if (mul.*.src[0] == rmsnorm)
        mul.*.src[1] orelse return null
    else if (mul.*.src[1] == rmsnorm)
        mul.*.src[0] orelse return null
    else
        return null;

    if (!isContiguousF32(mul) or !sameShape(rmsnorm, mul)) return null;
    if (!isRowBroadcastFor(weight, mul)) return null;
    const eps = opParamF32(rmsnorm, 0);
    if (eps < 0 or !std.math.isFinite(eps)) return null;

    var uses: usize = 0;
    var graph_index: c_int = 0;
    while (graph_index < n) : (graph_index += 1) {
        const node = c.ggml_graph_node(graph, graph_index) orelse continue;
        if (node.*.view_src == rmsnorm) return null;
        for (node.*.src) |src| {
            if (src == rmsnorm) uses += 1;
        }
    }
    if (uses != 1) return null;

    return .{ .rmsnorm = rmsnorm, .input = input, .weight = weight, .mul = mul };
}

fn tryLowerRmsNormMulF32(graph: *c.ggml_cgraph, rms_index: c_int, lookup: Lookup) ?wire.Command {
    const pair = rmsNormMulF32Pair(graph, rms_index) orelse return null;
    const rows = u32Dim(dim(pair.mul, 0)) catch return null;
    const cols = flattenedCols(pair.mul, rows) catch return null;
    const total_bytes = f32MatrixBytes(rows, cols) catch return null;
    const weight_bytes = f32MatrixBytes(rows, 1) catch return null;

    const input_binding = lookup.find(pair.input) orelse return null;
    _ = lookup.find(pair.rmsnorm) orelse return null;
    const weight_binding = lookup.find(pair.weight) orelse return null;
    const dst_binding = lookup.find(pair.mul) orelse return null;

    return .{ .rmsnorm = .{
        .input = backingRange(input_binding, total_bytes) catch return null,
        .weight = backingRange(weight_binding, weight_bytes) catch return null,
        .dst = backingRange(dst_binding, total_bytes) catch return null,
        .rows = rows,
        .cols = cols,
        .eps = opParamF32(pair.rmsnorm, 0),
        .has_weight = true,
    } };
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

fn supportsFlashAttnF32(op: *const c.ggml_tensor) bool {
    if (op.*.op != c.GGML_OP_FLASH_ATTN_EXT or op.*.type != c.GGML_TYPE_F32) return false;
    const q: *const c.ggml_tensor = op.*.src[0] orelse return false;
    const k: *const c.ggml_tensor = op.*.src[1] orelse return false;
    const v: *const c.ggml_tensor = op.*.src[2] orelse return false;
    const mask: ?*const c.ggml_tensor = op.*.src[3];
    if (op.*.src[4] != null) return false;

    if (q.*.type != c.GGML_TYPE_F32 or k.*.type != c.GGML_TYPE_F16 or v.*.type != c.GGML_TYPE_F16) return false;
    if (mask) |m| {
        if (m.*.type != c.GGML_TYPE_F16 or m.*.nb[0] != @sizeOf(f16)) return false;
        if (dim(m, 0) < dim(k, 1) or dim(m, 1) < dim(q, 1)) return false;
    }

    if (op.*.nb[0] != @sizeOf(f32) or q.*.nb[0] != @sizeOf(f32) or k.*.nb[0] != @sizeOf(f16) or v.*.nb[0] != @sizeOf(f16)) return false;
    if (dim(q, 3) != 1 or dim(k, 3) != 1 or dim(v, 3) != 1 or dim(op, 3) != 1) return false;
    if (dim(q, 0) != dim(k, 0)) return false;
    if (dim(q, 0) <= 0 or dim(v, 0) <= 0) return false;
    if (dim(q, 0) > max_flash_head_dim or dim(v, 0) > max_flash_head_dim) return false;
    if (dim(k, 1) <= 0 or dim(k, 1) != dim(v, 1)) return false;
    if (dim(k, 2) <= 0 or dim(k, 2) != dim(v, 2)) return false;
    if (dim(q, 1) <= 0 or dim(q, 2) <= 0) return false;
    if (@mod(dim(q, 2), dim(k, 2)) != 0) return false;
    if (dim(op, 0) != dim(v, 0) or dim(op, 1) != dim(q, 2) or dim(op, 2) != dim(q, 1)) return false;
    if (!std.math.isFinite(opParamF32(op, 0))) return false;
    if (opParamF32(op, 1) != 0 or opParamF32(op, 2) != 0) return false;
    if (dim(k, 1) > 8192) return false;
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
    if ((src0.*.type != c.GGML_TYPE_F32 and src0.*.type != c.GGML_TYPE_Q1_0 and src0.*.type != c.GGML_TYPE_Q2_0) or src1.*.type != c.GGML_TYPE_I32 or op.*.type != c.GGML_TYPE_F32) return false;
    if (dim(src0, 0) <= 0 or dim(src0, 1) <= 0 or dim(src0, 2) <= 0 or dim(src0, 3) <= 0) return false;
    if (dim(src1, 0) <= 0 or dim(src1, 1) <= 0 or dim(src1, 2) <= 0 or dim(src1, 3) != 1) return false;
    if (src0.*.type == c.GGML_TYPE_F32 and src0.*.nb[0] != @sizeOf(f32)) return false;
    if (src0.*.type == c.GGML_TYPE_Q1_0 and (src0.*.nb[0] != layout.q1_block_bytes or @mod(@as(usize, @intCast(dim(src0, 0))), layout.q1_block) != 0)) return false;
    if (src0.*.type == c.GGML_TYPE_Q2_0 and (src0.*.nb[0] != layout.q2_source_block_bytes or @mod(@as(usize, @intCast(dim(src0, 0))), layout.q1_block) != 0)) return false;
    if (src1.*.nb[0] != @sizeOf(i32) or op.*.nb[0] != @sizeOf(f32)) return false;
    if ((src0.*.type == c.GGML_TYPE_Q1_0 or src0.*.type == c.GGML_TYPE_Q2_0) and (dim(src0, 2) != 1 or dim(src0, 3) != 1)) return false;
    if (dim(src0, 2) != dim(src1, 1) or dim(src0, 3) != dim(src1, 2)) return false;
    if (dim(op, 0) != dim(src0, 0) or dim(op, 1) != dim(src1, 0) or dim(op, 2) != dim(src1, 1) or dim(op, 3) != dim(src1, 2)) return false;
    return true;
}

fn supportsArgmax(op: *const c.ggml_tensor) bool {
    // ggml_argmax: matrix f32 src -> 1-D i32 result of length src.ne[1].
    if (op.*.op != c.GGML_OP_ARGMAX or op.*.type != c.GGML_TYPE_I32) return false;
    const src: *const c.ggml_tensor = op.*.src[0] orelse return false;
    if (src.*.type != c.GGML_TYPE_F32 or !c.ggml_is_contiguous(src)) return false;
    if (src.*.nb[0] != @sizeOf(f32) or op.*.nb[0] != @sizeOf(i32)) return false;
    if (dim(src, 0) <= 0 or dim(src, 1) <= 0 or dim(src, 2) != 1 or dim(src, 3) != 1) return false;
    if (dim(op, 0) != dim(src, 1) or dim(op, 1) != 1 or dim(op, 2) != 1 or dim(op, 3) != 1) return false;
    return true;
}

fn supportsPad(op: *const c.ggml_tensor) bool {
    // ggml_pad: zero-pad. Only the front-aligned, right-side pad that
    // padZeroTailBytes implements — every left pad zero, not circular, dim0
    // unchanged, dims 2/3 singleton — so dst is src bytes followed by zeros.
    if (op.*.op != c.GGML_OP_PAD or !isContiguousF32(op)) return false;
    const src: *const c.ggml_tensor = op.*.src[0] orelse return false;
    if (!isContiguousF32(src)) return false;
    if (opParamI32(op, 0) != 0 or opParamI32(op, 2) != 0 or opParamI32(op, 4) != 0 or opParamI32(op, 6) != 0) return false;
    if (opParamI32(op, 8) != 0) return false; // circular flag
    if (dim(op, 0) != dim(src, 0)) return false;
    if (dim(src, 2) != 1 or dim(src, 3) != 1 or dim(op, 2) != 1 or dim(op, 3) != 1) return false;
    if (dim(op, 1) < dim(src, 1)) return false;
    return true;
}

// ===========================  Lowering — ggml node → wire.Command  ===========================

fn lowerCopy(node: *const c.ggml_tensor, lookup: Lookup) LowerError!wire.Command {
    const src: *const c.ggml_tensor = node.*.src[0] orelse return error.InvalidShape;
    const nbytes = tensorNbytes(node);
    const src_binding = lookup.find(src) orelse return error.MissingBinding;
    const dst_binding = lookup.find(node) orelse return error.MissingBinding;
    return .{ .copy = .{
        .src = try backingRange(src_binding, nbytes),
        .dst = try backingRange(dst_binding, nbytes),
    } };
}

fn lowerCopyF32ToF16(node: *const c.ggml_tensor, lookup: Lookup) LowerError!wire.Command {
    const src: *const c.ggml_tensor = node.*.src[0] orelse return error.InvalidShape;
    const src_binding = lookup.find(src) orelse return error.MissingBinding;
    const dst_binding = lookup.find(node) orelse return error.MissingBinding;
    return .{ .cpy_f32_to_f16 = .{
        .src = try backingRange(src_binding, tensorNbytes(src)),
        .dst = try backingRange(dst_binding, tensorNbytes(node)),
    } };
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

    const weight_fmt: wire.WeightFormat = if (weights.*.type == c.GGML_TYPE_Q2_0) .w158a8 else .w1a8;
    const weights_bytes = switch (weight_fmt) {
        .w1a8 => layout.packedWeightBytes(rows, k),
        .w158a8 => layout.packedTernaryWeightBytes(rows, k),
    } catch return error.InvalidShape;
    const acts_bytes = layout.actsF32Bytes(cols, k) catch return error.InvalidShape;
    const dst_bytes = layout.outputF32Bytes(rows, cols) catch return error.InvalidShape;

    return .{ .matmul_q1a8 = .{
        .weights = range(weights_binding, weights_bytes),
        .acts = range(acts_binding, acts_bytes),
        .dst = range(dst_binding, dst_bytes),
        .rows = rows,
        .cols = cols,
        .k = k,
        .weight_fmt = weight_fmt,
    } };
}

const LoweredMatmulGroup2 = struct {
    command: wire.Command,
    last_index: c_int,
};

fn tryLowerMatmulQ1A8Group2(graph: *c.ggml_cgraph, first_index: c_int, lookup: Lookup) ?LoweredMatmulGroup2 {
    const first: *const c.ggml_tensor = c.ggml_graph_node(graph, first_index) orelse return null;
    if (!safeGroupedMatmulNode(first)) return null;

    const n = c.ggml_graph_n_nodes(graph);
    var second_index = first_index + 1;
    var second: ?*const c.ggml_tensor = null;
    while (second_index < n) : (second_index += 1) {
        const candidate = c.ggml_graph_node(graph, second_index) orelse continue;
        if (isMetadataOp(candidate.*.op) or tensorElements(candidate) == 0) continue;
        second = candidate;
        break;
    }
    const second_node = second orelse return null;
    if (!safeGroupedMatmulNode(second_node)) return null;

    const first_acts: *const c.ggml_tensor = first.*.src[1] orelse return null;
    if (second_node.*.src[1] != first_acts) return null;
    if (dim(first, 0) != dim(second_node, 0) or dim(first, 1) != dim(second_node, 1)) return null;
    const first_weights: *const c.ggml_tensor = first.*.src[0] orelse return null;
    const second_weights: *const c.ggml_tensor = second_node.*.src[0] orelse return null;
    if (first_weights.*.type != second_weights.*.type or dim(first_weights, 0) != dim(second_weights, 0) or dim(first_weights, 1) != dim(second_weights, 1)) return null;

    // A view creates an additional alias/lifetime contract which the atomic
    // grouped command intentionally does not attempt to reconstruct.
    var graph_index: c_int = 0;
    while (graph_index < n) : (graph_index += 1) {
        const node = c.ggml_graph_node(graph, graph_index) orelse continue;
        if (node.*.view_src == first or node.*.view_src == second_node) return null;
    }

    const first_cmd = lowerMatmulQ1A8(first, lookup) catch return null;
    const second_cmd = lowerMatmulQ1A8(second_node, lookup) catch return null;
    const a = switch (first_cmd) {
        .matmul_q1a8 => |value| value,
        else => unreachable,
    };
    const b = switch (second_cmd) {
        .matmul_q1a8 => |value| value,
        else => unreachable,
    };
    if (a.rows != b.rows or a.cols != b.cols or a.k != b.k or a.weight_fmt != b.weight_fmt or !std.meta.eql(a.acts, b.acts)) return null;
    if (rangesOverlap(a.dst, b.dst) or
        rangesOverlap(a.dst, a.acts) or rangesOverlap(b.dst, a.acts) or
        rangesOverlap(a.dst, a.weights) or rangesOverlap(a.dst, b.weights) or
        rangesOverlap(b.dst, a.weights) or rangesOverlap(b.dst, b.weights)) return null;

    return .{
        .command = .{ .matmul_q1a8_group2 = .{
            .acts = a.acts,
            .projections = .{
                .{ .weights = a.weights, .dst = a.dst, .rows = a.rows, .weight_fmt = a.weight_fmt },
                .{ .weights = b.weights, .dst = b.dst, .rows = b.rows, .weight_fmt = b.weight_fmt },
            },
            .cols = a.cols,
            .k = a.k,
        } },
        .last_index = second_index,
    };
}

fn safeGroupedMatmulNode(node: *const c.ggml_tensor) bool {
    if (!supportsMatmulQ1A8(node) or node.*.view_src != null) return false;
    if ((node.*.flags & c.GGML_TENSOR_FLAG_COMPUTE) == 0) return false;
    const unsafe_flags = c.GGML_TENSOR_FLAG_INPUT | c.GGML_TENSOR_FLAG_OUTPUT | c.GGML_TENSOR_FLAG_PARAM | c.GGML_TENSOR_FLAG_LOSS;
    return (node.*.flags & unsafe_flags) == 0;
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

fn lowerFlashAttnF32(node: *const c.ggml_tensor, lookup: Lookup) LowerError!wire.Command {
    const q: *const c.ggml_tensor = node.*.src[0] orelse return error.InvalidShape;
    const k: *const c.ggml_tensor = node.*.src[1] orelse return error.InvalidShape;
    const v: *const c.ggml_tensor = node.*.src[2] orelse return error.InvalidShape;
    const mask: ?*const c.ggml_tensor = node.*.src[3];

    const head_dim_q = try u32Dim(dim(q, 0));
    const head_dim_v = try u32Dim(dim(v, 0));
    const n_tokens = try u32Dim(dim(q, 1));
    const n_heads = try u32Dim(dim(q, 2));
    const n_kv = try u32Dim(dim(k, 1));
    const n_head_kv = try u32Dim(dim(k, 2));

    const q_span = try stridedSpan(
        try checkedMul(@as(usize, @intCast(head_dim_q)), @sizeOf(f32)),
        n_tokens,
        n_heads,
        1,
        q.*.nb[1],
        q.*.nb[2],
        q.*.nb[2],
    );
    const k_span = try stridedSpan(
        try checkedMul(@as(usize, @intCast(head_dim_q)), @sizeOf(f16)),
        n_kv,
        n_head_kv,
        1,
        k.*.nb[1],
        k.*.nb[2],
        k.*.nb[2],
    );
    const v_span = try stridedSpan(
        try checkedMul(@as(usize, @intCast(head_dim_v)), @sizeOf(f16)),
        n_kv,
        n_head_kv,
        1,
        v.*.nb[1],
        v.*.nb[2],
        v.*.nb[2],
    );
    const dst_span = try stridedSpan(
        try checkedMul(@as(usize, @intCast(head_dim_v)), @sizeOf(f32)),
        n_heads,
        n_tokens,
        1,
        node.*.nb[1],
        node.*.nb[2],
        node.*.nb[2],
    );

    const q_binding = lookup.find(q) orelse return error.MissingBinding;
    const k_binding = lookup.find(k) orelse return error.MissingBinding;
    const v_binding = lookup.find(v) orelse return error.MissingBinding;
    const dst_binding = lookup.find(node) orelse return error.MissingBinding;

    var mask_range: wire.TensorRange = .{ .handle = 0, .offset = 0, .nbytes = 0 };
    var mask_nb1: u64 = 0;
    if (mask) |m| {
        const mask_row_bytes = try checkedMul(@as(usize, @intCast(n_kv)), @sizeOf(f16));
        const mask_span = try stridedSpan(mask_row_bytes, n_tokens, 1, 1, m.*.nb[1], mask_row_bytes, mask_row_bytes);
        const mask_binding = lookup.find(m) orelse return error.MissingBinding;
        mask_range = try backingRange(mask_binding, mask_span);
        mask_nb1 = @intCast(m.*.nb[1]);
    }

    return .{ .flash_attn_f32 = .{
        .q = try backingRange(q_binding, q_span),
        .k = try backingRange(k_binding, k_span),
        .v = try backingRange(v_binding, v_span),
        .mask = mask_range,
        .dst = try backingRange(dst_binding, dst_span),
        .has_mask = mask != null,
        .head_dim_q = head_dim_q,
        .head_dim_v = head_dim_v,
        .n_heads = n_heads,
        .n_head_kv = n_head_kv,
        .n_kv = n_kv,
        .n_tokens = n_tokens,
        .scale = opParamF32(node, 0),
        .q_nb1 = @intCast(q.*.nb[1]),
        .q_nb2 = @intCast(q.*.nb[2]),
        .k_nb1 = @intCast(k.*.nb[1]),
        .k_nb2 = @intCast(k.*.nb[2]),
        .v_nb1 = @intCast(v.*.nb[1]),
        .v_nb2 = @intCast(v.*.nb[2]),
        .mask_nb1 = mask_nb1,
        .dst_nb1 = @intCast(node.*.nb[1]),
        .dst_nb2 = @intCast(node.*.nb[2]),
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
    const src_type: wire.GetRowsSrcType = if (src0.*.type == c.GGML_TYPE_Q1_0)
        .q1_0
    else if (src0.*.type == c.GGML_TYPE_Q2_0)
        .q2_0
    else
        .f32;

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
        .q1_0 => layout.packedWeightBytes(src_rows, row_width) catch return error.InvalidShape,
        .q2_0 => layout.packedTernaryWeightBytes(src_rows, row_width) catch return error.InvalidShape,
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

fn lowerArgmax(node: *const c.ggml_tensor, lookup: Lookup) LowerError!wire.Command {
    const src: *const c.ggml_tensor = node.*.src[0] orelse return error.InvalidShape;
    const cols = try u32Dim(dim(src, 0));
    const rows = try u32Dim(dim(src, 1));
    const src_bytes = try checkedMul(try checkedMul(@as(usize, cols), @as(usize, rows)), @sizeOf(f32));
    const dst_bytes = try checkedMul(@as(usize, rows), @sizeOf(i32));

    const src_binding = lookup.find(src) orelse return error.MissingBinding;
    const dst_binding = lookup.find(node) orelse return error.MissingBinding;

    return .{ .argmax = .{
        .src = try backingRange(src_binding, src_bytes),
        .dst = try backingRange(dst_binding, dst_bytes),
        .rows = rows,
        .cols = cols,
    } };
}

fn lowerPad(node: *const c.ggml_tensor, lookup: Lookup) LowerError!wire.Command {
    const src: *const c.ggml_tensor = node.*.src[0] orelse return error.InvalidShape;
    const src_binding = lookup.find(src) orelse return error.MissingBinding;
    const dst_binding = lookup.find(node) orelse return error.MissingBinding;
    return .{ .pad = .{
        .src = try backingRange(src_binding, tensorNbytes(src)),
        .dst = try backingRange(dst_binding, tensorNbytes(node)),
    } };
}

// ===========================  Range / span helpers  ===========================

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

fn rangesOverlap(a: wire.TensorRange, b: wire.TensorRange) bool {
    if (a.handle != b.handle or a.nbytes == 0 or b.nbytes == 0) return false;
    const a_end = std.math.add(u64, a.offset, a.nbytes) catch return true;
    const b_end = std.math.add(u64, b.offset, b.nbytes) catch return true;
    return a.offset < b_end and b.offset < a_end;
}

fn stridedSpan(row_bytes: usize, ne1: u32, ne2: u32, ne3: u32, nb1: usize, nb2: usize, nb3: usize) LowerError!usize {
    if (row_bytes == 0 or ne1 == 0 or ne2 == 0 or ne3 == 0) return error.InvalidShape;
    var span = row_bytes;
    span = try checkedAdd(span, try checkedMul(@as(usize, @intCast(ne1 - 1)), nb1));
    span = try checkedAdd(span, try checkedMul(@as(usize, @intCast(ne2 - 1)), nb2));
    span = try checkedAdd(span, try checkedMul(@as(usize, @intCast(ne3 - 1)), nb3));
    return span;
}

// ===========================  Tensor geometry & op-param accessors  ===========================

fn tensorElements(tensor: *const c.ggml_tensor) usize {
    const raw = c.ggml_nelements(@constCast(tensor));
    if (raw <= 0) return 0;
    return @intCast(raw);
}

fn tensorNbytes(tensor: *const c.ggml_tensor) usize {
    return @intCast(c.ggml_nbytes(@constCast(tensor)));
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

const TestRmsNormMul = struct {
    input: *c.ggml_tensor,
    weight: *c.ggml_tensor,
    rmsnorm: *c.ggml_tensor,
    mul: *c.ggml_tensor,
};

const TestMatmulPair = struct {
    acts: *c.ggml_tensor,
    first_weights: *c.ggml_tensor,
    second_weights: *c.ggml_tensor,
    first: *c.ggml_tensor,
    second: *c.ggml_tensor,
};

fn testContext() !*c.ggml_context {
    return c.ggml_init(.{
        .mem_size = 1024 * 1024,
        .mem_buffer = null,
        .no_alloc = true,
    }) orelse error.OutOfMemory;
}

fn testGraph(ctx: *c.ggml_context) !*c.ggml_cgraph {
    return c.ggml_new_graph_custom(ctx, 128, false) orelse error.OutOfMemory;
}

fn testRmsNormMul(ctx: *c.ggml_context, rows: i64, cols: i64, eps: f32, reverse: bool) !TestRmsNormMul {
    const input = c.ggml_new_tensor_2d(ctx, c.GGML_TYPE_F32, rows, cols) orelse return error.OutOfMemory;
    const weight = c.ggml_new_tensor_1d(ctx, c.GGML_TYPE_F32, rows) orelse return error.OutOfMemory;
    const rmsnorm = c.ggml_rms_norm(ctx, input, eps) orelse return error.OutOfMemory;
    const mul = if (reverse)
        c.ggml_mul(ctx, weight, rmsnorm) orelse return error.OutOfMemory
    else
        c.ggml_mul(ctx, rmsnorm, weight) orelse return error.OutOfMemory;
    return .{ .input = input, .weight = weight, .rmsnorm = rmsnorm, .mul = mul };
}

fn testMatmulPair(ctx: *c.ggml_context, rows: i64, cols: i64, second_acts: bool, ternary: bool) !TestMatmulPair {
    const k: i64 = layout.q1_block;
    const acts = c.ggml_new_tensor_2d(ctx, c.GGML_TYPE_F32, k, cols) orelse return error.OutOfMemory;
    const other_acts = if (second_acts)
        c.ggml_new_tensor_2d(ctx, c.GGML_TYPE_F32, k, cols) orelse return error.OutOfMemory
    else
        acts;
    const weight_type: c.enum_ggml_type = @intCast(if (ternary) c.GGML_TYPE_Q2_0 else c.GGML_TYPE_Q1_0);
    const first_weights = c.ggml_new_tensor_2d(ctx, weight_type, k, rows) orelse return error.OutOfMemory;
    const second_weights = c.ggml_new_tensor_2d(ctx, weight_type, k, rows) orelse return error.OutOfMemory;
    const first = c.ggml_mul_mat(ctx, first_weights, acts) orelse return error.OutOfMemory;
    const second = c.ggml_mul_mat(ctx, second_weights, other_acts) orelse return error.OutOfMemory;
    return .{
        .acts = acts,
        .first_weights = first_weights,
        .second_weights = second_weights,
        .first = first,
        .second = second,
    };
}

fn testFindBinding(ctx: *anyopaque, tensor: ?*const c.ggml_tensor) ?Binding {
    _ = ctx;
    const t = tensor orelse return null;
    const nbytes = tensorNbytes(t);
    return .{
        .range = .{ .handle = @intCast(@intFromPtr(t)), .offset = 0, .nbytes = @intCast(nbytes) },
        .handle_nbytes = @intCast(nbytes),
    };
}

fn testLookup() Lookup {
    const dummy: *anyopaque = @ptrFromInt(@alignOf(usize));
    return .{ .ctx = dummy, .findFn = testFindBinding };
}

const TestMissingBinding = struct {
    tensor: *const c.ggml_tensor,

    fn find(ctx: *anyopaque, tensor: ?*const c.ggml_tensor) ?Binding {
        const self: *TestMissingBinding = @ptrCast(@alignCast(ctx));
        if (tensor == self.tensor) return null;
        return testFindBinding(ctx, tensor);
    }

    fn lookup(self: *TestMissingBinding) Lookup {
        return .{ .ctx = self, .findFn = find };
    }
};

const TestAliasedBindings = struct {
    alias: *const c.ggml_tensor,
    target: *const c.ggml_tensor,

    fn find(ctx: *anyopaque, tensor: ?*const c.ggml_tensor) ?Binding {
        const self: *TestAliasedBindings = @ptrCast(@alignCast(ctx));
        if (tensor == self.alias) return testFindBinding(ctx, self.target);
        return testFindBinding(ctx, tensor);
    }

    fn lookup(self: *TestAliasedBindings) Lookup {
        return .{ .ctx = self, .findFn = find };
    }
};

fn expectStandaloneRmsNormMul(commands: []const wire.Command) !void {
    try std.testing.expectEqual(@as(usize, 2), commands.len);
    switch (commands[0]) {
        .rmsnorm => |op| try std.testing.expect(!op.has_weight),
        else => return error.TestUnexpectedResult,
    }
    switch (commands[1]) {
        .mul_f32 => {},
        else => return error.TestUnexpectedResult,
    }
}

fn expectStandaloneMatmulPair(commands: []const wire.Command) !void {
    try std.testing.expectEqual(@as(usize, 2), commands.len);
    switch (commands[0]) {
        .matmul_q1a8 => {},
        else => return error.TestUnexpectedResult,
    }
    switch (commands[1]) {
        .matmul_q1a8 => {},
        else => return error.TestUnexpectedResult,
    }
}

test "lowering groups exact adjacent matmuls with one activation" {
    const ctx = try testContext();
    defer c.ggml_free(ctx);
    const pair = try testMatmulPair(ctx, 16, 3, false, false);
    const graph = try testGraph(ctx);
    c.ggml_build_forward_expand(graph, pair.first);
    c.ggml_build_forward_expand(graph, pair.second);

    const commands = try lowerGraph(std.testing.allocator, graph, testLookup());
    defer std.testing.allocator.free(commands);
    try std.testing.expectEqual(@as(usize, 1), commands.len);
    switch (commands[0]) {
        .matmul_q1a8_group2 => |group| {
            try std.testing.expectEqual(@as(u32, 16), group.projections[0].rows);
            try std.testing.expectEqual(@as(u32, 16), group.projections[1].rows);
            try std.testing.expectEqual(@as(u32, 3), group.cols);
            try std.testing.expectEqual(@as(u32, layout.q1_block), group.k);
            try std.testing.expectEqual(@as(u64, @intCast(@intFromPtr(pair.acts))), group.acts.handle);
            try std.testing.expectEqual(@as(u64, @intCast(@intFromPtr(pair.first_weights))), group.projections[0].weights.handle);
            try std.testing.expectEqual(@as(u64, @intCast(@intFromPtr(pair.second_weights))), group.projections[1].weights.handle);
        },
        else => return error.TestUnexpectedResult,
    }
}

test "lowering groups adjacent ternary matmuls" {
    const ctx = try testContext();
    defer c.ggml_free(ctx);
    const pair = try testMatmulPair(ctx, 16, 2, false, true);
    const graph = try testGraph(ctx);
    c.ggml_build_forward_expand(graph, pair.first);
    c.ggml_build_forward_expand(graph, pair.second);

    const commands = try lowerGraph(std.testing.allocator, graph, testLookup());
    defer std.testing.allocator.free(commands);
    try std.testing.expectEqual(@as(usize, 1), commands.len);
    try std.testing.expectEqual(wire.WeightFormat.w158a8, commands[0].matmul_q1a8_group2.projections[0].weight_fmt);
    try std.testing.expectEqual(wire.WeightFormat.w158a8, commands[0].matmul_q1a8_group2.projections[1].weight_fmt);
}

test "matmul grouping rejects different activation tensors" {
    const ctx = try testContext();
    defer c.ggml_free(ctx);
    const pair = try testMatmulPair(ctx, 16, 2, true, false);
    const graph = try testGraph(ctx);
    c.ggml_build_forward_expand(graph, pair.first);
    c.ggml_build_forward_expand(graph, pair.second);

    const commands = try lowerGraph(std.testing.allocator, graph, testLookup());
    defer std.testing.allocator.free(commands);
    try expectStandaloneMatmulPair(commands);
}

test "matmul grouping rejects unsafe output flags" {
    const ctx = try testContext();
    defer c.ggml_free(ctx);
    const pair = try testMatmulPair(ctx, 16, 2, false, false);
    c.ggml_set_output(pair.first);
    const graph = try testGraph(ctx);
    c.ggml_build_forward_expand(graph, pair.first);
    c.ggml_build_forward_expand(graph, pair.second);

    const commands = try lowerGraph(std.testing.allocator, graph, testLookup());
    defer std.testing.allocator.free(commands);
    try expectStandaloneMatmulPair(commands);
}

test "matmul grouping rejects output views and overlapping bindings" {
    const ctx = try testContext();
    defer c.ggml_free(ctx);
    const pair = try testMatmulPair(ctx, 16, 2, false, false);
    const view = c.ggml_view_2d(ctx, pair.first, 16, 2, 16 * @sizeOf(f32), 0) orelse return error.OutOfMemory;
    const graph = try testGraph(ctx);
    c.ggml_build_forward_expand(graph, pair.first);
    c.ggml_build_forward_expand(graph, pair.second);
    c.ggml_build_forward_expand(graph, view);

    const view_commands = try lowerGraph(std.testing.allocator, graph, testLookup());
    defer std.testing.allocator.free(view_commands);
    try expectStandaloneMatmulPair(view_commands);

    const clean_graph = try testGraph(ctx);
    c.ggml_build_forward_expand(clean_graph, pair.first);
    c.ggml_build_forward_expand(clean_graph, pair.second);
    var aliases = TestAliasedBindings{ .alias = pair.second, .target = pair.first };
    const alias_commands = try lowerGraph(std.testing.allocator, clean_graph, aliases.lookup());
    defer std.testing.allocator.free(alias_commands);
    try expectStandaloneMatmulPair(alias_commands);
}

test "matmul grouping requires every binding" {
    const ctx = try testContext();
    defer c.ggml_free(ctx);
    const pair = try testMatmulPair(ctx, 16, 2, false, false);
    const graph = try testGraph(ctx);
    c.ggml_build_forward_expand(graph, pair.first);
    c.ggml_build_forward_expand(graph, pair.second);

    var missing = TestMissingBinding{ .tensor = pair.second_weights };
    try std.testing.expect(tryLowerMatmulQ1A8Group2(graph, 0, missing.lookup()) == null);
}

test "matmul grouping rejects intervening compute" {
    const ctx = try testContext();
    defer c.ggml_free(ctx);
    const pair = try testMatmulPair(ctx, 16, 2, false, false);
    const middle_lhs = c.ggml_new_tensor_1d(ctx, c.GGML_TYPE_F32, 8) orelse return error.OutOfMemory;
    const middle_rhs = c.ggml_new_tensor_1d(ctx, c.GGML_TYPE_F32, 8) orelse return error.OutOfMemory;
    const middle = c.ggml_add(ctx, middle_lhs, middle_rhs) orelse return error.OutOfMemory;
    const graph = try testGraph(ctx);
    c.ggml_build_forward_expand(graph, pair.first);
    c.ggml_build_forward_expand(graph, middle);
    c.ggml_build_forward_expand(graph, pair.second);

    const commands = try lowerGraph(std.testing.allocator, graph, testLookup());
    defer std.testing.allocator.free(commands);
    try std.testing.expectEqual(@as(usize, 3), commands.len);
    switch (commands[0]) {
        .matmul_q1a8 => {},
        else => return error.TestUnexpectedResult,
    }
    switch (commands[1]) {
        .add_f32 => {},
        else => return error.TestUnexpectedResult,
    }
    switch (commands[2]) {
        .matmul_q1a8 => {},
        else => return error.TestUnexpectedResult,
    }
}

test "flash lowering preserves llama token-major Q strides" {
    const ctx = try testContext();
    defer c.ggml_free(ctx);

    const head_dim: i64 = 128;
    const n_heads: i64 = 16;
    const n_head_kv: i64 = 8;
    const n_tokens: i64 = 16;
    const n_kv: i64 = 256;

    // llama.cpp builds contiguous [D, H, T], then applies permute(0, 2, 1, 3)
    // before FLASH_ATTN_EXT, producing logical [D, T, H] with token-major backing.
    const q_storage = c.ggml_new_tensor_3d(ctx, c.GGML_TYPE_F32, head_dim, n_heads, n_tokens) orelse return error.OutOfMemory;
    const q = c.ggml_permute(ctx, q_storage, 0, 2, 1, 3) orelse return error.OutOfMemory;
    const k_storage = c.ggml_new_tensor_3d(ctx, c.GGML_TYPE_F16, head_dim, n_head_kv, n_kv) orelse return error.OutOfMemory;
    const k = c.ggml_permute(ctx, k_storage, 0, 2, 1, 3) orelse return error.OutOfMemory;
    const v_storage = c.ggml_new_tensor_3d(ctx, c.GGML_TYPE_F16, head_dim, n_head_kv, n_kv) orelse return error.OutOfMemory;
    const v = c.ggml_permute(ctx, v_storage, 0, 2, 1, 3) orelse return error.OutOfMemory;
    const mask = c.ggml_new_tensor_2d(ctx, c.GGML_TYPE_F16, n_kv, n_tokens) orelse return error.OutOfMemory;
    const attn = c.ggml_flash_attn_ext(ctx, q, k, v, mask, 0.08838835, 0, 0) orelse return error.OutOfMemory;

    try std.testing.expectEqual(@as(usize, n_heads * head_dim * @sizeOf(f32)), q[0].nb[1]);
    try std.testing.expectEqual(@as(usize, head_dim * @sizeOf(f32)), q[0].nb[2]);
    try std.testing.expect(supportsOp(attn));

    const graph = try testGraph(ctx);
    c.ggml_build_forward_expand(graph, attn);
    const commands = try lowerGraph(std.testing.allocator, graph, testLookup());
    defer std.testing.allocator.free(commands);
    try std.testing.expectEqual(@as(usize, 1), commands.len);
    switch (commands[0]) {
        .flash_attn_f32 => |op| {
            try std.testing.expectEqual(@as(u64, 8192), op.q_nb1);
            try std.testing.expectEqual(@as(u64, 512), op.q_nb2);
            try std.testing.expectEqual(@as(u64, 2048), op.k_nb1);
            try std.testing.expectEqual(@as(u64, 256), op.k_nb2);
            try std.testing.expectEqual(@as(u64, 512), op.mask_nb1);
            try std.testing.expectEqual(@as(u64, 512), op.dst_nb1);
            try std.testing.expectEqual(@as(u64, 8192), op.dst_nb2);
        },
        else => return error.TestUnexpectedResult,
    }
}

test "lowering fuses exact adjacent rmsnorm gamma multiply" {
    const ctx = try testContext();
    defer c.ggml_free(ctx);
    const pair = try testRmsNormMul(ctx, 8, 3, 1e-5, false);
    const graph = try testGraph(ctx);
    c.ggml_build_forward_expand(graph, pair.mul);

    const commands = try lowerGraph(std.testing.allocator, graph, testLookup());
    defer std.testing.allocator.free(commands);
    try std.testing.expectEqual(@as(usize, 1), commands.len);
    switch (commands[0]) {
        .rmsnorm => |op| {
            try std.testing.expect(op.has_weight);
            try std.testing.expectEqual(@as(u64, 8 * @sizeOf(f32)), op.weight.nbytes);
            try std.testing.expectEqual(@as(u64, 8 * 3 * @sizeOf(f32)), op.dst.nbytes);
            try std.testing.expectEqual(@as(u64, @intCast(@intFromPtr(pair.weight))), op.weight.handle);
            try std.testing.expectEqual(@as(u64, @intCast(@intFromPtr(pair.mul))), op.dst.handle);
        },
        else => return error.TestUnexpectedResult,
    }
}

test "lowering fuses rmsnorm when it is the reversed multiply operand" {
    const ctx = try testContext();
    defer c.ggml_free(ctx);
    const pair = try testRmsNormMul(ctx, 8, 1, 1e-5, true);
    const graph = try testGraph(ctx);
    c.ggml_build_forward_expand(graph, pair.mul);

    const commands = try lowerGraph(std.testing.allocator, graph, testLookup());
    defer std.testing.allocator.free(commands);
    try std.testing.expectEqual(@as(usize, 1), commands.len);
    try std.testing.expect(commands[0].rmsnorm.has_weight);
    try std.testing.expectEqual(@as(u64, @intCast(@intFromPtr(pair.weight))), commands[0].rmsnorm.weight.handle);
}

test "lowering retains rmsnorm and mul for extra consumers" {
    const ctx = try testContext();
    defer c.ggml_free(ctx);
    const pair = try testRmsNormMul(ctx, 8, 2, 1e-5, false);
    const other_weight = c.ggml_new_tensor_1d(ctx, c.GGML_TYPE_F32, 8) orelse return error.OutOfMemory;
    const other_mul = c.ggml_mul(ctx, pair.rmsnorm, other_weight) orelse return error.OutOfMemory;
    const graph = try testGraph(ctx);
    c.ggml_build_forward_expand(graph, pair.mul);
    c.ggml_build_forward_expand(graph, other_mul);

    const commands = try lowerGraph(std.testing.allocator, graph, testLookup());
    defer std.testing.allocator.free(commands);
    try std.testing.expectEqual(@as(usize, 3), commands.len);
    try std.testing.expect(!commands[0].rmsnorm.has_weight);
    switch (commands[1]) {
        .mul_f32 => {},
        else => return error.TestUnexpectedResult,
    }
    switch (commands[2]) {
        .mul_f32 => {},
        else => return error.TestUnexpectedResult,
    }
}

test "lowering retains rmsnorm and mul when rmsnorm is an output" {
    const ctx = try testContext();
    defer c.ggml_free(ctx);
    const pair = try testRmsNormMul(ctx, 8, 2, 1e-5, false);
    c.ggml_set_output(pair.rmsnorm);
    const graph = try testGraph(ctx);
    c.ggml_build_forward_expand(graph, pair.mul);

    const commands = try lowerGraph(std.testing.allocator, graph, testLookup());
    defer std.testing.allocator.free(commands);
    try expectStandaloneRmsNormMul(commands);
}

test "lowering retains rmsnorm and mul for rmsnorm views" {
    const ctx = try testContext();
    defer c.ggml_free(ctx);
    const input = c.ggml_new_tensor_2d(ctx, c.GGML_TYPE_F32, 8, 2) orelse return error.OutOfMemory;
    const weight = c.ggml_new_tensor_1d(ctx, c.GGML_TYPE_F32, 8) orelse return error.OutOfMemory;
    const rmsnorm = c.ggml_rms_norm_inplace(ctx, input, 1e-5) orelse return error.OutOfMemory;
    const mul = c.ggml_mul(ctx, rmsnorm, weight) orelse return error.OutOfMemory;
    const graph = try testGraph(ctx);
    c.ggml_build_forward_expand(graph, mul);

    const commands = try lowerGraph(std.testing.allocator, graph, testLookup());
    defer std.testing.allocator.free(commands);
    try expectStandaloneRmsNormMul(commands);
}

test "lowering retains rmsnorm and mul when a view also consumes rmsnorm" {
    const ctx = try testContext();
    defer c.ggml_free(ctx);
    const pair = try testRmsNormMul(ctx, 8, 2, 1e-5, false);
    const view = c.ggml_view_2d(ctx, pair.rmsnorm, 8, 2, 8 * @sizeOf(f32), 0) orelse return error.OutOfMemory;
    const view_weight = c.ggml_new_tensor_1d(ctx, c.GGML_TYPE_F32, 8) orelse return error.OutOfMemory;
    const view_mul = c.ggml_mul(ctx, view, view_weight) orelse return error.OutOfMemory;
    const graph = try testGraph(ctx);
    c.ggml_build_forward_expand(graph, pair.mul);
    c.ggml_build_forward_expand(graph, view_mul);

    const commands = try lowerGraph(std.testing.allocator, graph, testLookup());
    defer std.testing.allocator.free(commands);
    try std.testing.expectEqual(@as(usize, 3), commands.len);
    try std.testing.expect(!commands[0].rmsnorm.has_weight);
}

test "lowering requires raw adjacency for rmsnorm fusion" {
    const ctx = try testContext();
    defer c.ggml_free(ctx);
    const pair = try testRmsNormMul(ctx, 8, 2, 1e-5, false);
    const other_weight = c.ggml_new_tensor_1d(ctx, c.GGML_TYPE_F32, 8) orelse return error.OutOfMemory;
    const zero = c.ggml_new_tensor_1d(ctx, c.GGML_TYPE_F32, 8) orelse return error.OutOfMemory;
    const computed_weight = c.ggml_add(ctx, other_weight, zero) orelse return error.OutOfMemory;
    pair.mul.*.src[1] = computed_weight;
    const graph = try testGraph(ctx);
    c.ggml_build_forward_expand(graph, pair.mul);

    const commands = try lowerGraph(std.testing.allocator, graph, testLookup());
    defer std.testing.allocator.free(commands);
    try std.testing.expectEqual(@as(usize, 3), commands.len);
    try std.testing.expect(!commands[0].rmsnorm.has_weight);
}

test "lowering keeps same-shape non-gamma multiply generic" {
    const ctx = try testContext();
    defer c.ggml_free(ctx);
    const input = c.ggml_new_tensor_2d(ctx, c.GGML_TYPE_F32, 8, 2) orelse return error.OutOfMemory;
    const weight = c.ggml_new_tensor_2d(ctx, c.GGML_TYPE_F32, 8, 2) orelse return error.OutOfMemory;
    const rmsnorm = c.ggml_rms_norm(ctx, input, 1e-5) orelse return error.OutOfMemory;
    const mul = c.ggml_mul(ctx, rmsnorm, weight) orelse return error.OutOfMemory;
    const graph = try testGraph(ctx);
    c.ggml_build_forward_expand(graph, mul);

    const commands = try lowerGraph(std.testing.allocator, graph, testLookup());
    defer std.testing.allocator.free(commands);
    try expectStandaloneRmsNormMul(commands);
    try std.testing.expectEqual(wire.BinaryF32Mode.same_shape, commands[1].mul_f32.mode);
}

test "rmsnorm fusion rejects invalid epsilon type and stride" {
    const ctx = try testContext();
    defer c.ggml_free(ctx);

    const invalid_eps = try testRmsNormMul(ctx, 8, 2, -1, false);
    const eps_graph = try testGraph(ctx);
    c.ggml_build_forward_expand(eps_graph, invalid_eps.mul);
    try std.testing.expect(rmsNormMulF32Pair(eps_graph, 0) == null);
    const eps_commands = try lowerGraph(std.testing.allocator, eps_graph, testLookup());
    defer std.testing.allocator.free(eps_commands);
    try expectStandaloneRmsNormMul(eps_commands);

    const typed_input = c.ggml_new_tensor_2d(ctx, c.GGML_TYPE_F32, 8, 2) orelse return error.OutOfMemory;
    const typed_weight = c.ggml_new_tensor_1d(ctx, c.GGML_TYPE_F16, 8) orelse return error.OutOfMemory;
    const typed_rms = c.ggml_rms_norm(ctx, typed_input, 1e-5) orelse return error.OutOfMemory;
    const typed_mul = c.ggml_mul(ctx, typed_rms, typed_weight) orelse return error.OutOfMemory;
    const typed_graph = try testGraph(ctx);
    c.ggml_build_forward_expand(typed_graph, typed_mul);
    try std.testing.expect(rmsNormMulF32Pair(typed_graph, 0) == null);
    try std.testing.expect(!supportsOp(typed_mul));

    const strided = try testRmsNormMul(ctx, 8, 2, 1e-5, false);
    strided.weight.*.nb[0] = 2 * @sizeOf(f32);
    const strided_graph = try testGraph(ctx);
    c.ggml_build_forward_expand(strided_graph, strided.mul);
    try std.testing.expect(rmsNormMulF32Pair(strided_graph, 0) == null);
    try std.testing.expect(!supportsOp(strided.mul));

    const shaped = try testRmsNormMul(ctx, 8, 2, 1e-5, false);
    shaped.mul.*.ne[1] = 3;
    const shaped_graph = try testGraph(ctx);
    c.ggml_build_forward_expand(shaped_graph, shaped.mul);
    try std.testing.expect(rmsNormMulF32Pair(shaped_graph, 0) == null);
    try std.testing.expect(!supportsOp(shaped.mul));
}

test "rmsnorm fusion requires every resident binding" {
    const ctx = try testContext();
    defer c.ggml_free(ctx);
    const pair = try testRmsNormMul(ctx, 8, 2, 1e-5, false);
    const graph = try testGraph(ctx);
    c.ggml_build_forward_expand(graph, pair.mul);

    var missing_weight = TestMissingBinding{ .tensor = pair.weight };
    try std.testing.expect(tryLowerRmsNormMulF32(graph, 0, missing_weight.lookup()) == null);
    var missing_intermediate = TestMissingBinding{ .tensor = pair.rmsnorm };
    try std.testing.expect(tryLowerRmsNormMulF32(graph, 0, missing_intermediate.lookup()) == null);
    var missing_output = TestMissingBinding{ .tensor = pair.mul };
    try std.testing.expect(tryLowerRmsNormMulF32(graph, 0, missing_output.lookup()) == null);
}

test "lowering leaves standalone rmsnorm unchanged" {
    const ctx = try testContext();
    defer c.ggml_free(ctx);
    const input = c.ggml_new_tensor_2d(ctx, c.GGML_TYPE_F32, 8, 2) orelse return error.OutOfMemory;
    const rmsnorm = c.ggml_rms_norm(ctx, input, 1e-5) orelse return error.OutOfMemory;
    const graph = try testGraph(ctx);
    c.ggml_build_forward_expand(graph, rmsnorm);

    const commands = try lowerGraph(std.testing.allocator, graph, testLookup());
    defer std.testing.allocator.free(commands);
    try std.testing.expectEqual(@as(usize, 1), commands.len);
    try std.testing.expect(!commands[0].rmsnorm.has_weight);
}
