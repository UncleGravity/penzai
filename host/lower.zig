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
    return supportsMatmulQ1A8(tensor) or supportsSetRows(tensor);
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
        if (supportsSetRows(node)) {
            try commands.append(allocator, try lowerSetRows(node, lookup));
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
