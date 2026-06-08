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
    return supportsMatmulQ1A8(tensor);
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

fn range(binding: Binding, nbytes: usize) wire.TensorRange {
    return .{
        .handle = binding.range.handle,
        .offset = binding.range.offset,
        .nbytes = @intCast(nbytes),
    };
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
