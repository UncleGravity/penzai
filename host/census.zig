const std = @import("std");
const c = @import("c");
const lower = @import("lower");

const op_count: usize = @intCast(c.GGML_OP_COUNT);
const max_src: usize = @intCast(c.GGML_MAX_SRC);

pub const Census = struct {
    const Self = @This();

    graph_compute_calls: u64 = 0,
    node_count: u64 = 0,
    op_counts: [op_count]u64 = [_]u64{0} ** op_count,
    supported_counts: [op_count]u64 = [_]u64{0} ** op_count,
    unsupported_counts: [op_count]u64 = [_]u64{0} ** op_count,
    missing_src_bindings: u64 = 0,
    missing_dst_bindings: u64 = 0,
    matmul_q1_0: u64 = 0,
    matmul_f16: u64 = 0,
    matmul_f32: u64 = 0,
    matmul_other: u64 = 0,

    pub fn reset(self: *Self) void {
        self.* = .{};
    }

    pub fn recordGraph(self: *Self, graph: *c.ggml_cgraph, lookup: lower.Lookup) void {
        self.graph_compute_calls += 1;
        const n = c.ggml_graph_n_nodes(graph);
        var i: c_int = 0;
        while (i < n) : (i += 1) {
            const node = c.ggml_graph_node(graph, i) orelse continue;
            self.recordNode(node, lookup);
        }
    }

    pub fn report(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        try writer.print("penzai census\n", .{});
        try writer.print("graph_compute_calls={d} nodes={d}\n", .{ self.graph_compute_calls, self.node_count });
        try writer.print(
            "missing_bindings src={d} dst={d}\n",
            .{ self.missing_src_bindings, self.missing_dst_bindings },
        );
        try writer.print(
            "matmul_weight_dtype q1_0={d} f16={d} f32={d} other={d}\n\n",
            .{ self.matmul_q1_0, self.matmul_f16, self.matmul_f32, self.matmul_other },
        );

        try writer.print("ops:\n", .{});
        try writer.print("  {s: <18} {s: >8} {s: >10} {s: >10} {s: <9}\n", .{ "op", "count", "supported", "unsupported", "status" });
        for (0..op_count) |i| {
            if (self.op_counts[i] == 0) continue;
            try writer.print(
                "  {s: <18} {d: >8} {d: >10} {d: >10} {s: <9}\n",
                .{
                    opName(i),
                    self.op_counts[i],
                    self.supported_counts[i],
                    self.unsupported_counts[i],
                    supportStatus(self.supported_counts[i], self.unsupported_counts[i]),
                },
            );
        }

        try writer.print("\nworklist:\n", .{});
        var emitted: [op_count]bool = [_]bool{false} ** op_count;
        var any = false;
        while (true) {
            const next = self.nextWorklistIndex(&emitted) orelse break;
            emitted[next] = true;
            any = true;
            try writer.print("  {s: <18} {d}\n", .{ opName(next), self.unsupported_counts[next] });
        }
        if (!any) try writer.print("  none\n", .{});
    }

    fn recordNode(self: *Self, node: *c.ggml_tensor, lookup: lower.Lookup) void {
        self.node_count += 1;
        const index = opIndex(node.*.op) orelse return;
        self.op_counts[index] += 1;

        if (lower.supportsOp(node)) {
            self.supported_counts[index] += 1;
        } else {
            self.unsupported_counts[index] += 1;
        }

        if (lookup.find(node) == null) self.missing_dst_bindings += 1;
        for (0..max_src) |src_index| {
            const src: *const c.ggml_tensor = node.*.src[src_index] orelse continue;
            if (lookup.find(src) == null) self.missing_src_bindings += 1;
        }

        if (node.*.op == c.GGML_OP_MUL_MAT) {
            const weights: *const c.ggml_tensor = node.*.src[0] orelse {
                self.matmul_other += 1;
                return;
            };
            if (weights.*.type == c.GGML_TYPE_Q1_0) {
                self.matmul_q1_0 += 1;
            } else if (weights.*.type == c.GGML_TYPE_F16) {
                self.matmul_f16 += 1;
            } else if (weights.*.type == c.GGML_TYPE_F32) {
                self.matmul_f32 += 1;
            } else {
                self.matmul_other += 1;
            }
        }
    }

    fn nextWorklistIndex(self: *const Self, emitted: *const [op_count]bool) ?usize {
        var best: ?usize = null;
        var best_count: u64 = 0;
        for (0..op_count) |i| {
            const count = self.unsupported_counts[i];
            if (emitted[i] or count == 0 or isMetadataIndex(i)) continue;
            if (count > best_count) {
                best = i;
                best_count = count;
            }
        }
        return best;
    }
};

fn supportStatus(supported: u64, unsupported: u64) []const u8 {
    if (unsupported == 0) return "yes";
    if (supported == 0) return "no";
    return "mixed";
}

fn opIndex(op: c.enum_ggml_op) ?usize {
    const raw: usize = @intCast(op);
    if (raw >= op_count) return null;
    return raw;
}

fn isMetadataIndex(index: usize) bool {
    return index == @as(usize, @intCast(c.GGML_OP_NONE)) or
        index == @as(usize, @intCast(c.GGML_OP_RESHAPE)) or
        index == @as(usize, @intCast(c.GGML_OP_VIEW)) or
        index == @as(usize, @intCast(c.GGML_OP_PERMUTE)) or
        index == @as(usize, @intCast(c.GGML_OP_TRANSPOSE));
}

fn opName(index: usize) []const u8 {
    const name = c.ggml_op_name(@intCast(index));
    if (name == null) return "?";
    return std.mem.span(name);
}
