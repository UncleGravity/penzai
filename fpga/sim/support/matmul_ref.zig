//! Scalar Q1A8 matmul reference — the oracle for software, Verilator RTL, and
//! the packing contract. One activation column is multiplied by all rows.
//!
//! Per plan §11 the integer part is exact and order-independent: test the
//! int32 sub-sums with `==`. The fp16-scale application is normal float
//! arithmetic: test the scaled output within ε. We keep the two separable so a
//! bit-exact gate survives even though the final fp32 is only ε-comparable.

const std = @import("std");
const layout = @import("layout");

/// One matmul problem, in logical (unpacked) form. Activations are a single
/// column shared across every row.
pub const Problem = struct {
    rows: usize, // total rows (= num_rowblocks * ROWS)
    q1_blocks: usize, // K / Q1_BLOCK

    /// Weight sign bits, row-major: weight_bits[row * q1_blocks + blk] is a
    /// Q1_BLOCK-wide bitmask; bit i set => +1, clear => -1.
    weight_bits: []const u128,
    /// fp16 weight scale per (row, Q1 block), same indexing as weight_bits.
    weight_scales: []const f16,

    /// int8 activation quants, length q1_blocks * Q1_BLOCK.
    act_quants: []const i8,
    /// fp16 activation scale per Q8 sub-block, length q1_blocks * Q8_SUBBLOCKS.
    act_scales: []const f16,

    pub fn subblockCount(self: Problem) usize {
        return self.rows * self.q1_blocks * layout.Q8_SUBBLOCKS;
    }
};

/// Integer sub-sum for one (row, Q1 block, Q8 sub-block): Σ ±act over the
/// sub-block. Exact; this is the bit-exact gate against the fabric.
pub fn subSum(p: Problem, row: usize, blk: usize, sub: usize) i32 {
    const bits = p.weight_bits[row * p.q1_blocks + blk];
    var sum: i32 = 0;
    var i: usize = 0;
    while (i < layout.Q8_BLOCK) : (i += 1) {
        const bit_index = sub * layout.Q8_BLOCK + i;
        const act: i32 = p.act_quants[blk * layout.Q1_BLOCK + bit_index];
        const set = (bits >> @intCast(bit_index)) & 1 == 1;
        sum += if (set) act else -act;
    }
    return sum;
}

/// Fill `out` (len p.subblockCount()) with every int32 sub-sum, ordered
/// (row, blk, sub). The `==` oracle.
pub fn accumulateInt(p: Problem, out: []i32) void {
    std.debug.assert(out.len == p.subblockCount());
    var idx: usize = 0;
    var row: usize = 0;
    while (row < p.rows) : (row += 1) {
        var blk: usize = 0;
        while (blk < p.q1_blocks) : (blk += 1) {
            var sub: usize = 0;
            while (sub < layout.Q8_SUBBLOCKS) : (sub += 1) {
                out[idx] = subSum(p, row, blk, sub);
                idx += 1;
            }
        }
    }
}

/// fp32 output per row (len p.rows): Σ weight_scale·act_scale·sub_sum.
/// Standard f32 accumulation — compare within ε, not `==`.
pub fn scaledOutput(p: Problem, out: []f32) void {
    std.debug.assert(out.len == p.rows);
    var row: usize = 0;
    while (row < p.rows) : (row += 1) {
        var acc: f32 = 0;
        var blk: usize = 0;
        while (blk < p.q1_blocks) : (blk += 1) {
            const ws: f32 = @floatCast(p.weight_scales[row * p.q1_blocks + blk]);
            var sub: usize = 0;
            while (sub < layout.Q8_SUBBLOCKS) : (sub += 1) {
                const as_: f32 = @floatCast(p.act_scales[blk * layout.Q8_SUBBLOCKS + sub]);
                if (ws == 0 or as_ == 0) continue;
                const ss: f32 = @floatFromInt(subSum(p, row, blk, sub));
                acc += ws * as_ * ss;
            }
        }
        out[row] = acc;
    }
}

const testing = std.testing;

test "all-ones: sub_sum == Q8_BLOCK, output == blocks*Q8_SUBBLOCKS" {
    // bits all set (+1), acts all +1, scales all 1.0 -> each sub_sum = 32,
    // each row accumulates q1_blocks * 4 sub-blocks * 32 * 1 * 1.
    const blocks = 3;
    const rows = layout.ROWS;
    var bits = [_]u128{std.math.maxInt(u128)} ** (rows * blocks);
    var wscales = [_]f16{1.0} ** (rows * blocks);
    var aquants = [_]i8{1} ** (blocks * layout.Q1_BLOCK);
    var ascales = [_]f16{1.0} ** (blocks * layout.Q8_SUBBLOCKS);
    const p: Problem = .{
        .rows = rows,
        .q1_blocks = blocks,
        .weight_bits = &bits,
        .weight_scales = &wscales,
        .act_quants = &aquants,
        .act_scales = &ascales,
    };

    var sums: [rows * blocks * layout.Q8_SUBBLOCKS]i32 = undefined;
    accumulateInt(p, &sums);
    for (sums) |s| try testing.expectEqual(@as(i32, 32), s);

    var out: [rows]f32 = undefined;
    scaledOutput(p, &out);
    const expect: f32 = @floatFromInt(blocks * layout.Q8_SUBBLOCKS * 32);
    for (out) |v| try testing.expectApproxEqAbs(expect, v, 1e-3);
}

test "sign bits flip activation contribution" {
    // bits = 0 (-1), acts = 5 -> sub_sum = -160 over the 32-wide sub-block.
    var bits = [_]u128{0} ** layout.ROWS;
    var wscales = [_]f16{1.0} ** layout.ROWS;
    var aquants = [_]i8{5} ** layout.Q1_BLOCK;
    var ascales = [_]f16{1.0} ** layout.Q8_SUBBLOCKS;
    const p: Problem = .{
        .rows = layout.ROWS,
        .q1_blocks = 1,
        .weight_bits = &bits,
        .weight_scales = &wscales,
        .act_quants = &aquants,
        .act_scales = &ascales,
    };
    try testing.expectEqual(@as(i32, -160), subSum(p, 0, 0, 0));
}
