//! Pure result-gather for the PL matmul.
//!
//! The kernel's S2MM stream is ordered `[rowblock][col][row]`, lane-major, with
//! a fixed `rows_per_block` fp32 values per (rowblock, col) regardless of how
//! many rows that rowblock actually carries -- the pad lanes sit *after* the
//! valid rows. This scatters that stream into the col-major destination tensor.
//!
//! Isolated from the /dev/mem driver so it is unit-testable: the coarse-copy
//! production path (`gatherResults`) is checked bit-for-bit against the obvious
//! per-lane reference (`gatherResultsRef`, the oracle) over a matrix of shapes.

const std = @import("std");
const shared = @import("shared");
const q1a8 = shared.q1a8;

pub const result_bytes_per_rb: usize = (q1a8.rows_per_block / 2) * q1a8.beat_bytes;

/// Scatter one kernel run's results (`group` columns starting at `col0`) into
/// `dst` (col-major `rows*cols` f32). Coarse memcpy: one contiguous copy for the
/// single-column case (the decode hot path), rowblock-granular otherwise.
pub fn gatherResults(dst: []u8, result: []const u8, rows: usize, group: usize, col0: usize, num_rb: usize) void {
    const fp = @sizeOf(f32);
    if (group == 1) {
        // Single column: `result` fp32 index == dst row index, and only the last
        // rowblock can be partial (its pad lanes sit past the last valid row), so
        // the first rows*4 bytes are exactly rows 0..rows-1 in order.
        @memcpy(dst[col0 * rows * fp ..][0 .. rows * fp], result[0 .. rows * fp]);
        return;
    }
    for (0..num_rb) |rb| {
        const row0 = rb * q1a8.rows_per_block;
        // `valid: usize` is load-bearing: @min with the comptime `rows_per_block`
        // narrows the result type, and `valid * fp` would then overflow that
        // narrow type (silently wrapping to 0 in ReleaseFast -> empty copies).
        const valid: usize = @min(q1a8.rows_per_block, rows - row0);
        const nbytes = valid * fp;
        for (0..group) |c| {
            const col = col0 + c;
            const src_off = (rb * group + c) * result_bytes_per_rb;
            const dst_off = (col * rows + row0) * fp;
            @memcpy(dst[dst_off .. dst_off + nbytes], result[src_off .. src_off + nbytes]);
        }
    }
}

/// Oracle: the obvious per-lane 4-byte copy. Slow, obviously correct.
fn gatherResultsRef(dst: []u8, result: []const u8, rows: usize, group: usize, col0: usize, num_rb: usize) void {
    for (0..group) |c| {
        const col = col0 + c;
        for (0..num_rb) |rb| {
            const chunk = (rb * group + c) * result_bytes_per_rb;
            const valid = @min(q1a8.rows_per_block, rows - rb * q1a8.rows_per_block);
            for (0..valid) |lane| {
                const grow = rb * q1a8.rows_per_block + lane;
                @memcpy(dst[(col * rows + grow) * 4 ..][0..4], result[chunk + lane * 4 ..][0..4]);
            }
        }
    }
}

test "gatherResults matches the per-lane oracle and places correctly" {
    const A = std.testing.allocator;
    const mc_cols_max: usize = 8; // matches the bitstream COLS_MAX / matmul.zig
    // rows cover multiples and non-multiples of rows_per_block; cols cover
    // decode (1), single-group, and multi-group tiling incl. a trailing group=1.
    const shapes = [_][2]usize{
        .{ 2048, 1 }, .{ 6144, 1 }, .{ 1024, 1 }, .{ 2048, 13 }, .{ 6144, 13 },
        .{ 10, 1 },   .{ 10, 5 },   .{ 17, 9 },   .{ 8, 8 },     .{ 1, 1 },
        .{ 7, 3 },    .{ 9, 9 },    .{ 64, 16 },  .{ 3, 17 },
    };
    for (shapes) |s| {
        const rows = s[0];
        const cols = s[1];
        const num_rb = q1a8.rowblocksFor(rows);
        const dst_ref = try A.alloc(u8, rows * cols * 4);
        defer A.free(dst_ref);
        const dst_opt = try A.alloc(u8, rows * cols * 4);
        defer A.free(dst_opt);
        @memset(dst_ref, 0xAA);
        @memset(dst_opt, 0xAA);
        var col0: usize = 0;
        while (col0 < cols) {
            const group = @min(mc_cols_max, cols - col0);
            const result = try A.alloc(u8, num_rb * group * result_bytes_per_rb);
            @memset(result, 0xEE); // pad-lane sentinel; must never be copied out
            for (0..group) |c| {
                for (0..num_rb) |rb| {
                    const valid = @min(q1a8.rows_per_block, rows - rb * q1a8.rows_per_block);
                    const chunk = (rb * group + c) * result_bytes_per_rb;
                    for (0..valid) |lane| {
                        const grow = rb * q1a8.rows_per_block + lane;
                        const idx: u32 = @intCast((col0 + c) * rows + grow);
                        std.mem.writeInt(u32, result[chunk + lane * 4 ..][0..4], idx, .little);
                    }
                }
            }
            gatherResultsRef(dst_ref, result, rows, group, col0, num_rb);
            gatherResults(dst_opt, result, rows, group, col0, num_rb);
            A.free(result);
            col0 += group;
        }
        try std.testing.expectEqualSlices(u8, dst_ref, dst_opt);
        // Every dst slot must equal its own flat index (correct placement, no pad).
        for (0..rows * cols) |i| {
            try std.testing.expectEqual(@as(u32, @intCast(i)), std.mem.readInt(u32, dst_opt[i * 4 ..][0..4], .little));
        }
    }
}
