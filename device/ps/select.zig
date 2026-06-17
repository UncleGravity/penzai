const std = @import("std");

pub const SelectError = error{
    InvalidLength,
};

const lanes = 8;
const VecF = @Vector(lanes, f32);
const VecI = @Vector(lanes, i32);

/// f32 argmax along each contiguous row (ggml `GGML_OP_ARGMAX`): reduce `cols`
/// elements per row over `rows` rows, writing one little-endian i32 index per row.
///
/// Reads the source as an aligned `f32` view and reduces with `@Vector` so the ARM
/// PS core vectorizes it — the earlier `readInt`-per-element loop ran ~12x slower
/// than a `@memcpy` over the same buffer (34 vs 423 MiB/s on a 151k-vocab logits
/// row), which made on-device greedy cost more than the logits download it saved.
///
/// Tie-breaking matches `ggml_vec_argmax_f32`: the LAST index holding the maximum
/// wins. NaNs are ignored (a NaN is never `>=` the running max), which agrees with
/// ggml's `MAX` ordering for every case except all-NaN, where both return 0.
///
/// The source must be 4-byte aligned; the runtime's heap allocations are 64-byte
/// aligned and the logits view starts on a row boundary, so this always holds.
pub fn argmaxF32Bytes(src: []const u8, dst: []u8, rows: usize, cols: usize) SelectError!void {
    if (rows == 0 or cols == 0) return error.InvalidLength;
    const elements = std.math.mul(usize, rows, cols) catch return error.InvalidLength;
    const src_len = std.math.mul(usize, elements, @sizeOf(f32)) catch return error.InvalidLength;
    const dst_len = std.math.mul(usize, rows, @sizeOf(i32)) catch return error.InvalidLength;
    if (src.len != src_len or dst.len != dst_len) return error.InvalidLength;

    const floats: [*]const f32 = @ptrCast(@alignCast(src.ptr));
    for (0..rows) |r| {
        writeI32(dst, r, argmaxRow(floats[r * cols ..][0..cols]));
    }
}

fn argmaxRow(row: []const f32) i32 {
    var i: usize = 0;
    var best_v: f32 = -std.math.inf(f32);
    var best_i: i32 = 0;

    if (row.len >= lanes) {
        var vmax: VecF = @splat(-std.math.inf(f32));
        var vidx: VecI = @splat(0);
        var lane_idx: VecI = .{ 0, 1, 2, 3, 4, 5, 6, 7 };
        const stride: VecI = @splat(@intCast(lanes));
        while (i + lanes <= row.len) : (i += lanes) {
            const chunk: VecF = row[i..][0..lanes].*;
            // `>=` so a later equal value overwrites within a lane → last-wins.
            const take = chunk >= vmax;
            vmax = @select(f32, take, chunk, vmax);
            vidx = @select(i32, take, lane_idx, vidx);
            lane_idx += stride;
        }
        // Each lane holds its column's max and the last index achieving it; pick the
        // global max and, among ties, the largest index → global last-max.
        const max_arr: [lanes]f32 = vmax;
        const idx_arr: [lanes]i32 = vidx;
        for (max_arr, idx_arr) |v, ix| {
            if (v > best_v or (v == best_v and ix > best_i)) {
                best_v = v;
                best_i = ix;
            }
        }
    }

    // Scalar tail (also the whole row when cols < lanes). `>=` keeps last-wins, and
    // tail indices exceed every SIMD index so a tie here correctly wins.
    while (i < row.len) : (i += 1) {
        const v = row[i];
        if (v >= best_v) {
            best_v = v;
            best_i = @intCast(i);
        }
    }
    return best_i;
}

fn writeI32(bytes: []u8, index: usize, value: i32) void {
    std.mem.writeInt(i32, bytes[index * @sizeOf(i32) ..][0..4], value, .little);
}

fn expectArgmax(values: []const f32, expected: i32) !void {
    var buf: [4096]f32 = undefined;
    @memcpy(buf[0..values.len], values);
    const src = std.mem.sliceAsBytes(buf[0..values.len]); // f32-aligned bytes
    var dst: [4]u8 = undefined;
    try argmaxF32Bytes(src, &dst, 1, values.len);
    try std.testing.expectEqual(expected, std.mem.readInt(i32, &dst, .little));
}

test "argmax picks the maximum index" {
    try expectArgmax(&.{ 1, 3, 2 }, 1);
    try expectArgmax(&.{ -5, -1, -9 }, 1);
    try expectArgmax(&.{7}, 0);
}

test "argmax tie-break matches ggml: last maximum wins (scalar tail)" {
    try expectArgmax(&.{ 5, 3, 5, 2 }, 2);
    try expectArgmax(&.{ 9, 9, 9 }, 2);
    try expectArgmax(&.{ 4, 4, 1 }, 1);
}

test "argmax skips NaN like ggml MAX ordering" {
    const nan = std.math.nan(f32);
    try expectArgmax(&.{ nan, 2, 1 }, 1);
    try expectArgmax(&.{ 1, nan, 3 }, 2);
}

test "argmax over the SIMD path (cols >= lanes), incl. cross-lane and in-lane ties" {
    // 20 elements: exercises the vector body (16) + scalar tail (4).
    // max 9 at idx 3 (lane 3) and idx 11 (lane 3 again) and idx 18 (tail).
    var v = [_]f32{ 0, 1, 2, 9, 4, 5, 6, 7, 8, 1, 2, 9, 3, 4, 5, 6, 7, 8, 9, 0 };
    try expectArgmax(&v, 18); // last 9 is at index 18 (tail)
    // remove the tail max → last 9 now at idx 11 (in-lane, second occurrence)
    v[18] = 0;
    try expectArgmax(&v, 11);
    // cross-lane tie only inside the vector body: 9 at idx 3 and idx 5.
    const w = [_]f32{ 1, 1, 1, 9, 1, 9, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0 };
    try expectArgmax(&w, 5);
}

test "argmax handles multiple rows independently" {
    var data = [_]f32{ 1, 2, 9, 8, 0, 0 }; // row0 max@2, row1 max@0
    const src = std.mem.sliceAsBytes(data[0..]);
    var dst: [8]u8 = undefined;
    try argmaxF32Bytes(src, &dst, 2, 3);
    try std.testing.expectEqual(@as(i32, 2), std.mem.readInt(i32, dst[0..4], .little));
    try std.testing.expectEqual(@as(i32, 0), std.mem.readInt(i32, dst[4..8], .little));
}

test "argmax rejects mismatched lengths" {
    var src: [12]u8 = undefined;
    var dst: [4]u8 = undefined;
    try std.testing.expectError(error.InvalidLength, argmaxF32Bytes(&src, &dst, 1, 4)); // src too short
    try std.testing.expectError(error.InvalidLength, argmaxF32Bytes(&src, dst[0..0], 0, 3)); // empty
    var dst2: [8]u8 = undefined;
    try std.testing.expectError(error.InvalidLength, argmaxF32Bytes(&src, &dst2, 1, 3)); // dst too long
}
