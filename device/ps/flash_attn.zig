const std = @import("std");

pub const FlashAttnError = error{
    InvalidLength,
    InvalidShape,
    InvalidParam,
};

pub const max_head_dim = 256;

pub const Params = struct {
    head_dim_q: u32,
    head_dim_v: u32,
    n_heads: u32,
    n_head_kv: u32,
    n_kv: u32,
    n_tokens: u32,
    scale: f32,
    q_nb1: u64,
    q_nb2: u64,
    k_nb1: u64,
    k_nb2: u64,
    v_nb1: u64,
    v_nb2: u64,
    mask_nb1: u64,
    dst_nb1: u64,
    dst_nb2: u64,
};

pub fn runBytes(
    q_data: []const u8,
    k_data: []const u8,
    v_data: []const u8,
    mask_data: ?[]const u8,
    dst_data: []u8,
    params: Params,
) FlashAttnError!void {
    try validateInputs(q_data, k_data, v_data, mask_data, dst_data, params);

    const head_dim_q: usize = @intCast(params.head_dim_q);
    const head_dim_v: usize = @intCast(params.head_dim_v);
    const n_heads: usize = @intCast(params.n_heads);
    const n_head_kv: usize = @intCast(params.n_head_kv);
    const n_kv: usize = @intCast(params.n_kv);
    const n_tokens: usize = @intCast(params.n_tokens);
    const q_nb1: usize = @intCast(params.q_nb1);
    const q_nb2: usize = @intCast(params.q_nb2);
    const k_nb1: usize = @intCast(params.k_nb1);
    const k_nb2: usize = @intCast(params.k_nb2);
    const v_nb1: usize = @intCast(params.v_nb1);
    const v_nb2: usize = @intCast(params.v_nb2);
    const mask_nb1: usize = @intCast(params.mask_nb1);
    const dst_nb1: usize = @intCast(params.dst_nb1);
    const dst_nb2: usize = @intCast(params.dst_nb2);
    const head_ratio = n_heads / n_head_kv;

    var q_row: [max_head_dim]f32 = undefined;
    var acc: [max_head_dim]f32 = undefined;

    for (0..n_tokens) |token| {
        const mask_row = if (mask_data) |mask| mask[token * mask_nb1 ..] else null;
        for (0..n_heads) |head| {
            const kv_head = head / head_ratio;
            const q_base = token * q_nb1 + head * q_nb2;
            for (0..head_dim_q) |d| {
                q_row[d] = readF32(q_data, q_base + d * @sizeOf(f32));
            }

            var max_score = -std.math.inf(f32);
            var sum: f32 = 0;
            @memset(acc[0..head_dim_v], 0);

            for (0..n_kv) |kv| {
                var mask_value: f32 = 0;
                if (mask_row) |row| {
                    mask_value = readF16(row, kv * @sizeOf(f16));
                    if (!std.math.isFinite(mask_value) and mask_value < 0) continue;
                }

                const k_base = kv * k_nb1 + kv_head * k_nb2;
                var score: f32 = 0;
                for (0..head_dim_q) |d| {
                    score += q_row[d] * readF16(k_data, k_base + d * @sizeOf(f16));
                }
                score = score * params.scale + mask_value;

                const old_max = max_score;
                var max_scale: f32 = 1;
                var value_scale: f32 = 1;
                if (score > max_score) {
                    max_score = score;
                    max_scale = @exp(old_max - max_score);
                    for (0..head_dim_v) |d| {
                        acc[d] *= max_scale;
                    }
                } else {
                    value_scale = @exp(score - max_score);
                }

                const v_base = kv * v_nb1 + kv_head * v_nb2;
                for (0..head_dim_v) |d| {
                    acc[d] += value_scale * readF16(v_data, v_base + d * @sizeOf(f16));
                }
                sum = sum * max_scale + value_scale;
            }

            const dst_base = head * dst_nb1 + token * dst_nb2;
            const inv_sum: f32 = if (sum == 0) 0 else 1.0 / sum;
            for (0..head_dim_v) |d| {
                writeF32(dst_data, dst_base + d * @sizeOf(f32), acc[d] * inv_sum);
            }
        }
    }
}

fn validateInputs(q: []const u8, k: []const u8, v: []const u8, mask: ?[]const u8, dst: []const u8, params: Params) FlashAttnError!void {
    if (params.head_dim_q == 0 or params.head_dim_v == 0 or params.n_heads == 0 or params.n_head_kv == 0 or params.n_kv == 0 or params.n_tokens == 0) {
        return error.InvalidShape;
    }
    if (params.head_dim_q > max_head_dim or params.head_dim_v > max_head_dim) return error.InvalidShape;
    if (params.n_heads % params.n_head_kv != 0) return error.InvalidShape;
    if (!std.math.isFinite(params.scale)) return error.InvalidParam;

    const q_row_bytes = try checkedMul(params.head_dim_q, @sizeOf(f32));
    const k_row_bytes = try checkedMul(params.head_dim_q, @sizeOf(f16));
    const v_row_bytes = try checkedMul(params.head_dim_v, @sizeOf(f16));
    const dst_row_bytes = try checkedMul(params.head_dim_v, @sizeOf(f32));
    const mask_row_bytes = try checkedMul(params.n_kv, @sizeOf(f16));

    try validateStride(params.q_nb1, q_row_bytes);
    try validateStride(params.q_nb2, q_row_bytes);
    try validateStride(params.k_nb1, k_row_bytes);
    try validateStride(params.k_nb2, k_row_bytes);
    try validateStride(params.v_nb1, v_row_bytes);
    try validateStride(params.v_nb2, v_row_bytes);
    try validateStride(params.dst_nb1, dst_row_bytes);
    try validateStride(params.dst_nb2, dst_row_bytes);

    if (q.len < try requiredSpan(q_row_bytes, params.n_tokens, params.n_heads, params.q_nb1, params.q_nb2)) return error.InvalidLength;
    if (k.len < try requiredSpan(k_row_bytes, params.n_kv, params.n_head_kv, params.k_nb1, params.k_nb2)) return error.InvalidLength;
    if (v.len < try requiredSpan(v_row_bytes, params.n_kv, params.n_head_kv, params.v_nb1, params.v_nb2)) return error.InvalidLength;
    if (dst.len < try requiredSpan(dst_row_bytes, params.n_heads, params.n_tokens, params.dst_nb1, params.dst_nb2)) return error.InvalidLength;
    if (mask) |mask_bytes| {
        try validateStride(params.mask_nb1, mask_row_bytes);
        if (mask_bytes.len < try requiredSpan(mask_row_bytes, params.n_tokens, 1, params.mask_nb1, mask_row_bytes)) return error.InvalidLength;
    }
}

fn validateStride(stride: u64, row_bytes: usize) FlashAttnError!void {
    if (stride < row_bytes) return error.InvalidShape;
    if (stride > std.math.maxInt(usize)) return error.InvalidLength;
}

fn requiredSpan(row_bytes: usize, ne1_raw: u32, ne2_raw: u32, nb1_raw: u64, nb2_raw: u64) FlashAttnError!usize {
    if (row_bytes == 0 or ne1_raw == 0 or ne2_raw == 0) return error.InvalidShape;
    const ne1: usize = @intCast(ne1_raw);
    const ne2: usize = @intCast(ne2_raw);
    const nb1 = std.math.cast(usize, nb1_raw) orelse return error.InvalidLength;
    const nb2 = std.math.cast(usize, nb2_raw) orelse return error.InvalidLength;
    var span = row_bytes;
    span = try checkedAdd(span, try checkedMul(ne1 - 1, nb1));
    span = try checkedAdd(span, try checkedMul(ne2 - 1, nb2));
    return span;
}

fn checkedAdd(a: usize, b: usize) FlashAttnError!usize {
    return std.math.add(usize, a, b) catch return error.InvalidLength;
}

fn checkedMul(a: anytype, b: anytype) FlashAttnError!usize {
    const lhs = std.math.cast(usize, a) orelse return error.InvalidLength;
    const rhs = std.math.cast(usize, b) orelse return error.InvalidLength;
    return std.math.mul(usize, lhs, rhs) catch return error.InvalidLength;
}

fn readF16(bytes: []const u8, offset: usize) f32 {
    const half: f16 = @bitCast(std.mem.readInt(u16, bytes[offset..][0..2], .little));
    return @floatCast(half);
}

fn writeF16(bytes: []u8, offset: usize, value: f16) void {
    std.mem.writeInt(u16, bytes[offset..][0..2], @bitCast(value), .little);
}

fn readF32(bytes: []const u8, offset: usize) f32 {
    return @bitCast(std.mem.readInt(u32, bytes[offset..][0..4], .little));
}

fn writeF32(bytes: []u8, offset: usize, value: f32) void {
    std.mem.writeInt(u32, bytes[offset..][0..4], @bitCast(value), .little);
}

fn expectApprox(expected: f32, actual: f32, tolerance: f32) !void {
    try std.testing.expect(@abs(expected - actual) <= tolerance);
}

fn baseParams() Params {
    return .{
        .head_dim_q = 2,
        .head_dim_v = 2,
        .n_heads = 1,
        .n_head_kv = 1,
        .n_kv = 2,
        .n_tokens = 1,
        .scale = 1,
        .q_nb1 = 2 * @sizeOf(f32),
        .q_nb2 = 2 * @sizeOf(f32),
        .k_nb1 = 2 * @sizeOf(f16),
        .k_nb2 = 4 * @sizeOf(f16),
        .v_nb1 = 2 * @sizeOf(f16),
        .v_nb2 = 4 * @sizeOf(f16),
        .mask_nb1 = 2 * @sizeOf(f16),
        .dst_nb1 = 2 * @sizeOf(f32),
        .dst_nb2 = 2 * @sizeOf(f32),
    };
}

fn writeBaseQkv(q: []u8, k: []u8, v: []u8) void {
    writeF32(q, 0, 1);
    writeF32(q, 4, 0);
    writeF16(k, 0, 1);
    writeF16(k, 2, 0);
    writeF16(k, 4, 0);
    writeF16(k, 6, 1);
    writeF16(v, 0, 10);
    writeF16(v, 2, 20);
    writeF16(v, 4, 30);
    writeF16(v, 6, 40);
}

test "flash attention computes softmax qk times v" {
    var q: [8]u8 = undefined;
    var k: [8]u8 = undefined;
    var v: [8]u8 = undefined;
    var dst: [8]u8 = undefined;
    writeBaseQkv(&q, &k, &v);

    try runBytes(&q, &k, &v, null, &dst, baseParams());

    const w0 = @exp(@as(f32, 1)) / (@exp(@as(f32, 1)) + 1.0);
    const w1 = 1.0 - w0;
    try expectApprox(w0 * 10 + w1 * 30, readF32(&dst, 0), 0.00001);
    try expectApprox(w0 * 20 + w1 * 40, readF32(&dst, 4), 0.00001);
}

test "flash attention skips negative infinity mask entries" {
    var q: [8]u8 = undefined;
    var k: [8]u8 = undefined;
    var v: [8]u8 = undefined;
    var mask: [4]u8 = undefined;
    var dst: [8]u8 = undefined;
    writeBaseQkv(&q, &k, &v);
    writeF16(&mask, 0, 0);
    writeF16(&mask, 2, -std.math.inf(f16));

    try runBytes(&q, &k, &v, &mask, &dst, baseParams());

    try expectApprox(10, readF32(&dst, 0), 0.000001);
    try expectApprox(20, readF32(&dst, 4), 0.000001);
}

test "flash attention maps grouped query heads onto shared kv heads" {
    var q: [16]u8 = undefined;
    var k: [8]u8 = undefined;
    var v: [8]u8 = undefined;
    var dst: [16]u8 = undefined;
    writeBaseQkv(q[0..8], &k, &v);
    writeF32(&q, 8, 0);
    writeF32(&q, 12, 1);
    var params = baseParams();
    params.n_heads = 2;
    params.q_nb2 = 8;
    params.dst_nb1 = 8;
    params.dst_nb2 = 16;

    try runBytes(&q, &k, &v, null, &dst, params);

    const w0_head0 = @exp(@as(f32, 1)) / (@exp(@as(f32, 1)) + 1.0);
    const w1_head0 = 1.0 - w0_head0;
    const w1_head1 = w0_head0;
    const w0_head1 = w1_head0;
    try expectApprox(w0_head0 * 10 + w1_head0 * 30, readF32(&dst, 0), 0.00001);
    try expectApprox(w0_head0 * 20 + w1_head0 * 40, readF32(&dst, 4), 0.00001);
    try expectApprox(w0_head1 * 10 + w1_head1 * 30, readF32(&dst, 8), 0.00001);
    try expectApprox(w0_head1 * 20 + w1_head1 * 40, readF32(&dst, 12), 0.00001);
}
