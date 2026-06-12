const std = @import("std");

pub const RopeError = error{
    InvalidLength,
    InvalidShape,
    InvalidParams,
    InvalidPosition,
};

pub const Mode = enum {
    normal,
    neox,
};

// cos/sin depend on (position, dim) only, not the head, so the byte path computes
// them once per token and reuses across heads — an n_heads× cut in transcendentals.
// The neox rotation (contiguous halves) vectorizes; portable @Vector → NEON. Both
// reuse the exact scalar theta/cos/sin values, so the result is bit-exact.
const max_head_dim = 256;
const max_half = max_head_dim / 2;
const lanes = 16;
const Vf32 = @Vector(lanes, f32);

pub const Params = struct {
    head_dim: u32,
    n_heads: u32,
    n_tokens: u32,
    n_dims: u32,
    mode: Mode,
    n_ctx_orig: u32,
    freq_base: f32,
    freq_scale: f32,
    ext_factor: f32,
    attn_factor: f32,
    beta_fast: f32,
    beta_slow: f32,
};

const Derived = struct {
    inv_n_dims: f32,
    log_freq_base: f32,
    use_yarn: bool,
    corr_low: f32,
    corr_high: f32,
    mscale: f32,
    freq_scale: f32,
    ext_factor: f32,
};

pub fn applyF32(input: []const f32, positions: []const i32, dst: []f32, params: Params) RopeError!void {
    const derived = try derive(params);
    const head_dim: usize = @intCast(params.head_dim);
    const n_heads: usize = @intCast(params.n_heads);
    const n_tokens: usize = @intCast(params.n_tokens);
    const n_dims: usize = @intCast(params.n_dims);
    const elements = try elementCount(params);
    if (input.len != elements or dst.len != elements or positions.len != n_tokens) return error.InvalidLength;

    for (0..n_tokens) |t| {
        const pos = try positionF32(positions[t]);
        for (0..n_heads) |h| {
            const offset = (t * n_heads + h) * head_dim;
            applyOne(input[offset..][0..head_dim], dst[offset..][0..head_dim], pos, n_dims, params.mode, derived);
        }
    }
}

pub fn applyBytes(input: []const u8, positions: []const u8, dst: []u8, params: Params) RopeError!void {
    const derived = try derive(params);
    const head_dim: usize = @intCast(params.head_dim);
    const n_heads: usize = @intCast(params.n_heads);
    const n_tokens: usize = @intCast(params.n_tokens);
    const n_dims: usize = @intCast(params.n_dims);
    if (params.head_dim > max_head_dim) return error.InvalidShape;
    const elements = try elementCount(params);
    const input_bytes = try checkedMul(elements, @sizeOf(f32));
    const position_bytes = try checkedMul(n_tokens, @sizeOf(i32));
    if (input.len != input_bytes or dst.len != input_bytes or positions.len != position_bytes) return error.InvalidLength;

    const half = n_dims / 2;
    const in = std.mem.bytesAsSlice(f32, input);
    const out = std.mem.bytesAsSlice(f32, dst);

    var cos_t: [max_half]f32 = undefined;
    var sin_t: [max_half]f32 = undefined;
    for (0..n_tokens) |t| {
        const pos = try positionF32(readI32(positions, t));
        for (0..half) |j| {
            const theta = thetaFor(pos, 2 * j, derived);
            cos_t[j] = @cos(theta) * derived.mscale;
            sin_t[j] = @sin(theta) * derived.mscale;
        }
        for (0..n_heads) |h| {
            const base = (t * n_heads + h) * head_dim;
            switch (params.mode) {
                .normal => rotateNormal(in, out, base, half, n_dims, head_dim, &cos_t, &sin_t),
                .neox => rotateNeox(in, out, base, half, n_dims, head_dim, &cos_t, &sin_t),
            }
        }
    }
}

/// Normal pairing rotates adjacent lanes (2j, 2j+1) — interleaved, so scalar; the
/// win here is the precomputed cos/sin (no transcendentals in this loop).
fn rotateNormal(in: []align(1) const f32, out: []align(1) f32, base: usize, half: usize, n_dims: usize, head_dim: usize, cos_t: []const f32, sin_t: []const f32) void {
    for (0..half) |j| {
        const x0 = in[base + 2 * j];
        const x1 = in[base + 2 * j + 1];
        out[base + 2 * j] = x0 * cos_t[j] - x1 * sin_t[j];
        out[base + 2 * j + 1] = x0 * sin_t[j] + x1 * cos_t[j];
    }
    var i = n_dims;
    while (i < head_dim) : (i += 1) out[base + i] = in[base + i];
}

/// Neox pairing rotates (j, j+half) — two contiguous halves, so it vectorizes.
/// In-place safe: the two halves are disjoint and each lane group loads before storing.
fn rotateNeox(in: []align(1) const f32, out: []align(1) f32, base: usize, half: usize, n_dims: usize, head_dim: usize, cos_t: []const f32, sin_t: []const f32) void {
    var j: usize = 0;
    while (j + lanes <= half) : (j += lanes) {
        const x0: Vf32 = in[base + j ..][0..lanes].*;
        const x1: Vf32 = in[base + half + j ..][0..lanes].*;
        const cv: Vf32 = cos_t[j..][0..lanes].*;
        const sv: Vf32 = sin_t[j..][0..lanes].*;
        out[base + j ..][0..lanes].* = x0 * cv - x1 * sv;
        out[base + half + j ..][0..lanes].* = x0 * sv + x1 * cv;
    }
    while (j < half) : (j += 1) {
        const x0 = in[base + j];
        const x1 = in[base + half + j];
        out[base + j] = x0 * cos_t[j] - x1 * sin_t[j];
        out[base + half + j] = x0 * sin_t[j] + x1 * cos_t[j];
    }
    var i = n_dims;
    while (i < head_dim) : (i += 1) out[base + i] = in[base + i];
}

fn applyOne(input: []const f32, dst: []f32, pos: f32, n_dims: usize, mode: Mode, derived: Derived) void {
    var i: usize = 0;
    while (i < n_dims) : (i += 2) {
        const theta = thetaFor(pos, i, derived);
        const c = @cos(theta) * derived.mscale;
        const s = @sin(theta) * derived.mscale;
        const pair = ropePair(i, n_dims, mode);
        const x0 = input[pair.i0];
        const x1 = input[pair.i1];
        dst[pair.i0] = x0 * c - x1 * s;
        dst[pair.i1] = x0 * s + x1 * c;
    }
    while (i < input.len) : (i += 1) {
        dst[i] = input[i];
    }
}

fn thetaFor(pos: f32, i: usize, derived: Derived) f32 {
    const theta_extrap = pos * @exp(derived.log_freq_base * -@as(f32, @floatFromInt(i)) * derived.inv_n_dims);
    if (!derived.use_yarn) return theta_extrap * derived.freq_scale;

    const theta_interp = derived.freq_scale * theta_extrap;
    const ramp_mix = yarnRamp(derived.corr_low, derived.corr_high, i) * derived.ext_factor;
    return theta_interp * (1.0 - ramp_mix) + theta_extrap * ramp_mix;
}

fn derive(params: Params) RopeError!Derived {
    try validateParams(params);

    const n_dims_f: f32 = @floatFromInt(params.n_dims);
    var corr_low: f32 = 0;
    var corr_high: f32 = n_dims_f;
    var mscale = params.attn_factor;
    const use_yarn = params.ext_factor != 0;
    if (use_yarn) {
        if (params.beta_fast <= 0 or params.beta_slow <= 0) return error.InvalidParams;
        const ctx_orig: f32 = @floatFromInt(if (params.n_ctx_orig > 0) params.n_ctx_orig else 1);
        const start = @floor(yarnCorrDim(n_dims_f, ctx_orig, params.beta_fast, params.freq_base));
        const end = @ceil(yarnCorrDim(n_dims_f, ctx_orig, params.beta_slow, params.freq_base));
        corr_low = @max(0, start);
        corr_high = @min(n_dims_f - 1.0, end);
        mscale *= 1.0 + 0.1 * @log(1.0 / params.freq_scale);
    }

    return .{
        .inv_n_dims = 1.0 / n_dims_f,
        .log_freq_base = @log(params.freq_base),
        .use_yarn = use_yarn,
        .corr_low = corr_low,
        .corr_high = corr_high,
        .mscale = mscale,
        .freq_scale = params.freq_scale,
        .ext_factor = params.ext_factor,
    };
}

fn validateParams(params: Params) RopeError!void {
    if (params.head_dim == 0 or params.n_heads == 0 or params.n_tokens == 0) return error.InvalidShape;
    if (params.n_dims == 0 or params.n_dims > params.head_dim or params.n_dims % 2 != 0) return error.InvalidShape;
    if (params.freq_base <= 0 or params.freq_scale <= 0) return error.InvalidParams;
    if (!std.math.isFinite(params.freq_base) or
        !std.math.isFinite(params.freq_scale) or
        !std.math.isFinite(params.ext_factor) or
        !std.math.isFinite(params.attn_factor) or
        !std.math.isFinite(params.beta_fast) or
        !std.math.isFinite(params.beta_slow))
    {
        return error.InvalidParams;
    }
}

fn elementCount(params: Params) RopeError!usize {
    const heads = try checkedMul(params.head_dim, params.n_heads);
    return try checkedMul(heads, params.n_tokens);
}

fn positionF32(position: i32) RopeError!f32 {
    if (position < 0) return error.InvalidPosition;
    return @floatFromInt(position);
}

const Pair = struct {
    i0: usize,
    i1: usize,
};

fn ropePair(i: usize, n_dims: usize, mode: Mode) Pair {
    return switch (mode) {
        .normal => .{ .i0 = i, .i1 = i + 1 },
        .neox => .{ .i0 = i / 2, .i1 = i / 2 + n_dims / 2 },
    };
}

fn yarnCorrDim(n_dims: f32, n_ctx_orig: f32, n_rot: f32, base: f32) f32 {
    return n_dims * @log(n_ctx_orig / (n_rot * 2.0 * std.math.pi)) / (2.0 * @log(base));
}

fn yarnRamp(low: f32, high: f32, i: usize) f32 {
    const denom = @max(0.001, high - low);
    const y = (@as(f32, @floatFromInt(i / 2)) - low) / denom;
    return 1.0 - std.math.clamp(y, 0.0, 1.0);
}

fn checkedMul(a: anytype, b: anytype) RopeError!usize {
    const lhs = std.math.cast(usize, a) orelse return error.InvalidLength;
    const rhs = std.math.cast(usize, b) orelse return error.InvalidLength;
    return std.math.mul(usize, lhs, rhs) catch return error.InvalidLength;
}

fn readI32(bytes: []const u8, index: usize) i32 {
    return std.mem.readInt(i32, bytes[index * @sizeOf(i32) ..][0..4], .little);
}

fn writeI32(bytes: []u8, index: usize, value: i32) void {
    std.mem.writeInt(i32, bytes[index * @sizeOf(i32) ..][0..4], value, .little);
}

fn readF32(bytes: []const u8, index: usize) f32 {
    return @bitCast(std.mem.readInt(u32, bytes[index * @sizeOf(f32) ..][0..4], .little));
}

fn writeF32(bytes: []u8, index: usize, value: f32) void {
    std.mem.writeInt(u32, bytes[index * @sizeOf(f32) ..][0..4], @bitCast(value), .little);
}

fn expectApprox(expected: f32, actual: f32, tolerance: f32) !void {
    try std.testing.expect(@abs(expected - actual) <= tolerance);
}

fn normalParams(head_dim: u32, n_dims: u32) Params {
    return .{
        .head_dim = head_dim,
        .n_heads = 1,
        .n_tokens = 1,
        .n_dims = n_dims,
        .mode = .normal,
        .n_ctx_orig = 0,
        .freq_base = 10000,
        .freq_scale = 1,
        .ext_factor = 0,
        .attn_factor = 1,
        .beta_fast = 0,
        .beta_slow = 0,
    };
}

test "rope leaves values unchanged at position zero" {
    const input = [_]f32{ 1, 2, 3, 4 };
    const positions = [_]i32{0};
    var dst: [4]f32 = undefined;

    try applyF32(&input, &positions, &dst, normalParams(4, 4));

    for (input, dst) |expected, actual| {
        try expectApprox(expected, actual, 0.000001);
    }
}

test "rope rotates normal pairs by position" {
    const input = [_]f32{ 1, 2 };
    const positions = [_]i32{1};
    var dst: [2]f32 = undefined;

    try applyF32(&input, &positions, &dst, normalParams(2, 2));

    try expectApprox(1 * @cos(@as(f32, 1)) - 2 * @sin(@as(f32, 1)), dst[0], 0.000001);
    try expectApprox(1 * @sin(@as(f32, 1)) + 2 * @cos(@as(f32, 1)), dst[1], 0.000001);
}

test "rope supports neox pairing and partial rotation" {
    const input = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const positions = [_]i32{1};
    var dst: [6]f32 = undefined;
    var params = normalParams(6, 4);
    params.mode = .neox;

    try applyF32(&input, &positions, &dst, params);

    try expectApprox(1 * @cos(@as(f32, 1)) - 3 * @sin(@as(f32, 1)), dst[0], 0.000001);
    try expectApprox(2 * @cos(@as(f32, 0.01)) - 4 * @sin(@as(f32, 0.01)), dst[1], 0.000001);
    try expectApprox(1 * @sin(@as(f32, 1)) + 3 * @cos(@as(f32, 1)), dst[2], 0.000001);
    try expectApprox(2 * @sin(@as(f32, 0.01)) + 4 * @cos(@as(f32, 0.01)), dst[3], 0.000001);
    try expectApprox(5, dst[4], 0.000001);
    try expectApprox(6, dst[5], 0.000001);
}

test "rope byte wrapper reads positions and supports in-place rotation" {
    var values: [8]u8 = undefined;
    var positions: [4]u8 = undefined;
    writeF32(&values, 0, 1);
    writeF32(&values, 1, 0);
    writeI32(&positions, 0, 1);

    try applyBytes(&values, &positions, &values, normalParams(2, 2));

    try expectApprox(@cos(@as(f32, 1)), readF32(&values, 0), 0.000001);
    try expectApprox(@sin(@as(f32, 1)), readF32(&values, 1), 0.000001);
}

test "rope vectorized byte path is bit-exact vs scalar over multi-head wide rotation" {
    const head_dim = 40; // > 0 tail past n_dims
    const n_heads = 3; // shared cos/sin table across heads
    const n_tokens = 2;
    const n_dims = 36; // half = 18 -> vector (16) + remainder (2)
    const elems = head_dim * n_heads * n_tokens;

    var in_f: [elems]f32 = undefined;
    var in_b: [elems * @sizeOf(f32)]u8 = undefined;
    var positions: [n_tokens]i32 = undefined;
    var pos_b: [n_tokens * @sizeOf(i32)]u8 = undefined;
    var prng = std.Random.DefaultPrng.init(0x209E);
    const rnd = prng.random();
    for (0..elems) |i| {
        const v = rnd.float(f32) * 2 - 1;
        in_f[i] = v;
        writeF32(&in_b, i, v);
    }
    for (0..n_tokens) |t| {
        positions[t] = @intCast(t * 7 + 3);
        writeI32(&pos_b, t, positions[t]);
    }

    inline for (.{ Mode.normal, Mode.neox }) |mode| {
        var ref: [elems]f32 = undefined;
        var out_b: [elems * @sizeOf(f32)]u8 = undefined;
        var p = normalParams(head_dim, n_dims);
        p.n_heads = n_heads;
        p.n_tokens = n_tokens;
        p.mode = mode;

        try applyF32(&in_f, &positions, &ref, p);
        try applyBytes(&in_b, &pos_b, &out_b, p);

        for (0..elems) |i| try std.testing.expectEqual(ref[i], readF32(&out_b, i));
    }
}
