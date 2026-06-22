const std = @import("std");

pub const ActivationError = error{
    InvalidLength,
};

// Vectorized over fixed-width lanes (portable @Vector → NEON on the A53).
//
// The transcendental is the whole cost: `@exp` on a @Vector lowers to one libm
// `exp` call PER LANE, so the old "vector" path ran at scalar speed (~87 MiB/s).
// expVec replaces it with a true vector expf — range reduction + a degree-5
// polynomial, all @Vector arithmetic (FMUL/FADD/FRINTN → NEON, no libcall) — so a
// whole lane group is one transcendental's worth of work. siluScalar stays on libm
// exp as the exact reference (tests + the sub-lane remainder); the vector path
// tracks it to ~1 ULP, far inside this 1-bit model's quantization noise.
const lanes = 16;
const Vf32 = @Vector(lanes, f32);

// exp(x) for a vector lane group (Cephes single-precision algorithm): n=round(x/ln2),
// r=x-n·ln2 ∈ [-ln2/2, ln2/2], exp(r) via Horner poly, then scale by 2^n built from
// the float exponent bits. x is clamped so the result stays finite in f32.
inline fn expVec(x_in: Vf32) Vf32 {
    const log2ef: Vf32 = @splat(1.44269504088896341);
    const c1: Vf32 = @splat(0.693359375); // ln2, hi part
    const c2: Vf32 = @splat(-2.12194440e-4); // ln2, lo part (c1 + c2 ≈ ln2)
    const half: Vf32 = @splat(0.5);
    const one: Vf32 = @splat(1.0);
    // exp(88) ≈ 1.65e38 < FLT_MAX; below ~-88 the result underflows to 0. Clamping
    // here keeps SiLU correct at the tails (large +x → x, large -x → 0).
    const x0 = @min(@max(x_in, @as(Vf32, @splat(-88.0))), @as(Vf32, @splat(88.0)));

    const fx = @floor(x0 * log2ef + half); // n as float, round-to-nearest
    const r = (x0 - fx * c1) - fx * c2; // reduced argument

    const z = r * r;
    var y: Vf32 = @splat(1.9875691500e-4);
    y = y * r + @as(Vf32, @splat(1.3981999507e-3));
    y = y * r + @as(Vf32, @splat(8.3334519073e-3));
    y = y * r + @as(Vf32, @splat(4.1665795894e-2));
    y = y * r + @as(Vf32, @splat(1.6666665459e-1));
    y = y * r + @as(Vf32, @splat(5.0000001201e-1));
    y = y * z + r + one;

    // 2^n by writing n into the f32 exponent field. n+127 ∈ [0,254] over the clamp.
    const n: @Vector(lanes, i32) = @intFromFloat(fx);
    const biased: @Vector(lanes, u32) = @intCast(n + @as(@Vector(lanes, i32), @splat(127)));
    const pow2n: Vf32 = @bitCast(biased << @as(@Vector(lanes, u5), @splat(23)));
    return y * pow2n;
}

inline fn siluVec(x: Vf32) Vf32 {
    const one: Vf32 = @splat(1.0);
    return x / (one + expVec(-x));
}

pub fn siluScalar(x: f32) f32 {
    return x / (1.0 + @exp(-x));
}

pub fn siluF32(input: []const f32, dst: []f32) ActivationError!void {
    if (dst.len != input.len) return error.InvalidLength;
    for (input, dst) |x, *out| {
        out.* = siluScalar(x);
    }
}

pub fn swigluF32(gate: []const f32, up: []const f32, dst: []f32) ActivationError!void {
    if (up.len != gate.len or dst.len != gate.len) return error.InvalidLength;
    for (gate, up, dst) |g, u, *out| {
        out.* = siluScalar(g) * u;
    }
}

pub fn siluBytes(input: []const u8, dst: []u8) ActivationError!void {
    const count = try f32Count(input);
    if (dst.len != input.len) return error.InvalidLength;
    const in = std.mem.bytesAsSlice(f32, input);
    const out = std.mem.bytesAsSlice(f32, dst);
    var i: usize = 0;
    while (i + lanes <= count) : (i += lanes) {
        const xv: Vf32 = in[i..][0..lanes].*;
        out[i..][0..lanes].* = siluVec(xv);
    }
    while (i < count) : (i += 1) out[i] = siluScalar(in[i]);
}

pub fn swigluBytes(gate: []const u8, up: []const u8, dst: []u8) ActivationError!void {
    const count = try f32Count(gate);
    if (up.len != gate.len or dst.len != gate.len) return error.InvalidLength;
    const g = std.mem.bytesAsSlice(f32, gate);
    const u = std.mem.bytesAsSlice(f32, up);
    const out = std.mem.bytesAsSlice(f32, dst);
    var i: usize = 0;
    while (i + lanes <= count) : (i += lanes) {
        const gv: Vf32 = g[i..][0..lanes].*;
        const uv: Vf32 = u[i..][0..lanes].*;
        out[i..][0..lanes].* = siluVec(gv) * uv;
    }
    while (i < count) : (i += 1) out[i] = siluScalar(g[i]) * u[i];
}

fn f32Count(bytes: []const u8) ActivationError!usize {
    if (bytes.len % @sizeOf(f32) != 0) return error.InvalidLength;
    return bytes.len / @sizeOf(f32);
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

test "silu applies sigmoid-weighted linear activation" {
    const input = [_]f32{ 0, 1, -1 };
    var dst: [3]f32 = undefined;

    try siluF32(&input, &dst);

    try expectApprox(0, dst[0], 0.000001);
    try expectApprox(0.7310586, dst[1], 0.000001);
    try expectApprox(-0.26894143, dst[2], 0.000001);
}

test "swiglu multiplies up projection by silu gate" {
    const gate = [_]f32{ 0, 1 };
    const up = [_]f32{ 10, 2 };
    var dst: [2]f32 = undefined;

    try swigluF32(&gate, &up, &dst);

    try expectApprox(0, dst[0], 0.000001);
    try expectApprox(1.4621172, dst[1], 0.000001);
}

test "activation byte wrappers use little-endian f32 values" {
    var gate: [8]u8 = undefined;
    var up: [8]u8 = undefined;
    var dst: [8]u8 = undefined;
    writeF32(&gate, 0, 0);
    writeF32(&gate, 1, 1);
    writeF32(&up, 0, 10);
    writeF32(&up, 1, 2);

    try swigluBytes(&gate, &up, &dst);

    try expectApprox(0, readF32(&dst, 0), 0.000001);
    try expectApprox(1.4621172, readF32(&dst, 1), 0.000001);
}

test "swiglu/silu vector path tracks libm scalar within tolerance over a wide span" {
    const n = 200; // > lanes (16), with remainder; spans the range reduction + clamp
    var gate: [n * @sizeOf(f32)]u8 = undefined;
    var up: [n * @sizeOf(f32)]u8 = undefined;
    var dst: [n * @sizeOf(f32)]u8 = undefined;
    var prng = std.Random.DefaultPrng.init(0x5111);
    const rnd = prng.random();
    for (0..n) |i| {
        writeF32(&gate, i, rnd.float(f32) * 60 - 30); // [-30, 30]: positive/negative saturation + mid
        writeF32(&up, i, rnd.float(f32) * 2 - 1);
    }

    // The vector path uses a polynomial exp (no per-lane libm call), so it is no
    // longer bit-identical to siluScalar — it tracks it to ~1 ULP. The tolerance is
    // ~100x the approximation error: tight enough to catch a broken poly/range
    // reduction, loose enough to never flake.
    const tol = struct {
        fn of(want: f32) f32 {
            return 1e-4 + @abs(want) * 1e-5;
        }
    }.of;

    try siluBytes(&gate, &dst);
    for (0..n) |i| {
        const want = siluScalar(readF32(&gate, i));
        try expectApprox(want, readF32(&dst, i), tol(want));
    }

    try swigluBytes(&gate, &up, &dst);
    for (0..n) |i| {
        const want = siluScalar(readF32(&gate, i)) * readF32(&up, i);
        try expectApprox(want, readF32(&dst, i), tol(want));
    }
}
