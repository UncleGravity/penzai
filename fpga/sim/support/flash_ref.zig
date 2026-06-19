//! Flash-attention golden reference — the cosim oracle for the PL flash kernel.
//!
//! Unlike matmul (int-exact sub-sums + ε fp scale), flash attention is fp all the
//! way down: f16 K/V widened to f32, f32 Q, dot-products, a softmax with exp and a
//! reciprocal, and a weighted V sum. There is no integer-exact gate — the whole
//! cosim is ε. For the ε to be *tight* (so it catches bugs rather than hiding them)
//! the reference must model the kernel's chosen exp/reciprocal approximation, not
//! `@exp`/`1.0/x`. Otherwise the ε would have to swallow the approximation gap and
//! every real datapath bug with it.
//!
//! The schedule mirrors `device/ps/flash_attn.zig` exactly: single-pass *online*
//! softmax (running max `m`, denominator `l`, output `acc` rescaled by exp(Δm)) over
//! the *real* KV extent. That PS kernel is the shipping, tested oracle; the PL
//! kernel and this ref both implement the same schedule, differing only in that
//! exp/recip are the hardware approximations modeled below. Internal datapath is
//! fp32 throughout (matches rtl/fp/ and the PS path; the bf16-accumulator area
//! lever was considered and rejected to avoid reopening numerics).

const std = @import("std");

// ===================== committed hardware approximation ======================
//
// LUT address bits. B=8 (256-entry table + linear interp) is the safe-and-free
// point: exp rel error ~2.5e-6, ~8 Kbit of table (~0.15% of one BRAM, or LUTRAM),
// no fmax/area cost, and the residual sits far below the fp32 truncation + f16 K/V
// rounding that dominate total error. Characterized in the tests below.
pub const exp_lut_bits: u5 = 8;
pub const recip_lut_bits: u5 = 8;

// ============================ exp, the long pole =============================
//
// Softmax feeds exp only values ≤ 0 (p = exp(score−m), m ≥ score; corr = exp(Δm),
// Δm ≤ 0), so output ∈ (0,1] and the integer exponent part is ≤ 0 — a right-shift /
// underflow-to-0, never overflow. HW recipe (all from existing rtl/fp/ cells + one
// table + one lerp):
//
//   y = x · log2e          one fp32 multiply              (y ≤ 0)
//   i = floor(y) ; f = y−i  exponent split                (i ≤ 0, f ∈ [0,1))
//   2^f ≈ lut[f]           BRAM table + linear interp     (result ∈ [1,2))
//   2^x = ldexp(2^f, i)    add i to the fp32 exponent; underflow ⇒ +0

const log2e: f32 = 1.4426950408889634;

/// 2^f for f ∈ [0,1) via a (2^B + 1)-entry table of 2^(k/2^B) with linear
/// interpolation — the HW LUT, modeled in the same shape silicon would use.
fn exp2Frac(comptime B: u5, f: f32) f32 {
    const N = 1 << B;
    const table = comptime blk: {
        @setEvalBranchQuota(100000);
        var t: [N + 1]f32 = undefined;
        for (&t, 0..) |*e, k| {
            e.* = @floatCast(std.math.pow(f64, 2.0, @as(f64, @floatFromInt(k)) / @as(f64, N)));
        }
        break :blk t;
    };
    const scaled = f * @as(f32, @floatFromInt(N));
    // Clamp the address to the last segment (idx+1 stays in range) — the HW LUT
    // address saturates; guards the f→1.0⁻ fp-rounding edge.
    const idx: usize = @min(@as(usize, @intFromFloat(@floor(scaled))), N - 1);
    const t = scaled - @floor(scaled);
    return table[idx] * (1.0 - t) + table[idx + 1] * t;
}

/// Hardware-modeled exp for softmax: exact on x = 0, flush to 0 on deep underflow.
pub fn softmaxExp(comptime B: u5, x: f32) f32 {
    if (x >= 0) return 1.0; // softmax only feeds x ≤ 0; x == 0 ⇒ exactly 1
    // exp(−87) ≈ 1.6e−38 (≈ smallest normal f32); below that the result underflows.
    // This guard also catches x = −inf (the first-iteration m − m_new), which the HW
    // exponent subtractor saturates to underflow ⇒ 0 — and avoids @intFromFloat(−inf).
    if (x <= -87.0) return 0.0;
    const y = x * log2e;
    const i = @floor(y);
    const f = y - i;
    const frac = exp2Frac(B, f);
    return std.math.ldexp(frac, @intFromFloat(i));
}

// ============================ reciprocal (1/l) ===============================
//
// l = Σ p ≥ 1 (the running-max term contributes exp(0)=1), so the input is ≥ 1, but
// the model handles any positive x. Same recipe as exp: reduce to the mantissa, LUT
// the reciprocal there, negate the exponent. 1/l is a single per-(head,token) output
// scale, so it is far less sensitive than exp — modeled to the same standard only so
// the reference stays bit-faithful to what silicon will do.

/// 1/s for s ∈ [0.5,1) via a (2^B + 1)-entry table + linear interp. frexp puts the
/// significand in [0.5,1); u = (s−0.5)·2 ∈ [0,1) indexes the table.
fn recipMant(comptime B: u5, s: f32) f32 {
    const N = 1 << B;
    const table = comptime blk: {
        @setEvalBranchQuota(100000);
        var t: [N + 1]f32 = undefined;
        for (&t, 0..) |*e, k| {
            const sk = 0.5 + 0.5 * (@as(f64, @floatFromInt(k)) / @as(f64, N));
            e.* = @floatCast(1.0 / sk);
        }
        break :blk t;
    };
    const u = (s - 0.5) * 2.0;
    const scaled = u * @as(f32, @floatFromInt(N));
    const idx: usize = @intFromFloat(@floor(scaled));
    const t = scaled - @floor(scaled);
    return table[idx] * (1.0 - t) + table[idx + 1] * t;
}

pub fn recip(comptime B: u5, x: f32) f32 {
    if (x <= 0) return 0;
    const fe = std.math.frexp(x); // x = significand · 2^exponent, significand ∈ [0.5,1)
    return std.math.ldexp(recipMant(B, fe.significand), -fe.exponent);
}

// ============================ attention schedule =============================
//
// Mirrors device/ps/flash_attn.zig (per-head online softmax over the real KV extent).
// Generic over the exp/recip functions so the same schedule yields both the exact
// oracle (@exp, 1/l) and the hardware-modeled output — their difference is the pure
// approximation error the cosim ε must cover. f16 K/V are widened to f32 (as on
// silicon); when comparing exact-vs-approx the f16 rounding cancels, isolating exp/recip.

pub const Params = struct {
    head_dim_q: usize,
    head_dim_v: usize,
    n_heads: usize,
    n_head_kv: usize,
    n_kv: usize,
    n_tokens: usize,
    scale: f32,
};

const ExpFn = fn (f32) f32;
const RecipFn = fn (f32) f32;

fn exactExp(x: f32) f32 {
    return @exp(x);
}
fn exactRecip(x: f32) f32 {
    return if (x == 0) 0 else 1.0 / x;
}
fn hwExp(x: f32) f32 {
    return softmaxExp(exp_lut_bits, x);
}
fn hwRecip(x: f32) f32 {
    return recip(recip_lut_bits, x);
}

/// One head/token output o[0..head_dim_v] = softmax(scale·Q·Kᵀ + mask)·V, computed
/// with the given exp/recip. Q is f32; K/V are f16 (widened); mask is f16 additive,
/// −inf ⇒ skip. Row-major contiguous layout: Q[token][head], K/V[kv][kv_head].
fn attendHead(
    comptime expFn: ExpFn,
    comptime recipFn: RecipFn,
    p: Params,
    q: []const f32,
    k: []const f16,
    v: []const f16,
    mask: ?[]const f16, // per-token row, length n_kv
    out: []f32,
    token: usize,
    head: usize,
) void {
    const head_ratio = p.n_heads / p.n_head_kv;
    const kv_head = head / head_ratio;
    const q_row = q[(token * p.n_heads + head) * p.head_dim_q ..][0..p.head_dim_q];

    var m: f32 = -std.math.inf(f32);
    var l: f32 = 0;
    var acc = [_]f32{0} ** 256;
    const accv = acc[0..p.head_dim_v];

    for (0..p.n_kv) |kv| {
        var mask_value: f32 = 0;
        if (mask) |mrow| {
            mask_value = @floatCast(mrow[kv]);
            if (!std.math.isFinite(mask_value) and mask_value < 0) continue;
        }
        const k_row = k[(kv * p.n_head_kv + kv_head) * p.head_dim_q ..][0..p.head_dim_q];
        var dot: f32 = 0;
        for (q_row, k_row) |qd, kd| dot += qd * @as(f32, @floatCast(kd));
        const score = dot * p.scale + mask_value;

        const m_new = @max(m, score);
        if (m_new != m) {
            const corr = expFn(m - m_new);
            l *= corr;
            for (accv) |*a| a.* *= corr;
            m = m_new;
        }
        const pe = expFn(score - m);
        l += pe;
        const v_row = v[(kv * p.n_head_kv + kv_head) * p.head_dim_v ..][0..p.head_dim_v];
        for (accv, v_row) |*a, vd| a.* += pe * @as(f32, @floatCast(vd));
    }

    const inv_l = recipFn(l);
    for (out, accv) |*o, a| o.* = a * inv_l;
}

// ============================ characterization ===============================
//
// Run: zig test fpga/sim/support/flash_ref.zig
// The tests print tables and assert only loose sanity — the numbers are the product.

fn maxRelExpError(comptime B: u5) f64 {
    var max_rel: f64 = 0;
    var x: f64 = -40.0;
    while (x <= 0.0) : (x += 0.0009765625) {
        const exact = @exp(x);
        const got: f64 = softmaxExp(B, @floatCast(x));
        if (exact > 1e-12) max_rel = @max(max_rel, @abs(got - exact) / exact);
    }
    return max_rel;
}

fn maxRelRecipError(comptime B: u5) f64 {
    var max_rel: f64 = 0;
    var x: f64 = 1.0;
    while (x <= 8192.0) : (x += 0.5) {
        const got: f64 = recip(B, @floatCast(x));
        max_rel = @max(max_rel, @abs(got - 1.0 / x) * x);
    }
    return max_rel;
}

test "exp/recip approximation error vs LUT address bits" {
    std.debug.print("\n  flash exp/recip characterization (table + linear interp)\n", .{});
    std.debug.print("  {s:>4} | {s:>14} | {s:>16}\n", .{ "B", "max_rel(exp)", "max_rel(recip)" });
    inline for (.{ 4, 5, 6, 7, 8 }) |B| {
        std.debug.print("  {d:>4} | {e:>14.4} | {e:>16.4}\n", .{ B, maxRelExpError(B), maxRelRecipError(B) });
    }
    std.debug.print("  committed: B={d}\n\n", .{exp_lut_bits});
    try std.testing.expect(maxRelExpError(exp_lut_bits) < 1e-5);
    try std.testing.expect(maxRelRecipError(exp_lut_bits) < 1e-5);
}

test "hardware-modeled attention output matches the exact schedule within ε" {
    const a = std.testing.allocator;
    // Bonsai-ish GQA decode shape: head_dim 64, 4 query heads over 1 kv head, 256 keys.
    const p: Params = .{ .head_dim_q = 64, .head_dim_v = 64, .n_heads = 4, .n_head_kv = 1, .n_kv = 256, .n_tokens = 2, .scale = 0.125 };

    const q = try a.alloc(f32, p.n_tokens * p.n_heads * p.head_dim_q);
    defer a.free(q);
    const k = try a.alloc(f16, p.n_kv * p.n_head_kv * p.head_dim_q);
    defer a.free(k);
    const v = try a.alloc(f16, p.n_kv * p.n_head_kv * p.head_dim_v);
    defer a.free(v);
    const mask = try a.alloc(f16, p.n_tokens * p.n_kv);
    defer a.free(mask);
    var exact: [64]f32 = undefined;
    var hw: [64]f32 = undefined;

    var prng = std.Random.DefaultPrng.init(0x5EED);
    const rnd = prng.random();
    for (q) |*x| x.* = (rnd.float(f32) - 0.5) * 2.0;
    for (k) |*x| x.* = @floatCast((rnd.float(f32) - 0.5) * 2.0); // f16-rounded, as on silicon
    for (v) |*x| x.* = @floatCast((rnd.float(f32) - 0.5) * 2.0);
    // Causal-ish: token t attends kv ≤ (t+1)·128; the rest is −inf padding.
    for (0..p.n_tokens) |t| {
        for (0..p.n_kv) |kv| {
            mask[t * p.n_kv + kv] = if (kv < (t + 1) * 128) 0.0 else -std.math.inf(f16);
        }
    }

    var max_abs: f64 = 0;
    var max_rel: f64 = 0;
    for (0..p.n_tokens) |t| {
        const mrow = mask[t * p.n_kv ..][0..p.n_kv];
        for (0..p.n_heads) |h| {
            attendHead(exactExp, exactRecip, p, q, k, v, mrow, &exact, t, h);
            attendHead(hwExp, hwRecip, p, q, k, v, mrow, &hw, t, h);
            for (exact[0..p.head_dim_v], hw[0..p.head_dim_v]) |e, g| {
                max_abs = @max(max_abs, @abs(e - g));
                if (@abs(e) > 1e-4) max_rel = @max(max_rel, @abs(e - g) / @abs(e));
            }
        }
    }
    std.debug.print("  attention output exact-vs-HW (B={d}): max_abs={e:.3} max_rel={e:.3}\n\n", .{ exp_lut_bits, max_abs, max_rel });
    // The PS↔llama.cpp oracle floor is max_abs_diff ≈ 0.145 in logit space; the
    // exp/recip approximation must be orders below that to be a non-issue.
    try std.testing.expect(max_abs < 1e-3);
}

test "softmaxExp is exact at 0 and monotonic" {
    try std.testing.expectEqual(@as(f32, 1.0), softmaxExp(8, 0.0));
    var prev: f32 = 1.0;
    var x: f32 = -0.1;
    while (x > -20.0) : (x -= 0.1) {
        const val = softmaxExp(8, x);
        try std.testing.expect(val < prev and val > 0);
        prev = val;
    }
}
