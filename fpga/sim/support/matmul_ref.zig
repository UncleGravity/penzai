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

// ───────────────────── plan-7 phase-2: fixed-point accumulate reference ────────
//
// `scaledOutput` above is itself fp32, so it cannot gate a datapath whose claim is
// "exact-in-window, better-than-fp32" — the oracle would BE the thing under test.
// These add a higher-precision *truth* (`exactOutput`, an exact i128 accumulate) and
// a bit-accurate model of the hardware's windowed integer accumulate
// (`windowedFixedOutput`), so the phase-2 RTL can be gated `==` inside the window and
// proven ≤ fp32 error outside. The integer front-end (`subSum`) is unchanged and
// shared — this is purely the scale→accumulate→emit tail (plan-fpga-7.md Part 1).

/// f16 decomposed to an exact (signed significand, power-of-two exponent): the value
/// is exactly `sig · 2^e`, no rounding. Quant scales are finite (asserts non-inf/nan).
pub const F16Parts = struct { sig: i64, e: i32 };

pub fn decompose(x: f16) F16Parts {
    const bits: u16 = @bitCast(x);
    const sign: u16 = bits >> 15;
    const exp_field: u16 = (bits >> 10) & 0x1F;
    const mant: u16 = bits & 0x3FF;
    std.debug.assert(exp_field != 0x1F); // no inf/nan scales
    if (exp_field == 0) {
        // zero or subnormal: value = mant · 2^(1-15-10) = mant · 2^-24
        const s: i64 = mant;
        return .{ .sig = if (sign == 1) -s else s, .e = -24 };
    }
    // normal: value = (1024+mant) · 2^(exp_field-15-10)
    const s: i64 = 1024 + @as(i64, mant);
    return .{ .sig = if (sign == 1) -s else s, .e = @as(i32, @intCast(exp_field)) - 25 };
}

/// Exact exponent of one contribution ws·as·ss: value = contribMant · 2^contribExp.
fn contribExp(ws: f16, as_: f16) i32 {
    return decompose(ws).e + decompose(as_).e;
}

/// Exact integer mantissa (signed) of one contribution. With contribExp this is the
/// contribution losslessly: m · 2^e. Bounded by ~2^11·2^11·2^13 = 2^35.
fn contribMant(ws: f16, as_: f16, ss: i32) i128 {
    return @as(i128, decompose(ws).sig) * @as(i128, decompose(as_).sig) * @as(i128, ss);
}

/// One row's exact accumulator: the value is `acc · 2^emin`, summed with a full
/// window (emin = min contribution exponent) so nothing is lost. The ground truth.
const RowExact = struct { acc: i128, emin: i32, nonzero: bool };

fn exactRow(p: Problem, row: usize) RowExact {
    var emin: i32 = std.math.maxInt(i32);
    var any = false;
    for (0..p.q1_blocks) |blk| {
        const ws = p.weight_scales[row * p.q1_blocks + blk];
        for (0..layout.Q8_SUBBLOCKS) |sub| {
            const as_ = p.act_scales[blk * layout.Q8_SUBBLOCKS + sub];
            if (contribMant(ws, as_, subSum(p, row, blk, sub)) == 0) continue;
            emin = @min(emin, contribExp(ws, as_));
            any = true;
        }
    }
    if (!any) return .{ .acc = 0, .emin = 0, .nonzero = false };
    var acc: i128 = 0;
    for (0..p.q1_blocks) |blk| {
        const ws = p.weight_scales[row * p.q1_blocks + blk];
        for (0..layout.Q8_SUBBLOCKS) |sub| {
            const as_ = p.act_scales[blk * layout.Q8_SUBBLOCKS + sub];
            const m = contribMant(ws, as_, subSum(p, row, blk, sub));
            if (m == 0) continue;
            acc += m << @as(u7, @intCast(contribExp(ws, as_) - emin)); // exact (full window)
        }
    }
    return .{ .acc = acc, .emin = emin, .nonzero = true };
}

/// Effectively-exact reference output per row. The i128 accumulate is exact; the
/// final f64 carries it without loss whenever the result fits f64's 52-bit mantissa
/// (true for any realistic window). Both fp32 and fixed-point are measured AGAINST this.
pub fn exactOutput(p: Problem, out: []f64) void {
    std.debug.assert(out.len == p.rows);
    for (0..p.rows) |row| {
        const r = exactRow(p, row);
        out[row] = if (!r.nonzero) 0 else std.math.ldexp(@as(f64, @floatFromInt(r.acc)), r.emin);
    }
}

/// The fixed-point accumulator window (plan-fpga-7.md:116). Values whose exponent is
/// in [emin, emax] accumulate exactly; below emin they lose low bits (bounded loss,
/// still better than fp32's per-add rounding); above emax they saturate (rare, logged).
pub const Window = struct { emin: i32, emax: i32, acc_w: u16 };

/// Physical width of the hardware accumulator (numeric/fma ACC_W) — sized so a SINGLE FIXED
/// window covers the entire f16-scale range, making the gemm matmul exact for any model with
/// no per-model calibration (see `fixedWindow`). The clamp at this width is what `windowedRow`
/// models as the hardware saturation limit. (`Window.acc_w` is a per-problem prediction used
/// by the legacy `calibrateWindow`; the deployed datapath uses the fixed window below.)
pub const ACC_W_BITS: u16 = 104;

/// The fixed accumulator window the deployed gemm datapath bakes in (decode_top wires `emin`
/// to this constant; there is no runtime EMIN register). A single f16 scale's exponent lies
/// in [-24, +5], so a contribution exponent `e_ws + e_as` lies in **[-48, +10]** for ANY f16
/// model. Anchoring the window at the floor (-48) with a 104-bit accumulator covers that whole
/// range exactly: nothing truncates (every `e >= emin`) and nothing saturates (the top +
/// 36-bit product + Σ-headroom fits 104b for K ≲ 16384). So `windowedFixedOutput(p, fixedWindow)`
/// is the exact integer accumulate truncated to fp32 for every problem — the oracle the cosims
/// gate the RTL against, with the per-model EMIN concept deleted.
pub const fixed_emin: i32 = -48; // min contribution exponent: f16 subnormal (-24) × subnormal (-24)
pub const fixed_emax: i32 = 10; //  max contribution exponent: f16 max exp (+5) × (+5)

pub fn fixedWindow() Window {
    return .{ .emin = fixed_emin, .emax = fixed_emax, .acc_w = ACC_W_BITS };
}

/// One-time calibration sweep: derive EMIN/EMAX from the contribution exponents over a
/// representative problem set, and size the accumulator = window span + exact-product
/// mantissa (≤36b) + Σ headroom (log2 terms) + sign (plan-fpga-7.md:116-118).
pub fn calibrateWindow(problems: []const Problem) Window {
    var emin: i32 = std.math.maxInt(i32);
    var emax: i32 = std.math.minInt(i32);
    var max_terms: usize = 1;
    for (problems) |p| {
        for (0..p.rows) |row| {
            var terms: usize = 0;
            for (0..p.q1_blocks) |blk| {
                const ws = p.weight_scales[row * p.q1_blocks + blk];
                for (0..layout.Q8_SUBBLOCKS) |sub| {
                    const as_ = p.act_scales[blk * layout.Q8_SUBBLOCKS + sub];
                    if (contribMant(ws, as_, subSum(p, row, blk, sub)) == 0) continue;
                    const e = contribExp(ws, as_);
                    emin = @min(emin, e);
                    emax = @max(emax, e);
                    terms += 1;
                }
            }
            max_terms = @max(max_terms, terms);
        }
    }
    if (emin > emax) return .{ .emin = 0, .emax = 0, .acc_w = 64 }; // all-zero set
    const headroom: i32 = @intCast(std.math.log2_int_ceil(usize, max_terms + 1));
    const need: i32 = (emax - emin) + 36 + headroom + 1;
    return .{ .emin = emin, .emax = emax, .acc_w = @intCast(need) };
}

/// One row through the windowed fixed-point pipeline. Value = `acc · 2^emin`; `sats`
/// counts saturating clamps. Saturation is at the PHYSICAL accumulator width (ACC_W_BITS),
/// matching numeric/fma's hardware clamp — so an overflow saturates (bounded) here exactly
/// as the silicon does, instead of the model and RTL disagreeing. Inside a calibrated
/// window nothing clamps (sats=0) and the integer adds are exact + order-independent.
pub const RowWindowed = struct { acc: i128, sats: usize };

pub fn windowedRow(p: Problem, row: usize, w: Window) RowWindowed {
    std.debug.assert(w.acc_w >= 2 and w.acc_w <= 127);
    const limit: i128 = (@as(i128, 1) << @as(u7, @intCast(ACC_W_BITS - 1))) - 1;
    var acc: i128 = 0;
    var sats: usize = 0;
    for (0..p.q1_blocks) |blk| {
        const ws = p.weight_scales[row * p.q1_blocks + blk];
        for (0..layout.Q8_SUBBLOCKS) |sub| {
            const as_ = p.act_scales[blk * layout.Q8_SUBBLOCKS + sub];
            const m = contribMant(ws, as_, subSum(p, row, blk, sub));
            if (m == 0) continue;
            const e = contribExp(ws, as_);
            if (e > w.emax) { // above the window: cannot represent → saturate, log
                sats += 1;
                acc = if (m > 0) limit else -limit;
                continue;
            }
            const up = e - w.emin;
            acc += if (up >= 0)
                m << @as(u7, @intCast(up)) // in window: exact
            else
                m >> @as(u7, @intCast(-up)); // below window: truncate (bounded loss)
            if (acc > limit) {
                acc = limit;
                sats += 1;
            } else if (acc < -limit) {
                acc = -limit;
                sats += 1;
            }
        }
    }
    return .{ .acc = acc, .sats = sats };
}

/// Truncating emit: the wide accumulator value (acc · 2^emin) → fp32 bits, round-toward-
/// zero. Models gemm_emit EXACTLY — align the leading 1 to bit 23, drop the rest, exponent
/// = msb + emin + 127, saturate to max-finite on overflow. The hardware emit truncates
/// (one LZD+shift per output, no round logic, matching fmul/fadd/cvt), so the oracle must
/// too — otherwise the gemm end-to-end gate is ~1 ULP instead of bit-exact.
pub fn emitTrunc(acc: i128, emin: i32) u32 {
    if (acc == 0) return 0;
    const sign: u32 = if (acc < 0) 1 else 0;
    const mag: u128 = @intCast(if (acc < 0) -acc else acc);
    const msb: i32 = 127 - @as(i32, @clz(mag));
    const e_b: i32 = msb + emin + 127;
    if (e_b <= 0) return sign << 31; // underflow -> signed zero
    if (e_b >= 255) return (sign << 31) | (@as(u32, 0xFE) << 23) | 0x7FFFFF; // saturate
    const mant: u32 = if (msb >= 23)
        @truncate((mag >> @intCast(msb - 23)) & 0x7FFFFF)
    else
        @truncate((mag << @intCast(23 - msb)) & 0x7FFFFF);
    return (sign << 31) | (@as(u32, @intCast(e_b)) << 23) | mant;
}

/// Fixed-point reference output per row, emitted via the truncating `emitTrunc` (a faithful
/// model of the gemm_emit hardware — NOT f64 round-nearest), so the gemm cosim gates the
/// full pipeline bit-exact. `saturations` counts out-of-window contributions (rare, logged).
pub fn windowedFixedOutput(p: Problem, w: Window, out: []f32, saturations: *usize) void {
    std.debug.assert(out.len == p.rows);
    saturations.* = 0;
    for (0..p.rows) |row| {
        const r = windowedRow(p, row, w);
        saturations.* += r.sats;
        out[row] = @bitCast(emitTrunc(r.acc, w.emin));
    }
}

const testing = std.testing;

// Build a problem from per-term (ws, as, sub_sum) specs: term i lives at (blk=i,
// sub=0) — its own Q1 block so ws/as are independent per term — replicated across all
// ROWS so any row equals the spec. Only element 0 of sub 0 is nonzero, so
// subSum(row,i,0) == s_i and every other sub-block is zero (skipped).
const Term = struct { ws: f16, as_: f16, s: i8 };

fn TermProblem(comptime n: usize) type {
    return struct {
        wbits: [layout.ROWS * n]u128 = [_]u128{0} ** (layout.ROWS * n),
        wscales: [layout.ROWS * n]f16 = [_]f16{0} ** (layout.ROWS * n),
        aquants: [n * layout.Q1_BLOCK]i8 = [_]i8{0} ** (n * layout.Q1_BLOCK),
        ascales: [n * layout.Q8_SUBBLOCKS]f16 = [_]f16{0} ** (n * layout.Q8_SUBBLOCKS),
        const Self = @This();

        fn init(terms: [n]Term) Self {
            var self = Self{};
            for (0..layout.ROWS) |r| {
                for (0..n) |i| {
                    self.wscales[r * n + i] = terms[i].ws;
                    self.wbits[r * n + i] = 1; // bit 0 set (+); only act[0] of sub 0 nonzero
                }
            }
            for (0..n) |i| {
                self.ascales[i * layout.Q8_SUBBLOCKS + 0] = terms[i].as_;
                self.aquants[i * layout.Q1_BLOCK + 0] = terms[i].s;
            }
            return self;
        }

        fn problem(self: *const Self) Problem {
            return .{
                .rows = layout.ROWS,
                .q1_blocks = n,
                .weight_bits = &self.wbits,
                .weight_scales = &self.wscales,
                .act_quants = &self.aquants,
                .act_scales = &self.ascales,
            };
        }
    };
}

test "windowed accumulate is bit-exact vs the i128 truth inside a covering window" {
    var tp = TermProblem(4).init(.{
        .{ .ws = 1.0, .as_ = 1.0, .s = 3 },
        .{ .ws = 2.0, .as_ = 0.5, .s = 5 },
        .{ .ws = 0.25, .as_ = 4.0, .s = -2 },
        .{ .ws = 8.0, .as_ = 8.0, .s = 1 },
    });
    const p = tp.problem();
    const w = calibrateWindow(&.{p});
    // Same per-row emin (all rows identical) => integer accumulators must be identical.
    for (0..p.rows) |row| {
        const exact = exactRow(p, row);
        const win = windowedRow(p, row, w);
        try testing.expectEqual(@as(usize, 0), win.sats);
        try testing.expectEqual(exact.emin, w.emin);
        try testing.expectEqual(exact.acc, win.acc); // bit-exact, no float
    }
}

test "fixed-point beats fp32 on a wide-dynamic-range row" {
    // One big contribution (256·256·1 = 65536) then 16 tiny ones (2^-5·2^-5·1 = 2^-10).
    // In problem order the big term lands first, so each tiny add is < ½ ULP@65536 and
    // vanishes in fp32; the fixed-point window keeps them exactly.
    var terms: [17]Term = undefined;
    terms[0] = .{ .ws = 256.0, .as_ = 256.0, .s = 1 };
    for (1..17) |i| terms[i] = .{ .ws = 0.03125, .as_ = 0.03125, .s = 1 }; // 2^-5
    var tp = TermProblem(17).init(terms);
    const p = tp.problem();

    var truth: [layout.ROWS]f64 = undefined;
    var fp32: [layout.ROWS]f32 = undefined;
    var fixed: [layout.ROWS]f32 = undefined;
    var sats: usize = 0;
    exactOutput(p, &truth);
    scaledOutput(p, &fp32);
    windowedFixedOutput(p, calibrateWindow(&.{p}), &fixed, &sats);

    const expect_truth: f64 = 65536.0 + 16.0 * std.math.ldexp(@as(f64, 1.0), -10); // +0.015625
    try testing.expectEqual(expect_truth, truth[0]);
    try testing.expectEqual(@as(f32, 65536.0), fp32[0]); // tinies lost to fp32 rounding
    try testing.expectEqual(@as(usize, 0), sats);

    const err_fixed = @abs(@as(f64, fixed[0]) - truth[0]);
    const err_fp32 = @abs(@as(f64, fp32[0]) - truth[0]);
    try testing.expectEqual(@as(f64, 0.0), err_fixed); // exact in window
    try testing.expect(err_fixed < err_fp32); // strictly better than fp32
}

test "calibrateWindow brackets every contribution exponent" {
    var tp = TermProblem(4).init(.{
        .{ .ws = 64.0, .as_ = 64.0, .s = 7 },
        .{ .ws = 0.0625, .as_ = 0.0625, .s = 3 },
        .{ .ws = 1.0, .as_ = 1.0, .s = -1 },
        .{ .ws = 4.0, .as_ = 0.25, .s = 9 },
    });
    const p = tp.problem();
    const w = calibrateWindow(&.{p});
    for (0..4) |i| {
        const e = contribExp(tp.wscales[i], tp.ascales[i * layout.Q8_SUBBLOCKS]);
        try testing.expect(e >= w.emin and e <= w.emax);
    }
}

test "contribution above the window top is flagged as a saturation" {
    var tp = TermProblem(3).init(.{
        .{ .ws = 1.0, .as_ = 1.0, .s = 2 },
        .{ .ws = 2.0, .as_ = 2.0, .s = 2 },
        .{ .ws = 1024.0, .as_ = 1024.0, .s = 1 }, // largest exponent
    });
    const p = tp.problem();
    var w = calibrateWindow(&.{p});
    w.emax -= 4; // shrink the top below the largest term's exponent
    var out: [layout.ROWS]f32 = undefined;
    var sats: usize = 0;
    windowedFixedOutput(p, w, &out, &sats);
    try testing.expect(sats >= p.rows); // at least the out-of-window term, every row
}

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
