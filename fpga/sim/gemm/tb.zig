//! Oracle cosim for gemm_rowblock (the gemm decode datapath core, increment 1). Feeds the
//! RAW wire format — packed weight sign-bits, int8 activations, f16 scales — for all ROWS
//! lanes in parallel, one sub-block per beat, then reads each lane's 72-bit accumulator and
//! checks it BIT-EXACT against matmul_ref.windowedRow (the exact-in-window integer truth).
//! This extends fma's per-row proof to the full front-end (f16 decompose + Σ±a reduction)
//! driven from the wire — the genuinely-new gemm logic. The standalone decompose port is
//! swept exhaustively (every f16 except inf/nan) against matmul_ref.decompose.

const std = @import("std");
const ref = @import("matmul_ref");
const layout = @import("layout");
const c = @cImport(@cInclude("shim.h"));

const Dut = struct {
    h: *c.Dut,
    fn init() Dut {
        return .{ .h = c.dut_new().? };
    }
    fn deinit(self: *Dut) void {
        c.dut_free(self.h);
    }
    fn eval(self: *Dut) void {
        c.dut_eval(self.h);
    }
    fn step(self: *Dut) void {
        c.dut_set_clk(self.h, 1);
        c.dut_eval(self.h);
        c.dut_set_clk(self.h, 0);
        c.dut_eval(self.h);
    }
};

// Positive normal f16 with a narrow exponent band — realistic model scales, well inside
// the fixed [-48,+10] window the 104-bit accumulator covers. Same generator the fma cosim uses.
fn normalScale(rnd: std.Random) f16 {
    const exp: u16 = rnd.intRangeAtMost(u16, 12, 18);
    const mant: u16 = rnd.intRangeLessThan(u16, 0, 1024);
    return @bitCast((exp << 10) | mant);
}

fn readAcc(h: *c.Dut) i128 {
    const w0: u128 = c.dut_acc0(h);
    const w1: u128 = c.dut_acc1(h);
    const w2: u128 = c.dut_acc2(h);
    const w3: u128 = c.dut_acc3(h);
    var raw: u128 = w0 | (w1 << 32) | (w2 << 64) | (w3 << 96);
    raw &= (@as(u128, 1) << 104) - 1;
    if (raw >> 103 != 0) return @as(i128, @intCast(raw)) - (@as(i128, 1) << 104);
    return @intCast(raw);
}

fn signExtend(raw: u32, bits: u5) i64 {
    const sign_bit = @as(u32, 1) << (bits - 1);
    if (raw & sign_bit != 0) {
        return @as(i64, raw) - (@as(i64, 1) << bits);
    }
    return raw;
}

// Bit-exact model of gemm_emit: |acc|·2^emin -> truncated fp32 (round-toward-zero). Same
// algorithm as the RTL — align the leading 1 to bit 23, drop the rest, exponent msb+emin+127
// — so the gate pins every output bit. (Underflow/overflow branches mirror the RTL but the
// calibrated window keeps the exponent normal, so they aren't exercised here.)
fn truncEmitBits(acc: i128, emin: i32) u32 {
    if (acc == 0) return 0;
    const sign: u32 = if (acc < 0) 1 else 0;
    const mag: u128 = @intCast(if (acc < 0) -acc else acc);
    const msb: i32 = 127 - @as(i32, @clz(mag));
    const e_b: i32 = msb + emin + 127;
    if (e_b <= 0) return sign << 31;
    if (e_b >= 255) return (sign << 31) | (0xFE << 23) | 0x7FFFFF;
    const mant: u32 = if (msb >= 23)
        @truncate((mag >> @intCast(msb - 23)) & 0x7FFFFF)
    else
        @truncate((mag << @intCast(23 - msb)) & 0x7FFFFF);
    return (sign << 31) | (@as(u32, @intCast(e_b)) << 23) | mant;
}

// low 104 bits of a signed acc as {lo, mid, hi, top[7:0]} for the directed-sweep port.
fn accWords(acc: i128) [4]u32 {
    const raw: u128 = @as(u128, @bitCast(acc)) & ((@as(u128, 1) << 104) - 1);
    return .{ @truncate(raw), @truncate(raw >> 32), @truncate(raw >> 64), @truncate(raw >> 96) };
}

// Directed emit-sweep case space: every leading-1 position (msb 0..102) × sign × emin.
const DIR_EMINS = [_]i32{ -40, -7, 0, 5, 30 };
const DIR_NMSB: usize = 103;
const DIR_BASE_N: usize = 2 * DIR_NMSB * DIR_EMINS.len;
const DIR_N: usize = DIR_BASE_N + 2;
fn dirAcc(k: usize) i128 {
    if (k == DIR_BASE_N) return 0;
    if (k == DIR_BASE_N + 1) return -(@as(i128, 1) << 103);
    const msb = (k / DIR_EMINS.len) % DIR_NMSB;
    const neg = (k / DIR_EMINS.len) / DIR_NMSB;
    var mag: u128 = @as(u128, 1) << @intCast(msb);
    mag |= (0x5A5A5A5A5A5A5A5A5A5A5A5A5A5A5A5A & (mag - 1)); // arbitrary sub-leading bits to truncate
    return if (neg == 1) -@as(i128, @intCast(mag)) else @as(i128, @intCast(mag));
}
fn dirEmin(k: usize) i32 {
    if (k >= DIR_BASE_N) return -48;
    return DIR_EMINS[k % DIR_EMINS.len];
}

pub fn main() !void {
    const rows = layout.ROWS;
    const COLS_MAX: usize = 8; // must match gemm_top's COLS_MAX parameter
    const B: usize = 4;
    const Q1 = layout.Q1_BLOCK;
    const Q8 = layout.Q8_BLOCK;
    const SUBS = layout.Q8_SUBBLOCKS;

    var prng = std.Random.DefaultPrng.init(0x6E33A1);
    const rnd = prng.random();

    var wbits: [rows * B]u128 = undefined;
    var wscales: [rows * B]f16 = undefined;
    var aquants: [B * Q1]i8 = undefined;
    var ascales: [B * SUBS]f16 = undefined;
    for (&wbits) |*x| x.* = rnd.int(u128);
    for (&wscales) |*x| x.* = normalScale(rnd);
    for (&aquants) |*x| x.* = rnd.intRangeAtMost(i8, -127, 127);
    for (&ascales) |*x| x.* = normalScale(rnd);

    const p = ref.Problem{
        .rows = rows,
        .q1_blocks = B,
        .weight_bits = &wbits,
        .weight_scales = &wscales,
        .act_quants = &aquants,
        .act_scales = &ascales,
    };
    const w = ref.fixedWindow(); // the deployed fixed window (emin -48, ACC_W 104) — no calibration

    var dut = Dut.init();
    defer dut.deinit();

    // reset
    c.dut_set_rst_n(dut.h, 0);
    c.dut_set_clear(dut.h, 0);
    c.dut_set_valid(dut.h, 0);
    c.dut_set_emin(dut.h, @as(i8, @intCast(w.emin)));
    c.dut_set_act_scale(dut.h, 0);
    c.dut_set_read_row(dut.h, 0);
    c.dut_set_col_idx(dut.h, 0);
    c.dut_set_read_col(dut.h, 0);
    c.dut_set_dbg_f16(dut.h, 0);
    c.dut_set_dbg_emit_vin(dut.h, 0);
    c.dut_set_dbg_acc(dut.h, 0, 0, 0, 0);
    c.dut_set_dbg_emin(dut.h, 0);
    var zw = [_]u32{0} ** rows;
    c.dut_set_weight_bits(dut.h, &zw, @intCast(rows));
    c.dut_set_weight_scales(dut.h, &zw, @intCast(rows / 2));
    c.dut_set_acts(dut.h, &zw, 8);
    c.dut_set_clk(dut.h, 0);
    c.dut_eval(dut.h);
    for (0..6) |_| dut.step();
    c.dut_set_rst_n(dut.h, 1);

    // ---- exhaustive decompose sweep (every f16 except inf/nan) ----
    var dec_mm: usize = 0;
    var f16v: u32 = 0;
    while (f16v <= 0xFFFF) : (f16v += 1) {
        if ((f16v >> 10) & 0x1F == 0x1F) continue; // inf/nan: not represented (scales finite)
        c.dut_set_dbg_f16(dut.h, @intCast(f16v));
        dut.eval();
        const parts = ref.decompose(@bitCast(@as(u16, @intCast(f16v))));
        const got_sig = signExtend(c.dut_dbg_sig(dut.h), 12);
        const got_e = signExtend(c.dut_dbg_e(dut.h), 8);
        if (got_sig != parts.sig or got_e != @as(i64, parts.e)) {
            dec_mm += 1;
            if (dec_mm <= 8)
                std.debug.print("  decompose MISMATCH @0x{X:0>4}: sig {d} vs {d}, e {d} vs {d}\n", .{ f16v, got_sig, parts.sig, got_e, parts.e });
        }
    }
    c.dut_set_dbg_f16(dut.h, 0);

    // ---- directed emit sweep: every leading-1 position × sign × emin, STREAMED through the
    //      pipelined emit (valid_in→valid_out), checked vs the truncating model. Keeps the
    //      msb/sign edge coverage the realistic-window kernel cases (test-rtl-gemm-kernel,
    //      which gates the decode emit end-to-end) can't reach. ----
    var emit_dir_mm: usize = 0;
    var fed: usize = 0;
    var rcv: usize = 0;
    while (rcv < DIR_N) : (fed += 1) {
        if (fed < DIR_N) {
            const wds = accWords(dirAcc(fed));
            c.dut_set_dbg_acc(dut.h, wds[0], wds[1], wds[2], wds[3]);
            c.dut_set_dbg_emin(dut.h, @intCast(dirEmin(fed)));
            c.dut_set_dbg_emit_vin(dut.h, 1);
        } else {
            c.dut_set_dbg_emit_vin(dut.h, 0);
        }
        dut.step();
        if (c.dut_dbg_emit_vout(dut.h) != 0) {
            const got = c.dut_dbg_emit_f32(dut.h);
            const want = truncEmitBits(dirAcc(rcv), dirEmin(rcv));
            if (got != want) {
                emit_dir_mm += 1;
                if (emit_dir_mm <= 8)
                    std.debug.print("  emit dir case {d}: 0x{X:0>8} vs 0x{X:0>8}\n", .{ rcv, got, want });
            }
            rcv += 1;
        }
    }
    c.dut_set_dbg_emit_vin(dut.h, 0);
    c.dut_set_dbg_acc(dut.h, 0, 0, 0, 0);
    c.dut_set_dbg_emin(dut.h, 0);

    // ---- stream the sub-blocks: all ROWS lanes in parallel, one (blk,sub) per beat ----
    for (0..B) |blk| {
        for (0..SUBS) |sub| {
            // weight sign-bits: lane r -> its 32 bits for this sub-block.
            var wb: [rows]u32 = undefined;
            var ws: [rows / 2]u32 = undefined;
            for (0..rows) |r| {
                wb[r] = @truncate(wbits[r * B + blk] >> @intCast(sub * 32));
            }
            // weight scales: 2 f16 per 32-bit word (lane 2i in low half, 2i+1 in high).
            for (0..rows / 2) |i| {
                const s0: u32 = @as(u16, @bitCast(wscales[(2 * i) * B + blk]));
                const s1: u32 = @as(u16, @bitCast(wscales[(2 * i + 1) * B + blk]));
                ws[i] = s0 | (s1 << 16);
            }
            // activations: 32 int8 of this sub-block -> 8 words, 4 bytes each.
            var ap: [8]u32 = undefined;
            for (0..8) |k| {
                var word: u32 = 0;
                for (0..4) |b| {
                    const q: u8 = @bitCast(aquants[blk * Q1 + sub * Q8 + k * 4 + b]);
                    word |= @as(u32, q) << @intCast(b * 8);
                }
                ap[k] = word;
            }

            c.dut_set_weight_bits(dut.h, &wb, @intCast(rows));
            c.dut_set_weight_scales(dut.h, &ws, @intCast(rows / 2));
            c.dut_set_acts(dut.h, &ap, 8);
            c.dut_set_act_scale(dut.h, @as(u16, @bitCast(ascales[blk * SUBS + sub])));
            c.dut_set_clear(dut.h, if (blk == 0 and sub == 0) 1 else 0);
            c.dut_set_valid(dut.h, 1);
            dut.step();
        }
    }
    c.dut_set_valid(dut.h, 0);
    c.dut_set_clear(dut.h, 0);
    for (0..16) |_| dut.step(); // drain the pipelined front-end (FE_LAT) + fma latency

    // ---- read each lane's accumulator, gate bit-exact vs windowedRow. (The decode emit
    //      f32 is gated end-to-end by test-rtl-gemm-kernel; here we gate the integer
    //      accumulate, and the streamed directed sweep above gates the emit's math.) ----
    var mism: usize = 0;
    var sats: usize = 0;
    for (0..rows) |row| {
        c.dut_set_read_row(dut.h, @intCast(row));
        dut.eval(); // read_row -> acc is a combinational mux
        const got = readAcc(dut.h);
        const exp_row = ref.windowedRow(p, row, w);
        sats += exp_row.sats;
        if (got != exp_row.acc) {
            mism += 1;
            if (mism <= 8)
                std.debug.print("  row {d}: gemm acc={d} expected={d}\n", .{ row, got, exp_row.acc });
        }
    }

    std.debug.print(
        "\n  gemm decode cosim: {d} rows, acc mismatches {d}, decompose sweep mismatches {d}, emit directed-sweep mismatches {d}, window=[{d},{d}] acc_w={d}, sats {d}\n",
        .{ rows, mism, dec_mm, emit_dir_mm, w.emin, w.emax, w.acc_w, sats },
    );
    if (dec_mm != 0) {
        std.debug.print("  FAIL: gemm_f16_decompose !== matmul_ref.decompose\n", .{});
        return error.DecomposeMismatch;
    }
    if (mism != 0) {
        std.debug.print("  FAIL: gemm acc !== windowedRow (front-end + fma broken)\n", .{});
        return error.ResultMismatch;
    }
    if (sats != 0) {
        std.debug.print("  FAIL: unexpected saturations (window should cover the test problem)\n", .{});
        return error.WindowTooNarrow;
    }
    if (emit_dir_mm != 0) {
        std.debug.print("  FAIL: gemm_emit !== truncating model (directed sweep)\n", .{});
        return error.EmitMismatch;
    }
    std.debug.print("  gemm decode cosim passed (front-end+fma === windowedRow; emit math === trunc model)\n", .{});

    // ---- prefill (C>1): COLS_MAX columns share the weights, distinct acts per column;
    //      col_idx selects the per-row accumulator. Exercises the bank + the col_idx
    //      pipeline alignment (a wrong delay → contributions land in the wrong column). ----
    const C = COLS_MAX;
    var aq: [C][B * Q1]i8 = undefined;
    var asc: [C][B * SUBS]f16 = undefined;
    for (0..C) |col| {
        for (&aq[col]) |*x| x.* = rnd.intRangeAtMost(i8, -127, 127);
        for (&asc[col]) |*x| x.* = normalScale(rnd);
    }
    var probs: [C]ref.Problem = undefined;
    for (0..C) |col| probs[col] = .{
        .rows = rows,        .q1_blocks = B,
        .weight_bits = &wbits, .weight_scales = &wscales,
        .act_quants = &aq[col], .act_scales = &asc[col],
    };
    const wp = ref.fixedWindow();
    c.dut_set_emin(dut.h, @as(i8, @intCast(wp.emin)));

    for (0..B) |blk| {
        for (0..SUBS) |sub| {
            // weights held across the column sweep (depend only on blk,sub).
            var wb: [rows]u32 = undefined;
            var ws: [rows / 2]u32 = undefined;
            for (0..rows) |r| wb[r] = @truncate(wbits[r * B + blk] >> @intCast(sub * 32));
            for (0..rows / 2) |i| {
                const s0: u32 = @as(u16, @bitCast(wscales[(2 * i) * B + blk]));
                const s1: u32 = @as(u16, @bitCast(wscales[(2 * i + 1) * B + blk]));
                ws[i] = s0 | (s1 << 16);
            }
            for (0..C) |col| {
                var ap: [8]u32 = undefined;
                for (0..8) |k| {
                    var word: u32 = 0;
                    for (0..4) |b| {
                        const q: u8 = @bitCast(aq[col][blk * Q1 + sub * Q8 + k * 4 + b]);
                        word |= @as(u32, q) << @intCast(b * 8);
                    }
                    ap[k] = word;
                }
                c.dut_set_weight_bits(dut.h, &wb, @intCast(rows));
                c.dut_set_weight_scales(dut.h, &ws, @intCast(rows / 2));
                c.dut_set_acts(dut.h, &ap, 8);
                c.dut_set_act_scale(dut.h, @as(u16, @bitCast(asc[col][blk * SUBS + sub])));
                c.dut_set_col_idx(dut.h, @intCast(col));
                c.dut_set_clear(dut.h, if (blk == 0 and sub == 0) 1 else 0);
                c.dut_set_valid(dut.h, 1);
                dut.step();
            }
        }
    }
    c.dut_set_valid(dut.h, 0);
    c.dut_set_clear(dut.h, 0);
    for (0..16) |_| dut.step();

    var pf_mm: usize = 0;
    var pf_sats: usize = 0;
    for (0..C) |col| {
        c.dut_set_read_col(dut.h, @intCast(col));
        // One local read_col register followed by the redundant-pair resolve register.
        dut.step();
        dut.step();
        for (0..rows) |row| {
            c.dut_set_read_row(dut.h, @intCast(row));
            dut.eval();
            const got = readAcc(dut.h);
            const exp_row = ref.windowedRow(probs[col], row, wp);
            pf_sats += exp_row.sats;
            if (got != exp_row.acc) {
                pf_mm += 1;
                if (pf_mm <= 8)
                    std.debug.print("  prefill col {d} row {d}: gemm acc={d} expected={d}\n", .{ col, row, got, exp_row.acc });
            }
        }
    }
    std.debug.print(
        "  gemm prefill bank cosim: {d} cols × {d} rows, acc mismatches {d}, window=[{d},{d}] acc_w={d}, sats {d}\n",
        .{ C, rows, pf_mm, wp.emin, wp.emax, wp.acc_w, pf_sats },
    );
    if (pf_mm != 0) {
        std.debug.print("  FAIL: gemm prefill bank !== windowedRow (col_idx alignment / bank broken)\n", .{});
        return error.PrefillMismatch;
    }
    if (pf_sats != 0) {
        std.debug.print("  FAIL: unexpected saturations in prefill window\n", .{});
        return error.WindowTooNarrow;
    }
    std.debug.print("  gemm cosim passed (decode + prefill bank === windowedRow; emit === trunc model)\n\n", .{});
}
