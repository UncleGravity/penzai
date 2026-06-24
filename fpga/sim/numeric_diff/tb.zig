//! Differential cosim for the numeric/ leaves: each new leaf runs beside the proven
//! rtl/fp leaf it replaces, fed identical stimulus, and must match BIT-FOR-BIT. The
//! modules are deterministic functions of their input bits, so the gate is the
//! strongest possible — random/exhaustive full-range stimulus, new === old for every
//! input ⇒ safe to retire the old. No software oracle.
//!
//!   clocked (streamed): fadd≡fp32_add_pipe, fmul≡fp32_mul_pipe, reduce≡fp_addtree
//!   combinational (exhaustive): cvt_f16_f32≡fp16_to_fp32, cvt_i2f≡int_to_fp32

const std = @import("std");
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

pub fn main() !void {
    const N: usize = 4096;
    var prng = std.Random.DefaultPrng.init(0xADD);
    const rnd = prng.random();

    const edge = [_]u32{
        0x00000000, 0x80000000, 0x3F800000, 0xBF800000,
        0x7F7FFFFF, 0xFF7FFFFF, 0x00800000, 0x80800000,
        0x00000001, 0x007FFFFF, 0x7F800000, 0xFF800000,
        0x7FC00000, 0x40490FDB, 0x42F60000, 0x42F5FFFF,
    };
    var as: [N]u32 = undefined;
    var bs: [N]u32 = undefined;
    var k: usize = 0;
    outer: for (edge) |ea| {
        for (edge) |eb| {
            if (k >= N) break :outer;
            as[k] = ea;
            bs[k] = eb;
            k += 1;
        }
    }
    while (k < N) : (k += 1) {
        as[k] = rnd.int(u32);
        bs[k] = rnd.int(u32);
    }

    var dut = Dut.init();
    defer dut.deinit();

    // reset
    c.dut_set_rst_n(dut.h, 0);
    c.dut_set_valid(dut.h, 0);
    c.dut_set_a(dut.h, 0);
    c.dut_set_b(dut.h, 0);
    c.dut_set_c16(dut.h, 0);
    c.dut_set_cint(dut.h, 0);
    var zlanes = [_]u32{0} ** 16;
    c.dut_set_rin(dut.h, &zlanes);
    c.dut_set_ex_x(dut.h, 0);
    c.dut_set_rc_l(dut.h, 0);
    c.dut_set_clk(dut.h, 0);
    c.dut_eval(dut.h);
    for (0..6) |_| dut.step();
    c.dut_set_rst_n(dut.h, 1);

    // ---- combinational cvt sweeps (exhaustive) ----
    var cvt_f16_mm: usize = 0;
    var bf16_widen_mm: usize = 0; // cvt_bf16_f32(c16) must equal {c16, 16'd0}
    var f16v: u32 = 0;
    while (f16v <= 0xFFFF) : (f16v += 1) {
        c.dut_set_c16(dut.h, f16v);
        dut.eval();
        if (c.dut_cvt_f16_new(dut.h) != c.dut_cvt_f16_old(dut.h)) {
            cvt_f16_mm += 1;
            if (cvt_f16_mm <= 8)
                std.debug.print("  cvt_f16 MISMATCH @0x{X:0>4}: new=0x{X:0>8} old=0x{X:0>8}\n", .{ f16v, c.dut_cvt_f16_new(dut.h), c.dut_cvt_f16_old(dut.h) });
        }
        if (c.dut_cvt_bf16_widen(dut.h) != (f16v << 16)) bf16_widen_mm += 1;
    }
    var cvt_i2f_mm: usize = 0;
    var iv: u32 = 0;
    while (iv < 0x4000) : (iv += 1) { // all 2^14 int patterns
        c.dut_set_cint(dut.h, iv);
        dut.eval();
        if (c.dut_cvt_i2f_new(dut.h) != c.dut_cvt_i2f_old(dut.h)) {
            cvt_i2f_mm += 1;
            if (cvt_i2f_mm <= 8)
                std.debug.print("  cvt_i2f MISMATCH @0x{X:0>4}: new=0x{X:0>8} old=0x{X:0>8}\n", .{ iv, c.dut_cvt_i2f_new(dut.h), c.dut_cvt_i2f_old(dut.h) });
        }
    }
    c.dut_set_c16(dut.h, 0);
    c.dut_set_cint(dut.h, 0);

    // ---- clocked streaming: fadd / fmul / reduce ----
    var add_chk: usize = 0;
    var add_mm: usize = 0;
    var mul_chk: usize = 0;
    var mul_mm: usize = 0;
    var red_chk: usize = 0;
    var red_mm: usize = 0;
    var exp_chk: usize = 0;
    var exp_mm: usize = 0;
    var rec_chk: usize = 0;
    var rec_mm: usize = 0;
    var bf16_narrow_mm: usize = 0; // cvt_f32_bf16(a) must equal a >> 16
    var first_valid_cyc: ?usize = null;
    var cyc: usize = 0;

    const drain: usize = 24;
    var fed: usize = 0;
    while (fed < N + drain) : (fed += 1) {
        var lanes: [16]u32 = undefined;
        if (fed < N) {
            c.dut_set_valid(dut.h, 1);
            c.dut_set_a(dut.h, as[fed]);
            c.dut_set_b(dut.h, bs[fed]);
            for (0..16) |i| lanes[i] = rnd.int(u32);
            c.dut_set_rin(dut.h, &lanes);
            c.dut_set_ex_x(dut.h, rnd.int(u32));
            c.dut_set_rc_l(dut.h, rnd.int(u32));
        } else {
            c.dut_set_valid(dut.h, 0);
        }
        dut.step();
        cyc += 1;
        if (fed < N and c.dut_cvt_bf16_narrow(dut.h) != (as[fed] >> 16)) bf16_narrow_mm += 1;

        if (c.dut_fadd_new_valid(dut.h) != 0 and c.dut_fadd_old_valid(dut.h) != 0) {
            if (first_valid_cyc == null) first_valid_cyc = cyc;
            add_chk += 1;
            if (c.dut_fadd_new_out(dut.h) != c.dut_fadd_old_out(dut.h)) add_mm += 1;
        }
        if (c.dut_fmul_new_valid(dut.h) != 0 and c.dut_fmul_old_valid(dut.h) != 0) {
            mul_chk += 1;
            if (c.dut_fmul_new_out(dut.h) != c.dut_fmul_old_out(dut.h)) mul_mm += 1;
        }
        if (c.dut_reduce_new_valid(dut.h) != 0 and c.dut_reduce_old_valid(dut.h) != 0) {
            red_chk += 1;
            if (c.dut_reduce_new_out(dut.h) != c.dut_reduce_old_out(dut.h)) red_mm += 1;
        }
        if (c.dut_exp_new_valid(dut.h) != 0 and c.dut_exp_old_valid(dut.h) != 0) {
            exp_chk += 1;
            if (c.dut_exp_new_out(dut.h) != c.dut_exp_old_out(dut.h)) exp_mm += 1;
        }
        if (c.dut_recip_new_valid(dut.h) != 0 and c.dut_recip_old_valid(dut.h) != 0) {
            rec_chk += 1;
            if (c.dut_recip_new_out(dut.h) != c.dut_recip_old_out(dut.h)) rec_mm += 1;
        }
    }

    const lat: usize = if (first_valid_cyc) |fv| fv - 1 else 0;
    std.debug.print(
        "\n  numeric diff cosim ({d} streamed, fadd latency~{d}):\n" ++
            "    fadd  vs fp32_add_pipe : checked {d}, mismatches {d}\n" ++
            "    fmul  vs fp32_mul_pipe : checked {d}, mismatches {d}\n" ++
            "    reduce vs fp_addtree   : checked {d}, mismatches {d}\n" ++
            "    exp   vs fp_exp        : checked {d}, mismatches {d}\n" ++
            "    recip vs fp_recip      : checked {d}, mismatches {d}\n" ++
            "    cvt_f16_f32 vs fp16_to_fp32 : 65536 swept, mismatches {d}\n" ++
            "    cvt_i2f     vs int_to_fp32  : 16384 swept, mismatches {d}\n" ++
            "    cvt_f32_bf16 (narrow) vs hi16 : {d} checked, mismatches {d}\n" ++
            "    cvt_bf16_f32 (widen)  vs <<16 : 65536 swept, mismatches {d}\n",
        .{ N, lat, add_chk, add_mm, mul_chk, mul_mm, red_chk, red_mm, exp_chk, exp_mm, rec_chk, rec_mm, cvt_f16_mm, cvt_i2f_mm, N, bf16_narrow_mm, bf16_widen_mm },
    );
    if (add_chk < N or mul_chk < N or red_chk < N or exp_chk < N or rec_chk < N) {
        std.debug.print("  FAIL: too few streamed outputs — pipe stalled?\n", .{});
        return error.MissingOutputs;
    }
    if (add_mm + mul_mm + red_mm + exp_mm + rec_mm + cvt_f16_mm + cvt_i2f_mm + bf16_narrow_mm + bf16_widen_mm != 0) {
        std.debug.print("  FAIL: a numeric leaf diverges from its predecessor / definition\n", .{});
        return error.ResultMismatch;
    }
    std.debug.print("  all numeric diff cosim cases passed (fadd, fmul, reduce, exp, recip, cvt + bf16 cvt)\n\n", .{});
}
