//! Fixed P2 section scratch contract.
//!
//! Scratch is private PL storage owned by one section command. It is deliberately
//! not a `wire.TensorRange`: the host names external DDR tensors, while the section
//! controller assigns these fixed roles and addresses internally.

const std = @import("std");
pub const layout = @import("layout.zig");

pub const version: u16 = 1;
pub const ffn_kernel_version: u32 = 15;
/// First matmul contract that executes weighted RMSNorm and exact residual add
/// inside the named FFN section. `ffn_kernel_version` remains the minimum for
/// the older named-section wire command; callers must gate P3d behavior on this.
pub const p3d_kernel_version: u32 = 17;

// V1 is the first routed point. A command may contain a longer prompt; the
// controller walks it in tiles no larger than this value.
pub const query_tile_max: u32 = 4;
pub const model_dim_max: u32 = 4096;
pub const ffn_dim_max: u32 = 12288;
pub const head_dim_max: u32 = 128;
pub const heads_max: u32 = 32;
pub const kv_heads_max: u32 = 8;
pub const context_max: u32 = 8192;

pub const f32_banks_per_role: u32 = 4;
pub const f32_values_per_word: u32 = 2;
pub const f32_word_bytes: u32 = f32_values_per_word * @sizeOf(f32);
pub const f32_rows_per_group: u32 = f32_banks_per_role * f32_values_per_word;

/// Host/RTL bit contract for `SCRATCH_CTRL`. These are write-only strobes. Abort
/// has priority when several bits are written together. `resident_r` qualifies
/// `section_begin` in the same write and has no effect by itself.
pub const ScratchControl = struct {
    pub const drain_start: u32 = 1 << 0;
    pub const abort: u32 = 1 << 1;
    pub const section_begin: u32 = 1 << 2;
    pub const resident_r: u32 = 1 << 3;
};

/// Stable `SCRATCH_STATUS` layout through v17. A clean v17 `section_done`
/// coincides with `r_valid` only after the final exact residual word is committed
/// and accepted externally. Abort/fault of an active section clears tentative R
/// validity before restart-safe idle; an idle sealed R is not a command target.
pub const ScratchStatus = struct {
    pub const writer_busy: u32 = 1 << 0;
    pub const writer_done: u32 = 1 << 1;
    pub const drain_busy: u32 = 1 << 2;
    pub const drain_done: u32 = 1 << 3;
    pub const any_error: u32 = 1 << 4;
    pub const r_valid: u32 = 1 << 5;
    pub const x0_valid: u32 = 1 << 6;
    pub const x1_valid: u32 = 1 << 7;
    pub const x2_valid: u32 = 1 << 8;
    pub const producer_busy: u32 = 1 << 9;
    pub const producer_done: u32 = 1 << 10;
    pub const section_active: u32 = 1 << 11;
    pub const section_done: u32 = 1 << 12;
    pub const gate_ready: u32 = 1 << 13;
    pub const defined_mask: u32 = 0x3fff;
};

/// Coarse `SCRATCH_ERROR` categories. Bits 0..6 retain their v14/v16 meanings;
/// detailed v17 arithmetic status lives in `NORM_ERROR`/`RESIDUAL_ERROR`.
pub const ScratchError = struct {
    pub const config: u32 = 1 << 0;
    pub const writer: u32 = 1 << 1;
    pub const abort: u32 = 1 << 2;
    pub const drain: u32 = 1 << 3;
    pub const stale: u32 = 1 << 4;
    pub const section: u32 = 1 << 5;
    pub const swiglu_q8: u32 = 1 << 6;
    pub const rmsnorm: u32 = 1 << 7;
    pub const residual: u32 = 1 << 8;
    pub const fatal_mask: u32 = 0x1ff;
    pub const defined_mask: u32 = fatal_mask;
};

/// `NORM_CTRL` write-only commands. Gamma loading snapshots `MODEL_ROWS` only
/// when accepted from global shared-resource idle: legacy/raw activation ingress,
/// GEMM/Q8 ingress, scratch read/write/drain owners, retained responses, every v17
/// leaf, and the section controller must all be quiescent. Global cancellation
/// uses `ScratchControl.abort` so one abort source owns every section leaf.
pub const NormControl = struct {
    pub const load_gamma: u32 = 1 << 0;
};

/// `NORM_STATUS` lifecycle bits. Done/error bits are sticky per operation class:
/// gamma flags clear on an accepted gamma load, while norm/residual flags clear
/// atomically on an accepted section begin; global abort clears all three classes.
/// `idle` means global shared-resource quiescence: legacy/raw activation ingress,
/// GEMM/Q8 ingress, all scratch read/write/drain owners and retained responses,
/// every v17 leaf, and the section/abort-cleanup controller are quiescent. Both
/// gamma load and section begin require it. It may coexist with sealed gamma/R.
/// Any accepted section that terminates with an error clears `gamma_valid` and
/// requires a clean gamma reload, even if the fault occurs after norm completion.
pub const NormStatus = struct {
    pub const gamma_busy: u32 = 1 << 0;
    pub const gamma_done: u32 = 1 << 1;
    pub const gamma_error: u32 = 1 << 2;
    pub const gamma_valid: u32 = 1 << 3;
    pub const norm_busy: u32 = 1 << 4;
    pub const norm_done: u32 = 1 << 5;
    pub const norm_error: u32 = 1 << 6;
    pub const residual_busy: u32 = 1 << 7;
    pub const residual_done: u32 = 1 << 8;
    pub const residual_error: u32 = 1 << 9;
    pub const idle: u32 = 1 << 10;
    pub const busy_mask: u32 = gamma_busy | norm_busy | residual_busy;
    pub const done_mask: u32 = gamma_done | norm_done | residual_done;
    pub const error_mask: u32 = gamma_error | norm_error | residual_error;
    pub const defined_mask: u32 = 0x7ff;
};

/// `NORM_ERROR` preserves the raw standalone status layouts. Q8 quantizer detail
/// remains in `QUANT_STATUS`; `controller` covers arbitration/framing ownership
/// failures that no RMSNorm leaf can diagnose.
pub const NormError = struct {
    pub const scalar_shift: u5 = 0;
    pub const gamma_shift: u5 = 23;
    pub const controller: u32 = 1 << 27;

    pub fn scalar(status: u23) u32 {
        return @as(u32, status) << scalar_shift;
    }

    pub fn gamma(status: u4) u32 {
        return @as(u32, status) << gamma_shift;
    }

    pub const fatal_mask: u32 = 0x0fff_ffff;
    pub const defined_mask: u32 = fatal_mask;
};

/// `RESIDUAL_ERROR` is the raw seven-bit `ResidualAddStatus` value. Every bit
/// outside `defined_mask` reads zero, as do unlisted high bits in the other RO
/// status/error registers above.
pub const ResidualError = struct {
    pub const raw_mask: u32 = 0x7f;
    pub const defined_mask: u32 = raw_mask;
};

/// Pre-start legality shared by the v17 controller and its software fallback
/// gate. RMSNorm's fixed mean conversion requires a power-of-two row count.
pub fn p3dModelRowsValid(rows: u32) bool {
    return rows >= 128 and rows <= model_dim_max and (rows & (rows - 1)) == 0;
}

/// The inverse-RMS leaf accepts positive, finite, normal binary32 epsilon.
/// Subnormal epsilon is deliberately rejected rather than silently flushed.
pub fn p3dNormEpsValid(eps_bits: u32) bool {
    const sign = eps_bits >> 31;
    const exponent: u8 = @truncate(eps_bits >> 23);
    return sign == 0 and exponent != 0 and exponent != 0xff;
}

/// Diagnostic status bits shared with `section_rmsnorm_loader.v`.
pub const RmsNormLoaderStatus = struct {
    pub const bad_cfg: u4 = 1 << 0;
    pub const frame: u4 = 1 << 1;
    pub const sink: u4 = 1 << 2;
    pub const internal: u4 = 1 << 3;
    pub const fatal_mask: u4 = bad_cfg | frame | sink | internal;
};

/// Diagnostic status bits shared with `section_rmsnorm_maxexp.v`.
pub const RmsNormMaxExpStatus = struct {
    pub const bad_cfg: u6 = 1 << 0;
    pub const nonfinite: u6 = 1 << 1;
    pub const frame: u6 = 1 << 2;
    pub const scratch: u6 = 1 << 3;
    pub const internal: u6 = 1 << 4;
    pub const subnormal_warning: u6 = 1 << 5;
    pub const fatal_mask: u6 = bad_cfg | nonfinite | frame | scratch | internal;
};

pub const RmsNormMaxExpError = error{
    InvalidShape,
    Nonfinite,
};

pub const RmsNormMaxExpResult = struct {
    max_exp: u8,
    subnormal_warning: bool,
};

/// Exact token-local oracle for the P3d prior-exponent scan. Zeros and
/// subnormals do not contribute to the maximum; subnormals set a warning that
/// integration must reject before architectural publication.
pub fn rmsNormMaxExp(
    values: []const u32,
    rows: u32,
) RmsNormMaxExpError!RmsNormMaxExpResult {
    if (rows < f32_rows_per_group or rows > model_dim_max or
        rows % f32_rows_per_group != 0 or values.len != rows)
        return error.InvalidShape;

    var max_exp: u8 = 0;
    var subnormal_warning = false;
    for (values) |bits| {
        const exponent: u8 = @truncate(bits >> 23);
        const mantissa: u23 = @truncate(bits);
        if (exponent == 0xff) return error.Nonfinite;
        if (exponent == 0) {
            if (mantissa != 0) subnormal_warning = true;
        } else {
            max_exp = @max(max_exp, exponent);
        }
    }
    return .{
        .max_exp = max_exp,
        .subnormal_warning = subnormal_warning,
    };
}

/// Diagnostic status bits shared with `section_rmsnorm_sumsq.v`.
pub const RmsNormSumsqStatus = struct {
    pub const bad_cfg: u7 = 1 << 0;
    pub const nonfinite: u7 = 1 << 1;
    pub const max_mismatch: u7 = 1 << 2;
    pub const frame: u7 = 1 << 3;
    pub const scratch: u7 = 1 << 4;
    pub const internal: u7 = 1 << 5;
    pub const subnormal_warning: u7 = 1 << 6;
    pub const fatal_mask: u7 = bad_cfg | nonfinite | max_mismatch | frame | scratch | internal;
};

/// Coarse fail-closed status at the composed RMSNorm reduction boundary.
pub const RmsNormFrontendStatus = struct {
    pub const bad_cfg: u7 = 1 << 0;
    pub const loader: u7 = 1 << 1;
    pub const max_exp: u7 = 1 << 2;
    pub const sum_sq: u7 = 1 << 3;
    pub const scratch: u7 = 1 << 4;
    pub const subnormal: u7 = 1 << 5;
    pub const internal: u7 = 1 << 6;
    pub const fatal_mask: u7 = bad_cfg | loader | max_exp | sum_sq |
        scratch | subnormal | internal;
};

/// Diagnostic status bits shared with `section_rmsnorm_inv.v`.
pub const RmsNormInvStatus = struct {
    pub const bad_cfg: u4 = 1 << 0;
    pub const frame: u4 = 1 << 1;
    pub const arithmetic: u4 = 1 << 2;
    pub const internal: u4 = 1 << 3;
    pub const fatal_mask: u4 = bad_cfg | frame | arithmetic | internal;
};

/// Per-result diagnostics shared with `section_rmsnorm_mul_rne.v`.
pub const RmsNormMulStatus = struct {
    pub const nonfinite_input: u2 = 1 << 0;
    pub const overflow: u2 = 1 << 1;
    pub const fatal_mask: u2 = nonfinite_input | overflow;
};

pub const RmsNormMulResult = struct {
    bits: u32,
    status: u2,
};

/// Per-result diagnostics shared with `section_residual_add_rne.v`.
pub const ResidualAddArithmeticStatus = struct {
    pub const nonfinite_input: u2 = 1 << 0;
    pub const overflow: u2 = 1 << 1;
    pub const fatal_mask: u2 = nonfinite_input | overflow;
};

pub const ResidualAddResult = struct {
    bits: u32,
    status: u2,
};

/// Run diagnostics shared with `section_residual_add.v`.
pub const ResidualAddStatus = struct {
    pub const bad_cfg: u7 = 1 << 0;
    pub const frame: u7 = 1 << 1;
    pub const scratch_read: u7 = 1 << 2;
    pub const scratch_write: u7 = 1 << 3;
    pub const arithmetic_shift: u3 = 4;
    pub const internal: u7 = 1 << 6;

    pub fn arithmetic(status: u2) u7 {
        return @as(u7, status) << arithmetic_shift;
    }

    pub const fatal_mask: u7 = 0x7f;
};

/// Diagnostics shared with `section_rmsnorm_weighted_source.v`.
pub const RmsNormGammaStatus = struct {
    pub const bad_cfg: u4 = 1 << 0;
    pub const frame: u4 = 1 << 1;
    pub const nonfinite: u4 = 1 << 2;
    pub const internal: u4 = 1 << 3;
    pub const fatal_mask: u4 = bad_cfg | frame | nonfinite | internal;
};

pub const RmsNormWeightedSourceStatus = struct {
    pub const bad_cfg: u9 = 1 << 0;
    pub const gamma: u9 = 1 << 1;
    pub const inverse_frame: u9 = 1 << 2;
    pub const scratch: u9 = 1 << 3;
    pub const mul1_shift: u4 = 4;
    pub const mul2_shift: u4 = 6;
    pub const internal: u9 = 1 << 8;

    pub fn mul1(status: u2) u9 {
        return @as(u9, status) << mul1_shift;
    }

    pub fn mul2(status: u2) u9 {
        return @as(u9, status) << mul2_shift;
    }

    pub const fatal_mask: u9 = 0x1ff;
};

/// Aggregate diagnostics shared with `section_rmsnorm_q8_source.v`. Both child
/// layouts remain verbatim so software can attribute a failed tentative stream
/// without reinterpreting either leaf's raw status.
pub const RmsNormQ8SourceStatus = struct {
    pub const weighted_shift: u4 = 0;
    pub const q8_shift: u4 = 9;
    pub const internal: u16 = 1 << 15;

    pub fn weighted(status: u9) u16 {
        return @as(u16, status) << weighted_shift;
    }

    pub fn q8(status: u6) u16 {
        return @as(u16, status) << q8_shift;
    }

    pub const fatal_mask: u16 = 0xffff;
};

pub const RmsNormWeightedResult = struct {
    normalized_bits: u32,
    bits: u32,
    mul1_status: u2,
    mul2_status: u2,
};

/// Diagnostics shared with `section_rmsnorm_reduce.v`. Child status layouts are
/// preserved verbatim so the eventual v17 controller can distinguish scratch,
/// framing, subnormal, and arithmetic failures without re-decoding a coarse bit.
pub const RmsNormReduceStatus = struct {
    pub const bad_cfg: u13 = 1 << 0;
    pub const frontend_shift: u4 = 1;
    pub const inverse_shift: u4 = 8;
    pub const internal: u13 = 1 << 12;

    pub fn frontend(status: u7) u13 {
        return @as(u13, status) << frontend_shift;
    }

    pub fn inverse(status: u4) u13 {
        return @as(u13, status) << inverse_shift;
    }

    pub const fatal_mask: u13 = 0x1fff;
};

/// Aggregate diagnostics shared with `section_rmsnorm_scalar_pipeline.v`.
/// Reduction and weighted-source child layouts remain verbatim; the top bit is
/// reserved for scalar-pipeline ownership, framing, and lifecycle failures.
pub const RmsNormScalarPipelineStatus = struct {
    pub const reduce_shift: u5 = 0;
    pub const weighted_shift: u5 = 13;
    pub const internal: u23 = 1 << 22;

    pub fn reduce(status: u13) u23 {
        return @as(u23, status) << reduce_shift;
    }

    pub fn weighted(status: u9) u23 {
        return @as(u23, status) << weighted_shift;
    }

    pub const fatal_mask: u23 = 0x7f_ffff;
};

/// Aggregate diagnostics shared with `section_rmsnorm_q8_pipeline.v`. Both
/// child layouts remain verbatim; the top bit is reserved for wrapper ownership
/// and lifecycle failures.
pub const RmsNormQ8PipelineStatus = struct {
    pub const reduce_shift: u5 = 0;
    pub const source_shift: u5 = 13;
    pub const internal: u30 = 1 << 29;

    pub fn reduce(status: u13) u30 {
        return @as(u30, status) << reduce_shift;
    }

    pub fn source(status: u16) u30 {
        return @as(u30, status) << source_shift;
    }

    pub const fatal_mask: u30 = 0x3fff_ffff;
};

pub const RmsNormInvError = error{
    InvalidShape,
    InvalidEpsilon,
    InvalidRecord,
    ArithmeticOverflow,
};

fn roundRightEvenU48(value: u64, shift: u16) u64 {
    if (shift == 0) return value;
    if (shift > 48) return 0;

    const amount: u6 = @intCast(shift);
    const quotient = value >> amount;
    const remainder_mask = (@as(u64, 1) << amount) - 1;
    const remainder = value & remainder_mask;
    const halfway = @as(u64, 1) << @as(u6, @intCast(shift - 1));
    const round_up = remainder > halfway or
        (remainder == halfway and (quotient & 1) != 0);
    return quotient + @intFromBool(round_up);
}

/// Integer-only oracle for the serialized exact finite binary32 multiplier.
/// Non-finite inputs are rejected as deterministic +0; every finite input uses
/// gradual underflow and round-to-nearest-even.
pub fn rmsNormMulRneBits(a_bits: u32, b_bits: u32) RmsNormMulResult {
    const sign = (a_bits ^ b_bits) & 0x8000_0000;
    const exponent_a: u8 = @truncate(a_bits >> 23);
    const exponent_b: u8 = @truncate(b_bits >> 23);
    const fraction_a: u23 = @truncate(a_bits);
    const fraction_b: u23 = @truncate(b_bits);

    if (exponent_a == 0xff or exponent_b == 0xff)
        return .{ .bits = 0, .status = RmsNormMulStatus.nonfinite_input };
    if ((exponent_a == 0 and fraction_a == 0) or
        (exponent_b == 0 and fraction_b == 0))
        return .{ .bits = sign, .status = 0 };

    const significand_a: u24 =
        (@as(u24, @intFromBool(exponent_a != 0)) << 23) | fraction_a;
    const significand_b: u24 =
        (@as(u24, @intFromBool(exponent_b != 0)) << 23) | fraction_b;
    const effective_a: u16 = if (exponent_a == 0) 1 else exponent_a;
    const effective_b: u16 = if (exponent_b == 0) 1 else exponent_b;
    const exponent_sum: u16 = effective_a + effective_b;
    const product: u64 = @as(u64, significand_a) * significand_b;
    const lead: u6 = @intCast(63 - @clz(product));
    var biased: i16 = @as(i16, @intCast(exponent_sum)) +
        @as(i16, lead) - 173;

    if (biased >= 1) {
        var retained = if (lead >= 23)
            roundRightEvenU48(product, lead - 23)
        else
            product << @as(u6, @intCast(23 - lead));
        if (retained >= (@as(u64, 1) << 24)) {
            retained >>= 1;
            biased += 1;
        }
        if (biased >= 255)
            return .{
                .bits = sign | 0x7f80_0000,
                .status = RmsNormMulStatus.overflow,
            };
        return .{
            .bits = sign |
                (@as(u32, @intCast(biased)) << 23) |
                (@as(u32, @truncate(retained)) & 0x007f_ffff),
            .status = 0,
        };
    }

    const subnormal_shift = @as(i16, 151) - @as(i16, @intCast(exponent_sum));
    const fraction = if (subnormal_shift > 0)
        roundRightEvenU48(product, @intCast(subnormal_shift))
    else blk: {
        const left_shift: u16 = @intCast(-subnormal_shift);
        if (left_shift >= 64) break :blk std.math.maxInt(u64);
        break :blk product << @as(u6, @intCast(left_shift));
    };
    if (fraction >= (@as(u64, 1) << 23))
        return .{ .bits = sign | 0x0080_0000, .status = 0 };
    return .{ .bits = sign | @as(u32, @truncate(fraction)), .status = 0 };
}

/// Integer-only oracle for the exact finite binary32 residual addition leaf.
/// Finite operands are converted to an exact integer in minimum-subnormal
/// units, added without reassociation, then rounded once to binary32 RNE.
pub fn residualAddRneBits(a_bits: u32, b_bits: u32) ResidualAddResult {
    const exponent_a: u8 = @truncate(a_bits >> 23);
    const exponent_b: u8 = @truncate(b_bits >> 23);
    const fraction_a: u23 = @truncate(a_bits);
    const fraction_b: u23 = @truncate(b_bits);
    if (exponent_a == 0xff or exponent_b == 0xff)
        return .{
            .bits = 0,
            .status = ResidualAddArithmeticStatus.nonfinite_input,
        };

    const a_zero = exponent_a == 0 and fraction_a == 0;
    const b_zero = exponent_b == 0 and fraction_b == 0;
    if (a_zero and b_zero)
        return .{ .bits = (a_bits & b_bits) & 0x8000_0000, .status = 0 };
    if (a_zero) return .{ .bits = b_bits, .status = 0 };
    if (b_zero) return .{ .bits = a_bits, .status = 0 };

    const significand_a: u24 =
        (@as(u24, @intFromBool(exponent_a != 0)) << 23) | fraction_a;
    const significand_b: u24 =
        (@as(u24, @intFromBool(exponent_b != 0)) << 23) | fraction_b;
    const effective_a: u8 = if (exponent_a == 0) 1 else exponent_a;
    const effective_b: u8 = if (exponent_b == 0) 1 else exponent_b;
    const magnitude_a = @as(u278, significand_a) << @intCast(effective_a - 1);
    const magnitude_b = @as(u278, significand_b) << @intCast(effective_b - 1);
    const sign_a = a_bits >> 31 != 0;
    const sign_b = b_bits >> 31 != 0;

    var sign = sign_a;
    var magnitude: u278 = undefined;
    if (sign_a == sign_b) {
        magnitude = magnitude_a + magnitude_b;
    } else if (magnitude_a > magnitude_b) {
        magnitude = magnitude_a - magnitude_b;
    } else if (magnitude_b > magnitude_a) {
        magnitude = magnitude_b - magnitude_a;
        sign = sign_b;
    } else {
        // Exact cancellation is +0 in round-to-nearest mode.
        return .{ .bits = 0, .status = 0 };
    }

    const sign_bits: u32 = @as(u32, @intFromBool(sign)) << 31;
    const lead: u9 = @intCast(277 - @clz(magnitude));
    if (lead <= 22)
        return .{
            .bits = sign_bits | @as(u32, @truncate(magnitude)),
            .status = 0,
        };

    const shift: u9 = lead - 23;
    var retained = magnitude >> shift;
    if (shift != 0) {
        const remainder_mask = (@as(u278, 1) << shift) - 1;
        const remainder = magnitude & remainder_mask;
        const halfway = @as(u278, 1) << (shift - 1);
        if (remainder > halfway or
            (remainder == halfway and (retained & 1) != 0))
            retained += 1;
    }

    var biased: u9 = lead - 22;
    if (retained >= (@as(u278, 1) << 24)) {
        retained >>= 1;
        biased += 1;
    }
    if (biased >= 255)
        return .{
            .bits = sign_bits | 0x7f80_0000,
            .status = ResidualAddArithmeticStatus.overflow,
        };
    return .{
        .bits = sign_bits |
            (@as(u32, biased) << 23) |
            (@as(u32, @truncate(retained)) & 0x007f_ffff),
        .status = 0,
    };
}

/// Exact PS-order weighted RMSNorm scalar oracle. The second multiply is not
/// evaluated after a diagnosed first-stage result because hardware suppresses
/// that scalar and aborts the surrounding tentative Q8 run.
pub fn rmsNormWeightedBits(
    x_bits: u32,
    inverse_bits: u32,
    gamma_bits: u32,
) RmsNormWeightedResult {
    const normalized = rmsNormMulRneBits(x_bits, inverse_bits);
    if (normalized.status != 0) return .{
        .normalized_bits = normalized.bits,
        .bits = 0,
        .mul1_status = normalized.status,
        .mul2_status = 0,
    };
    const weighted = rmsNormMulRneBits(normalized.bits, gamma_bits);
    return .{
        .normalized_bits = normalized.bits,
        .bits = weighted.bits,
        .mul1_status = 0,
        .mul2_status = weighted.status,
    };
}

pub fn rmsNormGammaBank(pair_word: u11) u2 {
    return @truncate(pair_word);
}

pub fn rmsNormGammaAddress(pair_word: u11) u9 {
    return @truncate(pair_word >> 2);
}

pub const RmsNormInvResult = struct {
    mean_bits: u32,
    adjusted_mean_bits: u32,
    inv_rms_bits: u32,
};

pub const rmsnorm_sumsq_synthetic_model_relative_limit: f64 = 3e-5;
pub const rmsnorm_sumsq_adversarial_relative_limit: f64 = 1e-3;
pub const rmsnorm_inv_scalar_relative_limit: f64 = 6e-6;

pub const RmsNormSumsqError = error{
    InvalidShape,
    InvalidMaxExponent,
    Nonfinite,
    MaxMismatch,
    InternalOverflow,
};

pub const RmsNormSumsqResult = struct {
    sum_sq: u48,
    subnormal_warning: bool,
};

/// Exact integer oracle for the deliberately truncated P3d slice-1 reduction.
/// `max_exp` is the configured raw binary32 exponent field for one token.
pub fn rmsNormSumsqFixed(
    values: []const u32,
    rows: u32,
    max_exp: u8,
) RmsNormSumsqError!RmsNormSumsqResult {
    if (rows < 8 or rows > model_dim_max or rows % f32_rows_per_group != 0 or
        values.len != rows)
        return error.InvalidShape;
    if (max_exp == 0xff) return error.InvalidMaxExponent;

    var sum: u48 = 0;
    var subnormal_warning = false;
    var saw_max_exp = false;
    for (values) |bits| {
        const exponent: u8 = @truncate(bits >> 23);
        const mantissa: u23 = @truncate(bits);
        if (exponent == 0xff) return error.Nonfinite;
        if (exponent == 0) {
            if (mantissa != 0) subnormal_warning = true;
            continue;
        }
        if (max_exp == 0 or exponent > max_exp) return error.MaxMismatch;
        if (exponent == max_exp) saw_max_exp = true;

        const delta: u8 = max_exp - exponent;
        const significand: u24 = (@as(u24, 1) << 23) | mantissa;
        const quant: u18 = if (delta >= 18)
            0
        else
            @truncate(significand >> @intCast(6 + delta));
        const product: u36 = @as(u36, quant) * @as(u36, quant);
        const added = @addWithOverflow(sum, @as(u48, product));
        if (added[1] != 0) return error.InternalOverflow;
        sum = added[0];
    }
    if (max_exp != 0 and !saw_max_exp) return error.MaxMismatch;
    return .{ .sum_sq = sum, .subnormal_warning = subnormal_warning };
}

/// Decode the fixed reduction identity into a host-side characterization value.
pub fn rmsNormSumsqMeanF64(sum_sq: u48, rows: u32, max_exp: u8) f64 {
    if (sum_sq == 0 or rows == 0) return 0;
    const scaled = @as(f64, @floatFromInt(sum_sq)) /
        @as(f64, @floatFromInt(rows));
    return std.math.ldexp(scaled, 2 * (@as(i32, max_exp) - 144));
}

fn truncFp32MulBits(a: u32, b: u32) u32 {
    const sign = (a ^ b) & 0x8000_0000;
    const ea: u32 = (a >> 23) & 0xff;
    const eb: u32 = (b >> 23) & 0xff;
    if (ea == 0 or eb == 0) return sign;

    const siga: u64 = 0x80_0000 | (a & 0x7f_ffff);
    const sigb: u64 = 0x80_0000 | (b & 0x7f_ffff);
    const product = siga * sigb;
    const renormalizes = ((product >> 47) & 1) != 0;
    const mantissa: u32 = @truncate(if (renormalizes)
        (product >> 24) & 0x7f_ffff
    else
        (product >> 23) & 0x7f_ffff);
    const exponent = @as(i32, @intCast(ea)) + @as(i32, @intCast(eb)) - 127 +
        @as(i32, @intFromBool(renormalizes));
    if (exponent <= 0) return sign;
    if (exponent >= 255) return sign | 0x7f7f_ffff;
    return sign | (@as(u32, @intCast(exponent)) << 23) | mantissa;
}

fn truncFp32AddBits(a: u32, b: u32) u32 {
    const sa = a >> 31;
    const sb = b >> 31;
    const ea: u32 = (a >> 23) & 0xff;
    const eb: u32 = (b >> 23) & 0xff;
    const ma = a & 0x7f_ffff;
    const mb = b & 0x7f_ffff;
    const a_zero = ea == 0;
    const b_zero = eb == 0;

    const a_ge_b = ea >= eb;
    const exp_big = if (a_ge_b) ea else eb;
    const exp_diff = if (a_ge_b) ea - eb else eb - ea;
    const mant_big: u32 = 0x80_0000 | (if (a_ge_b) ma else mb);
    const mant_small: u32 = 0x80_0000 | (if (a_ge_b) mb else ma);
    const sign_big = if (a_ge_b) sa else sb;
    const sign_small = if (a_ge_b) sb else sa;
    const mant_small_aligned = if (exp_diff > 24) 0 else mant_small >> @intCast(exp_diff);
    const small_bigger = mant_small_aligned > mant_big;
    const m1 = if (small_bigger) mant_small_aligned else mant_big;
    const m2 = if (small_bigger) mant_big else mant_small_aligned;
    const result_sign = if (small_bigger) sign_small else sign_big;
    const mant_sum: u32 = if (sign_big == sign_small) m1 + m2 else m1 - m2;

    if (a_zero) return b;
    if (b_zero) return a;
    if (mant_sum == 0) return result_sign << 31;

    const lead_pos: u5 = @intCast(31 - @clz(mant_sum));
    const shift_right = lead_pos > 23;
    const right_amount: u5 = if (shift_right) lead_pos - 23 else 0;
    const left_amount: u5 = if (lead_pos < 23) 23 - lead_pos else 0;
    const exponent = if (shift_right)
        @as(i32, @intCast(exp_big)) + @as(i32, right_amount)
    else
        @as(i32, @intCast(exp_big)) - @as(i32, left_amount);
    if (exponent <= 0) return result_sign << 31;
    if (exponent >= 255) return (result_sign << 31) | 0x7f7f_ffff;
    const normalized = if (shift_right)
        mant_sum >> right_amount
    else
        mant_sum << left_amount;
    return (result_sign << 31) | (@as(u32, @intCast(exponent)) << 23) |
        (normalized & 0x7f_ffff);
}

/// Round the fixed reduction identity once to positive binary32. Values below
/// the normal range flush to zero, matching the numeric leaf convention.
pub fn rmsNormFixedMeanBits(
    sum_sq: u48,
    rows: u32,
    max_exp: u8,
) RmsNormInvError!u32 {
    if (rows < f32_rows_per_group or rows > model_dim_max or
        (rows & (rows - 1)) != 0)
        return error.InvalidShape;
    if (max_exp == 0xff or ((sum_sq == 0) != (max_exp == 0)))
        return error.InvalidRecord;
    if (sum_sq == 0) return 0;

    var msb: u6 = @intCast(47 - @clz(sum_sq));
    var significand: u64 = undefined;
    if (msb <= 23) {
        significand = @as(u64, sum_sq) << @intCast(23 - msb);
    } else {
        const shift: u6 = msb - 23;
        significand = @as(u64, sum_sq) >> shift;
        const remainder_mask = (@as(u64, 1) << shift) - 1;
        const remainder = @as(u64, sum_sq) & remainder_mask;
        const halfway = @as(u64, 1) << @intCast(shift - 1);
        if (remainder > halfway or
            (remainder == halfway and (significand & 1) != 0))
            significand += 1;
        if (significand == (@as(u64, 1) << 24)) {
            significand >>= 1;
            msb += 1;
        }
    }

    const row_shift: i32 = @intCast(@ctz(rows));
    const unbiased = @as(i32, msb) +
        2 * (@as(i32, max_exp) - 144) - row_shift;
    const biased = unbiased + 127;
    if (biased <= 0) return 0;
    if (biased >= 255) return error.ArithmeticOverflow;
    return (@as(u32, @intCast(biased)) << 23) |
        @as(u23, @truncate(significand));
}

/// Bit-faithful model of the serialized two-step Newton leaf. The seed is the
/// classic affine binary32 estimate; every refinement uses the repository's
/// truncating `fmul`/`fadd` semantics rather than host IEEE arithmetic.
pub fn rmsNormInvSqrtApproxBits(value_bits: u32) RmsNormInvError!u32 {
    const exponent: u8 = @truncate(value_bits >> 23);
    // The shared multiplier flushes subnormal intermediates. This bounded input
    // range keeps x/2 and both y*y products normal through the two refinements.
    if ((value_bits >> 31) != 0 or exponent < 2 or exponent > 251)
        return error.InvalidRecord;

    const half_value = value_bits - (1 << 23);
    var estimate = @as(u32, 0x5f37_59df) - (value_bits >> 1);
    for (0..2) |_| {
        const square = truncFp32MulBits(estimate, estimate);
        const scaled = truncFp32MulBits(half_value, square);
        const correction = truncFp32AddBits(0x3fc0_0000, scaled ^ 0x8000_0000);
        estimate = truncFp32MulBits(estimate, correction);
    }
    return estimate;
}

/// Full scalar oracle for the P3d inverse-RMS leaf. Epsilon is restricted to a
/// positive finite normal so unsupported requests can fall back before launch.
pub fn rmsNormInvFixed(
    sum_sq: u48,
    rows: u32,
    max_exp: u8,
    eps_bits: u32,
) RmsNormInvError!RmsNormInvResult {
    const eps_exp: u8 = @truncate(eps_bits >> 23);
    if ((eps_bits >> 31) != 0 or eps_exp == 0 or eps_exp == 0xff)
        return error.InvalidEpsilon;
    const mean_bits = try rmsNormFixedMeanBits(sum_sq, rows, max_exp);
    const adjusted = truncFp32AddBits(mean_bits, eps_bits);
    return .{
        .mean_bits = mean_bits,
        .adjusted_mean_bits = adjusted,
        .inv_rms_bits = try rmsNormInvSqrtApproxBits(adjusted),
    };
}

/// High-precision-enough characterization of the exact-real input sum. Every
/// individual binary32 square is exact in f64; only the positive accumulation
/// can round, far below the diagnostic approximation bound used here.
pub fn rmsNormSumsqExactMeanF64(values: []const u32) f64 {
    if (values.len == 0) return 0;
    var sum: f64 = 0;
    for (values) |bits| {
        const exponent: u8 = @truncate(bits >> 23);
        const mantissa: u23 = @truncate(bits);
        if (exponent == 0xff) return std.math.nan(f64);
        if (exponent == 0 and mantissa == 0) continue;
        const value = if (exponent == 0)
            std.math.ldexp(@as(f64, @floatFromInt(mantissa)), -149)
        else
            std.math.ldexp(
                @as(f64, @floatFromInt((@as(u24, 1) << 23) | mantissa)),
                @as(i32, exponent) - 150,
            );
        sum += value * value;
    }
    return sum / @as(f64, @floatFromInt(values.len));
}

/// Conservative error envelope for prior-maximum 18-bit quantization plus the
/// 48-bit aligned accumulation. It is below 0.001 for every legal row count.
pub fn rmsNormSumsqRelativeErrorBound(rows: u32) f64 {
    const n = @as(f64, @floatFromInt(rows));
    return 2.0 * @sqrt(n) / 131072.0 + n / 17179869184.0;
}

/// Named storage roles. These are schedule contracts, not host-addressable banks.
pub const F32Role = enum(u8) {
    residual,
    x0,
    x1,
    x2,

    pub fn rowCapacity(self: F32Role) u32 {
        return switch (self) {
            .residual, .x2 => model_dim_max,
            .x0, .x1 => ffn_dim_max,
        };
    }

    pub fn bytes(self: F32Role) u64 {
        return @as(u64, query_tile_max) * self.rowCapacity() * @sizeOf(f32);
    }
};

pub const F32Location = struct {
    bank: u2,
    address: u32,
};

/// Number of 256-bit row groups reserved for one token in a role.  Each group
/// contributes one 64-bit word to every physical bank.
pub fn f32GroupsPerToken(role: F32Role) u32 {
    return role.rowCapacity() / f32_rows_per_group;
}

/// Per-bank address span reserved for a role across the maximum query tile.
pub fn f32RoleSpan(role: F32Role) u32 {
    return query_tile_max * f32GroupsPerToken(role);
}

/// First physical address occupied by a role in each of the four banks.
pub fn f32RoleBase(role: F32Role) u32 {
    return switch (role) {
        .residual => 0,
        .x0 => f32RoleSpan(.residual),
        .x1 => f32RoleSpan(.residual) + f32RoleSpan(.x0),
        .x2 => f32RoleSpan(.residual) + f32RoleSpan(.x0) + f32RoleSpan(.x1),
    };
}

/// Physical word depth of each 64-bit bank across all roles.
pub fn f32BankDepth() u32 {
    return f32RoleBase(.x2) + f32RoleSpan(.x2);
}

pub const Error = error{
    InvalidBank,
    InvalidTokenCount,
    InvalidModelDim,
    InvalidFfnDim,
    InvalidHeadDim,
    InvalidHeadCount,
    InvalidKvCount,
    InvalidRow,
    InvalidRowPair,
    InvalidSubblock,
};

/// Map an even/odd f32 row pair into four physical 64-bit banks. Eight adjacent
/// logical rows occupy one address across the four banks. GEMM's two-row result
/// beats can therefore be written directly while consumers read a 256-bit group.
pub fn f32Location(role: F32Role, token: u32, even_row: u32) Error!F32Location {
    if (token >= query_tile_max) return error.InvalidTokenCount;
    if (even_row >= role.rowCapacity()) return error.InvalidRow;
    if (even_row % f32_values_per_word != 0) return error.InvalidRowPair;
    return .{
        .bank = @intCast((even_row % f32_rows_per_group) / f32_values_per_word),
        .address = token * f32GroupsPerToken(role) + even_row / f32_rows_per_group,
    };
}

/// Map a row pair to its absolute address in the shared four-bank memory.
pub fn f32PhysicalLocation(role: F32Role, token: u32, even_row: u32) Error!F32Location {
    const local = try f32Location(role, token, even_row);
    return .{
        .bank = local.bank,
        .address = f32RoleBase(role) + local.address,
    };
}

/// Map one token-major 64-bit residual input word to the direct R scratch port.
pub fn rmsNormResidualWordLocation(rows: u32, tokens: u32, ordinal: u32) Error!F32Location {
    if (rows < f32_rows_per_group or rows > model_dim_max or
        rows % f32_rows_per_group != 0)
        return error.InvalidModelDim;
    if (tokens == 0 or tokens > query_tile_max)
        return error.InvalidTokenCount;
    const words_per_token = rows / f32_values_per_word;
    if (ordinal >= words_per_token * tokens) return error.InvalidRowPair;
    const token = ordinal / words_per_token;
    const word = ordinal % words_per_token;
    return f32PhysicalLocation(.residual, token, word * f32_values_per_word);
}

/// The Q8 activation is the GEMM kernel's native pair of memories: 32 int8
/// values plus one f16 scale. DDR's padded 40-byte stream is only an ingress
/// representation and is not the scratch layout.
pub const Q8Location = struct {
    address: u32,
};

pub const q8_column_capacity: u32 = 8;

// P3's compact DOWN-input store holds one native Q8 record per FFN block and
// token in each ping-pong bank. The block-major physical layout accepts the
// producer schedule directly; consumers can request the same records in
// token-major order without a transpose copy.
pub const q8_buffer_bank_count: u32 = 2;
pub const q8_buffer_token_capacity: u32 = query_tile_max;
pub const q8_buffer_block_capacity: u32 = ffn_dim_max / layout.q8_block;
pub const q8_buffer_records_per_bank: u32 =
    q8_buffer_token_capacity * q8_buffer_block_capacity;
pub const q8_buffer_record_capacity: u32 =
    q8_buffer_bank_count * q8_buffer_records_per_bank;

// The streaming FFN pairer consumes one 32-value Q8 block as four adjacent
// eight-value FP32 scratch groups. Its GATE tag maps directly to the resident
// X1/UP role's role-local read group.
pub const ffn_pair_groups_per_block: u32 = layout.q8_block / f32_rows_per_group;

pub const FfnPairTag = struct {
    block: u32,
    token: u32,
    group: u32,
};

/// Decode the native GEMM arrival order. Each 32-row block arrives as a lower
/// 16-row sweep across all tokens followed by the corresponding upper sweep.
pub fn ffnPairNativeTag(tokens: u32, ordinal: u32) Error!FfnPairTag {
    if (tokens == 0 or tokens > query_tile_max) return error.InvalidTokenCount;
    const groups_per_block = tokens * ffn_pair_groups_per_block;
    const block = ordinal / groups_per_block;
    if (block >= q8_buffer_block_capacity) return error.InvalidSubblock;
    const local = ordinal % groups_per_block;
    const half_span = tokens * 2;
    const half = local / half_span;
    const within_half = local % half_span;
    return .{
        .block = block,
        .token = within_half / 2,
        .group = half * 2 + within_half % 2,
    };
}

/// Decode the canonical consumer order used for X1 reads and SwiGLU output.
pub fn ffnPairCanonicalTag(tokens: u32, ordinal: u32) Error!FfnPairTag {
    if (tokens == 0 or tokens > query_tile_max) return error.InvalidTokenCount;
    const groups_per_block = tokens * ffn_pair_groups_per_block;
    const block = ordinal / groups_per_block;
    if (block >= q8_buffer_block_capacity) return error.InvalidSubblock;
    const local = ordinal % groups_per_block;
    return .{
        .block = block,
        .token = local / ffn_pair_groups_per_block,
        .group = local % ffn_pair_groups_per_block,
    };
}

pub fn ffnPairScratchGroup(block: u32, group: u32) Error!u32 {
    if (block >= q8_buffer_block_capacity) return error.InvalidSubblock;
    if (group >= ffn_pair_groups_per_block) return error.InvalidRow;
    return block * ffn_pair_groups_per_block + group;
}

pub const Q8BufferLocation = struct {
    address: u32,
};

pub fn q8BufferLocation(bank: u32, token: u32, block: u32) Error!Q8BufferLocation {
    if (bank >= q8_buffer_bank_count) return error.InvalidBank;
    if (token >= q8_buffer_token_capacity) return error.InvalidTokenCount;
    if (block >= q8_buffer_block_capacity) return error.InvalidSubblock;
    return .{ .address = bank * q8_buffer_records_per_bank +
        block * q8_buffer_token_capacity + token };
}

pub fn q8Location(subblock: u32, token: u32) Error!Q8Location {
    if (token >= query_tile_max) return error.InvalidTokenCount;
    if (subblock >= layout.max_sub_index) return error.InvalidSubblock;
    return .{ .address = subblock * q8_column_capacity + token };
}

pub const FfnShape = struct {
    token_count: u32,
    model_dim: u32,
    ffn_dim: u32,

    pub fn validate(self: FfnShape) Error!void {
        try validateTokenCount(self.token_count);
        try validateModelDim(self.model_dim);
        if (self.ffn_dim == 0 or self.ffn_dim > ffn_dim_max or self.ffn_dim % layout.q1_block != 0)
            return error.InvalidFfnDim;
    }
};

/// Host-visible section extent. The controller tiles this full command into
/// `FfnShape` chunks, so it accepts the full context rather than one PL tile.
pub const FfnCommandShape = struct {
    token_count: u32,
    model_dim: u32,
    ffn_dim: u32,

    pub fn validate(self: FfnCommandShape) Error!void {
        if (self.token_count == 0 or self.token_count > context_max)
            return error.InvalidTokenCount;
        try validateModelDim(self.model_dim);
        if (self.ffn_dim == 0 or self.ffn_dim > ffn_dim_max or self.ffn_dim % layout.q1_block != 0)
            return error.InvalidFfnDim;
    }
};

pub const AttentionShape = struct {
    query_count: u32,
    model_dim: u32,
    head_dim_q: u32,
    head_dim_v: u32,
    n_heads: u32,
    n_head_kv: u32,
    kv_count: u32,

    pub fn validate(self: AttentionShape) Error!void {
        try validateTokenCount(self.query_count);
        try validateModelDim(self.model_dim);
        if (self.head_dim_q == 0 or self.head_dim_q > head_dim_max or self.head_dim_q % 8 != 0 or
            self.head_dim_v == 0 or self.head_dim_v > head_dim_max or self.head_dim_v % 8 != 0)
        {
            return error.InvalidHeadDim;
        }
        if (self.n_heads == 0 or self.n_heads > heads_max or
            self.model_dim != self.n_heads * self.head_dim_q or
            self.model_dim != self.n_heads * self.head_dim_v)
            return error.InvalidHeadCount;
        if (self.n_head_kv == 0 or self.n_head_kv > kv_heads_max or self.n_heads % self.n_head_kv != 0)
            return error.InvalidHeadCount;
        if (self.kv_count == 0 or self.kv_count > context_max) return error.InvalidKvCount;
    }

    pub fn newKvBytes(self: AttentionShape) Error!u64 {
        try self.validate();
        return @as(u64, self.query_count) * self.n_head_kv * (self.head_dim_q + self.head_dim_v) * @sizeOf(f16);
    }

    pub fn stateBytes(self: AttentionShape) Error!u64 {
        try self.validate();
        // Per (query, head): FP32 m and l. The output accumulator lives in X2.
        return @as(u64, self.query_count) * self.n_heads * 2 * @sizeOf(f32);
    }

    pub fn outputAccumulatorBytes(self: AttentionShape) Error!u64 {
        try self.validate();
        return @as(u64, self.query_count) * self.n_heads * self.head_dim_v * @sizeOf(f32);
    }
};

pub fn totalF32Bytes() u64 {
    return F32Role.residual.bytes() + F32Role.x0.bytes() +
        F32Role.x1.bytes() + F32Role.x2.bytes();
}

fn validateTokenCount(count: u32) Error!void {
    if (count == 0 or count > query_tile_max) return error.InvalidTokenCount;
}

fn validateModelDim(dim: u32) Error!void {
    if (dim == 0 or dim > model_dim_max or dim % layout.q1_block != 0)
        return error.InvalidModelDim;
}

test "f32 mapping accepts GEMM row-pair order without a transpose copy" {
    for (0..16) |row| {
        if (row % 2 != 0) continue;
        const loc = try f32Location(.x0, 2, @intCast(row));
        try std.testing.expectEqual(@as(u2, @intCast((row / 2) % 4)), loc.bank);
        try std.testing.expectEqual(2 * (ffn_dim_max / 8) + @as(u32, @intCast(row / 8)), loc.address);
    }
    try std.testing.expectError(error.InvalidRowPair, f32Location(.x0, 0, 1));
    try std.testing.expectError(error.InvalidTokenCount, f32Location(.x0, query_tile_max, 0));
}

test "physical f32 mapping exhaustively and uniquely fills every bank" {
    try std.testing.expectEqual(@as(u32, 0), f32RoleBase(.residual));
    try std.testing.expectEqual(@as(u32, 2048), f32RoleBase(.x0));
    try std.testing.expectEqual(@as(u32, 8192), f32RoleBase(.x1));
    try std.testing.expectEqual(@as(u32, 14336), f32RoleBase(.x2));
    try std.testing.expectEqual(@as(u32, 16384), f32BankDepth());

    const total_words: usize = @as(usize, f32_banks_per_role) * @as(usize, f32BankDepth());
    const seen = try std.testing.allocator.alloc(bool, total_words);
    defer std.testing.allocator.free(seen);
    @memset(seen, false);

    const roles = [_]F32Role{ .residual, .x0, .x1, .x2 };
    var visited: usize = 0;
    for (roles) |role| {
        try std.testing.expectEqual(
            f32RoleBase(role) + f32RoleSpan(role),
            if (role == .x2) f32BankDepth() else f32RoleBase(@enumFromInt(@intFromEnum(role) + 1)),
        );

        for (0..query_tile_max) |token| {
            for (0..role.rowCapacity() / f32_values_per_word) |pair| {
                const even_row: u32 = @intCast(pair * f32_values_per_word);
                const local = try f32Location(role, @intCast(token), even_row);
                const physical = try f32PhysicalLocation(role, @intCast(token), even_row);
                try std.testing.expectEqual(local.bank, physical.bank);
                try std.testing.expectEqual(f32RoleBase(role) + local.address, physical.address);
                try std.testing.expect(physical.address < f32BankDepth());

                const index = @as(usize, physical.bank) * @as(usize, f32BankDepth()) +
                    @as(usize, physical.address);
                try std.testing.expect(!seen[index]);
                seen[index] = true;
                visited += 1;
            }
        }

        try std.testing.expectError(error.InvalidRow, f32PhysicalLocation(role, 0, role.rowCapacity()));
        try std.testing.expectError(error.InvalidRowPair, f32PhysicalLocation(role, 0, 1));
        try std.testing.expectError(error.InvalidTokenCount, f32PhysicalLocation(role, query_tile_max, 0));
    }

    try std.testing.expectEqual(total_words, visited);
    for (seen) |word_was_mapped| try std.testing.expect(word_was_mapped);
}

test "native Q8 addresses preserve token reuse within GEMM storage" {
    try std.testing.expectEqual(@as(u32, 19), (try q8Location(2, 3)).address);
    try std.testing.expectError(error.InvalidTokenCount, q8Location(0, query_tile_max));
    try std.testing.expectError(error.InvalidSubblock, q8Location(layout.max_sub_index, 0));
}

test "P3 Q8 buffer mapping is block-major, exhaustive, and ping-ponged" {
    try std.testing.expectEqual(@as(u32, 384), q8_buffer_block_capacity);
    try std.testing.expectEqual(@as(u32, 1536), q8_buffer_records_per_bank);
    try std.testing.expectEqual(@as(u32, 3072), q8_buffer_record_capacity);
    try std.testing.expectEqual(@as(u32, 0), (try q8BufferLocation(0, 0, 0)).address);
    try std.testing.expectEqual(@as(u32, 7), (try q8BufferLocation(0, 3, 1)).address);
    try std.testing.expectEqual(@as(u32, 1536), (try q8BufferLocation(1, 0, 0)).address);
    try std.testing.expectEqual(@as(u32, 3071), (try q8BufferLocation(1, 3, 383)).address);

    var seen = [_]bool{false} ** q8_buffer_record_capacity;
    var visited: usize = 0;
    for (0..q8_buffer_bank_count) |bank| {
        for (0..q8_buffer_block_capacity) |block| {
            for (0..q8_buffer_token_capacity) |token| {
                const loc = try q8BufferLocation(
                    @intCast(bank),
                    @intCast(token),
                    @intCast(block),
                );
                try std.testing.expect(loc.address < q8_buffer_record_capacity);
                try std.testing.expect(!seen[loc.address]);
                seen[loc.address] = true;
                visited += 1;
            }
        }
    }
    try std.testing.expectEqual(@as(usize, q8_buffer_record_capacity), visited);
    for (seen) |mapped| try std.testing.expect(mapped);

    try std.testing.expectError(error.InvalidBank, q8BufferLocation(2, 0, 0));
    try std.testing.expectError(error.InvalidTokenCount, q8BufferLocation(0, 4, 0));
    try std.testing.expectError(error.InvalidSubblock, q8BufferLocation(0, 0, 384));
}

test "P3 FFN pair tags map exhaustively onto X1 scratch groups" {
    try std.testing.expectEqual(@as(u32, 4), ffn_pair_groups_per_block);

    for (0..q8_buffer_block_capacity) |block| {
        for (0..ffn_pair_groups_per_block) |group| {
            const got = try ffnPairScratchGroup(@intCast(block), @intCast(group));
            try std.testing.expectEqual(
                @as(u32, @intCast(block * ffn_pair_groups_per_block + group)),
                got,
            );
            try std.testing.expect(got < f32GroupsPerToken(.x1));
        }
    }

    try std.testing.expectError(
        error.InvalidSubblock,
        ffnPairScratchGroup(q8_buffer_block_capacity, 0),
    );
    try std.testing.expectError(
        error.InvalidRow,
        ffnPairScratchGroup(0, ffn_pair_groups_per_block),
    );
}

test "P3 FFN pair native ingress reorders into canonical token blocks" {
    for (1..query_tile_max + 1) |tokens| {
        const groups_per_block = tokens * ffn_pair_groups_per_block;
        for (0..q8_buffer_block_capacity) |block| {
            for (0..groups_per_block) |local| {
                const ordinal: u32 = @intCast(block * groups_per_block + local);
                const native = try ffnPairNativeTag(@intCast(tokens), ordinal);
                const canonical = try ffnPairCanonicalTag(@intCast(tokens), ordinal);
                const half_span = tokens * 2;
                const half = local / half_span;
                const within_half = local % half_span;

                try std.testing.expectEqual(@as(u32, @intCast(block)), native.block);
                try std.testing.expectEqual(@as(u32, @intCast(within_half / 2)), native.token);
                try std.testing.expectEqual(
                    @as(u32, @intCast(half * 2 + within_half % 2)),
                    native.group,
                );
                try std.testing.expectEqual(@as(u32, @intCast(block)), canonical.block);
                try std.testing.expectEqual(
                    @as(u32, @intCast(local / ffn_pair_groups_per_block)),
                    canonical.token,
                );
                try std.testing.expectEqual(
                    @as(u32, @intCast(local % ffn_pair_groups_per_block)),
                    canonical.group,
                );
            }
        }
    }

    try std.testing.expectError(error.InvalidTokenCount, ffnPairNativeTag(0, 0));
    try std.testing.expectError(error.InvalidTokenCount, ffnPairCanonicalTag(5, 0));
    try std.testing.expectError(
        error.InvalidSubblock,
        ffnPairNativeTag(4, q8_buffer_block_capacity * 4 * ffn_pair_groups_per_block),
    );
}

test "P3d v17 register lifecycle masks are exact and non-overlapping" {
    try std.testing.expectEqual(@as(u32, 15), ffn_kernel_version);
    try std.testing.expectEqual(@as(u32, 17), p3d_kernel_version);

    try std.testing.expectEqual(@as(u32, 0x1), ScratchControl.drain_start);
    try std.testing.expectEqual(@as(u32, 0x2), ScratchControl.abort);
    try std.testing.expectEqual(@as(u32, 0x4), ScratchControl.section_begin);
    try std.testing.expectEqual(@as(u32, 0x8), ScratchControl.resident_r);
    try std.testing.expectEqual(
        @as(u32, 0),
        ScratchControl.resident_r & ScratchControl.section_begin,
    );

    try std.testing.expectEqual(@as(u32, 0x20), ScratchStatus.r_valid);
    try std.testing.expectEqual(@as(u32, 0x800), ScratchStatus.section_active);
    try std.testing.expectEqual(@as(u32, 0x1000), ScratchStatus.section_done);
    try std.testing.expectEqual(@as(u32, 0x3fff), ScratchStatus.defined_mask);
    try std.testing.expectEqual(@as(u32, 0x1ff), ScratchError.fatal_mask);
    try std.testing.expectEqual(ScratchError.fatal_mask, ScratchError.defined_mask);

    try std.testing.expectEqual(@as(u32, 1), NormControl.load_gamma);
    try std.testing.expectEqual(@as(u32, 0x91), NormStatus.busy_mask);
    try std.testing.expectEqual(@as(u32, 0x122), NormStatus.done_mask);
    try std.testing.expectEqual(@as(u32, 0x244), NormStatus.error_mask);
    try std.testing.expectEqual(@as(u32, 0x400), NormStatus.idle);
    try std.testing.expectEqual(@as(u32, 0x7ff), NormStatus.defined_mask);

    try std.testing.expectEqual(@as(u32, 0x007f_ffff), NormError.scalar(0x7f_ffff));
    try std.testing.expectEqual(@as(u32, 0x0780_0000), NormError.gamma(0xf));
    try std.testing.expectEqual(@as(u32, 0x0800_0000), NormError.controller);
    try std.testing.expectEqual(@as(u32, 0x0fff_ffff), NormError.fatal_mask);
    try std.testing.expectEqual(NormError.fatal_mask, NormError.defined_mask);
    try std.testing.expectEqual(
        @as(u32, ResidualAddStatus.fatal_mask),
        ResidualError.raw_mask,
    );
    try std.testing.expectEqual(ResidualError.raw_mask, ResidualError.defined_mask);

    try std.testing.expect(!p3dModelRowsValid(127));
    try std.testing.expect(p3dModelRowsValid(128));
    try std.testing.expect(!p3dModelRowsValid(384));
    try std.testing.expect(p3dModelRowsValid(4096));
    try std.testing.expect(!p3dModelRowsValid(8192));
    try std.testing.expect(!p3dNormEpsValid(0x0000_0000));
    try std.testing.expect(!p3dNormEpsValid(0x0000_0001));
    try std.testing.expect(p3dNormEpsValid(0x0080_0000));
    try std.testing.expect(p3dNormEpsValid(0x7f7f_ffff));
    try std.testing.expect(!p3dNormEpsValid(0x7f80_0000));
    try std.testing.expect(!p3dNormEpsValid(0x7fc0_0000));
    try std.testing.expect(!p3dNormEpsValid(0x8080_0000));
}

test "P3d RMSNorm sumsq status and fixed integer oracle are explicit" {
    try std.testing.expectEqual(@as(u7, 0x3f), RmsNormSumsqStatus.fatal_mask);
    try std.testing.expectEqual(@as(u7, 0x40), RmsNormSumsqStatus.subnormal_warning);

    const ones = [_]u32{0x3f80_0000} ** 8;
    const one_result = try rmsNormSumsqFixed(&ones, 8, 127);
    try std.testing.expectEqual(@as(u48, 8) << 34, one_result.sum_sq);
    try std.testing.expect(!one_result.subnormal_warning);
    try std.testing.expectEqual(@as(f64, 1), rmsNormSumsqMeanF64(one_result.sum_sq, 8, 127));
    try std.testing.expectEqual(@as(f64, 1), rmsNormSumsqExactMeanF64(&ones));

    const signed_zero = [_]u32{
        0, 0x8000_0000, 0, 0x8000_0000,
        0, 0x8000_0000, 0, 0x8000_0000,
    };
    const zero_result = try rmsNormSumsqFixed(&signed_zero, 8, 0);
    try std.testing.expectEqual(@as(u48, 0), zero_result.sum_sq);
    try std.testing.expect(!zero_result.subnormal_warning);

    var subnormal = signed_zero;
    subnormal[3] = 1;
    const subnormal_result = try rmsNormSumsqFixed(&subnormal, 8, 0);
    try std.testing.expectEqual(@as(u48, 0), subnormal_result.sum_sq);
    try std.testing.expect(subnormal_result.subnormal_warning);

    var bad = ones;
    bad[4] = 0x7f80_0000;
    try std.testing.expectError(error.Nonfinite, rmsNormSumsqFixed(&bad, 8, 127));
    bad[4] = 0x4000_0000;
    try std.testing.expectError(error.MaxMismatch, rmsNormSumsqFixed(&bad, 8, 127));
    try std.testing.expectError(error.MaxMismatch, rmsNormSumsqFixed(&ones, 8, 128));
    try std.testing.expectError(error.InvalidMaxExponent, rmsNormSumsqFixed(&ones, 8, 0xff));
    try std.testing.expectError(error.InvalidShape, rmsNormSumsqFixed(ones[0..7], 7, 127));
}

test "P3d RMSNorm max exponent scan is exact and preserves warnings" {
    try std.testing.expectEqual(@as(u6, 0x1f), RmsNormMaxExpStatus.fatal_mask);
    try std.testing.expectEqual(@as(u6, 0x20), RmsNormMaxExpStatus.subnormal_warning);

    const values = [_]u32{
        0x0000_0000,
        0x8000_0000,
        0x0000_0001,
        0x3f80_0000,
        0xc120_0000,
        0x4080_0000,
        0x007f_ffff,
        0xbf00_0000,
    };
    const got = try rmsNormMaxExp(&values, values.len);
    try std.testing.expectEqual(@as(u8, 130), got.max_exp);
    try std.testing.expect(got.subnormal_warning);

    const zeros = [_]u32{0} ** 8;
    const all_zero = try rmsNormMaxExp(&zeros, zeros.len);
    try std.testing.expectEqual(@as(u8, 0), all_zero.max_exp);
    try std.testing.expect(!all_zero.subnormal_warning);
}

test "P3d RMSNorm max exponent scan rejects shape and nonfinite input" {
    const short = [_]u32{0} ** 7;
    try std.testing.expectError(error.InvalidShape, rmsNormMaxExp(&short, short.len));

    var values = [_]u32{0} ** 8;
    values[5] = 0x7f80_0000;
    try std.testing.expectError(error.Nonfinite, rmsNormMaxExp(&values, values.len));
    values[5] = 0x7fc0_0001;
    try std.testing.expectError(error.Nonfinite, rmsNormMaxExp(&values, values.len));
}

test "P3d residual loader maps token-major words into R scratch" {
    try std.testing.expectEqual(@as(u4, 0xf), RmsNormLoaderStatus.fatal_mask);
    for (1..query_tile_max + 1) |tokens| {
        const rows: u32 = model_dim_max;
        const words = rows / f32_values_per_word;
        for (0..tokens * words) |ordinal| {
            const got = try rmsNormResidualWordLocation(
                rows,
                @intCast(tokens),
                @intCast(ordinal),
            );
            const token: u32 = @intCast(ordinal / words);
            const word: u32 = @intCast(ordinal % words);
            const expected = try f32PhysicalLocation(
                .residual,
                token,
                word * f32_values_per_word,
            );
            try std.testing.expectEqual(expected, got);
        }
    }

    try std.testing.expectError(error.InvalidModelDim, rmsNormResidualWordLocation(7, 1, 0));
    try std.testing.expectError(error.InvalidTokenCount, rmsNormResidualWordLocation(8, 0, 0));
    try std.testing.expectError(error.InvalidRowPair, rmsNormResidualWordLocation(8, 1, 4));
}

test "P3d RMSNorm sumsq oracle covers every exponent-alignment class" {
    const max_exp: u8 = 140;
    for (0..20) |delta_usize| {
        const delta: u8 = @intCast(delta_usize);
        const exponent = max_exp - delta;
        const bits = (@as(u32, exponent) << 23) | 0x007f_ffff;
        var values = [_]u32{0} ** 8;
        values[0] = (@as(u32, max_exp) << 23) | 0x007f_ffff;
        values[1] = bits;
        const got = try rmsNormSumsqFixed(&values, 8, max_exp);
        const significand: u24 = 0x00ff_ffff;
        const max_quant: u18 = @truncate(significand >> 6);
        const quant: u18 = if (delta >= 18)
            0
        else
            @truncate(significand >> @intCast(6 + delta));
        const max_product: u36 = @as(u36, max_quant) * @as(u36, max_quant);
        const product: u36 = @as(u36, quant) * @as(u36, quant);
        try std.testing.expectEqual(
            @as(u48, max_product) + @as(u48, product),
            got.sum_sq,
        );
    }

    try std.testing.expect(rmsNormSumsqRelativeErrorBound(4096) <
        rmsnorm_sumsq_adversarial_relative_limit);
    try std.testing.expect(rmsNormSumsqRelativeErrorBound(2048) <
        rmsnorm_sumsq_adversarial_relative_limit);
}

test "P3d RMSNorm inverse scalar fixes mean rounding and request bounds" {
    const one_sum: u48 = @as(u48, 8) << 34;
    try std.testing.expectEqual(@as(u32, 0x3f80_0000), try rmsNormFixedMeanBits(one_sum, 8, 127));
    try std.testing.expectEqual(@as(u32, 0), try rmsNormFixedMeanBits(0, 2048, 0));

    const cases = [_]struct { sum: u48, rows: u32, exponent: u8 }{
        .{ .sum = 0x0000_1234_5678, .rows = 8, .exponent = 110 },
        .{ .sum = 0x0001_ffff_ffff, .rows = 128, .exponent = 127 },
        .{ .sum = 0x1234_5678_9abc, .rows = 2048, .exponent = 132 },
        .{ .sum = 0xffff_ff00_0000, .rows = 4096, .exponent = 144 },
    };
    for (cases) |case| {
        const value = std.math.ldexp(
            @as(f64, @floatFromInt(case.sum)) /
                @as(f64, @floatFromInt(case.rows)),
            2 * (@as(i32, case.exponent) - 144),
        );
        const expected: u32 = @bitCast(@as(f32, @floatCast(value)));
        try std.testing.expectEqual(expected, try rmsNormFixedMeanBits(case.sum, case.rows, case.exponent));
    }

    try std.testing.expectError(error.InvalidShape, rmsNormFixedMeanBits(one_sum, 24, 127));
    try std.testing.expectError(error.InvalidRecord, rmsNormFixedMeanBits(one_sum, 8, 0));
    try std.testing.expectError(error.InvalidRecord, rmsNormFixedMeanBits(0, 8, 127));
    try std.testing.expectError(error.InvalidEpsilon, rmsNormInvFixed(one_sum, 8, 127, 0));
    try std.testing.expectError(error.InvalidEpsilon, rmsNormInvFixed(one_sum, 8, 127, 0xbf80_0000));
}

test "P3d RMSNorm reduction composition preserves child diagnostics" {
    try std.testing.expectEqual(@as(u13, 0x1fff), RmsNormReduceStatus.fatal_mask);
    try std.testing.expectEqual(
        @as(u13, RmsNormFrontendStatus.scratch) << 1,
        RmsNormReduceStatus.frontend(RmsNormFrontendStatus.scratch),
    );
    try std.testing.expectEqual(
        @as(u13, RmsNormFrontendStatus.subnormal) << 1,
        RmsNormReduceStatus.frontend(RmsNormFrontendStatus.subnormal),
    );
    try std.testing.expectEqual(
        @as(u13, RmsNormInvStatus.arithmetic) << 8,
        RmsNormReduceStatus.inverse(RmsNormInvStatus.arithmetic),
    );
    try std.testing.expectEqual(@as(u13, 0x1000), RmsNormReduceStatus.internal);
    try std.testing.expectEqual(@as(u13, 1), RmsNormReduceStatus.bad_cfg);
}

test "P3d exact finite multiplier covers IEEE rounding boundaries" {
    try std.testing.expectEqual(@as(u2, 0x3), RmsNormMulStatus.fatal_mask);
    const cases = [_]struct {
        a: u32,
        b: u32,
        bits: u32,
        status: u2 = 0,
    }{
        .{ .a = 0x3f80_0000, .b = 0x3f80_0000, .bits = 0x3f80_0000 },
        .{ .a = 0xc000_0000, .b = 0x3f00_0000, .bits = 0xbf80_0000 },
        .{ .a = 0x0000_0000, .b = 0xbf80_0000, .bits = 0x8000_0000 },
        .{ .a = 0x8000_0000, .b = 0xbf80_0000, .bits = 0x0000_0000 },
        .{ .a = 0x3f8b_593f, .b = 0x3fb6_227b, .bits = 0x3fc6_486f },
        .{ .a = 0x3fdf_9fbe, .b = 0x3fe6_5296, .bits = 0x4049_31a9 },
        .{ .a = 0x3fe0_0000, .b = 0x3fd7_539c, .bits = 0x403c_6928 },
        .{ .a = 0x3ffd_f200, .b = 0x3f93_c000, .bits = 0x4012_906c },
        .{ .a = 0x3fc0_0000, .b = 0x3f80_0001, .bits = 0x3fc0_0002 },
        .{ .a = 0x3fc0_0000, .b = 0x3f80_0003, .bits = 0x3fc0_0004 },
        .{ .a = 0x3fff_f830, .b = 0x3f80_03e8, .bits = 0x4000_0000 },
        .{ .a = 0x7f7f_ffff, .b = 0x3f80_0000, .bits = 0x7f7f_ffff },
        .{
            .a = 0x7f7f_ffff,
            .b = 0x3f80_0001,
            .bits = 0x7f80_0000,
            .status = RmsNormMulStatus.overflow,
        },
        .{
            .a = 0xff7f_ffff,
            .b = 0x3f80_0001,
            .bits = 0xff80_0000,
            .status = RmsNormMulStatus.overflow,
        },
        .{ .a = 0x0080_0000, .b = 0x3f00_0000, .bits = 0x0040_0000 },
        .{ .a = 0x0000_0001, .b = 0x3f80_0000, .bits = 0x0000_0001 },
        .{ .a = 0x0000_0001, .b = 0x3f00_0000, .bits = 0x0000_0000 },
        .{ .a = 0x8000_0001, .b = 0x3f00_0000, .bits = 0x8000_0000 },
        .{ .a = 0x0000_0003, .b = 0x3f00_0000, .bits = 0x0000_0002 },
        .{ .a = 0x0000_0005, .b = 0x3f00_0000, .bits = 0x0000_0002 },
        .{ .a = 0x0000_0001, .b = 0x3f00_0001, .bits = 0x0000_0001 },
        .{ .a = 0x007f_ffff, .b = 0x3f80_0001, .bits = 0x0080_0000 },
        .{ .a = 0x007f_ffff, .b = 0x4000_0000, .bits = 0x00ff_fffe },
        .{ .a = 0x0000_0001, .b = 0x7f7f_ffff, .bits = 0x34ff_ffff },
        .{ .a = 0x007f_ffff, .b = 0x007f_ffff, .bits = 0x0000_0000 },
        .{ .a = 0x19ff_ffff, .b = 0x1a7f_ffff, .bits = 0x0000_0001 },
        .{ .a = 0x197f_ffff, .b = 0x1a7f_ffff, .bits = 0x0000_0000 },
        .{ .a = 0x997f_ffff, .b = 0x1a7f_ffff, .bits = 0x8000_0000 },
        .{
            .a = 0x5f61_2000,
            .b = 0x5f91_8e00,
            .bits = 0x7f80_0000,
            .status = RmsNormMulStatus.overflow,
        },
        .{
            .a = 0x7f80_0000,
            .b = 0x3f80_0000,
            .bits = 0,
            .status = RmsNormMulStatus.nonfinite_input,
        },
        .{
            .a = 0,
            .b = 0x7f80_0000,
            .bits = 0,
            .status = RmsNormMulStatus.nonfinite_input,
        },
        .{
            .a = 0x7fc1_2345,
            .b = 0xff80_0000,
            .bits = 0,
            .status = RmsNormMulStatus.nonfinite_input,
        },
    };

    for (cases) |case| {
        const got = rmsNormMulRneBits(case.a, case.b);
        try std.testing.expectEqual(case.bits, got.bits);
        try std.testing.expectEqual(case.status, got.status);
        const swapped = rmsNormMulRneBits(case.b, case.a);
        try std.testing.expectEqual(got.bits, swapped.bits);
        try std.testing.expectEqual(got.status, swapped.status);
    }
}

test "P3d exact residual addition covers RNE and status boundaries" {
    try std.testing.expectEqual(
        @as(u2, 0x3),
        ResidualAddArithmeticStatus.fatal_mask,
    );
    try std.testing.expectEqual(@as(u7, 0x7f), ResidualAddStatus.fatal_mask);
    try std.testing.expectEqual(
        @as(u7, 0x20),
        ResidualAddStatus.arithmetic(ResidualAddArithmeticStatus.overflow),
    );
    const cases = [_]struct {
        a: u32,
        b: u32,
        bits: u32,
        status: u2 = 0,
    }{
        .{ .a = 0x3f80_0000, .b = 0x3f80_0000, .bits = 0x4000_0000 },
        .{ .a = 0x3f80_0000, .b = 0x3380_0000, .bits = 0x3f80_0000 },
        .{ .a = 0x3f80_0001, .b = 0x3380_0000, .bits = 0x3f80_0002 },
        .{ .a = 0x3f80_0000, .b = 0xbf80_0000, .bits = 0 },
        .{ .a = 0x8000_0000, .b = 0x8000_0000, .bits = 0x8000_0000 },
        .{ .a = 0x0080_0000, .b = 0x807f_ffff, .bits = 1 },
        .{ .a = 0x007f_ffff, .b = 1, .bits = 0x0080_0000 },
        .{
            .a = 0x7f7f_ffff,
            .b = 0x7f7f_ffff,
            .bits = 0x7f80_0000,
            .status = ResidualAddArithmeticStatus.overflow,
        },
        .{
            .a = 0x7f80_0000,
            .b = 0x3f80_0000,
            .bits = 0,
            .status = ResidualAddArithmeticStatus.nonfinite_input,
        },
    };
    for (cases) |case| {
        const got = residualAddRneBits(case.a, case.b);
        try std.testing.expectEqual(case.bits, got.bits);
        try std.testing.expectEqual(case.status, got.status);
        try std.testing.expectEqual(got, residualAddRneBits(case.b, case.a));
    }
}

test "P3d weighted RMSNorm multiplication order is explicit" {
    const x: u32 = 0x3730_b2f5;
    const inverse: u32 = 0x451c_6e2e;
    const gamma: u32 = 0x3ffa_b1b6;
    const weighted = rmsNormWeightedBits(x, inverse, gamma);
    const reassociated_scale = rmsNormMulRneBits(inverse, gamma);
    const reassociated = rmsNormMulRneBits(x, reassociated_scale.bits);
    try std.testing.expectEqual(@as(u2, 0), weighted.mul1_status);
    try std.testing.expectEqual(@as(u2, 0), weighted.mul2_status);
    try std.testing.expectEqual(@as(u32, 0x3d53_786f), weighted.bits);
    try std.testing.expectEqual(@as(u32, 0x3d53_786e), reassociated.bits);

    // A normal first-stage input may produce a subnormal which the gamma stage
    // must consume rather than flush.
    const tiny_normalized = rmsNormMulRneBits(0x0080_0000, 0x3f00_0000);
    try std.testing.expectEqual(@as(u32, 0x0040_0000), tiny_normalized.bits);
    try std.testing.expectEqual(
        @as(u32, 0x3fff_ffff),
        rmsNormMulRneBits(tiny_normalized.bits, 0x7f7f_ffff).bits,
    );
}

test "P3d weighted source status and gamma bank mapping are exact" {
    try std.testing.expectEqual(@as(u2, 0), rmsNormGammaBank(0));
    try std.testing.expectEqual(@as(u2, 3), rmsNormGammaBank(3));
    try std.testing.expectEqual(@as(u2, 0), rmsNormGammaBank(4));
    try std.testing.expectEqual(@as(u9, 0), rmsNormGammaAddress(3));
    try std.testing.expectEqual(@as(u9, 1), rmsNormGammaAddress(4));
    try std.testing.expectEqual(@as(u9, 511), rmsNormGammaAddress(2047));

    try std.testing.expectEqual(
        @as(u9, 0x010),
        RmsNormWeightedSourceStatus.mul1(
            RmsNormMulStatus.nonfinite_input,
        ),
    );
    try std.testing.expectEqual(
        @as(u9, 0x080),
        RmsNormWeightedSourceStatus.mul2(RmsNormMulStatus.overflow),
    );

    const first_fault = rmsNormWeightedBits(
        0x7f80_0000,
        0x3f80_0000,
        0x3f80_0000,
    );
    try std.testing.expectEqual(
        RmsNormMulStatus.nonfinite_input,
        first_fault.mul1_status,
    );
    try std.testing.expectEqual(@as(u2, 0), first_fault.mul2_status);
    try std.testing.expectEqual(@as(u32, 0), first_fault.bits);

    const second_fault = rmsNormWeightedBits(
        0x7f7f_ffff,
        0x3f80_0000,
        0x4000_0000,
    );
    try std.testing.expectEqual(@as(u2, 0), second_fault.mul1_status);
    try std.testing.expectEqual(
        RmsNormMulStatus.overflow,
        second_fault.mul2_status,
    );
}

test "P3d RMSNorm Q8 source preserves both raw child status layouts" {
    try std.testing.expectEqual(
        @as(u16, 0x0001),
        RmsNormQ8SourceStatus.weighted(
            RmsNormWeightedSourceStatus.bad_cfg,
        ),
    );
    try std.testing.expectEqual(
        @as(u16, 0x0100),
        RmsNormQ8SourceStatus.weighted(
            RmsNormWeightedSourceStatus.internal,
        ),
    );
    try std.testing.expectEqual(
        @as(u16, 0x0200),
        RmsNormQ8SourceStatus.q8(0x01),
    );
    try std.testing.expectEqual(
        @as(u16, 0x4000),
        RmsNormQ8SourceStatus.q8(0x20),
    );
    try std.testing.expectEqual(
        @as(u16, 0xffff),
        RmsNormQ8SourceStatus.weighted(0x1ff) |
            RmsNormQ8SourceStatus.q8(0x3f) |
            RmsNormQ8SourceStatus.internal,
    );
    try std.testing.expectEqual(
        @as(u16, 0xffff),
        RmsNormQ8SourceStatus.fatal_mask,
    );
}

test "P3d RMSNorm scalar pipeline preserves both raw child layouts" {
    try std.testing.expectEqual(
        @as(u23, 0x001fff),
        RmsNormScalarPipelineStatus.reduce(0x1fff),
    );
    try std.testing.expectEqual(
        @as(u23, 0x3f_e000),
        RmsNormScalarPipelineStatus.weighted(0x1ff),
    );
    try std.testing.expectEqual(
        @as(u23, 0x7f_ffff),
        RmsNormScalarPipelineStatus.reduce(0x1fff) |
            RmsNormScalarPipelineStatus.weighted(0x1ff) |
            RmsNormScalarPipelineStatus.internal,
    );
    try std.testing.expectEqual(
        @as(u23, 0x7f_ffff),
        RmsNormScalarPipelineStatus.fatal_mask,
    );
}

test "P3d RMSNorm Q8 pipeline preserves both composed child layouts" {
    try std.testing.expectEqual(
        @as(u30, 0x0000_1fff),
        RmsNormQ8PipelineStatus.reduce(0x1fff),
    );
    try std.testing.expectEqual(
        @as(u30, 0x1fff_e000),
        RmsNormQ8PipelineStatus.source(0xffff),
    );
    try std.testing.expectEqual(
        @as(u30, 0x1000_0000),
        RmsNormQ8PipelineStatus.source(RmsNormQ8SourceStatus.internal),
    );
    try std.testing.expectEqual(
        @as(u30, 0x2000_0000),
        RmsNormQ8PipelineStatus.internal,
    );
    try std.testing.expectEqual(
        @as(u30, 0x3fff_ffff),
        RmsNormQ8PipelineStatus.reduce(0x1fff) |
            RmsNormQ8PipelineStatus.source(0xffff) |
            RmsNormQ8PipelineStatus.internal,
    );
}

test "P3d RMSNorm two-step inverse square root has a bounded scalar error" {
    var maximum_relative_error: f64 = 0;
    var exponent: u32 = 2;
    while (exponent <= 251) : (exponent += 1) {
        var sample: u32 = 0;
        while (sample <= 32) : (sample += 1) {
            const mantissa: u32 = @intCast(
                (@as(u64, sample) * 0x7f_ffff) / 32,
            );
            const bits = (exponent << 23) | mantissa;
            const value = @as(f64, @floatCast(@as(f32, @bitCast(bits))));
            const approx_bits = try rmsNormInvSqrtApproxBits(bits);
            const approx = @as(f64, @floatCast(@as(f32, @bitCast(approx_bits))));
            const expected = 1.0 / @sqrt(value);
            maximum_relative_error = @max(
                maximum_relative_error,
                @abs(approx - expected) / expected,
            );
        }
    }
    if (maximum_relative_error > rmsnorm_inv_scalar_relative_limit)
        std.debug.print("maximum inverse-sqrt relative error: {e}\n", .{
            maximum_relative_error,
        });
    try std.testing.expect(maximum_relative_error <= rmsnorm_inv_scalar_relative_limit);

    const one_sum: u48 = @as(u48, 2048) << 34;
    const result = try rmsNormInvFixed(one_sum, 2048, 127, 0x3586_37bd);
    try std.testing.expectEqual(@as(u32, 0x3f80_0000), result.mean_bits);
    const adjusted: f64 = @floatCast(@as(f32, @bitCast(result.adjusted_mean_bits)));
    const inverse: f64 = @floatCast(@as(f32, @bitCast(result.inv_rms_bits)));
    try std.testing.expect(@abs(inverse - 1.0 / @sqrt(adjusted)) <=
        rmsnorm_inv_scalar_relative_limit);
    try std.testing.expectError(error.InvalidRecord, rmsNormInvSqrtApproxBits(0x0080_0000));
    try std.testing.expectError(error.InvalidRecord, rmsNormInvSqrtApproxBits(0x7e00_0000));
}

test "v1 Bonsai section shapes and capacities are explicit" {
    try (FfnShape{ .token_count = 4, .model_dim = 4096, .ffn_dim = 12288 }).validate();
    const attention = AttentionShape{
        .query_count = 4,
        .model_dim = 4096,
        .head_dim_q = 128,
        .head_dim_v = 128,
        .n_heads = 32,
        .n_head_kv = 8,
        .kv_count = 8192,
    };
    try attention.validate();
    try std.testing.expectEqual(@as(u64, 16 * 1024), try attention.newKvBytes());
    try std.testing.expectEqual(@as(u64, 1024), try attention.stateBytes());
    try std.testing.expectEqual(F32Role.x2.bytes(), try attention.outputAccumulatorBytes());
    try std.testing.expectEqual(@as(u64, 512 * 1024), totalF32Bytes());
    try std.testing.expectError(error.InvalidTokenCount, (FfnShape{ .token_count = 5, .model_dim = 4096, .ffn_dim = 12288 }).validate());

    try (FfnCommandShape{ .token_count = 1, .model_dim = 4096, .ffn_dim = 12288 }).validate();
    try (FfnCommandShape{ .token_count = 16, .model_dim = 4096, .ffn_dim = 12288 }).validate();
    try (FfnCommandShape{ .token_count = context_max, .model_dim = 4096, .ffn_dim = 12288 }).validate();
    try std.testing.expectError(error.InvalidTokenCount, (FfnCommandShape{ .token_count = context_max + 1, .model_dim = 4096, .ffn_dim = 12288 }).validate());
}
