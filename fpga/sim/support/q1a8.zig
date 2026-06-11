//! Q1A8 layout constants — the single source of truth shared by the reference
//! oracle (matmul_ref.zig), the wire packer (pack.zig), and (via the regmap)
//! the FPGA decoder. Ported verbatim from the old PYNQ-Z1 `proto/q1a8_layout.py`.
//!
//! Shapes:
//!   - A rowblock is ROWS matmul rows computed in parallel by the fabric.
//!   - K is split into Q1 blocks of Q1_BLOCK weights; each Q1 block carries one
//!     fp16 weight scale per row.
//!   - Each Q1 block is 4 Q8 sub-blocks of Q8_BLOCK activations; each sub-block
//!     carries one fp16 activation scale.

pub const ROWS: usize = 8; // lanes per rowblock
pub const Q1_BLOCK: usize = 128; // weights per Q1 block (per row)
pub const Q8_BLOCK: usize = 32; // activations per Q8 sub-block
pub const Q8_SUBBLOCKS: usize = Q1_BLOCK / Q8_BLOCK; // = 4
pub const BEAT_BYTES: usize = 8; // 64-bit AXIS data width

// AXIS weight stream beats, per Q1 block per rowblock.
pub const SCALE_BEATS: usize = (ROWS + 3) / 4; // 4 fp16 scales per beat -> 2
pub const WBITS_BEATS: usize = (ROWS + 1) / 2; // 2 rows (u32) per beat   -> 4

/// Bytes of packed weights per Q1 block per rowblock (== ROWS * 18, the Q1_0 size).
pub const WEIGHT_BYTES_PER_BLOCK: usize =
    (SCALE_BEATS + Q8_SUBBLOCKS * WBITS_BEATS) * BEAT_BYTES; // = 144

/// Bytes of packed activations per Q1 block (one column; broadcast to all rowblocks).
/// Per sub-block: 4 beats of int8 acts + 1 beat holding the fp16 scale.
pub const ACT_BYTES_PER_BLOCK: usize =
    Q8_SUBBLOCKS * ((Q8_BLOCK / BEAT_BYTES) + 1) * BEAT_BYTES; // = 160

/// Bytes of results per rowblock (ROWS fp32, 2 fp32 per beat, lane-major).
pub const RESULT_BYTES_PER_ROWBLOCK: usize = (ROWS / 2) * BEAT_BYTES; // = 32

comptime {
    if (WEIGHT_BYTES_PER_BLOCK != 144) @compileError("weight block size drifted");
    if (ACT_BYTES_PER_BLOCK != 160) @compileError("act block size drifted");
    if (RESULT_BYTES_PER_ROWBLOCK != 32) @compileError("result block size drifted");
}
