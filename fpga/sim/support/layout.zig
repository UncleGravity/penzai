//! Cosim layout: the AXIS-beat geometry the Verilator testbench, the golden
//! reference (matmul_ref.zig), and the wire packer (pack.zig) consume. The block
//! constants are derived from the one production layout source (shared/layout.zig)
//! so the cosim and the device can never drift; only the cosim-specific AXIS feed
//! geometry lives here. ROWS is the production array width.

const shared = @import("shared_layout");

pub const ROWS: usize = shared.rows_per_block; // lanes per rowblock (16)
pub const Q1_BLOCK: usize = shared.q1_block; // weights per Q1 block (per row)
pub const Q8_BLOCK: usize = shared.q8_block; // activations per Q8 sub-block
pub const Q8_SUBBLOCKS: usize = shared.q8_subblocks; // = 4
pub const BEAT_BYTES: usize = shared.beat_bytes; // 64-bit AXIS data width
pub const q2_source_block: usize = shared.q2_source_block;
pub const ternary_block_bytes: usize = shared.ternary_block_bytes;
pub const ternary_packed_per_port_block: usize = shared.ternary_packed_per_port_block;
pub const ternary_beats_per_port_block: usize = shared.ternary_beats_per_port_block;

// AXIS weight stream beats, per Q1 block per rowblock.
pub const SCALE_BEATS: usize = (ROWS + 3) / 4; // 4 fp16 scales per beat
pub const WBITS_BEATS: usize = (ROWS + 1) / 2; // 2 rows (u32) per beat

/// Bytes of packed weights per Q1 block per rowblock (== ROWS * 18, the Q1_0 size).
pub const WEIGHT_BYTES_PER_BLOCK: usize =
    (SCALE_BEATS + Q8_SUBBLOCKS * WBITS_BEATS) * BEAT_BYTES;

/// Bytes of packed activations per Q1 block (one column; broadcast to all rowblocks).
/// Per sub-block: 4 beats of int8 acts + 1 beat holding the fp16 scale.
pub const ACT_BYTES_PER_BLOCK: usize =
    Q8_SUBBLOCKS * ((Q8_BLOCK / BEAT_BYTES) + 1) * BEAT_BYTES; // = 160

/// Bytes of results per rowblock (ROWS fp32, 2 fp32 per beat, lane-major).
pub const RESULT_BYTES_PER_ROWBLOCK: usize = (ROWS / 2) * BEAT_BYTES;

comptime {
    if (WEIGHT_BYTES_PER_BLOCK != ROWS * 18) @compileError("weight block size drifted");
    if (ACT_BYTES_PER_BLOCK != 160) @compileError("act block size drifted");
    if (RESULT_BYTES_PER_ROWBLOCK != ROWS * 4) @compileError("result block size drifted");
}
