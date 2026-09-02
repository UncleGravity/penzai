`ifndef VECTOR_DEFS_VH
`define VECTOR_DEFS_VH

// Values match the engine's vector request ABI.
`define VECTOR_OP_ATTN_NORM   4'd1
`define VECTOR_OP_FFN_NORM    4'd5
`define VECTOR_OP_FINAL_NORM  4'd8

`define VECTOR_STATUS_BAD_CMD       16'h0001
`define VECTOR_STATUS_GAMMA_STREAM  16'h0002
`define VECTOR_STATUS_SOURCE        16'h0008
`define VECTOR_STATUS_REDUCE        16'h0010
`define VECTOR_STATUS_Q8            16'h0020
`define VECTOR_STATUS_INTERNAL      16'h0040

`endif
