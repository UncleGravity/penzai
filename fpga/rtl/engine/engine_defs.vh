`ifndef ENGINE_DEFS_VH
`define ENGINE_DEFS_VH

// Fixed architectural bounds for the dense Bonsai/Qwen token engine.
`define ENGINE_MAX_TOKENS          8
`define ENGINE_MAX_LAYERS          40
`define MODEL_LAYER_WORDS         8
`define ENGINE_MAX_HIDDEN_BLOCKS   128
`define ENGINE_MAX_FFN_BLOCKS      512
`define ENGINE_MAX_CONTEXT         65536
`define ENGINE_MAX_VOCAB_ROWS      160000

`define ENGINE_INTERFACE_VERSION   32'h0001_0007
// First 64 bits of SHA-256 over the canonical v5 model-spec/memory/attention contract.
`define MODEL_LAYOUT_HASH   64'hc255_c7a5_2fc1_4a79

// Canonical model identities. A model_spec ID is inseparable from the complete
// geometry tuple checked by model_spec_store.
`define MODEL_BONSAI_1_7B       32'd1
`define MODEL_BONSAI_4B         32'd2
`define MODEL_BONSAI_8B         32'd3
`define MODEL_BONSAI_VOCAB_ROWS 18'd151669

// One sealed layer record. Every address is a 64-byte-aligned device address.
`define MODEL_LAYER_FUSED_QKV        3'd0
`define MODEL_LAYER_O                3'd1
`define MODEL_LAYER_FUSED_GATE_UP    3'd2
`define MODEL_LAYER_DOWN             3'd3
`define MODEL_LAYER_ATTN_NORM        3'd4
`define MODEL_LAYER_Q_NORM           3'd5
`define MODEL_LAYER_K_NORM           3'd6
`define MODEL_LAYER_FFN_NORM         3'd7

// The only production execution order. Values are also the trace ABI.
`define ENGINE_STAGE_EMBED         5'd0
`define ENGINE_STAGE_ATTN_NORM     5'd1
`define ENGINE_STAGE_QKV_ROPE      5'd2
`define ENGINE_STAGE_KV_APPEND     5'd4
`define ENGINE_STAGE_ATTENTION     5'd5
`define ENGINE_STAGE_O_PROJ_RESID  5'd6
`define ENGINE_STAGE_FFN_NORM      5'd8
`define ENGINE_STAGE_GATE_UP_SWIGLU_Q8 5'd9
`define ENGINE_STAGE_DOWN_RESID     5'd11
`define ENGINE_STAGE_FINAL_NORM    5'd13
`define ENGINE_STAGE_LM_HEAD       5'd14

`define VECTOR_OP_EMBED           4'd0
`define VECTOR_OP_ATTN_NORM       4'd1
`define VECTOR_OP_QK_ROPE         4'd2
`define VECTOR_OP_KV_APPEND       4'd3
`define VECTOR_OP_ATTN_RESID      4'd4
`define VECTOR_OP_FFN_NORM        4'd5
`define VECTOR_OP_SWIGLU_Q8       4'd6
`define VECTOR_OP_FFN_RESID       4'd7
`define VECTOR_OP_FINAL_NORM      4'd8

`define PROJECTION_OP_QKV            3'd0
`define PROJECTION_OP_O              3'd1
`define PROJECTION_OP_GATE_UP        3'd2
`define PROJECTION_OP_DOWN           3'd3
`define PROJECTION_OP_LM_HEAD        3'd4

`define MODEL_SPEC_ERROR_BAD_HEADER      8'h01
`define MODEL_SPEC_ERROR_BAD_LAYER_WORD  8'h02
`define MODEL_SPEC_ERROR_INCOMPLETE      8'h03

`define ENGINE_ERROR_MODEL_SPEC         16'h0101
`define ENGINE_ERROR_TOKEN_COUNT     16'h0102
`define ENGINE_ERROR_LANE_MASK       16'h0103
`define ENGINE_ERROR_CONTEXT         16'h0104
`define ENGINE_ERROR_KV_BASE         16'h0105
`define ENGINE_ERROR_LEAF            16'h0201

`endif
