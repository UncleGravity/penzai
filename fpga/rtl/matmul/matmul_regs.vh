// Generated from fpga/regmap/matmul.regmap — do not edit.
`ifndef MATMUL_REGS_VH
`define MATMUL_REGS_VH
localparam [11:0] MATMUL_OFF_ID = 12'h000;
localparam [11:0] MATMUL_OFF_VERSION = 12'h004;
localparam [11:0] MATMUL_OFF_CTRL = 12'h008;
localparam [11:0] MATMUL_OFF_STATUS = 12'h00C;
localparam [11:0] MATMUL_OFF_NUM_Q1_BLOCKS = 12'h010;
localparam [11:0] MATMUL_OFF_NUM_ROWBLOCKS = 12'h014;
localparam [11:0] MATMUL_OFF_CYCLES = 12'h018;
localparam [11:0] MATMUL_OFF_ROWS = 12'h01C;
localparam [11:0] MATMUL_OFF_W_STALL = 12'h020;
localparam [11:0] MATMUL_OFF_A_STALL = 12'h024;
localparam [11:0] MATMUL_OFF_R_STALL = 12'h028;
localparam [11:0] MATMUL_OFF_W_BEATS = 12'h02C;
localparam [11:0] MATMUL_OFF_A_BEATS = 12'h030;
localparam [11:0] MATMUL_OFF_R_BEATS = 12'h034;
localparam [11:0] MATMUL_OFF_NUM_COLS = 12'h038;
localparam [11:0] MATMUL_OFF_CLK_HZ = 12'h03C;
localparam [11:0] MATMUL_OFF_WEIGHT_PORTS = 12'h040;
localparam [31:0] MATMUL_RST_ID = 32'hB05A2000;
localparam [31:0] MATMUL_RST_VERSION = 32'h00000009;
localparam [31:0] MATMUL_RST_STATUS = 32'h00000000;
localparam [31:0] MATMUL_RST_CYCLES = 32'h00000000;
localparam [31:0] MATMUL_RST_ROWS = 32'h00000010;
localparam [31:0] MATMUL_RST_W_STALL = 32'h00000000;
localparam [31:0] MATMUL_RST_A_STALL = 32'h00000000;
localparam [31:0] MATMUL_RST_R_STALL = 32'h00000000;
localparam [31:0] MATMUL_RST_W_BEATS = 32'h00000000;
localparam [31:0] MATMUL_RST_A_BEATS = 32'h00000000;
localparam [31:0] MATMUL_RST_R_BEATS = 32'h00000000;
localparam [31:0] MATMUL_RST_CLK_HZ = 32'h00000000;
localparam [31:0] MATMUL_RST_WEIGHT_PORTS = 32'h00000004;
localparam integer MATMUL_COLS_MAX = 8;
`endif
