// Generated from fpga/regmap/flash_attn.regmap — do not edit.
`ifndef FLASH_REGS_VH
`define FLASH_REGS_VH
localparam [11:0] FLASH_OFF_ID = 12'h000;
localparam [11:0] FLASH_OFF_VERSION = 12'h004;
localparam [11:0] FLASH_OFF_CTRL = 12'h008;
localparam [11:0] FLASH_OFF_STATUS = 12'h00C;
localparam [11:0] FLASH_OFF_HEAD_DIM_Q = 12'h010;
localparam [11:0] FLASH_OFF_HEAD_DIM_V = 12'h014;
localparam [11:0] FLASH_OFF_N_HEADS = 12'h018;
localparam [11:0] FLASH_OFF_N_KV = 12'h01C;
localparam [11:0] FLASH_OFF_N_TOKENS = 12'h020;
localparam [11:0] FLASH_OFF_SCALE = 12'h024;
localparam [11:0] FLASH_OFF_CYCLES = 12'h028;
localparam [11:0] FLASH_OFF_CLK_HZ = 12'h02C;
localparam [11:0] FLASH_OFF_LANES = 12'h030;
localparam [11:0] FLASH_OFF_Q_BEATS = 12'h034;
localparam [11:0] FLASH_OFF_K_BEATS = 12'h038;
localparam [11:0] FLASH_OFF_K_STALL = 12'h03C;
localparam [11:0] FLASH_OFF_V_BEATS = 12'h040;
localparam [11:0] FLASH_OFF_V_STALL = 12'h044;
localparam [11:0] FLASH_OFF_O_BEATS = 12'h048;
localparam [11:0] FLASH_OFF_O_STALL = 12'h04C;
localparam [31:0] FLASH_RST_ID = 32'hF1A54A00;
localparam [31:0] FLASH_RST_VERSION = 32'h00000001;
localparam [31:0] FLASH_RST_STATUS = 32'h00000000;
localparam [31:0] FLASH_RST_CYCLES = 32'h00000000;
localparam [31:0] FLASH_RST_CLK_HZ = 32'h00000000;
localparam [31:0] FLASH_RST_LANES = 32'h00000008;
localparam [31:0] FLASH_RST_Q_BEATS = 32'h00000000;
localparam [31:0] FLASH_RST_K_BEATS = 32'h00000000;
localparam [31:0] FLASH_RST_K_STALL = 32'h00000000;
localparam [31:0] FLASH_RST_V_BEATS = 32'h00000000;
localparam [31:0] FLASH_RST_V_STALL = 32'h00000000;
localparam [31:0] FLASH_RST_O_BEATS = 32'h00000000;
localparam [31:0] FLASH_RST_O_STALL = 32'h00000000;
localparam integer FLASH_LANES = 8;
`endif
