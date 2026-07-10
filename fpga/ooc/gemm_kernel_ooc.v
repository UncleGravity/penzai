// gemm_kernel_ooc - OOC timing probe for the full gemm_kernel (FSM + banked rowblock +
// pipelined emit + result buffer).
//
// The rowblock (gemm_rb_ooc, 398.9 MHz) and the emit (gemm_emit_ooc, 386.8 MHz) close f300
// in isolation. The full kernel adds paths those don't cover — the FSM control, the BRAM
// acts read, and the result_buf 64:1 read mux → m_axis_tdata. Register all kernel outputs
// here so those internal paths end at a real reg (ooc_synth false-paths the top I/O). If this
// misses f300, register m_axis_tdata in gemm_kernel (pipeline the buffer read).

`default_nettype none

module gemm_kernel_ooc #(
    parameter integer ROWS     = 16,
    parameter integer COLS_MAX = 8
) (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start_kernel,
    input  wire [15:0] num_q1_blocks,
    input  wire [15:0] num_rowblocks,
    input  wire [15:0] num_cols,
    input  wire signed [7:0] emin,
    input  wire [ROWS*32-1:0] s_axis_tdata,
    input  wire        s_axis_tvalid,
    input  wire [63:0] s_axis_acts_tdata,
    input  wire        s_axis_acts_tvalid,
    input  wire        m_axis_tready,
    output reg  [63:0] m_data_q,
    output reg         m_valid_q,
    output reg         m_last_q,
    output reg         w_ready_q,
    output reg         a_ready_q,
    output reg         busy_q,
    output reg         done_q
);
    wire [63:0] m_data;
    wire        m_valid, m_last, w_ready, a_ready, busy, done;
    wire [7:0]  m_keep;
    wire [3:0]  dbg;
    gemm_kernel #(.ROWS(ROWS), .COLS_MAX(COLS_MAX), .MAX_SUB_INDEX(512)) u ( // 512 = deployed (decode_top)
        .clk(clk), .rst_n(rst_n), .start_kernel(start_kernel),
        .num_q1_blocks(num_q1_blocks), .num_rowblocks(num_rowblocks), .num_cols(num_cols),
        .emin(emin), .kernel_done(done), .busy(busy),
        .s_axis_tdata(s_axis_tdata), .s_axis_tvalid(s_axis_tvalid), .s_axis_tready(w_ready),
        .s_axis_acts_tdata(s_axis_acts_tdata), .s_axis_acts_tvalid(s_axis_acts_tvalid),
        .s_axis_acts_tready(a_ready),
        .m_axis_tdata(m_data), .m_axis_tvalid(m_valid), .m_axis_tready(m_axis_tready),
        .m_axis_tlast(m_last), .m_axis_tkeep(m_keep), .dbg_state(dbg)
    );
    always @(posedge clk) begin
        m_data_q  <= m_data;
        m_valid_q <= m_valid;
        m_last_q  <= m_last;
        w_ready_q <= w_ready;
        a_ready_q <= a_ready;
        busy_q    <= busy;
        done_q    <= done;
    end
endmodule
