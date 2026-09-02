// Registered out-of-context shell for the complete serialized four-lane, tile-8 engine.

`default_nettype none

module projection_ooc (
    input  wire                  clk,
    input  wire                  rst_n,
    input  wire                  clear,
    input  wire                  start,
    input  wire [15:0]           model_spec_k,
    input  wire [17:0]           model_spec_m,
    input  wire [15:0]           model_spec_rowblocks,
    input  wire [1:0]            weight_fmt,
    input  wire signed [7:0]     emin,
    input  wire [7:0]            token_mask,
    output wire                  busy,
    output wire                  done,
    output wire                  error,

    input  wire [511:0]          w_data,
    input  wire                  w_valid,
    output wire                  w_ready,

    output wire                  act_req_valid,
    input  wire                  act_req_ready,
    output wire [8:0]            act_req_addr,
    output wire                  act_req_wave,
    input  wire                  act_rsp_valid,
    output wire                  act_rsp_ready,
    input  wire [1087:0]         act_rsp_data,

    output wire signed [103:0]   out_acc,
    output wire signed [7:0]     out_emin,
    output wire [2:0]            out_token,
    output wire [17:0]           out_row,
    output wire                  out_last,
    output wire                  out_valid,
    input  wire                  out_ready,
    output wire [31:0]           weight_beat_count,
    output wire [31:0]           wave_issue_count
);
    reg clear_q;
    reg start_q;
    reg [15:0] model_spec_k_q;
    reg [17:0] model_spec_m_q;
    reg [15:0] model_spec_rowblocks_q;
    reg [1:0] weight_fmt_q;
    reg signed [7:0] emin_q;
    reg [7:0] token_mask_q;
    reg [511:0] w_data_q;
    reg w_valid_q;
    reg act_req_ready_q;
    reg act_rsp_valid_q;
    reg [1087:0] act_rsp_data_q;
    reg out_ready_q;

    always @(posedge clk) begin
        if (!rst_n) begin
            clear_q <= 1'b0;
            start_q <= 1'b0;
            w_valid_q <= 1'b0;
            act_req_ready_q <= 1'b0;
            act_rsp_valid_q <= 1'b0;
            out_ready_q <= 1'b0;
        end else begin
            clear_q <= clear;
            start_q <= start;
            w_valid_q <= w_valid;
            act_req_ready_q <= act_req_ready;
            act_rsp_valid_q <= act_rsp_valid;
            out_ready_q <= out_ready;
        end
        model_spec_k_q <= model_spec_k;
        model_spec_m_q <= model_spec_m;
        model_spec_rowblocks_q <= model_spec_rowblocks;
        weight_fmt_q <= weight_fmt;
        emin_q <= emin;
        token_mask_q <= token_mask;
        w_data_q <= w_data;
        act_rsp_data_q <= act_rsp_data;
    end

     projection_engine u_engine (
        .clk(clk),
        .rst_n(rst_n),
        .clear(clear_q),
        .start(start_q),
        .model_spec_k(model_spec_k_q),
        .model_spec_m(model_spec_m_q),
        .model_spec_rowblocks(model_spec_rowblocks_q),
        .weight_fmt(weight_fmt_q),
        .emin(emin_q),
        .token_mask(token_mask_q),
        .busy(busy),
        .done(done),
        .error(error),
        .w_data(w_data_q),
        .w_valid(w_valid_q),
        .w_ready(w_ready),
        .act_req_valid(act_req_valid),
        .act_req_ready(act_req_ready_q),
        .act_req_addr(act_req_addr),
        .act_req_wave(act_req_wave),
        .act_rsp_valid(act_rsp_valid_q),
        .act_rsp_ready(act_rsp_ready),
        .act_rsp_data(act_rsp_data_q),
        .out_acc(out_acc),
        .out_emin(out_emin),
        .out_token(out_token),
        .out_row(out_row),
        .out_last(out_last),
        .out_valid(out_valid),
        .out_ready(out_ready_q),
        .weight_beat_count(weight_beat_count),
        .wave_issue_count(wave_issue_count)
    );
endmodule

`default_nettype wire
