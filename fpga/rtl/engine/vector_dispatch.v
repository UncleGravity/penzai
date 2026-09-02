`default_nettype none

`include "engine_defs.vh"

// Lock one controller vector request to its physical service until the done
// record is accepted.  The lock also owns every shared-arena response routed
// by engine_datapath.
module vector_dispatch (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         clear,

    input  wire         req_valid,
    output wire         req_ready,
    input  wire [3:0]   req_op,
    output wire         done_valid,
    input  wire         done_ready,
    output wire         done_error,
    output wire [7:0]   done_status,

    output wire         embed_cmd_valid,
    input  wire         embed_cmd_ready,
    input  wire         embed_done_valid,
    output wire         embed_done_ready,
    input  wire         embed_done_error,
    input  wire [7:0]   embed_done_status,

    output wire         norm_cmd_valid,
    input  wire         norm_cmd_ready,
    input  wire         norm_done_valid,
    output wire         norm_done_ready,
    input  wire         norm_done_error,
    input  wire [15:0]  norm_done_status,

    output wire         append_cmd_valid,
    input  wire         append_cmd_ready,
    input  wire         append_done_valid,
    output wire         append_done_ready,
    input  wire         append_done_error,
    input  wire [7:0]   append_done_status,

    output wire [1:0]   owner,
    output wire         busy,
    output reg          protocol_error
);
    localparam [1:0] OWNER_NONE   = 2'd0;
    localparam [1:0] OWNER_EMBED  = 2'd1;
    localparam [1:0] OWNER_NORM   = 2'd2;
    localparam [1:0] OWNER_APPEND = 2'd3;

    reg [1:0] owner_q;

    wire op_embed = req_op == `VECTOR_OP_EMBED;
    wire op_norm = (req_op == `VECTOR_OP_ATTN_NORM) ||
                   (req_op == `VECTOR_OP_FFN_NORM) ||
                   (req_op == `VECTOR_OP_FINAL_NORM);
    wire op_append = req_op == `VECTOR_OP_KV_APPEND;
    wire op_legal = op_embed || op_norm || op_append;

    wire idle = owner_q == OWNER_NONE;
    assign embed_cmd_valid = idle && req_valid && op_embed;
    assign norm_cmd_valid = idle && req_valid && op_norm;
    assign append_cmd_valid = idle && req_valid && op_append;
    assign req_ready = idle && op_legal &&
                       (op_embed ? embed_cmd_ready :
                        op_norm ? norm_cmd_ready : append_cmd_ready);

    assign done_valid = (owner_q == OWNER_EMBED) ? embed_done_valid :
                        (owner_q == OWNER_NORM) ? norm_done_valid :
                        (owner_q == OWNER_APPEND) ? append_done_valid : 1'b0;
    assign done_error = (owner_q == OWNER_EMBED) ? embed_done_error :
                        (owner_q == OWNER_NORM) ? norm_done_error :
                        (owner_q == OWNER_APPEND) ? append_done_error : 1'b0;
    assign done_status = (owner_q == OWNER_EMBED) ? embed_done_status :
                         (owner_q == OWNER_NORM) ?
                            (norm_done_status[7:0] |
                             ((|norm_done_status[15:8]) ? 8'h80 : 8'h00)) :
                         (owner_q == OWNER_APPEND) ? append_done_status : 8'd0;

    assign embed_done_ready = (owner_q == OWNER_EMBED) && done_ready;
    assign norm_done_ready = (owner_q == OWNER_NORM) && done_ready;
    assign append_done_ready = (owner_q == OWNER_APPEND) && done_ready;
    assign owner = owner_q;
    assign busy = owner_q != OWNER_NONE;

    always @(posedge clk) begin
        if (!rst_n || clear) begin
            owner_q <= OWNER_NONE;
            protocol_error <= 1'b0;
        end else begin
            if (req_valid && idle && !op_legal)
                protocol_error <= 1'b1;

            if (req_valid && req_ready)
                owner_q <= op_embed ? OWNER_EMBED :
                           op_norm ? OWNER_NORM : OWNER_APPEND;
            else if (done_valid && done_ready)
                owner_q <= OWNER_NONE;
        end
    end

`ifndef SYNTHESIS
    always @(posedge clk) begin
        if (rst_n && !clear && (owner_q == OWNER_NONE) && done_valid)
            $fatal(1, " vector_dispatch done without owner");
    end
`endif
endmodule

`default_nettype wire
