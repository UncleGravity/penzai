// One shared exact-accumulator -> FP32 stream boundary for every projection.

`default_nettype none

module emit_stream #(
    parameter integer DEPTH = 8
) (
    input  wire                  clk,
    input  wire                  rst_n,
    input  wire                  clear,

    input  wire                  in_valid,
    output wire                  in_ready,
    input  wire signed [103:0]   in_acc,
    input  wire signed [7:0]     in_emin,
    input  wire [2:0]            in_token,
    input  wire [17:0]           in_row,
    input  wire                  in_last,

    output wire                  out_valid,
    input  wire                  out_ready,
    output wire [31:0]           out_data,
    output wire [2:0]            out_token,
    output wire [17:0]           out_row,
    output wire                  out_last,
    output wire [$clog2(DEPTH+1)-1:0] reserved
);
    localparam integer PTR_W = $clog2(DEPTH);
    localparam integer CNT_W = $clog2(DEPTH + 1);
    localparam integer TAG_W = 22;
    localparam integer PACKET_W = 32 + TAG_W;

    initial begin
        if (DEPTH < 4 || (DEPTH & (DEPTH - 1)) != 0)
            $error(" emit_stream DEPTH must be a power of two >= 4");
    end

    reg [PACKET_W-1:0] fifo [0:DEPTH-1];
    reg [PTR_W-1:0] wr_ptr_q;
    reg [PTR_W-1:0] rd_ptr_q;
    reg [CNT_W-1:0] fifo_count_q;
    reg [CNT_W-1:0] reserved_q;
    reg [TAG_W-1:0] tag1_q;
    reg [TAG_W-1:0] tag2_q;
    reg [TAG_W-1:0] tag3_q;
    reg [TAG_W-1:0] tag4_q;

    wire [TAG_W-1:0] in_tag = {in_last, in_token, in_row};
    wire emit_valid;
    wire [31:0] emit_data;
    wire out_fire = out_valid && out_ready;
    assign in_ready = (reserved_q < DEPTH) || out_fire;
    wire in_fire = in_valid && in_ready;

    gemm_emit #(.ACC_W(104), .EXP_W(8)) u_emit (
        .clk(clk),
        .rst_n(rst_n && !clear),
        .valid_in(in_fire),
        .acc(in_acc),
        .emin(in_emin),
        .valid_out(emit_valid),
        .f32(emit_data)
    );

    assign out_valid = fifo_count_q != 0;
    assign {out_last, out_token, out_row, out_data} = fifo[rd_ptr_q];
    assign reserved = reserved_q;

    always @(posedge clk) begin
        if (!rst_n || clear) begin
            wr_ptr_q <= {PTR_W{1'b0}};
            rd_ptr_q <= {PTR_W{1'b0}};
            fifo_count_q <= {CNT_W{1'b0}};
            reserved_q <= {CNT_W{1'b0}};
            tag1_q <= {TAG_W{1'b0}};
            tag2_q <= {TAG_W{1'b0}};
            tag3_q <= {TAG_W{1'b0}};
            tag4_q <= {TAG_W{1'b0}};
        end else begin
            tag1_q <= in_tag;
            tag2_q <= tag1_q;
            tag3_q <= tag2_q;
            tag4_q <= tag3_q;

            if (emit_valid) begin
                fifo[wr_ptr_q] <= {tag4_q, emit_data};
                wr_ptr_q <= wr_ptr_q + 1'b1;
            end
            if (out_fire)
                rd_ptr_q <= rd_ptr_q + 1'b1;

            case ({emit_valid, out_fire})
                2'b10: fifo_count_q <= fifo_count_q + 1'b1;
                2'b01: fifo_count_q <= fifo_count_q - 1'b1;
                default: ;
            endcase
            case ({in_fire, out_fire})
                2'b10: reserved_q <= reserved_q + 1'b1;
                2'b01: reserved_q <= reserved_q - 1'b1;
                default: ;
            endcase
        end
    end
endmodule

`default_nettype wire
