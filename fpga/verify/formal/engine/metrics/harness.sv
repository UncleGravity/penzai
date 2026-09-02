`default_nettype none

module engine_metrics_harness (
    input wire clk
);
    (* anyseq *) reg rst_n;
    (* anyseq *) reg start;
    (* anyseq *) reg finish;
    (* anyseq *) reg acknowledge;
    (* anyseq *) reg [1:0] finish_outcome;
    (* anyseq *) reg core_stage_active;
    (* anyseq *) reg core_stage_call;
    (* anyseq *) reg [12:0] projection_probe;
    (* anyseq *) reg [2:0] weight_axi_r_beats;
    (* anyseq *) reg [2:0] weight_axi_r_gap_ports;
    (* anyseq *) reg weight_zip_skew;
    (* anyseq *) reg [1:0] history_axi_r_beats;
    (* anyseq *) reg kv_axi_w_beat;

    wire [31:0] read_data_lo;
    wire [31:0] read_data_hi;
    wire [31:0] schema;
    wire [31:0] capabilities;
    wire [31:0] status;
    wire [31:0] snapshot_tag;
    wire [31:0] overflow0;
    wire [31:0] overflow1;
    wire [31:0] overflow2;
    wire [31:0] overflow3;
    wire [63:0] total_cycles;
    wire recording;
    wire sealing;
    wire snapshot_valid;

    engine_metrics #(
        .COUNTER_W(4),
        .CALL_COUNTER_W(4)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .start_tag(32'h1234_5678),
        .finish(finish),
        .finish_outcome(finish_outcome),
        .acknowledge(acknowledge),
        .core_stage_active(core_stage_active),
        .core_stage(5'd0),
        .core_stage_call(core_stage_call),
        .core_stage_call_id(5'd0),
        .projection_probe(projection_probe),
        .weight_axi_r_beats(weight_axi_r_beats),
        .weight_axi_r_gap_ports(weight_axi_r_gap_ports),
        .weight_zip_skew(weight_zip_skew),
        .history_axi_r_beats(history_axi_r_beats),
        .kv_axi_w_beat(kv_axi_w_beat),
        .read_index(7'd0),
        .read_data_lo(read_data_lo),
        .read_data_hi(read_data_hi),
        .schema(schema),
        .capabilities(capabilities),
        .status(status),
        .snapshot_tag(snapshot_tag),
        .overflow0(overflow0),
        .overflow1(overflow1),
        .overflow2(overflow2),
        .overflow3(overflow3),
        .total_cycles(total_cycles),
        .recording(recording),
        .sealing(sealing),
        .snapshot_valid(snapshot_valid)
    );

    reg f_past_valid = 1'b0;
    always @(posedge clk) begin
        f_past_valid <= 1'b1;
        if (!f_past_valid)
            assume(!rst_n);
        else
            assume(rst_n);

        assume(!start || (!recording && !sealing && !snapshot_valid));
        assume(!finish || recording);
        assume(!acknowledge || snapshot_valid);
        assume(!(start && finish));
        assume(!(start && acknowledge));

        if (rst_n) begin
            assert(schema == 32'h0001_0000);
            assert(capabilities == 32'h0028_0b1d);
            assert(status[3]);
            assert(total_cycles == {read_data_hi, read_data_lo});

            cover(snapshot_valid && status[6:5] == 2'd1);
            cover(snapshot_valid && status[6:5] == 2'd3);
            cover(snapshot_valid && overflow0[24]);
        end
    end
endmodule

`default_nettype wire
