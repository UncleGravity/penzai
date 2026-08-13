`default_nettype none

module decode_ffn_formal(input wire clk);
    (* anyconst *) reg abort_case;
    reg rst_n = 1'b0;
    reg [8:0] cycle = 9'd0;

    wire [6:0] command = cycle[8:2];
    wire [1:0] phase = cycle[1:0];
    reg [7:0] write_addr;
    reg [31:0] write_data;
    reg write_enable;

    localparam [7:0] REG_CTRL = 8'h08;
    localparam [7:0] REG_NUM_Q1 = 8'h10;
    localparam [7:0] REG_NUM_RB = 8'h14;
    localparam [7:0] REG_NUM_COLS = 8'h38;
    localparam [7:0] REG_WEIGHT_FMT = 8'h44;
    localparam [7:0] REG_NUM_ROWS = 8'h48;
    localparam [7:0] REG_ACT_MODE = 8'h4c;
    localparam [7:0] REG_ACT_EPOCH = 8'h50;
    localparam [7:0] REG_SCRATCH_MODE = 8'h68;
    localparam [7:0] REG_SCRATCH_ROLE = 8'h6c;
    localparam [7:0] REG_SCRATCH_ROWS = 8'h70;
    localparam [7:0] REG_SCRATCH_TOKENS = 8'h74;
    localparam [7:0] REG_SCRATCH_CTRL = 8'h78;

    always @* begin
        write_addr = 8'd0;
        write_data = 32'd0;
        write_enable = 1'b1;
        case (command)
            7'd0:  begin write_addr = REG_SCRATCH_MODE; write_data = 0; end
            7'd1:  begin write_addr = REG_SCRATCH_ROWS; write_data = 128; end
            7'd2:  begin write_addr = REG_SCRATCH_TOKENS; write_data = 1; end
            7'd3:  begin write_addr = REG_SCRATCH_CTRL; write_data = 4; end
            7'd4:  begin write_addr = REG_SCRATCH_MODE; write_data = 3; end
            7'd5:  begin write_addr = REG_SCRATCH_ROLE; write_data = 2; end
            7'd6:  begin write_addr = REG_NUM_Q1; write_data = 1; end
            7'd7:  begin write_addr = REG_NUM_RB; write_data = 8; end
            7'd8:  begin write_addr = REG_NUM_ROWS; write_data = 128; end
            7'd9:  begin write_addr = REG_NUM_COLS; write_data = 1; end
            7'd10: begin write_addr = REG_WEIGHT_FMT; write_data = 1; end
            7'd11: begin write_addr = REG_ACT_MODE; write_data = 2; end
            7'd12: begin write_addr = REG_ACT_EPOCH; write_data = 1; end
            7'd13: begin write_addr = REG_CTRL; write_data = 1; end
            7'd16: begin write_addr = REG_SCRATCH_ROLE; write_data = 1; end
            7'd17: begin write_addr = REG_ACT_MODE; write_data = 1; end
            7'd18: begin write_addr = REG_CTRL; write_data = 1; end
            7'd23: begin write_addr = REG_SCRATCH_MODE; write_data = 0; end
            7'd24: begin write_addr = REG_ACT_MODE; write_data = 3; end
            7'd25: begin write_addr = REG_ACT_EPOCH; write_data = 2; end
            7'd26: begin write_addr = REG_CTRL; write_data = 1; end
            7'd35: begin
                write_addr = REG_SCRATCH_CTRL;
                write_data = 2;
                write_enable = abort_case;
            end
            default: write_enable = 1'b0;
        endcase
    end

    wire axi_valid = rst_n && write_enable && (phase < 2);
    wire [1:0] bresp;
    wire bvalid;
    wire awready, wready;
    wire [31:0] rdata;
    wire [1:0] rresp;
    wire arready, rvalid;
    wire [127:0] weight_word = 128'd0;
    wire w0_ready, w1_ready, w2_ready, w3_ready;
    wire acts_ready;
    wire [63:0] result_data;
    wire [7:0] result_keep;
    wire result_valid, result_last;

    decode_top dut (
        .s_axi_aclk(clk), .s_axi_aresetn(rst_n),
        .s_axi_awaddr(write_addr), .s_axi_awprot(3'd0),
        .s_axi_awvalid(axi_valid), .s_axi_awready(awready),
        .s_axi_wdata(write_data), .s_axi_wstrb(4'hf),
        .s_axi_wvalid(axi_valid), .s_axi_wready(wready),
        .s_axi_bresp(bresp), .s_axi_bvalid(bvalid), .s_axi_bready(1'b1),
        .s_axi_araddr(8'd0), .s_axi_arprot(3'd0),
        .s_axi_arvalid(1'b0), .s_axi_arready(arready),
        .s_axi_rdata(rdata), .s_axi_rresp(rresp),
        .s_axi_rvalid(rvalid), .s_axi_rready(1'b1),
        .s_axis_w0_tdata(weight_word), .s_axis_w0_tkeep(16'hffff),
        .s_axis_w0_tvalid(1'b1), .s_axis_w0_tready(w0_ready),
        .s_axis_w0_tlast(1'b0),
        .s_axis_w1_tdata(weight_word), .s_axis_w1_tkeep(16'hffff),
        .s_axis_w1_tvalid(1'b1), .s_axis_w1_tready(w1_ready),
        .s_axis_w1_tlast(1'b0),
        .s_axis_w2_tdata(weight_word), .s_axis_w2_tkeep(16'hffff),
        .s_axis_w2_tvalid(1'b1), .s_axis_w2_tready(w2_ready),
        .s_axis_w2_tlast(1'b0),
        .s_axis_w3_tdata(weight_word), .s_axis_w3_tkeep(16'hffff),
        .s_axis_w3_tvalid(1'b1), .s_axis_w3_tready(w3_ready),
        .s_axis_w3_tlast(1'b0),
        .s_axis_acts_tdata(64'd0), .s_axis_acts_tkeep(8'hff),
        .s_axis_acts_tvalid(1'b1), .s_axis_acts_tready(acts_ready),
        .s_axis_acts_tlast(1'b0),
        .m_axis_tdata(result_data), .m_axis_tkeep(result_keep),
        .m_axis_tvalid(result_valid), .m_axis_tready(1'b1),
        .m_axis_tlast(result_last)
    );

    always @(posedge clk) begin
        rst_n <= 1'b1;
        if (!rst_n) cycle <= 9'd0;
        else if (cycle != 9'h1ff) cycle <= cycle + 1'b1;

        if (rst_n) begin
            if (axi_valid) assert(write_addr != 0);
        end
    end

    wire _unused = &{1'b0, bresp, bvalid, awready, wready, rdata, rresp,
                     arready, rvalid, w0_ready, w1_ready, w2_ready, w3_ready,
                     acts_ready, result_data, result_keep, result_valid,
                     result_last};
endmodule

`default_nettype wire
