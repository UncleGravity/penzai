`default_nettype none

module q8_internal_ingress_formal(input wire clk);
    (* anyconst *) reg [1:0] fault_mode;
    (* anyseq *) reg [31:0] internal_data;
    (* anyseq *) reg m_axis_tready;
    (* anyseq *) reg [63:0] external_data;
    (* anyseq *) reg external_valid;
    (* anyseq *) reg abort_request;

    reg f_past_valid = 1'b0;
    reg [3:0] step = 4'd0;
    reg [5:0] source_lane = 6'd0;
    reg restarted = 1'b0;
    reg fault_seen = 1'b0;
    reg explicit_abort_seen = 1'b0;
    reg post_abort_restart = 1'b0;

    wire rst_n = f_past_valid;
    wire abort = rst_n && (fault_mode == 2'd0) &&
                 !explicit_abort_seen && abort_request;
    wire restart = (fault_seen || explicit_abort_seen) && !restarted && !abort;
    wire start = (step == 4'd1) || restart;
    wire inject_fault = (source_lane == 6'd7);
    wire internal_last = (source_lane == 6'd31) ||
                         ((fault_mode == 2'd2) && inject_fault);
    wire [1:0] internal_status =
        ((fault_mode == 2'd1) && inject_fault) ? 2'd1 : 2'd0;
    wire internal_valid = rst_n && !start && !abort;
    wire internal_ready;
    wire internal_record_done;
    wire external_ready;
    wire [63:0] m_axis_tdata;
    wire m_axis_tvalid;
    wire activation_abort;
    wire [5:0] quantizer_status;

    q8_ingress dut (
        .clk(clk), .rst_n(rst_n), .start(start), .abort(abort),
        .raw_mode(1'b1), .internal_mode(1'b1),
        .num_q1_blocks(16'd1), .num_cols(16'd1),
        .s_axis_tdata(external_data), .s_axis_tvalid(external_valid),
        .s_axis_tready(external_ready), .s_axis_tlast(1'b0),
        .internal_data(internal_data), .internal_last(internal_last),
        .internal_status(internal_status), .internal_valid(internal_valid),
        .internal_ready(internal_ready),
        .internal_record_done(internal_record_done),
        .m_axis_tdata(m_axis_tdata), .m_axis_tvalid(m_axis_tvalid),
        .m_axis_tready(m_axis_tready), .activation_abort(activation_abort),
        .quantizer_status(quantizer_status)
    );

    wire internal_accept = internal_valid && internal_ready;
    always @(posedge clk) begin
        f_past_valid <= 1'b1;
        assume(fault_mode <= 2'd2);
        if (!rst_n) begin
            step <= 4'd0;
            source_lane <= 6'd0;
            restarted <= 1'b0;
            fault_seen <= 1'b0;
            explicit_abort_seen <= 1'b0;
            post_abort_restart <= 1'b0;
        end else begin
            if (step != 4'hf) step <= step + 1'b1;
            if (start) source_lane <= 6'd0;
            else if (internal_accept) begin
                if (source_lane == 6'd31) source_lane <= 6'd0;
                else source_lane <= source_lane + 1'b1;
                if ((internal_status != 0) ||
                    (internal_last != (source_lane == 6'd31)))
                    fault_seen <= 1'b1;
            end
            if (abort) begin
                explicit_abort_seen <= 1'b1;
            end
            if (restart) begin
                restarted <= 1'b1;
                if (explicit_abort_seen)
                    post_abort_restart <= 1'b1;
            end
        end

        if (f_past_valid && $past(rst_n) &&
            $past(internal_valid && !internal_ready)) begin
            assume(internal_data == $past(internal_data));
        end

        if (rst_n) begin
            assert(!external_ready);
            if (abort) begin
                assert(!internal_ready);
                assert(!internal_record_done);
                assert(!m_axis_tvalid);
                assert(!activation_abort);
                assert(quantizer_status == 6'd0);
            end
            if (activation_abort) begin
                assert(fault_mode != 2'd0);
                assert(!m_axis_tvalid && !internal_ready);
            end
            if (f_past_valid && $past(restart)) begin
                assert(!activation_abort);
                assert(internal_ready);
                assert(quantizer_status == 6'd0);
            end
            if (f_past_valid && $past(abort)) begin
                assert(!m_axis_tvalid && !internal_ready);
                assert(!internal_record_done && !activation_abort);
                assert(quantizer_status == 6'd0);
            end
        end

        cover(rst_n && (fault_mode == 2'd0) && internal_record_done);
        cover(rst_n && (fault_mode != 2'd0) && restarted && internal_accept);
        cover(rst_n && (fault_mode == 2'd0) && post_abort_restart &&
              internal_record_done);
    end

    wire _unused = &{1'b0, m_axis_tdata};
endmodule

`default_nettype wire
