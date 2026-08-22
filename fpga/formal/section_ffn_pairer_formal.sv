`default_nettype none

// Control and ordering proof boundary for section_ffn_pairer. A maximum-token,
// one-block run retains the native GEMM ingress order:
//
//   token 0..3 groups 0/1, then token 0..3 groups 2/3
//
// while scratch requests and scalar output must be canonical token-major. The
// environment holds every offered input and response stable until handshake and
// keeps an accepted scratch request alive across abort for orphan-drain coverage.
module section_ffn_pairer_formal(input wire clk);
    reg f_past_valid = 1'b0;
    wire rst_n = f_past_valid;

`ifdef FORMAL_DATA
    localparam [2:0] RUN_TOKENS = 3'd4;
    localparam [4:0] RUN_GROUPS = 5'd16;
    localparam [7:0] RUN_SCALARS = 8'd128;
`else
    // Two tokens retain the complete lower-half/upper-half reorder transition
    // while keeping arbitrary-stall/fault control proofs compact.
    localparam [2:0] RUN_TOKENS = 3'd2;
    localparam [4:0] RUN_GROUPS = 5'd8;
    localparam [7:0] RUN_SCALARS = 8'd64;
`endif

    (* anyseq *) reg f_input_offer;
    (* anyseq *) reg f_gate_tag_fault_choice;
    (* anyseq *) reg f_gate_frame_fault_choice;
    (* anyseq *) reg f_request_ready;
    (* anyseq *) reg f_response_offer;
    (* anyseq *) reg f_response_error_choice;
    (* anyseq *) reg f_output_ready;
    (* anyseq *) reg f_abort_choice;

    reg want_start = 1'b1;
    reg [4:0] sent_groups = 5'd0;
    reg [4:0] accepted_requests = 5'd0;
    reg [4:0] accepted_responses = 5'd0;
    reg [7:0] accepted_scalars = 8'd0;
    reg expected_failure_q = 1'b0;

    reg gate_present_q = 1'b0;
    reg gate_tag_fault_q = 1'b0;
    reg gate_frame_fault_q = 1'b0;

    reg env_read_pending = 1'b0;
    reg env_rsp_valid = 1'b0;
    reg env_rsp_error = 1'b0;
    reg [1:0] env_rsp_token = 2'd0;
    reg [1:0] env_rsp_group = 2'd0;

    reg saw_abort = 1'b0;
    reg saw_restart = 1'b0;
    reg saw_gate_fault = 1'b0;
    reg saw_read_error = 1'b0;
    reg saw_request_stall = 1'b0;
    reg saw_output_stall = 1'b0;

`ifdef FORMAL_DATA
    // The data task keeps one real stall at each elastic boundary but removes
    // unrelated fault choices from the reorder-memory payload proof.
    wire input_offer = 1'b1;
    wire request_ready = !rd_req_valid || saw_request_stall;
    wire response_offer = 1'b1;
    wire output_ready = !out_valid || saw_output_stall;
    wire abort_choice = 1'b0;
`elsif FORMAL_COVER
    wire input_offer = 1'b1;
    wire request_ready = !rd_req_valid || saw_request_stall;
    wire response_offer = 1'b1;
    wire output_ready = !out_valid || saw_output_stall;
    // Force one outstanding-read abort; later runs remain free to finish.
    wire abort_choice = !saw_abort && formal_read_inflight && !env_rsp_valid;
`else
    wire input_offer = f_input_offer;
    wire request_ready = f_request_ready;
    wire response_offer = f_response_offer;
    wire output_ready = f_output_ready;
    wire abort_choice = f_abort_choice;
`endif

`ifdef FORMAL_DATA
    wire gate_tag_fault_choice = 1'b0;
    wire gate_frame_fault_choice = 1'b0;
    wire response_error_choice = 1'b0;
`else
    wire gate_tag_fault_choice = f_gate_tag_fault_choice;
    wire gate_frame_fault_choice = f_gate_frame_fault_choice;
    wire response_error_choice = f_response_error_choice;
`endif

    function automatic [31:0] scalar_value(
        input is_up,
        input [1:0] token,
        input [1:0] group,
        input [2:0] lane
    );
        begin
            scalar_value = {is_up, 24'd0, token, group, lane};
        end
    endfunction

    function automatic [255:0] group_value(
        input is_up,
        input [1:0] token,
        input [1:0] group
    );
        integer lane;
        begin
            for (lane = 0; lane < 8; lane = lane + 1)
                group_value[lane * 32 +: 32] =
                    scalar_value(is_up, token, group, lane[2:0]);
        end
    endfunction

    wire start_valid = want_start;
    wire start_ready;
    wire abort_run = rst_n && busy && !saw_abort && abort_choice;
    wire busy;
    wire done;
    wire error;

    // The data task uses the maximum four-token shape. Control tasks use two
    // tokens but retain both native 16-row halves and the canonical replay.
`ifdef FORMAL_DATA
    wire [1:0] native_gate_token = sent_groups[2:1];
    wire [1:0] native_gate_group = {sent_groups[3], sent_groups[0]};
`else
    wire [1:0] native_gate_token = {1'b0, sent_groups[1]};
    wire [1:0] native_gate_group = {sent_groups[2], sent_groups[0]};
`endif
    wire gate_valid = busy && gate_present_q;
    wire gate_ready;
    wire [1:0] gate_token = native_gate_token;
    wire [1:0] gate_group = native_gate_group;
    wire [8:0] gate_block = gate_tag_fault_q ? 9'd1 : 9'd0;
    wire gate_last = (native_gate_group == 2'd3) ^ gate_frame_fault_q;
`ifdef FORMAL_DATA
    wire [255:0] gate_data = group_value(
        1'b0, native_gate_token, native_gate_group);
`else
    wire [255:0] gate_data = 256'd0;
`endif

    wire rd_req_valid;
    wire [1:0] rd_req_role;
    wire [2:0] rd_req_token;
    wire [10:0] rd_req_group;
    wire rd_rsp_ready;
`ifdef FORMAL_DATA
    wire [255:0] env_rsp_data = group_value(
        1'b1, env_rsp_token, env_rsp_group);
`else
    wire [255:0] env_rsp_data = 256'd0;
`endif

    wire out_valid;
    wire [31:0] out_gate;
    wire [31:0] out_up;
    wire out_last;
    wire formal_read_inflight;
    wire formal_orphan;
    wire formal_emit_active;
    wire [2:0] formal_emit_lane;

    wire start_fire = start_valid && start_ready;
    wire gate_fire = gate_valid && gate_ready;
    wire gate_bad = gate_fire && (gate_tag_fault_q || gate_frame_fault_q);
    wire request_fire = rd_req_valid && request_ready;
    wire response_fire = env_rsp_valid && rd_rsp_ready;
    wire live_bad_response = response_fire && busy && formal_read_inflight &&
                             env_rsp_error && !abort_run;
    wire output_fire = out_valid && output_ready;

`ifdef FORMAL_DATA
    wire [1:0] canonical_request_token = accepted_requests[3:2];
    wire [1:0] canonical_output_token = accepted_scalars[6:5];
`else
    wire [1:0] canonical_request_token = {1'b0, accepted_requests[2]};
    wire [1:0] canonical_output_token = {1'b0, accepted_scalars[5]};
`endif
    wire [1:0] canonical_request_group = accepted_requests[1:0];
    wire [1:0] canonical_output_group = accepted_scalars[4:3];
    wire [2:0] canonical_output_lane = accepted_scalars[2:0];

    section_ffn_pairer dut (
        .clk(clk), .rst_n(rst_n),
        .start_valid(start_valid), .start_ready(start_ready),
        .start_tokens(RUN_TOKENS), .start_blocks(9'd1),
        .abort_run(abort_run), .busy(busy), .done(done), .error(error),
        .s_axis_tdata(gate_data), .s_axis_tvalid(gate_valid),
        .s_axis_tready(gate_ready), .s_axis_tready_core(),
        .s_axis_tlast(gate_last),
        .s_axis_token(gate_token), .s_axis_block(gate_block),
        .s_axis_group(gate_group),
        .rd_req_valid(rd_req_valid), .rd_req_ready(request_ready),
        .rd_req_role(rd_req_role), .rd_req_token(rd_req_token),
        .rd_req_group(rd_req_group),
        .rd_rsp_valid(env_rsp_valid), .rd_rsp_ready(rd_rsp_ready),
        .rd_rsp_data(env_rsp_data), .rd_rsp_error(env_rsp_error),
        .out_valid(out_valid), .out_ready(output_ready),
        .out_gate(out_gate), .out_up(out_up), .out_last(out_last),
        .formal_read_inflight(formal_read_inflight),
        .formal_orphan(formal_orphan),
        .formal_emit_active(formal_emit_active),
        .formal_emit_lane(formal_emit_lane)
    );

    always @(posedge clk) begin
        f_past_valid <= 1'b1;

        if (!rst_n) begin
            want_start <= 1'b1;
            sent_groups <= 5'd0;
            accepted_requests <= 5'd0;
            accepted_responses <= 5'd0;
            accepted_scalars <= 8'd0;
            expected_failure_q <= 1'b0;
            gate_present_q <= 1'b0;
            gate_tag_fault_q <= 1'b0;
            gate_frame_fault_q <= 1'b0;
            env_read_pending <= 1'b0;
            env_rsp_valid <= 1'b0;
            env_rsp_error <= 1'b0;
            env_rsp_token <= 2'd0;
            env_rsp_group <= 2'd0;
            saw_abort <= 1'b0;
            saw_restart <= 1'b0;
            saw_gate_fault <= 1'b0;
            saw_read_error <= 1'b0;
            saw_request_stall <= 1'b0;
            saw_output_stall <= 1'b0;
        end else begin
            if (start_fire) begin
                want_start <= 1'b0;
                sent_groups <= 5'd0;
                accepted_requests <= 5'd0;
                accepted_responses <= 5'd0;
                accepted_scalars <= 8'd0;
                expected_failure_q <= 1'b0;
                gate_present_q <= 1'b0;
                gate_tag_fault_q <= 1'b0;
                gate_frame_fault_q <= 1'b0;
                if (saw_abort)
                    saw_restart <= 1'b1;
            end

            if (done)
                want_start <= 1'b1;

            if (!busy) begin
                gate_present_q <= 1'b0;
            end else if (gate_fire) begin
                gate_present_q <= 1'b0;
            end else if (!gate_present_q && (sent_groups < RUN_GROUPS) &&
                         input_offer) begin
                gate_present_q <= 1'b1;
                gate_tag_fault_q <= gate_tag_fault_choice;
                gate_frame_fault_q <= gate_frame_fault_choice;
            end

            if (gate_fire && !gate_bad)
                sent_groups <= sent_groups + 1'b1;
            if (gate_bad)
                saw_gate_fault <= 1'b1;
            if (gate_bad || live_bad_response || abort_run)
                expected_failure_q <= 1'b1;

            if (request_fire) begin
                assert(!env_read_pending);
                env_read_pending <= 1'b1;
                env_rsp_token <= rd_req_token[1:0];
                env_rsp_group <= rd_req_group[1:0];
                accepted_requests <= accepted_requests + 1'b1;
            end

            if (!env_rsp_valid && env_read_pending && response_offer) begin
                env_rsp_valid <= 1'b1;
                env_rsp_error <= response_error_choice;
            end else if (response_fire) begin
                env_rsp_valid <= 1'b0;
                env_rsp_error <= 1'b0;
                env_read_pending <= 1'b0;
                if (busy && formal_read_inflight && !env_rsp_error && !abort_run)
                    accepted_responses <= accepted_responses + 1'b1;
            end

            if (live_bad_response)
                saw_read_error <= 1'b1;

            if (output_fire)
                accepted_scalars <= accepted_scalars + 1'b1;

            if (abort_run) begin
                saw_abort <= 1'b1;
                sent_groups <= 5'd0;
                accepted_requests <= 5'd0;
                accepted_responses <= 5'd0;
                accepted_scalars <= 8'd0;
                gate_present_q <= 1'b0;
            end

            if (rd_req_valid && !request_ready)
                saw_request_stall <= 1'b1;
            if (out_valid && !output_ready)
                saw_output_stall <= 1'b1;

            assert(sent_groups <= RUN_GROUPS);
            assert(accepted_requests <= RUN_GROUPS);
            assert(accepted_requests <= sent_groups);
            assert(accepted_responses <= accepted_requests);
            assert(accepted_scalars <= {accepted_responses, 3'b000});
            assert(formal_emit_lane <= 3'd7);

            if (gate_valid) begin
                assert(sent_groups < RUN_GROUPS);
                assert(gate_token == native_gate_token);
                assert(gate_group == native_gate_group);
                assert(gate_block == (gate_tag_fault_q ? 9'd1 : 9'd0));
                assert(gate_last ==
                       ((native_gate_group == 2'd3) ^ gate_frame_fault_q));
            end

            if (rd_req_valid) begin
                assert(rd_req_role == 2'd2);
                assert(accepted_requests < RUN_GROUPS);
                assert(rd_req_token == {1'b0, canonical_request_token});
                assert(rd_req_group == {9'd0, canonical_request_group});
            end

            if (out_valid) begin
                assert(accepted_scalars < RUN_SCALARS);
`ifdef FORMAL_DATA
                assert(out_gate == scalar_value(
                    1'b0, canonical_output_token,
                    canonical_output_group, canonical_output_lane));
                assert(out_up == scalar_value(
                    1'b1, canonical_output_token,
                    canonical_output_group, canonical_output_lane));
`endif
                assert(out_last == (accepted_scalars[4:0] == 5'd31));
            end

            if (done && !error) begin
                assert(!expected_failure_q);
                assert(sent_groups == RUN_GROUPS);
                assert(accepted_requests == RUN_GROUPS);
                assert(accepted_responses == RUN_GROUPS);
                assert(accepted_scalars == RUN_SCALARS);
            end
            if (done && error)
                assert(expected_failure_q);

            if (start_fire)
                assert(!env_read_pending);
            if (env_rsp_valid)
                assert(env_read_pending);
            if (formal_read_inflight)
                assert(env_read_pending);
            if (formal_orphan)
                assert(env_read_pending);
        end

        if (f_past_valid && rst_n && busy && !abort_run &&
            $past(rst_n && busy && gate_valid && !gate_ready && !abort_run)) begin
            assert(gate_valid);
            assert(gate_data == $past(gate_data));
            assert(gate_last == $past(gate_last));
            assert(gate_token == $past(gate_token));
            assert(gate_block == $past(gate_block));
            assert(gate_group == $past(gate_group));
        end

        if (f_past_valid && rst_n && busy && !abort_run &&
            $past(rst_n && busy && rd_req_valid && !request_ready && !abort_run)) begin
            assert(rd_req_valid);
            assert(rd_req_role == $past(rd_req_role));
            assert(rd_req_token == $past(rd_req_token));
            assert(rd_req_group == $past(rd_req_group));
        end

        if (f_past_valid && rst_n && busy && !abort_run &&
            $past(rst_n && busy && out_valid && !output_ready && !abort_run)) begin
            assert(out_valid);
`ifdef FORMAL_DATA
            assert(out_gate == $past(out_gate));
            assert(out_up == $past(out_up));
`endif
            assert(out_last == $past(out_last));
        end

        if (f_past_valid &&
            $past(rst_n && (abort_run || gate_bad || live_bad_response))) begin
            assert(!busy);
            assert(!gate_ready);
            assert(!rd_req_valid);
            assert(!out_valid);
            assert(error);
        end

        if (rst_n && formal_orphan) begin
            assert(!busy);
            assert(!start_ready);
            assert(rd_rsp_ready);
        end

`ifdef FORMAL_COVER
        cover(rst_n && formal_emit_active && env_rsp_valid &&
              !rd_rsp_ready && saw_output_stall);
        cover(rst_n && saw_abort && formal_orphan && env_rsp_valid);
`endif
    end
endmodule

`default_nettype wire
