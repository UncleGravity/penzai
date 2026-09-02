`default_nettype none

// Serialize embedding reads and the two coefficient streams onto one logical
// client of  quad_read_arbiter. Embedding keeps the native 512-bit service
// contract; coefficient clients use only port zero and see 128-bit records.
module small_read_mux (
    input  wire          clk,
    input  wire          rst_n,
    input  wire          clear,
    input  wire          abort_run,

    input  wire          embed_cmd_valid,
    output wire          embed_cmd_ready,
    input  wire [63:0]   embed_cmd_base_addr,
    input  wire [31:0]   embed_cmd_port_beats,
    input  wire [3:0]    embed_cmd_port_mask,
    input  wire          embed_abort,
    output wire [511:0]  embed_data,
    output wire          embed_valid,
    input  wire          embed_ready,
    output wire          embed_last,
    output wire          embed_error,
    output wire          embed_busy,
    output wire          embed_done_valid,
    input  wire          embed_done_ready,
    output wire          embed_done_error,
    output wire [7:0]    embed_done_status,

    input  wire          vector_req_valid,
    output wire          vector_req_ready,
    input  wire [63:0]   vector_req_addr,
    input  wire [10:0]   vector_req_words,
    output wire          vector_rsp_valid,
    input  wire          vector_rsp_ready,
    output wire [127:0]  vector_rsp_data,
    output wire          vector_rsp_last,
    output wire          vector_rsp_error,

    input  wire          sink_req_valid,
    output wire          sink_req_ready,
    input  wire [63:0]   sink_req_addr,
    input  wire [6:0]    sink_req_words,
    output wire          sink_rsp_valid,
    input  wire          sink_rsp_ready,
    output wire [127:0]  sink_rsp_data,
    output wire          sink_rsp_last,
    output wire          sink_rsp_error,

    output wire          svc_cmd_valid,
    input  wire          svc_cmd_ready,
    output wire [63:0]   svc_cmd_base_addr,
    output wire [31:0]   svc_cmd_port_beats,
    output wire [3:0]    svc_cmd_port_mask,
    output wire          svc_abort_run,
    input  wire [511:0]  svc_data,
    input  wire          svc_valid,
    output wire          svc_ready,
    input  wire          svc_last,
    input  wire          svc_error,
    input  wire          svc_busy,
    input  wire          svc_done_valid,
    output wire          svc_done_ready,
    input  wire          svc_done_error,
    input  wire [7:0]    svc_done_status
);
    localparam [1:0] OWNER_NONE   = 2'd0;
    localparam [1:0] OWNER_EMBED  = 2'd1;
    localparam [1:0] OWNER_VECTOR = 2'd2;
    localparam [1:0] OWNER_SINK   = 2'd3;

    reg [1:0] owner_q;
    reg aborting_q;

    wire choose_embed = (owner_q == OWNER_NONE) && embed_cmd_valid;
    wire choose_vector = (owner_q == OWNER_NONE) && !embed_cmd_valid &&
                         vector_req_valid;
    wire choose_sink = (owner_q == OWNER_NONE) && !embed_cmd_valid &&
                       !vector_req_valid && sink_req_valid;
    wire command_fire = svc_cmd_valid && svc_cmd_ready;
    wire owned_abort = abort_run || clear ||
                       ((owner_q == OWNER_EMBED) && embed_abort);

    assign svc_cmd_valid = !clear && !abort_run && !aborting_q &&
                           (choose_embed || choose_vector || choose_sink);
    assign svc_cmd_base_addr = choose_embed ? embed_cmd_base_addr :
                               choose_vector ? vector_req_addr : sink_req_addr;
    assign svc_cmd_port_beats = choose_embed ? embed_cmd_port_beats :
                                choose_vector ? {21'd0, vector_req_words} :
                                                {25'd0, sink_req_words};
    assign svc_cmd_port_mask = choose_embed ? embed_cmd_port_mask : 4'h1;
    assign svc_abort_run = owned_abort || aborting_q;

    assign embed_cmd_ready = choose_embed && svc_cmd_ready && !aborting_q;
    assign vector_req_ready = choose_vector && svc_cmd_ready && !aborting_q;
    assign sink_req_ready = choose_sink && svc_cmd_ready && !aborting_q;

    assign embed_data = svc_data;
    assign embed_valid = (owner_q == OWNER_EMBED) && svc_valid && !aborting_q;
    assign embed_last = svc_last;
    assign embed_error = svc_error;
    assign embed_busy = (owner_q == OWNER_EMBED) && svc_busy;
    assign embed_done_valid = (owner_q == OWNER_EMBED) && svc_done_valid &&
                              !aborting_q;
    assign embed_done_error = svc_done_error;
    assign embed_done_status = svc_done_status;

    assign vector_rsp_valid = (owner_q == OWNER_VECTOR) && svc_valid &&
                              !aborting_q;
    assign vector_rsp_data = svc_data[127:0];
    assign vector_rsp_last = svc_last;
    assign vector_rsp_error = svc_error;
    assign sink_rsp_valid = (owner_q == OWNER_SINK) && svc_valid &&
                            !aborting_q;
    assign sink_rsp_data = svc_data[127:0];
    assign sink_rsp_last = svc_last;
    assign sink_rsp_error = svc_error;

    assign svc_ready = aborting_q ? 1'b1 :
                       (owner_q == OWNER_EMBED) ? embed_ready :
                       (owner_q == OWNER_VECTOR) ? vector_rsp_ready :
                       (owner_q == OWNER_SINK) ? sink_rsp_ready : 1'b0;
    assign svc_done_ready = aborting_q ? 1'b1 :
                            (owner_q == OWNER_EMBED) ? embed_done_ready :
                            ((owner_q == OWNER_VECTOR) ||
                             (owner_q == OWNER_SINK));

    always @(posedge clk) begin
        if (!rst_n || clear) begin
            owner_q <= OWNER_NONE;
            aborting_q <= 1'b0;
        end else begin
            if (command_fire)
                owner_q <= choose_embed ? OWNER_EMBED :
                           choose_vector ? OWNER_VECTOR : OWNER_SINK;

            if (owned_abort && (owner_q != OWNER_NONE))
                aborting_q <= 1'b1;

            if (!aborting_q && svc_done_valid && svc_done_ready)
                owner_q <= OWNER_NONE;

            if (aborting_q && !svc_busy && svc_cmd_ready) begin
                owner_q <= OWNER_NONE;
                aborting_q <= 1'b0;
            end

            if ((owner_q == OWNER_NONE) && (clear || abort_run))
                aborting_q <= 1'b0;
        end
    end

`ifndef SYNTHESIS
    always @(posedge clk) begin
        if (rst_n && (owner_q == OWNER_NONE) && svc_valid)
            $fatal(1, " small_read_mux response without owner");
        if (rst_n && command_fire && choose_vector)
            assert(vector_req_words != 0);
        if (rst_n && command_fire && choose_sink)
            assert(sink_req_words != 0);
    end
`endif
endmodule

`default_nettype wire
