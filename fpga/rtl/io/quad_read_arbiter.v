`default_nettype none

// Phase-locked arbitration for the single four-port packed-data reader.
// Projection has fixed priority over RoPE, which has priority over the small
// embedding/gamma client.  The accepted client owns every response channel
// until completion is acknowledged or an abort has fully drained the reader.
module quad_read_arbiter (
    input  wire          clk,
    input  wire          rst_n,
    input  wire          clear,

    input  wire          proj_cmd_valid,
    output wire          proj_cmd_ready,
    input  wire [63:0]   proj_cmd_base_addr,
    input  wire [31:0]   proj_cmd_port_beats,
    input  wire [3:0]    proj_cmd_port_mask,
    input  wire          proj_abort,
    output wire [511:0]  proj_data,
    output wire          proj_valid,
    input  wire          proj_ready,
    output wire          proj_last,
    output wire          proj_error,
    output wire          proj_busy,
    output wire          proj_done_valid,
    input  wire          proj_done_ready,
    output wire          proj_done_error,
    output wire [7:0]    proj_done_status,

    input  wire          rope_cmd_valid,
    output wire          rope_cmd_ready,
    input  wire [63:0]   rope_cmd_base_addr,
    input  wire [31:0]   rope_cmd_port_beats,
    input  wire [3:0]    rope_cmd_port_mask,
    input  wire          rope_abort,
    output wire [511:0]  rope_data,
    output wire          rope_valid,
    input  wire          rope_ready,
    output wire          rope_last,
    output wire          rope_error,
    output wire          rope_busy,
    output wire          rope_done_valid,
    input  wire          rope_done_ready,
    output wire          rope_done_error,
    output wire [7:0]    rope_done_status,

    input  wire          small_cmd_valid,
    output wire          small_cmd_ready,
    input  wire [63:0]   small_cmd_base_addr,
    input  wire [31:0]   small_cmd_port_beats,
    input  wire [3:0]    small_cmd_port_mask,
    input  wire          small_abort,
    output wire [511:0]  small_data,
    output wire          small_valid,
    input  wire          small_ready,
    output wire          small_last,
    output wire          small_error,
    output wire          small_busy,
    output wire          small_done_valid,
    input  wire          small_done_ready,
    output wire          small_done_error,
    output wire [7:0]    small_done_status,

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
    localparam [1:0] OWNER_NONE  = 2'd0;
    localparam [1:0] OWNER_PROJ  = 2'd1;
    localparam [1:0] OWNER_ROPE  = 2'd2;
    localparam [1:0] OWNER_SMALL = 2'd3;

    reg [1:0] owner_q;
    reg aborting_q;

    wire unowned = owner_q == OWNER_NONE;
    wire any_cmd_valid = proj_cmd_valid || rope_cmd_valid ||
                         small_cmd_valid;
    wire [1:0] grant_owner = proj_cmd_valid ? OWNER_PROJ :
                             rope_cmd_valid ? OWNER_ROPE : OWNER_SMALL;
    wire grant_abort = proj_cmd_valid ? proj_abort :
                       rope_cmd_valid ? rope_abort : small_abort;

    wire owner_is_proj = owner_q == OWNER_PROJ;
    wire owner_is_rope = owner_q == OWNER_ROPE;
    wire owner_is_small = owner_q == OWNER_SMALL;
    wire owner_abort = owner_is_proj ? proj_abort :
                       owner_is_rope ? rope_abort :
                       owner_is_small ? small_abort : 1'b0;
    wire abort_active = !unowned && (aborting_q || owner_abort);
    wire command_fire = svc_cmd_valid && svc_cmd_ready;
    wire completion_fire = svc_done_valid && svc_done_ready;
    wire abort_idle = abort_active && !svc_busy && !svc_done_valid;

    // With no presented command, every idle client sees ready.  This matters
    // for clients whose abort state waits for the shared service to return
    // ready while keeping cmd_valid low.  Once any valid is present, only the
    // fixed-priority winner sees ready and can observe a command handshake.
    wire idle_ready = rst_n && !clear && unowned && svc_cmd_ready;
    assign proj_cmd_ready = idle_ready &&
                            (!any_cmd_valid ||
                             ((grant_owner == OWNER_PROJ) && !grant_abort));
    assign rope_cmd_ready = idle_ready &&
                            (!any_cmd_valid ||
                             ((grant_owner == OWNER_ROPE) && !grant_abort));
    assign small_cmd_ready = idle_ready &&
                             (!any_cmd_valid ||
                              ((grant_owner == OWNER_SMALL) && !grant_abort));

    assign svc_cmd_valid = rst_n && !clear && unowned && any_cmd_valid &&
                           !grant_abort;
    assign svc_cmd_base_addr = grant_owner == OWNER_PROJ ?
                               proj_cmd_base_addr :
                               grant_owner == OWNER_ROPE ?
                               rope_cmd_base_addr : small_cmd_base_addr;
    assign svc_cmd_port_beats = grant_owner == OWNER_PROJ ?
                                proj_cmd_port_beats :
                                grant_owner == OWNER_ROPE ?
                                rope_cmd_port_beats :
                                small_cmd_port_beats;
    assign svc_cmd_port_mask = grant_owner == OWNER_PROJ ?
                               proj_cmd_port_mask :
                               grant_owner == OWNER_ROPE ?
                               rope_cmd_port_mask : small_cmd_port_mask;

    // Abort is sticky only while a client owns the service.  It is dropped
    // after the service reports idle, even if the client holds abort high;
    // that lets the client observe cmd_ready and leave its abort state.
    assign svc_abort_run = abort_active;
    assign svc_ready = !abort_active &&
                       (owner_is_proj ? proj_ready :
                        owner_is_rope ? rope_ready :
                        owner_is_small ? small_ready : 1'b0);
    assign svc_done_ready = !unowned &&
                            (abort_active ||
                             (owner_is_proj ? proj_done_ready :
                              owner_is_rope ? rope_done_ready :
                              small_done_ready));

    // Payload wires are broadcast; ready/valid qualification is the ownership
    // boundary.  Zeroing three inactive 512-bit buses would add 1536 LUTs
    // without changing ready/valid semantics.
    assign proj_data = svc_data;
    assign proj_valid = owner_is_proj && !abort_active && svc_valid;
    assign proj_last = svc_last;
    assign proj_error = svc_error;
    assign proj_busy = owner_is_proj && svc_busy;
    assign proj_done_valid = owner_is_proj && !abort_active && svc_done_valid;
    assign proj_done_error = svc_done_error;
    assign proj_done_status = svc_done_status;

    assign rope_data = svc_data;
    assign rope_valid = owner_is_rope && !abort_active && svc_valid;
    assign rope_last = svc_last;
    assign rope_error = svc_error;
    assign rope_busy = owner_is_rope && svc_busy;
    assign rope_done_valid = owner_is_rope && !abort_active && svc_done_valid;
    assign rope_done_error = svc_done_error;
    assign rope_done_status = svc_done_status;

    assign small_data = svc_data;
    assign small_valid = owner_is_small && !abort_active && svc_valid;
    assign small_last = svc_last;
    assign small_error = svc_error;
    assign small_busy = owner_is_small && svc_busy;
    assign small_done_valid = owner_is_small && !abort_active &&
                              svc_done_valid;
    assign small_done_error = svc_done_error;
    assign small_done_status = svc_done_status;

    always @(posedge clk) begin
        if (!rst_n) begin
            owner_q <= OWNER_NONE;
            aborting_q <= 1'b0;
        end else if (clear) begin
            owner_q <= OWNER_NONE;
            aborting_q <= 1'b0;
        end else if (unowned) begin
            aborting_q <= 1'b0;
            if (command_fire)
                owner_q <= grant_owner;
        end else begin
            if (owner_abort)
                aborting_q <= 1'b1;

            if (completion_fire || abort_idle) begin
                owner_q <= OWNER_NONE;
                aborting_q <= 1'b0;
            end
        end
    end

`ifdef FORMAL
    always @(posedge clk) begin
        if (rst_n && !clear && any_cmd_valid) begin
            assert((proj_cmd_ready + rope_cmd_ready + small_cmd_ready) <= 1);
        end
        if (rst_n && !clear && svc_valid)
            assert(owner_q != OWNER_NONE || !svc_ready);
        if (rst_n && abort_active) begin
            assert(!svc_ready);
            assert(!proj_valid && !rope_valid && !small_valid);
        end
    end
`endif
endmodule

`default_nettype wire
