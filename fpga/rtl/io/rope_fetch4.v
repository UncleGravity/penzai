`default_nettype none

// Fetch precomputed FP32 RoPE coefficients for a tile-8 request through the shared
// four-port reader.  Each position row is 512 bytes (64 {cos,sin} pairs).
// A 128-bit beat contains two adjacent pairs for one token; two output cycles
// transpose the four token records into four-lane coefficient order.
module rope_fetch4 (
    input  wire                  clk,
    input  wire                  rst_n,
    input  wire                  clear,
    input  wire                  abort_run,

    input  wire                  cmd_valid,
    output wire                  cmd_ready,
    input  wire [63:0]           cmd_table_addr,
    input  wire [16:0]           cmd_position_base,
    input  wire [3:0]            cmd_token_count,

    output wire [255:0]          coeff_data,
    output wire                  coeff_valid,
    input  wire                  coeff_ready,
    output wire                  coeff_last,
    output wire                  coeff_error,

    output wire                  busy,
    output wire                  done_valid,
    input  wire                  done_ready,
    output wire                  done_error,
    output wire [7:0]            done_status,

    // Shared quad-reader client port. The production top arbitrates this with
    // packed-weight and embedding reads; no AXI fabric is duplicated here.
    output wire                  read_cmd_valid,
    input  wire                  read_cmd_ready,
    output wire [63:0]           read_cmd_base_addr,
    output wire [31:0]           read_cmd_port_beats,
    output wire [3:0]            read_cmd_port_mask,
    output wire                  read_abort,
    input  wire [511:0]          read_data,
    input  wire                  read_valid,
    output wire                  read_ready,
    input  wire                  read_last,
    input  wire                  read_error,
    input  wire                  read_busy,
    input  wire                  read_done_valid,
    output wire                  read_done_ready,
    input  wire                  read_done_error,
    input  wire [7:0]            read_done_status
);
    localparam [2:0] ST_IDLE   = 3'd0;
    localparam [2:0] ST_START  = 3'd1;
    localparam [2:0] ST_STREAM = 3'd2;
    localparam [2:0] ST_WAIT   = 3'd3;
    localparam [2:0] ST_ABORT  = 3'd4;
    localparam [2:0] ST_DONE   = 3'd5;

    localparam [7:0] STATUS_OK          = 8'h00;
    localparam [7:0] STATUS_BAD_COMMAND = 8'h01;

    reg [2:0] state_q;
    reg [63:0] table_addr_q;
    reg [16:0] position_base_q;
    reg [3:0] token_count_q;
    reg wave_q;
    reg pair_half_q;
    reg done_error_q;
    reg [7:0] done_status_q;

    wire [17:0] position_end = {1'b0, cmd_position_base} +
                               {14'd0, cmd_token_count};
    wire command_ok = (cmd_table_addr != 64'd0) &&
                      (cmd_table_addr[5:0] == 6'd0) &&
                      (cmd_token_count >= 4'd1) &&
                      (cmd_token_count <= 4'd8) &&
                      (position_end <= 18'd65536);

    wire [3:0] wave_lane_count = !wave_q ?
        (token_count_q >= 4 ? 4'd4 : token_count_q) :
        (token_count_q - 4'd4);
    wire [3:0] wave_port_mask = wave_lane_count == 4'd1 ? 4'h1 :
                                wave_lane_count == 4'd2 ? 4'h3 :
                                wave_lane_count == 4'd3 ? 4'h7 : 4'hf;
    wire [17:0] wave_first_position =
        {1'b0, position_base_q} + (wave_q ? 18'd4 : 18'd0);
    wire [63:0] wave_base_addr = table_addr_q +
                                 ({46'd0, wave_first_position} << 9);

    wire [255:0] pair0_data = {
        read_data[3*128 +: 64], read_data[2*128 +: 64],
        read_data[1*128 +: 64], read_data[0*128 +: 64]
    };
    wire [255:0] pair1_data = {
        read_data[3*128 + 64 +: 64], read_data[2*128 + 64 +: 64],
        read_data[1*128 + 64 +: 64], read_data[0*128 + 64 +: 64]
    };
    wire final_wave = (token_count_q <= 4) || wave_q;

    assign cmd_ready = rst_n && !clear && !abort_run &&
                       (state_q == ST_IDLE);
    assign coeff_data = pair_half_q ? pair1_data : pair0_data;
    assign coeff_valid = (state_q == ST_STREAM) && read_valid;
    assign coeff_last = coeff_valid && pair_half_q && read_last && final_wave;
    assign coeff_error = read_error;
    assign read_cmd_valid = state_q == ST_START;
    assign read_cmd_base_addr = wave_base_addr;
    assign read_cmd_port_beats = 32'd32;
    assign read_cmd_port_mask = wave_port_mask;
    assign read_abort = abort_run || (state_q == ST_ABORT);
    assign read_ready = (state_q == ST_STREAM) && coeff_ready && pair_half_q;
    assign read_done_ready = state_q == ST_WAIT;
    assign busy = (state_q == ST_START) || (state_q == ST_STREAM) ||
                  (state_q == ST_WAIT) || (state_q == ST_ABORT);
    assign done_valid = state_q == ST_DONE;
    assign done_error = done_error_q;
    assign done_status = done_status_q;

    always @(posedge clk) begin
        if (!rst_n) begin
            state_q <= ST_IDLE;
            table_addr_q <= 64'd0;
            position_base_q <= 17'd0;
            token_count_q <= 4'd0;
            wave_q <= 1'b0;
            pair_half_q <= 1'b0;
            done_error_q <= 1'b0;
            done_status_q <= STATUS_OK;
        end else begin
            if (clear || abort_run) begin
                pair_half_q <= 1'b0;
                done_error_q <= 1'b0;
                done_status_q <= STATUS_OK;
                state_q <= read_busy ? ST_ABORT : ST_IDLE;
            end else begin
                case (state_q)
                    ST_IDLE: if (cmd_valid) begin
                        table_addr_q <= cmd_table_addr;
                        position_base_q <= cmd_position_base;
                        token_count_q <= cmd_token_count;
                        wave_q <= 1'b0;
                        pair_half_q <= 1'b0;
                        done_error_q <= !command_ok;
                        done_status_q <= command_ok ? STATUS_OK :
                                                      STATUS_BAD_COMMAND;
                        state_q <= command_ok ? ST_START : ST_DONE;
                    end

                    ST_START: if (read_cmd_valid && read_cmd_ready) begin
                        pair_half_q <= 1'b0;
                        state_q <= ST_STREAM;
                    end

                    ST_STREAM: if (coeff_valid && coeff_ready) begin
                        if (!pair_half_q) begin
                            pair_half_q <= 1'b1;
                        end else begin
                            pair_half_q <= 1'b0;
                            if (read_last)
                                state_q <= ST_WAIT;
                        end
                    end

                    ST_WAIT: if (read_done_valid) begin
                        if (read_done_error) begin
                            done_error_q <= 1'b1;
                            done_status_q <= read_done_status;
                            state_q <= ST_DONE;
                        end else if (!final_wave) begin
                            wave_q <= 1'b1;
                            state_q <= ST_START;
                        end else begin
                            state_q <= ST_DONE;
                        end
                    end

                    ST_ABORT: if (!read_busy && read_cmd_ready)
                        state_q <= ST_IDLE;

                    ST_DONE: if (done_ready)
                        state_q <= ST_IDLE;

                    default: state_q <= ST_IDLE;
                endcase
            end
        end
    end

`ifdef FORMAL
    always @(posedge clk) begin
        if (rst_n && coeff_valid && !coeff_ready) begin
            assert($stable(coeff_data));
            assert($stable(coeff_last));
            assert($stable(coeff_error));
        end
    end
`endif
endmodule

`default_nettype wire
