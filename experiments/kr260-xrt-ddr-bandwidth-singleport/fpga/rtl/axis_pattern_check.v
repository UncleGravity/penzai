module axis_pattern_check (
    input  wire         aclk,
    input  wire         aresetn,

    input  wire         start,
    input  wire [63:0]  length_bytes,
    input  wire [63:0]  base_index,
    input  wire [7:0]   seed,

    output reg          busy,
    output reg          done,
    output reg          error_seen,
    output reg  [63:0]  first_error_index,
    output reg  [7:0]   expected,
    output reg  [7:0]   actual,
    output reg  [63:0]  bytes_checked,
    output reg  [63:0]  cycles,

    input  wire [127:0] s_axis_tdata,
    input  wire [15:0]  s_axis_tkeep,
    input  wire         s_axis_tvalid,
    output wire         s_axis_tready,
    input  wire         s_axis_tlast
);
    localparam integer DATA_BYTES = 16;
    localparam [63:0] DATA_BYTES_U64 = 64'd16;

    reg [63:0] byte_index;
    reg [63:0] remaining_bytes;

    reg        cmp_error;
    reg [63:0] cmp_index;
    reg [7:0]  cmp_expected;
    reg [7:0]  cmp_actual;
    reg [63:0] beat_bytes;
    reg [63:0] lane_u64;
    reg [7:0]  lane_u8;
    reg        expected_tlast;
    integer lane;

    function [7:0] pattern_byte;
        input [7:0] index;
        input [7:0] seed_value;
        begin
            pattern_byte = (index * 8'd7) + seed_value;
        end
    endfunction

    assign s_axis_tready = busy;

    always @* begin
        cmp_error = 1'b0;
        cmp_index = 64'd0;
        cmp_expected = 8'd0;
        cmp_actual = 8'd0;
        beat_bytes = (remaining_bytes < DATA_BYTES_U64) ? remaining_bytes : DATA_BYTES_U64;
        expected_tlast = (remaining_bytes <= DATA_BYTES_U64);
        lane_u64 = 64'd0;
        lane_u8 = 8'd0;

        for (lane = 0; lane < DATA_BYTES; lane = lane + 1) begin
            lane_u64 = {32'd0, lane};
            lane_u8 = lane[7:0];
            if (!cmp_error && (lane_u64 < beat_bytes)) begin
                if (s_axis_tkeep[lane] != 1'b1) begin
                    cmp_error = 1'b1;
                    cmp_index = byte_index + lane_u64;
                    cmp_expected = pattern_byte(byte_index[7:0] + lane_u8, seed);
                    cmp_actual = 8'h00;
                end else if (s_axis_tdata[(lane * 8) +: 8] != pattern_byte(byte_index[7:0] + lane_u8, seed)) begin
                    cmp_error = 1'b1;
                    cmp_index = byte_index + lane_u64;
                    cmp_expected = pattern_byte(byte_index[7:0] + lane_u8, seed);
                    cmp_actual = s_axis_tdata[(lane * 8) +: 8];
                end
            end
        end

        if (!cmp_error && (s_axis_tlast != expected_tlast)) begin
            cmp_error = 1'b1;
            cmp_index = byte_index + beat_bytes;
            cmp_expected = {7'd0, expected_tlast};
            cmp_actual = {7'd0, s_axis_tlast};
        end
    end

    always @(posedge aclk) begin
        if (!aresetn) begin
            byte_index <= 64'd0;
            remaining_bytes <= 64'd0;
            busy <= 1'b0;
            done <= 1'b0;
            error_seen <= 1'b0;
            first_error_index <= 64'd0;
            expected <= 8'd0;
            actual <= 8'd0;
            bytes_checked <= 64'd0;
            cycles <= 64'd0;
        end else if (start) begin
            byte_index <= base_index;
            remaining_bytes <= length_bytes;
            busy <= (length_bytes != 64'd0);
            done <= (length_bytes == 64'd0);
            error_seen <= 1'b0;
            first_error_index <= 64'd0;
            expected <= 8'd0;
            actual <= 8'd0;
            bytes_checked <= 64'd0;
            cycles <= 64'd0;
        end else begin
            if (busy) begin
                cycles <= cycles + 64'd1;
            end

            if (busy && s_axis_tvalid && s_axis_tready) begin
                if (cmp_error && !error_seen) begin
                    error_seen <= 1'b1;
                    first_error_index <= cmp_index;
                    expected <= cmp_expected;
                    actual <= cmp_actual;
                end

                bytes_checked <= bytes_checked + beat_bytes;

                if (remaining_bytes <= DATA_BYTES_U64) begin
                    byte_index <= byte_index + beat_bytes;
                    remaining_bytes <= 64'd0;
                    busy <= 1'b0;
                    done <= 1'b1;
                end else begin
                    byte_index <= byte_index + DATA_BYTES_U64;
                    remaining_bytes <= remaining_bytes - DATA_BYTES_U64;
                end
            end
        end
    end
endmodule
