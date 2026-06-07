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
    localparam [31:0] DATA_BYTES_U32 = 32'd16;

    reg [63:0] byte_index;
    reg [31:0] remaining_beats;
    reg [127:0] expected_data;

    reg        data_error;
    reg [3:0]  first_error_lane;
    reg [7:0]  first_error_expected;
    reg [7:0]  first_error_actual;
    integer lane;
    integer update_lane;

    function [7:0] pattern_byte;
        input [7:0] index;
        input [7:0] seed_value;
        begin
            pattern_byte = (index * 8'd7) + seed_value;
        end
    endfunction

    assign s_axis_tready = busy;

    always @* begin
        data_error = 1'b0;
        first_error_lane = 4'd0;
        first_error_expected = 8'd0;
        first_error_actual = 8'd0;

        for (lane = 0; lane < DATA_BYTES; lane = lane + 1) begin
            if (!data_error &&
                (s_axis_tdata[(lane * 8) +: 8] != expected_data[(lane * 8) +: 8])) begin
                data_error = 1'b1;
                first_error_lane = lane[3:0];
                first_error_expected = expected_data[(lane * 8) +: 8];
                first_error_actual = s_axis_tdata[(lane * 8) +: 8];
            end
        end
    end

    always @(posedge aclk) begin
        if (!aresetn) begin
            byte_index <= 64'd0;
            remaining_beats <= 32'd0;
            expected_data <= 128'd0;
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
            remaining_beats <= length_bytes[35:4];
            for (update_lane = 0; update_lane < DATA_BYTES; update_lane = update_lane + 1) begin
                expected_data[(update_lane * 8) +: 8] <= pattern_byte(base_index[7:0] + update_lane[7:0], seed);
            end
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
                if (!error_seen) begin
                    if (data_error) begin
                        error_seen <= 1'b1;
                        first_error_index <= {byte_index[63:4], first_error_lane};
                        expected <= first_error_expected;
                        actual <= first_error_actual;
                    end else if (s_axis_tlast != (remaining_beats == 32'd1)) begin
                        error_seen <= 1'b1;
                        first_error_index <= {byte_index[63:4] + 60'd1, 4'd0};
                        expected <= {7'd0, (remaining_beats == 32'd1)};
                        actual <= {7'd0, s_axis_tlast};
                    end
                end

                bytes_checked <= bytes_checked + {32'd0, DATA_BYTES_U32};
                for (update_lane = 0; update_lane < DATA_BYTES; update_lane = update_lane + 1) begin
                    expected_data[(update_lane * 8) +: 8] <= expected_data[(update_lane * 8) +: 8] + 8'd112;
                end

                if (remaining_beats <= 32'd1) begin
                    byte_index <= {byte_index[63:4] + 60'd1, byte_index[3:0]};
                    remaining_beats <= 32'd0;
                    busy <= 1'b0;
                    done <= 1'b1;
                end else begin
                    byte_index <= {byte_index[63:4] + 60'd1, byte_index[3:0]};
                    remaining_beats <= remaining_beats - 32'd1;
                end
            end
        end
    end

    wire unused_tkeep = |s_axis_tkeep;
endmodule
