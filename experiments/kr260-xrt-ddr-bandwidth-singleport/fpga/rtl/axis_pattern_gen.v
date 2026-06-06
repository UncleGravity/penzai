module axis_pattern_gen (
    input  wire         aclk,
    input  wire         aresetn,

    input  wire         start,
    input  wire [63:0]  length_bytes,
    input  wire [63:0]  base_index,
    input  wire [7:0]   seed,

    output reg          busy,
    output reg          done,
    output reg  [63:0]  cycles,

    output wire [127:0] m_axis_tdata,
    output wire [15:0]  m_axis_tkeep,
    output wire         m_axis_tvalid,
    input  wire         m_axis_tready,
    output wire         m_axis_tlast
);
    localparam integer DATA_BYTES = 16;
    localparam [63:0] DATA_BYTES_U64 = 64'd16;

    reg [63:0] byte_index;
    reg [63:0] remaining_bytes;
    reg        valid;

    function [7:0] pattern_byte;
        input [7:0] index;
        input [7:0] seed_value;
        begin
            pattern_byte = (index * 8'd7) + seed_value;
        end
    endfunction

    genvar lane;
    generate
        for (lane = 0; lane < DATA_BYTES; lane = lane + 1) begin : gen_lanes
            localparam [7:0] LANE_INDEX = lane[7:0];
            assign m_axis_tdata[(lane * 8) +: 8] = pattern_byte(byte_index[7:0] + LANE_INDEX, seed);
        end
    endgenerate

    assign m_axis_tkeep = 16'hffff;
    assign m_axis_tvalid = valid;
    assign m_axis_tlast = valid && (remaining_bytes <= DATA_BYTES_U64);

    always @(posedge aclk) begin
        if (!aresetn) begin
            byte_index <= 64'd0;
            remaining_bytes <= 64'd0;
            valid <= 1'b0;
            busy <= 1'b0;
            done <= 1'b0;
            cycles <= 64'd0;
        end else if (start) begin
            byte_index <= base_index;
            remaining_bytes <= length_bytes;
            cycles <= 64'd0;
            done <= (length_bytes == 64'd0);
            busy <= (length_bytes != 64'd0);
            valid <= (length_bytes != 64'd0);
        end else begin
            if (busy) begin
                cycles <= cycles + 64'd1;
            end

            if (valid && m_axis_tready) begin
                if (remaining_bytes <= DATA_BYTES_U64) begin
                    byte_index <= byte_index + remaining_bytes;
                    remaining_bytes <= 64'd0;
                    valid <= 1'b0;
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
