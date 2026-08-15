`default_nettype none

// Control-accurate abstraction of the independently proven exact multiplier.
// The noncommutative data function lets the wrapper prove operand order without
// importing the full IEEE normalization cone into this composition proof.
module section_rmsnorm_mul_rne (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        abort_run,
    output wire        busy,
    input  wire        s_valid,
    output wire        s_ready,
    input  wire [31:0] s_a,
    input  wire [31:0] s_b,
    output wire        result_valid,
    input  wire        result_ready,
    output wire [31:0] result_data,
    output wire [1:0]  result_status
);
    reg        busy_q;
    reg [2:0]  age_q;
    reg        valid_q;
    reg [31:0] data_q;
    reg [1:0]  status_q;

    function automatic [31:0] model_data(
        input [31:0] a,
        input [31:0] b
    );
        model_data = a ^ {b[15:0], b[31:16]} ^ 32'h6d2b_79f5;
    endfunction

    function automatic [1:0] model_status(
        input [31:0] a,
        input [31:0] b
    );
        begin
            if (a == 32'h7f80_0000 || b == 32'h7f80_0000)
                model_status = 2'b01;
            else if (a == 32'h7f7f_ffff || b == 32'h7f7f_ffff)
                model_status = 2'b10;
            else
                model_status = 2'b00;
        end
    endfunction

    assign busy = busy_q;
    assign s_ready = rst_n && !abort_run && !busy_q;
    assign result_valid = valid_q;
    assign result_data = data_q;
    assign result_status = status_q;

    always @(posedge clk) begin
        if (!rst_n || abort_run) begin
            busy_q <= 1'b0;
            age_q <= 3'd0;
            valid_q <= 1'b0;
            data_q <= 32'd0;
            status_q <= 2'd0;
        end else begin
            if (s_valid && s_ready) begin
                busy_q <= 1'b1;
                age_q <= 3'd5;
                valid_q <= 1'b0;
                data_q <= model_data(s_a, s_b);
                status_q <= model_status(s_a, s_b);
            end else if (busy_q && !valid_q) begin
                if (age_q == 3'd1) begin
                    age_q <= 3'd0;
                    valid_q <= 1'b1;
                end else begin
                    age_q <= age_q - 1'b1;
                end
            end else if (valid_q && result_ready) begin
                busy_q <= 1'b0;
                valid_q <= 1'b0;
                data_q <= 32'd0;
                status_q <= 2'd0;
            end
        end
    end
endmodule

`default_nettype wire
