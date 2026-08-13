`default_nettype none

// Pairwise proof of the frozen P3 native-Q8 address contract. The production
// cosim separately drives every one of the 3072 RTL commit/read addresses.
module section_q8_buffer_map_formal;
    (* anyconst *) reg bank_a;
    (* anyconst *) reg bank_b;
    (* anyconst *) reg [1:0] token_a;
    (* anyconst *) reg [1:0] token_b;
    (* anyconst *) reg [8:0] block_a;
    (* anyconst *) reg [8:0] block_b;

    function automatic [11:0] address(
        input bank,
        input [1:0] token,
        input [8:0] block
    );
        address = (bank ? 12'd1536 : 12'd0) +
                  {1'b0, block, 2'b00} + {10'd0, token};
    endfunction

    wire [11:0] address_a = address(bank_a, token_a, block_a);
    wire [11:0] address_b = address(bank_b, token_b, block_b);

    always @* begin
        assume(block_a < 9'd384);
        assume(block_b < 9'd384);
        assert(address_a < 12'd3072);
        assert(address_b < 12'd3072);
        if (address_a == address_b) begin
            assert(bank_a == bank_b);
            assert(token_a == token_b);
            assert(block_a == block_b);
        end

        cover(bank_a && token_a == 2'd3 && block_a == 9'd383 &&
              address_a == 12'd3071);
    end
endmodule

`default_nettype wire
