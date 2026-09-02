`default_nettype none

// Tile-8 storage with a four-lane hot interface. Wave 0 addresses tokens 0..3 and wave 1
// addresses tokens 4..7. Folding the wave into each physical bank keeps the
// service width at four lanes without reducing logical tile capacity.
module resident_arenas (
    input  wire          clk,
    input  wire          rst_n,
    input  wire          clear,

    // Resident FP32 residual R: four 4096 x 64-bit URAM banks.
    input  wire          r_wr_valid,
    output wire          r_wr_ready,
    input  wire          r_wr_wave,
    input  wire [11:0]   r_wr_addr,
    input  wire [3:0]    r_wr_lane_mask,
    input  wire [127:0]  r_wr_data,
    input  wire          r_rd_req_valid,
    output wire          r_rd_req_ready,
    input  wire          r_rd_req_wave,
    input  wire [11:0]   r_rd_req_addr,
    output wire          r_rd_rsp_valid,
    input  wire          r_rd_rsp_ready,
    output wire [127:0]  r_rd_rsp_data,

    // Resident normalized/rotated Q. It cannot alias R before O projection.
    input  wire          query_wr_valid,
    output wire          query_wr_ready,
    input  wire          query_wr_wave,
    input  wire [11:0]   query_wr_addr,
    input  wire [3:0]    query_wr_lane_mask,
    input  wire [127:0]  query_wr_data,
    input  wire          query_rd_req_valid,
    output wire          query_rd_req_ready,
    input  wire          query_rd_req_wave,
    input  wire [11:0]   query_rd_req_addr,
    output wire          query_rd_rsp_valid,
    input  wire          query_rd_rsp_ready,
    output wire [127:0]  query_rd_rsp_data,

    // Canonical Q8 record per lane: {32 int8 values, fp16 scale}.
    // Physical address {wave, block} spans tile-8 x 512 FFN blocks.
    input  wire          q8_wr_valid,
    output wire          q8_wr_ready,
    input  wire          q8_wr_wave,
    input  wire [8:0]    q8_wr_addr,
    input  wire [3:0]    q8_wr_lane_mask,
    input  wire [1087:0] q8_wr_data,
    input  wire          q8_rd_req_valid,
    output wire          q8_rd_req_ready,
    input  wire          q8_rd_req_wave,
    input  wire [8:0]    q8_rd_req_addr,
    output wire          q8_rd_rsp_valid,
    input  wire          q8_rd_rsp_ready,
    output wire [1087:0] q8_rd_rsp_data,

    // New K/V staging. Logical address = {kind, kv_head[2:0], dim[6:0]}.
    // Physical address {wave, logical address} spans every tile-8 token.
    input  wire          newkv_wr_valid,
    output wire          newkv_wr_ready,
    input  wire          newkv_wr_wave,
    input  wire [10:0]   newkv_wr_addr,
    input  wire [3:0]    newkv_wr_lane_mask,
    input  wire [63:0]   newkv_wr_data,
    input  wire          newkv_rd_req_valid,
    output wire          newkv_rd_req_ready,
    input  wire          newkv_rd_req_wave,
    input  wire [10:0]   newkv_rd_req_addr,
    output wire          newkv_rd_rsp_valid,
    input  wire          newkv_rd_rsp_ready,
    output wire [63:0]   newkv_rd_rsp_data
);
    reg r_wr_valid_q;
    reg r_wr_wave_q;
    reg [11:0] r_wr_addr_q;
    reg [3:0] r_wr_lane_mask_q;
    reg [127:0] r_wr_data_q;
    reg query_wr_valid_q;
    reg query_wr_wave_q;
    reg [11:0] query_wr_addr_q;
    reg [3:0] query_wr_lane_mask_q;
    reg [127:0] query_wr_data_q;
    reg q8_wr_valid_q;
    reg q8_wr_wave_q;
    reg [8:0] q8_wr_addr_q;
    reg [3:0] q8_wr_lane_mask_q;
    reg [1087:0] q8_wr_data_q;
    reg newkv_wr_valid_q;
    reg newkv_wr_wave_q;
    reg [10:0] newkv_wr_addr_q;
    reg [3:0] newkv_wr_lane_mask_q;
    reg [63:0] newkv_wr_data_q;

    reg r_rd_mem_valid_q;
    reg r_rd_mem_wave_q;
    reg r_rd_rsp_valid_q;
    reg [127:0] r_rd_rsp_data_q;
    reg query_rd_mem_valid_q;
    reg query_rd_mem_wave_q;
    reg q8_rd_mem_valid_q;
    reg q8_rd_rsp_valid_q;
    reg [1087:0] q8_rd_rsp_data_q;
    reg newkv_rd_mem_valid_q;

    wire [255:0] r_bank_rd_data;
    wire [127:0] r_selected_bank_data;
    wire [255:0] query_bank_rd_data;
    wire [1087:0] q8_bank_rd_data;
    wire [63:0] newkv_bank_rd_data;

    // The intent stages can retire on the same edge that a following phase
    // first presents a read. Stall only that same-address collision for one
    // cycle so RAM inference stays canonical and read-during-write behavior is
    // deterministic. Unrelated and streaming accesses remain full-rate.
    wire r_rw_conflict = r_wr_valid_q && (|r_wr_lane_mask_q) &&
        (r_wr_addr_q == r_rd_req_addr) && (r_wr_wave_q == r_rd_req_wave);
    wire query_rw_conflict = query_wr_valid_q && (|query_wr_lane_mask_q) &&
        (query_wr_addr_q == query_rd_req_addr) &&
        (query_wr_wave_q == query_rd_req_wave);
    wire q8_rw_conflict = q8_wr_valid_q && (|q8_wr_lane_mask_q) &&
        ({q8_wr_wave_q, q8_wr_addr_q} ==
         {q8_rd_req_wave, q8_rd_req_addr});
    wire newkv_rw_conflict = newkv_wr_valid_q && (|newkv_wr_lane_mask_q) &&
        ({newkv_wr_wave_q, newkv_wr_addr_q} ==
         {newkv_rd_req_wave, newkv_rd_req_addr});

    wire r_rsp_slot_available = !r_rd_rsp_valid_q || r_rd_rsp_ready;
    wire r_mem_to_rsp = r_rd_mem_valid_q && r_rsp_slot_available;
    wire r_mem_slot_available = !r_rd_mem_valid_q || r_mem_to_rsp;
    wire r_req_fire = r_rd_req_valid && r_rd_req_ready;
    wire r_mem_read_issue = r_rd_req_valid && r_mem_slot_available &&
                            !r_rw_conflict;

    wire query_mem_slot_available = !query_rd_mem_valid_q ||
                                    query_rd_rsp_ready;
    wire query_req_fire = query_rd_req_valid && query_rd_req_ready;
    wire query_mem_read_issue = query_rd_req_valid &&
                                query_mem_slot_available &&
                                !query_rw_conflict;

    wire q8_rsp_slot_available = !q8_rd_rsp_valid_q || q8_rd_rsp_ready;
    wire q8_mem_to_rsp = q8_rd_mem_valid_q && q8_rsp_slot_available;
    wire q8_mem_slot_available = !q8_rd_mem_valid_q || q8_mem_to_rsp;
    wire q8_req_fire = q8_rd_req_valid && q8_rd_req_ready;
    wire q8_mem_read_issue = q8_rd_req_valid && q8_mem_slot_available &&
                             !q8_rw_conflict;

    wire newkv_mem_slot_available = !newkv_rd_mem_valid_q ||
                                    newkv_rd_rsp_ready;
    wire newkv_req_fire = newkv_rd_req_valid && newkv_rd_req_ready;
    wire newkv_mem_read_issue = newkv_rd_req_valid &&
                                newkv_mem_slot_available &&
                                !newkv_rw_conflict;

    // Write handshakes enter local intent registers. Physical RAM inputs are
    // consequently driven only by nearby registers, not service state/clear
    // cones. The stages drain every cycle and preserve full write throughput.
    assign r_wr_ready = rst_n && !clear;
    assign query_wr_ready = rst_n && !clear;
    assign q8_wr_ready = rst_n && !clear;
    assign newkv_wr_ready = rst_n && !clear;

    // Timing-critical R/Q8 reads have an elastic memory-output slot and held
    // response. Query/NewKV consumers already register their inputs, so those
    // ports expose the synchronous RAM output directly and avoid a repeated
    // cycle in the single-outstanding KV append path. Every port remains II=1.
    // Physical read enables deliberately ignore clear: an untracked
    // clear-cycle read is side-effect free, while valid state is canceled
    // synchronously and no stale response can escape.
    assign r_rd_req_ready = rst_n && !clear && r_mem_slot_available &&
                            !r_rw_conflict;
    assign r_rd_rsp_valid = r_rd_rsp_valid_q;
    assign r_rd_rsp_data = r_rd_rsp_data_q;
    assign query_rd_req_ready = rst_n && !clear &&
                                query_mem_slot_available &&
                                !query_rw_conflict;
    assign query_rd_rsp_valid = query_rd_mem_valid_q;
    assign query_rd_rsp_data = {
        query_rd_mem_wave_q ? query_bank_rd_data[255:224] :
                              query_bank_rd_data[223:192],
        query_rd_mem_wave_q ? query_bank_rd_data[191:160] :
                              query_bank_rd_data[159:128],
        query_rd_mem_wave_q ? query_bank_rd_data[127:96] :
                              query_bank_rd_data[95:64],
        query_rd_mem_wave_q ? query_bank_rd_data[63:32] :
                              query_bank_rd_data[31:0]
    };
    assign q8_rd_req_ready = rst_n && !clear && q8_mem_slot_available &&
                             !q8_rw_conflict;
    assign q8_rd_rsp_valid = q8_rd_rsp_valid_q;
    assign q8_rd_rsp_data = q8_rd_rsp_data_q;
    assign newkv_rd_req_ready = rst_n && !clear &&
                                newkv_mem_slot_available &&
                                !newkv_rw_conflict;
    assign newkv_rd_rsp_valid = newkv_rd_mem_valid_q;
    assign newkv_rd_rsp_data = newkv_bank_rd_data;

    genvar lane;
    generate
        for (lane = 0; lane < 4; lane = lane + 1) begin : g_response_lane
            assign r_selected_bank_data[lane*32 +: 32] =
                r_rd_mem_wave_q ? r_bank_rd_data[lane*64 + 32 +: 32] :
                                  r_bank_rd_data[lane*64 +: 32];
        end
    endgenerate

    always @(posedge clk) begin
        if (!rst_n || clear) begin
            r_wr_valid_q <= 1'b0;
            query_wr_valid_q <= 1'b0;
            q8_wr_valid_q <= 1'b0;
            newkv_wr_valid_q <= 1'b0;
            r_rd_mem_valid_q <= 1'b0;
            r_rd_rsp_valid_q <= 1'b0;
            query_rd_mem_valid_q <= 1'b0;
            q8_rd_mem_valid_q <= 1'b0;
            q8_rd_rsp_valid_q <= 1'b0;
            newkv_rd_mem_valid_q <= 1'b0;
        end else begin
            r_wr_valid_q <= r_wr_valid;
            if (r_wr_valid) begin
                r_wr_wave_q <= r_wr_wave;
                r_wr_addr_q <= r_wr_addr;
                r_wr_lane_mask_q <= r_wr_lane_mask;
                r_wr_data_q <= r_wr_data;
            end
            query_wr_valid_q <= query_wr_valid;
            if (query_wr_valid) begin
                query_wr_wave_q <= query_wr_wave;
                query_wr_addr_q <= query_wr_addr;
                query_wr_lane_mask_q <= query_wr_lane_mask;
                query_wr_data_q <= query_wr_data;
            end
            q8_wr_valid_q <= q8_wr_valid;
            if (q8_wr_valid) begin
                q8_wr_wave_q <= q8_wr_wave;
                q8_wr_addr_q <= q8_wr_addr;
                q8_wr_lane_mask_q <= q8_wr_lane_mask;
                q8_wr_data_q <= q8_wr_data;
            end
            newkv_wr_valid_q <= newkv_wr_valid;
            if (newkv_wr_valid) begin
                newkv_wr_wave_q <= newkv_wr_wave;
                newkv_wr_addr_q <= newkv_wr_addr;
                newkv_wr_lane_mask_q <= newkv_wr_lane_mask;
                newkv_wr_data_q <= newkv_wr_data;
            end

            if (r_mem_to_rsp) begin
                r_rd_rsp_valid_q <= 1'b1;
                r_rd_rsp_data_q <= r_selected_bank_data;
            end else if (r_rd_rsp_valid_q && r_rd_rsp_ready) begin
                r_rd_rsp_valid_q <= 1'b0;
            end
            if (r_req_fire) begin
                r_rd_mem_valid_q <= 1'b1;
                r_rd_mem_wave_q <= r_rd_req_wave;
            end else if (r_mem_to_rsp) begin
                r_rd_mem_valid_q <= 1'b0;
            end

            if (query_req_fire) begin
                query_rd_mem_valid_q <= 1'b1;
                query_rd_mem_wave_q <= query_rd_req_wave;
            end else if (query_rd_mem_valid_q && query_rd_rsp_ready) begin
                query_rd_mem_valid_q <= 1'b0;
            end

            if (q8_mem_to_rsp) begin
                q8_rd_rsp_valid_q <= 1'b1;
                q8_rd_rsp_data_q <= q8_bank_rd_data;
            end else if (q8_rd_rsp_valid_q && q8_rd_rsp_ready) begin
                q8_rd_rsp_valid_q <= 1'b0;
            end
            if (q8_req_fire) begin
                q8_rd_mem_valid_q <= 1'b1;
            end else if (q8_mem_to_rsp) begin
                q8_rd_mem_valid_q <= 1'b0;
            end

            if (newkv_req_fire) begin
                newkv_rd_mem_valid_q <= 1'b1;
            end else if (newkv_rd_mem_valid_q && newkv_rd_rsp_ready) begin
                newkv_rd_mem_valid_q <= 1'b0;
            end
        end
    end

    generate
        for (lane = 0; lane < 4; lane = lane + 1) begin : g_lane
            f32_wave_bank u_r (
                .clk(clk),
                .wr_en(r_wr_valid_q && r_wr_lane_mask_q[lane]),
                .wr_wave(r_wr_wave_q), .wr_addr(r_wr_addr_q),
                .wr_data(r_wr_data_q[lane*32 +: 32]),
                .rd_en(r_mem_read_issue), .rd_addr(r_rd_req_addr),
                .rd_data(r_bank_rd_data[lane*64 +: 64])
            );
            f32_wave_bank u_query (
                .clk(clk),
                .wr_en(query_wr_valid_q && query_wr_lane_mask_q[lane]),
                .wr_wave(query_wr_wave_q), .wr_addr(query_wr_addr_q),
                .wr_data(query_wr_data_q[lane*32 +: 32]),
                .rd_en(query_mem_read_issue), .rd_addr(query_rd_req_addr),
                .rd_data(query_bank_rd_data[lane*64 +: 64])
            );
            q8_wave_bank u_q8 (
                .clk(clk),
                .wr_en(q8_wr_valid_q && q8_wr_lane_mask_q[lane]),
                .wr_addr({q8_wr_wave_q, q8_wr_addr_q}),
                .wr_data(q8_wr_data_q[lane*272 +: 272]),
                .rd_en(q8_mem_read_issue),
                .rd_addr({q8_rd_req_wave, q8_rd_req_addr}),
                .rd_data(q8_bank_rd_data[lane*272 +: 272])
            );
            newkv_wave_bank u_newkv (
                .clk(clk),
                .wr_en(newkv_wr_valid_q && newkv_wr_lane_mask_q[lane]),
                .wr_addr({newkv_wr_wave_q, newkv_wr_addr_q}),
                .wr_data(newkv_wr_data_q[lane*16 +: 16]),
                .rd_en(newkv_mem_read_issue),
                .rd_addr({newkv_rd_req_wave, newkv_rd_req_addr}),
                .rd_data(newkv_bank_rd_data[lane*16 +: 16])
            );
        end
    endgenerate
endmodule

module f32_wave_bank (
    input  wire        clk,
    input  wire        wr_en,
    input  wire        wr_wave,
    input  wire [11:0] wr_addr,
    input  wire [31:0] wr_data,
    input  wire        rd_en,
    input  wire [11:0] rd_addr,
    output reg  [63:0] rd_data
);
    (* ram_style = "ultra" *) reg [63:0] mem [0:4095];
    always @(posedge clk) begin
        if (wr_en && !wr_wave)
            mem[wr_addr][31:0] <= wr_data;
        if (wr_en && wr_wave)
            mem[wr_addr][63:32] <= wr_data;
        if (rd_en)
            rd_data <= mem[rd_addr];
    end
endmodule

module q8_wave_bank (
    input  wire         clk,
    input  wire         wr_en,
    input  wire [9:0]   wr_addr,
    input  wire [271:0] wr_data,
    input  wire         rd_en,
    input  wire [9:0]   rd_addr,
    output reg  [271:0] rd_data
);
    (* ram_style = "block" *) reg [271:0] mem [0:1023];
    always @(posedge clk) begin
        if (wr_en)
            mem[wr_addr] <= wr_data;
        if (rd_en)
            rd_data <= mem[rd_addr];
    end
endmodule

module newkv_wave_bank (
    input  wire        clk,
    input  wire        wr_en,
    input  wire [11:0] wr_addr,
    input  wire [15:0] wr_data,
    input  wire        rd_en,
    input  wire [11:0] rd_addr,
    output reg  [15:0] rd_data
);
    (* ram_style = "block" *) reg [15:0] mem [0:4095];
    always @(posedge clk) begin
        if (wr_en)
            mem[wr_addr] <= wr_data;
        if (rd_en)
            rd_data <= mem[rd_addr];
    end
endmodule

`default_nettype wire
