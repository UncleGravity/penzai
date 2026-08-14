// section_q8_buffer - ping-ponged native-Q8 record storage for streaming FFN.
//
// Each bank holds four tokens x 384 Q8 blocks. A canonical record is captured
// as four little-endian quant-byte beats followed by one scale beat whose f16
// value occupies bits [15:0]. The producer tags every beat with
// {bank, token, block}; tags and TLAST must describe exactly one five-beat
// record. Complete records map as:
//
//   local_address = block * 4 + token
//   address       = bank * 1536 + local_address
//
// This block-major layout accepts paired-projection production directly. The
// request port can later replay arbitrary records, including token-major order,
// as the same five-beat stream.
//
// Configuration clears only the selected bank's duplicate bitmap. Payload and
// scale RAMs are never reset. A bank becomes readable only after an explicit,
// clean seal observes every configured tag exactly once. Abort invalidates the
// selected bank, discards its in-flight capture/replay, and sets sticky error.

`default_nettype none

module section_q8_buffer (
    input  wire          clk,
    input  wire          rst_n,

    // Configure one ping-pong bank. A valid shape starts a 1536-cycle metadata
    // clear; bank_active rises only after that clear completes.
    input  wire          cfg_valid,
    output wire          cfg_ready,
    input  wire          cfg_bank,
    input  wire [2:0]    cfg_tokens,
    input  wire [8:0]    cfg_blocks,

    // Seal exactly one configured bank after all unique records have arrived.
    input  wire          seal_valid,
    output wire          seal_ready,
    input  wire          seal_bank,
    output reg           seal_done,
    output reg           seal_error,

    // Abort wins over same-bank capture, replay, configuration, and sealing.
    input  wire          abort_valid,
    input  wire          abort_bank,

    output wire [1:0]    bank_clearing,
    output wire [1:0]    bank_active,
    output wire [1:0]    bank_valid,
    output wire [1:0]    bank_error,
    output wire [10:0]   bank0_record_count,
    output wire [10:0]   bank1_record_count,

    // Tagged canonical record capture. TLAST is required only on beat four.
    input  wire [63:0]   s_axis_tdata,
    input  wire          s_axis_tvalid,
    output wire          s_axis_tready,
    input  wire          s_axis_tlast,
    input  wire          s_axis_bank,
    input  wire [1:0]    s_axis_token,
    input  wire [8:0]    s_axis_block,
    output reg           cap_record_done,
    output reg           cap_record_error,
    output wire          cap_commit_valid,
    output wire [11:0]   cap_commit_address,

    // One tagged record request. Invalid/unsealed requests return a stable,
    // five-beat all-zero response with m_axis_error asserted.
    input  wire          rd_req_valid,
    output wire          rd_req_ready,
    input  wire          rd_req_bank,
    input  wire [1:0]    rd_req_token,
    input  wire [8:0]    rd_req_block,
    output wire          rd_issue_valid,
    output wire [11:0]   rd_issue_address,

    output wire [63:0]   m_axis_tdata,
    output wire          m_axis_tvalid,
    input  wire          m_axis_tready,
    output wire          m_axis_tlast,
    output wire          m_axis_error,
    output wire          m_axis_bank,
    output wire [1:0]    m_axis_token,
    output wire [8:0]    m_axis_block
`ifdef FORMAL
    , output wire         formal_cap_busy
    , output wire [2:0]   formal_cap_index
    , output wire         formal_rd_pending
    , output wire [2:0]   formal_emit_index
`endif
);
    localparam integer BANK_RECORDS = 1536;
    localparam integer TOTAL_RECORDS = 3072;
    localparam [8:0] BLOCK_CAPACITY = 9'd384;
`ifdef FORMAL_REDUCED_STORAGE
    // Formal lifecycle/storage proofs exercise the minimum legal integrated
    // section (F=128), whose four blocks occupy addresses 0,4,8,12. Production
    // geometry and all address arithmetic stay fixed.
    localparam integer PAYLOAD_DEPTH = 16;
    localparam integer SEEN_DEPTH = 16;
    localparam [10:0] LAST_CLEAR_ADDRESS = 11'd15;
`else
    localparam integer PAYLOAD_DEPTH = TOTAL_RECORDS;
    localparam integer SEEN_DEPTH = BANK_RECORDS;
    localparam [10:0] LAST_LOCAL_ADDRESS = 11'd1535;
    localparam [10:0] LAST_CLEAR_ADDRESS = LAST_LOCAL_ADDRESS;
`endif

    function automatic [10:0] record_count_for_shape(
        input [2:0] tokens,
        input [8:0] blocks
    );
        case (tokens)
            3'd1: record_count_for_shape = {2'b00, blocks};
            3'd2: record_count_for_shape = {1'b0, blocks, 1'b0};
            3'd3: record_count_for_shape =
                {1'b0, blocks, 1'b0} + {2'b00, blocks};
            default: record_count_for_shape = {blocks, 2'b00};
        endcase
    endfunction

    function automatic [10:0] local_address(
        input [1:0] token,
        input [8:0] block
    );
        local_address = {block, 2'b00} + {9'd0, token};
    endfunction

    function automatic [11:0] physical_address(
        input bank,
        input [1:0] token,
        input [8:0] block
    );
        physical_address = (bank ? 12'd1536 : 12'd0) +
                           {1'b0, local_address(token, block)};
    endfunction

    // Full native records use one wide payload write and one narrow scale write.
    // The independent synchronous read port is compatible with simple dual-port
    // BRAM/UltraRAM inference; neither data memory carries reset logic.
    (* ram_style = "ultra" *) reg [255:0] payload_mem [0:PAYLOAD_DEPTH-1];
    (* ram_style = "block" *) reg [15:0] scale_mem [0:PAYLOAD_DEPTH-1];

    // Duplicate metadata is split per ping-pong bank so the inactive bank can
    // clear while the other bank captures or replays. These arrays are also
    // reset-free synchronous RAMs.
    // Pad the one-bit flags to a BRAM-native word width. Vivado otherwise maps
    // the independent simple-dual-port arrays into distributed RAM despite the
    // block style; bit zero remains the sole semantic duplicate flag.
    (* ram_style = "block" *) reg [7:0] seen0_mem [0:SEEN_DEPTH-1];
    (* ram_style = "block" *) reg [7:0] seen1_mem [0:SEEN_DEPTH-1];

    reg bank0_clearing_q;
    reg bank1_clearing_q;
    reg bank0_active_q;
    reg bank1_active_q;
    reg bank0_valid_q;
    reg bank1_valid_q;
    reg bank0_error_q;
    reg bank1_error_q;
    reg [2:0] bank0_tokens_q;
    reg [2:0] bank1_tokens_q;
    reg [8:0] bank0_blocks_q;
    reg [8:0] bank1_blocks_q;
    reg [10:0] bank0_expected_q;
    reg [10:0] bank1_expected_q;
    reg [10:0] bank0_count_q;
    reg [10:0] bank1_count_q;
    reg [10:0] bank0_clear_address_q;
    reg [10:0] bank1_clear_address_q;

    assign bank_clearing = {bank1_clearing_q, bank0_clearing_q};
    assign bank_active = {bank1_active_q, bank0_active_q};
    assign bank_valid = {bank1_valid_q, bank0_valid_q};
    assign bank_error = {bank1_error_q, bank0_error_q};
    assign bank0_record_count = bank0_count_q;
    assign bank1_record_count = bank1_count_q;

    wire cfg_shape_ok = (cfg_tokens != 3'd0) && (cfg_tokens <= 3'd4) &&
                        (cfg_blocks != 9'd0) &&
                        (cfg_blocks <= BLOCK_CAPACITY);
    wire cfg_bank_clearing = cfg_bank ? bank1_clearing_q : bank0_clearing_q;

    // ---- Capture staging and duplicate probe ----

    reg cap_busy_q;
    reg cap_bank_q;
    reg [1:0] cap_token_q;
    reg [8:0] cap_block_q;
    reg [10:0] cap_local_address_q;
    reg [2:0] cap_index_q;
    reg [255:0] cap_payload_q;
    reg cap_bad_q;
    reg cap_duplicate_q;
    reg cap_probe_pending_q;
    reg [7:0] seen0_probe_q;
    reg [7:0] seen1_probe_q;

    wire cap_target_active = s_axis_bank ? bank1_active_q : bank0_active_q;
    wire cap_target_valid = s_axis_bank ? bank1_valid_q : bank0_valid_q;
    wire cap_target_error = s_axis_bank ? bank1_error_q : bank0_error_q;
    wire cap_target_clearing = s_axis_bank ? bank1_clearing_q : bank0_clearing_q;
    wire [2:0] cap_target_tokens = s_axis_bank ? bank1_tokens_q : bank0_tokens_q;
    wire [8:0] cap_target_blocks = s_axis_bank ? bank1_blocks_q : bank0_blocks_q;
    wire cap_target_available = cap_target_active && !cap_target_valid &&
                                !cap_target_error && !cap_target_clearing;
    wire cap_current_abort = abort_valid &&
                             (abort_bank == (cap_busy_q ? cap_bank_q :
                                                           s_axis_bank));

    // Configuration and sealing only suppress a same-bank record boundary. A
    // record already in flight drains continuously unless it is explicitly
    // aborted, avoiding a control-dependent ready chain through either RAM.
    wire cfg_accept;
    wire seal_accept;
    assign s_axis_tready = rst_n && !cap_current_abort &&
        (cap_busy_q || (cap_target_available &&
         !(cfg_accept && (cfg_bank == s_axis_bank)) &&
         !(seal_accept && (seal_bank == s_axis_bank))));
    wire cap_accept = s_axis_tvalid && s_axis_tready;
    wire cap_start = cap_accept && !cap_busy_q;
    wire cap_body = cap_accept && cap_busy_q;

    wire cap_first_tag_bad = ({1'b0, s_axis_token} >= cap_target_tokens) ||
                             (s_axis_block >= BLOCK_CAPACITY) ||
                             (s_axis_block >= cap_target_blocks);
    wire cap_first_fault = cap_start &&
                           (cap_first_tag_bad || s_axis_tlast);
    wire cap_body_tag_bad = (s_axis_bank != cap_bank_q) ||
                            (s_axis_token != cap_token_q) ||
                            (s_axis_block != cap_block_q);
    wire cap_body_frame_bad = s_axis_tlast != (cap_index_q == 3'd4);
    wire cap_scale_pad_bad = (cap_index_q == 3'd4) &&
                             (s_axis_tdata[63:16] != 48'd0);
    wire cap_body_fault = cap_body &&
                          (cap_body_tag_bad || cap_body_frame_bad ||
                           cap_scale_pad_bad);
    wire cap_probe_fault = cap_probe_pending_q &&
                           (cap_bank_q ? seen1_probe_q[0] :
                                         seen0_probe_q[0]);

    wire cap_final = cap_body && (cap_index_q == 3'd4);
    wire cap_bank_sticky_error = cap_bank_q ? bank1_error_q : bank0_error_q;
    wire cap_commit_ok = cap_final && !cap_bad_q && !cap_duplicate_q &&
                         !cap_body_fault && !cap_bank_sticky_error;
    assign cap_commit_valid = cap_commit_ok;
    assign cap_commit_address = physical_address(cap_bank_q, cap_token_q,
                                                 cap_block_q);

    // ---- Elastic synchronous replay ----

    reg rd_pending_q;
    reg rd_pending_bank_q;
    reg [1:0] rd_pending_token_q;
    reg [8:0] rd_pending_block_q;
    reg rd_pending_error_q;
    reg [255:0] rd_mem_payload_q;
    reg [15:0] rd_mem_scale_q;

    reg out_valid_q;
    reg [255:0] out_payload_q;
    reg [15:0] out_scale_q;
    reg [2:0] out_index_q;
    reg out_error_q;
    reg out_bank_q;
    reg [1:0] out_token_q;
    reg [8:0] out_block_q;

    wire rd_target_valid = rd_req_bank ? bank1_valid_q : bank0_valid_q;
    wire rd_target_error = rd_req_bank ? bank1_error_q : bank0_error_q;
    wire [2:0] rd_target_tokens = rd_req_bank ? bank1_tokens_q : bank0_tokens_q;
    wire [8:0] rd_target_blocks = rd_req_bank ? bank1_blocks_q : bank0_blocks_q;
    wire rd_request_bad = !rd_target_valid || rd_target_error ||
                          ({1'b0, rd_req_token} >= rd_target_tokens) ||
                          (rd_req_block >= BLOCK_CAPACITY) ||
                          (rd_req_block >= rd_target_blocks);
    wire [11:0] rd_address_unchecked = physical_address(
        rd_req_bank, rd_req_token, rd_req_block);
    wire [11:0] rd_address = ((rd_req_block >= BLOCK_CAPACITY) ?
                              12'd0 : rd_address_unchecked);

    wire rd_inflight = rd_pending_q || out_valid_q;
    wire rd_inflight_bank = out_valid_q ? out_bank_q : rd_pending_bank_q;
    wire rd_target_abort = abort_valid && (abort_bank == rd_req_bank);
    assign rd_req_ready = rst_n && !rd_inflight && !rd_target_abort &&
                          !(cfg_accept && (cfg_bank == rd_req_bank)) &&
                          !(seal_accept && (seal_bank == rd_req_bank));
    wire rd_accept = rd_req_valid && rd_req_ready;
    assign rd_issue_valid = rd_accept;
    assign rd_issue_address = rd_address;

    wire out_abort = abort_valid && out_valid_q &&
                     (abort_bank == out_bank_q);
    assign m_axis_tvalid = out_valid_q && !out_abort;
    assign m_axis_tlast = out_index_q == 3'd4;
    assign m_axis_error = out_error_q;
    assign m_axis_bank = out_bank_q;
    assign m_axis_token = out_token_q;
    assign m_axis_block = out_block_q;
    assign m_axis_tdata = out_error_q ? 64'd0 :
        ((out_index_q == 3'd0) ? out_payload_q[63:0] :
         (out_index_q == 3'd1) ? out_payload_q[127:64] :
         (out_index_q == 3'd2) ? out_payload_q[191:128] :
         (out_index_q == 3'd3) ? out_payload_q[255:192] :
                                  {48'd0, out_scale_q});
    wire out_accept = m_axis_tvalid && m_axis_tready;

    wire bank0_rd_in_use = rd_inflight && !rd_inflight_bank;
    wire bank1_rd_in_use = rd_inflight && rd_inflight_bank;
    wire bank0_cap_in_use = cap_busy_q && !cap_bank_q;
    wire bank1_cap_in_use = cap_busy_q && cap_bank_q;

    // Abort is globally prioritized over new lifecycle requests. Configuration
    // wins over a simultaneous same-bank seal request.
    assign cfg_ready = rst_n && !abort_valid && !cfg_bank_clearing &&
                       !(cfg_bank ? bank1_cap_in_use : bank0_cap_in_use) &&
                       !(cfg_bank ? bank1_rd_in_use : bank0_rd_in_use);
    assign cfg_accept = cfg_valid && cfg_ready;
    assign seal_ready = rst_n && !abort_valid &&
                        !(cfg_valid && (cfg_bank == seal_bank)) &&
                        !(seal_bank ? bank1_clearing_q : bank0_clearing_q) &&
                        !(seal_bank ? bank1_cap_in_use : bank0_cap_in_use) &&
                        !(seal_bank ? bank1_rd_in_use : bank0_rd_in_use);
    assign seal_accept = seal_valid && seal_ready;

    wire seal0_ok = bank0_active_q && !bank0_error_q &&
                    (bank0_count_q == bank0_expected_q) &&
                    (bank0_expected_q != 11'd0);
    wire seal1_ok = bank1_active_q && !bank1_error_q &&
                    (bank1_count_q == bank1_expected_q) &&
                    (bank1_expected_q != 11'd0);

    // Data and duplicate memories. A complete capture performs exactly one
    // payload/scale write. Read data is registered once at the hard-memory
    // boundary and once again in the elastic output slot below.
    always @(posedge clk) begin
        if (cap_commit_valid) begin
            payload_mem[cap_commit_address] <= cap_payload_q;
            scale_mem[cap_commit_address] <= s_axis_tdata[15:0];
        end
        if (rd_accept) begin
            rd_mem_payload_q <= payload_mem[rd_address];
            rd_mem_scale_q <= scale_mem[rd_address];
        end
    end

    wire seen0_write_enable = bank0_clearing_q ||
                              (cap_commit_valid && !cap_bank_q);
    wire [10:0] seen0_write_address = bank0_clearing_q ?
        bank0_clear_address_q : cap_local_address_q;
    wire [7:0] seen0_write_data = bank0_clearing_q ? 8'd0 : 8'd1;
    wire seen1_write_enable = bank1_clearing_q ||
                              (cap_commit_valid && cap_bank_q);
    wire [10:0] seen1_write_address = bank1_clearing_q ?
        bank1_clear_address_q : cap_local_address_q;
    wire [7:0] seen1_write_data = bank1_clearing_q ? 8'd0 : 8'd1;
    wire [10:0] cap_probe_address = (s_axis_block < BLOCK_CAPACITY) ?
        local_address(s_axis_token, s_axis_block) : 11'd0;

    always @(posedge clk) begin
        if (seen0_write_enable)
            seen0_mem[seen0_write_address] <= seen0_write_data;
    end

    always @(posedge clk) begin
        if (cap_start && !s_axis_bank)
            seen0_probe_q <= seen0_mem[cap_probe_address];
    end

    always @(posedge clk) begin
        if (seen1_write_enable)
            seen1_mem[seen1_write_address] <= seen1_write_data;
    end

    always @(posedge clk) begin
        if (cap_start && s_axis_bank)
            seen1_probe_q <= seen1_mem[cap_probe_address];
    end

    // Bank-zero lifecycle and exact-record accounting.
    always @(posedge clk) begin
        if (!rst_n) begin
            bank0_clearing_q <= 1'b0;
            bank0_active_q <= 1'b0;
            bank0_valid_q <= 1'b0;
            bank0_error_q <= 1'b0;
            bank0_tokens_q <= 3'd0;
            bank0_blocks_q <= 9'd0;
            bank0_expected_q <= 11'd0;
            bank0_count_q <= 11'd0;
            bank0_clear_address_q <= 11'd0;
        end else if (abort_valid && !abort_bank) begin
            bank0_clearing_q <= 1'b0;
            bank0_active_q <= 1'b0;
            bank0_valid_q <= 1'b0;
            bank0_error_q <= 1'b1;
            bank0_count_q <= 11'd0;
            bank0_clear_address_q <= 11'd0;
        end else if (cfg_accept && !cfg_bank) begin
            bank0_clearing_q <= cfg_shape_ok;
            bank0_active_q <= 1'b0;
            bank0_valid_q <= 1'b0;
            bank0_error_q <= !cfg_shape_ok;
            bank0_tokens_q <= cfg_tokens;
            bank0_blocks_q <= cfg_blocks;
            bank0_expected_q <= cfg_shape_ok ?
                record_count_for_shape(cfg_tokens, cfg_blocks) : 11'd0;
            bank0_count_q <= 11'd0;
            bank0_clear_address_q <= 11'd0;
        end else begin
            if (bank0_clearing_q) begin
                if (bank0_clear_address_q == LAST_CLEAR_ADDRESS) begin
                    bank0_clearing_q <= 1'b0;
                    bank0_active_q <= 1'b1;
                    bank0_clear_address_q <= 11'd0;
                end else begin
                    bank0_clear_address_q <= bank0_clear_address_q + 1'b1;
                end
            end
            if (seal_accept && !seal_bank) begin
                bank0_clearing_q <= 1'b0;
                bank0_active_q <= 1'b0;
                bank0_valid_q <= seal0_ok;
                if (!seal0_ok)
                    bank0_error_q <= 1'b1;
            end
            if ((cap_first_fault && !s_axis_bank) ||
                (cap_body_fault && !cap_bank_q) ||
                (cap_probe_fault && !cap_bank_q)) begin
                bank0_valid_q <= 1'b0;
                bank0_error_q <= 1'b1;
            end
            if (cap_commit_valid && !cap_bank_q)
                bank0_count_q <= bank0_count_q + 1'b1;
        end
    end

    // Bank-one lifecycle and exact-record accounting.
    always @(posedge clk) begin
        if (!rst_n) begin
            bank1_clearing_q <= 1'b0;
            bank1_active_q <= 1'b0;
            bank1_valid_q <= 1'b0;
            bank1_error_q <= 1'b0;
            bank1_tokens_q <= 3'd0;
            bank1_blocks_q <= 9'd0;
            bank1_expected_q <= 11'd0;
            bank1_count_q <= 11'd0;
            bank1_clear_address_q <= 11'd0;
        end else if (abort_valid && abort_bank) begin
            bank1_clearing_q <= 1'b0;
            bank1_active_q <= 1'b0;
            bank1_valid_q <= 1'b0;
            bank1_error_q <= 1'b1;
            bank1_count_q <= 11'd0;
            bank1_clear_address_q <= 11'd0;
        end else if (cfg_accept && cfg_bank) begin
            bank1_clearing_q <= cfg_shape_ok;
            bank1_active_q <= 1'b0;
            bank1_valid_q <= 1'b0;
            bank1_error_q <= !cfg_shape_ok;
            bank1_tokens_q <= cfg_tokens;
            bank1_blocks_q <= cfg_blocks;
            bank1_expected_q <= cfg_shape_ok ?
                record_count_for_shape(cfg_tokens, cfg_blocks) : 11'd0;
            bank1_count_q <= 11'd0;
            bank1_clear_address_q <= 11'd0;
        end else begin
            if (bank1_clearing_q) begin
                if (bank1_clear_address_q == LAST_CLEAR_ADDRESS) begin
                    bank1_clearing_q <= 1'b0;
                    bank1_active_q <= 1'b1;
                    bank1_clear_address_q <= 11'd0;
                end else begin
                    bank1_clear_address_q <= bank1_clear_address_q + 1'b1;
                end
            end
            if (seal_accept && seal_bank) begin
                bank1_clearing_q <= 1'b0;
                bank1_active_q <= 1'b0;
                bank1_valid_q <= seal1_ok;
                if (!seal1_ok)
                    bank1_error_q <= 1'b1;
            end
            if ((cap_first_fault && s_axis_bank) ||
                (cap_body_fault && cap_bank_q) ||
                (cap_probe_fault && cap_bank_q)) begin
                bank1_valid_q <= 1'b0;
                bank1_error_q <= 1'b1;
            end
            if (cap_commit_valid && cap_bank_q)
                bank1_count_q <= bank1_count_q + 1'b1;
        end
    end

    // Five-beat capture staging. The complete payload reaches memory only on a
    // clean scale-beat handshake, so malformed or aborted records cannot expose
    // partially updated data.
    always @(posedge clk) begin
        if (!rst_n) begin
            cap_busy_q <= 1'b0;
            cap_bank_q <= 1'b0;
            cap_token_q <= 2'd0;
            cap_block_q <= 9'd0;
            cap_local_address_q <= 11'd0;
            cap_index_q <= 3'd0;
            cap_payload_q <= 256'd0;
            cap_bad_q <= 1'b0;
            cap_duplicate_q <= 1'b0;
            cap_probe_pending_q <= 1'b0;
            cap_record_done <= 1'b0;
            cap_record_error <= 1'b0;
        end else begin
            cap_record_done <= 1'b0;
            cap_record_error <= 1'b0;

            if (abort_valid && cap_busy_q && (abort_bank == cap_bank_q)) begin
                cap_busy_q <= 1'b0;
                cap_index_q <= 3'd0;
                cap_bad_q <= 1'b0;
                cap_duplicate_q <= 1'b0;
                cap_probe_pending_q <= 1'b0;
            end else begin
                if (cap_probe_pending_q) begin
                    cap_probe_pending_q <= 1'b0;
                    cap_duplicate_q <= cap_bank_q ? seen1_probe_q[0] :
                                                    seen0_probe_q[0];
                    if (cap_bank_q ? seen1_probe_q[0] : seen0_probe_q[0])
                        cap_bad_q <= 1'b1;
                end

                if (cap_start) begin
                    cap_busy_q <= 1'b1;
                    cap_bank_q <= s_axis_bank;
                    cap_token_q <= s_axis_token;
                    cap_block_q <= s_axis_block;
                    cap_local_address_q <= (s_axis_block < BLOCK_CAPACITY) ?
                        local_address(s_axis_token, s_axis_block) : 11'd0;
                    cap_index_q <= 3'd1;
                    cap_payload_q[63:0] <= s_axis_tdata;
                    cap_bad_q <= cap_first_tag_bad || s_axis_tlast;
                    cap_duplicate_q <= 1'b0;
                    cap_probe_pending_q <= 1'b1;
                end else if (cap_body) begin
                    if (cap_body_fault)
                        cap_bad_q <= 1'b1;
                    case (cap_index_q)
                        3'd1: cap_payload_q[127:64] <= s_axis_tdata;
                        3'd2: cap_payload_q[191:128] <= s_axis_tdata;
                        3'd3: cap_payload_q[255:192] <= s_axis_tdata;
                        default: ;
                    endcase
                    if (cap_index_q == 3'd4) begin
                        cap_busy_q <= 1'b0;
                        cap_index_q <= 3'd0;
                        cap_bad_q <= 1'b0;
                        cap_duplicate_q <= 1'b0;
                        cap_probe_pending_q <= 1'b0;
                        cap_record_done <= 1'b1;
                        cap_record_error <= !cap_commit_ok;
                    end else begin
                        cap_index_q <= cap_index_q + 1'b1;
                    end
                end
            end
        end
    end

    // One-request/one-response elastic replay. The registered payload, scale,
    // tags, error, and beat index are all held while the consumer is stalled.
    always @(posedge clk) begin
        if (!rst_n) begin
            rd_pending_q <= 1'b0;
            rd_pending_bank_q <= 1'b0;
            rd_pending_token_q <= 2'd0;
            rd_pending_block_q <= 9'd0;
            rd_pending_error_q <= 1'b0;
            out_valid_q <= 1'b0;
            out_payload_q <= 256'd0;
            out_scale_q <= 16'd0;
            out_index_q <= 3'd0;
            out_error_q <= 1'b0;
            out_bank_q <= 1'b0;
            out_token_q <= 2'd0;
            out_block_q <= 9'd0;
        end else if (abort_valid && rd_inflight &&
                     (abort_bank == rd_inflight_bank)) begin
            rd_pending_q <= 1'b0;
            out_valid_q <= 1'b0;
            out_index_q <= 3'd0;
            out_error_q <= 1'b0;
        end else begin
            rd_pending_q <= rd_accept;
            if (rd_accept) begin
                rd_pending_bank_q <= rd_req_bank;
                rd_pending_token_q <= rd_req_token;
                rd_pending_block_q <= rd_req_block;
                rd_pending_error_q <= rd_request_bad;
            end

            if (rd_pending_q) begin
                out_valid_q <= 1'b1;
                out_payload_q <= rd_mem_payload_q;
                out_scale_q <= rd_mem_scale_q;
                out_index_q <= 3'd0;
                out_error_q <= rd_pending_error_q;
                out_bank_q <= rd_pending_bank_q;
                out_token_q <= rd_pending_token_q;
                out_block_q <= rd_pending_block_q;
            end else if (out_accept) begin
                if (out_index_q == 3'd4) begin
                    out_valid_q <= 1'b0;
                    out_index_q <= 3'd0;
                    out_error_q <= 1'b0;
                end else begin
                    out_index_q <= out_index_q + 1'b1;
                end
            end
        end
    end

    always @(posedge clk) begin
        if (!rst_n) begin
            seal_done <= 1'b0;
            seal_error <= 1'b0;
        end else begin
            seal_done <= seal_accept;
            if (seal_accept)
                seal_error <= seal_bank ? !seal1_ok : !seal0_ok;
            else
                seal_error <= 1'b0;
        end
    end

`ifdef FORMAL
    assign formal_cap_busy = cap_busy_q;
    assign formal_cap_index = cap_index_q;
    assign formal_rd_pending = rd_pending_q;
    assign formal_emit_index = out_index_q;

    always @(posedge clk) begin
        if (rst_n) begin
            assert(bank0_count_q <= 11'd1536);
            assert(bank1_count_q <= 11'd1536);
            if (bank0_valid_q) begin
                assert(!bank0_active_q && !bank0_clearing_q && !bank0_error_q);
                assert(bank0_count_q == bank0_expected_q);
            end
            if (bank1_valid_q) begin
                assert(!bank1_active_q && !bank1_clearing_q && !bank1_error_q);
                assert(bank1_count_q == bank1_expected_q);
            end
            if (cap_commit_valid) begin
                assert(cap_commit_address < 12'd3072);
                assert(cap_local_address_q < 11'd1536);
            end
            if (rd_issue_valid)
                assert(rd_issue_address < 12'd3072);
            if (cap_busy_q)
                assert(cap_index_q >= 3'd1 && cap_index_q <= 3'd4);
            if (out_valid_q)
                assert(out_index_q <= 3'd4);
        end
    end
`endif

endmodule

`default_nettype wire
