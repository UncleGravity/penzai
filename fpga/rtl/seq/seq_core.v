// seq_core - the command-executor core of seq.v (docs/plan-seq-impl-v2.md, v2.1).
//
// Walks a {WRITE|WAIT} entry list and replays it onto the control bus, with no PS in the inner
// loop — the clock-invariant ~103us/op register dance moved into the PL. This is the op-AGNOSTIC
// core: it knows only "write a register" and "poll a register until it matches", so plan-6 PL-op
// fusion lands as longer runs with ZERO change here. Strictly in-order by design, forever:
// op-to-op overlap is the HOST's job (emit op N+1's read-DMA starts before op N's WAITs).
//
// Interfaces are simple req/gnt (1-cycle) memory + register ports. seq_top wraps the desc port
// around the command BRAM (v2.1: PS-filled, no DRAM) and the reg port in an AXI-Lite master;
// keeping them abstract lets this core be cosim-gated in isolation (test-rtl-seq) for the part
// that matters: write-replay ORDER, WAIT poll-until-match, done, and both safety nets (per-WAIT
// poll budget + the global no-progress watchdog). Abort is a synchronous reset from seq_top —
// this core carries no abort logic.
//
// Entry = 128b, little-endian u32x4 { tag @ [31:0], addr @ [63:32], a @ [95:64], b @ [127:96] }:
//   tag WRITE(0): reg.wr(addr, a)                          // b ignored
//   tag WAIT (1): poll reg.rd(addr) until (rdata & a) == b // a=mask, b=expected
//   other tags  : stop early (a run normally ends after desc_count entries)

`default_nettype none

module seq_core #(
    parameter integer ADDR_W       = 32,   // control-bus byte address width
    parameter integer COUNT_W      = 16,   // entries per run
    // Per-WAIT poll budget before erroring out. SMALL default for cosim; BD integration overrides
    // to ~2_000_000 (a matmul op is ~100us ~= 30k cycles @300MHz, so the budget must exceed it).
    parameter integer POLL_TIMEOUT = 1024,
    // Global no-progress watchdog: cycles without any gnt while busy before erroring out. Covers
    // what POLL_TIMEOUT cannot — a bus transaction that never completes (dead slave). SMALL
    // default for cosim; BD integration overrides to ~8_000_000 (~27ms @300MHz).
    parameter integer WATCHDOG_TIMEOUT = 4096
) (
    input  wire                 clk,
    input  wire                 rst_n,

    // ---- control (PS-facing; a thin AXI-Lite slave wraps this in integration) ----
    input  wire                 go,           // 1-cycle pulse: start a run of desc_count entries
    input  wire [COUNT_W-1:0]   desc_count,
    output reg                  busy,
    output reg                  done,         // level: high from finish until the next `go`
    output reg                  err_timeout,  // a WAIT exceeded POLL_TIMEOUT
    output reg                  err_watchdog, // no gnt for WATCHDOG_TIMEOUT cycles while busy
    output reg  [COUNT_W-1:0]   err_index,    // entry index at the fault (debug)

    // ---- descriptor fetch (req/gnt, 1-cycle): present idx, gnt returns the 128b entry ----
    output reg                  desc_req,
    output reg  [COUNT_W-1:0]   desc_idx,
    input  wire                 desc_gnt,
    input  wire [127:0]         desc_data,

    // ---- register replay master (req/gnt): we=1 -> write(addr,wdata); we=0 -> read(addr)->rdata
    output reg                  reg_req,
    output reg                  reg_we,
    output reg  [ADDR_W-1:0]    reg_addr,
    output reg  [31:0]          reg_wdata,
    input  wire                 reg_gnt,
    input  wire [31:0]          reg_rdata
);
    localparam [1:0] TAG_WRITE = 2'd0, TAG_WAIT = 2'd1; // any other tag stops the run

    localparam [2:0]
        S_IDLE     = 3'd0,
        S_FETCH    = 3'd1,   // desc_req held until gnt; latch the entry
        S_DEC      = 3'd2,   // classify the latched entry
        S_WR       = 3'd3,   // reg write req held until gnt
        S_RD_ISSUE = 3'd4,   // start one poll read
        S_RD_WAIT  = 3'd5,   // wait the poll's gnt; match / re-poll / timeout
        S_ADV      = 3'd6,   // advance to the next entry (or finish)
        S_FIN      = 3'd7;

    reg [2:0]            state;
    reg [COUNT_W-1:0]    idx;
    // Snapshot the PS-facing count with `go`. RUN_COUNT remains writable in seq_top while busy;
    // observing it live in S_ADV can lengthen/shorten an in-flight run.
    reg [COUNT_W-1:0]    run_count;
    reg [1:0]            e_tag;
    reg [ADDR_W-1:0]     e_addr;
    reg [31:0]           e_a, e_b;
    reg [31:0]           poll_cnt;
    reg [31:0]           wd_cnt;

    // Progress = any gnt. The counter only runs while executing (not IDLE/FIN), so a fired
    // watchdog can never re-fire from DONE and an idle core never counts.
    wire executing = busy && state != S_IDLE && state != S_FIN;
    wire progress  = desc_gnt || reg_gnt;

    // entry field views of the freshly-fetched 128b word
    wire [1:0]        d_tag  = desc_data[1:0];
    wire [ADDR_W-1:0] d_addr = desc_data[32 +: ADDR_W];
    wire [31:0]       d_a    = desc_data[64 +: 32];
    wire [31:0]       d_b    = desc_data[96 +: 32];

    wire _unused = &{1'b0, desc_data[31:2]};

    always @(posedge clk) begin
        if (!rst_n) begin
            state <= S_IDLE; busy <= 1'b0; done <= 1'b0; err_timeout <= 1'b0; err_watchdog <= 1'b0;
            err_index <= {COUNT_W{1'b0}};
            idx <= {COUNT_W{1'b0}}; run_count <= {COUNT_W{1'b0}};
            poll_cnt <= 32'd0; wd_cnt <= 32'd0;
            desc_req <= 1'b0; desc_idx <= {COUNT_W{1'b0}};
            reg_req <= 1'b0; reg_we <= 1'b0; reg_addr <= {ADDR_W{1'b0}}; reg_wdata <= 32'd0;
            e_tag <= 2'd0; e_addr <= {ADDR_W{1'b0}}; e_a <= 32'd0; e_b <= 32'd0;
        end else if (executing && !progress && wd_cnt >= WATCHDOG_TIMEOUT[31:0]) begin
            // No forward progress for the whole budget: something on the bus is dead. Stop
            // reporting err_watchdog; the stuck adapter (if any) is reclaimed by ABORT's reset.
            err_watchdog <= 1'b1; err_index <= idx;
            desc_req <= 1'b0; reg_req <= 1'b0;
            state <= S_FIN;
        end else begin
            wd_cnt <= (progress || !executing) ? 32'd0 : wd_cnt + 32'd1;
            case (state)
                S_IDLE: begin
                    if (go) begin
                        busy <= 1'b1; done <= 1'b0; err_timeout <= 1'b0; err_watchdog <= 1'b0;
                        idx <= {COUNT_W{1'b0}}; run_count <= desc_count;
                        if (desc_count == {COUNT_W{1'b0}}) begin
                            state <= S_FIN;                       // empty run
                        end else begin
                            desc_req <= 1'b1; desc_idx <= {COUNT_W{1'b0}}; state <= S_FETCH;
                        end
                    end
                end

                S_FETCH: begin
                    if (desc_req && desc_gnt) begin
                        desc_req <= 1'b0;
                        e_tag <= d_tag; e_addr <= d_addr; e_a <= d_a; e_b <= d_b;
                        state <= S_DEC;
                    end
                end

                S_DEC: begin
                    case (e_tag)
                        TAG_WRITE: begin
                            reg_req <= 1'b1; reg_we <= 1'b1; reg_addr <= e_addr; reg_wdata <= e_a;
                            state <= S_WR;
                        end
                        TAG_WAIT: begin
                            poll_cnt <= 32'd0;
                            state <= S_RD_ISSUE;
                        end
                        default: state <= S_FIN;                  // TAG_END (or unknown): stop
                    endcase
                end

                S_WR: begin
                    if (reg_req && reg_gnt) begin
                        reg_req <= 1'b0;
                        state <= S_ADV;
                    end
                end

                S_RD_ISSUE: begin
                    reg_req <= 1'b1; reg_we <= 1'b0; reg_addr <= e_addr;
                    state <= S_RD_WAIT;
                end

                S_RD_WAIT: begin
                    if (reg_req && reg_gnt) begin
                        reg_req <= 1'b0;
                        if ((reg_rdata & e_a) == e_b) begin
                            state <= S_ADV;                        // poll satisfied
                        end else if (poll_cnt >= POLL_TIMEOUT[31:0]) begin
                            err_timeout <= 1'b1; err_index <= idx; state <= S_FIN;
                        end else begin
                            poll_cnt <= poll_cnt + 32'd1;
                            state <= S_RD_ISSUE;                   // re-poll
                        end
                    end
                end

                S_ADV: begin
                    if (idx + {{(COUNT_W-1){1'b0}}, 1'b1} == run_count) begin
                        state <= S_FIN;
                    end else begin
                        idx <= idx + {{(COUNT_W-1){1'b0}}, 1'b1};
                        desc_req <= 1'b1; desc_idx <= idx + {{(COUNT_W-1){1'b0}}, 1'b1};
                        state <= S_FETCH;
                    end
                end

                S_FIN: begin
                    busy <= 1'b0; done <= 1'b1; state <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end
endmodule
