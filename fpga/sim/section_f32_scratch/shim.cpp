#include "Vsection_f32_scratch.h"
#include "verilated.h"
#include "shim.h"

struct Dut {
    Vsection_f32_scratch *top;
};

double sc_time_stamp() { return 0; }

extern "C" {

Dut *dut_new(void) {
    Dut *d = new Dut();
    d->top = new Vsection_f32_scratch();
    return d;
}

void dut_free(Dut *d) {
    delete d->top;
    delete d;
}

void dut_eval(Dut *d) { d->top->eval(); }
void dut_set_clk(Dut *d, int value) { d->top->clk = value; }
void dut_set_rst_n(Dut *d, int value) { d->top->rst_n = value; }

void dut_set_write_config(Dut *d, int valid, uint8_t role, uint16_t rows, uint8_t tokens) {
    d->top->wr_cfg_valid = valid;
    d->top->wr_cfg_role = role;
    d->top->wr_cfg_rows = rows;
    d->top->wr_cfg_tokens = tokens;
}

void dut_set_write_abort(Dut *d, int value) { d->top->wr_abort = value; }

void dut_set_write_stream(Dut *d, int valid, uint64_t data, uint8_t keep, int last) {
    d->top->s_axis_tvalid = valid;
    d->top->s_axis_tdata = data;
    d->top->s_axis_tkeep = keep;
    d->top->s_axis_tlast = last;
}

void dut_set_read_request(Dut *d, int valid, uint8_t role, uint8_t token, uint16_t group) {
    d->top->rd_req_valid = valid;
    d->top->rd_req_role = role;
    d->top->rd_req_token = token;
    d->top->rd_req_group = group;
}

void dut_set_read_ready(Dut *d, int value) { d->top->rd_rsp_ready = value; }

int dut_write_config_ready(Dut *d) { return d->top->wr_cfg_ready; }
int dut_write_busy(Dut *d) { return d->top->wr_busy; }
int dut_write_done(Dut *d) { return d->top->wr_done; }
int dut_write_error(Dut *d) { return d->top->wr_error; }
int dut_write_stream_ready(Dut *d) { return d->top->s_axis_tready; }
int dut_write_commit_valid(Dut *d) { return d->top->wr_commit_valid; }
uint8_t dut_write_commit_bank(Dut *d) { return d->top->wr_commit_bank; }
uint16_t dut_write_commit_address(Dut *d) { return d->top->wr_commit_address; }
int dut_read_request_ready(Dut *d) { return d->top->rd_req_ready; }
int dut_read_issue_valid(Dut *d) { return d->top->rd_issue_valid; }
uint16_t dut_read_issue_address(Dut *d) { return d->top->rd_issue_address; }
int dut_read_response_valid(Dut *d) { return d->top->rd_rsp_valid; }
int dut_read_response_error(Dut *d) { return d->top->rd_rsp_error; }

uint64_t dut_read_response_lane(Dut *d, uint32_t lane) {
    const uint32_t lo = d->top->rd_rsp_data[lane * 2];
    const uint32_t hi = d->top->rd_rsp_data[lane * 2 + 1];
    return static_cast<uint64_t>(lo) | (static_cast<uint64_t>(hi) << 32);
}

}
