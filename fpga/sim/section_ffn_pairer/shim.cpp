#include "Vsection_ffn_pairer.h"
#include "verilated.h"
#include "shim.h"

struct Dut {
    Vsection_ffn_pairer *top;
};

double sc_time_stamp() { return 0; }

static void set_wide(VlWide<8> &dst, const uint32_t lanes[8]) {
    for (int i = 0; i < 8; ++i) dst[i] = lanes[i];
}

extern "C" {

Dut *dut_new(void) {
    Dut *d = new Dut();
    d->top = new Vsection_ffn_pairer();
    return d;
}

void dut_free(Dut *d) {
    delete d->top;
    delete d;
}

void dut_eval(Dut *d) { d->top->eval(); }
void dut_set_clk(Dut *d, int value) { d->top->clk = value; }
void dut_set_rst_n(Dut *d, int value) { d->top->rst_n = value; }

void dut_set_start(Dut *d, int valid, uint8_t tokens, uint16_t blocks) {
    d->top->start_valid = valid;
    d->top->start_tokens = tokens;
    d->top->start_blocks = blocks;
}

void dut_set_abort(Dut *d, int value) { d->top->abort_run = value; }

void dut_set_gate(Dut *d, int valid, const uint32_t lanes[8], int last,
                  uint8_t token, uint16_t block, uint8_t group) {
    d->top->s_axis_tvalid = valid;
    set_wide(d->top->s_axis_tdata, lanes);
    d->top->s_axis_tlast = last;
    d->top->s_axis_token = token;
    d->top->s_axis_block = block;
    d->top->s_axis_group = group;
}

void dut_set_read_request_ready(Dut *d, int value) {
    d->top->rd_req_ready = value;
}

void dut_set_read_response(Dut *d, int valid, const uint32_t lanes[8], int error) {
    d->top->rd_rsp_valid = valid;
    set_wide(d->top->rd_rsp_data, lanes);
    d->top->rd_rsp_error = error;
}

void dut_set_output_ready(Dut *d, int value) { d->top->out_ready = value; }

int dut_start_ready(Dut *d) { return d->top->start_ready; }
int dut_busy(Dut *d) { return d->top->busy; }
int dut_done(Dut *d) { return d->top->done; }
int dut_error(Dut *d) { return d->top->error; }
int dut_gate_ready(Dut *d) { return d->top->s_axis_tready; }
int dut_read_request_valid(Dut *d) { return d->top->rd_req_valid; }
uint8_t dut_read_request_role(Dut *d) { return d->top->rd_req_role; }
uint8_t dut_read_request_token(Dut *d) { return d->top->rd_req_token; }
uint16_t dut_read_request_group(Dut *d) { return d->top->rd_req_group; }
int dut_read_response_ready(Dut *d) { return d->top->rd_rsp_ready; }
int dut_output_valid(Dut *d) { return d->top->out_valid; }
uint32_t dut_output_gate(Dut *d) { return d->top->out_gate; }
uint32_t dut_output_up(Dut *d) { return d->top->out_up; }
int dut_output_last(Dut *d) { return d->top->out_last; }

}
