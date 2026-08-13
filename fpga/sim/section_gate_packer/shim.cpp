#include "Vsection_gate_packer.h"
#include "verilated.h"
#include "shim.h"

struct Dut {
    Vsection_gate_packer *top;
};

double sc_time_stamp() { return 0; }

extern "C" {

Dut *dut_new(void) {
    Dut *d = new Dut();
    d->top = new Vsection_gate_packer();
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

void dut_set_input(Dut *d, int valid, uint64_t data, uint8_t keep, int last) {
    d->top->s_axis_tvalid = valid;
    d->top->s_axis_tdata = data;
    d->top->s_axis_tkeep = keep;
    d->top->s_axis_tlast = last;
}

void dut_set_output_ready(Dut *d, int value) { d->top->m_axis_tready = value; }

int dut_start_ready(Dut *d) { return d->top->start_ready; }
int dut_busy(Dut *d) { return d->top->busy; }
int dut_done(Dut *d) { return d->top->done; }
int dut_error(Dut *d) { return d->top->error; }
int dut_input_ready(Dut *d) { return d->top->s_axis_tready; }
int dut_output_valid(Dut *d) { return d->top->m_axis_tvalid; }

void dut_output_data(Dut *d, uint32_t lanes[8]) {
    for (int i = 0; i < 8; ++i) lanes[i] = d->top->m_axis_tdata[i];
}

int dut_output_last(Dut *d) { return d->top->m_axis_tlast; }
uint8_t dut_output_token(Dut *d) { return d->top->m_axis_token; }
uint16_t dut_output_block(Dut *d) { return d->top->m_axis_block; }
uint8_t dut_output_group(Dut *d) { return d->top->m_axis_group; }

}
