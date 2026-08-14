#include "Vsection_rmsnorm_loader.h"
#include "verilated.h"
#include "shim.h"

struct Dut {
    Vsection_rmsnorm_loader *top;
};

double sc_time_stamp() { return 0; }

extern "C" {

Dut *dut_new(void) {
    Dut *d = new Dut();
    d->top = new Vsection_rmsnorm_loader();
    return d;
}

void dut_free(Dut *d) {
    delete d->top;
    delete d;
}

void dut_eval(Dut *d) { d->top->eval(); }
void dut_set_clk(Dut *d, int value) { d->top->clk = value; }
void dut_set_rst_n(Dut *d, int value) { d->top->rst_n = value; }

void dut_set_config(Dut *d, int valid, uint16_t rows, uint8_t tokens) {
    d->top->cfg_valid = valid;
    d->top->cfg_rows = rows;
    d->top->cfg_tokens = tokens;
}

void dut_set_abort(Dut *d, int value) { d->top->abort_run = value; }

void dut_set_input(Dut *d, int valid, uint64_t data, uint8_t keep, int last) {
    d->top->s_axis_tvalid = valid;
    d->top->s_axis_tdata = data;
    d->top->s_axis_tkeep = keep;
    d->top->s_axis_tlast = last;
}

void dut_set_write_sink(Dut *d, int ready, int error) {
    d->top->wr_ready = ready;
    d->top->wr_error = error;
}

void dut_set_group_ready(Dut *d, int value) { d->top->group_ready = value; }

int dut_config_ready(Dut *d) { return d->top->cfg_ready; }
int dut_busy(Dut *d) { return d->top->busy; }
int dut_done(Dut *d) { return d->top->done; }
int dut_error(Dut *d) { return d->top->error; }
uint8_t dut_status(Dut *d) { return d->top->status; }
int dut_input_ready(Dut *d) { return d->top->s_axis_tready; }
int dut_write_valid(Dut *d) { return d->top->wr_valid; }
uint8_t dut_write_bank(Dut *d) { return d->top->wr_bank; }
uint16_t dut_write_address(Dut *d) { return d->top->wr_address; }
uint64_t dut_write_data(Dut *d) { return d->top->wr_data; }
int dut_group_valid(Dut *d) { return d->top->group_valid; }

void dut_group_data(Dut *d, uint32_t lanes[8]) {
    for (int i = 0; i < 8; ++i) lanes[i] = d->top->group_data[i];
}

int dut_group_last(Dut *d) { return d->top->group_last; }

}
