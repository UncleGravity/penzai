#include "Vsection_swiglu.h"
#include "verilated.h"
#include "shim.h"

struct Dut {
    Vsection_swiglu *top;
};

double sc_time_stamp() { return 0; }

extern "C" {

Dut *dut_new(void) {
    Dut *d = new Dut();
    d->top = new Vsection_swiglu();
    return d;
}

void dut_free(Dut *d) {
    delete d->top;
    delete d;
}

void dut_eval(Dut *d) { d->top->eval(); }
void dut_set_clk(Dut *d, int value) { d->top->clk = value; }
void dut_set_rst_n(Dut *d, int value) { d->top->rst_n = value; }
void dut_set_abort(Dut *d, int value) { d->top->__SYM__abort = value; }
void dut_set_input(Dut *d, int valid, uint32_t gate, uint32_t up, int last) {
    d->top->in_valid = valid;
    d->top->in_gate = gate;
    d->top->in_up = up;
    d->top->in_last = last;
}
void dut_set_out_ready(Dut *d, int value) { d->top->out_ready = value; }

int dut_in_ready(Dut *d) { return d->top->in_ready; }
int dut_out_valid(Dut *d) { return d->top->out_valid; }
uint32_t dut_out_data(Dut *d) { return d->top->out_data; }
int dut_out_last(Dut *d) { return d->top->out_last; }
uint8_t dut_out_status(Dut *d) { return d->top->out_status; }

}
