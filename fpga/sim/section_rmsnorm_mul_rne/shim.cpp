#include "Vsection_rmsnorm_mul_rne.h"
#include "verilated.h"
#include "shim.h"

struct Dut {
    Vsection_rmsnorm_mul_rne *top;
};

double sc_time_stamp() { return 0; }

extern "C" {

Dut *dut_new(void) {
    Dut *d = new Dut();
    d->top = new Vsection_rmsnorm_mul_rne();
    return d;
}

void dut_free(Dut *d) {
    delete d->top;
    delete d;
}

void dut_eval(Dut *d) { d->top->eval(); }
void dut_set_clk(Dut *d, int value) { d->top->clk = value; }
void dut_set_rst_n(Dut *d, int value) { d->top->rst_n = value; }
void dut_set_abort(Dut *d, int value) { d->top->abort_run = value; }

void dut_set_input(Dut *d, int valid, uint32_t a, uint32_t b) {
    d->top->s_valid = valid;
    d->top->s_a = a;
    d->top->s_b = b;
}

void dut_set_result_ready(Dut *d, int value) {
    d->top->result_ready = value;
}

int dut_busy(Dut *d) { return d->top->busy; }
int dut_input_ready(Dut *d) { return d->top->s_ready; }
int dut_result_valid(Dut *d) { return d->top->result_valid; }
uint32_t dut_result_data(Dut *d) { return d->top->result_data; }
uint8_t dut_result_status(Dut *d) { return d->top->result_status; }

}
