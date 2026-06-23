#include "Vfma_top.h"
#include "verilated.h"
#include "shim.h"

struct Dut {
    Vfma_top *t;
};

double sc_time_stamp() { return 0; }

extern "C" {

Dut *dut_new(void) {
    Dut *d = new Dut();
    d->t = new Vfma_top();
    return d;
}
void dut_free(Dut *d) {
    delete d->t;
    delete d;
}
void dut_eval(Dut *d) { d->t->eval(); }

void dut_set_clk(Dut *d, int v) { d->t->clk = v; }
void dut_set_rst_n(Dut *d, int v) { d->t->rst_n = v; }
void dut_set_clear(Dut *d, int v) { d->t->clear = v; }
void dut_set_valid(Dut *d, int v) { d->t->valid_in = v; }
void dut_set_ws_sig(Dut *d, uint32_t v) { d->t->ws_sig = v; }
void dut_set_as_sig(Dut *d, uint32_t v) { d->t->as_sig = v; }
void dut_set_p_exp(Dut *d, uint32_t v) { d->t->p_exp = v; }
void dut_set_emin(Dut *d, uint32_t v) { d->t->emin = v; }
void dut_set_s_sum(Dut *d, uint32_t v) { d->t->s_sum = v; }

uint32_t dut_acc0(Dut *d) { return d->t->acc0; }
uint32_t dut_acc1(Dut *d) { return d->t->acc1; }
uint32_t dut_acc2(Dut *d) { return d->t->acc2; }
uint32_t dut_acc3(Dut *d) { return d->t->acc3; }
}
