#include "Vflash_softmax.h"
#include "verilated.h"
#include "shim.h"

struct Dut {
    Vflash_softmax *t;
};

double sc_time_stamp() { return 0; }

extern "C" {

Dut *dut_new(void) {
    Dut *d = new Dut();
    d->t = new Vflash_softmax();
    return d;
}
void dut_free(Dut *d) {
    delete d->t;
    delete d;
}
void dut_eval(Dut *d) { d->t->eval(); }

void dut_set_clk(Dut *d, int v) { d->t->clk = v; }
void dut_set_rst_n(Dut *d, int v) { d->t->rst_n = v; }
void dut_set_valid(Dut *d, int v) { d->t->valid_in = v; }
void dut_set_m_in(Dut *d, uint32_t v) { d->t->m_in = v; }
void dut_set_l_in(Dut *d, uint32_t v) { d->t->l_in = v; }
void dut_set_score(Dut *d, uint32_t v) { d->t->score = v; }

int dut_valid_out(Dut *d) { return d->t->valid_out; }
uint32_t dut_m_out(Dut *d) { return d->t->m_out; }
uint32_t dut_l_out(Dut *d) { return d->t->l_out; }
uint32_t dut_p(Dut *d) { return d->t->p; }
uint32_t dut_corr(Dut *d) { return d->t->corr; }
int dut_grew(Dut *d) { return d->t->grew; }
}
