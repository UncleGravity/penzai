#include "Vflash_fp_top.h"
#include "verilated.h"
#include "shim.h"

struct Dut {
    Vflash_fp_top *t;
};

double sc_time_stamp() { return 0; }

extern "C" {

Dut *dut_new(void) {
    Dut *d = new Dut();
    d->t = new Vflash_fp_top();
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
void dut_set_x(Dut *d, uint32_t v) { d->t->x = v; }
void dut_set_l(Dut *d, uint32_t v) { d->t->l = v; }
void dut_set_dq(Dut *d, const uint32_t *words8) {
    for (int i = 0; i < 8; i++) d->t->dq[i] = words8[i];
}
void dut_set_dk(Dut *d, const uint32_t *words4) {
    for (int i = 0; i < 4; i++) d->t->dk[i] = words4[i];
}

int dut_exp_valid(Dut *d) { return d->t->exp_valid; }
uint32_t dut_exp_y(Dut *d) { return d->t->exp_y; }
int dut_recip_valid(Dut *d) { return d->t->recip_valid; }
uint32_t dut_recip_y(Dut *d) { return d->t->recip_y; }
int dut_dot_valid(Dut *d) { return d->t->dot_valid; }
uint32_t dut_dot_sum(Dut *d) { return d->t->dot_sum; }
}
