#include "Vnumeric_diff_top.h"
#include "verilated.h"
#include "shim.h"

struct Dut {
    Vnumeric_diff_top *t;
};

double sc_time_stamp() { return 0; }

extern "C" {

Dut *dut_new(void) {
    Dut *d = new Dut();
    d->t = new Vnumeric_diff_top();
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
void dut_set_a(Dut *d, uint32_t v) { d->t->a = v; }
void dut_set_b(Dut *d, uint32_t v) { d->t->b = v; }
void dut_set_c16(Dut *d, uint32_t v) { d->t->c16 = v; }
void dut_set_cint(Dut *d, uint32_t v) { d->t->cint = v; }
void dut_set_rin(Dut *d, const uint32_t *words16) {
    for (int i = 0; i < 16; i++) d->t->rin[i] = words16[i];
}
void dut_set_ex_x(Dut *d, uint32_t v) { d->t->ex_x = v; }
void dut_set_rc_l(Dut *d, uint32_t v) { d->t->rc_l = v; }

int dut_fadd_new_valid(Dut *d) { return d->t->fadd_new_valid; }
uint32_t dut_fadd_new_out(Dut *d) { return d->t->fadd_new_out; }
int dut_fadd_old_valid(Dut *d) { return d->t->fadd_old_valid; }
uint32_t dut_fadd_old_out(Dut *d) { return d->t->fadd_old_out; }

int dut_fmul_new_valid(Dut *d) { return d->t->fmul_new_valid; }
uint32_t dut_fmul_new_out(Dut *d) { return d->t->fmul_new_out; }
int dut_fmul_old_valid(Dut *d) { return d->t->fmul_old_valid; }
uint32_t dut_fmul_old_out(Dut *d) { return d->t->fmul_old_out; }

uint32_t dut_cvt_f16_new(Dut *d) { return d->t->cvt_f16_new_out; }
uint32_t dut_cvt_f16_old(Dut *d) { return d->t->cvt_f16_old_out; }
uint32_t dut_cvt_i2f_new(Dut *d) { return d->t->cvt_i2f_new_out; }
uint32_t dut_cvt_i2f_old(Dut *d) { return d->t->cvt_i2f_old_out; }

int dut_reduce_new_valid(Dut *d) { return d->t->reduce_new_valid; }
uint32_t dut_reduce_new_out(Dut *d) { return d->t->reduce_new_out; }
int dut_reduce_old_valid(Dut *d) { return d->t->reduce_old_valid; }
uint32_t dut_reduce_old_out(Dut *d) { return d->t->reduce_old_out; }

int dut_exp_new_valid(Dut *d) { return d->t->exp_new_valid; }
uint32_t dut_exp_new_out(Dut *d) { return d->t->exp_new_out; }
int dut_exp_old_valid(Dut *d) { return d->t->exp_old_valid; }
uint32_t dut_exp_old_out(Dut *d) { return d->t->exp_old_out; }

int dut_recip_new_valid(Dut *d) { return d->t->recip_new_valid; }
uint32_t dut_recip_new_out(Dut *d) { return d->t->recip_new_out; }
int dut_recip_old_valid(Dut *d) { return d->t->recip_old_valid; }
uint32_t dut_recip_old_out(Dut *d) { return d->t->recip_old_out; }
}
