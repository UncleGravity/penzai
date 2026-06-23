#include "Vgemm_top.h"
#include "verilated.h"
#include "shim.h"

struct Dut {
    Vgemm_top *t;
};

double sc_time_stamp() { return 0; }

extern "C" {

Dut *dut_new(void) {
    Dut *d = new Dut();
    d->t = new Vgemm_top();
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
void dut_set_col_idx(Dut *d, int v) { d->t->col_idx = v; }
void dut_set_read_col(Dut *d, int v) { d->t->read_col = v; }
void dut_set_emin(Dut *d, int v) { d->t->emin = v; }
void dut_set_weight_bits(Dut *d, const uint32_t *w, int nwords) {
    for (int i = 0; i < nwords; i++) d->t->weight_bits_flat[i] = w[i];
}
void dut_set_weight_scales(Dut *d, const uint32_t *w, int nwords) {
    for (int i = 0; i < nwords; i++) d->t->weight_scales_flat[i] = w[i];
}
void dut_set_acts(Dut *d, const uint32_t *w, int nwords) {
    for (int i = 0; i < nwords; i++) d->t->acts_packed[i] = w[i];
}
void dut_set_act_scale(Dut *d, int v) { d->t->act_scale = v; }
void dut_set_read_row(Dut *d, int v) { d->t->read_row = v; }
void dut_set_dbg_f16(Dut *d, int v) { d->t->dbg_f16 = v; }
void dut_set_dbg_emit_vin(Dut *d, int v) { d->t->dbg_emit_vin = v; }
void dut_set_dbg_acc(Dut *d, uint32_t lo, uint32_t mid, uint32_t hi, uint32_t top) {
    d->t->dbg_acc0 = lo;
    d->t->dbg_acc1 = mid;
    d->t->dbg_acc2 = hi;
    d->t->dbg_acc3 = top;
}
void dut_set_dbg_emin(Dut *d, int v) { d->t->dbg_emin = v; }

uint32_t dut_acc0(Dut *d) { return d->t->acc0; }
uint32_t dut_acc1(Dut *d) { return d->t->acc1; }
uint32_t dut_acc2(Dut *d) { return d->t->acc2; }
uint32_t dut_acc3(Dut *d) { return d->t->acc3; }
uint32_t dut_dbg_sig(Dut *d) { return d->t->dbg_sig; }
uint32_t dut_dbg_e(Dut *d) { return d->t->dbg_e; }
uint32_t dut_dbg_emit_f32(Dut *d) { return d->t->dbg_emit_f32; }
int dut_dbg_emit_vout(Dut *d) { return d->t->dbg_emit_vout; }
}
