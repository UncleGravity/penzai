#include "Vgemm_kernel.h"
#include "verilated.h"
#include "shim.h"

struct Dut {
    Vgemm_kernel *t;
};

double sc_time_stamp() { return 0; }

extern "C" {

Dut *dut_new(void) {
    Dut *d = new Dut();
    d->t = new Vgemm_kernel();
    return d;
}
void dut_free(Dut *d) {
    delete d->t;
    delete d;
}
void dut_eval(Dut *d) { d->t->eval(); }

void dut_set_clk(Dut *d, int v) { d->t->clk = v; }
void dut_set_rst_n(Dut *d, int v) { d->t->rst_n = v; }
void dut_set_start(Dut *d, int v) { d->t->start_kernel = v; }
void dut_set_num_q1(Dut *d, int v) { d->t->num_q1_blocks = v; }
void dut_set_num_rb(Dut *d, int v) { d->t->num_rowblocks = v; }
void dut_set_num_cols(Dut *d, int v) { d->t->num_cols = v; }
void dut_set_emin(Dut *d, int v) { d->t->emin = v; }
void dut_set_weight_fmt(Dut *d, int v) { d->t->weight_fmt = v; }
void dut_set_w(Dut *d, const uint32_t *w, int nwords, int valid) {
    for (int i = 0; i < nwords; i++) d->t->s_axis_tdata[i] = w[i];
    d->t->s_axis_tvalid = valid;
}
void dut_set_a(Dut *d, uint64_t data, int valid) {
    d->t->s_axis_acts_tdata = data;
    d->t->s_axis_acts_tvalid = valid;
}
void dut_set_m_ready(Dut *d, int v) { d->t->m_axis_tready = v; }

int dut_w_ready(Dut *d) { return d->t->s_axis_tready; }
int dut_a_ready(Dut *d) { return d->t->s_axis_acts_tready; }
int dut_m_valid(Dut *d) { return d->t->m_axis_tvalid; }
int dut_m_last(Dut *d) { return d->t->m_axis_tlast; }
uint64_t dut_m_data(Dut *d) { return d->t->m_axis_tdata; }
int dut_busy(Dut *d) { return d->t->busy; }
int dut_done(Dut *d) { return d->t->kernel_done; }
int dut_state(Dut *d) { return d->t->dbg_state; }
}
