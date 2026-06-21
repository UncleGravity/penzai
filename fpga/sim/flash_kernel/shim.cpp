#include "Vflash_kernel.h"
#include "verilated.h"
#include "shim.h"

struct Dut {
    Vflash_kernel *t;
};

double sc_time_stamp() { return 0; }

static void set_w8(VlWide<8> &dst, const uint32_t *src) {
    for (int i = 0; i < 8; i++) dst[i] = src[i];
}
static void set_w4(VlWide<4> &dst, const uint32_t *src) {
    for (int i = 0; i < 4; i++) dst[i] = src[i];
}

extern "C" {

Dut *dut_new(void) {
    Dut *d = new Dut();
    d->t = new Vflash_kernel();
    return d;
}
void dut_free(Dut *d) {
    delete d->t;
    delete d;
}
void dut_eval(Dut *d) { d->t->eval(); }

void dut_set_clk(Dut *d, int v) { d->t->clk = v; }
void dut_set_rst_n(Dut *d, int v) { d->t->rst_n = v; }
void dut_set_start(Dut *d, int v) { d->t->start = v; }
void dut_set_config(Dut *d, uint16_t hdq, uint16_t hdv, uint16_t nh, uint16_t nhkv, uint16_t ratio, uint16_t nkv, uint16_t ntok, uint32_t scale) {
    d->t->head_dim_q = hdq;
    d->t->head_dim_v = hdv;
    d->t->n_heads = nh;
    d->t->n_head_kv = nhkv;
    d->t->head_ratio = ratio;
    d->t->n_kv = nkv;
    d->t->n_tokens = ntok;
    d->t->scale = scale;
}
int dut_busy(Dut *d) { return d->t->busy; }
int dut_done(Dut *d) { return d->t->done; }

void dut_set_q(Dut *d, const uint32_t *w8, int valid) {
    set_w8(d->t->q_tdata, w8);
    d->t->q_tvalid = valid;
}
void dut_set_k(Dut *d, const uint32_t *w4, int valid) {
    set_w4(d->t->k_tdata, w4);
    d->t->k_tvalid = valid;
}
void dut_set_v(Dut *d, const uint32_t *w4, int valid) {
    set_w4(d->t->v_tdata, w4);
    d->t->v_tvalid = valid;
}
void dut_set_mask(Dut *d, uint16_t v, int valid) {
    d->t->mask_tdata = v;
    d->t->mask_tvalid = valid;
}
void dut_set_o_ready(Dut *d, int v) { d->t->o_tready = v; }

int dut_q_ready(Dut *d) { return d->t->q_tready; }
int dut_k_ready(Dut *d) { return d->t->k_tready; }
int dut_v_ready(Dut *d) { return d->t->v_tready; }
int dut_mask_ready(Dut *d) { return d->t->mask_tready; }
int dut_o_valid(Dut *d) { return d->t->o_tvalid; }
void dut_o_data(Dut *d, uint32_t *w8) {
    for (int i = 0; i < 8; i++) w8[i] = d->t->o_tdata[i];
}
}
