#include "Vseq_core.h"
#include "verilated.h"
#include "shim.h"

struct Dut {
    Vseq_core *t;
};

double sc_time_stamp() { return 0; }

extern "C" {

Dut *dut_new(void) {
    Dut *d = new Dut();
    d->t = new Vseq_core();
    return d;
}
void dut_free(Dut *d) {
    delete d->t;
    delete d;
}
void dut_eval(Dut *d) { d->t->eval(); }

void dut_set_clk(Dut *d, int v) { d->t->clk = v; }
void dut_set_rst_n(Dut *d, int v) { d->t->rst_n = v; }

void dut_set_go(Dut *d, int v) { d->t->go = v; }
void dut_set_desc_count(Dut *d, uint32_t v) { d->t->desc_count = v; }
void dut_set_desc_gnt(Dut *d, int v) { d->t->desc_gnt = v; }
void dut_set_desc_data(Dut *d, uint32_t w0, uint32_t w1, uint32_t w2, uint32_t w3) {
    // 128-bit input: Verilator lays it out little-endian as uint32_t[4].
    d->t->desc_data[0] = w0;
    d->t->desc_data[1] = w1;
    d->t->desc_data[2] = w2;
    d->t->desc_data[3] = w3;
}
void dut_set_reg_gnt(Dut *d, int v) { d->t->reg_gnt = v; }
void dut_set_reg_rdata(Dut *d, uint32_t v) { d->t->reg_rdata = v; }

int dut_busy(Dut *d) { return d->t->busy; }
int dut_done(Dut *d) { return d->t->done; }
int dut_err_timeout(Dut *d) { return d->t->err_timeout; }
int dut_err_watchdog(Dut *d) { return d->t->err_watchdog; }
uint32_t dut_err_index(Dut *d) { return d->t->err_index; }
int dut_desc_req(Dut *d) { return d->t->desc_req; }
uint32_t dut_desc_idx(Dut *d) { return d->t->desc_idx; }
int dut_reg_req(Dut *d) { return d->t->reg_req; }
int dut_reg_we(Dut *d) { return d->t->reg_we; }
uint32_t dut_reg_addr(Dut *d) { return d->t->reg_addr; }
uint32_t dut_reg_wdata(Dut *d) { return d->t->reg_wdata; }
}
