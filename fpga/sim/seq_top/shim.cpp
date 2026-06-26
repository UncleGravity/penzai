#include "Vseq_top.h"
#include "verilated.h"
#include "shim.h"

struct Dut {
    Vseq_top *t;
};

double sc_time_stamp() { return 0; }

extern "C" {

Dut *dut_new(void) {
    Dut *d = new Dut();
    d->t = new Vseq_top();
    return d;
}
void dut_free(Dut *d) {
    delete d->t;
    delete d;
}
void dut_eval(Dut *d) { d->t->eval(); }

void dut_set_clk(Dut *d, int v) { d->t->clk = v; }
void dut_set_rst_n(Dut *d, int v) { d->t->rst_n = v; }

void dut_set_s_awaddr(Dut *d, uint32_t v) { d->t->s_awaddr = v; }
void dut_set_s_awvalid(Dut *d, int v) { d->t->s_awvalid = v; }
int dut_s_awready(Dut *d) { return d->t->s_awready; }
void dut_set_s_wdata(Dut *d, uint32_t v) { d->t->s_wdata = v; }
void dut_set_s_wvalid(Dut *d, int v) { d->t->s_wvalid = v; }
int dut_s_wready(Dut *d) { return d->t->s_wready; }
int dut_s_bvalid(Dut *d) { return d->t->s_bvalid; }
void dut_set_s_bready(Dut *d, int v) { d->t->s_bready = v; }
void dut_set_s_araddr(Dut *d, uint32_t v) { d->t->s_araddr = v; }
void dut_set_s_arvalid(Dut *d, int v) { d->t->s_arvalid = v; }
int dut_s_arready(Dut *d) { return d->t->s_arready; }
uint32_t dut_s_rdata(Dut *d) { return d->t->s_rdata; }
int dut_s_rvalid(Dut *d) { return d->t->s_rvalid; }
void dut_set_s_rready(Dut *d, int v) { d->t->s_rready = v; }

uint32_t dut_reg_awaddr(Dut *d) { return d->t->reg_awaddr; }
int dut_reg_awvalid(Dut *d) { return d->t->reg_awvalid; }
void dut_set_reg_awready(Dut *d, int v) { d->t->reg_awready = v; }
uint32_t dut_reg_wdata(Dut *d) { return d->t->reg_wdata; }
int dut_reg_wvalid(Dut *d) { return d->t->reg_wvalid; }
void dut_set_reg_wready(Dut *d, int v) { d->t->reg_wready = v; }
int dut_reg_bready(Dut *d) { return d->t->reg_bready; }
void dut_set_reg_bvalid(Dut *d, int v) { d->t->reg_bvalid = v; }
uint32_t dut_reg_araddr(Dut *d) { return d->t->reg_araddr; }
int dut_reg_arvalid(Dut *d) { return d->t->reg_arvalid; }
void dut_set_reg_arready(Dut *d, int v) { d->t->reg_arready = v; }
int dut_reg_rready(Dut *d) { return d->t->reg_rready; }
void dut_set_reg_rdata(Dut *d, uint32_t v) { d->t->reg_rdata = v; }
void dut_set_reg_rvalid(Dut *d, int v) { d->t->reg_rvalid = v; }

uint64_t dut_desc_araddr(Dut *d) { return d->t->desc_araddr; }
int dut_desc_arvalid(Dut *d) { return d->t->desc_arvalid; }
void dut_set_desc_arready(Dut *d, int v) { d->t->desc_arready = v; }
int dut_desc_rready(Dut *d) { return d->t->desc_rready; }
void dut_set_desc_rdata(Dut *d, uint32_t w0, uint32_t w1, uint32_t w2, uint32_t w3) {
    d->t->desc_rdata[0] = w0;
    d->t->desc_rdata[1] = w1;
    d->t->desc_rdata[2] = w2;
    d->t->desc_rdata[3] = w3;
}
void dut_set_desc_rvalid(Dut *d, int v) { d->t->desc_rvalid = v; }
void dut_set_desc_rlast(Dut *d, int v) { d->t->desc_rlast = v; }
}
