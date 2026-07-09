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
int dut_s_rvalid(Dut *d) { return d->t->s_rvalid; }
uint32_t dut_s_rdata(Dut *d) { return d->t->s_rdata; }
void dut_set_s_rready(Dut *d, int v) { d->t->s_rready = v; }

int dut_reg_awvalid(Dut *d) { return d->t->reg_awvalid; }
uint32_t dut_reg_awaddr(Dut *d) { return d->t->reg_awaddr; }
void dut_set_reg_awready(Dut *d, int v) { d->t->reg_awready = v; }
int dut_reg_wvalid(Dut *d) { return d->t->reg_wvalid; }
uint32_t dut_reg_wdata(Dut *d) { return d->t->reg_wdata; }
void dut_set_reg_wready(Dut *d, int v) { d->t->reg_wready = v; }
void dut_set_reg_bvalid(Dut *d, int v) { d->t->reg_bvalid = v; }
int dut_reg_bready(Dut *d) { return d->t->reg_bready; }
int dut_reg_arvalid(Dut *d) { return d->t->reg_arvalid; }
uint32_t dut_reg_araddr(Dut *d) { return d->t->reg_araddr; }
void dut_set_reg_arready(Dut *d, int v) { d->t->reg_arready = v; }
void dut_set_reg_rdata(Dut *d, uint32_t v) { d->t->reg_rdata = v; }
void dut_set_reg_rvalid(Dut *d, int v) { d->t->reg_rvalid = v; }
int dut_reg_rready(Dut *d) { return d->t->reg_rready; }
}
