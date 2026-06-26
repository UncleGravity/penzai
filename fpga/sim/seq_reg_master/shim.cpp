#include "Vseq_reg_master.h"
#include "verilated.h"
#include "shim.h"

struct Dut {
    Vseq_reg_master *t;
};

double sc_time_stamp() { return 0; }

extern "C" {

Dut *dut_new(void) {
    Dut *d = new Dut();
    d->t = new Vseq_reg_master();
    return d;
}
void dut_free(Dut *d) {
    delete d->t;
    delete d;
}
void dut_eval(Dut *d) { d->t->eval(); }

void dut_set_clk(Dut *d, int v) { d->t->clk = v; }
void dut_set_rst_n(Dut *d, int v) { d->t->rst_n = v; }

void dut_set_req(Dut *d, int v) { d->t->req = v; }
void dut_set_we(Dut *d, int v) { d->t->we = v; }
void dut_set_addr(Dut *d, uint32_t v) { d->t->addr = v; }
void dut_set_wdata(Dut *d, uint32_t v) { d->t->wdata = v; }
int dut_gnt(Dut *d) { return d->t->gnt; }
uint32_t dut_rdata(Dut *d) { return d->t->rdata; }

uint32_t dut_awaddr(Dut *d) { return d->t->m_awaddr; }
int dut_awvalid(Dut *d) { return d->t->m_awvalid; }
uint32_t dut_wdata_m(Dut *d) { return d->t->m_wdata; }
int dut_wstrb(Dut *d) { return d->t->m_wstrb; }
int dut_wvalid(Dut *d) { return d->t->m_wvalid; }
int dut_bready(Dut *d) { return d->t->m_bready; }
uint32_t dut_araddr(Dut *d) { return d->t->m_araddr; }
int dut_arvalid(Dut *d) { return d->t->m_arvalid; }
int dut_rready(Dut *d) { return d->t->m_rready; }

void dut_set_awready(Dut *d, int v) { d->t->m_awready = v; }
void dut_set_wready(Dut *d, int v) { d->t->m_wready = v; }
void dut_set_bresp(Dut *d, int v) { d->t->m_bresp = v; }
void dut_set_bvalid(Dut *d, int v) { d->t->m_bvalid = v; }
void dut_set_arready(Dut *d, int v) { d->t->m_arready = v; }
void dut_set_rdata_m(Dut *d, uint32_t v) { d->t->m_rdata = v; }
void dut_set_rresp(Dut *d, int v) { d->t->m_rresp = v; }
void dut_set_rvalid(Dut *d, int v) { d->t->m_rvalid = v; }
}
