#include "Vseq_desc_reader.h"
#include "verilated.h"
#include "shim.h"

struct Dut {
    Vseq_desc_reader *t;
};

double sc_time_stamp() { return 0; }

extern "C" {

Dut *dut_new(void) {
    Dut *d = new Dut();
    d->t = new Vseq_desc_reader();
    return d;
}
void dut_free(Dut *d) {
    delete d->t;
    delete d;
}
void dut_eval(Dut *d) { d->t->eval(); }

void dut_set_clk(Dut *d, int v) { d->t->clk = v; }
void dut_set_rst_n(Dut *d, int v) { d->t->rst_n = v; }

void dut_set_desc_base(Dut *d, uint64_t v) { d->t->desc_base = v; }
void dut_set_desc_req(Dut *d, int v) { d->t->desc_req = v; }
void dut_set_desc_idx(Dut *d, uint32_t v) { d->t->desc_idx = v; }
int dut_desc_gnt(Dut *d) { return d->t->desc_gnt; }
uint32_t dut_desc_data(Dut *d, int word) { return d->t->desc_data[word]; }

uint64_t dut_araddr(Dut *d) { return d->t->m_araddr; }
int dut_arvalid(Dut *d) { return d->t->m_arvalid; }
int dut_rready(Dut *d) { return d->t->m_rready; }

void dut_set_arready(Dut *d, int v) { d->t->m_arready = v; }
void dut_set_rdata(Dut *d, uint32_t w0, uint32_t w1, uint32_t w2, uint32_t w3) {
    d->t->m_rdata[0] = w0;
    d->t->m_rdata[1] = w1;
    d->t->m_rdata[2] = w2;
    d->t->m_rdata[3] = w3;
}
void dut_set_rresp(Dut *d, int v) { d->t->m_rresp = v; }
void dut_set_rlast(Dut *d, int v) { d->t->m_rlast = v; }
void dut_set_rvalid(Dut *d, int v) { d->t->m_rvalid = v; }
}
