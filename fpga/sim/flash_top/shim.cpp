#include "Vflash_top.h"
#include "verilated.h"
#include "shim.h"

struct Dut {
    Vflash_top *t;
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
    d->t = new Vflash_top();
    return d;
}
void dut_free(Dut *d) {
    delete d->t;
    delete d;
}
void dut_eval(Dut *d) { d->t->eval(); }

void dut_set_clk(Dut *d, int v) { d->t->s_axi_aclk = v; }
void dut_set_rst_n(Dut *d, int v) { d->t->s_axi_aresetn = v; }

void dut_axi_write(Dut *d, uint8_t addr, uint32_t data, int valid) {
    d->t->s_axi_awaddr = addr;
    d->t->s_axi_awprot = 0;
    d->t->s_axi_awvalid = valid;
    d->t->s_axi_wdata = data;
    d->t->s_axi_wstrb = 0xf;
    d->t->s_axi_wvalid = valid;
    d->t->s_axi_bready = 1;
    d->t->s_axi_araddr = 0;
    d->t->s_axi_arprot = 0;
    d->t->s_axi_arvalid = 0;
    d->t->s_axi_rready = 1;
}
void dut_axi_read(Dut *d, uint8_t addr, int valid) {
    d->t->s_axi_awvalid = 0;
    d->t->s_axi_wvalid = 0;
    d->t->s_axi_bready = 1;
    d->t->s_axi_araddr = addr;
    d->t->s_axi_arprot = 0;
    d->t->s_axi_arvalid = valid;
    d->t->s_axi_rready = 1;
}
void dut_axi_idle(Dut *d) {
    d->t->s_axi_awvalid = 0;
    d->t->s_axi_wvalid = 0;
    d->t->s_axi_bready = 1;
    d->t->s_axi_arvalid = 0;
    d->t->s_axi_rready = 1;
}
int dut_axi_rvalid(Dut *d) { return d->t->s_axi_rvalid; }
uint32_t dut_axi_rdata(Dut *d) { return d->t->s_axi_rdata; }

void dut_set_q(Dut *d, const uint32_t *w8, int valid) {
    set_w8(d->t->s_axis_q_tdata, w8);
    d->t->s_axis_q_tvalid = valid;
    // Sideband the DMA path would drive; the kernel counts beats from shape regs,
    // so tie TKEEP all-valid and TLAST low (cosim doesn't model packet framing).
    d->t->s_axis_q_tkeep = 0xFFFFFFFFu;
    d->t->s_axis_q_tlast = 0;
}
void dut_set_k(Dut *d, const uint32_t *w4, int valid) {
    set_w4(d->t->s_axis_k_tdata, w4);
    d->t->s_axis_k_tvalid = valid;
    d->t->s_axis_k_tkeep = 0xFFFFu;
    d->t->s_axis_k_tlast = 0;
}
void dut_set_v(Dut *d, const uint32_t *w4, int valid) {
    set_w4(d->t->s_axis_v_tdata, w4);
    d->t->s_axis_v_tvalid = valid;
    d->t->s_axis_v_tkeep = 0xFFFFu;
    d->t->s_axis_v_tlast = 0;
}
void dut_set_mask(Dut *d, uint16_t v, int valid) {
    d->t->s_axis_mask_tdata = v;
    d->t->s_axis_mask_tvalid = valid;
    d->t->s_axis_mask_tkeep = 0x3u;
    d->t->s_axis_mask_tlast = 0;
}
void dut_set_o_ready(Dut *d, int v) { d->t->m_axis_o_tready = v; }

int dut_q_ready(Dut *d) { return d->t->s_axis_q_tready; }
int dut_k_ready(Dut *d) { return d->t->s_axis_k_tready; }
int dut_v_ready(Dut *d) { return d->t->s_axis_v_tready; }
int dut_mask_ready(Dut *d) { return d->t->s_axis_mask_tready; }
int dut_o_valid(Dut *d) { return d->t->m_axis_o_tvalid; }
void dut_o_data(Dut *d, uint32_t *w8) {
    for (int i = 0; i < 8; i++) w8[i] = d->t->m_axis_o_tdata[i];
}
uint32_t dut_o_keep(Dut *d) { return d->t->m_axis_o_tkeep; }
int dut_o_last(Dut *d) { return d->t->m_axis_o_tlast; }
}
