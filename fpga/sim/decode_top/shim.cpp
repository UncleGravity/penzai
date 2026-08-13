#include "Vdecode_top.h"
#include "Vdecode_top___024root.h"
#include "verilated.h"
#include "shim.h"

struct Dut {
    Vdecode_top *t;
};

double sc_time_stamp() { return 0; }

static void set_wide128(VlWide<4> &dst, const uint32_t *src) {
    for (int i = 0; i < 4; i++) dst[i] = src[i];
}

extern "C" {

Dut *dut_new(void) {
    Dut *d = new Dut();
    d->t = new Vdecode_top();
    return d;
}
void dut_free(Dut *d) {
    delete d->t;
    delete d;
}
void dut_eval(Dut *d) { d->t->eval(); }

void dut_set_clk(Dut *d, int v) { d->t->s_axi_aclk = v; }
void dut_set_rst_n(Dut *d, int v) { d->t->s_axi_aresetn = v; }

void dut_set_axi_write(Dut *d, uint8_t addr, uint32_t data, int valid) {
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
void dut_set_axi_read(Dut *d, uint8_t addr, int valid) {
    d->t->s_axi_awvalid = 0;
    d->t->s_axi_wvalid = 0;
    d->t->s_axi_bready = 1;
    d->t->s_axi_araddr = addr;
    d->t->s_axi_arprot = 0;
    d->t->s_axi_arvalid = valid;
    d->t->s_axi_rready = 1;
}
void dut_set_axi_idle(Dut *d) {
    d->t->s_axi_awvalid = 0;
    d->t->s_axi_wvalid = 0;
    d->t->s_axi_bready = 1;
    d->t->s_axi_arvalid = 0;
    d->t->s_axi_rready = 1;
}

void dut_set_w(Dut *d, int port, const uint32_t *w, int valid) {
    switch (port) {
    case 0:
        set_wide128(d->t->s_axis_w0_tdata, w);
        d->t->s_axis_w0_tkeep = 0xffff;
        d->t->s_axis_w0_tvalid = valid;
        d->t->s_axis_w0_tlast = 0;
        break;
    case 1:
        set_wide128(d->t->s_axis_w1_tdata, w);
        d->t->s_axis_w1_tkeep = 0xffff;
        d->t->s_axis_w1_tvalid = valid;
        d->t->s_axis_w1_tlast = 0;
        break;
    case 2:
        set_wide128(d->t->s_axis_w2_tdata, w);
        d->t->s_axis_w2_tkeep = 0xffff;
        d->t->s_axis_w2_tvalid = valid;
        d->t->s_axis_w2_tlast = 0;
        break;
    case 3:
        set_wide128(d->t->s_axis_w3_tdata, w);
        d->t->s_axis_w3_tkeep = 0xffff;
        d->t->s_axis_w3_tvalid = valid;
        d->t->s_axis_w3_tlast = 0;
        break;
    default:
        break;
    }
}

void dut_set_a(Dut *d, uint64_t data, int valid, int last) {
    d->t->s_axis_acts_tdata = data;
    d->t->s_axis_acts_tkeep = 0xff;
    d->t->s_axis_acts_tvalid = valid;
    d->t->s_axis_acts_tlast = last;
}
void dut_set_m_ready(Dut *d, int v) { d->t->m_axis_tready = v; }

int dut_w_ready(Dut *d, int port) {
    switch (port) {
    case 0: return d->t->s_axis_w0_tready;
    case 1: return d->t->s_axis_w1_tready;
    case 2: return d->t->s_axis_w2_tready;
    case 3: return d->t->s_axis_w3_tready;
    default: return 0;
    }
}
int dut_a_ready(Dut *d) { return d->t->s_axis_acts_tready; }
int dut_m_valid(Dut *d) { return d->t->m_axis_tvalid; }
int dut_m_last(Dut *d) { return d->t->m_axis_tlast; }
int dut_m_keep(Dut *d) { return d->t->m_axis_tkeep; }
uint64_t dut_m_data(Dut *d) { return d->t->m_axis_tdata; }
int dut_dbg_scratch_read_fire(Dut *d) {
    return d->t->rootp->decode_top__DOT__scratch_rd_req_valid &&
           d->t->rootp->decode_top__DOT__scratch_rd_req_ready;
}
int dut_dbg_swiglu_input_fire(Dut *d) {
    return d->t->rootp->decode_top__DOT__scratch_consumer_state_q == 5 &&
           d->t->rootp->decode_top__DOT__swiglu_in_ready;
}
int dut_dbg_internal_record_done(Dut *d) {
    return d->t->rootp->decode_top__DOT__u_q8_ingress__DOT__state == 5 &&
           d->t->rootp->decode_top__DOT__u_q8_ingress__DOT__emit_index == 4 &&
           d->t->rootp->decode_top__DOT__native_acts_tready;
}
int dut_dbg_q8_state(Dut *d) {
    return d->t->rootp->decode_top__DOT__u_q8_ingress__DOT__state;
}
int dut_dbg_q8_scalar_index(Dut *d) {
    return d->t->rootp->decode_top__DOT__u_q8_ingress__DOT__scalar_index;
}
int dut_dbg_q8_emit_index(Dut *d) {
    return d->t->rootp->decode_top__DOT__u_q8_ingress__DOT__emit_index;
}
int dut_dbg_q8_quantizer_state(Dut *d) {
    return d->t->rootp->decode_top__DOT__u_q8_ingress__DOT__u_quantizer__DOT__state;
}
uint32_t dut_dbg_q8_handshakes(Dut *d) {
    return (d->t->s_axis_acts_tready ? 1u : 0u) |
           (d->t->rootp->decode_top__DOT__u_q8_ingress__DOT__internal_source_accept ? 2u : 0u) |
           (d->t->rootp->decode_top__DOT__native_acts_tvalid ? 4u : 0u) |
           (d->t->rootp->decode_top__DOT__q8_internal_record_done ? 8u : 0u) |
           (d->t->rootp->decode_top__DOT__q8_activation_abort ? 16u : 0u) |
           (d->t->rootp->decode_top__DOT__q8_ingress_abort ? 0x80000000u : 0u);
}
int dut_dbg_scratch_consumer_state(Dut *d) {
    return d->t->rootp->decode_top__DOT__scratch_consumer_state_q;
}
uint32_t dut_dbg_scratch_total_blocks(Dut *d) {
    return d->t->rootp->decode_top__DOT__scratch_consumer_total_blocks_q;
}
void dut_dbg_seed_scratch_roles(Dut *d, uint32_t rows, uint32_t tokens) {
    // Test-only metadata setup for production-width arithmetic and lifecycle
    // relaunch checks. The ordinary FFN path populates it through real writes.
    d->t->rootp->decode_top__DOT__scratch_valid_q = 0x6;
    d->t->rootp->decode_top__DOT__scratch_valid_rows_q[1] = rows;
    d->t->rootp->decode_top__DOT__scratch_valid_rows_q[2] = rows;
    d->t->rootp->decode_top__DOT__scratch_valid_tokens_q[1] = tokens;
    d->t->rootp->decode_top__DOT__scratch_valid_tokens_q[2] = tokens;
}
int dut_axi_rvalid(Dut *d) { return d->t->s_axi_rvalid; }
uint32_t dut_axi_rdata(Dut *d) { return d->t->s_axi_rdata; }
}
