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
    d->t->sim_inject_q8_numeric_error = 0;
    d->t->sim_inject_p3d_scratch_error = 0;
    d->t->sim_inject_residual_numeric_error = 0;
    d->t->sim_inject_down_activation_error = 0;
    d->t->sim_hold_p3d_rd_rsp = 0;
    d->t->sim_force_scratch_abort_strobe = 0;
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
    return d->t->rootp->decode_top__DOT__u_section_ffn_pairer__DOT__out_valid_q &&
           d->t->rootp->decode_top__DOT__swiglu_in_ready;
}
int dut_dbg_internal_record_done(Dut *d) {
    return d->t->rootp->decode_top__DOT__q8_internal_record_done;
}
int dut_dbg_capture_fire(Dut *d) {
    return d->t->rootp->decode_top__DOT__q8_capture_fire;
}
int dut_dbg_replay_fire(Dut *d) {
    return d->t->rootp->decode_top__DOT__q8_buffer_m_axis_tvalid &&
           d->t->rootp->decode_top__DOT__q8_buffer_replay_healthy &&
           d->t->rootp->decode_top__DOT__kernel_acts_tready;
}
int dut_dbg_ffn_phase(Dut *d) {
    return d->t->rootp->decode_top__DOT__ffn_phase_q;
}
uint32_t dut_dbg_capture_tag(Dut *d) {
    return d->t->rootp->decode_top__DOT__ffn_capture_beat_q |
           (d->t->rootp->decode_top__DOT__ffn_capture_token_q << 3) |
           (d->t->rootp->decode_top__DOT__ffn_capture_block_q << 5);
}
uint32_t dut_dbg_replay_tag(Dut *d) {
    return d->t->rootp->decode_top__DOT__ffn_replay_beat_q |
           (d->t->rootp->decode_top__DOT__ffn_replay_token_q << 3) |
           (d->t->rootp->decode_top__DOT__ffn_replay_block_q << 5);
}
uint32_t dut_dbg_ffn_lifecycle(Dut *d) {
    auto *r = d->t->rootp;
    return (r->decode_top__DOT__scratch_section_active_q ? 1u : 0u) |
           (r->decode_top__DOT__ffn_producer_busy_q ? 2u : 0u) |
           (r->decode_top__DOT__ffn_abort_cleanup_q ? 4u : 0u) |
           (r->decode_top__DOT__scratch_rd_owner_q << 3) |
           (r->decode_top__DOT__u_section_scratch__DOT__rd_rsp_valid_q ? 0x20u : 0u) |
           (r->decode_top__DOT__u_section_ffn_pairer__DOT__orphan_q ? 0x40u : 0u) |
           (r->decode_top__DOT__u_section_ffn_pairer__DOT__busy_q ? 0x80u : 0u) |
           (r->decode_top__DOT__u_kernel__DOT__busy_q ? 0x100u : 0u) |
           (r->decode_top__DOT__u_section_gate_packer__DOT__busy_q ? 0x200u : 0u) |
           (r->decode_top__DOT__scratch_section_done_q ? 0x400u : 0u) |
           (r->decode_top__DOT__scratch_error_q ? 0x800u : 0u) |
           (r->decode_top__DOT__u_section_ffn_pairer__DOT__reorder_to_req ? 0x1000u : 0u);
}
uint32_t dut_dbg_scratch_error(Dut *d) {
    return d->t->rootp->decode_top__DOT__scratch_error_q;
}
uint32_t dut_dbg_p3d_lifecycle(Dut *d) {
    auto *r = d->t->rootp;
    return (r->decode_top__DOT__p3d_active_q ? 1u : 0u) |
           (r->decode_top__DOT__p3d_cleanup_q ? 2u : 0u) |
           (r->decode_top__DOT__p3d_kill_q ? 4u : 0u) |
           (r->decode_top__DOT__p3d_r_load_complete_q ? 8u : 0u) |
           (r->decode_top__DOT__p3d_norm_sealed_q ? 0x10u : 0u) |
           (r->decode_top__DOT__p3d_residual_started_q ? 0x20u : 0u) |
           (r->decode_top__DOT__q8_owner_q << 6) |
           (r->decode_top__DOT__scratch_rd_owner_q << 8) |
           (r->decode_top__DOT__p3d_rd_owner_q << 10) |
           (r->decode_top__DOT__u_section_scratch__DOT__rd_rsp_valid_q ?
                0x1000u : 0u);
}
uint32_t dut_dbg_p3d_launch(Dut *d) {
    auto *r = d->t->rootp;
    return (r->decode_top__DOT__p3d_section_begin_ok ? 1u : 0u) |
           (r->decode_top__DOT__p3d_leaf_start_q ? 2u : 0u) |
           (r->decode_top__DOT__p3d_leaf_start ? 4u : 0u) |
           (r->decode_top__DOT__q8_ingress_start ? 8u : 0u) |
           (r->decode_top__DOT__scratch_abort_strobe ? 0x10u : 0u) |
           (r->decode_top__DOT__rms_busy ? 0x20u : 0u);
}
uint32_t dut_dbg_shared_activation(Dut *d) {
    auto *r = d->t->rootp;
    return (r->decode_top__DOT__compute_s_axis_acts_tvalid ? 1u : 0u) |
           (r->decode_top__DOT__q8_ingress_start ? 2u : 0u) |
           (r->decode_top__DOT__kernel_start ? 4u : 0u) |
           (r->decode_top__DOT__rms_gamma_busy ? 8u : 0u) |
           (r->decode_top__DOT__rms_gamma_tready ? 0x40u : 0u);
}
uint32_t dut_dbg_legacy_q8_cfg(Dut *d) {
    auto *r = d->t->rootp;
    return (r->decode_top__DOT__legacy_section_begin_ok ? 1u : 0u) |
           (r->decode_top__DOT__legacy_q8_cfg_start_q ? 2u : 0u) |
           (r->decode_top__DOT__q8_buffer_cfg_legacy ? 4u : 0u) |
           (r->decode_top__DOT__q8_buffer_cfg_valid ? 8u : 0u) |
           (r->decode_top__DOT__q8_buffer_cfg_p3d ? 0x10u : 0u) |
           ((r->decode_top__DOT__q8_buffer_cfg_legacy &&
             r->decode_top__DOT__q8_buffer_cfg_ready)
                ? 0x20u
                : 0u);
}
uint32_t dut_dbg_p3d_q8_accounting(Dut *d) {
    auto *r = d->t->rootp;
    return (r->decode_top__DOT__rms_q8_record_accept ? 1u : 0u) |
           (r->decode_top__DOT__rms_q8_record_fire ? 2u : 0u) |
           (r->decode_top__DOT__rms_q8_final_record_fire ? 4u : 0u) |
           (r->decode_top__DOT__p3d_norm_q8_done_q ? 8u : 0u) |
           (r->decode_top__DOT__p3d_norm_seal_event ? 0x10u : 0u) |
           (r->decode_top__DOT__p3d_norm_sealed_q ? 0x20u : 0u) |
           ((r->decode_top__DOT__q8_owner_q == 1) ? 0x40u : 0u) |
           (r->decode_top__DOT__p3d_active_q ? 0x80u : 0u) |
           (r->decode_top__DOT__p3d_q8_record_count_q << 8) |
           (r->decode_top__DOT__p3d_q8_record_expected_q << 18);
}
void dut_set_gate_q8_numeric_error(Dut *d, int v) {
    d->t->sim_inject_q8_numeric_error = v;
}
void dut_set_p3d_scratch_error(Dut *d, int v) {
    d->t->sim_inject_p3d_scratch_error = v;
}
void dut_set_residual_numeric_error(Dut *d, int v) {
    d->t->sim_inject_residual_numeric_error = v;
}
void dut_set_down_activation_error(Dut *d, int v) {
    d->t->sim_inject_down_activation_error = v;
}
void dut_set_p3d_read_response_hold(Dut *d, int v) {
    d->t->sim_hold_p3d_rd_rsp = v;
}
void dut_force_scratch_abort_strobe(Dut *d, int v) {
    d->t->sim_force_scratch_abort_strobe = v;
    d->t->rootp->decode_top__DOT__scratch_abort_strobe = v;
}
int dut_axi_rvalid(Dut *d) { return d->t->s_axi_rvalid; }
uint32_t dut_axi_rdata(Dut *d) { return d->t->s_axi_rdata; }
}
