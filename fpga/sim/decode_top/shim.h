/* C ABI over the Verilated decode_top model. */
#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct Dut Dut;

Dut *dut_new(void);
void dut_free(Dut *);
void dut_eval(Dut *);

void dut_set_clk(Dut *, int v);
void dut_set_rst_n(Dut *, int v);

void dut_set_axi_write(Dut *, uint8_t addr, uint32_t data, int valid);
void dut_set_axi_read(Dut *, uint8_t addr, int valid);
void dut_set_axi_idle(Dut *);
int dut_axi_rvalid(Dut *);
uint32_t dut_axi_rdata(Dut *);

void dut_set_w(Dut *, int port, const uint32_t *w, int valid); /* 128-bit weight port beat */
void dut_set_a(Dut *, uint64_t data, int valid, int last);
void dut_set_m_ready(Dut *, int v);

int dut_w_ready(Dut *, int port);
int dut_a_ready(Dut *);
int dut_m_valid(Dut *);
int dut_m_last(Dut *);
int dut_m_keep(Dut *);
uint64_t dut_m_data(Dut *);

int dut_dbg_scratch_read_fire(Dut *);
int dut_dbg_swiglu_input_fire(Dut *);
int dut_dbg_internal_record_done(Dut *);
int dut_dbg_capture_fire(Dut *);
int dut_dbg_replay_fire(Dut *);
int dut_dbg_ffn_phase(Dut *);
uint32_t dut_dbg_capture_tag(Dut *);
uint32_t dut_dbg_replay_tag(Dut *);
uint32_t dut_dbg_ffn_lifecycle(Dut *);
uint32_t dut_dbg_scratch_error(Dut *);
uint32_t dut_dbg_p3d_lifecycle(Dut *);
uint32_t dut_dbg_p3d_launch(Dut *);
uint32_t dut_dbg_shared_activation(Dut *);
uint32_t dut_dbg_legacy_q8_cfg(Dut *);
uint32_t dut_dbg_p3d_q8_accounting(Dut *);
uint32_t dut_dbg_control_boundaries(Dut *);
void dut_set_gate_q8_numeric_error(Dut *, int);
void dut_set_p3d_scratch_error(Dut *, int);
void dut_set_residual_numeric_error(Dut *, int);
void dut_set_down_activation_error(Dut *, int);
void dut_set_p3d_read_response_hold(Dut *, int);
void dut_force_scratch_abort_strobe(Dut *, int);
void dut_set_inactive_norm_owner(Dut *, int);
void dut_set_inactive_residual_owner(Dut *, int);
void dut_arm_ownerless_response(Dut *, uint32_t route, int error);
uint32_t dut_ownerless_response_result(Dut *);

#ifdef __cplusplus
}
#endif
