/* C ABI over the Verilated gemm_kernel model. */
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
void dut_set_start(Dut *, int v);
void dut_set_num_q1(Dut *, int v);
void dut_set_num_rb(Dut *, int v);
void dut_set_num_rows(Dut *, uint32_t v);
void dut_set_num_cols(Dut *, int v);
void dut_set_emin(Dut *, int v);                     /* signed 8-bit window floor */
void dut_set_weight_fmt(Dut *, int v);
void dut_set_w(Dut *, const uint32_t *w, int nwords, int valid); /* ROWS*32-bit weight beat */
void dut_set_a(Dut *, uint64_t data, int valid);     /* 64-bit acts beat */
void dut_set_m_ready(Dut *, int v);

int dut_w_ready(Dut *);
int dut_a_ready(Dut *);
int dut_m_valid(Dut *);
int dut_m_last(Dut *);
int dut_m_keep(Dut *);
uint64_t dut_m_data(Dut *);
int dut_busy(Dut *);
int dut_done(Dut *);
int dut_state(Dut *);

#ifdef __cplusplus
}
#endif
