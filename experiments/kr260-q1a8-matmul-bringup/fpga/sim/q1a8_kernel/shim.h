/* C ABI over the Verilated q1a8_kernel model, so the Zig testbench can drive
 * it without touching C++. One opaque handle; typed accessors per port. */
#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct Dut Dut;

Dut *dut_new(void);
void dut_free(Dut *);
void dut_eval(Dut *);

/* inputs */
void dut_set_clk(Dut *, int v);
void dut_set_rst_n(Dut *, int v);
void dut_set_start(Dut *, int v);
void dut_set_num_q1(Dut *, int v);
void dut_set_num_rb(Dut *, int v);
void dut_set_w(Dut *, uint64_t data, int valid);     /* weights stream */
void dut_set_a(Dut *, uint64_t data, int valid);     /* acts stream */
void dut_set_m_ready(Dut *, int v);                  /* results sink ready */

/* outputs */
int dut_w_ready(Dut *);
int dut_a_ready(Dut *);
int dut_m_valid(Dut *);
int dut_m_last(Dut *);
uint64_t dut_m_data(Dut *);
int dut_busy(Dut *);
int dut_done(Dut *);
int dut_state(Dut *);

#ifdef __cplusplus
}
#endif
