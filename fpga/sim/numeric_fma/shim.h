#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct Dut Dut;

Dut *dut_new(void);
void dut_free(Dut *d);
void dut_eval(Dut *d);

void dut_set_clk(Dut *d, int v);
void dut_set_rst_n(Dut *d, int v);
void dut_set_clear(Dut *d, int v);
void dut_set_valid(Dut *d, int v);
void dut_set_ws_sig(Dut *d, uint32_t v);
void dut_set_as_sig(Dut *d, uint32_t v);
void dut_set_p_exp(Dut *d, uint32_t v);
void dut_set_emin(Dut *d, uint32_t v);
void dut_set_s_sum(Dut *d, uint32_t v);

uint32_t dut_acc0(Dut *d);
uint32_t dut_acc1(Dut *d);
uint32_t dut_acc2(Dut *d);
uint32_t dut_acc3(Dut *d);

#ifdef __cplusplus
}
#endif
