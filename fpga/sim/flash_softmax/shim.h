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
void dut_set_valid(Dut *d, int v);
void dut_set_m_in(Dut *d, uint32_t v);
void dut_set_l_in(Dut *d, uint32_t v);
void dut_set_score(Dut *d, uint32_t v);

int dut_valid_out(Dut *d);
uint32_t dut_m_out(Dut *d);
uint32_t dut_l_out(Dut *d);
uint32_t dut_p(Dut *d);
uint32_t dut_corr(Dut *d);
int dut_grew(Dut *d);

#ifdef __cplusplus
}
#endif
