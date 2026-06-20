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
void dut_set_x(Dut *d, uint32_t v);
void dut_set_l(Dut *d, uint32_t v);
void dut_set_dq(Dut *d, const uint32_t *words8); // 8 × f32
void dut_set_dk(Dut *d, const uint32_t *words4); // 8 × f16 packed (4 × u32)

int dut_exp_valid(Dut *d);
uint32_t dut_exp_y(Dut *d);
int dut_recip_valid(Dut *d);
uint32_t dut_recip_y(Dut *d);
int dut_dot_valid(Dut *d);
uint32_t dut_dot_sum(Dut *d);

#ifdef __cplusplus
}
#endif
