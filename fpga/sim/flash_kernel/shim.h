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
void dut_set_start(Dut *d, int v);
void dut_set_config(Dut *d, uint16_t hdq, uint16_t hdv, uint16_t nh, uint16_t nhkv, uint16_t ratio, uint16_t nkv, uint16_t ntok, uint32_t scale);
int dut_busy(Dut *d);
int dut_done(Dut *d);

void dut_set_q(Dut *d, const uint32_t *w8, int valid);
void dut_set_k(Dut *d, const uint32_t *w4, int valid);
void dut_set_v(Dut *d, const uint32_t *w4, int valid);
void dut_set_mask(Dut *d, uint16_t v, int valid);
void dut_set_o_ready(Dut *d, int v);

int dut_q_ready(Dut *d);
int dut_k_ready(Dut *d);
int dut_v_ready(Dut *d);
int dut_mask_ready(Dut *d);
int dut_o_valid(Dut *d);
void dut_o_data(Dut *d, uint32_t *w8); // packed 256-bit O beat: 8 × f32

#ifdef __cplusplus
}
#endif
