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

// AXI-Lite
void dut_axi_write(Dut *d, uint8_t addr, uint32_t data, int valid);
void dut_axi_read(Dut *d, uint8_t addr, int valid);
void dut_axi_idle(Dut *d);
int dut_axi_rvalid(Dut *d);
uint32_t dut_axi_rdata(Dut *d);

// AXIS streams
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
uint32_t dut_o_keep(Dut *d);
int dut_o_last(Dut *d);

#ifdef __cplusplus
}
#endif
