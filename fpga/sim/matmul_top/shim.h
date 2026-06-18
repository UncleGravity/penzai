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
void dut_set_axi_write(Dut *d, uint8_t addr, uint32_t data, int valid);
void dut_set_axi_idle(Dut *d);
void dut_set_w(Dut *d, int port, const uint32_t *w, int valid);
void dut_set_a(Dut *d, uint64_t data, int valid);
void dut_set_m_ready(Dut *d, int v);

int dut_w_ready(Dut *d, int port);
int dut_a_ready(Dut *d);
int dut_m_valid(Dut *d);
int dut_m_last(Dut *d);
uint64_t dut_m_data(Dut *d);

#ifdef __cplusplus
}
#endif
