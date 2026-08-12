#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct Dut Dut;

Dut *dut_new(void);
void dut_free(Dut *d);
void dut_eval(Dut *d);

void dut_set_clk(Dut *d, int value);
void dut_set_rst_n(Dut *d, int value);
void dut_set_input(Dut *d, int valid, uint32_t data, int last);
void dut_set_out_ready(Dut *d, int value);

int dut_in_ready(Dut *d);
int dut_out_valid(Dut *d);
uint8_t dut_out_quant(Dut *d, uint32_t index);
uint16_t dut_out_scale(Dut *d);
uint8_t dut_out_status(Dut *d);

#ifdef __cplusplus
}
#endif
