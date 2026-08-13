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
void dut_set_abort(Dut *d, int value);
void dut_set_input(Dut *d, int valid, uint32_t gate, uint32_t up, int last);
void dut_set_out_ready(Dut *d, int value);

int dut_in_ready(Dut *d);
int dut_out_valid(Dut *d);
uint32_t dut_out_data(Dut *d);
int dut_out_last(Dut *d);
uint8_t dut_out_status(Dut *d);

#ifdef __cplusplus
}
#endif
