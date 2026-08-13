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
void dut_set_start(Dut *d, int valid, uint8_t tokens, uint16_t blocks);
void dut_set_abort(Dut *d, int value);
void dut_set_input(Dut *d, int valid, uint64_t data, uint8_t keep, int last);
void dut_set_output_ready(Dut *d, int value);

int dut_start_ready(Dut *d);
int dut_busy(Dut *d);
int dut_done(Dut *d);
int dut_error(Dut *d);
int dut_input_ready(Dut *d);
int dut_output_valid(Dut *d);
void dut_output_data(Dut *d, uint32_t lanes[8]);
int dut_output_last(Dut *d);
uint8_t dut_output_token(Dut *d);
uint16_t dut_output_block(Dut *d);
uint8_t dut_output_group(Dut *d);

#ifdef __cplusplus
}
#endif
