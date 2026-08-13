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
void dut_set_gate(Dut *d, int valid, const uint32_t lanes[8], int last,
                  uint8_t token, uint16_t block, uint8_t group);
void dut_set_read_request_ready(Dut *d, int value);
void dut_set_read_response(Dut *d, int valid, const uint32_t lanes[8], int error);
void dut_set_output_ready(Dut *d, int value);

int dut_start_ready(Dut *d);
int dut_busy(Dut *d);
int dut_done(Dut *d);
int dut_error(Dut *d);
int dut_gate_ready(Dut *d);
int dut_read_request_valid(Dut *d);
uint8_t dut_read_request_role(Dut *d);
uint8_t dut_read_request_token(Dut *d);
uint16_t dut_read_request_group(Dut *d);
int dut_read_response_ready(Dut *d);
int dut_output_valid(Dut *d);
uint32_t dut_output_gate(Dut *d);
uint32_t dut_output_up(Dut *d);
int dut_output_last(Dut *d);

#ifdef __cplusplus
}
#endif
