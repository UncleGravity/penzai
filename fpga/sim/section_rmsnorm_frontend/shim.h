#ifndef PENZAI_SECTION_RMSNORM_FRONTEND_SHIM_H
#define PENZAI_SECTION_RMSNORM_FRONTEND_SHIM_H

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
void dut_set_config(Dut *d, int valid, uint16_t rows, uint8_t tokens, int resident);
void dut_set_abort(Dut *d, int value);
void dut_set_stream(Dut *d, int valid, uint64_t data, uint8_t keep, int last);
void dut_set_r_write_sink(Dut *d, int ready, int error);
void dut_set_read_request_ready(Dut *d, int ready);
void dut_set_read_response(Dut *d, int valid, const uint64_t lanes[4], int error);
void dut_set_result_ready(Dut *d, int ready);

int dut_config_ready(Dut *d);
int dut_busy(Dut *d);
int dut_done(Dut *d);
int dut_error(Dut *d);
uint8_t dut_status(Dut *d);
int dut_stream_ready(Dut *d);
int dut_r_write_valid(Dut *d);
uint8_t dut_r_write_bank(Dut *d);
uint16_t dut_r_write_address(Dut *d);
uint64_t dut_r_write_data(Dut *d);
int dut_read_request_valid(Dut *d);
uint8_t dut_read_request_token(Dut *d);
uint16_t dut_read_request_group(Dut *d);
int dut_read_response_ready(Dut *d);
int dut_result_valid(Dut *d);
uint8_t dut_result_token(Dut *d);
uint8_t dut_result_max_exp(Dut *d);
uint64_t dut_result_sum_sq(Dut *d);
uint16_t dut_result_rows(Dut *d);
int dut_result_final(Dut *d);

#ifdef __cplusplus
}
#endif

#endif
