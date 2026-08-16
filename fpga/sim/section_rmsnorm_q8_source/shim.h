#ifndef PENZAI_SECTION_RMSNORM_Q8_SOURCE_SHIM_H
#define PENZAI_SECTION_RMSNORM_Q8_SOURCE_SHIM_H

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

void dut_set_gamma_config(Dut *d, int valid, uint16_t rows);
void dut_set_gamma_stream(Dut *d, int valid, uint64_t data, uint8_t keep,
                          int last);
void dut_set_run_config(Dut *d, int valid, uint16_t rows, uint8_t tokens);
void dut_set_inverse(Dut *d, int valid, uint8_t token, uint32_t data,
                     int final);
void dut_set_read_request_ready(Dut *d, int ready);
void dut_set_read_response(Dut *d, int valid, const uint64_t lanes[4],
                           int error);
void dut_set_output_ready(Dut *d, int ready);

int dut_gamma_config_ready(Dut *d);
int dut_gamma_stream_ready(Dut *d);
int dut_gamma_busy(Dut *d);
int dut_gamma_done(Dut *d);
int dut_gamma_error(Dut *d);
uint8_t dut_gamma_status(Dut *d);
int dut_gamma_valid(Dut *d);

int dut_run_config_ready(Dut *d);
int dut_busy(Dut *d);
int dut_done(Dut *d);
int dut_error(Dut *d);
uint16_t dut_status(Dut *d);
int dut_inverse_ready(Dut *d);
int dut_read_request_valid(Dut *d);
uint8_t dut_read_request_role(Dut *d);
uint8_t dut_read_request_token(Dut *d);
uint16_t dut_read_request_group(Dut *d);
int dut_read_response_ready(Dut *d);

int dut_output_valid(Dut *d);
uint64_t dut_output_data(Dut *d);
int dut_output_last(Dut *d);
uint8_t dut_output_token(Dut *d);
uint16_t dut_output_block(Dut *d);

int dut_debug_scalar_fire(Dut *d);
int dut_debug_source_done(Dut *d);
uint8_t dut_debug_wrapper_state(Dut *d);
uint8_t dut_debug_source_state(Dut *d);
uint8_t dut_debug_q8_state(Dut *d);
uint8_t dut_debug_q8_emit_index(Dut *d);
int dut_debug_q8_record_done(Dut *d);

#ifdef __cplusplus
}
#endif

#endif
