#ifndef PENZAI_SECTION_RESIDUAL_ADD_SHIM_H
#define PENZAI_SECTION_RESIDUAL_ADD_SHIM_H

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
void dut_set_config(Dut *d, int valid, uint16_t rows, uint8_t tokens);
void dut_set_input(Dut *d, int valid, uint64_t data, uint8_t keep, int last);
void dut_set_read_request_ready(Dut *d, int ready);
void dut_set_read_response(Dut *d, int valid, const uint64_t lanes[4],
                           int error);
void dut_set_write_sink(Dut *d, int ready, int error);
void dut_set_output_ready(Dut *d, int ready);

int dut_config_ready(Dut *d);
int dut_busy(Dut *d);
int dut_done(Dut *d);
int dut_error(Dut *d);
uint8_t dut_status(Dut *d);
int dut_input_ready(Dut *d);

int dut_read_request_valid(Dut *d);
uint8_t dut_read_request_role(Dut *d);
uint8_t dut_read_request_token(Dut *d);
uint16_t dut_read_request_group(Dut *d);
int dut_read_response_ready(Dut *d);

int dut_write_valid(Dut *d);
uint8_t dut_write_bank(Dut *d);
uint16_t dut_write_address(Dut *d);
uint64_t dut_write_data(Dut *d);

int dut_output_valid(Dut *d);
uint64_t dut_output_data(Dut *d);
uint8_t dut_output_keep(Dut *d);
int dut_output_last(Dut *d);
uint8_t dut_debug_state(Dut *d);
uint8_t dut_debug_add_state(Dut *d);

#ifdef __cplusplus
}
#endif

#endif
