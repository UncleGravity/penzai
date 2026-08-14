#ifndef PENZAI_SECTION_RMSNORM_LOADER_SHIM_H
#define PENZAI_SECTION_RMSNORM_LOADER_SHIM_H

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
void dut_set_config(Dut *d, int valid, uint16_t rows, uint8_t tokens);
void dut_set_abort(Dut *d, int value);
void dut_set_input(Dut *d, int valid, uint64_t data, uint8_t keep, int last);
void dut_set_write_sink(Dut *d, int ready, int error);
void dut_set_group_ready(Dut *d, int value);

int dut_config_ready(Dut *d);
int dut_busy(Dut *d);
int dut_done(Dut *d);
int dut_error(Dut *d);
uint8_t dut_status(Dut *d);
int dut_input_ready(Dut *d);
int dut_write_valid(Dut *d);
uint8_t dut_write_bank(Dut *d);
uint16_t dut_write_address(Dut *d);
uint64_t dut_write_data(Dut *d);
int dut_group_valid(Dut *d);
void dut_group_data(Dut *d, uint32_t lanes[8]);
int dut_group_last(Dut *d);

#ifdef __cplusplus
}
#endif

#endif
