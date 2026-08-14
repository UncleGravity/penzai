#ifndef PENZAI_SECTION_RMSNORM_INV_SHIM_H
#define PENZAI_SECTION_RMSNORM_INV_SHIM_H

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
void dut_set_config(Dut *d, int valid, uint16_t rows, uint8_t tokens,
                    uint32_t eps);
void dut_set_abort(Dut *d, int value);
void dut_set_record(Dut *d, int valid, uint8_t token, uint8_t max_exp,
                    uint64_t sum_sq, uint16_t rows, int final);
void dut_set_result_ready(Dut *d, int value);

int dut_config_ready(Dut *d);
int dut_busy(Dut *d);
int dut_done(Dut *d);
int dut_error(Dut *d);
uint8_t dut_status(Dut *d);
int dut_record_ready(Dut *d);
int dut_result_valid(Dut *d);
uint8_t dut_result_token(Dut *d);
uint32_t dut_result_inv_rms(Dut *d);
int dut_result_final(Dut *d);

#ifdef __cplusplus
}
#endif

#endif
