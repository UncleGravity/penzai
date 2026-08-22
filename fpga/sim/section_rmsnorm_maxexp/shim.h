#ifndef PENZAI_SECTION_RMSNORM_MAXEXP_SHIM_H
#define PENZAI_SECTION_RMSNORM_MAXEXP_SHIM_H

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
void dut_set_group(Dut *d, int valid, const uint32_t lanes[8], int error,
                   int last);
void dut_set_result_ready(Dut *d, int value);
void dut_force_summary_fatal(Dut *d, int value);

int dut_config_ready(Dut *d);
int dut_busy(Dut *d);
int dut_done(Dut *d);
int dut_error(Dut *d);
uint8_t dut_status(Dut *d);
int dut_group_ready(Dut *d);
int dut_result_valid(Dut *d);
uint8_t dut_result_token(Dut *d);
uint8_t dut_result_max_exp(Dut *d);
uint16_t dut_result_rows(Dut *d);
int dut_result_subnormal_warning(Dut *d);
int dut_result_final(Dut *d);

#ifdef __cplusplus
}
#endif

#endif
