#ifndef PENZAI_SECTION_RESIDUAL_ADD_RNE_SHIM_H
#define PENZAI_SECTION_RESIDUAL_ADD_RNE_SHIM_H

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
void dut_set_input(Dut *d, int valid, uint32_t a, uint32_t b);
void dut_set_result_ready(Dut *d, int value);

int dut_busy(Dut *d);
int dut_input_ready(Dut *d);
int dut_result_valid(Dut *d);
uint32_t dut_result_data(Dut *d);
uint8_t dut_result_status(Dut *d);
uint8_t dut_debug_state(Dut *d);
uint32_t dut_debug_operand_a(Dut *d);
uint32_t dut_debug_operand_b(Dut *d);
uint8_t dut_debug_align_distance(Dut *d);
uint16_t dut_debug_exponent(Dut *d);
uint32_t dut_debug_small_ext(Dut *d);
uint32_t dut_debug_magnitude(Dut *d);

#ifdef __cplusplus
}
#endif

#endif
