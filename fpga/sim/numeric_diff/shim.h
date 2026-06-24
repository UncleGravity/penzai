#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct Dut Dut;

Dut *dut_new(void);
void dut_free(Dut *d);
void dut_eval(Dut *d);

void dut_set_clk(Dut *d, int v);
void dut_set_rst_n(Dut *d, int v);
void dut_set_valid(Dut *d, int v);
void dut_set_a(Dut *d, uint32_t v);
void dut_set_b(Dut *d, uint32_t v);
void dut_set_c16(Dut *d, uint32_t v);            // 16-bit f16
void dut_set_cint(Dut *d, uint32_t v);           // 14-bit signed int
void dut_set_rin(Dut *d, const uint32_t *words16); // 16 × f32
void dut_set_ex_x(Dut *d, uint32_t v);           // exp input
void dut_set_rc_l(Dut *d, uint32_t v);           // recip input

int dut_fadd_new_valid(Dut *d);
uint32_t dut_fadd_new_out(Dut *d);
int dut_fadd_old_valid(Dut *d);
uint32_t dut_fadd_old_out(Dut *d);

int dut_fmul_new_valid(Dut *d);
uint32_t dut_fmul_new_out(Dut *d);
int dut_fmul_old_valid(Dut *d);
uint32_t dut_fmul_old_out(Dut *d);

uint32_t dut_cvt_f16_new(Dut *d);
uint32_t dut_cvt_f16_old(Dut *d);
uint32_t dut_cvt_i2f_new(Dut *d);
uint32_t dut_cvt_i2f_old(Dut *d);
uint32_t dut_cvt_bf16_narrow(Dut *d);   // cvt_f32_bf16(a), 16-bit
uint32_t dut_cvt_bf16_widen(Dut *d);    // cvt_bf16_f32(c16)

int dut_reduce_new_valid(Dut *d);
uint32_t dut_reduce_new_out(Dut *d);
int dut_reduce_old_valid(Dut *d);
uint32_t dut_reduce_old_out(Dut *d);

int dut_exp_new_valid(Dut *d);
uint32_t dut_exp_new_out(Dut *d);
int dut_exp_old_valid(Dut *d);
uint32_t dut_exp_old_out(Dut *d);

int dut_recip_new_valid(Dut *d);
uint32_t dut_recip_new_out(Dut *d);
int dut_recip_old_valid(Dut *d);
uint32_t dut_recip_old_out(Dut *d);

#ifdef __cplusplus
}
#endif
