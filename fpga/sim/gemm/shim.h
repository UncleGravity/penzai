/* C ABI over the Verilated gemm_top model (increment 1: the decode datapath core).
 * Wide buses are set word-by-word, as Verilator exposes >64-bit signals as uint32_t[]. */
#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct Dut Dut;

Dut *dut_new(void);
void dut_free(Dut *);
void dut_eval(Dut *);

void dut_set_clk(Dut *, int v);
void dut_set_rst_n(Dut *, int v);
void dut_set_clear(Dut *, int v);
void dut_set_valid(Dut *, int v);
void dut_set_col_idx(Dut *, int v);                               /* accumulator column to issue */
void dut_set_read_col(Dut *, int v);                              /* accumulator column to expose */
void dut_set_emin(Dut *, int v);                                  /* signed 8-bit */
void dut_set_weight_bits(Dut *, const uint32_t *w, int nwords);   /* ROWS*32-bit (nwords = ROWS) */
void dut_set_weight_scales(Dut *, const uint32_t *w, int nwords); /* ROWS*16-bit (nwords = ROWS/2) */
void dut_set_acts(Dut *, const uint32_t *w, int nwords);          /* 256-bit (nwords = 8) */
void dut_set_act_scale(Dut *, int v);                             /* f16 bits */
void dut_set_read_row(Dut *, int v);
void dut_set_dbg_f16(Dut *, int v);
void dut_set_dbg_emit_vin(Dut *, int v);                             /* directed-sweep valid_in */
void dut_set_dbg_acc(Dut *, uint32_t lo, uint32_t mid, uint32_t hi, uint32_t top); /* 104-bit acc */
void dut_set_dbg_emin(Dut *, int v);                                 /* signed 8-bit */

uint32_t dut_acc0(Dut *);
uint32_t dut_acc1(Dut *);
uint32_t dut_acc2(Dut *);
uint32_t dut_acc3(Dut *);
uint32_t dut_emit_f32(Dut *);   /* gemm_emit(acc_sel, emin) -> fp32 bits */
uint32_t dut_dbg_sig(Dut *);    /* 12-bit, caller sign-extends */
uint32_t dut_dbg_e(Dut *);      /* 8-bit, caller sign-extends */
uint32_t dut_dbg_emit_f32(Dut *);
int dut_dbg_emit_vout(Dut *);

#ifdef __cplusplus
}
#endif
