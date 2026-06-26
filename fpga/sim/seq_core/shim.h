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

// control inputs
void dut_set_go(Dut *d, int v);
void dut_set_desc_count(Dut *d, uint32_t v);
// descriptor-port response inputs
void dut_set_desc_gnt(Dut *d, int v);
void dut_set_desc_data(Dut *d, uint32_t w0, uint32_t w1, uint32_t w2, uint32_t w3);
// register-port response inputs
void dut_set_reg_gnt(Dut *d, int v);
void dut_set_reg_rdata(Dut *d, uint32_t v);

// status outputs
int dut_busy(Dut *d);
int dut_done(Dut *d);
int dut_err_timeout(Dut *d);
uint32_t dut_err_index(Dut *d);
// descriptor-port request outputs
int dut_desc_req(Dut *d);
uint32_t dut_desc_idx(Dut *d);
// register-port request outputs
int dut_reg_req(Dut *d);
int dut_reg_we(Dut *d);
uint32_t dut_reg_addr(Dut *d);
uint32_t dut_reg_wdata(Dut *d);

#ifdef __cplusplus
}
#endif
