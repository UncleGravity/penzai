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

// S_AXI (tb is the PS): write channel
void dut_set_s_awaddr(Dut *d, uint32_t v);
void dut_set_s_awvalid(Dut *d, int v);
int dut_s_awready(Dut *d);
void dut_set_s_wdata(Dut *d, uint32_t v);
void dut_set_s_wvalid(Dut *d, int v);
int dut_s_wready(Dut *d);
int dut_s_bvalid(Dut *d);
void dut_set_s_bready(Dut *d, int v);
// S_AXI: read channel
void dut_set_s_araddr(Dut *d, uint32_t v);
void dut_set_s_arvalid(Dut *d, int v);
int dut_s_arready(Dut *d);
int dut_s_rvalid(Dut *d);
uint32_t dut_s_rdata(Dut *d);
void dut_set_s_rready(Dut *d, int v);

// M_AXI_REG (tb is the sc_ctrl target): write channel
int dut_reg_awvalid(Dut *d);
uint32_t dut_reg_awaddr(Dut *d);
void dut_set_reg_awready(Dut *d, int v);
int dut_reg_wvalid(Dut *d);
uint32_t dut_reg_wdata(Dut *d);
void dut_set_reg_wready(Dut *d, int v);
void dut_set_reg_bvalid(Dut *d, int v);
int dut_reg_bready(Dut *d);
// M_AXI_REG: read channel
int dut_reg_arvalid(Dut *d);
uint32_t dut_reg_araddr(Dut *d);
void dut_set_reg_arready(Dut *d, int v);
void dut_set_reg_rdata(Dut *d, uint32_t v);
void dut_set_reg_rvalid(Dut *d, int v);
int dut_reg_rready(Dut *d);

#ifdef __cplusplus
}
#endif
