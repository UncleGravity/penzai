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

// ---- S_AXI control slave (tb is master) ----
void dut_set_s_awaddr(Dut *d, uint32_t v);
void dut_set_s_awvalid(Dut *d, int v);
int dut_s_awready(Dut *d);
void dut_set_s_wdata(Dut *d, uint32_t v);
void dut_set_s_wvalid(Dut *d, int v);
int dut_s_wready(Dut *d);
int dut_s_bvalid(Dut *d);
void dut_set_s_bready(Dut *d, int v);
void dut_set_s_araddr(Dut *d, uint32_t v);
void dut_set_s_arvalid(Dut *d, int v);
int dut_s_arready(Dut *d);
uint32_t dut_s_rdata(Dut *d);
int dut_s_rvalid(Dut *d);
void dut_set_s_rready(Dut *d, int v);

// ---- M_AXI_REG replay master (tb is slave) ----
uint32_t dut_reg_awaddr(Dut *d);
int dut_reg_awvalid(Dut *d);
void dut_set_reg_awready(Dut *d, int v);
uint32_t dut_reg_wdata(Dut *d);
int dut_reg_wvalid(Dut *d);
void dut_set_reg_wready(Dut *d, int v);
int dut_reg_bready(Dut *d);
void dut_set_reg_bvalid(Dut *d, int v);
uint32_t dut_reg_araddr(Dut *d);
int dut_reg_arvalid(Dut *d);
void dut_set_reg_arready(Dut *d, int v);
int dut_reg_rready(Dut *d);
void dut_set_reg_rdata(Dut *d, uint32_t v);
void dut_set_reg_rvalid(Dut *d, int v);

// ---- M_AXI_DESC read master (tb is slave) ----
uint64_t dut_desc_araddr(Dut *d);
int dut_desc_arvalid(Dut *d);
void dut_set_desc_arready(Dut *d, int v);
int dut_desc_rready(Dut *d);
void dut_set_desc_rdata(Dut *d, uint32_t w0, uint32_t w1, uint32_t w2, uint32_t w3);
void dut_set_desc_rvalid(Dut *d, int v);
void dut_set_desc_rlast(Dut *d, int v);

#ifdef __cplusplus
}
#endif
