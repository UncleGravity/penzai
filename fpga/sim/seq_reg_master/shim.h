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

// seq_core req/gnt side (tb drives a transaction)
void dut_set_req(Dut *d, int v);
void dut_set_we(Dut *d, int v);
void dut_set_addr(Dut *d, uint32_t v);
void dut_set_wdata(Dut *d, uint32_t v);
int dut_gnt(Dut *d);
uint32_t dut_rdata(Dut *d);

// AXI-Lite master outputs (tb-as-slave reads these)
uint32_t dut_awaddr(Dut *d);
int dut_awvalid(Dut *d);
uint32_t dut_wdata_m(Dut *d);
int dut_wstrb(Dut *d);
int dut_wvalid(Dut *d);
int dut_bready(Dut *d);
uint32_t dut_araddr(Dut *d);
int dut_arvalid(Dut *d);
int dut_rready(Dut *d);

// AXI-Lite slave responses (tb-as-slave drives these)
void dut_set_awready(Dut *d, int v);
void dut_set_wready(Dut *d, int v);
void dut_set_bresp(Dut *d, int v);
void dut_set_bvalid(Dut *d, int v);
void dut_set_arready(Dut *d, int v);
void dut_set_rdata_m(Dut *d, uint32_t v);
void dut_set_rresp(Dut *d, int v);
void dut_set_rvalid(Dut *d, int v);

#ifdef __cplusplus
}
#endif
