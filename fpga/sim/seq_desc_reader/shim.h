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

void dut_set_desc_base(Dut *d, uint64_t v);
void dut_set_desc_req(Dut *d, int v);
void dut_set_desc_idx(Dut *d, uint32_t v);
int dut_desc_gnt(Dut *d);
uint32_t dut_desc_data(Dut *d, int word); // word in 0..3

// AXI4 read master outputs (tb-as-slave reads)
uint64_t dut_araddr(Dut *d);
int dut_arvalid(Dut *d);
int dut_rready(Dut *d);

// AXI4 read slave responses (tb-as-slave drives)
void dut_set_arready(Dut *d, int v);
void dut_set_rdata(Dut *d, uint32_t w0, uint32_t w1, uint32_t w2, uint32_t w3);
void dut_set_rresp(Dut *d, int v);
void dut_set_rlast(Dut *d, int v);
void dut_set_rvalid(Dut *d, int v);

#ifdef __cplusplus
}
#endif
