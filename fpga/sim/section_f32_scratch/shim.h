#pragma once

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
void dut_set_write_config(Dut *d, int valid, uint8_t role, uint16_t rows, uint8_t tokens);
void dut_set_write_abort(Dut *d, int value);
void dut_set_write_stream(Dut *d, int valid, uint64_t data, uint8_t keep, int last);
void dut_set_read_request(Dut *d, int valid, uint8_t role, uint8_t token, uint16_t group);
void dut_set_read_ready(Dut *d, int value);

int dut_write_config_ready(Dut *d);
int dut_write_busy(Dut *d);
int dut_write_done(Dut *d);
int dut_write_error(Dut *d);
int dut_write_stream_ready(Dut *d);
int dut_write_commit_valid(Dut *d);
uint8_t dut_write_commit_bank(Dut *d);
uint16_t dut_write_commit_address(Dut *d);
int dut_read_request_ready(Dut *d);
int dut_read_issue_valid(Dut *d);
uint16_t dut_read_issue_address(Dut *d);
int dut_read_response_valid(Dut *d);
int dut_read_response_error(Dut *d);
uint64_t dut_read_response_lane(Dut *d, uint32_t lane);

#ifdef __cplusplus
}
#endif
