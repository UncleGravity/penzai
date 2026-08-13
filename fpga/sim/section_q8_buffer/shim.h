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

void dut_set_config(Dut *d, int valid, int bank, uint8_t tokens, uint16_t blocks);
void dut_set_seal(Dut *d, int valid, int bank);
void dut_set_abort(Dut *d, int valid, int bank);
void dut_set_capture(Dut *d, int valid, uint64_t data, int last,
                     int bank, uint8_t token, uint16_t block);
void dut_set_read_request(Dut *d, int valid, int bank,
                          uint8_t token, uint16_t block);
void dut_set_output_ready(Dut *d, int ready);

int dut_config_ready(Dut *d);
int dut_seal_ready(Dut *d);
int dut_seal_done(Dut *d);
int dut_seal_error(Dut *d);
uint8_t dut_bank_clearing(Dut *d);
uint8_t dut_bank_active(Dut *d);
uint8_t dut_bank_valid(Dut *d);
uint8_t dut_bank_error(Dut *d);
uint16_t dut_bank_record_count(Dut *d, int bank);

int dut_capture_ready(Dut *d);
int dut_capture_done(Dut *d);
int dut_capture_error(Dut *d);
int dut_capture_commit_valid(Dut *d);
uint16_t dut_capture_commit_address(Dut *d);

int dut_read_request_ready(Dut *d);
int dut_read_issue_valid(Dut *d);
uint16_t dut_read_issue_address(Dut *d);
int dut_output_valid(Dut *d);
uint64_t dut_output_data(Dut *d);
int dut_output_last(Dut *d);
int dut_output_error(Dut *d);
int dut_output_bank(Dut *d);
uint8_t dut_output_token(Dut *d);
uint16_t dut_output_block(Dut *d);

#ifdef __cplusplus
}
#endif
