#include "Vsection_q8_buffer.h"
#include "verilated.h"
#include "shim.h"

struct Dut {
    Vsection_q8_buffer *top;
};

double sc_time_stamp() { return 0; }

extern "C" {

Dut *dut_new(void) {
    Dut *d = new Dut();
    d->top = new Vsection_q8_buffer();
    return d;
}

void dut_free(Dut *d) {
    delete d->top;
    delete d;
}

void dut_eval(Dut *d) { d->top->eval(); }
void dut_set_clk(Dut *d, int value) { d->top->clk = value; }
void dut_set_rst_n(Dut *d, int value) { d->top->rst_n = value; }

void dut_set_config(Dut *d, int valid, int bank, uint8_t tokens, uint16_t blocks) {
    d->top->cfg_valid = valid;
    d->top->cfg_bank = bank;
    d->top->cfg_tokens = tokens;
    d->top->cfg_blocks = blocks;
}

void dut_set_seal(Dut *d, int valid, int bank) {
    d->top->seal_valid = valid;
    d->top->seal_bank = bank;
}

void dut_set_abort(Dut *d, int valid, int bank) {
    d->top->abort_valid = valid;
    d->top->abort_bank = bank;
}

void dut_set_capture(Dut *d, int valid, uint64_t data, int last,
                     int bank, uint8_t token, uint16_t block) {
    d->top->s_axis_tvalid = valid;
    d->top->s_axis_tdata = data;
    d->top->s_axis_tlast = last;
    d->top->s_axis_bank = bank;
    d->top->s_axis_token = token;
    d->top->s_axis_block = block;
}

void dut_set_read_request(Dut *d, int valid, int bank,
                          uint8_t token, uint16_t block) {
    d->top->rd_req_valid = valid;
    d->top->rd_req_bank = bank;
    d->top->rd_req_token = token;
    d->top->rd_req_block = block;
}

void dut_set_output_ready(Dut *d, int ready) { d->top->m_axis_tready = ready; }

int dut_config_ready(Dut *d) { return d->top->cfg_ready; }
int dut_seal_ready(Dut *d) { return d->top->seal_ready; }
int dut_seal_done(Dut *d) { return d->top->seal_done; }
int dut_seal_error(Dut *d) { return d->top->seal_error; }
uint8_t dut_bank_clearing(Dut *d) { return d->top->bank_clearing; }
uint8_t dut_bank_active(Dut *d) { return d->top->bank_active; }
uint8_t dut_bank_valid(Dut *d) { return d->top->bank_valid; }
uint8_t dut_bank_error(Dut *d) { return d->top->bank_error; }

uint16_t dut_bank_record_count(Dut *d, int bank) {
    return bank ? d->top->bank1_record_count : d->top->bank0_record_count;
}

int dut_capture_ready(Dut *d) { return d->top->s_axis_tready; }
int dut_capture_done(Dut *d) { return d->top->cap_record_done; }
int dut_capture_error(Dut *d) { return d->top->cap_record_error; }
int dut_capture_commit_valid(Dut *d) { return d->top->cap_commit_valid; }
uint16_t dut_capture_commit_address(Dut *d) { return d->top->cap_commit_address; }

int dut_read_request_ready(Dut *d) { return d->top->rd_req_ready; }
int dut_read_issue_valid(Dut *d) { return d->top->rd_issue_valid; }
uint16_t dut_read_issue_address(Dut *d) { return d->top->rd_issue_address; }
int dut_output_valid(Dut *d) { return d->top->m_axis_tvalid; }
uint64_t dut_output_data(Dut *d) { return d->top->m_axis_tdata; }
int dut_output_last(Dut *d) { return d->top->m_axis_tlast; }
int dut_output_error(Dut *d) { return d->top->m_axis_error; }
int dut_output_bank(Dut *d) { return d->top->m_axis_bank; }
uint8_t dut_output_token(Dut *d) { return d->top->m_axis_token; }
uint16_t dut_output_block(Dut *d) { return d->top->m_axis_block; }

}
