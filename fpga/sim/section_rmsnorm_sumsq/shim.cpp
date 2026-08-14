#include "Vsection_rmsnorm_sumsq.h"
#include "verilated.h"
#include "shim.h"

struct Dut {
    Vsection_rmsnorm_sumsq *top;
};

double sc_time_stamp() { return 0; }

extern "C" {

Dut *dut_new(void) {
    Dut *d = new Dut();
    d->top = new Vsection_rmsnorm_sumsq();
    return d;
}

void dut_free(Dut *d) {
    delete d->top;
    delete d;
}

void dut_eval(Dut *d) { d->top->eval(); }
void dut_set_clk(Dut *d, int value) { d->top->clk = value; }
void dut_set_rst_n(Dut *d, int value) { d->top->rst_n = value; }

void dut_set_config(Dut *d, int valid, uint16_t rows, uint8_t tokens,
                    uint32_t max_exp) {
    d->top->cfg_valid = valid;
    d->top->cfg_rows = rows;
    d->top->cfg_tokens = tokens;
    d->top->cfg_max_exp = max_exp;
}

void dut_set_abort(Dut *d, int value) { d->top->abort_run = value; }

void dut_set_group(Dut *d, int valid, const uint32_t lanes[8], int error,
                   int last) {
    d->top->s_group_valid = valid;
    for (int i = 0; i < 8; ++i) d->top->s_group_data[i] = lanes[i];
    d->top->s_group_error = error;
    d->top->s_group_last = last;
}

void dut_set_result_ready(Dut *d, int value) {
    d->top->result_ready = value;
}

int dut_config_ready(Dut *d) { return d->top->cfg_ready; }
int dut_busy(Dut *d) { return d->top->busy; }
int dut_done(Dut *d) { return d->top->done; }
int dut_error(Dut *d) { return d->top->error; }
uint8_t dut_status(Dut *d) { return d->top->status; }
int dut_group_ready(Dut *d) { return d->top->s_group_ready; }
int dut_result_valid(Dut *d) { return d->top->result_valid; }
uint8_t dut_result_token(Dut *d) { return d->top->result_token; }
uint8_t dut_result_max_exp(Dut *d) { return d->top->result_max_exp; }
uint64_t dut_result_sum_sq(Dut *d) { return d->top->result_sum_sq; }
uint16_t dut_result_rows(Dut *d) { return d->top->result_rows; }
int dut_result_subnormal_warning(Dut *d) {
    return d->top->result_subnormal_warning;
}
int dut_result_final(Dut *d) { return d->top->result_final; }

}
