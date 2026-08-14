#include "Vsection_rmsnorm_inv.h"
#include "verilated.h"
#include "shim.h"

struct Dut {
    Vsection_rmsnorm_inv *top;
};

double sc_time_stamp() { return 0; }

extern "C" {

Dut *dut_new(void) {
    Dut *d = new Dut();
    d->top = new Vsection_rmsnorm_inv();
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
                    uint32_t eps) {
    d->top->cfg_valid = valid;
    d->top->cfg_rows = rows;
    d->top->cfg_tokens = tokens;
    d->top->cfg_eps = eps;
}

void dut_set_abort(Dut *d, int value) { d->top->abort_run = value; }

void dut_set_record(Dut *d, int valid, uint8_t token, uint8_t max_exp,
                    uint64_t sum_sq, uint16_t rows, int final) {
    d->top->s_valid = valid;
    d->top->s_token = token;
    d->top->s_max_exp = max_exp;
    d->top->s_sum_sq = sum_sq;
    d->top->s_rows = rows;
    d->top->s_final = final;
}

void dut_set_result_ready(Dut *d, int value) {
    d->top->result_ready = value;
}

int dut_config_ready(Dut *d) { return d->top->cfg_ready; }
int dut_busy(Dut *d) { return d->top->busy; }
int dut_done(Dut *d) { return d->top->done; }
int dut_error(Dut *d) { return d->top->error; }
uint8_t dut_status(Dut *d) { return d->top->status; }
int dut_record_ready(Dut *d) { return d->top->s_ready; }
int dut_result_valid(Dut *d) { return d->top->result_valid; }
uint8_t dut_result_token(Dut *d) { return d->top->result_token; }
uint32_t dut_result_inv_rms(Dut *d) { return d->top->result_inv_rms; }
int dut_result_final(Dut *d) { return d->top->result_final; }

}
