#include "shim.h"

#include "Vsection_rmsnorm_weighted_source.h"
#include "verilated.h"

struct Dut {
    Vsection_rmsnorm_weighted_source *top;
};

double sc_time_stamp() { return 0; }

extern "C" {

Dut *dut_new(void) {
    Dut *d = new Dut();
    d->top = new Vsection_rmsnorm_weighted_source();
    d->top->gamma_cfg_valid = 0;
    d->top->gamma_tvalid = 0;
    d->top->cfg_valid = 0;
    d->top->inv_valid = 0;
    d->top->rd_req_ready = 0;
    d->top->rd_rsp_valid = 0;
    d->top->rd_rsp_error = 0;
    d->top->scalar_ready = 0;
    d->top->abort_run = 0;
    return d;
}

void dut_free(Dut *d) {
    d->top->final();
    delete d->top;
    delete d;
}

void dut_eval(Dut *d) { d->top->eval(); }
void dut_set_clk(Dut *d, int value) { d->top->clk = value; }
void dut_set_rst_n(Dut *d, int value) { d->top->rst_n = value; }
void dut_set_abort(Dut *d, int value) { d->top->abort_run = value; }

void dut_set_gamma_config(Dut *d, int valid, uint16_t rows) {
    d->top->gamma_cfg_valid = valid;
    d->top->gamma_cfg_rows = rows;
}

void dut_set_gamma_stream(Dut *d, int valid, uint64_t data, uint8_t keep,
                          int last) {
    d->top->gamma_tvalid = valid;
    d->top->gamma_tdata = data;
    d->top->gamma_tkeep = keep;
    d->top->gamma_tlast = last;
}

void dut_set_run_config(Dut *d, int valid, uint16_t rows, uint8_t tokens) {
    d->top->cfg_valid = valid;
    d->top->cfg_rows = rows;
    d->top->cfg_tokens = tokens;
}

void dut_set_inverse(Dut *d, int valid, uint8_t token, uint32_t data,
                     int final) {
    d->top->inv_valid = valid;
    d->top->inv_token = token;
    d->top->inv_rms = data;
    d->top->inv_final = final;
}

void dut_set_read_request_ready(Dut *d, int ready) {
    d->top->rd_req_ready = ready;
}

void dut_set_read_response(Dut *d, int valid, const uint64_t lanes[4],
                           int error) {
    d->top->rd_rsp_valid = valid;
    for (int lane = 0; lane < 4; ++lane) {
        d->top->rd_rsp_data[lane * 2] = static_cast<uint32_t>(lanes[lane]);
        d->top->rd_rsp_data[lane * 2 + 1] =
            static_cast<uint32_t>(lanes[lane] >> 32);
    }
    d->top->rd_rsp_error = error;
}

void dut_set_scalar_ready(Dut *d, int ready) {
    d->top->scalar_ready = ready;
}

int dut_gamma_config_ready(Dut *d) { return d->top->gamma_cfg_ready; }
int dut_gamma_stream_ready(Dut *d) { return d->top->gamma_tready; }
int dut_gamma_busy(Dut *d) { return d->top->gamma_busy; }
int dut_gamma_done(Dut *d) { return d->top->gamma_done; }
int dut_gamma_error(Dut *d) { return d->top->gamma_error; }
uint8_t dut_gamma_status(Dut *d) { return d->top->gamma_status; }
int dut_gamma_valid(Dut *d) { return d->top->gamma_valid; }

int dut_run_config_ready(Dut *d) { return d->top->cfg_ready; }
int dut_busy(Dut *d) { return d->top->busy; }
int dut_done(Dut *d) { return d->top->done; }
int dut_error(Dut *d) { return d->top->error; }
uint16_t dut_status(Dut *d) { return d->top->status; }
int dut_inverse_ready(Dut *d) { return d->top->inv_ready; }
int dut_read_request_valid(Dut *d) { return d->top->rd_req_valid; }
uint8_t dut_read_request_role(Dut *d) { return d->top->rd_req_role; }
uint8_t dut_read_request_token(Dut *d) { return d->top->rd_req_token; }
uint16_t dut_read_request_group(Dut *d) { return d->top->rd_req_group; }
int dut_read_response_ready(Dut *d) { return d->top->rd_rsp_ready; }
int dut_scalar_valid(Dut *d) { return d->top->scalar_valid; }
uint32_t dut_scalar_data(Dut *d) { return d->top->scalar_data; }
int dut_scalar_last(Dut *d) { return d->top->scalar_last; }
uint8_t dut_scalar_status(Dut *d) { return d->top->scalar_status; }

}
