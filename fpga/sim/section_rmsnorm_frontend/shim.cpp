#include "shim.h"

#include "Vsection_rmsnorm_frontend.h"
#include "verilated.h"

struct Dut {
    Vsection_rmsnorm_frontend *top;
};

double sc_time_stamp() { return 0; }

extern "C" {

Dut *dut_new(void) {
    Dut *d = new Dut();
    d->top = new Vsection_rmsnorm_frontend();
    d->top->cfg_valid = 0;
    d->top->cfg_resident = 0;
    d->top->abort_run = 0;
    d->top->s_axis_tvalid = 0;
    d->top->r_wr_ready = 0;
    d->top->r_wr_error = 0;
    d->top->rd_req_ready = 0;
    d->top->rd_rsp_valid = 0;
    d->top->result_ready = 0;
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
void dut_set_config(Dut *d, int valid, uint16_t rows, uint8_t tokens) {
    d->top->cfg_valid = valid;
    d->top->cfg_rows = rows;
    d->top->cfg_tokens = tokens;
}
void dut_set_abort(Dut *d, int value) { d->top->abort_run = value; }
void dut_set_stream(Dut *d, int valid, uint64_t data, uint8_t keep, int last) {
    d->top->s_axis_tvalid = valid;
    d->top->s_axis_tdata = data;
    d->top->s_axis_tkeep = keep;
    d->top->s_axis_tlast = last;
}
void dut_set_r_write_sink(Dut *d, int ready, int error) {
    d->top->r_wr_ready = ready;
    d->top->r_wr_error = error;
}
void dut_set_read_request_ready(Dut *d, int ready) { d->top->rd_req_ready = ready; }
void dut_set_read_response(Dut *d, int valid, const uint64_t lanes[4], int error) {
    d->top->rd_rsp_valid = valid;
    for (int lane = 0; lane < 4; ++lane) {
        d->top->rd_rsp_data[lane * 2] = static_cast<uint32_t>(lanes[lane]);
        d->top->rd_rsp_data[lane * 2 + 1] =
            static_cast<uint32_t>(lanes[lane] >> 32);
    }
    d->top->rd_rsp_error = error;
}
void dut_set_result_ready(Dut *d, int ready) { d->top->result_ready = ready; }

int dut_config_ready(Dut *d) { return d->top->cfg_ready; }
int dut_busy(Dut *d) { return d->top->busy; }
int dut_done(Dut *d) { return d->top->done; }
int dut_error(Dut *d) { return d->top->error; }
uint8_t dut_status(Dut *d) { return d->top->status; }
int dut_stream_ready(Dut *d) { return d->top->s_axis_tready; }
int dut_r_write_valid(Dut *d) { return d->top->r_wr_valid; }
uint8_t dut_r_write_bank(Dut *d) { return d->top->r_wr_bank; }
uint16_t dut_r_write_address(Dut *d) { return d->top->r_wr_address; }
uint64_t dut_r_write_data(Dut *d) { return d->top->r_wr_data; }
int dut_read_request_valid(Dut *d) { return d->top->rd_req_valid; }
uint8_t dut_read_request_token(Dut *d) { return d->top->rd_req_token; }
uint16_t dut_read_request_group(Dut *d) { return d->top->rd_req_group; }
int dut_read_response_ready(Dut *d) { return d->top->rd_rsp_ready; }
int dut_result_valid(Dut *d) { return d->top->result_valid; }
uint8_t dut_result_token(Dut *d) { return d->top->result_token; }
uint8_t dut_result_max_exp(Dut *d) { return d->top->result_max_exp; }
uint64_t dut_result_sum_sq(Dut *d) { return d->top->result_sum_sq; }
uint16_t dut_result_rows(Dut *d) { return d->top->result_rows; }
int dut_result_final(Dut *d) { return d->top->result_final; }

}
