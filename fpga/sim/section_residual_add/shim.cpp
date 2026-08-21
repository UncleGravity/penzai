#include "shim.h"

#include "Vsection_residual_add.h"
#include "Vsection_residual_add___024root.h"
#include "verilated.h"

struct Dut {
    Vsection_residual_add *top;
};

double sc_time_stamp() { return 0; }

extern "C" {

Dut *dut_new(void) {
    Dut *d = new Dut();
    d->top = new Vsection_residual_add();
    d->top->cfg_valid = 0;
    d->top->s_axis_tvalid = 0;
    d->top->rd_req_ready = 0;
    d->top->rd_rsp_valid = 0;
    d->top->rd_rsp_error = 0;
    d->top->r_wr_ready = 0;
    d->top->r_wr_error = 0;
    d->top->m_axis_tready = 0;
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

void dut_set_config(Dut *d, int valid, uint16_t rows, uint8_t tokens) {
    d->top->cfg_valid = valid;
    d->top->cfg_rows = rows;
    d->top->cfg_tokens = tokens;
}

void dut_set_input(Dut *d, int valid, uint64_t data, uint8_t keep, int last) {
    d->top->s_axis_tvalid = valid;
    d->top->s_axis_tdata = data;
    d->top->s_axis_tkeep = keep;
    d->top->s_axis_tlast = last;
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

void dut_set_write_sink(Dut *d, int ready, int error) {
    d->top->r_wr_ready = ready;
    d->top->r_wr_error = error;
}

void dut_set_output_ready(Dut *d, int ready) {
    d->top->m_axis_tready = ready;
}

int dut_config_ready(Dut *d) { return d->top->cfg_ready; }
int dut_busy(Dut *d) { return d->top->busy; }
int dut_done(Dut *d) { return d->top->done; }
int dut_error(Dut *d) { return d->top->error; }
uint8_t dut_status(Dut *d) { return d->top->status; }
int dut_input_ready(Dut *d) { return d->top->s_axis_tready; }

int dut_read_request_valid(Dut *d) { return d->top->rd_req_valid; }
uint8_t dut_read_request_role(Dut *d) { return d->top->rd_req_role; }
uint8_t dut_read_request_token(Dut *d) { return d->top->rd_req_token; }
uint16_t dut_read_request_group(Dut *d) { return d->top->rd_req_group; }
int dut_read_response_ready(Dut *d) { return d->top->rd_rsp_ready; }

int dut_write_valid(Dut *d) { return d->top->r_wr_valid; }
uint8_t dut_write_bank(Dut *d) { return d->top->r_wr_bank; }
uint16_t dut_write_address(Dut *d) { return d->top->r_wr_address; }
uint64_t dut_write_data(Dut *d) { return d->top->r_wr_data; }

int dut_output_valid(Dut *d) { return d->top->m_axis_tvalid; }
uint64_t dut_output_data(Dut *d) { return d->top->m_axis_tdata; }
uint8_t dut_output_keep(Dut *d) { return d->top->m_axis_tkeep; }
int dut_output_last(Dut *d) { return d->top->m_axis_tlast; }
uint8_t dut_debug_state(Dut *d) {
    return d->top->rootp->section_residual_add__DOT__state_q;
}
uint8_t dut_debug_add_state(Dut *d) {
    return d->top->rootp->section_residual_add__DOT__u_add__DOT__state_q;
}

}
