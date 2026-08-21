#include "shim.h"

#include "Vsection_rmsnorm_q8_pipeline.h"
#include "Vsection_rmsnorm_q8_pipeline___024root.h"
#include "verilated.h"

struct Dut {
    Vsection_rmsnorm_q8_pipeline *top;
};

double sc_time_stamp() { return 0; }

extern "C" {

Dut *dut_new(void) {
    Dut *d = new Dut();
    d->top = new Vsection_rmsnorm_q8_pipeline();
    d->top->abort_run = 0;
    d->top->gamma_cfg_valid = 0;
    d->top->gamma_tvalid = 0;
    d->top->cfg_valid = 0;
    d->top->s_axis_tvalid = 0;
    d->top->r_wr_ready = 0;
    d->top->r_wr_error = 0;
    d->top->rd_req_ready = 0;
    d->top->rd_rsp_valid = 0;
    d->top->rd_rsp_error = 0;
    d->top->m_axis_tready = 0;
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

void dut_set_run_config(Dut *d, int valid, uint16_t rows, uint8_t tokens,
                        uint32_t eps) {
    d->top->cfg_valid = valid;
    d->top->cfg_rows = rows;
    d->top->cfg_tokens = tokens;
    d->top->cfg_eps = eps;
}

void dut_set_residual_stream(Dut *d, int valid, uint64_t data, uint8_t keep,
                             int last) {
    d->top->s_axis_tvalid = valid;
    d->top->s_axis_tdata = data;
    d->top->s_axis_tkeep = keep;
    d->top->s_axis_tlast = last;
}

void dut_set_r_write_sink(Dut *d, int ready, int error) {
    d->top->r_wr_ready = ready;
    d->top->r_wr_error = error;
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

void dut_set_output_ready(Dut *d, int ready) {
    d->top->m_axis_tready = ready;
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
uint32_t dut_status(Dut *d) { return d->top->status; }
int dut_residual_stream_ready(Dut *d) { return d->top->s_axis_tready; }
int dut_r_write_valid(Dut *d) { return d->top->r_wr_valid; }
uint8_t dut_r_write_bank(Dut *d) { return d->top->r_wr_bank; }
uint16_t dut_r_write_address(Dut *d) { return d->top->r_wr_address; }
uint64_t dut_r_write_data(Dut *d) { return d->top->r_wr_data; }
int dut_read_request_valid(Dut *d) { return d->top->rd_req_valid; }
uint8_t dut_read_request_role(Dut *d) { return d->top->rd_req_role; }
uint8_t dut_read_request_token(Dut *d) { return d->top->rd_req_token; }
uint16_t dut_read_request_group(Dut *d) { return d->top->rd_req_group; }
int dut_read_response_ready(Dut *d) { return d->top->rd_rsp_ready; }

int dut_output_valid(Dut *d) { return d->top->m_axis_tvalid; }
uint64_t dut_output_data(Dut *d) { return d->top->m_axis_tdata; }
int dut_output_last(Dut *d) { return d->top->m_axis_tlast; }
uint8_t dut_output_token(Dut *d) { return d->top->m_axis_token; }
uint16_t dut_output_block(Dut *d) { return d->top->m_axis_block; }

#define DEBUG_NAME(field) section_rmsnorm_q8_pipeline__DOT__##field
#define DEBUG(field) d->top->rootp->DEBUG_NAME(field)

uint8_t dut_debug_state(Dut *d) { return DEBUG(debug_state); }
uint8_t dut_debug_read_owner(Dut *d) { return DEBUG(debug_read_owner); }
uint8_t dut_debug_reduce_state(Dut *d) { return DEBUG(debug_reduce_state); }
uint8_t dut_debug_reduce_frontend_state(Dut *d) {
    return DEBUG(debug_reduce_frontend_state);
}
uint8_t dut_debug_reduce_inverse_state(Dut *d) {
    return DEBUG(debug_reduce_inverse_state);
}
uint8_t dut_debug_source_state(Dut *d) { return DEBUG(debug_source_state); }
uint8_t dut_debug_weighted_state(Dut *d) { return DEBUG(debug_weighted_state); }
uint8_t dut_debug_q8_state(Dut *d) { return DEBUG(debug_q8_state); }
uint8_t dut_debug_q8_quantizer_state(Dut *d) {
    return DEBUG(debug_q8_quantizer_state);
}
uint8_t dut_debug_q8_emit_index(Dut *d) {
    return DEBUG(debug_q8_emit_index);
}
int dut_debug_q8_record_done(Dut *d) { return DEBUG(debug_q8_record_done); }
int dut_debug_output_fire(Dut *d) { return DEBUG(debug_output_fire); }
int dut_debug_final_output_fire(Dut *d) {
    return DEBUG(debug_final_output_fire);
}
int dut_debug_reduce_busy(Dut *d) { return DEBUG(debug_reduce_busy); }
int dut_debug_reduce_done(Dut *d) { return DEBUG(debug_reduce_done); }
int dut_debug_reduce_error(Dut *d) { return DEBUG(debug_reduce_error); }
int dut_debug_source_busy(Dut *d) { return DEBUG(debug_source_busy); }
int dut_debug_source_done(Dut *d) { return DEBUG(debug_source_done); }
int dut_debug_source_error(Dut *d) { return DEBUG(debug_source_error); }
int dut_debug_child_fault(Dut *d) { return DEBUG(debug_child_fault); }
int dut_debug_child_abort(Dut *d) { return DEBUG(debug_child_abort); }

#undef DEBUG
#undef DEBUG_NAME

}
