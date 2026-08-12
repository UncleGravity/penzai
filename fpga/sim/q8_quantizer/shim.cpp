#include "Vq8_quantizer.h"
#include "verilated.h"
#include "shim.h"

struct Dut {
    Vq8_quantizer *top;
};

double sc_time_stamp() { return 0; }

extern "C" {

Dut *dut_new(void) {
    Dut *d = new Dut();
    d->top = new Vq8_quantizer();
    return d;
}

void dut_free(Dut *d) {
    delete d->top;
    delete d;
}

void dut_eval(Dut *d) { d->top->eval(); }
void dut_set_clk(Dut *d, int value) { d->top->clk = value; }
void dut_set_rst_n(Dut *d, int value) { d->top->rst_n = value; }
void dut_set_input(Dut *d, int valid, uint32_t data, int last) {
    d->top->in_valid = valid;
    d->top->in_data = data;
    d->top->in_last = last;
}
void dut_set_out_ready(Dut *d, int value) { d->top->out_ready = value; }

int dut_in_ready(Dut *d) { return d->top->in_ready; }
int dut_out_valid(Dut *d) { return d->top->out_valid; }
uint8_t dut_out_quant(Dut *d, uint32_t index) {
    const uint32_t word = d->top->out_quants[index / 4];
    return static_cast<uint8_t>(word >> ((index % 4) * 8));
}
uint16_t dut_out_scale(Dut *d) { return d->top->out_scale; }
uint8_t dut_out_status(Dut *d) { return d->top->out_status; }

}
