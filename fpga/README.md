# FPGA

This tree contains the production FPGA implementation and its verification
flow. The source manifest at `build/sources.f` is the design boundary: it lists
all 55 Verilog modules and four headers admitted to the bitstream.

| Directory | Contents |
|---|---|
| `rtl/engine/` | Token controller, model specification, arenas, and datapath integration |
| `rtl/projection/` | Packed-weight projection pipeline and result sinks |
| `rtl/attention/` | KV storage path, attention kernel, and softmax datapath |
| `rtl/vector/` | Embedding, normalization, residual, RoPE, SwiGLU, and Q8 services |
| `rtl/io/` | Top-level AXI boundary and shared memory readers/writers |
| `rtl/lib/` | Shared arithmetic primitives |
| `regmap/` | MMIO definitions and generated language bindings |
| `build/` | Manifest, Vivado bitstream flow, overlay, and table generators |
| `verify/` | Simulation, formal, lint, synthesis, and routed-checkpoint analysis |

Run the complete local verification gate with:

```sh
zig build verify-rtl
```

Focused commands are `zig build lint-rtl`, `zig build test-rtl`,
`zig build synth-rtl`, and `zig build formal`. Extended synthesis and formal
suites are available as `zig build synth-rtl-all` and `zig build formal-all`.

The qualified routed target is `f225`. Production and verification outputs are
generated under `.zig-cache/fpga-build/` and `.zig-cache/fpga-verify/`; generated
artifacts do not belong in this source tree. See `build/README.md` for bitstream
and deployment details and `verify/README.md` for the test registry.
