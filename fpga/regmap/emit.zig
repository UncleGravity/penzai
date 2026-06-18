//! Generator for the RTL register header. Writes the matmul_regs.vh contents to
//! stdout; the build's `regmap` step copies it to fpga/rtl/matmul/matmul_regs.vh so
//! the Verilog `case` decode and the Zig MMIO driver share one source.

const std = @import("std");
const regmap = @import("matmul.zig");

pub fn main(init: std.process.Init) !void {
    var buf: [8192]u8 = undefined;
    const text = try regmap.emitVerilog(&buf);
    var out_buf: [4096]u8 = undefined;
    var stdout = std.Io.File.stdout().writerStreaming(init.io, &out_buf);
    try stdout.interface.writeAll(text);
    try stdout.interface.flush();
}
