//! Generator for the flash-attention LUT header (`flash_luts.vh`). Emits the two
//! softmax tables to stdout as packed Verilog localparams. Its output is checked in as
//! `fpga/rtl/flash_attn/flash_luts.vh` (a generated
//! artifact, like `fpga/regmap/matmul_regs.vh`). Single source: the same closed formulas the
//! `flash_ref` model uses —
//!
//!   FLASH_EXP_LUT[k]   = 2^(−k/2^B)      for numeric/exp, af∈[0,1)
//!   FLASH_RECIP_LUT[k] = 1/(1 + k/2^B)   for numeric/recip, sig∈[1,2)
//!
//! Entry k occupies bits [k*32 +: 32]; the leaf reads table[idx*32 +: 32] and
//! table[(idx+1)*32 +: 32] for the two interpolation endpoints (k = 0 … 2^B).

const std = @import("std");

const B: u32 = 8;
const N: u32 = 1 << B;

fn expBits(k: u32) u32 {
    const v: f32 = @floatCast(std.math.pow(f64, 2.0, -@as(f64, @floatFromInt(k)) / @as(f64, N)));
    return @bitCast(v);
}
fn recipBits(k: u32) u32 {
    const v: f32 = @floatCast(1.0 / (1.0 + @as(f64, @floatFromInt(k)) / @as(f64, N)));
    return @bitCast(v);
}

fn emitTable(buf: []u8, cursor: *usize, name: []const u8, comptime bitsFn: fn (u32) u32) std.fmt.BufPrintError!void {
    const width = (N + 1) * 32;
    cursor.* += (try std.fmt.bufPrint(buf[cursor.*..], "localparam [{d}:0] {s} = {{\n", .{ width - 1, name })).len;
    // {} concatenation is MSB-first; list k = N … 0 so entry k lands at [k*32 +: 32].
    var k: u32 = N + 1;
    while (k > 0) {
        k -= 1;
        const sep = if (k == 0) "" else ",";
        cursor.* += (try std.fmt.bufPrint(buf[cursor.*..], "    32'h{X:0>8}{s} // {d}\n", .{ bitsFn(k), sep, k })).len;
    }
    cursor.* += (try std.fmt.bufPrint(buf[cursor.*..], "}};\n", .{})).len;
}

pub fn main(init: std.process.Init) !void {
    var buf: [96 * 1024]u8 = undefined;
    var cursor: usize = 0;
    // No `ifndef include guard: these are module-scoped localparams and each
    // including module (exp, recip) needs its own textual copy. A guard would
    // let only the first include define them (global `define), starving the rest.
    cursor += (try std.fmt.bufPrint(buf[cursor..], "// Generated from fpga/rtl/flash_attn/gen_luts.zig — do not edit.\n", .{})).len;
    cursor += (try std.fmt.bufPrint(buf[cursor..], "localparam integer FLASH_LUT_BITS = {d};\n", .{B})).len;
    try emitTable(&buf, &cursor, "FLASH_EXP_LUT", expBits);
    try emitTable(&buf, &cursor, "FLASH_RECIP_LUT", recipBits);

    var out_buf: [4096]u8 = undefined;
    var stdout = std.Io.File.stdout().writerStreaming(init.io, &out_buf);
    try stdout.interface.writeAll(buf[0..cursor]);
    try stdout.interface.flush();
}
