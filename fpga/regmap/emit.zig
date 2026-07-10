//! Generator for the bitstream contract files. Writes one artifact to stdout,
//! selected by argv `<op> <kind>`; the build's `regmap` step captures each and copies
//! it to its destination so the Verilog decode, the Vivado address assignment, and the
//! Zig MMIO driver all share one source (fpga/regmap/{matmul,flash_attn}.zig).
//!
//!   regmap-emit matmul vh   -> fpga/regmap/matmul_regs.vh           (register header)
//!   regmap-emit matmul tcl  -> fpga/regmap/matmul_address_map.tcl   (Vivado address map)
//!   regmap-emit flash  vh   -> fpga/regmap/flash_regs.vh            (register header)
//!   regmap-emit flash  tcl  -> fpga/regmap/flash_address_map.tcl    (Vivado address map)

const std = @import("std");
const matmul = @import("matmul.zig");
const flash = @import("flash_attn.zig");

pub fn main(init: std.process.Init) !void {
    var args = try std.process.Args.Iterator.initAllocator(init.minimal.args, init.gpa);
    defer args.deinit();
    _ = args.next(); // exe name
    const op = args.next() orelse "matmul";
    const kind = args.next() orelse "vh";

    var buf: [8192]u8 = undefined;
    const tcl = std.mem.eql(u8, kind, "tcl");
    const text = if (std.mem.eql(u8, op, "flash"))
        (if (tcl) try flash.emitAddrTcl(&buf) else try flash.emitVerilog(&buf))
    else
        (if (tcl) try matmul.emitAddrTcl(&buf) else try matmul.emitVerilog(&buf));

    var out_buf: [4096]u8 = undefined;
    var stdout = std.Io.File.stdout().writerStreaming(init.io, &out_buf);
    try stdout.interface.writeAll(text);
    try stdout.interface.flush();
}
