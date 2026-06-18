//! Generator for the bitstream contract files. Writes one artifact to stdout,
//! selected by argv[1]; the build's `regmap` step captures each and copies it to
//! its destination so the Verilog decode, the Vivado address assignment, and the
//! Zig MMIO driver all share one source (fpga/regmap/matmul.zig).
//!
//!   regmap-emit vh   -> fpga/rtl/matmul/matmul_regs.vh        (register header)
//!   regmap-emit tcl  -> .../tcl/address_map.tcl               (Vivado address map)

const std = @import("std");
const regmap = @import("matmul.zig");

pub fn main(init: std.process.Init) !void {
    var args = try std.process.Args.Iterator.initAllocator(init.minimal.args, init.gpa);
    defer args.deinit();
    _ = args.next(); // exe name
    const mode = args.next() orelse "vh";

    var buf: [8192]u8 = undefined;
    const text = if (std.mem.eql(u8, mode, "tcl"))
        try regmap.emitAddrTcl(&buf)
    else
        try regmap.emitVerilog(&buf);

    var out_buf: [4096]u8 = undefined;
    var stdout = std.Io.File.stdout().writerStreaming(init.io, &out_buf);
    try stdout.interface.writeAll(text);
    try stdout.interface.flush();
}
