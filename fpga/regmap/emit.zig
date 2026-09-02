//! Generator for the engine bitstream contract files. Writes one artifact to
//! stdout, selected by argv `engine <kind>`; the build's `regmap` step captures
//! each and copies it to its destination so the Verilog decode, the Vivado
//! address assignment, and the Zig MMIO driver all share one source.
//!
//!   regmap-emit engine vh   -> fpga/regmap/engine_regs.vh           (register header)
//!   regmap-emit engine tcl  -> fpga/regmap/engine_address_map.tcl   (Vivado address map)

const std = @import("std");
const engine = @import("engine.zig");

pub fn main(init: std.process.Init) !void {
    var args = try std.process.Args.Iterator.initAllocator(init.minimal.args, init.gpa);
    defer args.deinit();
    _ = args.next(); // exe name
    const op = args.next() orelse return error.MissingRegisterMap;
    const kind = args.next() orelse "vh";
    if (!std.mem.eql(u8, op, "engine"))
        return error.UnknownRegisterMap;

    var buf: [32 * 1024]u8 = undefined;
    const tcl = std.mem.eql(u8, kind, "tcl");
    const text = if (tcl)
        try engine.emitAddrTcl(&buf)
    else if (std.mem.eql(u8, kind, "vh"))
        try engine.emitVerilog(&buf)
    else
        return error.UnknownArtifactKind;

    var out_buf: [4096]u8 = undefined;
    var stdout = std.Io.File.stdout().writerStreaming(init.io, &out_buf);
    try stdout.interface.writeAll(text);
    try stdout.interface.flush();
}
