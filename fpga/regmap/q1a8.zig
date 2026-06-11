//! Single source of truth for the Q1A8 kernel AXI-Lite register map.
//!
//! `q1a8.regmap` is the schema; this module comptime-parses it so the Zig MMIO
//! driver (device/pl, P2) reads register offsets straight from the table and the
//! RTL gets the same constants via `emitVerilog` (the generated `q1a8_regs.vh`).
//! Nothing else may hand-duplicate these constants — change the `.regmap`.

const std = @import("std");

const raw = @embedFile("q1a8.regmap");

pub const Access = enum { ro, rw, wo };

pub const Reg = struct {
    name: []const u8,
    offset: u32,
    access: Access,
    reset: u32,
};

/// All registers, in schema order. Parsed at comptime from `q1a8.regmap`.
pub const table: [countRegs(raw)]Reg = parse(raw);

/// Offset of a named register, resolved at comptime (`@compileError` if absent).
pub fn offsetOf(comptime name: []const u8) u32 {
    inline for (table) |reg| {
        if (comptime std.mem.eql(u8, reg.name, name)) return reg.offset;
    }
    @compileError("unknown q1a8 register: " ++ name);
}

/// Reset value of a named register, resolved at comptime.
pub fn resetOf(comptime name: []const u8) u32 {
    inline for (table) |reg| {
        if (comptime std.mem.eql(u8, reg.name, name)) return reg.reset;
    }
    @compileError("unknown q1a8 register: " ++ name);
}

fn isDataLine(line: []const u8) bool {
    const trimmed = std.mem.trim(u8, line, " \t\r");
    return trimmed.len != 0 and trimmed[0] != '#';
}

fn countRegs(comptime text: []const u8) usize {
    @setEvalBranchQuota(100_000);
    var n: usize = 0;
    var it = std.mem.tokenizeScalar(u8, text, '\n');
    while (it.next()) |line| {
        if (isDataLine(line)) n += 1;
    }
    return n;
}

fn parse(comptime text: []const u8) [countRegs(text)]Reg {
    @setEvalBranchQuota(100_000);
    var regs: [countRegs(text)]Reg = undefined;
    var n: usize = 0;
    var it = std.mem.tokenizeScalar(u8, text, '\n');
    while (it.next()) |line| {
        if (!isDataLine(line)) continue;
        var fields = std.mem.tokenizeAny(u8, line, " \t\r");
        const offset_s = fields.next() orelse @compileError("regmap: missing offset");
        const name = fields.next() orelse @compileError("regmap: missing name");
        const access_s = fields.next() orelse @compileError("regmap: missing access");
        const reset_s = fields.next() orelse @compileError("regmap: missing reset");
        regs[n] = .{
            .name = name,
            .offset = std.fmt.parseInt(u32, offset_s, 0) catch @compileError("regmap: bad offset " ++ offset_s),
            .access = std.meta.stringToEnum(Access, access_s) orelse @compileError("regmap: bad access " ++ access_s),
            .reset = std.fmt.parseInt(u32, reset_s, 0) catch @compileError("regmap: bad reset " ++ reset_s),
        };
        n += 1;
    }
    return regs;
}

/// Emit a Verilog header of `localparam` offsets and reset values into `buf`,
/// returning the written slice. The RTL `\`include`s the generated file so its
/// register decode and this driver never drift. (Wired into a `zig build regmap`
/// step alongside the RTL in P2.)
pub fn emitVerilog(buf: []u8) std.fmt.BufPrintError![]const u8 {
    var cursor: usize = 0;
    cursor += (try std.fmt.bufPrint(buf[cursor..], "// Generated from fpga/regmap/q1a8.regmap — do not edit.\n", .{})).len;
    cursor += (try std.fmt.bufPrint(buf[cursor..], "`ifndef Q1A8_REGS_VH\n`define Q1A8_REGS_VH\n", .{})).len;
    for (table) |reg| {
        cursor += (try std.fmt.bufPrint(buf[cursor..], "localparam [11:0] Q1A8_OFF_{s} = 12'h{X:0>3};\n", .{ reg.name, reg.offset })).len;
    }
    for (table) |reg| {
        if (reg.access != .ro) continue;
        cursor += (try std.fmt.bufPrint(buf[cursor..], "localparam [31:0] Q1A8_RST_{s} = 32'h{X:0>8};\n", .{ reg.name, reg.reset })).len;
    }
    cursor += (try std.fmt.bufPrint(buf[cursor..], "`endif\n", .{})).len;
    return buf[0..cursor];
}

test "regmap parses known offsets and resets" {
    try std.testing.expect(table.len >= 14);
    try std.testing.expectEqual(@as(u32, 0x00), offsetOf("ID"));
    try std.testing.expectEqual(@as(u32, 0xB05A2000), resetOf("ID"));
    try std.testing.expectEqual(@as(u32, 0x08), offsetOf("CTRL"));
    try std.testing.expectEqual(@as(u32, 0x1C), offsetOf("ROWS"));
    try std.testing.expectEqual(@as(u32, 0x20), offsetOf("W_STALL"));
    try std.testing.expectEqual(@as(u32, 0x34), offsetOf("R_BEATS"));
    try std.testing.expectEqual(Access.rw, table[4].access);
}

test "emitVerilog contains offsets and ro reset values" {
    var buf: [4096]u8 = undefined;
    const out = try emitVerilog(&buf);
    try std.testing.expect(std.mem.indexOf(u8, out, "Q1A8_OFF_W_STALL = 12'h020;") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "Q1A8_RST_ID = 32'hB05A2000;") != null);
    // wo/rw registers must not get a reset localparam.
    try std.testing.expect(std.mem.indexOf(u8, out, "Q1A8_RST_CTRL") == null);
}
