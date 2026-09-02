//! Single-source AXI-Lite and address contract for the Penzai accelerator.

const std = @import("std");

const raw = @embedFile("engine.regmap");

pub const Access = enum { ro, rw, wo };
pub const Reg = struct {
    name: []const u8,
    offset: u32,
    access: Access,
    reset: u32,
};

pub const table: [countRegs(raw)]Reg = parse(raw);

pub fn offsetOf(comptime name: []const u8) u32 {
    inline for (table) |reg| {
        if (comptime std.mem.eql(u8, reg.name, name)) return reg.offset;
    }
    @compileError("unknown engine register: " ++ name);
}

pub fn resetOf(comptime name: []const u8) u32 {
    inline for (table) |reg| {
        if (comptime std.mem.eql(u8, reg.name, name)) return reg.reset;
    }
    @compileError("unknown engine register: " ++ name);
}

pub const addr = struct {
    pub const kernel: i64 = 0xA000_0000;
};

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
            .offset = std.fmt.parseInt(u32, offset_s, 0) catch
                @compileError("regmap: bad offset " ++ offset_s),
            .access = std.meta.stringToEnum(Access, access_s) orelse
                @compileError("regmap: bad access " ++ access_s),
            .reset = std.fmt.parseInt(u32, reset_s, 0) catch
                @compileError("regmap: bad reset " ++ reset_s),
        };
        n += 1;
    }
    return regs;
}

pub fn emitVerilog(buf: []u8) std.fmt.BufPrintError![]const u8 {
    var cursor: usize = 0;
    cursor += (try std.fmt.bufPrint(buf[cursor..], "// Generated from fpga/regmap/engine.regmap - do not edit.\n", .{})).len;
    cursor += (try std.fmt.bufPrint(buf[cursor..], "`ifndef ENGINE_REGS_VH\n`define ENGINE_REGS_VH\n", .{})).len;
    for (table) |reg| {
        cursor += (try std.fmt.bufPrint(buf[cursor..], "localparam [11:0] ENGINE_REG_OFF_{s} = 12'h{X:0>3};\n", .{ reg.name, reg.offset })).len;
    }
    for (table) |reg| {
        cursor += (try std.fmt.bufPrint(buf[cursor..], "localparam [31:0] ENGINE_REG_RST_{s} = 32'h{X:0>8};\n", .{ reg.name, reg.reset })).len;
    }
    cursor += (try std.fmt.bufPrint(buf[cursor..], "`endif\n", .{})).len;
    return buf[0..cursor];
}

pub fn emitAddrTcl(buf: []u8) std.fmt.BufPrintError![]const u8 {
    var cursor: usize = 0;
    cursor += (try std.fmt.bufPrint(buf[cursor..], "# Generated from fpga/regmap/engine.zig - do not edit.\n", .{})).len;
    cursor += (try std.fmt.bufPrint(buf[cursor..], "set engine_address_map {{\n", .{})).len;
    cursor += (try std.fmt.bufPrint(buf[cursor..], "    {{engine S_AXI 0x{X:0>8}}}\n", .{@as(u32, @intCast(addr.kernel))})).len;
    cursor += (try std.fmt.bufPrint(buf[cursor..], "}}\n", .{})).len;
    return buf[0..cursor];
}

test "engine interface v7 register contract is frozen" {
    try std.testing.expectEqual(@as(u32, 0xB05A_4000), resetOf("ID"));
    try std.testing.expectEqual(@as(u32, 0x0001_0007), resetOf("VERSION"));
    try std.testing.expectEqual(@as(u32, 0x2FC1_4A79), resetOf("LAYOUT_HASH_LO"));
    try std.testing.expectEqual(@as(u32, 0xC255_C7A5), resetOf("LAYOUT_HASH_HI"));
    try std.testing.expectEqual(@as(u32, 0x10), offsetOf("CTRL"));
    try std.testing.expectEqual(@as(u32, 0x0B0), offsetOf("CMD_KV_CAPACITY"));
    try std.testing.expectEqual(@as(u32, 0x0C0), offsetOf("CMD_TAG"));
    try std.testing.expectEqual(@as(u32, 0x150), offsetOf("MODEL_SPEC_ERROR"));
    try std.testing.expectEqual(@as(u32, 0x160), offsetOf("METRICS_SCHEMA"));
    try std.testing.expectEqual(@as(u32, 0x188), offsetOf("METRICS_OVERFLOW3"));
    try std.testing.expectEqual(@as(u32, 0x0028_0b1d), resetOf("METRICS_CAPABILITIES"));
    try std.testing.expectEqual(@as(u32, 7), resetOf("AXI_MASTERS"));
    try std.testing.expectEqual(@as(i64, 0xA000_0000), addr.kernel);
}

test "engine generated artifacts carry the same contract" {
    var buf: [32 * 1024]u8 = undefined;
    const vh = try emitVerilog(&buf);
    try std.testing.expect(std.mem.indexOf(u8, vh, "ENGINE_REG_RST_ID = 32'hB05A4000") != null);
    const tcl = try emitAddrTcl(&buf);
    try std.testing.expect(std.mem.indexOf(u8, tcl, "{engine S_AXI 0xA0000000}") != null);
}
