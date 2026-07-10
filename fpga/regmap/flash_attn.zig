//! Single source of truth for the flash_attn kernel AXI-Lite register map.
//!
//! `flash_attn.regmap` is the schema; this module comptime-parses it so the Zig MMIO
//! driver (device/pl flash tenant) reads register offsets straight from the table and
//! the RTL gets the same constants via `emitVerilog` (the generated `flash_regs.vh`).
//! Nothing else may hand-duplicate these constants — change the `.regmap`.
//!
//! Mirrors `fpga/regmap/matmul.zig`: the AXI-Lite block base `addr`esses (assigned by
//! the Vivado build, mapped by device/pl) and the build `caps` live here beside the
//! register table, so host and gateware cannot drift. (The generic parser is mirrored
//! from matmul.zig per plan-fpga-6's "each op gets its own regmap" — kept self-contained.)

const std = @import("std");

const raw = @embedFile("flash_attn.regmap");

pub const Access = enum { ro, rw, wo };

pub const Reg = struct {
    name: []const u8,
    offset: u32,
    access: Access,
    reset: u32,
};

/// All registers, in schema order. Parsed at comptime from `flash_attn.regmap`.
pub const table: [countRegs(raw)]Reg = parse(raw);

/// Offset of a named register, resolved at comptime (`@compileError` if absent).
pub fn offsetOf(comptime name: []const u8) u32 {
    inline for (table) |reg| {
        if (comptime std.mem.eql(u8, reg.name, name)) return reg.offset;
    }
    @compileError("unknown flash register: " ++ name);
}

/// Reset value of a named register, resolved at comptime.
pub fn resetOf(comptime name: []const u8) u32 {
    inline for (table) |reg| {
        if (comptime std.mem.eql(u8, reg.name, name)) return reg.reset;
    }
    @compileError("unknown flash register: " ++ name);
}

// ---- The rest of the bitstream contract (not AXI-Lite registers) ------------

/// AXI-Lite block base addresses in the PS M_AXI_HPM0_FPD window. Allocated in the
/// 0xA01x range so a future single bitstream can host flash alongside matmul (0xA00x).
/// One MM2S DMA per input stream (Q/K/V/mask) + one S2MM for O + the kernel window.
pub const addr = struct {
    pub const dma_q: i64 = 0xA010_0000; // Q MM2S
    pub const dma_k: i64 = 0xA011_0000; // K MM2S
    pub const dma_v: i64 = 0xA012_0000; // V MM2S
    pub const dma_mask: i64 = 0xA013_0000; // mask MM2S
    pub const dma_o: i64 = 0xA014_0000; // O S2MM
    pub const kernel: i64 = 0xA015_0000; // kernel AXI-Lite
};

/// Build/layout caps of the deployed bitstream the device driver must match. The
/// kernel version gates changes to these compile-time capacities.
pub const caps = struct {
    pub const lanes: u32 = 8;
    pub const head_dim_max: u32 = 128;
    pub const max_heads: usize = 32;
    pub const max_head_kv: usize = 8;
    // The only host staging the v2 tenant reserves: the DMA sink for the kernel's
    // packed 8-wide O emit (dense n_tokens·n_heads·head_dim_v f32). Q/K/V/mask are
    // DMA'd straight from the resident tensors in their native layout — there is no
    // input staging (that materialized-and-replicated gather was the v1 bottleneck).
    // 128 KiB covers the 1.7B/8B decode output with margin; a larger O falls back to PS.
    pub const o_staging_bytes: usize = 128 * 1024;
};

comptime {
    // The LANES register self-describes the MAC width; caps must agree.
    std.debug.assert(caps.lanes == resetOf("LANES"));
    // head_dim_max must be a multiple of the lane count.
    std.debug.assert(caps.head_dim_max % caps.lanes == 0);
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

/// Emit a Verilog header of `localparam` offsets and reset values into `buf`. The RTL
/// `\`include`s the generated file so its register decode and the driver never drift.
pub fn emitVerilog(buf: []u8) std.fmt.BufPrintError![]const u8 {
    var cursor: usize = 0;
    cursor += (try std.fmt.bufPrint(buf[cursor..], "// Generated from fpga/regmap/flash_attn.regmap — do not edit.\n", .{})).len;
    cursor += (try std.fmt.bufPrint(buf[cursor..], "`ifndef FLASH_REGS_VH\n`define FLASH_REGS_VH\n", .{})).len;
    for (table) |reg| {
        cursor += (try std.fmt.bufPrint(buf[cursor..], "localparam [11:0] FLASH_OFF_{s} = 12'h{X:0>3};\n", .{ reg.name, reg.offset })).len;
    }
    for (table) |reg| {
        if (reg.access != .ro) continue;
        cursor += (try std.fmt.bufPrint(buf[cursor..], "localparam [31:0] FLASH_RST_{s} = 32'h{X:0>8};\n", .{ reg.name, reg.reset })).len;
    }
    cursor += (try std.fmt.bufPrint(buf[cursor..], "localparam integer FLASH_LANES = {d};\n", .{caps.lanes})).len;
    cursor += (try std.fmt.bufPrint(buf[cursor..], "`endif\n", .{})).len;
    return buf[0..cursor];
}

/// Emit the Vivado address-map TCL into `buf`. The bitstream build.tcl `source`s the
/// generated file and iterates `$flash_address_map` ({cell intf offset} per AXI-Lite
/// block) so the addresses Vivado assigns and the addresses device/pl maps never drift.
pub fn emitAddrTcl(buf: []u8) std.fmt.BufPrintError![]const u8 {
    var cursor: usize = 0;
    cursor += (try std.fmt.bufPrint(buf[cursor..], "# Generated from fpga/regmap/flash_attn.zig — do not edit.\n", .{})).len;
    cursor += (try std.fmt.bufPrint(buf[cursor..], "# AXI-Lite address map for the flash bitstream; {{cell intf offset}} per block.\n", .{})).len;
    cursor += (try std.fmt.bufPrint(buf[cursor..], "set flash_address_map {{\n", .{})).len;
    const dmas = .{
        .{ "dma_q", addr.dma_q }, .{ "dma_k", addr.dma_k },   .{ "dma_v", addr.dma_v },
        .{ "dma_mask", addr.dma_mask }, .{ "dma_o", addr.dma_o },
    };
    inline for (dmas) |d| {
        cursor += (try std.fmt.bufPrint(buf[cursor..], "    {{{s} S_AXI_LITE 0x{X:0>8}}}\n", .{ d[0], @as(u32, @intCast(d[1])) })).len;
    }
    cursor += (try std.fmt.bufPrint(buf[cursor..], "    {{kernel S_AXI 0x{X:0>8}}}\n", .{@as(u32, @intCast(addr.kernel))})).len;
    cursor += (try std.fmt.bufPrint(buf[cursor..], "}}\n", .{})).len;
    return buf[0..cursor];
}

test "regmap parses known offsets and resets" {
    try std.testing.expect(table.len >= 20);
    try std.testing.expectEqual(@as(u32, 0x00), offsetOf("ID"));
    try std.testing.expectEqual(@as(u32, 0xF1A54A00), resetOf("ID"));
    try std.testing.expectEqual(@as(u32, 3), resetOf("VERSION"));
    try std.testing.expectEqual(@as(u32, 0x08), offsetOf("CTRL"));
    try std.testing.expectEqual(@as(u32, 0x1C), offsetOf("N_HEAD_KV"));
    try std.testing.expectEqual(@as(u32, 0x20), offsetOf("HEAD_RATIO"));
    try std.testing.expectEqual(@as(u32, 0x2C), offsetOf("SCALE"));
    try std.testing.expectEqual(@as(u32, 0x3C), offsetOf("Q_BEATS"));
    try std.testing.expectEqual(@as(u32, 0x54), offsetOf("O_STALL"));
    try std.testing.expectEqual(@as(u32, 8), resetOf("LANES"));
}

test "emitVerilog contains offsets and ro reset values" {
    var buf: [4096]u8 = undefined;
    const out = try emitVerilog(&buf);
    try std.testing.expect(std.mem.indexOf(u8, out, "FLASH_OFF_SCALE = 12'h02C;") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "FLASH_RST_ID = 32'hF1A54A00;") != null);
    // wo/rw registers must not get a reset localparam.
    try std.testing.expect(std.mem.indexOf(u8, out, "FLASH_RST_CTRL") == null);
    try std.testing.expect(std.mem.indexOf(u8, out, "FLASH_RST_SCALE") == null);
    try std.testing.expect(std.mem.indexOf(u8, out, "FLASH_LANES = 8;") != null);
}

test "address map and caps are internally consistent" {
    try std.testing.expectEqual(@as(i64, 0xA010_0000), addr.dma_q);
    try std.testing.expectEqual(@as(i64, 0xA015_0000), addr.kernel);
    try std.testing.expectEqual(@as(u32, 8), caps.lanes);
    try std.testing.expectEqual(@as(u32, 0), caps.head_dim_max % caps.lanes);
}

test "emitAddrTcl matches the deployed address map" {
    var buf: [4096]u8 = undefined;
    const out = try emitAddrTcl(&buf);
    try std.testing.expect(std.mem.indexOf(u8, out, "set flash_address_map {") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "{dma_q S_AXI_LITE 0xA0100000}") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "{dma_o S_AXI_LITE 0xA0140000}") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "{kernel S_AXI 0xA0150000}") != null);
}
