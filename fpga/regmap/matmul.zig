//! Single source of truth for the matmul kernel AXI-Lite register map.
//!
//! `matmul.regmap` is the schema; this module comptime-parses it so the Zig MMIO
//! driver (device/pl, P2) reads register offsets straight from the table and the
//! RTL gets the same constants via `emitVerilog` (the generated `matmul_regs.vh`).
//! Nothing else may hand-duplicate these constants — change the `.regmap`.
//!
//! This module also owns the rest of the bitstream contract that is not an
//! AXI-Lite register: the block base `addr`esses (assigned by the Vivado build,
//! which sources the generated `address_map.tcl`, and mapped by device/pl) and
//! the build `caps` (COLS_MAX — also emitted to the RTL as `MATMUL_COLS_MAX` —
//! and the host staging reservations). Host and gateware both derive from here,
//! so the address map and column count cannot drift across the three layers.

const std = @import("std");

const raw = @embedFile("matmul.regmap");

pub const Access = enum { ro, rw, wo };

pub const Reg = struct {
    name: []const u8,
    offset: u32,
    access: Access,
    reset: u32,
};

/// All registers, in schema order. Parsed at comptime from `matmul.regmap`.
pub const table: [countRegs(raw)]Reg = parse(raw);

/// Offset of a named register, resolved at comptime (`@compileError` if absent).
pub fn offsetOf(comptime name: []const u8) u32 {
    inline for (table) |reg| {
        if (comptime std.mem.eql(u8, reg.name, name)) return reg.offset;
    }
    @compileError("unknown matmul register: " ++ name);
}

/// Reset value of a named register, resolved at comptime.
pub fn resetOf(comptime name: []const u8) u32 {
    inline for (table) |reg| {
        if (comptime std.mem.eql(u8, reg.name, name)) return reg.reset;
    }
    @compileError("unknown matmul register: " ++ name);
}

// ---- The rest of the bitstream contract (not AXI-Lite registers) ------------

/// AXI-Lite block base addresses in the PS M_AXI_HPM0_FPD window. The Vivado
/// build assigns them (build.tcl sources the generated `address_map.tcl`); the
/// device PL driver maps them (device/pl/{matmul,mmio}.zig). One table, both
/// sides — change addresses here only.
pub const addr = struct {
    /// Weight-port DMA AXI-Lite bases (port N -> dma_wN). dma_w[0] also carries
    /// the result S2MM channel. Length is WEIGHT_PORTS. Typed i64 to match the
    /// /dev/mem mmap offset the device passes straight through.
    pub const dma_w = [_]i64{ 0xA000_0000, 0xA001_0000, 0xA002_0000, 0xA003_0000 };
    /// Activation MM2S DMA AXI-Lite base.
    pub const dma_a: i64 = 0xA004_0000;
    /// Matmul kernel AXI-Lite base.
    pub const kernel: i64 = 0xA005_0000;
};

/// Build/layout caps of the deployed bitstream the host must match. `cols_max`
/// is the kernel COLS_MAX (also emitted to the RTL as `MATMUL_COLS_MAX`, so the
/// device and gateware agree on it); the staging sizes are host DMA-buffer
/// reservations sized to one COLS_MAX group.
pub const caps = struct {
    /// Activation columns multiplied per kernel run (gemm_kernel COLS_MAX).
    pub const cols_max: u32 = 8;
    /// Activation staging reservation: COLS_MAX columns of packed acts.
    pub const acts_staging_bytes: usize = 256 * 1024;
    /// Result staging reservation: num_rb * COLS_MAX * result-bytes-per-rb.
    pub const result_staging_bytes: usize = 8 * 1024 * 1024;
};

comptime {
    // The weight-port count is self-described by the WEIGHT_PORTS register; the
    // address table must expose exactly that many weight DMAs.
    std.debug.assert(addr.dma_w.len == resetOf("WEIGHT_PORTS"));
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
    cursor += (try std.fmt.bufPrint(buf[cursor..], "// Generated from fpga/regmap/matmul.regmap — do not edit.\n", .{})).len;
    cursor += (try std.fmt.bufPrint(buf[cursor..], "`ifndef MATMUL_REGS_VH\n`define MATMUL_REGS_VH\n", .{})).len;
    for (table) |reg| {
        cursor += (try std.fmt.bufPrint(buf[cursor..], "localparam [11:0] MATMUL_OFF_{s} = 12'h{X:0>3};\n", .{ reg.name, reg.offset })).len;
    }
    for (table) |reg| {
        if (reg.access != .ro) continue;
        cursor += (try std.fmt.bufPrint(buf[cursor..], "localparam [31:0] MATMUL_RST_{s} = 32'h{X:0>8};\n", .{ reg.name, reg.reset })).len;
    }
    // Build caps the RTL needs (not registers). decode_top drives the kernel's
    // COLS_MAX from this so the deployed column count and the host's caps.cols_max
    // share one source.
    cursor += (try std.fmt.bufPrint(buf[cursor..], "localparam integer MATMUL_COLS_MAX = {d};\n", .{caps.cols_max})).len;
    cursor += (try std.fmt.bufPrint(buf[cursor..], "`endif\n", .{})).len;
    return buf[0..cursor];
}

/// Emit the Vivado address-map TCL into `buf`, returning the written slice. The
/// bitstream `build.tcl` `source`s the generated file and iterates
/// `$matmul_address_map` (one `{cell intf offset}` per AXI-Lite block) so the
/// addresses Vivado assigns and the addresses device/pl maps never drift.
pub fn emitAddrTcl(buf: []u8) std.fmt.BufPrintError![]const u8 {
    var cursor: usize = 0;
    cursor += (try std.fmt.bufPrint(buf[cursor..], "# Generated from fpga/regmap/matmul.zig — do not edit.\n", .{})).len;
    cursor += (try std.fmt.bufPrint(buf[cursor..], "# AXI-Lite address map for the matmul bitstream; {{cell intf offset}} per block.\n", .{})).len;
    cursor += (try std.fmt.bufPrint(buf[cursor..], "set matmul_address_map {{\n", .{})).len;
    // These AXI-Lite bases are 32-bit; cast to unsigned so the hex prints
    // without a sign (signed {X} would emit a leading '+').
    for (addr.dma_w, 0..) |base, i| {
        cursor += (try std.fmt.bufPrint(buf[cursor..], "    {{dma_w{d} S_AXI_LITE 0x{X:0>8}}}\n", .{ i, @as(u32, @intCast(base)) })).len;
    }
    cursor += (try std.fmt.bufPrint(buf[cursor..], "    {{dma_a S_AXI_LITE 0x{X:0>8}}}\n", .{@as(u32, @intCast(addr.dma_a))})).len;
    cursor += (try std.fmt.bufPrint(buf[cursor..], "    {{kernel S_AXI 0x{X:0>8}}}\n", .{@as(u32, @intCast(addr.kernel))})).len;
    cursor += (try std.fmt.bufPrint(buf[cursor..], "}}\n", .{})).len;
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
    try std.testing.expectEqual(@as(u32, 0x48), offsetOf("NUM_ROWS"));
    try std.testing.expectEqual(@as(u32, 15), resetOf("VERSION"));
    try std.testing.expectEqual(@as(u32, 0x4C), offsetOf("ACT_MODE"));
    try std.testing.expectEqual(@as(u32, 0x60), offsetOf("LOADED_COLS"));
    try std.testing.expectEqual(@as(u32, 0x68), offsetOf("SCRATCH_MODE"));
    try std.testing.expectEqual(@as(u32, 0x80), offsetOf("SCRATCH_ERROR"));
    try std.testing.expectEqual(Access.rw, table[4].access);
}

test "emitVerilog contains offsets and ro reset values" {
    var buf: [4096]u8 = undefined;
    const out = try emitVerilog(&buf);
    try std.testing.expect(std.mem.indexOf(u8, out, "MATMUL_OFF_W_STALL = 12'h020;") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "MATMUL_RST_ID = 32'hB05A2000;") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "MATMUL_OFF_NUM_ROWS = 12'h048;") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "MATMUL_OFF_ACT_MODE = 12'h04C;") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "MATMUL_RST_VERSION = 32'h0000000F;") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "MATMUL_OFF_SCRATCH_MODE = 12'h068;") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "MATMUL_OFF_SCRATCH_ERROR = 12'h080;") != null);
    // wo/rw registers must not get a reset localparam.
    try std.testing.expect(std.mem.indexOf(u8, out, "MATMUL_RST_CTRL") == null);
    // COLS_MAX flows to the RTL so decode_top can drive the kernel from it.
    try std.testing.expect(std.mem.indexOf(u8, out, "MATMUL_COLS_MAX = 8;") != null);
}

test "address map and caps are internally consistent" {
    try std.testing.expectEqual(@as(usize, 4), addr.dma_w.len);
    try std.testing.expectEqual(@as(usize, addr.dma_w.len), @as(usize, resetOf("WEIGHT_PORTS")));
    try std.testing.expectEqual(@as(i64, 0xA000_0000), addr.dma_w[0]);
    try std.testing.expectEqual(@as(i64, 0xA004_0000), addr.dma_a);
    try std.testing.expectEqual(@as(i64, 0xA005_0000), addr.kernel);
    try std.testing.expectEqual(@as(u32, 8), caps.cols_max);
}

test "emitAddrTcl matches the deployed address map" {
    var buf: [4096]u8 = undefined;
    const out = try emitAddrTcl(&buf);
    try std.testing.expect(std.mem.indexOf(u8, out, "set matmul_address_map {") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "{dma_w0 S_AXI_LITE 0xA0000000}") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "{dma_w3 S_AXI_LITE 0xA0030000}") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "{dma_a S_AXI_LITE 0xA0040000}") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "{kernel S_AXI 0xA0050000}") != null);
}
