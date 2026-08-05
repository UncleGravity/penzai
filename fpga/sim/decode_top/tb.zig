//! Cosim for decode_top: the deployable AXI-Lite wrapper for the fixed-point gemm kernel.
//! Drives the four 128-bit resident weight streams + acts, writes the config registers, and
//! checks the result stream BIT-EXACT against matmul_ref.windowedFixedOutput (the fixed window
//! decode_top bakes in — no EMIN register). Proves the four-port zip feeds gemm_kernel in the
//! right order through the AXI/zip BFM.

const std = @import("std");
const layout = @import("layout");
const pack = @import("pack");
const ref = @import("matmul_ref");

const c = @cImport(@cInclude("shim.h"));

const CYCLE_LIMIT: usize = 2_000_000;
const PORTS: usize = 4;
const ROWS = layout.ROWS;
const ROWS_PER_PORT = ROWS / PORTS;
const PORT_BEAT_BYTES = ROWS_PER_PORT * 4;

const REG_CTRL: u8 = 0x08;
const REG_NUM_Q1_BLOCKS: u8 = 0x10;
const REG_NUM_ROWBLOCKS: u8 = 0x14;
const REG_NUM_COLS: u8 = 0x38;
const REG_WEIGHT_FMT: u8 = 0x44;
const REG_NUM_ROWS: u8 = 0x48;

const WEIGHT_FMT_BINARY: u32 = 1;
const WEIGHT_FMT_TERNARY: u32 = 2;

comptime {
    if (ROWS != 16) @compileError("top cosim expects ROWS=16");
    if (ROWS_PER_PORT != 4) @compileError("top cosim expects four 4-row weight ports");
}

const Dut = struct {
    h: *c.Dut,
    fn init() Dut {
        return .{ .h = c.dut_new().? };
    }
    fn deinit(self: *Dut) void {
        c.dut_free(self.h);
    }
    fn step(self: *Dut) void {
        c.dut_set_clk(self.h, 1);
        c.dut_eval(self.h);
        c.dut_set_clk(self.h, 0);
        c.dut_eval(self.h);
    }
};

fn axiWrite(dut: *Dut, addr: u8, value: u32) void {
    c.dut_set_axi_write(dut.h, addr, value, 1);
    dut.step();
    dut.step();
    c.dut_set_axi_idle(dut.h);
    dut.step();
}

fn reset(dut: *Dut) void {
    c.dut_set_rst_n(dut.h, 0);
    c.dut_set_axi_idle(dut.h);
    c.dut_set_a(dut.h, 0, 0);
    c.dut_set_m_ready(dut.h, 1);
    var zero = [_]u32{0} ** ROWS_PER_PORT;
    for (0..PORTS) |port| c.dut_set_w(dut.h, @intCast(port), &zero, 0);
    c.dut_set_clk(dut.h, 0);
    c.dut_eval(dut.h);
    for (0..4) |_| dut.step();
    c.dut_set_rst_n(dut.h, 1);
    dut.step();
}

fn weightPortBytes(num_rb: usize, q1_blocks: usize, weight_fmt: u32) usize {
    const beats_per_block: usize = if (weight_fmt == WEIGHT_FMT_TERNARY) layout.ternary_beats_per_port_block else 1 + layout.Q8_SUBBLOCKS;
    return num_rb * q1_blocks * beats_per_block * PORT_BEAT_BYTES;
}

// Pack weights into the four resident 128-bit port streams (4 rows each), matching the
// host's packWeightsFromLogical port split that decode_top's zip reassembles.
fn packWeightPorts(rows: usize, q1_blocks: usize, weight_bits: []const u128, weight_scales: []const f16, ports: [PORTS][]u8) void {
    const num_rb = rows / ROWS;
    for (0..PORTS) |port| {
        var off: usize = 0;
        for (0..num_rb) |rb| {
            for (0..q1_blocks) |blk| {
                @memset(ports[port][off..][0..PORT_BEAT_BYTES], 0);
                for (0..ROWS_PER_PORT) |lane| {
                    const row = rb * ROWS + port * ROWS_PER_PORT + lane;
                    const scale: u16 = @bitCast(weight_scales[row * q1_blocks + blk]);
                    std.mem.writeInt(u16, ports[port][off + lane * 4 ..][0..2], scale, .little);
                }
                off += PORT_BEAT_BYTES;
                for (0..layout.Q8_SUBBLOCKS) |sub| {
                    for (0..ROWS_PER_PORT) |lane| {
                        const row = rb * ROWS + port * ROWS_PER_PORT + lane;
                        const bits = weight_bits[row * q1_blocks + blk];
                        const word: u32 = @truncate(bits >> @intCast(sub * 32));
                        std.mem.writeInt(u32, ports[port][off + lane * 4 ..][0..4], word, .little);
                    }
                    off += PORT_BEAT_BYTES;
                }
            }
        }
    }
}

fn readPortBeat(bytes: []const u8, index: usize) [ROWS_PER_PORT]u32 {
    var words: [ROWS_PER_PORT]u32 = undefined;
    const base = index * PORT_BEAT_BYTES;
    for (&words, 0..) |*word, lane| word.* = std.mem.readInt(u32, bytes[base + lane * 4 ..][0..4], .little);
    return words;
}

fn runTop(a: std.mem.Allocator, physical_rows: usize, logical_rows: usize, program_rows: usize, q1_blocks: usize, num_cols: usize, weight_fmt: u32, port_bytes: [PORTS][]const u8, act_bytes: []const u8) ![]u8 {
    const num_rb = physical_rows / ROWS;
    var dut = Dut.init();
    defer dut.deinit();
    reset(&dut);

    axiWrite(&dut, REG_NUM_Q1_BLOCKS, @intCast(q1_blocks));
    axiWrite(&dut, REG_NUM_ROWBLOCKS, @intCast(num_rb));
    axiWrite(&dut, REG_NUM_COLS, @intCast(num_cols));
    axiWrite(&dut, REG_NUM_ROWS, @intCast(program_rows));
    axiWrite(&dut, REG_WEIGHT_FMT, weight_fmt);
    // No EMIN write: v9 bakes the window floor in (decode_top.EMIN_FLOOR), no register.
    axiWrite(&dut, REG_CTRL, 1);

    const a_beats = std.mem.bytesAsSlice(u64, act_bytes);
    const w_beats = port_bytes[0].len / PORT_BEAT_BYTES;
    var wi = [_]usize{0} ** PORTS;
    var ai: usize = 0;
    var ri: usize = 0;
    var saw_last = false;
    const result = try a.alloc(u8, logical_rows * num_cols * @sizeOf(f32));

    var cycle: usize = 0;
    while (cycle < CYCLE_LIMIT) : (cycle += 1) {
        var w_valid = [_]bool{false} ** PORTS;
        for (0..PORTS) |port| {
            const valid = wi[port] < w_beats;
            w_valid[port] = valid;
            const words = if (valid) readPortBeat(port_bytes[port], wi[port]) else [_]u32{0} ** ROWS_PER_PORT;
            c.dut_set_w(dut.h, @intCast(port), &words, @intFromBool(valid));
        }
        const a_valid = ai < a_beats.len;
        c.dut_set_a(dut.h, if (a_valid) a_beats[ai] else 0, @intFromBool(a_valid));
        c.dut_set_m_ready(dut.h, 1);
        c.dut_eval(dut.h);

        var w_fire = [_]bool{false} ** PORTS;
        for (0..PORTS) |port| w_fire[port] = w_valid[port] and c.dut_w_ready(dut.h, @intCast(port)) != 0;
        const a_fire = a_valid and c.dut_a_ready(dut.h) != 0;
        if (c.dut_m_valid(dut.h) != 0) {
            const keep: u8 = @intCast(c.dut_m_keep(dut.h));
            const beat_bytes: usize = switch (keep) {
                0xFF => 8,
                0x0F => 4,
                else => return error.InvalidTkeep,
            };
            if (ri + beat_bytes > result.len) return error.TooManyResultBytes;
            var beat: [8]u8 = undefined;
            std.mem.writeInt(u64, &beat, c.dut_m_data(dut.h), .little);
            @memcpy(result[ri..][0..beat_bytes], beat[0..beat_bytes]);
            ri += beat_bytes;
            if (c.dut_m_last(dut.h) != 0) {
                if (saw_last or ri != result.len) return error.InvalidTlast;
                saw_last = true;
            }
        }

        dut.step();
        for (0..PORTS) |port| {
            if (w_fire[port]) wi[port] += 1;
        }
        if (a_fire) ai += 1;
        if (saw_last and ri == result.len) break;
    }
    if (!saw_last or ri != result.len) return error.MissingResultBytes;
    return result;
}

fn runCase(a: std.mem.Allocator, num_rb: usize, logical_rows_raw: ?usize, program_rows_raw: ?usize, blocks: usize, num_cols: usize, ternary: bool, seed: u64, note: []const u8) !void {
    const physical_rows = num_rb * ROWS;
    const logical_rows = logical_rows_raw orelse physical_rows;
    const program_rows = program_rows_raw orelse logical_rows;
    if (logical_rows <= physical_rows - ROWS or logical_rows > physical_rows) return error.InvalidLogicalRows;
    var prng = std.Random.DefaultPrng.init(seed);
    const rnd = prng.random();

    const bits = try a.alloc(u128, physical_rows * blocks);
    defer a.free(bits);
    const nonzero = try a.alloc(u128, physical_rows * blocks);
    defer a.free(nonzero);
    const wscales = try a.alloc(f16, physical_rows * blocks);
    defer a.free(wscales);
    const wscales_hi = try a.alloc(f16, physical_rows * blocks);
    defer a.free(wscales_hi);
    for (bits) |*b| b.* = (@as(u128, rnd.int(u64)) << 64) | rnd.int(u64);
    for (nonzero) |*nz| nz.* = if (ternary) (@as(u128, rnd.int(u64)) << 64) | rnd.int(u64) else std.math.maxInt(u128);
    for (wscales) |*s| {
        const e: u16 = rnd.intRangeAtMost(u16, 13, 17); // narrow band → covering window
        s.* = @bitCast((e << 10) | rnd.intRangeLessThan(u16, 0, 1024));
    }
    for (wscales_hi) |*s| {
        const e: u16 = rnd.intRangeAtMost(u16, 13, 17);
        s.* = @bitCast((e << 10) | rnd.intRangeLessThan(u16, 0, 1024));
    }

    var port_storage: [PORTS][]u8 = undefined;
    var port_views: [PORTS][]const u8 = undefined;
    const weight_fmt: u32 = if (ternary) WEIGHT_FMT_TERNARY else WEIGHT_FMT_BINARY;
    for (&port_storage, &port_views) |*storage, *view| {
        storage.* = try a.alloc(u8, weightPortBytes(num_rb, blocks, weight_fmt));
        view.* = storage.*;
    }
    defer for (port_storage) |storage| a.free(storage);
    if (ternary) {
        const wide = try a.alloc(u8, pack.ternaryWeightBytesWide(num_rb, blocks));
        defer a.free(wide);
        pack.packTernaryWeightsWide(physical_rows, blocks, bits, nonzero, wscales, wscales_hi, wide);
        for (0..num_rb * blocks) |resident_block| {
            for (0..layout.ternary_beats_per_port_block) |beat| {
                for (0..PORTS) |port| {
                    const src = (resident_block * layout.ternary_beats_per_port_block + beat) * pack.WIDE_BEAT_BYTES + port * PORT_BEAT_BYTES;
                    const dst = (resident_block * layout.ternary_beats_per_port_block + beat) * PORT_BEAT_BYTES;
                    @memcpy(port_storage[port][dst..][0..PORT_BEAT_BYTES], wide[src..][0..PORT_BEAT_BYTES]);
                }
            }
        }
    } else {
        packWeightPorts(physical_rows, blocks, bits, wscales, port_storage);
    }

    const acts = try a.alloc(u8, num_cols * pack.actBytes(blocks));
    defer a.free(acts);
    const aquants = try a.alloc(i8, num_cols * blocks * layout.Q1_BLOCK);
    defer a.free(aquants);
    const ascales = try a.alloc(f16, num_cols * blocks * layout.Q8_SUBBLOCKS);
    defer a.free(ascales);
    const column = try a.alloc(f32, blocks * layout.Q1_BLOCK);
    defer a.free(column);
    const aq_per = blocks * layout.Q1_BLOCK;
    const as_per = blocks * layout.Q8_SUBBLOCKS;
    for (0..num_cols) |col| {
        for (column) |*v| v.* = (rnd.float(f32) - 0.5) * 4.0;
        const aq = aquants[col * aq_per ..][0..aq_per];
        const as_ = ascales[col * as_per ..][0..as_per];
        pack.quantizeActs(column, aq, as_);
        pack.packActs(blocks, aq, as_, acts[col * pack.actBytes(blocks) ..][0..pack.actBytes(blocks)]);
    }

    var probs = try a.alloc(ref.Problem, num_cols);
    defer a.free(probs);
    for (0..num_cols) |col| probs[col] = .{
        .rows = physical_rows,
        .q1_blocks = blocks,
        .weight_bits = bits,
        .weight_scales = wscales,
        .weight_nonzero = if (ternary) nonzero else null,
        .weight_scales_hi = if (ternary) wscales_hi else null,
        .act_quants = aquants[col * aq_per ..][0..aq_per],
        .act_scales = ascales[col * as_per ..][0..as_per],
    };
    const w = ref.fixedWindow(); // the deployed fixed window (emin -48, ACC_W 104) — no calibration

    const expected = try a.alloc(f32, num_cols * physical_rows);
    defer a.free(expected);
    var sats: usize = 0;
    for (0..num_cols) |col| {
        var s: usize = 0;
        ref.windowedFixedOutput(probs[col], w, expected[col * physical_rows ..][0..physical_rows], &s);
        sats += s;
    }

    const got = try runTop(a, physical_rows, logical_rows, program_rows, blocks, num_cols, weight_fmt, port_views, acts);
    defer a.free(got);

    var mism: usize = 0;
    var result_offset: usize = 0;
    for (0..num_rb) |rb| {
        const row0 = rb * ROWS;
        const valid_rows = @min(ROWS, logical_rows - row0);
        for (0..num_cols) |col| {
            for (0..valid_rows) |lane| {
                const result_bits = std.mem.readInt(u32, got[result_offset..][0..4], .little);
                result_offset += 4;
                const expected_bits: u32 = @bitCast(expected[col * physical_rows + row0 + lane]);
                if (result_bits != expected_bits) mism += 1;
            }
        }
    }
    if (result_offset != got.len) return error.InvalidResultLength;
    std.debug.print("decode_top case rows={d}/{d} blocks={d} cols={d} emin={d} sats={d} mism={d}  {s}\n", .{ logical_rows, physical_rows, blocks, num_cols, w.emin, sats, mism, note });
    if (sats != 0) return error.WindowTooNarrow;
    if (mism != 0) return error.ResultMismatch;
}

pub fn main() !void {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer _ = gpa.deinit();
    const a = gpa.allocator();

    try runCase(a, 1, null, null, 2, 1, false, 0x4101, "decode, 1 rb");
    try runCase(a, 2, null, null, 3, 3, false, 0x4102, "multi-rb prefill C=3");
    try runCase(a, 1, null, null, 4, 8, false, 0x4103, "prefill C=8 full bank");
    try runCase(a, 1, null, null, 76, 1, false, 0x4104, "decode, K=9728 (4B down, 76 blk)");
    try runCase(a, 1, null, null, 96, 1, false, 0x4105, "decode, K=12288 (8B down, 96 blk)");
    try runCase(a, 3, null, null, 96, 4, false, 0x4106, "prefill, K=12288 96 blk, multi-rb C=4");
    try runCase(a, 2, null, null, 3, 3, true, 0x4107, "ternary multi-rb prefill C=3");
    try runCase(a, 1, null, 0, 2, 1, false, 0x4108, "legacy zero NUM_ROWS emits full rowblock");
    try runCase(a, 1, 5, null, 2, 1, false, 0x4109, "partial one-column output, odd final TKEEP");
    try runCase(a, 2, 22, null, 2, 1, false, 0x410A, "partial one-column output, even final TKEEP");
    std.debug.print("all decode_top cosim cases passed (decode_top === windowedFixedOutput, bit-exact)\n", .{});
}
