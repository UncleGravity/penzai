//! Cosim for matmul_top: drives the deployable AXI-Lite wrapper with
//! four 128-bit resident weight streams and checks that the top zips them into
//! the same 512-bit core order as packWeightsWide.

const std = @import("std");
const q1a8 = @import("q1a8");
const pack = @import("pack");
const ref = @import("matmul_ref");

const c = @cImport(@cInclude("shim.h"));

const CYCLE_LIMIT: usize = 1_000_000;
const PORTS: usize = 4;
const ROWS = q1a8.ROWS;
const ROWS_PER_PORT = ROWS / PORTS;
const PORT_BEAT_BYTES = ROWS_PER_PORT * 4;
const RESULT_BYTES_PER_RB = q1a8.RESULT_BYTES_PER_ROWBLOCK;

const REG_CTRL: u8 = 0x08;
const REG_NUM_Q1_BLOCKS: u8 = 0x10;
const REG_NUM_ROWBLOCKS: u8 = 0x14;
const REG_NUM_COLS: u8 = 0x38;

comptime {
    if (ROWS != 16) @compileError("top cosim expects ROWS=16");
    if (ROWS_PER_PORT != 4) @compileError("top cosim expects four 4-row weight ports");
    if (PORT_BEAT_BYTES != 16) @compileError("top cosim expects 128-bit weight ports");
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

fn weightPortBytes(num_rb: usize, q1_blocks: usize) usize {
    return num_rb * q1_blocks * (1 + q1a8.Q8_SUBBLOCKS) * PORT_BEAT_BYTES;
}

fn packWeightPorts(
    rows: usize,
    q1_blocks: usize,
    weight_bits: []const u128,
    weight_scales: []const f16,
    ports: [PORTS][]u8,
) void {
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
                for (0..q1a8.Q8_SUBBLOCKS) |sub| {
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
    for (&words, 0..) |*word, lane| {
        word.* = std.mem.readInt(u32, bytes[base + lane * 4 ..][0..4], .little);
    }
    return words;
}

fn runTop(
    a: std.mem.Allocator,
    rows: usize,
    q1_blocks: usize,
    num_cols: usize,
    port_bytes: [PORTS][]const u8,
    act_bytes: []const u8,
    stagger_port2: bool,
) ![]u8 {
    const num_rb = rows / ROWS;
    var dut = Dut.init();
    defer dut.deinit();
    reset(&dut);

    axiWrite(&dut, REG_NUM_Q1_BLOCKS, @intCast(q1_blocks));
    axiWrite(&dut, REG_NUM_ROWBLOCKS, @intCast(num_rb));
    axiWrite(&dut, REG_NUM_COLS, @intCast(num_cols));
    axiWrite(&dut, REG_CTRL, 1);

    const a_beats = std.mem.bytesAsSlice(u64, act_bytes);
    const w_beats = port_bytes[0].len / PORT_BEAT_BYTES;
    var wi = [_]usize{0} ** PORTS;
    var ai: usize = 0;
    var ri: usize = 0;
    var saw_last = false;
    const result = try a.alloc(u8, num_rb * num_cols * RESULT_BYTES_PER_RB);

    var cycle: usize = 0;
    while (cycle < CYCLE_LIMIT) : (cycle += 1) {
        var w_valid = [_]bool{false} ** PORTS;
        for (0..PORTS) |port| {
            const have = wi[port] < w_beats;
            const hold = stagger_port2 and port == 2 and have and cycle % 11 == 3;
            const valid = have and !hold;
            w_valid[port] = valid;
            const words = if (have) readPortBeat(port_bytes[port], wi[port]) else [_]u32{0} ** ROWS_PER_PORT;
            c.dut_set_w(dut.h, @intCast(port), &words, @intFromBool(valid));
        }

        const a_valid = ai < a_beats.len;
        c.dut_set_a(dut.h, if (a_valid) a_beats[ai] else 0, @intFromBool(a_valid));
        c.dut_set_m_ready(dut.h, 1);
        c.dut_eval(dut.h);

        var w_fire = [_]bool{false} ** PORTS;
        for (0..PORTS) |port| {
            w_fire[port] = w_valid[port] and c.dut_w_ready(dut.h, @intCast(port)) != 0;
        }
        const a_fire = a_valid and c.dut_a_ready(dut.h) != 0;
        if (c.dut_m_valid(dut.h) != 0) {
            if (ri + 8 > result.len) return error.TooManyResultBeats;
            std.mem.writeInt(u64, result[ri..][0..8], c.dut_m_data(dut.h), .little);
            ri += 8;
            saw_last = c.dut_m_last(dut.h) != 0;
        }

        dut.step();
        for (0..PORTS) |port| {
            if (w_fire[port]) wi[port] += 1;
        }
        if (a_fire) ai += 1;
        if (saw_last and ri == result.len) break;
    }

    if (!saw_last or ri != result.len) return error.MissingResultBeats;
    return result;
}

fn runCase(a: std.mem.Allocator, rows: usize, blocks: usize, cols: usize, seed: u64, stagger: bool) !void {
    const num_rb = rows / ROWS;
    var prng = std.Random.DefaultPrng.init(seed);
    const rnd = prng.random();

    const bits = try a.alloc(u128, rows * blocks);
    defer a.free(bits);
    const wscales = try a.alloc(f16, rows * blocks);
    defer a.free(wscales);
    for (bits) |*b| b.* = (@as(u128, rnd.int(u64)) << 64) | rnd.int(u64);
    for (wscales) |*s| s.* = @floatCast(0.01 + rnd.float(f32) * 0.2);

    var port_storage: [PORTS][]u8 = undefined;
    var port_views: [PORTS][]const u8 = undefined;
    for (&port_storage, &port_views) |*storage, *view| {
        storage.* = try a.alloc(u8, weightPortBytes(num_rb, blocks));
        view.* = storage.*;
    }
    defer for (port_storage) |storage| a.free(storage);
    packWeightPorts(rows, blocks, bits, wscales, port_storage);

    const act_len = cols * pack.actBytes(blocks);
    const acts = try a.alloc(u8, act_len);
    defer a.free(acts);
    const expected = try a.alloc(f32, cols * rows);
    defer a.free(expected);
    const column = try a.alloc(f32, blocks * q1a8.Q1_BLOCK);
    defer a.free(column);
    const aquants = try a.alloc(i8, column.len);
    defer a.free(aquants);
    const ascales = try a.alloc(f16, blocks * q1a8.Q8_SUBBLOCKS);
    defer a.free(ascales);

    for (0..cols) |col| {
        for (column) |*v| v.* = (rnd.float(f32) - 0.5) * 4.0;
        pack.quantizeActs(column, aquants, ascales);
        pack.packActs(blocks, aquants, ascales, acts[col * pack.actBytes(blocks) ..][0..pack.actBytes(blocks)]);
        ref.scaledOutput(.{
            .rows = rows,
            .q1_blocks = blocks,
            .weight_bits = bits,
            .weight_scales = wscales,
            .act_quants = aquants,
            .act_scales = ascales,
        }, expected[col * rows ..][0..rows]);
    }

    const got = try runTop(a, rows, blocks, cols, port_views, acts, stagger);
    defer a.free(got);

    var max_rel: f32 = 0;
    for (0..cols) |col| {
        for (0..num_rb) |rb| {
            const chunk = got[(rb * cols + col) * RESULT_BYTES_PER_RB ..][0..RESULT_BYTES_PER_RB];
            for (0..ROWS / 2) |beat| {
                const word = std.mem.readInt(u64, chunk[beat * 8 ..][0..8], .little);
                const row0 = rb * ROWS + beat * 2;
                const got0: f32 = @bitCast(@as(u32, @truncate(word)));
                const got1: f32 = @bitCast(@as(u32, @truncate(word >> 32)));
                const exp0 = expected[col * rows + row0];
                const exp1 = expected[col * rows + row0 + 1];
                max_rel = @max(max_rel, @abs(got0 - exp0) / @max(@abs(exp0), 1.0));
                max_rel = @max(max_rel, @abs(got1 - exp1) / @max(@abs(exp1), 1.0));
            }
        }
    }

    std.debug.print("top case rows={d} blocks={d} cols={d} stagger={d} ok={d} max_rel={d:.4}\n", .{
        rows, blocks, cols, @intFromBool(stagger), @intFromBool(max_rel <= 0.02), max_rel,
    });
    if (max_rel > 0.02) return error.ResultMismatch;
}

pub fn main() !void {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer _ = gpa.deinit();
    const a = gpa.allocator();

    try runCase(a, ROWS, 2, 3, 0x4001, false);
    try runCase(a, ROWS * 2, 3, 1, 0x4002, true);
    std.debug.print("all matmul top cosim cases passed\n", .{});
}
