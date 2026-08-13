//! Cosim for decode_top: the deployable AXI-Lite wrapper for the fixed-point gemm kernel.
//! Drives the four 128-bit resident weight streams + acts, writes the config registers, and
//! checks the result stream BIT-EXACT against matmul_ref.windowedFixedOutput (the fixed window
//! decode_top bakes in — no EMIN register). Proves the four-port zip feeds gemm_kernel in the
//! right order through the AXI/zip BFM.

const std = @import("std");
const layout = @import("layout");
const shared_layout = @import("shared_layout");
const pack = @import("pack");
const ref = @import("matmul_ref");
const swiglu_ref = @import("swiglu_ref");

const c = @cImport(@cInclude("shim.h"));

const CYCLE_LIMIT: usize = 2_000_000;
const PORTS: usize = 4;
const ROWS = layout.ROWS;
const ROWS_PER_PORT = ROWS / PORTS;
const PORT_BEAT_BYTES = ROWS_PER_PORT * 4;

const REG_CTRL: u8 = 0x08;
const REG_STATUS: u8 = 0x0C;
const REG_NUM_Q1_BLOCKS: u8 = 0x10;
const REG_NUM_ROWBLOCKS: u8 = 0x14;
const REG_CYCLES: u8 = 0x18;
const REG_W_BEATS: u8 = 0x2C;
const REG_A_BEATS: u8 = 0x30;
const REG_R_BEATS: u8 = 0x34;
const REG_NUM_COLS: u8 = 0x38;
const REG_WEIGHT_FMT: u8 = 0x44;
const REG_NUM_ROWS: u8 = 0x48;
const REG_ACT_MODE: u8 = 0x4C;
const REG_ACT_EPOCH: u8 = 0x50;
const REG_ACT_STATE: u8 = 0x54;
const REG_LOADED_EPOCH: u8 = 0x58;
const REG_LOADED_Q1_BLOCKS: u8 = 0x5C;
const REG_LOADED_COLS: u8 = 0x60;
const REG_QUANT_STATUS: u8 = 0x64;
const REG_SCRATCH_MODE: u8 = 0x68;
const REG_SCRATCH_ROLE: u8 = 0x6C;
const REG_SCRATCH_ROWS: u8 = 0x70;
const REG_SCRATCH_TOKENS: u8 = 0x74;
const REG_SCRATCH_CTRL: u8 = 0x78;
const REG_SCRATCH_STATUS: u8 = 0x7C;
const REG_SCRATCH_ERROR: u8 = 0x80;

const ACT_PACKED_LOAD: u32 = 0;
const ACT_REUSE: u32 = 1;
const ACT_RAW_LOAD: u32 = 2;
const ACT_SCRATCH_SWIGLU: u32 = 3;
const QUANT_NONFINITE: u32 = 1 << 0;
const QUANT_FRAME: u32 = 1 << 2;

const SCRATCH_MODE_DDR: u32 = 0;
const SCRATCH_MODE_TEE: u32 = 1;
const SCRATCH_MODE_DRAIN: u32 = 2;
const SCRATCH_MODE_ONLY: u32 = 3;
const SCRATCH_ROLE_X0: u32 = 1;
const SCRATCH_ROLE_X1: u32 = 2;
const SCRATCH_CTRL_DRAIN_START: u32 = 1;
const SCRATCH_CTRL_ABORT: u32 = 1 << 1;
const SCRATCH_CTRL_SECTION_BEGIN: u32 = 1 << 2;
const SCRATCH_WRITER_BUSY: u32 = 1 << 0;
const SCRATCH_WRITER_DONE: u32 = 1 << 1;
const SCRATCH_DRAIN_BUSY: u32 = 1 << 2;
const SCRATCH_DRAIN_DONE: u32 = 1 << 3;
const SCRATCH_ANY_ERROR: u32 = 1 << 4;
const SCRATCH_X0_VALID: u32 = 1 << 6;
const SCRATCH_X1_VALID: u32 = 1 << 7;
const SCRATCH_CONSUMER_BUSY: u32 = 1 << 9;
const SCRATCH_CONSUMER_DONE: u32 = 1 << 10;
const SCRATCH_SECTION_ACTIVE: u32 = 1 << 11;
const SCRATCH_SECTION_DONE: u32 = 1 << 12;
const SCRATCH_ERROR_CONFIG: u32 = 1 << 0;
const SCRATCH_ERROR_WRITER: u32 = 1 << 1;
const SCRATCH_ERROR_ABORT: u32 = 1 << 2;
const SCRATCH_ERROR_STALE: u32 = 1 << 4;

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

fn axiRead(dut: *Dut, addr: u8) !u32 {
    c.dut_set_axi_read(dut.h, addr, 1);
    var value: u32 = 0;
    for (0..16) |_| {
        dut.step();
        if (c.dut_axi_rvalid(dut.h) != 0) {
            value = c.dut_axi_rdata(dut.h);
            c.dut_set_axi_idle(dut.h);
            dut.step();
            return value;
        }
    }
    return error.AxiReadTimeout;
}

fn reset(dut: *Dut) void {
    c.dut_set_rst_n(dut.h, 0);
    c.dut_set_axi_idle(dut.h);
    c.dut_set_a(dut.h, 0, 0, 0);
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

const ProjectionRun = struct {
    result: []u8,
    cycles: u32,
    weight_beats: u32,
    act_beats: u32,
    result_beats: u32,
    act_state: u32,
    quant_status: u32,
};

const RawFault = enum {
    none,
    early_last,
    no_last,
};

fn configureProjection(
    dut: *Dut,
    physical_rows: usize,
    logical_rows: usize,
    q1_blocks: usize,
    num_cols: usize,
    weight_fmt: u32,
    act_mode: u32,
    act_epoch: u32,
) void {
    axiWrite(dut, REG_NUM_Q1_BLOCKS, @intCast(q1_blocks));
    axiWrite(dut, REG_NUM_ROWBLOCKS, @intCast(physical_rows / ROWS));
    axiWrite(dut, REG_NUM_COLS, @intCast(num_cols));
    axiWrite(dut, REG_NUM_ROWS, @intCast(logical_rows));
    axiWrite(dut, REG_WEIGHT_FMT, weight_fmt);
    axiWrite(dut, REG_ACT_MODE, act_mode);
    axiWrite(dut, REG_ACT_EPOCH, act_epoch);
    axiWrite(dut, REG_CTRL, 1);
}

fn configureScratch(dut: *Dut, mode: u32, role: u32, rows: usize, tokens: usize) void {
    axiWrite(dut, REG_SCRATCH_ROLE, role);
    axiWrite(dut, REG_SCRATCH_ROWS, @intCast(rows));
    axiWrite(dut, REG_SCRATCH_TOKENS, @intCast(tokens));
    axiWrite(dut, REG_SCRATCH_MODE, mode);
}

fn runProjection(
    a: std.mem.Allocator,
    dut: *Dut,
    physical_rows: usize,
    logical_rows: usize,
    q1_blocks: usize,
    num_cols: usize,
    weight_fmt: u32,
    act_mode: u32,
    act_epoch: u32,
    port_bytes: [PORTS][]const u8,
    activation_beats: []const u64,
    activation_limit: usize,
    raw_fault: RawFault,
    stall_output: bool,
    expect_error: bool,
) !ProjectionRun {
    configureProjection(dut, physical_rows, logical_rows, q1_blocks, num_cols, weight_fmt, act_mode, act_epoch);

    const w_beats = port_bytes[0].len / PORT_BEAT_BYTES;
    for (port_bytes[1..]) |port| {
        if (port.len / PORT_BEAT_BYTES != w_beats) return error.WeightPortLengthMismatch;
    }
    if (activation_limit > activation_beats.len) return error.ActivationLimitTooLarge;

    var wi = [_]usize{0} ** PORTS;
    var ai: usize = 0;
    var ri: usize = 0;
    var saw_last = false;
    const result_len = if (expect_error) 0 else logical_rows * num_cols * @sizeOf(f32);
    const result = try a.alloc(u8, result_len);
    errdefer a.free(result);

    c.dut_set_m_ready(dut.h, @intFromBool(!expect_error));
    var cycle: usize = 0;
    while (cycle < CYCLE_LIMIT) : (cycle += 1) {
        var w_valid = [_]bool{false} ** PORTS;
        for (0..PORTS) |port| {
            const valid = wi[port] < w_beats;
            w_valid[port] = valid;
            const words = if (valid) readPortBeat(port_bytes[port], wi[port]) else [_]u32{0} ** ROWS_PER_PORT;
            c.dut_set_w(dut.h, @intCast(port), &words, @intFromBool(valid));
        }

        const a_valid = ai < activation_limit;
        const a_last = a_valid and switch (raw_fault) {
            .none => ai + 1 == activation_limit,
            .early_last => ai == 0,
            .no_last => false,
        };
        c.dut_set_a(dut.h, if (a_valid) activation_beats[ai] else 0, @intFromBool(a_valid), @intFromBool(a_last));
        const result_ready = !expect_error and (!stall_output or cycle % 11 < 7);
        c.dut_set_m_ready(dut.h, @intFromBool(result_ready));
        c.dut_eval(dut.h);

        var w_fire = [_]bool{false} ** PORTS;
        for (0..PORTS) |port| w_fire[port] = w_valid[port] and c.dut_w_ready(dut.h, @intCast(port)) != 0;
        const a_fire = a_valid and c.dut_a_ready(dut.h) != 0;
        if (act_mode == ACT_REUSE and c.dut_a_ready(dut.h) != 0)
            return error.ReuseRequestedActivationInput;

        if (c.dut_m_valid(dut.h) != 0 and result_ready) {
            if (expect_error) return error.ErrorRunProducedOutput;
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
        if (!expect_error and saw_last) break;
        if (expect_error and ai == activation_limit) break;
    }
    if (cycle >= CYCLE_LIMIT) return error.ProjectionTimeout;

    // Keep weights asserted during an error drain. Any accidental post-abort
    // consumption is caught by W_BEATS; hold result ready low so output cannot hide.
    c.dut_set_a(dut.h, 0, 0, 0);
    c.dut_set_m_ready(dut.h, @intFromBool(!expect_error));
    var done = false;
    var drain: usize = 0;
    while (drain < CYCLE_LIMIT and !done) : (drain += 256) {
        if (!expect_error) {
            var zero = [_]u32{0} ** ROWS_PER_PORT;
            for (0..PORTS) |port| c.dut_set_w(dut.h, @intCast(port), &zero, 0);
        }
        for (0..256) |_| {
            c.dut_eval(dut.h);
            if (expect_error and c.dut_m_valid(dut.h) != 0) return error.ErrorRunProducedOutput;
            dut.step();
        }
        done = (try axiRead(dut, REG_STATUS)) & 2 != 0;
        if (expect_error and c.dut_m_valid(dut.h) != 0) return error.ErrorRunProducedOutput;
    }
    if (!done) return error.DoneTimeout;

    const run: ProjectionRun = .{
        .result = result,
        .cycles = try axiRead(dut, REG_CYCLES),
        .weight_beats = try axiRead(dut, REG_W_BEATS),
        .act_beats = try axiRead(dut, REG_A_BEATS),
        .result_beats = try axiRead(dut, REG_R_BEATS),
        .act_state = try axiRead(dut, REG_ACT_STATE),
        .quant_status = try axiRead(dut, REG_QUANT_STATUS),
    };
    if (expect_error) {
        if (run.act_state & 2 == 0 or run.weight_beats != 0 or run.result_beats != 0 or
            c.dut_m_valid(dut.h) != 0)
            return error.RawFailureDidNotCloseStreams;
    } else if (!saw_last or ri != result.len or run.act_state != 1) {
        return error.ProjectionResultIncomplete;
    }
    return run;
}

fn drainScratch(a: std.mem.Allocator, dut: *Dut, role: u32, rows: usize, tokens: usize, expected: []const u8) !void {
    configureScratch(dut, SCRATCH_MODE_DRAIN, role, rows, tokens);
    axiWrite(dut, REG_SCRATCH_CTRL, SCRATCH_CTRL_DRAIN_START);

    const got = try a.alloc(u8, expected.len);
    defer a.free(got);
    var offset: usize = 0;
    var saw_last = false;
    var held_valid = false;
    var held_data: u64 = 0;
    var held_keep: u8 = 0;
    var held_last = false;

    for (0..CYCLE_LIMIT) |cycle| {
        const ready = cycle % 9 < 5;
        c.dut_set_m_ready(dut.h, @intFromBool(ready));
        c.dut_eval(dut.h);
        const valid = c.dut_m_valid(dut.h) != 0;
        if (held_valid) {
            if (!valid or c.dut_m_data(dut.h) != held_data or
                @as(u8, @intCast(c.dut_m_keep(dut.h))) != held_keep or
                (c.dut_m_last(dut.h) != 0) != held_last)
                return error.ScratchDrainChangedWhileStalled;
        }
        held_valid = valid and !ready;
        if (held_valid) {
            held_data = c.dut_m_data(dut.h);
            held_keep = @intCast(c.dut_m_keep(dut.h));
            held_last = c.dut_m_last(dut.h) != 0;
        }

        if (valid and ready) {
            if (c.dut_m_keep(dut.h) != 0xFF or offset + 8 > got.len)
                return error.InvalidScratchDrainBeat;
            std.mem.writeInt(u64, got[offset..][0..8], c.dut_m_data(dut.h), .little);
            offset += 8;
            if (c.dut_m_last(dut.h) != 0) {
                if (saw_last or offset != got.len) return error.InvalidScratchDrainLast;
                saw_last = true;
            }
        }
        dut.step();
        if (saw_last) break;
    } else return error.ScratchDrainTimeout;

    if (!std.mem.eql(u8, expected, got)) return error.ScratchDrainMismatch;
    const status = try axiRead(dut, REG_SCRATCH_STATUS);
    if (status & (SCRATCH_DRAIN_BUSY | SCRATCH_ANY_ERROR) != 0 or
        status & SCRATCH_DRAIN_DONE == 0 or try axiRead(dut, REG_SCRATCH_ERROR) != 0)
        return error.ScratchDrainStatusMismatch;
}

fn abortScratchDrain(dut: *Dut, role: u32, rows: usize, tokens: usize) !void {
    configureScratch(dut, SCRATCH_MODE_DRAIN, role, rows, tokens);
    c.dut_set_m_ready(dut.h, 0);
    axiWrite(dut, REG_SCRATCH_CTRL, SCRATCH_CTRL_DRAIN_START);

    var saw_held_beat = false;
    for (0..64) |_| {
        c.dut_set_m_ready(dut.h, 0);
        c.dut_eval(dut.h);
        if (c.dut_m_valid(dut.h) != 0) {
            saw_held_beat = true;
            break;
        }
        dut.step();
    }
    if (!saw_held_beat) return error.ScratchDrainDidNotReachHeldBeat;

    // Keep the sink closed while the control write commits. Once abort is
    // observed, the held response must be discarded rather than emitted.
    axiWrite(dut, REG_SCRATCH_CTRL, SCRATCH_CTRL_ABORT);
    c.dut_set_m_ready(dut.h, 1);
    for (0..8) |_| {
        c.dut_eval(dut.h);
        if (c.dut_m_valid(dut.h) != 0) return error.ScratchDrainEmittedAfterAbort;
        dut.step();
    }

    const status = try axiRead(dut, REG_SCRATCH_STATUS);
    const err = try axiRead(dut, REG_SCRATCH_ERROR);
    if (status & SCRATCH_DRAIN_BUSY != 0 or
        status & (SCRATCH_DRAIN_DONE | SCRATCH_ANY_ERROR) !=
            (SCRATCH_DRAIN_DONE | SCRATCH_ANY_ERROR) or
        err & SCRATCH_ERROR_ABORT == 0)
        return error.ScratchDrainAbortStatusMismatch;
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
    axiWrite(&dut, REG_ACT_MODE, 0);
    axiWrite(&dut, REG_ACT_EPOCH, 0xD00D_0001);
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
        c.dut_set_a(dut.h, if (a_valid) a_beats[ai] else 0, @intFromBool(a_valid), 0);
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
    if (try axiRead(&dut, REG_ACT_STATE) != 1 or
        try axiRead(&dut, REG_LOADED_EPOCH) != 0xD00D_0001 or
        try axiRead(&dut, REG_LOADED_Q1_BLOCKS) != q1_blocks or
        try axiRead(&dut, REG_LOADED_COLS) != num_cols)
        return error.ActivationStateReadbackMismatch;
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

fn packRawF32(values: []const f32, beats: []u64) void {
    std.debug.assert(values.len == beats.len * 2);
    for (beats, 0..) |*beat, i| {
        const lo: u32 = @bitCast(values[i * 2]);
        const hi: u32 = @bitCast(values[i * 2 + 1]);
        beat.* = @as(u64, lo) | (@as(u64, hi) << 32);
    }
}

fn runScratchOnlyProjection(
    dut: *Dut,
    rows: usize,
    q1_blocks: usize,
    tokens: usize,
    role: u32,
    act_mode: u32,
    epoch: u32,
    ports: [PORTS][]const u8,
    raw_beats: []const u64,
) !void {
    configureScratch(dut, SCRATCH_MODE_ONLY, role, rows, tokens);
    configureProjection(dut, rows, rows, q1_blocks, tokens, WEIGHT_FMT_BINARY, act_mode, epoch);
    const w_beats = ports[0].len / PORT_BEAT_BYTES;
    var wi = [_]usize{0} ** PORTS;
    var ai: usize = 0;
    var cycle: usize = 0;
    while (cycle < CYCLE_LIMIT and (wi[0] < w_beats or ai < raw_beats.len)) : (cycle += 1) {
        var w_valid = [_]bool{false} ** PORTS;
        for (0..PORTS) |port| {
            const valid = wi[port] < w_beats;
            w_valid[port] = valid;
            const words = if (valid) readPortBeat(ports[port], wi[port]) else [_]u32{0} ** ROWS_PER_PORT;
            c.dut_set_w(dut.h, @intCast(port), &words, @intFromBool(valid));
        }
        const a_valid = ai < raw_beats.len;
        c.dut_set_a(
            dut.h,
            if (a_valid) raw_beats[ai] else 0,
            @intFromBool(a_valid),
            @intFromBool(a_valid and ai + 1 == raw_beats.len),
        );
        c.dut_set_m_ready(dut.h, @intFromBool(cycle % 7 < 3));
        c.dut_eval(dut.h);
        if (c.dut_m_valid(dut.h) != 0) return error.ScratchOnlyExposedResult;
        var w_fire = [_]bool{false} ** PORTS;
        for (0..PORTS) |port| w_fire[port] = w_valid[port] and c.dut_w_ready(dut.h, @intCast(port)) != 0;
        const a_fire = a_valid and c.dut_a_ready(dut.h) != 0;
        if (act_mode == ACT_REUSE and c.dut_a_ready(dut.h) != 0)
            return error.ScratchOnlyReuseRequestedActivation;
        dut.step();
        for (0..PORTS) |port| {
            if (w_fire[port]) wi[port] += 1;
        }
        if (a_fire) ai += 1;
    }
    if (cycle == CYCLE_LIMIT) return error.ScratchOnlyInputTimeout;

    var zero = [_]u32{0} ** ROWS_PER_PORT;
    for (0..PORTS) |port| c.dut_set_w(dut.h, @intCast(port), &zero, 0);
    c.dut_set_a(dut.h, 0, 0, 0);
    var done = false;
    var poll: usize = 0;
    while (poll < CYCLE_LIMIT and !done) : (poll += 128) {
        for (0..128) |_| {
            c.dut_eval(dut.h);
            if (c.dut_m_valid(dut.h) != 0) return error.ScratchOnlyExposedResult;
            dut.step();
        }
        done = (try axiRead(dut, REG_STATUS)) & 2 != 0;
    }
    if (!done) return error.ScratchOnlyDoneTimeout;
    const expected_a_beats: u32 = if (act_mode == ACT_RAW_LOAD) @intCast(raw_beats.len) else 0;
    if (try axiRead(dut, REG_W_BEATS) != w_beats or
        try axiRead(dut, REG_A_BEATS) != expected_a_beats or
        try axiRead(dut, REG_R_BEATS) != 0)
        return error.ScratchOnlyCounterMismatch;
    const valid_mask = if (role == SCRATCH_ROLE_X1) SCRATCH_X1_VALID else SCRATCH_X0_VALID;
    const status = try axiRead(dut, REG_SCRATCH_STATUS);
    if (status & (SCRATCH_WRITER_DONE | valid_mask) != (SCRATCH_WRITER_DONE | valid_mask) or
        status & SCRATCH_ANY_ERROR != 0)
        return error.ScratchOnlyStatusMismatch;
}

const InternalRun = struct {
    stream: []u8,
    read_fires: usize,
    swiglu_fires: usize,
    record_fires: usize,
    cycles: usize,
    peak_pending_records: usize,
};

fn runInternalDown(
    a: std.mem.Allocator,
    dut: *Dut,
    rows: usize,
    ffn_dim: usize,
    tokens: usize,
    ports: [PORTS][]const u8,
) !InternalRun {
    configureScratch(dut, SCRATCH_MODE_DDR, SCRATCH_ROLE_X0, ffn_dim, tokens);
    configureProjection(
        dut,
        rows,
        rows,
        ffn_dim / layout.Q1_BLOCK,
        tokens,
        WEIGHT_FMT_BINARY,
        ACT_SCRATCH_SWIGLU,
        0xF15F_0002,
    );
    const w_beats = ports[0].len / PORT_BEAT_BYTES;
    const stream = try a.alloc(u8, rows * tokens * @sizeOf(f32));
    errdefer a.free(stream);
    var wi = [_]usize{0} ** PORTS;
    var out_offset: usize = 0;
    var saw_last = false;
    var read_fires: usize = 0;
    var swiglu_fires: usize = 0;
    var record_fires: usize = 0;
    var cycles: usize = 0;
    var peak_pending_records: usize = 0;

    for (0..CYCLE_LIMIT) |cycle| {
        cycles = cycle + 1;
        var w_valid = [_]bool{false} ** PORTS;
        for (0..PORTS) |port| {
            const valid = wi[port] < w_beats;
            w_valid[port] = valid;
            const words = if (valid) readPortBeat(ports[port], wi[port]) else [_]u32{0} ** ROWS_PER_PORT;
            c.dut_set_w(dut.h, @intCast(port), &words, @intFromBool(valid));
        }
        // Internal mode must ignore even a continuously asserted external source.
        c.dut_set_a(dut.h, 0x7fc0_0000_7fc0_0000, 1, 1);
        const ready = cycle % 13 < 8;
        c.dut_set_m_ready(dut.h, @intFromBool(ready));
        c.dut_eval(dut.h);
        if (c.dut_a_ready(dut.h) != 0) return error.InternalModeAcceptedExternalActivation;
        if (c.dut_dbg_scratch_read_fire(dut.h) != 0) {
            read_fires += 1;
        }
        if (c.dut_dbg_swiglu_input_fire(dut.h) != 0) {
            swiglu_fires += 1;
        }
        if (c.dut_dbg_internal_record_done(dut.h) != 0) {
            if (read_fires < (record_fires + 1) * 8 or
                swiglu_fires < (record_fires + 1) * shared_layout.q8_block)
                return error.InternalRecordScheduleMismatch;
            record_fires += 1;
        }
        peak_pending_records = @max(
            peak_pending_records,
            swiglu_fires / shared_layout.q8_block - record_fires,
        );
        var w_fire = [_]bool{false} ** PORTS;
        for (0..PORTS) |port| w_fire[port] = w_valid[port] and c.dut_w_ready(dut.h, @intCast(port)) != 0;
        if (c.dut_m_valid(dut.h) != 0 and ready) {
            if (c.dut_m_keep(dut.h) != 0xff or out_offset + 8 > stream.len)
                return error.InternalDownFraming;
            std.mem.writeInt(u64, stream[out_offset..][0..8], c.dut_m_data(dut.h), .little);
            out_offset += 8;
            if (c.dut_m_last(dut.h) != 0) {
                if (saw_last or out_offset != stream.len) return error.InternalDownFraming;
                saw_last = true;
            }
        }
        dut.step();
        for (0..PORTS) |port| {
            if (w_fire[port]) wi[port] += 1;
        }
        if (saw_last) break;
    } else return error.InternalDownTimeout;

    c.dut_set_a(dut.h, 0, 0, 0);
    var zero = [_]u32{0} ** ROWS_PER_PORT;
    for (0..PORTS) |port| c.dut_set_w(dut.h, @intCast(port), &zero, 0);
    for (0..128) |_| dut.step();
    if ((try axiRead(dut, REG_STATUS)) & 2 == 0 or
        try axiRead(dut, REG_A_BEATS) != 0 or
        try axiRead(dut, REG_R_BEATS) != stream.len / 8 or
        try axiRead(dut, REG_QUANT_STATUS) != 0)
        return error.InternalDownCounterMismatch;
    const status = try axiRead(dut, REG_SCRATCH_STATUS);
    if (status & (SCRATCH_SECTION_DONE | SCRATCH_CONSUMER_DONE) !=
        (SCRATCH_SECTION_DONE | SCRATCH_CONSUMER_DONE) or
        status & (SCRATCH_SECTION_ACTIVE | SCRATCH_CONSUMER_BUSY | SCRATCH_ANY_ERROR |
            SCRATCH_X0_VALID | SCRATCH_X1_VALID) != 0)
        return error.InternalDownStatusMismatch;
    return .{
        .stream = stream,
        .read_fires = read_fires,
        .swiglu_fires = swiglu_fires,
        .record_fires = record_fires,
        .cycles = cycles,
        .peak_pending_records = peak_pending_records,
    };
}

fn abortInternalAndRestart(
    dut: *Dut,
    rows: usize,
    ffn_dim: usize,
    tokens: usize,
) !void {
    configureScratch(dut, SCRATCH_MODE_DDR, SCRATCH_ROLE_X0, ffn_dim, tokens);
    configureProjection(
        dut,
        rows,
        rows,
        ffn_dim / layout.Q1_BLOCK,
        tokens,
        WEIGHT_FMT_BINARY,
        ACT_SCRATCH_SWIGLU,
        0xF15F_A807,
    );
    c.dut_set_m_ready(dut.h, 0);
    c.dut_set_a(dut.h, 0x7fc0_0000_7fc0_0000, 1, 1);

    var saw_request = false;
    for (0..64) |_| {
        c.dut_eval(dut.h);
        if (c.dut_a_ready(dut.h) != 0 or c.dut_m_valid(dut.h) != 0)
            return error.InternalAbortExposedExternalStream;
        if (c.dut_dbg_scratch_read_fire(dut.h) != 0) {
            saw_request = true;
            break;
        }
        dut.step();
    }
    if (!saw_request) return error.InternalAbortNoScratchRequest;

    // Abort on the first accepted read, then begin the next section without an
    // intervening reset.  Any response left by the killed consumer must drain
    // while idle and cannot deadlock or poison the new section.
    axiWrite(dut, REG_SCRATCH_CTRL, SCRATCH_CTRL_ABORT);
    c.dut_set_a(dut.h, 0, 0, 0);
    c.dut_set_m_ready(dut.h, 1);
    var status = try axiRead(dut, REG_SCRATCH_STATUS);
    if (status & (SCRATCH_CONSUMER_BUSY | SCRATCH_SECTION_ACTIVE) != 0 or
        status & (SCRATCH_CONSUMER_DONE | SCRATCH_SECTION_DONE | SCRATCH_ANY_ERROR) !=
            (SCRATCH_CONSUMER_DONE | SCRATCH_SECTION_DONE | SCRATCH_ANY_ERROR) or
        try axiRead(dut, REG_SCRATCH_ERROR) & SCRATCH_ERROR_ABORT == 0)
        return error.InternalAbortStatusMismatch;

    axiWrite(dut, REG_SCRATCH_CTRL, SCRATCH_CTRL_SECTION_BEGIN);
    status = try axiRead(dut, REG_SCRATCH_STATUS);
    if (status & SCRATCH_SECTION_ACTIVE == 0 or
        status & (SCRATCH_SECTION_DONE | SCRATCH_ANY_ERROR | SCRATCH_X0_VALID | SCRATCH_X1_VALID) != 0 or
        try axiRead(dut, REG_SCRATCH_ERROR) != 0)
        return error.InternalImmediateRestartFailed;
}

fn verifyMaxScratchBlockTarget(dut: *Dut) !void {
    const max_rows: usize = 12288;
    const max_tokens: usize = 4;
    configureScratch(dut, SCRATCH_MODE_DDR, SCRATCH_ROLE_X0, max_rows, max_tokens);
    axiWrite(dut, REG_SCRATCH_CTRL, SCRATCH_CTRL_SECTION_BEGIN);
    if (try axiRead(dut, REG_SCRATCH_STATUS) & SCRATCH_SECTION_ACTIVE == 0)
        return error.MaxScratchSectionBeginRejected;

    // The bit-exact case exercises real role commits.  For this isolated
    // production-width arithmetic check, seed matching metadata and require
    // the normal preflight/start path to snapshot the target register.
    c.dut_dbg_seed_scratch_roles(dut.h, max_rows, max_tokens);
    configureProjection(
        dut,
        ROWS,
        ROWS,
        max_rows / layout.Q1_BLOCK,
        max_tokens,
        WEIGHT_FMT_BINARY,
        ACT_SCRATCH_SWIGLU,
        0xF15F_1536,
    );
    // Accepted launches are registered after shape/ownership preflight.
    // Advance through that boundary before observing the consumer snapshot.
    dut.step();
    c.dut_eval(dut.h);
    if (c.dut_dbg_scratch_total_blocks(dut.h) != 1536 or
        c.dut_dbg_scratch_consumer_state(dut.h) == 0)
        return error.MaxScratchBlockTargetTruncated;

    axiWrite(dut, REG_SCRATCH_CTRL, SCRATCH_CTRL_ABORT);
    const status = try axiRead(dut, REG_SCRATCH_STATUS);
    if (status & (SCRATCH_CONSUMER_BUSY | SCRATCH_SECTION_ACTIVE) != 0 or
        status & (SCRATCH_SECTION_DONE | SCRATCH_ANY_ERROR) !=
            (SCRATCH_SECTION_DONE | SCRATCH_ANY_ERROR))
        return error.MaxScratchAbortFailed;
}

fn runFfnSectionCase(a: std.mem.Allocator) !void {
    const model_dim: usize = 128;
    const ffn_dim: usize = 128;
    const tokens: usize = 2;
    const q1_blocks: usize = 1;
    var prng = std.Random.DefaultPrng.init(0xF15F_5E07_10C0_0001);
    const rnd = prng.random();

    var weight_bits: [3][model_dim]u128 = undefined;
    var weight_scales: [3][model_dim]f16 = undefined;
    for (&weight_bits) |*projection| {
        for (projection) |*bits| {
            bits.* = (@as(u128, rnd.int(u64)) << 64) | rnd.int(u64);
        }
    }
    for (&weight_scales) |*projection| {
        for (projection) |*scale| {
            scale.* = @floatCast(@as(f32, 1.0 / 128.0));
        }
    }

    var port_storage: [3][PORTS][]u8 = undefined;
    var port_views: [3][PORTS][]const u8 = undefined;
    for (0..3) |projection| {
        for (0..PORTS) |port| {
            port_storage[projection][port] = try a.alloc(
                u8,
                weightPortBytes(model_dim / ROWS, q1_blocks, WEIGHT_FMT_BINARY),
            );
            port_views[projection][port] = port_storage[projection][port];
        }
        packWeightPorts(
            model_dim,
            q1_blocks,
            weight_bits[projection][0..],
            weight_scales[projection][0..],
            port_storage[projection],
        );
    }
    defer {
        for (&port_storage) |*projection| {
            for (projection) |storage| a.free(storage);
        }
    }

    var raw: [tokens * model_dim]f32 = undefined;
    for (&raw) |*value| value.* = (rnd.float(f32) - 0.5) * 0.5;
    var raw_beats: [raw.len / 2]u64 = undefined;
    packRawF32(&raw, &raw_beats);
    var raw_quants: [raw.len]i8 = undefined;
    var raw_scales: [raw.len / shared_layout.q8_block]f16 = undefined;
    try shared_layout.quantizeQ8_0(&raw, &raw_quants, &raw_scales);

    var up: [tokens * ffn_dim]f32 = undefined;
    var gate: [tokens * ffn_dim]f32 = undefined;
    const upstream = [_]*[tokens * ffn_dim]f32{ &up, &gate };
    var saturations: usize = 0;
    for (0..2) |projection| {
        for (0..tokens) |token| {
            var sat: usize = 0;
            ref.windowedFixedOutput(.{
                .rows = ffn_dim,
                .q1_blocks = q1_blocks,
                .weight_bits = weight_bits[projection][0..],
                .weight_scales = weight_scales[projection][0..],
                .act_quants = raw_quants[token * model_dim ..][0..model_dim],
                .act_scales = raw_scales[token * layout.Q8_SUBBLOCKS ..][0..layout.Q8_SUBBLOCKS],
            }, ref.fixedWindow(), upstream[projection][token * ffn_dim ..][0..ffn_dim], &sat);
            saturations += sat;
        }
    }
    if (saturations != 0) return error.WindowTooNarrow;

    var swiglu: [tokens * ffn_dim]f32 = undefined;
    for (&swiglu, gate, up) |*dst, gate_value, up_value| {
        const modeled = swiglu_ref.model(gate_value, up_value);
        if (modeled.status != 0) return error.SwigluOracleStatus;
        dst.* = @bitCast(modeled.bits);
    }
    var swiglu_quants: [swiglu.len]i8 = undefined;
    var swiglu_scales: [swiglu.len / shared_layout.q8_block]f16 = undefined;
    try shared_layout.quantizeQ8_0(&swiglu, &swiglu_quants, &swiglu_scales);
    var expected: [tokens * model_dim]f32 = undefined;
    for (0..tokens) |token| {
        var sat: usize = 0;
        ref.windowedFixedOutput(.{
            .rows = model_dim,
            .q1_blocks = q1_blocks,
            .weight_bits = weight_bits[2][0..],
            .weight_scales = weight_scales[2][0..],
            .act_quants = swiglu_quants[token * ffn_dim ..][0..ffn_dim],
            .act_scales = swiglu_scales[token * layout.Q8_SUBBLOCKS ..][0..layout.Q8_SUBBLOCKS],
        }, ref.fixedWindow(), expected[token * model_dim ..][0..model_dim], &sat);
        saturations += sat;
    }
    if (saturations != 0) return error.WindowTooNarrow;

    var dut = Dut.init();
    defer dut.deinit();
    reset(&dut);
    try verifyMaxScratchBlockTarget(&dut);
    configureScratch(&dut, SCRATCH_MODE_DDR, SCRATCH_ROLE_X0, ffn_dim, tokens);
    axiWrite(&dut, REG_SCRATCH_CTRL, SCRATCH_CTRL_SECTION_BEGIN);
    if (try axiRead(&dut, REG_SCRATCH_STATUS) & SCRATCH_SECTION_ACTIVE == 0)
        return error.SectionBeginRejected;
    try runScratchOnlyProjection(&dut, ffn_dim, q1_blocks, tokens, SCRATCH_ROLE_X1, ACT_RAW_LOAD, 0xF15F_0001, port_views[0], &raw_beats);
    try runScratchOnlyProjection(&dut, ffn_dim, q1_blocks, tokens, SCRATCH_ROLE_X0, ACT_REUSE, 0xF15F_0001, port_views[1], &.{});
    try abortInternalAndRestart(&dut, model_dim, ffn_dim, tokens);
    try runScratchOnlyProjection(&dut, ffn_dim, q1_blocks, tokens, SCRATCH_ROLE_X1, ACT_RAW_LOAD, 0xF15F_0003, port_views[0], &raw_beats);
    try runScratchOnlyProjection(&dut, ffn_dim, q1_blocks, tokens, SCRATCH_ROLE_X0, ACT_REUSE, 0xF15F_0003, port_views[1], &.{});
    const down = try runInternalDown(a, &dut, model_dim, ffn_dim, tokens, port_views[2]);
    defer a.free(down.stream);

    var offset: usize = 0;
    for (0..model_dim / ROWS) |rb| {
        for (0..tokens) |token| {
            for (0..ROWS) |lane| {
                const got = std.mem.readInt(u32, down.stream[offset..][0..4], .little);
                offset += 4;
                const want: u32 = @bitCast(expected[token * model_dim + rb * ROWS + lane]);
                if (got != want) return error.FfnSectionResultMismatch;
            }
        }
    }
    const expected_records = tokens * ffn_dim / shared_layout.q8_block;
    if (down.read_fires != expected_records * 8 or
        down.swiglu_fires != expected_records * shared_layout.q8_block or
        down.record_fires != expected_records)
        return error.FfnSectionTraversalMismatch;
    if (down.peak_pending_records < 2)
        return error.FfnSectionDidNotPipelineBlocks;
    // The former record-at-a-time walker takes 3,095 cycles for this exact
    // deterministic case. Keep modest headroom while requiring the elastic
    // feeder to retain its measured throughput gain.
    if (down.cycles > 2800)
        return error.FfnSectionFeederCycleRegression;
    std.debug.print(
        "decode_top v15 FFN section: scratch-only A/R={d}/0 then {d}/0, internal reads/y/records={d}/{d}/{d}, pending={d}, cycles={d}, DOWN bit-exact\n",
        .{ raw_beats.len, @as(usize, 0), down.read_fires, down.swiglu_fires, down.record_fires, down.peak_pending_records, down.cycles },
    );
}

fn runRawResidentCase(a: std.mem.Allocator, ternary: bool, blocks: usize, cols: usize) !void {
    const rows: usize = ROWS;
    const epoch: u32 = if (ternary) 0xF320_0158 else 0xF320_0001;
    const weight_fmt: u32 = if (ternary) WEIGHT_FMT_TERNARY else WEIGHT_FMT_BINARY;
    const scalar_count = cols * blocks * layout.Q1_BLOCK;
    var prng = std.Random.DefaultPrng.init(0xF320_A8E5_1D3C_7701 ^ @as(u64, @intFromBool(ternary)));
    const rnd = prng.random();

    const bits_load = try a.alloc(u128, rows * blocks);
    defer a.free(bits_load);
    const bits_reuse = try a.alloc(u128, rows * blocks);
    defer a.free(bits_reuse);
    const scales_load = try a.alloc(f16, rows * blocks);
    defer a.free(scales_load);
    const scales_reuse = try a.alloc(f16, rows * blocks);
    defer a.free(scales_reuse);
    const nonzero_load = try a.alloc(u128, rows * blocks);
    defer a.free(nonzero_load);
    const nonzero_reuse = try a.alloc(u128, rows * blocks);
    defer a.free(nonzero_reuse);
    const scales_hi_load = try a.alloc(f16, rows * blocks);
    defer a.free(scales_hi_load);
    const scales_hi_reuse = try a.alloc(f16, rows * blocks);
    defer a.free(scales_hi_reuse);
    for (bits_load, bits_reuse) |*first, *second| {
        first.* = (@as(u128, rnd.int(u64)) << 64) | rnd.int(u64);
        second.* = ~first.*;
    }
    for (nonzero_load, nonzero_reuse) |*first, *second| {
        first.* = (@as(u128, rnd.int(u64)) << 64) | rnd.int(u64);
        second.* = (@as(u128, rnd.int(u64)) << 64) | rnd.int(u64);
    }
    for (scales_load, scales_reuse) |*first, *second| {
        const exponent: u16 = rnd.intRangeAtMost(u16, 13, 17);
        first.* = @bitCast((exponent << 10) | rnd.intRangeLessThan(u16, 0, 1024));
        second.* = @bitCast((exponent << 10) | rnd.intRangeLessThan(u16, 0, 1024));
    }
    for (scales_hi_load, scales_hi_reuse) |*first, *second| {
        const exponent: u16 = rnd.intRangeAtMost(u16, 13, 17);
        first.* = @bitCast((exponent << 10) | rnd.intRangeLessThan(u16, 0, 1024));
        second.* = @bitCast((exponent << 10) | rnd.intRangeLessThan(u16, 0, 1024));
    }

    const port_len = weightPortBytes(1, blocks, weight_fmt);
    var storage_load: [PORTS][]u8 = undefined;
    var storage_reuse: [PORTS][]u8 = undefined;
    var ports_load: [PORTS][]const u8 = undefined;
    var ports_reuse: [PORTS][]const u8 = undefined;
    for (0..PORTS) |port| {
        storage_load[port] = try a.alloc(u8, port_len);
        storage_reuse[port] = try a.alloc(u8, port_len);
        ports_load[port] = storage_load[port];
        ports_reuse[port] = storage_reuse[port];
    }
    defer for (storage_load) |storage| a.free(storage);
    defer for (storage_reuse) |storage| a.free(storage);
    if (ternary) {
        const wide_load = try a.alloc(u8, pack.ternaryWeightBytesWide(1, blocks));
        defer a.free(wide_load);
        const wide_reuse = try a.alloc(u8, pack.ternaryWeightBytesWide(1, blocks));
        defer a.free(wide_reuse);
        pack.packTernaryWeightsWide(rows, blocks, bits_load, nonzero_load, scales_load, scales_hi_load, wide_load);
        pack.packTernaryWeightsWide(rows, blocks, bits_reuse, nonzero_reuse, scales_reuse, scales_hi_reuse, wide_reuse);
        for (0..blocks) |resident_block| {
            for (0..layout.ternary_beats_per_port_block) |beat| {
                for (0..PORTS) |port| {
                    const src = (resident_block * layout.ternary_beats_per_port_block + beat) *
                        pack.WIDE_BEAT_BYTES + port * PORT_BEAT_BYTES;
                    const dst = (resident_block * layout.ternary_beats_per_port_block + beat) *
                        PORT_BEAT_BYTES;
                    @memcpy(storage_load[port][dst..][0..PORT_BEAT_BYTES], wide_load[src..][0..PORT_BEAT_BYTES]);
                    @memcpy(storage_reuse[port][dst..][0..PORT_BEAT_BYTES], wide_reuse[src..][0..PORT_BEAT_BYTES]);
                }
            }
        }
    } else {
        packWeightPorts(rows, blocks, bits_load, scales_load, storage_load);
        packWeightPorts(rows, blocks, bits_reuse, scales_reuse, storage_reuse);
    }

    const raw = try a.alloc(f32, scalar_count);
    defer a.free(raw);
    for (raw, 0..) |*value, i| {
        const centered: i32 = @as(i32, @intCast(i % 37)) - 18;
        value.* = @as(f32, @floatFromInt(centered)) * 0.0625 +
            (rnd.float(f32) - 0.5) * 0.01;
    }
    const raw_beats = try a.alloc(u64, scalar_count / 2);
    defer a.free(raw_beats);
    packRawF32(raw, raw_beats);

    const aquants = try a.alloc(i8, scalar_count);
    defer a.free(aquants);
    const ascales = try a.alloc(f16, scalar_count / shared_layout.q8_block);
    defer a.free(ascales);
    try shared_layout.quantizeQ8_0(raw, aquants, ascales);

    const expected_load = try a.alloc(f32, cols * rows);
    defer a.free(expected_load);
    const expected_reuse = try a.alloc(f32, cols * rows);
    defer a.free(expected_reuse);
    const aq_per_col = blocks * layout.Q1_BLOCK;
    const as_per_col = blocks * layout.Q8_SUBBLOCKS;
    const window = ref.fixedWindow();
    var saturations: usize = 0;
    for (0..cols) |col| {
        const aq = aquants[col * aq_per_col ..][0..aq_per_col];
        const as_ = ascales[col * as_per_col ..][0..as_per_col];
        var sat: usize = 0;
        ref.windowedFixedOutput(.{
            .rows = rows,
            .q1_blocks = blocks,
            .weight_bits = bits_load,
            .weight_scales = scales_load,
            .act_quants = aq,
            .act_scales = as_,
            .weight_nonzero = if (ternary) nonzero_load else null,
            .weight_scales_hi = if (ternary) scales_hi_load else null,
        }, window, expected_load[col * rows ..][0..rows], &sat);
        saturations += sat;
        sat = 0;
        ref.windowedFixedOutput(.{
            .rows = rows,
            .q1_blocks = blocks,
            .weight_bits = bits_reuse,
            .weight_scales = scales_reuse,
            .act_quants = aq,
            .act_scales = as_,
            .weight_nonzero = if (ternary) nonzero_reuse else null,
            .weight_scales_hi = if (ternary) scales_hi_reuse else null,
        }, window, expected_reuse[col * rows ..][0..rows], &sat);
        saturations += sat;
    }
    if (saturations != 0) return error.WindowTooNarrow;

    var dut = Dut.init();
    defer dut.deinit();
    reset(&dut);

    const load = try runProjection(a, &dut, rows, rows, blocks, cols, weight_fmt, ACT_RAW_LOAD, epoch, ports_load, raw_beats, raw_beats.len, .none, false, false);
    defer a.free(load.result);
    if (!std.mem.eql(u8, load.result, std.mem.sliceAsBytes(expected_load)) or
        load.quant_status != 0 or load.act_beats != raw_beats.len or
        load.weight_beats != port_len / PORT_BEAT_BYTES or load.result_beats != rows * cols / 2 or
        try axiRead(&dut, REG_LOADED_EPOCH) != epoch or
        try axiRead(&dut, REG_LOADED_Q1_BLOCKS) != blocks or
        try axiRead(&dut, REG_LOADED_COLS) != cols)
        return error.RawProjectionMismatch;

    const reuse = try runProjection(a, &dut, rows, rows, blocks, cols, weight_fmt, ACT_REUSE, epoch, ports_reuse, &.{}, 0, .none, false, false);
    defer a.free(reuse.result);
    if (!std.mem.eql(u8, reuse.result, std.mem.sliceAsBytes(expected_reuse)) or
        reuse.act_beats != 0 or reuse.weight_beats != port_len / PORT_BEAT_BYTES or
        reuse.result_beats != rows * cols / 2 or reuse.act_state != 1 or
        try axiRead(&dut, REG_LOADED_EPOCH) != epoch or
        try axiRead(&dut, REG_LOADED_Q1_BLOCKS) != blocks or
        try axiRead(&dut, REG_LOADED_COLS) != cols)
        return error.RawReuseMismatch;

    const malformed = try runProjection(a, &dut, rows, rows, blocks, cols, weight_fmt, ACT_RAW_LOAD, epoch + 1, ports_load, raw_beats, 1, .early_last, false, true);
    defer a.free(malformed.result);
    if (malformed.act_beats != 1 or malformed.quant_status != QUANT_FRAME or
        malformed.act_state != 2)
        return error.MalformedRawFrameDidNotFailClosed;

    // Exercise the nested end predicate itself: a complete finite stream without
    // TLAST must be rejected only on the final col/Q1/sub-block input beat.
    const missing_last = try runProjection(a, &dut, rows, rows, blocks, cols, weight_fmt, ACT_RAW_LOAD, epoch + 2, ports_load, raw_beats, raw_beats.len, .no_last, false, true);
    defer a.free(missing_last.result);
    if (missing_last.act_beats != raw_beats.len or missing_last.quant_status != QUANT_FRAME or
        missing_last.act_state != 2)
        return error.MissingRawFrameEndDidNotFailClosed;

    var nonfinite_values = [_]f32{0.25} ** shared_layout.q8_block;
    nonfinite_values[7] = std.math.nan(f32);
    var nonfinite_beats: [shared_layout.q8_block / 2]u64 = undefined;
    packRawF32(&nonfinite_values, &nonfinite_beats);
    const nonfinite = try runProjection(a, &dut, rows, rows, blocks, cols, weight_fmt, ACT_RAW_LOAD, epoch + 3, ports_load, &nonfinite_beats, nonfinite_beats.len, .no_last, false, true);
    defer a.free(nonfinite.result);
    if (nonfinite.act_beats != nonfinite_beats.len or
        nonfinite.quant_status != QUANT_NONFINITE or nonfinite.act_state != 2)
        return error.NonfiniteRawInputDidNotFailClosed;

    if (blocks == 2 and cols == 3) {
        reset(&dut);
        if (try axiRead(&dut, 0x04) != 15) return error.ScratchVersionMismatch;

        configureScratch(&dut, SCRATCH_MODE_TEE, SCRATCH_ROLE_X1, rows, cols);
        const tee_up = try runProjection(a, &dut, rows, rows, blocks, cols, weight_fmt, ACT_RAW_LOAD, epoch, ports_load, raw_beats, raw_beats.len, .none, true, false);
        defer a.free(tee_up.result);
        if (!std.mem.eql(u8, tee_up.result, std.mem.sliceAsBytes(expected_load)))
            return error.ScratchTeeUpMismatch;
        var scratch_status = try axiRead(&dut, REG_SCRATCH_STATUS);
        if (scratch_status & (SCRATCH_WRITER_DONE | SCRATCH_X1_VALID) !=
            (SCRATCH_WRITER_DONE | SCRATCH_X1_VALID) or
            scratch_status & (SCRATCH_ANY_ERROR | SCRATCH_X0_VALID) != 0)
            return error.ScratchX1CommitMismatch;

        configureScratch(&dut, SCRATCH_MODE_TEE, SCRATCH_ROLE_X0, rows, cols);
        const tee_gate = try runProjection(a, &dut, rows, rows, blocks, cols, weight_fmt, ACT_REUSE, epoch, ports_reuse, &.{}, 0, .none, true, false);
        defer a.free(tee_gate.result);
        if (!std.mem.eql(u8, tee_gate.result, std.mem.sliceAsBytes(expected_reuse)))
            return error.ScratchTeeGateMismatch;
        scratch_status = try axiRead(&dut, REG_SCRATCH_STATUS);
        if (scratch_status & (SCRATCH_WRITER_DONE | SCRATCH_X0_VALID | SCRATCH_X1_VALID) !=
            (SCRATCH_WRITER_DONE | SCRATCH_X0_VALID | SCRATCH_X1_VALID) or
            scratch_status & SCRATCH_ANY_ERROR != 0)
            return error.ScratchX0CommitMismatch;

        try drainScratch(a, &dut, SCRATCH_ROLE_X1, rows, cols, tee_up.result);
        try drainScratch(a, &dut, SCRATCH_ROLE_X0, rows, cols, tee_gate.result);

        // A reuse-state error terminates the GEMM before output. The coupled
        // writer must be aborted too, otherwise it would remain busy forever.
        configureScratch(&dut, SCRATCH_MODE_TEE, SCRATCH_ROLE_X1, rows, cols);
        const bad_reuse = try runProjection(a, &dut, rows, rows, blocks, cols, weight_fmt, ACT_REUSE, epoch + 99, ports_reuse, &.{}, 0, .none, false, true);
        defer a.free(bad_reuse.result);
        scratch_status = try axiRead(&dut, REG_SCRATCH_STATUS);
        if (scratch_status & SCRATCH_WRITER_BUSY != 0 or
            scratch_status & (SCRATCH_WRITER_DONE | SCRATCH_ANY_ERROR) !=
                (SCRATCH_WRITER_DONE | SCRATCH_ANY_ERROR) or
            try axiRead(&dut, REG_SCRATCH_ERROR) & SCRATCH_ERROR_WRITER == 0 or
            scratch_status & SCRATCH_X1_VALID != 0 or
            scratch_status & SCRATCH_X0_VALID == 0)
            return error.BadReuseDidNotAbortScratchWriter;

        try abortScratchDrain(&dut, SCRATCH_ROLE_X0, rows, cols);
        scratch_status = try axiRead(&dut, REG_SCRATCH_STATUS);
        if (scratch_status & SCRATCH_X0_VALID == 0)
            return error.ScratchDrainAbortInvalidatedRole;

        // A mismatched tee must retire immediately without opening any stream.
        configureScratch(&dut, SCRATCH_MODE_TEE, SCRATCH_ROLE_X1, rows, cols - 1);
        configureProjection(&dut, rows, rows, blocks, cols, weight_fmt, ACT_RAW_LOAD, epoch + 1);
        var zero = [_]u32{0} ** ROWS_PER_PORT;
        for (0..16) |_| {
            for (0..PORTS) |port| c.dut_set_w(dut.h, @intCast(port), &zero, 1);
            c.dut_set_a(dut.h, raw_beats[0], 1, 0);
            c.dut_set_m_ready(dut.h, 1);
            c.dut_eval(dut.h);
            for (0..PORTS) |port| {
                if (c.dut_w_ready(dut.h, @intCast(port)) != 0)
                    return error.RejectedScratchTeeConsumedWeight;
            }
            if (c.dut_a_ready(dut.h) != 0 or c.dut_m_valid(dut.h) != 0)
                return error.RejectedScratchTeeOpenedStream;
            dut.step();
        }
        if (try axiRead(&dut, REG_STATUS) & 2 == 0 or
            try axiRead(&dut, REG_W_BEATS) != 0 or
            try axiRead(&dut, REG_A_BEATS) != 0 or
            try axiRead(&dut, REG_R_BEATS) != 0)
            return error.RejectedScratchTeeDidNotRetire;
        scratch_status = try axiRead(&dut, REG_SCRATCH_STATUS);
        if (scratch_status & SCRATCH_ANY_ERROR == 0 or
            try axiRead(&dut, REG_SCRATCH_ERROR) & SCRATCH_ERROR_CONFIG == 0 or
            scratch_status & SCRATCH_X1_VALID != 0 or
            scratch_status & SCRATCH_X0_VALID == 0)
            return error.RejectedScratchTeeOwnershipMismatch;

        configureScratch(&dut, SCRATCH_MODE_DRAIN, SCRATCH_ROLE_X1, rows, cols);
        axiWrite(&dut, REG_SCRATCH_CTRL, SCRATCH_CTRL_DRAIN_START);
        scratch_status = try axiRead(&dut, REG_SCRATCH_STATUS);
        if (scratch_status & (SCRATCH_DRAIN_DONE | SCRATCH_ANY_ERROR) !=
            (SCRATCH_DRAIN_DONE | SCRATCH_ANY_ERROR) or
            try axiRead(&dut, REG_SCRATCH_ERROR) & SCRATCH_ERROR_STALE == 0)
            return error.StaleScratchDrainDidNotFailClosed;

        axiWrite(&dut, REG_SCRATCH_MODE, SCRATCH_MODE_DDR);
        std.debug.print(
            "decode_top scratch tee/drain {s}: X1/X0 exact, stalled, retained; rejected shape consumed W/A/R=0/0/0\n",
            .{if (ternary) "ternary" else "binary"},
        );
    }

    std.debug.print(
        "decode_top raw F32 {s} blocks={d} cols={d} load/reuse passed: epoch=0x{X:0>8}, cycles={d}/{d}, hashes=0x{X:0>16}/0x{X:0>16}, A beats={d}/0; frame early/missing/nonfinite consumed A={d}/{d}/{d}, W/R=0/0\n",
        .{ if (ternary) "ternary" else "binary", blocks, cols, epoch, load.cycles, reuse.cycles, std.hash.Wyhash.hash(0, load.result), std.hash.Wyhash.hash(0, reuse.result), load.act_beats, malformed.act_beats, missing_last.act_beats, nonfinite.act_beats },
    );
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
    // Preserve the original one-Q1/two-column characterization, then cross both
    // nested ingress boundaries so TLAST/order cannot accidentally flatten Q1/cols.
    try runRawResidentCase(a, false, 1, 2);
    try runRawResidentCase(a, true, 1, 2);
    try runRawResidentCase(a, false, 2, 3);
    try runRawResidentCase(a, true, 2, 3);
    try runFfnSectionCase(a);
    std.debug.print("all decode_top cosim cases passed (decode_top === windowedFixedOutput, bit-exact)\n", .{});
}
