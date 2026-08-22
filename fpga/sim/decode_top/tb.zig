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
const REG_MODEL_ROWS: u8 = 0x84;
const REG_NORM_EPS: u8 = 0x88;
const REG_NORM_CTRL: u8 = 0x8C;
const REG_NORM_STATUS: u8 = 0x90;
const REG_NORM_ERROR: u8 = 0x94;
const REG_RESIDUAL_ERROR: u8 = 0x98;

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
const SCRATCH_ROLE_R: u32 = 0;
const SCRATCH_ROLE_X0: u32 = 1;
const SCRATCH_ROLE_X1: u32 = 2;
const SCRATCH_CTRL_DRAIN_START: u32 = 1;
const SCRATCH_CTRL_ABORT: u32 = 1 << 1;
const SCRATCH_CTRL_SECTION_BEGIN: u32 = 1 << 2;
const SCRATCH_CTRL_RESIDENT_R: u32 = 1 << 3;
const SCRATCH_WRITER_BUSY: u32 = 1 << 0;
const SCRATCH_WRITER_DONE: u32 = 1 << 1;
const SCRATCH_DRAIN_BUSY: u32 = 1 << 2;
const SCRATCH_DRAIN_DONE: u32 = 1 << 3;
const SCRATCH_ANY_ERROR: u32 = 1 << 4;
const SCRATCH_R_VALID: u32 = 1 << 5;
const SCRATCH_X0_VALID: u32 = 1 << 6;
const SCRATCH_X1_VALID: u32 = 1 << 7;
const SCRATCH_CONSUMER_BUSY: u32 = 1 << 9;
const SCRATCH_CONSUMER_DONE: u32 = 1 << 10;
const SCRATCH_SECTION_ACTIVE: u32 = 1 << 11;
const SCRATCH_SECTION_DONE: u32 = 1 << 12;
const SCRATCH_GATE_READY: u32 = 1 << 13;
const SCRATCH_ERROR_CONFIG: u32 = 1 << 0;
const SCRATCH_ERROR_WRITER: u32 = 1 << 1;
const SCRATCH_ERROR_ABORT: u32 = 1 << 2;
const SCRATCH_ERROR_STALE: u32 = 1 << 4;
const SCRATCH_ERROR_SECTION: u32 = 1 << 5;
const SCRATCH_ERROR_SWIGLU_Q8: u32 = 1 << 6;
const SCRATCH_ERROR_RMS: u32 = 1 << 7;
const SCRATCH_ERROR_RESIDUAL: u32 = 1 << 8;

const NORM_GAMMA_BUSY: u32 = 1 << 0;
const NORM_GAMMA_DONE: u32 = 1 << 1;
const NORM_GAMMA_ERROR: u32 = 1 << 2;
const NORM_GAMMA_VALID: u32 = 1 << 3;
const NORM_BUSY: u32 = 1 << 4;
const NORM_DONE: u32 = 1 << 5;
const NORM_ERROR: u32 = 1 << 6;
const RESIDUAL_BUSY: u32 = 1 << 7;
const RESIDUAL_DONE: u32 = 1 << 8;
const RESIDUAL_ERROR: u32 = 1 << 9;
const NORM_GLOBAL_IDLE: u32 = 1 << 10;

const DBG_P3D_ACTIVE: u32 = 1 << 0;
const DBG_P3D_CLEANUP: u32 = 1 << 1;
const DBG_P3D_KILL: u32 = 1 << 2;
const DBG_P3D_R_COMPLETE: u32 = 1 << 3;
const DBG_P3D_NORM_SEALED: u32 = 1 << 4;
const DBG_P3D_RESIDUAL_STARTED: u32 = 1 << 5;
const DBG_P3D_Q8_OWNER_MASK: u32 = 3 << 6;
const DBG_P3D_Q8_OWNER_RMS: u32 = 1 << 6;
const DBG_P3D_Q8_OWNER_GATE: u32 = 2 << 6;
const DBG_P3D_OUTER_OWNER_MASK: u32 = 3 << 8;
const DBG_P3D_OUTER_OWNER: u32 = 3 << 8;
const DBG_P3D_SUBOWNER_MASK: u32 = 3 << 10;
const DBG_P3D_SUBOWNER_RMS: u32 = 1 << 10;
const DBG_P3D_SUBOWNER_RESIDUAL: u32 = 2 << 10;
const DBG_P3D_RSP_VALID: u32 = 1 << 12;
const DBG_P3D_Q8_FAULT: u32 = 1 << 13;
const DBG_P3D_Q8_FAULT_OWNER_MASK: u32 = 3 << 14;
const DBG_P3D_Q8_FAULT_OWNER_RMS: u32 = 1 << 14;
const DBG_P3D_Q8_FAULT_OWNER_GATE: u32 = 2 << 14;
const DBG_P3D_Q8_FAULT_HELD: u32 = 1 << 16;
const DBG_P3D_DOWN_COMMITTED: u32 = 1 << 17;
const DBG_P3D_DOWN_LAUNCH: u32 = 1 << 18;

const DBG_P3D_BEGIN_OK: u32 = 1 << 0;
const DBG_P3D_LEAF_START_Q: u32 = 1 << 1;
const DBG_P3D_LEAF_START: u32 = 1 << 2;
const DBG_P3D_LAUNCH_Q8: u32 = 1 << 3;
const DBG_P3D_ABORT_STROBE: u32 = 1 << 4;
const DBG_P3D_RMS_BUSY: u32 = 1 << 5;

const DBG_FFN_ACTIVE: u32 = 1 << 0;
const DBG_FFN_PRODUCER_BUSY: u32 = 1 << 1;
const DBG_FFN_ABORT_CLEANUP: u32 = 1 << 2;
const DBG_FFN_OWNER_MASK: u32 = 3 << 3;
const DBG_FFN_OWNER_PAIRER: u32 = 2 << 3;
const DBG_FFN_RSP_VALID: u32 = 1 << 5;
const DBG_FFN_PAIRER_ORPHAN: u32 = 1 << 6;
const DBG_FFN_PAIRER_BUSY: u32 = 1 << 7;
const DBG_FFN_KERNEL_BUSY: u32 = 1 << 8;
const DBG_FFN_PACKER_BUSY: u32 = 1 << 9;
const DBG_FFN_SECTION_DONE: u32 = 1 << 10;
const DBG_FFN_ANY_ERROR: u32 = 1 << 11;
const DBG_FFN_PAIRER_STAGING_REQ: u32 = 1 << 12;

const DBG_SHARED_COMPUTE_VALID: u32 = 1 << 0;
const DBG_SHARED_Q8_START: u32 = 1 << 1;
const DBG_SHARED_KERNEL_START: u32 = 1 << 2;
const DBG_SHARED_GAMMA_BUSY: u32 = 1 << 3;
const DBG_SHARED_GAMMA_READY: u32 = 1 << 6;

const DBG_CTRL_GAMMA_ADMIT: u32 = 1 << 0;
const DBG_CTRL_GAMMA_PENDING: u32 = 1 << 1;
const DBG_CTRL_GAMMA_FIRE: u32 = 1 << 2;
const DBG_CTRL_GAMMA_READY: u32 = 1 << 3;
const DBG_CTRL_GAMMA_RAW_BUSY: u32 = 1 << 4;
const DBG_CTRL_GAMMA_BUSY: u32 = 1 << 5;
const DBG_CTRL_GAMMA_RAW_VALID: u32 = 1 << 6;
const DBG_CTRL_GAMMA_VALID: u32 = 1 << 7;
const DBG_CTRL_KERNEL_ABORT_NOW: u32 = 1 << 8;
const DBG_CTRL_KERNEL_ABORT_Q: u32 = 1 << 9;
const DBG_CTRL_KERNEL_BUSY: u32 = 1 << 10;
const DBG_CTRL_WEIGHT_READY: u32 = 1 << 11;
const DBG_CTRL_ACTS_READY: u32 = 1 << 12;
const DBG_CTRL_OUTPUT_VALID: u32 = 1 << 13;
const DBG_CTRL_SCRATCH_R_WRITE: u32 = 1 << 14;
const DBG_CTRL_SCRATCH_READ_ACCEPT: u32 = 1 << 15;
const DBG_CTRL_KERNEL_STATE_SHIFT: u5 = 16;
const DBG_CTRL_KERNEL_STATE_MASK: u32 = 0xF << DBG_CTRL_KERNEL_STATE_SHIFT;
const DBG_CTRL_ACTIVATION_VALID: u32 = 1 << 20;
const DBG_CTRL_ACTIVATION_ERROR: u32 = 1 << 21;
const DBG_CTRL_KERNEL_READY_CORE: u32 = 1 << 22;
const DBG_CTRL_KERNEL_SINK_ACCEPT: u32 = 1 << 23;
const DBG_CTRL_KERNEL_RAW_OUTPUT_VALID: u32 = 1 << 24;
const DBG_CTRL_KERNEL_RAW_OUTPUT_LAST: u32 = 1 << 25;
const DBG_CTRL_RMS_KILL_Q: u32 = 1 << 26;
const DBG_CTRL_RMS_ABORT: u32 = 1 << 27;
const DBG_CTRL_SCRATCH_WRITER_ABORT: u32 = 1 << 28;
const DBG_CTRL_SCRATCH_R_ABORT: u32 = 1 << 29;
const DBG_CTRL_SCRATCH_R_READY: u32 = 1 << 30;
const DBG_CTRL_RMS_ABORT_DRAIN_READY: u32 = 1 << 31;

const KERNEL_ST_IDLE: u32 = 0;
const KERNEL_ST_LOAD_ACTS: u32 = 1;
const KERNEL_ST_WISSUE: u32 = 4;
const KERNEL_ST_PRECOMPUTE: u32 = 6;
const KERNEL_ST_EMIT: u32 = 7;
const KERNEL_ST_FINISH: u32 = 8;

const DBG_LEGACY_BEGIN_OK: u32 = 1 << 0;
const DBG_LEGACY_CFG_PENDING: u32 = 1 << 1;
const DBG_LEGACY_CFG_FIRE: u32 = 1 << 2;
const DBG_LEGACY_CFG_VALID: u32 = 1 << 3;
const DBG_LEGACY_CFG_P3D: u32 = 1 << 4;
const DBG_LEGACY_CFG_ACCEPTED: u32 = 1 << 5;

const DBG_RMS_Q8_ACCEPT: u32 = 1 << 0;
const DBG_RMS_Q8_FIRE: u32 = 1 << 1;
const DBG_RMS_Q8_FINAL: u32 = 1 << 2;
const DBG_RMS_Q8_DONE: u32 = 1 << 3;
const DBG_RMS_Q8_SEAL_EVENT: u32 = 1 << 4;
const DBG_RMS_Q8_SEALED: u32 = 1 << 5;
const DBG_RMS_Q8_OWNER: u32 = 1 << 6;
const DBG_RMS_Q8_ACTIVE: u32 = 1 << 7;

const OWNERLESS_ROUTE_P3D_RMS: u32 = 1;
const OWNERLESS_ROUTE_PAIRER: u32 = 2;
const OWNERLESS_ROUTE_DRAIN: u32 = 3;
const OWNERLESS_COLLISION: u32 = 1 << 0;
const OWNERLESS_OLD_READY: u32 = 1 << 1;
const OWNERLESS_OLD_ISOLATED: u32 = 1 << 2;
const OWNERLESS_OWNER_CAPTURED: u32 = 1 << 3;
const OWNERLESS_NEW_ROUTED: u32 = 1 << 4;
const OWNERLESS_VIOLATION: u32 = 1 << 5;
const OWNERLESS_ERROR: u32 = 1 << 6;
const OWNERLESS_COMPLETE: u32 = OWNERLESS_COLLISION | OWNERLESS_OLD_READY |
    OWNERLESS_OLD_ISOLATED | OWNERLESS_OWNER_CAPTURED |
    OWNERLESS_NEW_ROUTED;

fn dbgRmsQ8Count(value: u32) u32 {
    return (value >> 8) & 0x3ff;
}

fn dbgRmsQ8Expected(value: u32) u32 {
    return (value >> 18) & 0x3ff;
}

fn expectOwnerlessResponse(
    dut: *Dut,
    route: u32,
    injected_error: bool,
) !void {
    c.dut_eval(dut.h);
    const result = c.dut_ownerless_response_result(dut.h);
    if (result & OWNERLESS_COMPLETE != OWNERLESS_COMPLETE or
        result & OWNERLESS_VIOLATION != 0 or
        ((result & OWNERLESS_ERROR != 0) != injected_error) or
        ((result >> 8) & 3) != route)
    {
        std.debug.print(
            "ownerless response mismatch: route={d} error={} result=0x{x}\n",
            .{ route, injected_error, result },
        );
        return error.OwnerlessResponseCollisionMismatch;
    }
}

const WEIGHT_FMT_BINARY: u32 = 1;
const WEIGHT_FMT_TERNARY: u32 = 2;

comptime {
    if (ROWS != 16) @compileError("top cosim expects ROWS=16");
    if (ROWS_PER_PORT != 4) @compileError("top cosim expects four 4-row weight ports");
}

const RmsQ8Monitor = struct {
    armed: bool = false,
    failed: bool = false,
    expected: u32 = 0,
    accepts: u32 = 0,
    fires: u32 = 0,
    finals: u32 = 0,
    seal_events: u32 = 0,

    fn arm(self: *RmsQ8Monitor, state: u32, expected: u32) void {
        self.* = .{ .armed = true, .expected = expected };
        self.failed = dbgRmsQ8Count(state) != 0 or
            dbgRmsQ8Expected(state) != expected or
            state & (DBG_RMS_Q8_ACCEPT | DBG_RMS_Q8_FIRE |
                DBG_RMS_Q8_DONE | DBG_RMS_Q8_SEALED) != 0;
    }

    fn observe(self: *RmsQ8Monitor, before: u32, after: u32) void {
        if (!self.armed) return;
        if (before & DBG_RMS_Q8_ACTIVE != 0 and
            dbgRmsQ8Expected(before) != self.expected)
            self.failed = true;

        if (before & DBG_RMS_Q8_ACCEPT != 0) {
            self.accepts += 1;
            if (before & (DBG_RMS_Q8_FIRE | DBG_RMS_Q8_DONE |
                DBG_RMS_Q8_SEALED) != 0 or
                after & DBG_RMS_Q8_FIRE == 0 or
                dbgRmsQ8Count(after) != dbgRmsQ8Count(before))
                self.failed = true;
        }
        if (before & DBG_RMS_Q8_FIRE != 0) {
            self.fires += 1;
            if (dbgRmsQ8Count(after) != dbgRmsQ8Count(before) + 1)
                self.failed = true;
            if (before & DBG_RMS_Q8_FINAL != 0) {
                self.finals += 1;
                if (after & DBG_RMS_Q8_DONE == 0 or
                    after & DBG_RMS_Q8_OWNER != 0)
                    self.failed = true;
            }
        }
        if (before & DBG_RMS_Q8_SEAL_EVENT != 0) {
            self.seal_events += 1;
            if (after & DBG_RMS_Q8_SEALED == 0)
                self.failed = true;
        }
    }
};

const Dut = struct {
    h: *c.Dut,
    rms_q8_monitor: RmsQ8Monitor = .{},
    fn init() Dut {
        return .{ .h = c.dut_new().? };
    }
    fn deinit(self: *Dut) void {
        c.dut_free(self.h);
    }
    fn step(self: *Dut) void {
        const rms_q8_before = if (self.rms_q8_monitor.armed)
            c.dut_dbg_p3d_q8_accounting(self.h)
        else
            0;
        c.dut_set_clk(self.h, 1);
        c.dut_eval(self.h);
        c.dut_set_clk(self.h, 0);
        c.dut_eval(self.h);
        if (self.rms_q8_monitor.armed) {
            const rms_q8_after = c.dut_dbg_p3d_q8_accounting(self.h);
            self.rms_q8_monitor.observe(rms_q8_before, rms_q8_after);
        }
    }
    fn armRmsQ8Monitor(self: *Dut, expected: u32) void {
        c.dut_eval(self.h);
        self.rms_q8_monitor.arm(
            c.dut_dbg_p3d_q8_accounting(self.h),
            expected,
        );
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

fn beginLegacySectionChecked(dut: *Dut) !void {
    c.dut_set_axi_write(
        dut.h,
        REG_SCRATCH_CTRL,
        SCRATCH_CTRL_SECTION_BEGIN,
        1,
    );
    var saw_begin = false;
    for (0..8) |_| {
        dut.step();
        const cfg = c.dut_dbg_legacy_q8_cfg(dut.h);
        if (cfg & DBG_LEGACY_BEGIN_OK != 0) {
            if (cfg & (DBG_LEGACY_CFG_PENDING | DBG_LEGACY_CFG_FIRE |
                DBG_LEGACY_CFG_VALID | DBG_LEGACY_CFG_P3D) != 0)
                return error.LegacyBeginConfiguredQ8SameCycle;
            saw_begin = true;
            break;
        }
    }
    if (!saw_begin) return error.LegacyBeginTimingTimeout;

    c.dut_set_axi_idle(dut.h);
    dut.step();
    const delayed = c.dut_dbg_legacy_q8_cfg(dut.h);
    if (delayed & (DBG_LEGACY_CFG_PENDING | DBG_LEGACY_CFG_FIRE |
        DBG_LEGACY_CFG_VALID | DBG_LEGACY_CFG_ACCEPTED) !=
        (DBG_LEGACY_CFG_PENDING | DBG_LEGACY_CFG_FIRE |
            DBG_LEGACY_CFG_VALID | DBG_LEGACY_CFG_ACCEPTED) or
        delayed & (DBG_LEGACY_BEGIN_OK | DBG_LEGACY_CFG_P3D) != 0)
        return error.LegacyDelayedQ8ConfigMismatch;

    dut.step();
    if (c.dut_dbg_legacy_q8_cfg(dut.h) != 0)
        return error.LegacyQ8ConfigWasNotSingleCycle;
}

fn abortLegacyAtPendingQ8Config(dut: *Dut) !void {
    c.dut_set_axi_write(
        dut.h,
        REG_SCRATCH_CTRL,
        SCRATCH_CTRL_SECTION_BEGIN,
        1,
    );
    var saw_begin = false;
    for (0..8) |_| {
        dut.step();
        const cfg = c.dut_dbg_legacy_q8_cfg(dut.h);
        if (cfg & DBG_LEGACY_BEGIN_OK != 0) {
            if (cfg & (DBG_LEGACY_CFG_PENDING | DBG_LEGACY_CFG_FIRE |
                DBG_LEGACY_CFG_VALID | DBG_LEGACY_CFG_P3D) != 0)
                return error.LegacyAbortBeginConfiguredQ8SameCycle;
            saw_begin = true;
            break;
        }
    }
    if (!saw_begin) return error.LegacyAbortBeginTimingTimeout;

    c.dut_set_axi_idle(dut.h);
    dut.step();
    const pending = c.dut_dbg_legacy_q8_cfg(dut.h);
    if (pending & (DBG_LEGACY_CFG_PENDING | DBG_LEGACY_CFG_FIRE |
        DBG_LEGACY_CFG_VALID | DBG_LEGACY_CFG_ACCEPTED) !=
        (DBG_LEGACY_CFG_PENDING | DBG_LEGACY_CFG_FIRE |
            DBG_LEGACY_CFG_VALID | DBG_LEGACY_CFG_ACCEPTED))
        return error.LegacyAbortPendingConfigMissing;

    c.dut_force_scratch_abort_strobe(dut.h, 1);
    c.dut_eval(dut.h);
    const suppressed = c.dut_dbg_legacy_q8_cfg(dut.h);
    if (suppressed & DBG_LEGACY_CFG_PENDING == 0 or
        suppressed & (DBG_LEGACY_CFG_FIRE | DBG_LEGACY_CFG_VALID |
            DBG_LEGACY_CFG_P3D) != 0)
        return error.LegacyInterveningAbortDidNotSuppressConfig;

    dut.step();
    c.dut_force_scratch_abort_strobe(dut.h, 0);
    c.dut_set_axi_idle(dut.h);
    dut.step();
    for (0..64) |_| {
        if (c.dut_dbg_ffn_lifecycle(dut.h) &
            (DBG_FFN_ACTIVE | DBG_FFN_ABORT_CLEANUP) == 0) break;
        dut.step();
    } else return error.LegacyPendingConfigAbortCleanupTimeout;

    const status = try axiRead(dut, REG_SCRATCH_STATUS);
    if (c.dut_dbg_legacy_q8_cfg(dut.h) != 0 or
        status & (SCRATCH_SECTION_ACTIVE | SCRATCH_CONSUMER_BUSY) != 0 or
        status & (SCRATCH_SECTION_DONE | SCRATCH_ANY_ERROR) !=
            (SCRATCH_SECTION_DONE | SCRATCH_ANY_ERROR) or
        try axiRead(dut, REG_SCRATCH_ERROR) & SCRATCH_ERROR_ABORT == 0)
        return error.LegacyPendingConfigAbortStatusMismatch;
}

fn verifyAbortDominatesCoencodedStarts() !void {
    var dut = Dut.init();
    defer dut.deinit();

    reset(&dut);
    configureScratch(&dut, SCRATCH_MODE_DDR, SCRATCH_ROLE_X0, 128, 1);
    axiWrite(&dut, REG_SCRATCH_CTRL, SCRATCH_CTRL_SECTION_BEGIN | SCRATCH_CTRL_ABORT);
    var status = try axiRead(&dut, REG_SCRATCH_STATUS);
    if (status & (SCRATCH_SECTION_ACTIVE | SCRATCH_SECTION_DONE |
        SCRATCH_DRAIN_BUSY | SCRATCH_DRAIN_DONE | SCRATCH_ANY_ERROR) != 0 or
        try axiRead(&dut, REG_SCRATCH_ERROR) != 0 or
        c.dut_dbg_ffn_phase(dut.h) != 0)
        return error.CoencodedBeginAbortDidNotStayIdle;

    // The aborted bank must remain reconfigurable without resetting the DUT.
    axiWrite(&dut, REG_SCRATCH_CTRL, SCRATCH_CTRL_SECTION_BEGIN);
    status = try axiRead(&dut, REG_SCRATCH_STATUS);
    if (status & SCRATCH_SECTION_ACTIVE == 0 or
        status & SCRATCH_ANY_ERROR != 0 or
        c.dut_dbg_ffn_phase(dut.h) != 1)
        return error.BeginAfterCoencodedAbortFailed;
    axiWrite(&dut, REG_SCRATCH_CTRL, SCRATCH_CTRL_ABORT);

    reset(&dut);
    configureScratch(&dut, SCRATCH_MODE_DRAIN, SCRATCH_ROLE_X0, 16, 1);
    axiWrite(&dut, REG_SCRATCH_CTRL, SCRATCH_CTRL_DRAIN_START | SCRATCH_CTRL_ABORT);
    status = try axiRead(&dut, REG_SCRATCH_STATUS);
    if (status & (SCRATCH_DRAIN_BUSY | SCRATCH_DRAIN_DONE |
        SCRATCH_ANY_ERROR) != 0 or try axiRead(&dut, REG_SCRATCH_ERROR) != 0)
        return error.CoencodedDrainAbortDidNotStayIdle;

    std.debug.print(
        "decode_top scratch ABORT dominates co-encoded BEGIN/DRAIN_START; later BEGIN accepted\n",
        .{},
    );
}

fn verifyInternalModeRequiresSection() !void {
    var dut = Dut.init();
    defer dut.deinit();

    reset(&dut);
    configureScratch(&dut, SCRATCH_MODE_DDR, SCRATCH_ROLE_X0, 128, 1);
    configureProjection(
        &dut,
        ROWS,
        ROWS,
        1,
        1,
        WEIGHT_FMT_BINARY,
        ACT_SCRATCH_SWIGLU,
        0xF15F_BAD3,
    );

    var zero = [_]u32{0} ** ROWS_PER_PORT;
    for (0..16) |_| {
        for (0..PORTS) |port| c.dut_set_w(dut.h, @intCast(port), &zero, 1);
        c.dut_set_a(dut.h, 0x7fc0_0000_7fc0_0000, 1, 1);
        c.dut_set_m_ready(dut.h, 1);
        c.dut_eval(dut.h);
        for (0..PORTS) |port| {
            if (c.dut_w_ready(dut.h, @intCast(port)) != 0)
                return error.UnownedInternalModeConsumedWeight;
        }
        if (c.dut_a_ready(dut.h) != 0 or c.dut_m_valid(dut.h) != 0)
            return error.UnownedInternalModeOpenedStream;
        dut.step();
    }
    for (0..PORTS) |port| c.dut_set_w(dut.h, @intCast(port), &zero, 0);
    c.dut_set_a(dut.h, 0, 0, 0);

    const status = try axiRead(&dut, REG_SCRATCH_STATUS);
    if (try axiRead(&dut, REG_STATUS) & 2 == 0 or
        try axiRead(&dut, REG_W_BEATS) != 0 or
        try axiRead(&dut, REG_A_BEATS) != 0 or
        try axiRead(&dut, REG_R_BEATS) != 0 or
        status & SCRATCH_ANY_ERROR == 0 or
        status & SCRATCH_SECTION_ACTIVE != 0 or
        try axiRead(&dut, REG_SCRATCH_ERROR) & SCRATCH_ERROR_CONFIG == 0 or
        c.dut_dbg_ffn_phase(dut.h) != 0)
        return error.UnownedInternalModeDidNotRejectAtPreflight;

    std.debug.print(
        "decode_top ACT_MODE=3 outside a section rejected before W/A/R streams open\n",
        .{},
    );
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

fn dbgKernelState(control: u32) u32 {
    return (control & DBG_CTRL_KERNEL_STATE_MASK) >>
        DBG_CTRL_KERNEL_STATE_SHIFT;
}

fn driveKernelBoundaryInputs(dut: *Dut, output_ready: bool) void {
    var words = [_]u32{0x8000_0000} ** ROWS_PER_PORT;
    for (0..PORTS) |port|
        c.dut_set_w(dut.h, @intCast(port), &words, 1);
    c.dut_set_a(dut.h, 0, 1, 0);
    c.dut_set_m_ready(dut.h, @intFromBool(output_ready));
}

fn clearKernelBoundaryInputs(dut: *Dut) void {
    var words = [_]u32{0} ** ROWS_PER_PORT;
    for (0..PORTS) |port|
        c.dut_set_w(dut.h, @intCast(port), &words, 0);
    c.dut_set_a(dut.h, 0, 0, 0);
    c.dut_set_m_ready(dut.h, 1);
}

fn abortPlainKernelAtState(
    dut: *Dut,
    target_state: u32,
    epoch: u32,
    live_final_emit: bool,
) !void {
    configureProjection(
        dut,
        ROWS,
        ROWS,
        1,
        1,
        WEIGHT_FMT_BINARY,
        ACT_PACKED_LOAD,
        epoch,
    );
    const output_ready = target_state != KERNEL_ST_EMIT or live_final_emit;
    var reached = false;
    for (0..4096) |_| {
        driveKernelBoundaryInputs(dut, output_ready);
        c.dut_eval(dut.h);
        const control = c.dut_dbg_control_boundaries(dut.h);
        if (dbgKernelState(control) == target_state) {
            if (target_state == KERNEL_ST_EMIT and live_final_emit and
                control & DBG_CTRL_KERNEL_RAW_OUTPUT_LAST == 0)
            {
                dut.step();
                continue;
            }
            if (control & DBG_CTRL_KERNEL_BUSY == 0)
                return error.KernelAbortTargetWasNotBusy;
            if ((control & DBG_CTRL_KERNEL_READY_CORE != 0) != output_ready)
                return error.KernelAbortReadyCoreMismatch;
            if (target_state == KERNEL_ST_LOAD_ACTS and
                control & DBG_CTRL_ACTS_READY == 0)
                return error.KernelLoadActsBoundaryWasNotOpen;
            if (target_state == KERNEL_ST_WISSUE and
                control & DBG_CTRL_WEIGHT_READY == 0)
                return error.KernelWeightIssueBoundaryWasNotOpen;
            if (target_state == KERNEL_ST_EMIT and
                control & (DBG_CTRL_OUTPUT_VALID |
                    DBG_CTRL_KERNEL_RAW_OUTPUT_VALID) !=
                    (DBG_CTRL_OUTPUT_VALID |
                        DBG_CTRL_KERNEL_RAW_OUTPUT_VALID))
                return error.KernelEmitBoundaryWasNotValid;
            if (target_state == KERNEL_ST_EMIT and live_final_emit and
                control & (DBG_CTRL_KERNEL_READY_CORE |
                    DBG_CTRL_KERNEL_SINK_ACCEPT |
                    DBG_CTRL_KERNEL_RAW_OUTPUT_LAST) !=
                    (DBG_CTRL_KERNEL_READY_CORE |
                        DBG_CTRL_KERNEL_SINK_ACCEPT |
                        DBG_CTRL_KERNEL_RAW_OUTPUT_LAST))
                return error.KernelFinalEmitBoundaryWasNotLive;
            if (target_state == KERNEL_ST_FINISH and
                control & DBG_CTRL_ACTIVATION_VALID == 0)
                return error.KernelFinishBoundaryWasNotResident;
            reached = true;
            break;
        }
        dut.step();
    }
    if (!reached) return error.KernelAbortTargetTimeout;

    c.dut_force_scratch_abort_strobe(dut.h, 1);
    c.dut_eval(dut.h);
    var control = c.dut_dbg_control_boundaries(dut.h);
    if (control & (DBG_CTRL_KERNEL_ABORT_NOW | DBG_CTRL_RMS_ABORT |
        DBG_CTRL_SCRATCH_WRITER_ABORT | DBG_CTRL_SCRATCH_R_ABORT) !=
        (DBG_CTRL_KERNEL_ABORT_NOW | DBG_CTRL_RMS_ABORT |
            DBG_CTRL_SCRATCH_WRITER_ABORT | DBG_CTRL_SCRATCH_R_ABORT) or
        control & DBG_CTRL_KERNEL_ABORT_Q != 0 or
        control & DBG_CTRL_RMS_ABORT_DRAIN_READY == 0 or
        control & (DBG_CTRL_RMS_KILL_Q | DBG_CTRL_SCRATCH_R_READY) != 0 or
        control & (DBG_CTRL_WEIGHT_READY | DBG_CTRL_ACTS_READY |
            DBG_CTRL_OUTPUT_VALID | DBG_CTRL_SCRATCH_R_WRITE |
            DBG_CTRL_SCRATCH_READ_ACCEPT |
            DBG_CTRL_KERNEL_SINK_ACCEPT) != 0)
        return error.KernelAbortSameCycleQuarantineMismatch;
    if (target_state == KERNEL_ST_EMIT and live_final_emit and
        control & (DBG_CTRL_KERNEL_READY_CORE |
            DBG_CTRL_KERNEL_RAW_OUTPUT_VALID |
            DBG_CTRL_KERNEL_RAW_OUTPUT_LAST) !=
            (DBG_CTRL_KERNEL_READY_CORE |
                DBG_CTRL_KERNEL_RAW_OUTPUT_VALID |
                DBG_CTRL_KERNEL_RAW_OUTPUT_LAST))
        return error.KernelFinalEmitAbortDidNotPreserveCoreBoundary;

    // Hold raw abort long enough to cross registered ingress, enter ST_ERROR,
    // and retire it. A sticky section fault must not deadlock cleanup there.
    var retired = false;
    var checked_finish_abort_status = false;
    for (0..8) |abort_cycle| {
        dut.step();
        control = c.dut_dbg_control_boundaries(dut.h);
        if (abort_cycle == 0 and
            control & DBG_CTRL_KERNEL_ABORT_Q == 0)
            return error.KernelAbortWasNotRegistered;
        if (abort_cycle == 0 and target_state == KERNEL_ST_EMIT and
            live_final_emit and dbgKernelState(control) != KERNEL_ST_FINISH)
            return error.KernelFinalEmitDidNotReachFinishUnderAbort;
        if (target_state == KERNEL_ST_FINISH and
            control & DBG_CTRL_KERNEL_ABORT_Q != 0 and
            !checked_finish_abort_status)
        {
            if (try axiRead(dut, REG_ACT_STATE) != 2)
                return error.KernelFinishAbortTransientStatusWasStale;
            checked_finish_abort_status = true;
            c.dut_eval(dut.h);
            control = c.dut_dbg_control_boundaries(dut.h);
        }
        if (control & (DBG_CTRL_WEIGHT_READY | DBG_CTRL_ACTS_READY |
            DBG_CTRL_OUTPUT_VALID | DBG_CTRL_SCRATCH_R_WRITE |
            DBG_CTRL_SCRATCH_READ_ACCEPT |
            DBG_CTRL_KERNEL_SINK_ACCEPT) != 0)
            return error.KernelAbortHeldQuarantineMismatch;
        if (control & DBG_CTRL_KERNEL_BUSY == 0 and
            dbgKernelState(control) == KERNEL_ST_IDLE and
            control & DBG_CTRL_KERNEL_ABORT_Q == 0)
        {
            retired = true;
            break;
        }
    }
    if (!retired) return error.KernelAbortHeldCleanupTimeout;
    if (target_state == KERNEL_ST_FINISH and !checked_finish_abort_status)
        return error.KernelFinishAbortTransientStatusWasNotChecked;
    if (control & DBG_CTRL_ACTIVATION_VALID != 0 or
        control & DBG_CTRL_ACTIVATION_ERROR == 0)
        return error.KernelAbortActivationStateMismatch;
    if (target_state == KERNEL_ST_EMIT and live_final_emit and
        try axiRead(dut, REG_ACT_STATE) != 2)
        return error.KernelFinalEmitAbortStatusWasNotPersistent;

    c.dut_force_scratch_abort_strobe(dut.h, 0);
    clearKernelBoundaryInputs(dut);
    dut.step();
}

fn verifyKernelAbortBoundaries() !void {
    var dut = Dut.init();
    defer dut.deinit();
    reset(&dut);

    // A truly idle raw abort was ignored by the old direct kernel port and must
    // neither create a delayed pulse nor poison an already-resident activation.
    configureProjection(
        &dut,
        ROWS,
        ROWS,
        1,
        1,
        WEIGHT_FMT_BINARY,
        ACT_PACKED_LOAD,
        0xA807_0000,
    );
    var saw_resident = false;
    var saw_busy = false;
    for (0..4096) |_| {
        driveKernelBoundaryInputs(&dut, true);
        c.dut_eval(dut.h);
        const control = c.dut_dbg_control_boundaries(dut.h);
        saw_busy = saw_busy or control & DBG_CTRL_KERNEL_BUSY != 0;
        if (saw_busy and control & DBG_CTRL_KERNEL_BUSY == 0 and
            control & DBG_CTRL_ACTIVATION_VALID != 0 and
            control & DBG_CTRL_ACTIVATION_ERROR == 0)
        {
            saw_resident = true;
            break;
        }
        dut.step();
    }
    if (!saw_resident) return error.IdleKernelAbortResidentSetupTimeout;
    clearKernelBoundaryInputs(&dut);
    dut.step();
    const status_before_idle_abort = try axiRead(&dut, REG_STATUS);
    const act_state_before_idle_abort = try axiRead(&dut, REG_ACT_STATE);
    if (act_state_before_idle_abort != 1)
        return error.IdleKernelAbortResidentSetupMismatch;

    c.dut_force_scratch_abort_strobe(dut.h, 1);
    for (0..3) |_| {
        c.dut_eval(dut.h);
        const control = c.dut_dbg_control_boundaries(dut.h);
        if (control & DBG_CTRL_KERNEL_ABORT_NOW == 0 or
            control & DBG_CTRL_ACTIVATION_VALID == 0 or
            control & (DBG_CTRL_KERNEL_ABORT_Q | DBG_CTRL_KERNEL_BUSY |
                DBG_CTRL_ACTIVATION_ERROR |
                DBG_CTRL_WEIGHT_READY | DBG_CTRL_ACTS_READY |
                DBG_CTRL_OUTPUT_VALID | DBG_CTRL_KERNEL_SINK_ACCEPT) != 0)
            return error.IdleKernelAbortChangedState;
        dut.step();
    }
    c.dut_force_scratch_abort_strobe(dut.h, 0);
    dut.step();
    if (try axiRead(&dut, REG_STATUS) != status_before_idle_abort or
        try axiRead(&dut, REG_ACT_STATE) != act_state_before_idle_abort)
        return error.IdleKernelAbortChangedPublicStatus;

    try abortPlainKernelAtState(&dut, KERNEL_ST_LOAD_ACTS, 0xA807_0001, false);
    try abortPlainKernelAtState(&dut, KERNEL_ST_WISSUE, 0xA807_0002, false);
    try abortPlainKernelAtState(&dut, KERNEL_ST_PRECOMPUTE, 0xA807_0006, false);
    try abortPlainKernelAtState(&dut, KERNEL_ST_EMIT, 0xA807_0003, false);
    try abortPlainKernelAtState(&dut, KERNEL_ST_EMIT, 0xA807_0004, true);
    try abortPlainKernelAtState(&dut, KERNEL_ST_FINISH, 0xA807_0005, false);

    if (try axiRead(&dut, REG_ACT_STATE) != 2)
        return error.KernelFinishAbortStatusWasNotPublished;
    configureProjection(
        &dut,
        ROWS,
        ROWS,
        1,
        1,
        WEIGHT_FMT_BINARY,
        ACT_REUSE,
        0xA807_0005,
    );
    clearKernelBoundaryInputs(&dut);
    dut.step();
    const reuse = c.dut_dbg_control_boundaries(dut.h);
    if (reuse & (DBG_CTRL_ACTIVATION_VALID | DBG_CTRL_WEIGHT_READY |
        DBG_CTRL_ACTS_READY | DBG_CTRL_OUTPUT_VALID) != 0 or
        reuse & (DBG_CTRL_ACTIVATION_ERROR | DBG_CTRL_KERNEL_BUSY) !=
            (DBG_CTRL_ACTIVATION_ERROR | DBG_CTRL_KERNEL_BUSY))
        return error.KernelFinishAbortWasReusable;
    dut.step();
    if (c.dut_dbg_control_boundaries(dut.h) & DBG_CTRL_KERNEL_BUSY != 0)
        return error.KernelBadReuseDidNotRetire;

    std.debug.print(
        "decode_top r7 kernel abort: idle unchanged; LOAD_ACTS/WISSUE/" ++
            "stalled/live-final EMIT quarantined; held ERROR retired; " ++
            "FINISH reuse rejected\n",
        .{},
    );
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

fn beginGammaLoadChecked(dut: *Dut) !void {
    c.dut_eval(dut.h);
    const before = c.dut_dbg_control_boundaries(dut.h);
    if (before & (DBG_CTRL_GAMMA_BUSY | DBG_CTRL_GAMMA_RAW_BUSY |
        DBG_CTRL_GAMMA_PENDING | DBG_CTRL_GAMMA_FIRE) != 0)
        return error.P3dGammaBoundaryWasNotIdle;

    c.dut_set_axi_write(dut.h, REG_NORM_CTRL, 1, 1);
    dut.step();
    dut.step();
    const admitted = c.dut_dbg_control_boundaries(dut.h);
    if (admitted & (DBG_CTRL_GAMMA_ADMIT | DBG_CTRL_GAMMA_READY) !=
        (DBG_CTRL_GAMMA_ADMIT | DBG_CTRL_GAMMA_READY) or
        admitted & (DBG_CTRL_GAMMA_PENDING | DBG_CTRL_GAMMA_FIRE |
            DBG_CTRL_GAMMA_RAW_BUSY | DBG_CTRL_GAMMA_BUSY) != 0 or
        admitted & (DBG_CTRL_GAMMA_RAW_VALID | DBG_CTRL_GAMMA_VALID) !=
            before & (DBG_CTRL_GAMMA_RAW_VALID | DBG_CTRL_GAMMA_VALID))
        return error.P3dGammaAdmissionBoundaryMismatch;

    c.dut_set_axi_idle(dut.h);
    dut.step();
    const pending = c.dut_dbg_control_boundaries(dut.h);
    if (pending & (DBG_CTRL_GAMMA_PENDING | DBG_CTRL_GAMMA_FIRE |
        DBG_CTRL_GAMMA_READY | DBG_CTRL_GAMMA_BUSY) !=
        (DBG_CTRL_GAMMA_PENDING | DBG_CTRL_GAMMA_FIRE |
            DBG_CTRL_GAMMA_READY | DBG_CTRL_GAMMA_BUSY) or
        pending & (DBG_CTRL_GAMMA_RAW_BUSY | DBG_CTRL_GAMMA_VALID |
            DBG_CTRL_ACTS_READY) != 0 or
        pending & DBG_CTRL_GAMMA_RAW_VALID !=
            before & DBG_CTRL_GAMMA_RAW_VALID)
        return error.P3dGammaPendingBoundaryMismatch;

    dut.step();
    const started = c.dut_dbg_control_boundaries(dut.h);
    if (started & (DBG_CTRL_GAMMA_RAW_BUSY | DBG_CTRL_GAMMA_BUSY |
        DBG_CTRL_ACTS_READY) !=
        (DBG_CTRL_GAMMA_RAW_BUSY | DBG_CTRL_GAMMA_BUSY |
            DBG_CTRL_ACTS_READY) or
        started & (DBG_CTRL_GAMMA_ADMIT | DBG_CTRL_GAMMA_PENDING |
            DBG_CTRL_GAMMA_FIRE | DBG_CTRL_GAMMA_RAW_VALID |
            DBG_CTRL_GAMMA_VALID) != 0)
        return error.P3dGammaLeafBoundaryMismatch;
}

fn loadP3dGamma(dut: *Dut, rows: usize, gamma_beats: []const u64) !void {
    if (gamma_beats.len != rows / 2) return error.P3dGammaShapeMismatch;
    axiWrite(dut, REG_MODEL_ROWS, @intCast(rows));
    try beginGammaLoadChecked(dut);

    for (gamma_beats, 0..) |beat, index| {
        if (index % 5 == 2) {
            c.dut_set_a(dut.h, 0, 0, 0);
            dut.step();
        }
        var accepted = false;
        for (0..256) |_| {
            c.dut_set_a(
                dut.h,
                beat,
                1,
                @intFromBool(index + 1 == gamma_beats.len),
            );
            c.dut_eval(dut.h);
            const fire = c.dut_a_ready(dut.h) != 0;
            dut.step();
            if (fire) {
                accepted = true;
                break;
            }
        }
        if (!accepted) return error.P3dGammaReadyTimeout;
    }
    c.dut_set_a(dut.h, 0, 0, 0);

    for (0..256) |_| {
        const status = try axiRead(dut, REG_NORM_STATUS);
        if (status & NORM_GAMMA_DONE != 0) {
            if (status & (NORM_GAMMA_BUSY | NORM_GAMMA_ERROR) != 0 or
                status & NORM_GAMMA_VALID == 0 or
                try axiRead(dut, REG_QUANT_STATUS) != 0)
                return error.P3dGammaStatusMismatch;
            return;
        }
        dut.step();
    }
    return error.P3dGammaDoneTimeout;
}

fn abortGammaReplacementAtPending(dut: *Dut, rows: usize) !void {
    axiWrite(dut, REG_MODEL_ROWS, @intCast(rows));
    c.dut_eval(dut.h);
    const before = c.dut_dbg_control_boundaries(dut.h);
    if (before & (DBG_CTRL_GAMMA_RAW_VALID | DBG_CTRL_GAMMA_VALID) !=
        (DBG_CTRL_GAMMA_RAW_VALID | DBG_CTRL_GAMMA_VALID) or
        before & DBG_CTRL_GAMMA_BUSY != 0)
        return error.P3dGammaReplacementRequiresValidTable;

    c.dut_set_axi_write(dut.h, REG_NORM_CTRL, 1, 1);
    dut.step();
    dut.step();
    const admitted = c.dut_dbg_control_boundaries(dut.h);
    if (admitted & (DBG_CTRL_GAMMA_ADMIT | DBG_CTRL_GAMMA_READY |
        DBG_CTRL_GAMMA_RAW_VALID | DBG_CTRL_GAMMA_VALID) !=
        (DBG_CTRL_GAMMA_ADMIT | DBG_CTRL_GAMMA_READY |
            DBG_CTRL_GAMMA_RAW_VALID | DBG_CTRL_GAMMA_VALID) or
        admitted & (DBG_CTRL_GAMMA_PENDING | DBG_CTRL_GAMMA_FIRE |
            DBG_CTRL_GAMMA_BUSY) != 0)
        return error.P3dGammaReplacementAdmissionMismatch;

    c.dut_set_axi_idle(dut.h);
    dut.step();
    const pending = c.dut_dbg_control_boundaries(dut.h);
    if (pending & (DBG_CTRL_GAMMA_PENDING | DBG_CTRL_GAMMA_FIRE |
        DBG_CTRL_GAMMA_READY | DBG_CTRL_GAMMA_BUSY |
        DBG_CTRL_GAMMA_RAW_VALID) !=
        (DBG_CTRL_GAMMA_PENDING | DBG_CTRL_GAMMA_FIRE |
            DBG_CTRL_GAMMA_READY | DBG_CTRL_GAMMA_BUSY |
            DBG_CTRL_GAMMA_RAW_VALID) or
        pending & (DBG_CTRL_GAMMA_VALID | DBG_CTRL_GAMMA_RAW_BUSY |
            DBG_CTRL_ACTS_READY) != 0)
        return error.P3dGammaReplacementPendingMismatch;

    c.dut_force_scratch_abort_strobe(dut.h, 1);
    c.dut_eval(dut.h);
    const suppressed = c.dut_dbg_control_boundaries(dut.h);
    if (suppressed & (DBG_CTRL_GAMMA_PENDING | DBG_CTRL_GAMMA_BUSY |
        DBG_CTRL_GAMMA_RAW_VALID | DBG_CTRL_KERNEL_ABORT_NOW) !=
        (DBG_CTRL_GAMMA_PENDING | DBG_CTRL_GAMMA_BUSY |
            DBG_CTRL_GAMMA_RAW_VALID | DBG_CTRL_KERNEL_ABORT_NOW) or
        suppressed & (DBG_CTRL_GAMMA_FIRE | DBG_CTRL_GAMMA_VALID |
            DBG_CTRL_KERNEL_ABORT_Q | DBG_CTRL_WEIGHT_READY |
            DBG_CTRL_ACTS_READY | DBG_CTRL_OUTPUT_VALID |
            DBG_CTRL_SCRATCH_R_WRITE | DBG_CTRL_SCRATCH_READ_ACCEPT) != 0)
        return error.P3dGammaReplacementAbortDidNotQuarantine;

    dut.step();
    const cleared = c.dut_dbg_control_boundaries(dut.h);
    if (cleared & (DBG_CTRL_GAMMA_PENDING | DBG_CTRL_GAMMA_FIRE |
        DBG_CTRL_GAMMA_RAW_BUSY | DBG_CTRL_GAMMA_BUSY |
        DBG_CTRL_GAMMA_RAW_VALID | DBG_CTRL_GAMMA_VALID |
        DBG_CTRL_KERNEL_ABORT_Q | DBG_CTRL_SCRATCH_R_WRITE |
        DBG_CTRL_SCRATCH_READ_ACCEPT) != 0)
        return error.P3dGammaReplacementAbortWasNotCleared;

    c.dut_force_scratch_abort_strobe(dut.h, 0);
    c.dut_set_axi_idle(dut.h);
    dut.step();
    if (try axiRead(dut, REG_NORM_STATUS) != NORM_GLOBAL_IDLE)
        return error.P3dGammaReplacementAbortStatusMismatch;
}

fn verifyGammaOverlapRejection(
    dut: *Dut,
    rows: usize,
    gamma_beats: []const u64,
) !void {
    if (gamma_beats.len != rows / 2) return error.P3dGammaShapeMismatch;
    if (try axiRead(dut, REG_SCRATCH_STATUS) & SCRATCH_R_VALID == 0)
        return error.P3dGammaOverlapRequiresResidentR;

    axiWrite(dut, REG_MODEL_ROWS, @intCast(rows));
    axiWrite(dut, REG_NORM_CTRL, 1);
    if (try axiRead(dut, REG_NORM_STATUS) & NORM_GAMMA_BUSY == 0)
        return error.P3dGammaOverlapDidNotStart;

    // R has valid metadata, so this drain would be legal except that gamma owns
    // the shared activation interface. It must retire rejected without reading R.
    configureScratch(dut, SCRATCH_MODE_DRAIN, SCRATCH_ROLE_R, rows, 1);
    axiWrite(dut, REG_SCRATCH_CTRL, SCRATCH_CTRL_DRAIN_START);
    var scratch_status = try axiRead(dut, REG_SCRATCH_STATUS);
    if (scratch_status & (SCRATCH_DRAIN_DONE | SCRATCH_R_VALID) !=
        (SCRATCH_DRAIN_DONE | SCRATCH_R_VALID) or
        scratch_status & SCRATCH_DRAIN_BUSY != 0 or
        try axiRead(dut, REG_SCRATCH_ERROR) & SCRATCH_ERROR_CONFIG == 0)
        return error.P3dGammaOverlapDrainWasNotRejected;

    // MODEL_ROWS=0 selects valid legacy CTRL.START/SECTION_BEGIN shapes. Both
    // must fail closed while the gamma frame is still open.
    axiWrite(dut, REG_MODEL_ROWS, 0);
    configureScratch(dut, SCRATCH_MODE_DDR, SCRATCH_ROLE_R, rows, 1);
    configureProjection(
        dut,
        rows,
        rows,
        rows / layout.Q1_BLOCK,
        1,
        WEIGHT_FMT_BINARY,
        ACT_RAW_LOAD,
        0xF15F_6A00,
    );
    if (try axiRead(dut, REG_STATUS) & 3 != 2 or
        c.dut_dbg_ffn_lifecycle(dut.h) &
            (DBG_FFN_ACTIVE | DBG_FFN_KERNEL_BUSY) != 0 or
        c.dut_dbg_ffn_phase(dut.h) != 0)
        return error.P3dGammaOverlapKernelStartWasNotRejected;

    axiWrite(dut, REG_SCRATCH_CTRL, SCRATCH_CTRL_SECTION_BEGIN);
    scratch_status = try axiRead(dut, REG_SCRATCH_STATUS);
    if (scratch_status & SCRATCH_SECTION_ACTIVE != 0 or
        scratch_status & (SCRATCH_SECTION_DONE | SCRATCH_ANY_ERROR) !=
            (SCRATCH_SECTION_DONE | SCRATCH_ANY_ERROR) or
        c.dut_dbg_ffn_phase(dut.h) != 0)
        return error.P3dGammaOverlapSectionBeginWasNotRejected;

    axiWrite(dut, REG_MODEL_ROWS, @intCast(rows));
    var zero = [_]u32{0} ** ROWS_PER_PORT;
    for (0..PORTS) |port| c.dut_set_w(dut.h, @intCast(port), &zero, 1);
    for (gamma_beats, 0..) |beat, index| {
        var accepted = false;
        for (0..256) |_| {
            c.dut_set_a(
                dut.h,
                beat,
                1,
                @intFromBool(index + 1 == gamma_beats.len),
            );
            c.dut_eval(dut.h);
            const owners = c.dut_dbg_shared_activation(dut.h);
            if (owners & DBG_SHARED_GAMMA_BUSY == 0 or
                owners & (DBG_SHARED_COMPUTE_VALID | DBG_SHARED_Q8_START |
                    DBG_SHARED_KERNEL_START) != 0)
                return error.P3dGammaOverlapSharedConsumerOpened;
            if ((c.dut_a_ready(dut.h) != 0) !=
                (owners & DBG_SHARED_GAMMA_READY != 0))
                return error.P3dGammaOverlapReadyOwnerMismatch;
            for (0..PORTS) |port| {
                if (c.dut_w_ready(dut.h, @intCast(port)) != 0)
                    return error.P3dGammaOverlapConsumedWeight;
            }
            if (c.dut_m_valid(dut.h) != 0 or
                c.dut_dbg_ffn_lifecycle(dut.h) & DBG_FFN_KERNEL_BUSY != 0)
                return error.P3dGammaOverlapOpenedKernelStream;
            const fire = c.dut_a_ready(dut.h) != 0;
            dut.step();
            if (fire) {
                accepted = true;
                break;
            }
        }
        if (!accepted) return error.P3dGammaOverlapReadyTimeout;
    }
    for (0..PORTS) |port| c.dut_set_w(dut.h, @intCast(port), &zero, 0);
    c.dut_set_a(dut.h, 0, 0, 0);

    for (0..256) |_| {
        const status = try axiRead(dut, REG_NORM_STATUS);
        if (status & NORM_GAMMA_DONE != 0) {
            if (status & (NORM_GAMMA_BUSY | NORM_GAMMA_ERROR) != 0 or
                status & NORM_GAMMA_VALID == 0 or
                try axiRead(dut, REG_W_BEATS) != 0 or
                try axiRead(dut, REG_A_BEATS) != 0 or
                try axiRead(dut, REG_R_BEATS) != 0)
                return error.P3dGammaOverlapTerminalMismatch;
            return;
        }
        dut.step();
    }
    return error.P3dGammaOverlapDoneTimeout;
}

fn rejectEarlyP3dGammaFrame(dut: *Dut, rows: usize, first_beat: u64) !void {
    axiWrite(dut, REG_MODEL_ROWS, @intCast(rows));
    axiWrite(dut, REG_NORM_CTRL, 1);
    var accepted = false;
    for (0..256) |_| {
        c.dut_set_a(dut.h, first_beat, 1, 1);
        c.dut_eval(dut.h);
        const fire = c.dut_a_ready(dut.h) != 0;
        dut.step();
        if (fire) {
            accepted = true;
            break;
        }
    }
    c.dut_set_a(dut.h, 0, 0, 0);
    if (!accepted) return error.P3dGammaFaultReadyTimeout;

    for (0..256) |_| {
        const status = try axiRead(dut, REG_NORM_STATUS);
        if (status & NORM_GAMMA_DONE != 0) {
            if (status & NORM_GAMMA_ERROR == 0 or
                status & (NORM_GAMMA_BUSY | NORM_GAMMA_VALID) != 0 or
                try axiRead(dut, REG_NORM_ERROR) == 0)
                return error.P3dGammaFaultStatusMismatch;
            return;
        }
        dut.step();
    }
    return error.P3dGammaFaultDoneTimeout;
}

fn beginP3dSection(
    dut: *Dut,
    model_rows: usize,
    ffn_rows: usize,
    tokens: usize,
    resident: bool,
) !void {
    axiWrite(dut, REG_MODEL_ROWS, @intCast(model_rows));
    axiWrite(dut, REG_NORM_EPS, 0x3586_37bd);
    configureScratch(dut, SCRATCH_MODE_DDR, SCRATCH_ROLE_X0, ffn_rows, tokens);
    axiWrite(
        dut,
        REG_SCRATCH_CTRL,
        SCRATCH_CTRL_SECTION_BEGIN |
            (if (resident) SCRATCH_CTRL_RESIDENT_R else 0),
    );
    const scratch_status = try axiRead(dut, REG_SCRATCH_STATUS);
    const norm_status = try axiRead(dut, REG_NORM_STATUS);
    if (scratch_status & SCRATCH_SECTION_ACTIVE == 0 or
        scratch_status & (SCRATCH_SECTION_DONE | SCRATCH_ANY_ERROR) != 0 or
        norm_status & (NORM_BUSY | NORM_GAMMA_VALID) !=
            (NORM_BUSY | NORM_GAMMA_VALID) or
        norm_status & (NORM_DONE | NORM_ERROR | RESIDUAL_DONE |
            RESIDUAL_ERROR) != 0)
        return error.P3dSectionBeginRejected;
    if (c.dut_dbg_ffn_phase(dut.h) != 1)
        return error.P3dSectionBeginPhaseMismatch;
    if (resident and c.dut_a_ready(dut.h) != 0)
        return error.P3dResidentOpenedResidualIngress;
}

fn abortP3dImmediatelyAfterBegin(
    dut: *Dut,
    model_rows: usize,
    ffn_rows: usize,
    tokens: usize,
) !void {
    axiWrite(dut, REG_MODEL_ROWS, @intCast(model_rows));
    axiWrite(dut, REG_NORM_EPS, 0x3586_37bd);
    configureScratch(dut, SCRATCH_MODE_DDR, SCRATCH_ROLE_X0, ffn_rows, tokens);

    // Stop between controller acceptance and the delayed leaf handshake. The
    // direct strobe drive models the already-decoded AXI abort pulse without
    // spending another AXI transaction's cycles before reaching this boundary.
    c.dut_set_axi_write(dut.h, REG_SCRATCH_CTRL, SCRATCH_CTRL_SECTION_BEGIN, 1);
    var saw_begin = false;
    for (0..8) |_| {
        dut.step();
        const launch = c.dut_dbg_p3d_launch(dut.h);
        if (launch & DBG_P3D_BEGIN_OK != 0) {
            if (launch & (DBG_P3D_LEAF_START_Q | DBG_P3D_LEAF_START |
                DBG_P3D_LAUNCH_Q8 | DBG_P3D_ABORT_STROBE |
                DBG_P3D_RMS_BUSY) != 0)
                return error.P3dBeginLaunchedLeafSameCycle;
            saw_begin = true;
            break;
        }
    }
    if (!saw_begin) return error.P3dImmediateAbortBeginTimeout;

    dut.step();
    var launch = c.dut_dbg_p3d_launch(dut.h);
    if (launch & (DBG_P3D_LEAF_START_Q | DBG_P3D_LEAF_START |
        DBG_P3D_LAUNCH_Q8) != (DBG_P3D_LEAF_START_Q |
        DBG_P3D_LEAF_START | DBG_P3D_LAUNCH_Q8) or
        launch & (DBG_P3D_BEGIN_OK | DBG_P3D_RMS_BUSY) != 0)
        return error.P3dDelayedLeafLaunchMismatch;

    c.dut_set_axi_idle(dut.h);
    c.dut_force_scratch_abort_strobe(dut.h, 1);
    c.dut_eval(dut.h);
    launch = c.dut_dbg_p3d_launch(dut.h);
    if (launch & (DBG_P3D_LEAF_START_Q | DBG_P3D_ABORT_STROBE) !=
        (DBG_P3D_LEAF_START_Q | DBG_P3D_ABORT_STROBE) or
        launch & (DBG_P3D_LEAF_START | DBG_P3D_LAUNCH_Q8 |
            DBG_P3D_RMS_BUSY) != 0 or
        c.dut_a_ready(dut.h) != 0 or c.dut_m_valid(dut.h) != 0)
    {
        std.debug.print(
            "P3d immediate-abort quarantine mismatch: launch=0x{x} a_ready={d} m_valid={d}\n",
            .{ launch, c.dut_a_ready(dut.h), c.dut_m_valid(dut.h) },
        );
        return error.P3dImmediateAbortDidNotSuppressLeaf;
    }

    dut.step();
    c.dut_force_scratch_abort_strobe(dut.h, 0);
    c.dut_set_axi_idle(dut.h);
    dut.step();
    try waitP3dTerminal(dut, false, SCRATCH_ERROR_ABORT);
    if (try axiRead(dut, REG_NORM_STATUS) != NORM_GLOBAL_IDLE or
        try axiRead(dut, REG_NORM_ERROR) != 0 or
        try axiRead(dut, REG_RESIDUAL_ERROR) != 0)
        return error.P3dImmediateAbortStatusMismatch;
}

fn sendP3dResidual(dut: *Dut, residual_beats: []const u64) !void {
    for (residual_beats, 0..) |beat, index| {
        if (index % 7 == 3) {
            c.dut_set_a(dut.h, 0, 0, 0);
            dut.step();
            dut.step();
        }
        var accepted = false;
        for (0..256) |_| {
            c.dut_set_a(
                dut.h,
                beat,
                1,
                @intFromBool(index + 1 == residual_beats.len),
            );
            c.dut_eval(dut.h);
            if (c.dut_m_valid(dut.h) != 0)
                return error.P3dResidualLoadExposedOutput;
            const fire = c.dut_a_ready(dut.h) != 0;
            dut.step();
            if (fire) {
                accepted = true;
                break;
            }
        }
        if (!accepted) return error.P3dResidualReadyTimeout;
    }
    c.dut_set_a(dut.h, 0, 0, 0);
}

fn verifyP3dRmsQ8Accounting(dut: *Dut, expected_records: u32) !void {
    for (0..CYCLE_LIMIT) |_| {
        c.dut_eval(dut.h);
        if (c.dut_dbg_p3d_q8_accounting(dut.h) &
            DBG_RMS_Q8_SEALED != 0) break;
        dut.step();
    } else return error.P3dRmsQ8AccountingTimeout;

    const state = c.dut_dbg_p3d_q8_accounting(dut.h);
    const monitor = &dut.rms_q8_monitor;
    if (!monitor.armed or monitor.failed or
        monitor.expected != expected_records or
        monitor.accepts != expected_records or
        monitor.fires != expected_records or monitor.finals != 1 or
        monitor.seal_events != 1 or
        dbgRmsQ8Count(state) != expected_records or
        state & (DBG_RMS_Q8_DONE | DBG_RMS_Q8_SEALED) !=
            (DBG_RMS_Q8_DONE | DBG_RMS_Q8_SEALED) or
        state & (DBG_RMS_Q8_ACCEPT | DBG_RMS_Q8_FIRE |
            DBG_RMS_Q8_OWNER) != 0)
    {
        std.debug.print(
            "P3d RMS accounting mismatch: state=0x{x} failed={} accepts/fires/finals/seals={d}/{d}/{d}/{d}\n",
            .{
                state,
                monitor.failed,
                monitor.accepts,
                monitor.fires,
                monitor.finals,
                monitor.seal_events,
            },
        );
        return error.P3dRmsQ8AccountingMismatch;
    }
    monitor.armed = false;
}

fn abortP3dAtRmsRecord(
    dut: *Dut,
    residual_beats: []const u64,
    rows: usize,
    q1_blocks: usize,
    tokens: usize,
    epoch: u32,
    ports: [PORTS][]const u8,
    abort_after_capture: bool,
) !void {
    const initial = c.dut_dbg_p3d_q8_accounting(dut.h);
    if (initial & (DBG_RMS_Q8_ACCEPT | DBG_RMS_Q8_FIRE |
        DBG_RMS_Q8_DONE | DBG_RMS_Q8_SEALED) != 0 or
        dbgRmsQ8Count(initial) != 0)
        return error.P3dRmsQ8AbortStartRetainedStaleEvent;

    try sendP3dResidual(dut, residual_beats);
    configureScratch(dut, SCRATCH_MODE_ONLY, SCRATCH_ROLE_X1, rows, tokens);
    configureProjection(
        dut,
        rows,
        rows,
        q1_blocks,
        tokens,
        WEIGHT_FMT_BINARY,
        ACT_RAW_LOAD,
        epoch,
    );

    const w_beats = ports[0].len / PORT_BEAT_BYTES;
    var wi = [_]usize{0} ** PORTS;
    var saw_pending = false;
    for (0..CYCLE_LIMIT) |_| {
        var w_valid = [_]bool{false} ** PORTS;
        for (0..PORTS) |port| {
            const valid = wi[port] < w_beats;
            w_valid[port] = valid;
            const words = if (valid)
                readPortBeat(ports[port], wi[port])
            else
                [_]u32{0} ** ROWS_PER_PORT;
            c.dut_set_w(
                dut.h,
                @intCast(port),
                &words,
                @intFromBool(valid),
            );
        }
        c.dut_set_a(dut.h, 0, 0, 0);
        c.dut_eval(dut.h);
        const pending = c.dut_dbg_p3d_q8_accounting(dut.h);
        if (pending & DBG_RMS_Q8_ACCEPT != 0) {
            if (pending & (DBG_RMS_Q8_FIRE | DBG_RMS_Q8_DONE |
                DBG_RMS_Q8_SEALED) != 0 or dbgRmsQ8Count(pending) != 0)
                return error.P3dRmsQ8AbortWasNotAtFirstRawCompletion;
            saw_pending = true;
            break;
        }
        var w_fire = [_]bool{false} ** PORTS;
        for (0..PORTS) |port|
            w_fire[port] = w_valid[port] and
                c.dut_w_ready(dut.h, @intCast(port)) != 0;
        dut.step();
        for (0..PORTS) |port| {
            if (w_fire[port]) wi[port] += 1;
        }
    }
    if (!saw_pending) {
        std.debug.print(
            "P3d RMS pending-abort timeout: accounting=0x{x} lifecycle=0x{x}\n",
            .{
                c.dut_dbg_p3d_q8_accounting(dut.h),
                c.dut_dbg_p3d_lifecycle(dut.h),
            },
        );
        return error.P3dRmsQ8AbortPendingTimeout;
    }

    var zero = [_]u32{0} ** ROWS_PER_PORT;
    for (0..PORTS) |port|
        c.dut_set_w(dut.h, @intCast(port), &zero, 0);
    if (abort_after_capture) {
        dut.step();
        const registered = c.dut_dbg_p3d_q8_accounting(dut.h);
        if (registered & DBG_RMS_Q8_FIRE == 0 or
            registered & (DBG_RMS_Q8_ACCEPT | DBG_RMS_Q8_DONE |
                DBG_RMS_Q8_SEALED) != 0 or
            dbgRmsQ8Count(registered) != 0)
            return error.P3dRmsQ8RegisteredEventWasNotPending;
    }
    c.dut_force_scratch_abort_strobe(dut.h, 1);
    c.dut_eval(dut.h);
    const suppressed = c.dut_dbg_p3d_q8_accounting(dut.h);
    if (suppressed & (DBG_RMS_Q8_ACCEPT | DBG_RMS_Q8_FIRE) != 0 or
        dbgRmsQ8Count(suppressed) != 0)
        return error.P3dRmsQ8InterveningAbortDidNotSuppressEvent;
    dut.step();
    const cleared = c.dut_dbg_p3d_q8_accounting(dut.h);
    if (cleared & (DBG_RMS_Q8_ACCEPT | DBG_RMS_Q8_FIRE |
        DBG_RMS_Q8_DONE | DBG_RMS_Q8_SEALED) != 0 or
        dbgRmsQ8Count(cleared) != 0)
        return error.P3dRmsQ8AbortLatchedStaleEvent;

    c.dut_force_scratch_abort_strobe(dut.h, 0);
    c.dut_set_axi_idle(dut.h);
    dut.step();
    try waitP3dTerminal(dut, false, SCRATCH_ERROR_ABORT);
    const terminal = c.dut_dbg_p3d_q8_accounting(dut.h);
    if (terminal & (DBG_RMS_Q8_ACCEPT | DBG_RMS_Q8_FIRE |
        DBG_RMS_Q8_DONE | DBG_RMS_Q8_SEALED | DBG_RMS_Q8_OWNER |
        DBG_RMS_Q8_ACTIVE) != 0 or dbgRmsQ8Count(terminal) != 0)
        return error.P3dRmsQ8AbortWasNotRestartSafe;
}

fn waitP3dTerminal(
    dut: *Dut,
    section_done: bool,
    expected_error_mask: u32,
) !void {
    var terminal = false;
    for (0..16384) |_| {
        c.dut_eval(dut.h);
        if (c.dut_m_valid(dut.h) != 0)
            return error.P3dFaultEscapedQuarantine;
        const lifecycle = c.dut_dbg_p3d_lifecycle(dut.h);
        if (lifecycle & (DBG_P3D_ACTIVE | DBG_P3D_CLEANUP |
            DBG_P3D_KILL | DBG_P3D_OUTER_OWNER_MASK |
            DBG_P3D_SUBOWNER_MASK | DBG_P3D_RSP_VALID) == 0)
        {
            terminal = true;
            break;
        }
        dut.step();
    }
    if (!terminal) {
        std.debug.print(
            "P3d cleanup timeout: lifecycle=0x{x} phase={d} scratch=0x{x}/0x{x} norm=0x{x}/0x{x}\n",
            .{
                c.dut_dbg_p3d_lifecycle(dut.h),
                c.dut_dbg_ffn_phase(dut.h),
                try axiRead(dut, REG_SCRATCH_STATUS),
                try axiRead(dut, REG_SCRATCH_ERROR),
                try axiRead(dut, REG_NORM_STATUS),
                try axiRead(dut, REG_NORM_ERROR),
            },
        );
        return error.P3dFaultCleanupTimeout;
    }

    const scratch_status = try axiRead(dut, REG_SCRATCH_STATUS);
    const scratch_error = try axiRead(dut, REG_SCRATCH_ERROR);
    if ((scratch_status & SCRATCH_SECTION_DONE != 0) != section_done or
        scratch_status & SCRATCH_SECTION_ACTIVE != 0 or
        scratch_status & SCRATCH_R_VALID != 0 or
        scratch_status & SCRATCH_ANY_ERROR == 0 or
        scratch_error & expected_error_mask != expected_error_mask)
        return error.P3dFaultTerminalStatusMismatch;
}

fn faultP3dInactiveNormOwner(dut: *Dut) !void {
    const before = c.dut_dbg_p3d_lifecycle(dut.h);
    if (before & DBG_P3D_ACTIVE == 0 or
        before & (DBG_P3D_CLEANUP | DBG_P3D_RESIDUAL_STARTED) != 0)
        return error.P3dInactiveNormSetupMismatch;

    c.dut_set_inactive_norm_owner(dut.h, 1);
    c.dut_set_m_ready(dut.h, 1);
    c.dut_eval(dut.h);
    if (c.dut_m_valid(dut.h) != 0)
        return error.P3dInactiveNormExposedOutput;
    dut.step();
    c.dut_set_inactive_norm_owner(dut.h, 0);
    c.dut_eval(dut.h);

    const diagnosed = c.dut_dbg_p3d_lifecycle(dut.h);
    if (diagnosed & (DBG_P3D_ACTIVE | DBG_P3D_CLEANUP | DBG_P3D_KILL) !=
        (DBG_P3D_ACTIVE | DBG_P3D_CLEANUP | DBG_P3D_KILL) or
        diagnosed & DBG_P3D_RESIDUAL_STARTED != 0 or
        c.dut_m_valid(dut.h) != 0)
        return error.P3dInactiveNormDiagnosisMismatch;

    try waitP3dTerminal(dut, true, SCRATCH_ERROR_RMS);
    const status = try axiRead(dut, REG_NORM_STATUS);
    if (status & (NORM_DONE | NORM_ERROR) != (NORM_DONE | NORM_ERROR) or
        status & (RESIDUAL_DONE | RESIDUAL_ERROR) != 0 or
        try axiRead(dut, REG_NORM_ERROR) == 0)
        return error.P3dInactiveNormStatusMismatch;
}

fn faultP3dInactiveResidualAtLaunch(
    dut: *Dut,
    model_rows: usize,
    ffn_rows: usize,
    tokens: usize,
    epoch: u32,
) !void {
    configureScratch(dut, SCRATCH_MODE_DDR, SCRATCH_ROLE_X0, ffn_rows, tokens);
    configureProjection(
        dut,
        model_rows,
        model_rows,
        ffn_rows / layout.Q1_BLOCK,
        tokens,
        WEIGHT_FMT_BINARY,
        ACT_SCRATCH_SWIGLU,
        epoch,
    );

    var collided = false;
    for (0..64) |_| {
        c.dut_eval(dut.h);
        const lifecycle = c.dut_dbg_p3d_lifecycle(dut.h);
        if (lifecycle & (DBG_P3D_DOWN_COMMITTED | DBG_P3D_DOWN_LAUNCH) ==
            (DBG_P3D_DOWN_COMMITTED | DBG_P3D_DOWN_LAUNCH))
        {
            if (c.dut_dbg_shared_activation(dut.h) &
                DBG_SHARED_KERNEL_START == 0)
                return error.P3dCommittedDownDidNotLaunchKernel;
            c.dut_set_inactive_residual_owner(dut.h, 1);
            c.dut_eval(dut.h);
            const collided_lifecycle = c.dut_dbg_p3d_lifecycle(dut.h);
            const control = c.dut_dbg_control_boundaries(dut.h);
            if (collided_lifecycle & DBG_P3D_DOWN_LAUNCH == 0 or
                c.dut_dbg_shared_activation(dut.h) &
                    DBG_SHARED_KERNEL_START == 0 or
                c.dut_m_valid(dut.h) != 0 or
                control & DBG_CTRL_SCRATCH_R_WRITE != 0)
                return error.P3dInactiveResidualLaunchCollisionMismatch;
            dut.step();
            c.dut_set_inactive_residual_owner(dut.h, 0);
            collided = true;
            break;
        }
        dut.step();
    }
    if (!collided) return error.P3dCommittedDownLaunchTimeout;

    c.dut_eval(dut.h);
    const diagnosed = c.dut_dbg_p3d_lifecycle(dut.h);
    if (diagnosed & (DBG_P3D_ACTIVE | DBG_P3D_CLEANUP | DBG_P3D_KILL |
        DBG_P3D_RESIDUAL_STARTED) != (DBG_P3D_ACTIVE | DBG_P3D_CLEANUP |
        DBG_P3D_KILL | DBG_P3D_RESIDUAL_STARTED) or
        c.dut_m_valid(dut.h) != 0)
        return error.P3dInactiveResidualLaunchDiagnosisMismatch;

    try waitP3dTerminal(dut, true, SCRATCH_ERROR_RESIDUAL);
    const status = try axiRead(dut, REG_NORM_STATUS);
    if (status & (NORM_DONE | RESIDUAL_DONE | RESIDUAL_ERROR) !=
        (NORM_DONE | RESIDUAL_DONE | RESIDUAL_ERROR) or
        status & NORM_ERROR != 0 or
        try axiRead(dut, REG_RESIDUAL_ERROR) == 0)
        return error.P3dInactiveResidualLaunchStatusMismatch;
}

fn faultP3dInactiveResidualAtFinalOutput(
    dut: *Dut,
    model_rows: usize,
    ffn_rows: usize,
    tokens: usize,
    ports: [PORTS][]const u8,
) !void {
    configureScratch(dut, SCRATCH_MODE_DDR, SCRATCH_ROLE_X0, ffn_rows, tokens);
    configureProjection(
        dut,
        model_rows,
        model_rows,
        ffn_rows / layout.Q1_BLOCK,
        tokens,
        WEIGHT_FMT_BINARY,
        ACT_SCRATCH_SWIGLU,
        0xF15F_FA22,
    );

    const w_beats = ports[0].len / PORT_BEAT_BYTES;
    var wi = [_]usize{0} ** PORTS;
    var collided = false;
    for (0..CYCLE_LIMIT) |_| {
        var w_valid = [_]bool{false} ** PORTS;
        for (0..PORTS) |port| {
            const valid = wi[port] < w_beats;
            w_valid[port] = valid;
            const words = if (valid)
                readPortBeat(ports[port], wi[port])
            else
                [_]u32{0} ** ROWS_PER_PORT;
            c.dut_set_w(
                dut.h,
                @intCast(port),
                &words,
                @intFromBool(valid),
            );
        }
        c.dut_set_a(dut.h, 0x7fc0_0000_7fc0_0000, 1, 1);
        c.dut_set_m_ready(dut.h, 1);
        c.dut_eval(dut.h);
        var w_fire = [_]bool{false} ** PORTS;
        for (0..PORTS) |port|
            w_fire[port] = w_valid[port] and
                c.dut_w_ready(dut.h, @intCast(port)) != 0;

        if (c.dut_m_valid(dut.h) != 0 and c.dut_m_last(dut.h) != 0) {
            if (c.dut_dbg_p3d_lifecycle(dut.h) &
                DBG_P3D_RESIDUAL_STARTED == 0)
                return error.P3dInactiveResidualFinalPhaseMismatch;
            c.dut_set_inactive_residual_owner(dut.h, 1);
            c.dut_eval(dut.h);
            if (c.dut_m_valid(dut.h) != 0 or
                c.dut_dbg_control_boundaries(dut.h) &
                    DBG_CTRL_SCRATCH_R_WRITE != 0)
                return error.P3dInactiveResidualFinalWasNotQuarantined;
            dut.step();
            c.dut_set_inactive_residual_owner(dut.h, 0);
            for (0..PORTS) |port| {
                if (w_fire[port]) wi[port] += 1;
            }
            collided = true;
            break;
        }

        dut.step();
        for (0..PORTS) |port| {
            if (w_fire[port]) wi[port] += 1;
        }
    }
    c.dut_set_a(dut.h, 0, 0, 0);
    var zero = [_]u32{0} ** ROWS_PER_PORT;
    for (0..PORTS) |port| c.dut_set_w(dut.h, @intCast(port), &zero, 0);
    if (!collided) return error.P3dInactiveResidualFinalTimeout;

    c.dut_eval(dut.h);
    const diagnosed = c.dut_dbg_p3d_lifecycle(dut.h);
    if (diagnosed & (DBG_P3D_ACTIVE | DBG_P3D_CLEANUP | DBG_P3D_KILL |
        DBG_P3D_RESIDUAL_STARTED) != (DBG_P3D_ACTIVE | DBG_P3D_CLEANUP |
        DBG_P3D_KILL | DBG_P3D_RESIDUAL_STARTED) or
        c.dut_m_valid(dut.h) != 0)
        return error.P3dInactiveResidualFinalDiagnosisMismatch;

    try waitP3dTerminal(dut, true, SCRATCH_ERROR_RESIDUAL);
    const status = try axiRead(dut, REG_NORM_STATUS);
    if (status & (NORM_DONE | RESIDUAL_DONE | RESIDUAL_ERROR) !=
        (NORM_DONE | RESIDUAL_DONE | RESIDUAL_ERROR) or
        status & NORM_ERROR != 0 or
        try axiRead(dut, REG_RESIDUAL_ERROR) == 0)
        return error.P3dInactiveResidualFinalStatusMismatch;
}

fn rejectEarlyP3dResidualFrame(dut: *Dut, first_beat: u64) !void {
    var accepted = false;
    for (0..256) |_| {
        c.dut_set_a(dut.h, first_beat, 1, 1);
        c.dut_eval(dut.h);
        const fire = c.dut_a_ready(dut.h) != 0;
        dut.step();
        if (fire) {
            accepted = true;
            break;
        }
    }
    c.dut_set_a(dut.h, 0, 0, 0);
    if (!accepted) return error.P3dResidualFrameReadyTimeout;
    try waitP3dTerminal(
        dut,
        true,
        SCRATCH_ERROR_SECTION | SCRATCH_ERROR_RMS,
    );
    const norm_status = try axiRead(dut, REG_NORM_STATUS);
    if (norm_status & (NORM_DONE | NORM_ERROR) !=
        (NORM_DONE | NORM_ERROR) or
        norm_status & (NORM_GAMMA_VALID | RESIDUAL_DONE |
            RESIDUAL_ERROR) != 0 or
        try axiRead(dut, REG_NORM_ERROR) == 0)
        return error.P3dResidualFrameStatusMismatch;
}

fn abortP3dWithRetainedRmsResponse(dut: *Dut) !void {
    var retained = false;
    for (0..4096) |_| {
        c.dut_eval(dut.h);
        const lifecycle = c.dut_dbg_p3d_lifecycle(dut.h);
        if (lifecycle & (DBG_P3D_OUTER_OWNER_MASK |
            DBG_P3D_SUBOWNER_MASK | DBG_P3D_RSP_VALID) ==
            (DBG_P3D_OUTER_OWNER | DBG_P3D_SUBOWNER_RMS |
                DBG_P3D_RSP_VALID))
        {
            retained = true;
            break;
        }
        dut.step();
    }
    if (!retained) return error.P3dRetainedResponseTimeout;

    axiWrite(dut, REG_SCRATCH_CTRL, SCRATCH_CTRL_ABORT);
    c.dut_eval(dut.h);
    const aborted = c.dut_dbg_p3d_lifecycle(dut.h);
    if (aborted & (DBG_P3D_ACTIVE | DBG_P3D_CLEANUP | DBG_P3D_KILL |
        DBG_P3D_OUTER_OWNER_MASK | DBG_P3D_SUBOWNER_MASK |
        DBG_P3D_RSP_VALID) !=
        (DBG_P3D_ACTIVE | DBG_P3D_CLEANUP | DBG_P3D_KILL |
            DBG_P3D_OUTER_OWNER | DBG_P3D_SUBOWNER_RMS |
            DBG_P3D_RSP_VALID))
        return error.P3dAbortDidNotRetainResponse;

    c.dut_set_p3d_read_response_hold(dut.h, 0);
    try waitP3dTerminal(dut, false, SCRATCH_ERROR_ABORT);
    if (try axiRead(dut, REG_NORM_STATUS) != NORM_GLOBAL_IDLE or
        try axiRead(dut, REG_NORM_ERROR) != 0 or
        try axiRead(dut, REG_RESIDUAL_ERROR) != 0)
        return error.P3dAbortStatusMismatch;
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

const GateRun = struct {
    scratch_reads: usize,
    swiglu_fires: usize,
    record_fires: usize,
    capture_fires: usize,
    phases_seen: u8,
};

fn waitGateReady(dut: *Dut) !void {
    var cycles: usize = 0;
    while (cycles < CYCLE_LIMIT) : (cycles += 64) {
        for (0..64) |_| dut.step();
        const status = try axiRead(dut, REG_SCRATCH_STATUS);
        if (status & SCRATCH_ANY_ERROR != 0) return error.FfnGateReadyFault;
        if (status & SCRATCH_GATE_READY != 0) {
            if (c.dut_dbg_ffn_phase(dut.h) != 3)
                return error.FfnGateReadyPhaseMismatch;
            return;
        }
    }
    std.debug.print("FFN gate-ready timeout: phase={d} status=0x{x} error=0x{x}\n", .{
        c.dut_dbg_ffn_phase(dut.h),
        try axiRead(dut, REG_SCRATCH_STATUS),
        try axiRead(dut, REG_SCRATCH_ERROR),
    });
    return error.FfnGateReadyTimeout;
}

fn runStreamingGateProjection(
    dut: *Dut,
    rows: usize,
    tokens: usize,
    epoch: u32,
    ports: [PORTS][]const u8,
) !GateRun {
    try waitGateReady(dut);
    configureScratch(dut, SCRATCH_MODE_ONLY, SCRATCH_ROLE_X0, rows, tokens);
    configureProjection(
        dut,
        rows,
        rows,
        rows / layout.Q1_BLOCK,
        tokens,
        WEIGHT_FMT_BINARY,
        ACT_REUSE,
        epoch,
    );

    const w_beats = ports[0].len / PORT_BEAT_BYTES;
    const blocks = rows / shared_layout.q8_block;
    const expected_records = blocks * tokens;
    var wi = [_]usize{0} ** PORTS;
    var scratch_reads: usize = 0;
    var swiglu_fires: usize = 0;
    var record_fires: usize = 0;
    var capture_fires: usize = 0;
    var phases_seen: u8 = 0;

    for (0..CYCLE_LIMIT) |cycle| {
        var w_valid = [_]bool{false} ** PORTS;
        for (0..PORTS) |port| {
            const valid = wi[port] < w_beats;
            w_valid[port] = valid;
            const words = if (valid) readPortBeat(ports[port], wi[port]) else [_]u32{0} ** ROWS_PER_PORT;
            c.dut_set_w(dut.h, @intCast(port), &words, @intFromBool(valid));
        }
        c.dut_set_a(dut.h, 0x7fc0_0000_7fc0_0000, 1, 1);
        c.dut_set_m_ready(dut.h, @intFromBool(cycle % 11 < 5));
        c.dut_eval(dut.h);

        const phase: u3 = @intCast(c.dut_dbg_ffn_phase(dut.h));
        phases_seen |= @as(u8, 1) << phase;
        if (c.dut_m_valid(dut.h) != 0)
            return error.FfnGateExposedResult;
        if (c.dut_a_ready(dut.h) != 0)
            return error.FfnGateAcceptedExternalActivation;
        if (c.dut_dbg_scratch_read_fire(dut.h) != 0)
            scratch_reads += 1;
        if (c.dut_dbg_swiglu_input_fire(dut.h) != 0)
            swiglu_fires += 1;
        if (c.dut_dbg_internal_record_done(dut.h) != 0)
            record_fires += 1;
        if (c.dut_dbg_capture_fire(dut.h) != 0) {
            const tag = c.dut_dbg_capture_tag(dut.h);
            const record = capture_fires / 5;
            const want_beat = capture_fires % 5;
            const want_token = record % tokens;
            const want_block = record / tokens;
            if ((tag & 7) != want_beat or
                ((tag >> 3) & 3) != want_token or
                (tag >> 5) != want_block)
                return error.FfnCaptureOrderMismatch;
            capture_fires += 1;
        }

        var w_fire = [_]bool{false} ** PORTS;
        for (0..PORTS) |port|
            w_fire[port] = w_valid[port] and c.dut_w_ready(dut.h, @intCast(port)) != 0;
        dut.step();
        for (0..PORTS) |port| {
            if (w_fire[port]) wi[port] += 1;
        }

        const post_phase: u3 = @intCast(c.dut_dbg_ffn_phase(dut.h));
        phases_seen |= @as(u8, 1) << post_phase;
        if (post_phase == 6) break;
    } else {
        std.debug.print("FFN producer timeout: phase={d} status=0x{x} error=0x{x}\n", .{
            c.dut_dbg_ffn_phase(dut.h),
            try axiRead(dut, REG_SCRATCH_STATUS),
            try axiRead(dut, REG_SCRATCH_ERROR),
        });
        return error.FfnProducerTimeout;
    }

    var zero = [_]u32{0} ** ROWS_PER_PORT;
    for (0..PORTS) |port| c.dut_set_w(dut.h, @intCast(port), &zero, 0);
    c.dut_set_a(dut.h, 0, 0, 0);
    if (wi[0] != w_beats or scratch_reads != rows / 8 * tokens or
        swiglu_fires != rows * tokens or record_fires != expected_records or
        capture_fires != expected_records * 5)
        return error.FfnProducerTraversalMismatch;
    if (phases_seen & ((@as(u8, 1) << 4) | (@as(u8, 1) << 5) |
        (@as(u8, 1) << 6)) != ((@as(u8, 1) << 4) |
        (@as(u8, 1) << 5) | (@as(u8, 1) << 6)))
        return error.FfnProducerPhaseMismatch;
    if ((try axiRead(dut, REG_STATUS)) & 2 == 0 or
        try axiRead(dut, REG_W_BEATS) != w_beats or
        try axiRead(dut, REG_A_BEATS) != 0 or
        try axiRead(dut, REG_R_BEATS) != 0)
        return error.FfnGateCounterMismatch;
    const status = try axiRead(dut, REG_SCRATCH_STATUS);
    if (status & (SCRATCH_X1_VALID | SCRATCH_CONSUMER_DONE |
        SCRATCH_SECTION_ACTIVE) != (SCRATCH_X1_VALID |
        SCRATCH_CONSUMER_DONE | SCRATCH_SECTION_ACTIVE) or
        status & (SCRATCH_X0_VALID | SCRATCH_CONSUMER_BUSY |
            SCRATCH_SECTION_DONE | SCRATCH_GATE_READY | SCRATCH_ANY_ERROR) != 0)
        return error.FfnProducerStatusMismatch;
    return .{
        .scratch_reads = scratch_reads,
        .swiglu_fires = swiglu_fires,
        .record_fires = record_fires,
        .capture_fires = capture_fires,
        .phases_seen = phases_seen,
    };
}

fn faultP3dGateQ8(
    dut: *Dut,
    rows: usize,
    tokens: usize,
    epoch: u32,
) !void {
    try waitGateReady(dut);
    configureScratch(dut, SCRATCH_MODE_ONLY, SCRATCH_ROLE_X0, rows, tokens);
    configureProjection(
        dut,
        rows,
        rows,
        rows / layout.Q1_BLOCK,
        tokens,
        WEIGHT_FMT_BINARY,
        ACT_REUSE,
        epoch,
    );

    var entered_cleanup = false;
    var injected = false;
    for (0..128) |_| {
        c.dut_eval(dut.h);
        if (c.dut_a_ready(dut.h) != 0 or c.dut_m_valid(dut.h) != 0)
            return error.P3dGateFaultExposedStream;
        const lifecycle = c.dut_dbg_p3d_lifecycle(dut.h);
        if (!injected and lifecycle & DBG_P3D_Q8_OWNER_MASK ==
            DBG_P3D_Q8_OWNER_GATE)
        {
            c.dut_set_gate_q8_numeric_error(dut.h, 1);
            c.dut_eval(dut.h);
            dut.step();
            c.dut_set_gate_q8_numeric_error(dut.h, 0);
            c.dut_eval(dut.h);
            const captured = c.dut_dbg_p3d_lifecycle(dut.h);
            if (captured & DBG_P3D_Q8_FAULT == 0 or
                captured & DBG_P3D_Q8_FAULT_OWNER_MASK !=
                    DBG_P3D_Q8_FAULT_OWNER_GATE)
                return error.P3dGateFaultOwnerWasNotCaptured;
            injected = true;
            continue;
        }
        if (c.dut_dbg_p3d_lifecycle(dut.h) & DBG_P3D_CLEANUP != 0) {
            entered_cleanup = true;
            break;
        }
        dut.step();
    }
    c.dut_set_gate_q8_numeric_error(dut.h, 0);
    if (!injected) return error.P3dGateOwnerWasNotAcquired;
    if (!entered_cleanup) return error.P3dGateFaultDidNotTrigger;
    try waitP3dTerminal(
        dut,
        true,
        SCRATCH_ERROR_SECTION | SCRATCH_ERROR_SWIGLU_Q8,
    );
    const norm_status = try axiRead(dut, REG_NORM_STATUS);
    if (norm_status & (NORM_DONE | NORM_ERROR) !=
        (NORM_DONE | NORM_ERROR) or
        norm_status & (NORM_GAMMA_VALID | RESIDUAL_DONE |
            RESIDUAL_ERROR) != 0 or
        try axiRead(dut, REG_NORM_ERROR) & (1 << 27) == 0)
        return error.P3dGateFaultStatusMismatch;
    try expectPreservedP3dQuantStatus(dut);
}

fn expectPreservedP3dQuantStatus(dut: *Dut) !void {
    const terminal = try axiRead(dut, REG_QUANT_STATUS);
    if (terminal == 0) return error.P3dQuantFaultStatusMissing;
    for (0..8) |_| dut.step();
    if (try axiRead(dut, REG_QUANT_STATUS) != terminal)
        return error.P3dQuantFaultStatusNotPreserved;
}

fn faultP3dRmsQ8(dut: *Dut, residual_beats: []const u64) !void {
    try sendP3dResidual(dut, residual_beats);
    if (c.dut_dbg_p3d_lifecycle(dut.h) & DBG_P3D_Q8_OWNER_MASK !=
        DBG_P3D_Q8_OWNER_RMS)
        return error.P3dRmsQ8OwnerMissing;

    c.dut_set_gate_q8_numeric_error(dut.h, 1);
    var entered_cleanup = false;
    var captured_owner = false;
    for (0..128) |_| {
        c.dut_eval(dut.h);
        if (c.dut_m_valid(dut.h) != 0)
            return error.P3dRmsQ8FaultExposedStream;
        const lifecycle = c.dut_dbg_p3d_lifecycle(dut.h);
        if (lifecycle & DBG_P3D_Q8_FAULT != 0) {
            if (lifecycle & DBG_P3D_Q8_FAULT_OWNER_MASK !=
                DBG_P3D_Q8_FAULT_OWNER_RMS)
                return error.P3dRmsQ8FaultOwnerMismatch;
            captured_owner = true;
        }
        if (c.dut_dbg_p3d_lifecycle(dut.h) & DBG_P3D_CLEANUP != 0) {
            entered_cleanup = true;
            break;
        }
        dut.step();
    }
    c.dut_set_gate_q8_numeric_error(dut.h, 0);
    if (!captured_owner) return error.P3dRmsQ8FaultOwnerWasNotCaptured;
    if (!entered_cleanup) return error.P3dRmsQ8FaultDidNotTrigger;
    try waitP3dTerminal(
        dut,
        true,
        SCRATCH_ERROR_SECTION | SCRATCH_ERROR_RMS,
    );
    const norm_status = try axiRead(dut, REG_NORM_STATUS);
    if (norm_status & (NORM_DONE | NORM_ERROR) !=
        (NORM_DONE | NORM_ERROR) or
        norm_status & (NORM_GAMMA_VALID | RESIDUAL_DONE |
            RESIDUAL_ERROR) != 0 or
        try axiRead(dut, REG_NORM_ERROR) == 0)
        return error.P3dRmsQ8FaultStatusMismatch;
    try expectPreservedP3dQuantStatus(dut);
}

fn rejectDuplicateGateAtWaitDown(
    dut: *Dut,
    rows: usize,
    tokens: usize,
    epoch: u32,
) !void {
    if (c.dut_dbg_ffn_phase(dut.h) != 6)
        return error.DuplicateGateTestNotAtWaitDown;
    configureScratch(dut, SCRATCH_MODE_ONLY, SCRATCH_ROLE_X0, rows, tokens);
    configureProjection(
        dut,
        rows,
        rows,
        rows / layout.Q1_BLOCK,
        tokens,
        WEIGHT_FMT_BINARY,
        ACT_REUSE,
        epoch,
    );

    var zero = [_]u32{0} ** ROWS_PER_PORT;
    for (0..16) |_| {
        for (0..PORTS) |port| c.dut_set_w(dut.h, @intCast(port), &zero, 1);
        c.dut_set_a(dut.h, 0x7fc0_0000_7fc0_0000, 1, 1);
        c.dut_set_m_ready(dut.h, 1);
        c.dut_eval(dut.h);
        for (0..PORTS) |port| {
            if (c.dut_w_ready(dut.h, @intCast(port)) != 0)
                return error.DuplicateGateConsumedWeight;
        }
        if (c.dut_a_ready(dut.h) != 0 or c.dut_m_valid(dut.h) != 0)
            return error.DuplicateGateOpenedStream;
        dut.step();
    }
    for (0..PORTS) |port| c.dut_set_w(dut.h, @intCast(port), &zero, 0);
    c.dut_set_a(dut.h, 0, 0, 0);

    const status = try axiRead(dut, REG_SCRATCH_STATUS);
    if (try axiRead(dut, REG_STATUS) & 2 == 0 or
        try axiRead(dut, REG_W_BEATS) != 0 or
        try axiRead(dut, REG_A_BEATS) != 0 or
        try axiRead(dut, REG_R_BEATS) != 0 or
        status & (SCRATCH_X1_VALID | SCRATCH_CONSUMER_DONE |
            SCRATCH_SECTION_ACTIVE | SCRATCH_ANY_ERROR) !=
            (SCRATCH_X1_VALID | SCRATCH_CONSUMER_DONE |
                SCRATCH_SECTION_ACTIVE | SCRATCH_ANY_ERROR) or
        status & (SCRATCH_WRITER_BUSY | SCRATCH_X0_VALID |
            SCRATCH_CONSUMER_BUSY | SCRATCH_SECTION_DONE |
            SCRATCH_GATE_READY) != 0 or
        try axiRead(dut, REG_SCRATCH_ERROR) != SCRATCH_ERROR_CONFIG or
        c.dut_dbg_ffn_phase(dut.h) != 6)
        return error.DuplicateGateDidNotRejectAtPreflight;
}

fn abortStreamingGateWithOutstandingRead(
    dut: *Dut,
    rows: usize,
    tokens: usize,
    epoch: u32,
    ports: [PORTS][]const u8,
) !void {
    try waitGateReady(dut);
    configureScratch(dut, SCRATCH_MODE_ONLY, SCRATCH_ROLE_X0, rows, tokens);
    configureProjection(
        dut,
        rows,
        rows,
        rows / layout.Q1_BLOCK,
        tokens,
        WEIGHT_FMT_BINARY,
        ACT_REUSE,
        epoch,
    );

    const w_beats = ports[0].len / PORT_BEAT_BYTES;
    var wi = [_]usize{0} ** PORTS;
    var armed = false;
    for (0..CYCLE_LIMIT) |_| {
        var w_valid = [_]bool{false} ** PORTS;
        for (0..PORTS) |port| {
            const valid = wi[port] < w_beats;
            w_valid[port] = valid;
            const words = if (valid) readPortBeat(ports[port], wi[port]) else [_]u32{0} ** ROWS_PER_PORT;
            c.dut_set_w(dut.h, @intCast(port), &words, @intFromBool(valid));
        }
        c.dut_set_a(dut.h, 0x7fc0_0000_7fc0_0000, 1, 1);
        c.dut_set_m_ready(dut.h, 1);
        c.dut_eval(dut.h);
        if (c.dut_m_valid(dut.h) != 0 or c.dut_a_ready(dut.h) != 0)
            return error.FfnAbortExposedExternalStream;

        var w_fire = [_]bool{false} ** PORTS;
        for (0..PORTS) |port|
            w_fire[port] = w_valid[port] and c.dut_w_ready(dut.h, @intCast(port)) != 0;

        const lifecycle = c.dut_dbg_ffn_lifecycle(dut.h);
        if (lifecycle & DBG_FFN_PAIRER_STAGING_REQ != 0) {
            if (lifecycle & (DBG_FFN_OWNER_MASK | DBG_FFN_RSP_VALID |
                DBG_FFN_PAIRER_ORPHAN | DBG_FFN_ABORT_CLEANUP) != 0)
                return error.FfnAbortRequestWasNotFresh;

            // Advance the staged request to the shared scratch arbiter first.
            // The Q8 diagnostic is then captured on the request-accept edge, so
            // its registered self-abort sees an already-retained owner.
            dut.step();
            for (0..PORTS) |port| {
                if (w_fire[port]) wi[port] += 1;
            }
            var zero = [_]u32{0} ** ROWS_PER_PORT;
            for (0..PORTS) |port| c.dut_set_w(dut.h, @intCast(port), &zero, 0);
            c.dut_set_a(dut.h, 0, 0, 0);
            c.dut_eval(dut.h);
            if (c.dut_dbg_scratch_read_fire(dut.h) == 0)
                return error.FfnAbortReadDidNotIssue;

            c.dut_set_gate_q8_numeric_error(dut.h, 1);
            dut.step();
            c.dut_set_gate_q8_numeric_error(dut.h, 0);
            c.dut_eval(dut.h);
            const captured = c.dut_dbg_p3d_lifecycle(dut.h);
            if (captured & DBG_P3D_Q8_FAULT == 0 or
                captured & DBG_P3D_Q8_FAULT_OWNER_MASK !=
                    DBG_P3D_Q8_FAULT_OWNER_GATE)
                return error.FfnGateNumericFaultOwnerMismatch;
            const owned = c.dut_dbg_ffn_lifecycle(dut.h);
            if (owned & DBG_FFN_OWNER_MASK != DBG_FFN_OWNER_PAIRER or
                owned & DBG_FFN_ACTIVE == 0 or
                owned & (DBG_FFN_RSP_VALID | DBG_FFN_PAIRER_ORPHAN) != 0)
                return error.FfnAbortReadWasNotOutstanding;

            c.dut_set_axi_write(dut.h, REG_SCRATCH_CTRL, SCRATCH_CTRL_ABORT, 1);
            dut.step();
            c.dut_set_axi_idle(dut.h);
            c.dut_eval(dut.h);
            const fault_error = c.dut_dbg_scratch_error(dut.h);
            if (fault_error & (SCRATCH_ERROR_SECTION | SCRATCH_ERROR_SWIGLU_Q8) !=
                (SCRATCH_ERROR_SECTION | SCRATCH_ERROR_SWIGLU_Q8))
            {
                std.debug.print("GATE numeric diagnostic after registered capture: 0x{x}\n", .{fault_error});
                return error.FfnGateNumericDiagnosticMismatch;
            }
            const aborted = c.dut_dbg_ffn_lifecycle(dut.h);
            if (aborted & (DBG_FFN_ACTIVE | DBG_FFN_ABORT_CLEANUP |
                DBG_FFN_PAIRER_ORPHAN) != (DBG_FFN_ACTIVE |
                DBG_FFN_ABORT_CLEANUP | DBG_FFN_PAIRER_ORPHAN) or
                aborted & DBG_FFN_OWNER_MASK != DBG_FFN_OWNER_PAIRER or
                aborted & DBG_FFN_RSP_VALID != 0)
                return error.FfnAbortDidNotRetainOrphan;
            armed = true;
            break;
        }

        dut.step();
        for (0..PORTS) |port| {
            if (w_fire[port]) wi[port] += 1;
        }
    }
    if (!armed) return error.FfnAbortStagingTimeout;

    var saw_owner = false;
    var saw_orphan = false;
    var saw_response = false;
    var saw_post_drain_hold = false;
    var reached_idle = false;
    for (0..256) |_| {
        c.dut_eval(dut.h);
        if (c.dut_m_valid(dut.h) != 0 or c.dut_a_ready(dut.h) != 0)
            return error.FfnAbortCleanupExposedStream;
        const lifecycle = c.dut_dbg_ffn_lifecycle(dut.h);
        const owner = lifecycle & DBG_FFN_OWNER_MASK;
        saw_owner = saw_owner or owner == DBG_FFN_OWNER_PAIRER;
        saw_orphan = saw_orphan or lifecycle & DBG_FFN_PAIRER_ORPHAN != 0;
        saw_response = saw_response or lifecycle & DBG_FFN_RSP_VALID != 0;
        if (lifecycle & DBG_FFN_ACTIVE != 0 and
            lifecycle & DBG_FFN_ABORT_CLEANUP != 0 and owner == 0 and
            lifecycle & (DBG_FFN_RSP_VALID | DBG_FFN_PAIRER_ORPHAN) == 0)
            saw_post_drain_hold = true;

        const retained = lifecycle & (DBG_FFN_PRODUCER_BUSY |
            DBG_FFN_ABORT_CLEANUP | DBG_FFN_OWNER_MASK | DBG_FFN_RSP_VALID |
            DBG_FFN_PAIRER_ORPHAN | DBG_FFN_PAIRER_BUSY |
            DBG_FFN_KERNEL_BUSY | DBG_FFN_PACKER_BUSY);
        if (retained != 0 and lifecycle & DBG_FFN_ACTIVE == 0)
            return error.FfnSectionActiveClearedBeforeDrain;
        if (lifecycle & DBG_FFN_ACTIVE == 0) {
            if (retained != 0 or lifecycle & (DBG_FFN_SECTION_DONE |
                DBG_FFN_ANY_ERROR) != (DBG_FFN_SECTION_DONE | DBG_FFN_ANY_ERROR))
                return error.FfnAbortCleanupWasNotRestartSafe;
            reached_idle = true;
            break;
        }
        dut.step();
    }
    if (!reached_idle or !saw_owner or !saw_orphan or !saw_response or
        !saw_post_drain_hold)
        return error.FfnAbortLifecycleCoverageMissing;

    const status = try axiRead(dut, REG_SCRATCH_STATUS);
    if (status & (SCRATCH_SECTION_ACTIVE | SCRATCH_CONSUMER_BUSY |
        SCRATCH_CONSUMER_DONE) != 0 or
        status & (SCRATCH_SECTION_DONE | SCRATCH_ANY_ERROR) !=
            (SCRATCH_SECTION_DONE | SCRATCH_ANY_ERROR) or
        try axiRead(dut, REG_SCRATCH_ERROR) &
            (SCRATCH_ERROR_ABORT | SCRATCH_ERROR_SECTION |
                SCRATCH_ERROR_SWIGLU_Q8) !=
            (SCRATCH_ERROR_ABORT | SCRATCH_ERROR_SECTION |
                SCRATCH_ERROR_SWIGLU_Q8) or
        try axiRead(dut, REG_QUANT_STATUS) != 0 or
        try axiRead(dut, REG_STATUS) & 2 == 0)
        return error.FfnAbortTerminalStatusMismatch;
}

const InternalRun = struct {
    stream: []u8,
    replay_fires: usize,
    cycles: usize,
};

fn runInternalDown(
    a: std.mem.Allocator,
    dut: *Dut,
    rows: usize,
    ffn_dim: usize,
    tokens: usize,
    ports: [PORTS][]const u8,
    expected_scratch_error: u32,
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
    var replay_fires: usize = 0;
    var cycles: usize = 0;
    const blocks = ffn_dim / shared_layout.q8_block;
    const expected_records = blocks * tokens;

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
        if (c.dut_dbg_replay_fire(dut.h) != 0) {
            const tag = c.dut_dbg_replay_tag(dut.h);
            const record = replay_fires / 5;
            const want_beat = replay_fires % 5;
            const want_token = record / blocks;
            const want_block = record % blocks;
            if ((tag & 7) != want_beat or
                ((tag >> 3) & 3) != want_token or
                (tag >> 5) != want_block)
                return error.FfnReplayOrderMismatch;
            replay_fires += 1;
        }
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
        try axiRead(dut, REG_A_BEATS) != expected_records * 5 or
        try axiRead(dut, REG_R_BEATS) != stream.len / 8 or
        try axiRead(dut, REG_QUANT_STATUS) != 0)
        return error.InternalDownCounterMismatch;
    const status = try axiRead(dut, REG_SCRATCH_STATUS);
    const expected_any_error: u32 = if (expected_scratch_error != 0)
        SCRATCH_ANY_ERROR
    else
        0;
    if (status & (SCRATCH_SECTION_DONE | SCRATCH_CONSUMER_DONE |
        SCRATCH_ANY_ERROR) != (SCRATCH_SECTION_DONE | SCRATCH_CONSUMER_DONE |
        expected_any_error) or
        status & (SCRATCH_SECTION_ACTIVE | SCRATCH_CONSUMER_BUSY |
            SCRATCH_X0_VALID | SCRATCH_X1_VALID) != 0)
        return error.InternalDownStatusMismatch;
    if (try axiRead(dut, REG_SCRATCH_ERROR) != expected_scratch_error)
        return error.InternalDownErrorMismatch;
    if (c.dut_dbg_ffn_phase(dut.h) != 0 or replay_fires != expected_records * 5)
        return error.InternalDownTraversalMismatch;
    return .{
        .stream = stream,
        .replay_fires = replay_fires,
        .cycles = cycles,
    };
}

fn abortInternalDownAtKernelState(
    dut: *Dut,
    rows: usize,
    ffn_dim: usize,
    tokens: usize,
    ports: [PORTS][]const u8,
    target_state: u32,
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
        0xF15F_A900 | target_state,
    );
    const w_beats = ports[0].len / PORT_BEAT_BYTES;
    const expected_replays = ffn_dim / shared_layout.q8_block * tokens * 5;
    var wi = [_]usize{0} ** PORTS;
    var replay_fires: usize = 0;
    const output_ready = target_state != KERNEL_ST_EMIT;
    var reached = false;

    for (0..CYCLE_LIMIT) |_| {
        var w_valid = [_]bool{false} ** PORTS;
        for (0..PORTS) |port| {
            const valid = wi[port] < w_beats;
            w_valid[port] = valid;
            const words = if (valid)
                readPortBeat(ports[port], wi[port])
            else
                [_]u32{0} ** ROWS_PER_PORT;
            c.dut_set_w(dut.h, @intCast(port), &words, @intFromBool(valid));
        }
        c.dut_set_a(dut.h, 0x7fc0_0000_7fc0_0000, 1, 1);
        c.dut_set_m_ready(dut.h, @intFromBool(output_ready));
        c.dut_eval(dut.h);
        if (c.dut_a_ready(dut.h) != 0)
            return error.InternalDownAbortAcceptedExternalActivation;
        if (c.dut_dbg_replay_fire(dut.h) != 0)
            replay_fires += 1;

        const control = c.dut_dbg_control_boundaries(dut.h);
        if (dbgKernelState(control) == target_state) {
            if (control & DBG_CTRL_KERNEL_BUSY == 0 or
                (target_state == KERNEL_ST_EMIT and
                    control & DBG_CTRL_OUTPUT_VALID == 0) or
                (target_state == KERNEL_ST_FINISH and
                    (wi[0] != w_beats or replay_fires != expected_replays)))
                return error.InternalDownAbortTargetMismatch;
            reached = true;
            break;
        }

        var w_fire = [_]bool{false} ** PORTS;
        for (0..PORTS) |port|
            w_fire[port] = w_valid[port] and
                c.dut_w_ready(dut.h, @intCast(port)) != 0;
        dut.step();
        for (0..PORTS) |port| {
            if (w_fire[port]) wi[port] += 1;
        }
    }
    if (!reached) return error.InternalDownAbortTargetTimeout;

    c.dut_force_scratch_abort_strobe(dut.h, 1);
    c.dut_eval(dut.h);
    var control = c.dut_dbg_control_boundaries(dut.h);
    if (control & DBG_CTRL_KERNEL_ABORT_NOW == 0 or
        control & DBG_CTRL_KERNEL_ABORT_Q != 0 or
        control & (DBG_CTRL_WEIGHT_READY | DBG_CTRL_ACTS_READY |
            DBG_CTRL_OUTPUT_VALID | DBG_CTRL_SCRATCH_R_WRITE |
            DBG_CTRL_SCRATCH_READ_ACCEPT |
            DBG_CTRL_KERNEL_SINK_ACCEPT) != 0)
        return error.InternalDownAbortSameCycleQuarantineMismatch;
    dut.step();

    control = c.dut_dbg_control_boundaries(dut.h);
    const lifecycle = c.dut_dbg_ffn_lifecycle(dut.h);
    if (control & DBG_CTRL_KERNEL_ABORT_Q == 0 or
        control & (DBG_CTRL_WEIGHT_READY | DBG_CTRL_ACTS_READY |
            DBG_CTRL_OUTPUT_VALID | DBG_CTRL_KERNEL_SINK_ACCEPT) != 0 or
        lifecycle & (DBG_FFN_ACTIVE | DBG_FFN_ABORT_CLEANUP) !=
            (DBG_FFN_ACTIVE | DBG_FFN_ABORT_CLEANUP))
        return error.InternalDownAbortRegisteredBoundaryMismatch;

    c.dut_force_scratch_abort_strobe(dut.h, 0);
    clearKernelBoundaryInputs(dut);
    var retired = false;
    for (0..256) |_| {
        c.dut_eval(dut.h);
        if (c.dut_m_valid(dut.h) != 0 or c.dut_a_ready(dut.h) != 0)
            return error.InternalDownAbortCleanupExposedStream;
        const state = c.dut_dbg_ffn_lifecycle(dut.h);
        if (state & DBG_FFN_ACTIVE == 0) {
            if (state & DBG_FFN_ABORT_CLEANUP != 0)
                return error.InternalDownAbortCleanupOwnerMismatch;
            retired = true;
            break;
        }
        dut.step();
    }
    if (!retired) return error.InternalDownAbortCleanupTimeout;

    const status = try axiRead(dut, REG_SCRATCH_STATUS);
    if (status & (SCRATCH_SECTION_DONE | SCRATCH_ANY_ERROR) !=
        (SCRATCH_SECTION_DONE | SCRATCH_ANY_ERROR) or
        status & SCRATCH_SECTION_ACTIVE != 0 or
        try axiRead(dut, REG_SCRATCH_ERROR) & SCRATCH_ERROR_ABORT == 0 or
        try axiRead(dut, REG_ACT_STATE) != 2)
        return error.InternalDownAbortStatusMismatch;
}

fn runP3dDown(
    a: std.mem.Allocator,
    dut: *Dut,
    model_rows: usize,
    ffn_rows: usize,
    tokens: usize,
    ports: [PORTS][]const u8,
) !InternalRun {
    configureScratch(dut, SCRATCH_MODE_DDR, SCRATCH_ROLE_X0, ffn_rows, tokens);
    configureProjection(
        dut,
        model_rows,
        model_rows,
        ffn_rows / layout.Q1_BLOCK,
        tokens,
        WEIGHT_FMT_BINARY,
        ACT_SCRATCH_SWIGLU,
        0xF15F_D003,
    );

    const w_beats = ports[0].len / PORT_BEAT_BYTES;
    const stream = try a.alloc(u8, model_rows * tokens * @sizeOf(f32));
    errdefer a.free(stream);
    var wi = [_]usize{0} ** PORTS;
    var out_offset: usize = 0;
    var saw_last = false;
    var replay_fires: usize = 0;
    var cycles: usize = 0;
    const blocks = ffn_rows / shared_layout.q8_block;
    const expected_records = blocks * tokens;

    // M_AXIS is the private S2MM quarantine allocation for this section. Beats
    // are tentative until the clean terminal status below; software discards
    // the allocation after any section error.
    for (0..CYCLE_LIMIT) |cycle| {
        cycles = cycle + 1;
        var w_valid = [_]bool{false} ** PORTS;
        for (0..PORTS) |port| {
            const valid = wi[port] < w_beats;
            w_valid[port] = valid;
            const words = if (valid)
                readPortBeat(ports[port], wi[port])
            else
                [_]u32{0} ** ROWS_PER_PORT;
            c.dut_set_w(
                dut.h,
                @intCast(port),
                &words,
                @intFromBool(valid),
            );
        }
        c.dut_set_a(dut.h, 0x7fc0_0000_7fc0_0000, 1, 1);
        const ready = cycle % 17 < 9;
        c.dut_set_m_ready(dut.h, @intFromBool(ready));
        c.dut_eval(dut.h);
        if (c.dut_a_ready(dut.h) != 0)
            return error.P3dDownAcceptedExternalActivation;
        if (c.dut_dbg_replay_fire(dut.h) != 0) replay_fires += 1;

        var w_fire = [_]bool{false} ** PORTS;
        for (0..PORTS) |port|
            w_fire[port] = w_valid[port] and
                c.dut_w_ready(dut.h, @intCast(port)) != 0;
        if (c.dut_m_valid(dut.h) != 0 and ready) {
            if (c.dut_m_keep(dut.h) != 0xff or
                out_offset + 8 > stream.len)
                return error.P3dDownFraming;
            std.mem.writeInt(
                u64,
                stream[out_offset..][0..8],
                c.dut_m_data(dut.h),
                .little,
            );
            out_offset += 8;
            if (c.dut_m_last(dut.h) != 0) {
                if (saw_last or out_offset != stream.len)
                    return error.P3dDownFraming;
                saw_last = true;
            }
        }
        dut.step();
        for (0..PORTS) |port| {
            if (w_fire[port]) wi[port] += 1;
        }
        if (saw_last) break;
    } else return error.P3dDownTimeout;

    c.dut_set_a(dut.h, 0, 0, 0);
    var zero = [_]u32{0} ** ROWS_PER_PORT;
    for (0..PORTS) |port| c.dut_set_w(dut.h, @intCast(port), &zero, 0);
    c.dut_set_m_ready(dut.h, 1);

    var terminal = false;
    for (0..512) |_| {
        const status = try axiRead(dut, REG_SCRATCH_STATUS);
        if (status & SCRATCH_SECTION_DONE != 0) {
            terminal = true;
            break;
        }
        dut.step();
    }
    if (!terminal) return error.P3dCleanTerminalTimeout;
    if (wi[0] != w_beats or replay_fires != expected_records * 5 or
        !saw_last or out_offset != stream.len)
        return error.P3dDownTraversalMismatch;

    const scratch_status = try axiRead(dut, REG_SCRATCH_STATUS);
    if (scratch_status & (SCRATCH_SECTION_DONE | SCRATCH_R_VALID) !=
        (SCRATCH_SECTION_DONE | SCRATCH_R_VALID) or
        scratch_status & (SCRATCH_SECTION_ACTIVE | SCRATCH_ANY_ERROR |
            SCRATCH_X0_VALID | SCRATCH_X1_VALID | SCRATCH_CONSUMER_BUSY) != 0 or
        try axiRead(dut, REG_SCRATCH_ERROR) != 0)
    {
        std.debug.print("P3d clean scratch status=0x{x} error=0x{x}\n", .{
            scratch_status,
            try axiRead(dut, REG_SCRATCH_ERROR),
        });
        return error.P3dCleanScratchStatusMismatch;
    }

    const norm_status = try axiRead(dut, REG_NORM_STATUS);
    const expected_norm = NORM_GAMMA_DONE | NORM_GAMMA_VALID |
        NORM_DONE | RESIDUAL_DONE | NORM_GLOBAL_IDLE;
    if (norm_status != expected_norm or
        try axiRead(dut, REG_NORM_ERROR) != 0 or
        try axiRead(dut, REG_RESIDUAL_ERROR) != 0)
        return error.P3dCleanNormStatusMismatch;
    return .{
        .stream = stream,
        .replay_fires = replay_fires,
        .cycles = cycles,
    };
}

fn abortP3dAtStalledResidualOutput(
    dut: *Dut,
    model_rows: usize,
    ffn_rows: usize,
    tokens: usize,
    ports: [PORTS][]const u8,
) !void {
    c.dut_set_m_ready(dut.h, 0);
    configureScratch(dut, SCRATCH_MODE_DDR, SCRATCH_ROLE_X0, ffn_rows, tokens);
    configureProjection(
        dut,
        model_rows,
        model_rows,
        ffn_rows / layout.Q1_BLOCK,
        tokens,
        WEIGHT_FMT_BINARY,
        ACT_SCRATCH_SWIGLU,
        0xF15F_FA19,
    );

    const w_beats = ports[0].len / PORT_BEAT_BYTES;
    var wi = [_]usize{0} ** PORTS;
    var held_data: u64 = 0;
    var held_last: c_int = 0;
    var saw_stalled_output = false;
    for (0..CYCLE_LIMIT) |_| {
        var w_valid = [_]bool{false} ** PORTS;
        for (0..PORTS) |port| {
            const valid = wi[port] < w_beats;
            w_valid[port] = valid;
            const words = if (valid)
                readPortBeat(ports[port], wi[port])
            else
                [_]u32{0} ** ROWS_PER_PORT;
            c.dut_set_w(
                dut.h,
                @intCast(port),
                &words,
                @intFromBool(valid),
            );
        }
        c.dut_set_a(dut.h, 0x7fc0_0000_7fc0_0000, 1, 1);
        c.dut_set_m_ready(dut.h, 0);
        c.dut_eval(dut.h);
        if (c.dut_a_ready(dut.h) != 0)
            return error.P3dStalledAbortAcceptedExternalActivation;
        if (c.dut_m_valid(dut.h) != 0) {
            if (c.dut_m_keep(dut.h) != 0xff or
                c.dut_dbg_p3d_lifecycle(dut.h) &
                    DBG_P3D_RESIDUAL_STARTED == 0)
                return error.P3dStalledAbortOutputMismatch;
            held_data = c.dut_m_data(dut.h);
            held_last = c.dut_m_last(dut.h);
            saw_stalled_output = true;
            break;
        }

        var w_fire = [_]bool{false} ** PORTS;
        for (0..PORTS) |port|
            w_fire[port] = w_valid[port] and
                c.dut_w_ready(dut.h, @intCast(port)) != 0;
        dut.step();
        for (0..PORTS) |port| {
            if (w_fire[port]) wi[port] += 1;
        }
    }
    if (!saw_stalled_output) return error.P3dStalledAbortOutputTimeout;

    var zero = [_]u32{0} ** ROWS_PER_PORT;
    for (0..PORTS) |port| c.dut_set_w(dut.h, @intCast(port), &zero, 0);
    c.dut_set_a(dut.h, 0, 0, 0);
    for (0..3) |_| {
        c.dut_eval(dut.h);
        if (c.dut_m_valid(dut.h) == 0 or
            c.dut_m_keep(dut.h) != 0xff or
            c.dut_m_data(dut.h) != held_data or
            c.dut_m_last(dut.h) != held_last)
            return error.P3dStalledOutputWasNotStable;
        dut.step();
    }

    c.dut_set_axi_write(dut.h, REG_SCRATCH_CTRL, SCRATCH_CTRL_ABORT, 1);
    var saw_abort_strobe = false;
    for (0..16) |_| {
        c.dut_eval(dut.h);
        const abort_now = c.dut_dbg_p3d_launch(dut.h) &
            DBG_P3D_ABORT_STROBE != 0;
        if (abort_now) {
            if (c.dut_m_valid(dut.h) != 0 or c.dut_a_ready(dut.h) != 0)
                return error.P3dStalledAbortEscapedSameCycle;
            saw_abort_strobe = true;
            dut.step();
            break;
        }
        if (c.dut_m_valid(dut.h) == 0 or
            c.dut_m_data(dut.h) != held_data or
            c.dut_m_last(dut.h) != held_last)
            return error.P3dStalledOutputChangedBeforeAbort;
        dut.step();
    }
    if (!saw_abort_strobe) return error.P3dStalledAbortStrobeTimeout;

    c.dut_set_axi_idle(dut.h);
    c.dut_set_m_ready(dut.h, 1);
    dut.step();
    try waitP3dTerminal(dut, false, SCRATCH_ERROR_ABORT);
    if (try axiRead(dut, REG_NORM_STATUS) != NORM_GLOBAL_IDLE or
        try axiRead(dut, REG_NORM_ERROR) != 0 or
        try axiRead(dut, REG_RESIDUAL_ERROR) != 0)
        return error.P3dStalledAbortStatusMismatch;
}

fn faultP3dResidual(
    dut: *Dut,
    model_rows: usize,
    ffn_rows: usize,
    tokens: usize,
    ports: [PORTS][]const u8,
) !void {
    configureScratch(dut, SCRATCH_MODE_DDR, SCRATCH_ROLE_X0, ffn_rows, tokens);
    c.dut_set_residual_numeric_error(dut.h, 1);
    configureProjection(
        dut,
        model_rows,
        model_rows,
        ffn_rows / layout.Q1_BLOCK,
        tokens,
        WEIGHT_FMT_BINARY,
        ACT_SCRATCH_SWIGLU,
        0xF15F_FA17,
    );

    const w_beats = ports[0].len / PORT_BEAT_BYTES;
    var wi = [_]usize{0} ** PORTS;
    var entered_cleanup = false;
    for (0..CYCLE_LIMIT) |_| {
        var w_valid = [_]bool{false} ** PORTS;
        for (0..PORTS) |port| {
            const valid = wi[port] < w_beats;
            w_valid[port] = valid;
            const words = if (valid)
                readPortBeat(ports[port], wi[port])
            else
                [_]u32{0} ** ROWS_PER_PORT;
            c.dut_set_w(
                dut.h,
                @intCast(port),
                &words,
                @intFromBool(valid),
            );
        }
        c.dut_set_a(dut.h, 0x7fc0_0000_7fc0_0000, 1, 1);
        c.dut_set_m_ready(dut.h, 1);
        c.dut_eval(dut.h);
        if (c.dut_a_ready(dut.h) != 0 or c.dut_m_valid(dut.h) != 0)
            return error.P3dResidualFaultExposedStream;
        var w_fire = [_]bool{false} ** PORTS;
        for (0..PORTS) |port|
            w_fire[port] = w_valid[port] and
                c.dut_w_ready(dut.h, @intCast(port)) != 0;
        if (c.dut_dbg_p3d_lifecycle(dut.h) & DBG_P3D_CLEANUP != 0) {
            entered_cleanup = true;
            break;
        }
        dut.step();
        for (0..PORTS) |port| {
            if (w_fire[port]) wi[port] += 1;
        }
    }

    c.dut_set_residual_numeric_error(dut.h, 0);
    c.dut_set_a(dut.h, 0, 0, 0);
    var zero = [_]u32{0} ** ROWS_PER_PORT;
    for (0..PORTS) |port| c.dut_set_w(dut.h, @intCast(port), &zero, 0);
    if (!entered_cleanup) return error.P3dResidualFaultDidNotTrigger;
    try waitP3dTerminal(
        dut,
        true,
        SCRATCH_ERROR_SECTION | SCRATCH_ERROR_RESIDUAL,
    );
    const norm_status = try axiRead(dut, REG_NORM_STATUS);
    if (norm_status & (NORM_DONE | RESIDUAL_DONE | RESIDUAL_ERROR) !=
        (NORM_DONE | RESIDUAL_DONE | RESIDUAL_ERROR) or
        norm_status & (NORM_GAMMA_VALID | NORM_ERROR) != 0 or
        try axiRead(dut, REG_RESIDUAL_ERROR) == 0)
        return error.P3dResidualFaultStatusMismatch;
}

fn faultP3dDownActivation(
    dut: *Dut,
    model_rows: usize,
    ffn_rows: usize,
    tokens: usize,
) !void {
    configureScratch(dut, SCRATCH_MODE_DDR, SCRATCH_ROLE_X0, ffn_rows, tokens);
    configureProjection(
        dut,
        model_rows,
        model_rows,
        ffn_rows / layout.Q1_BLOCK,
        tokens,
        WEIGHT_FMT_BINARY,
        ACT_SCRATCH_SWIGLU,
        0xF15F_FA18,
    );
    c.dut_set_down_activation_error(dut.h, 1);

    var entered_cleanup = false;
    var saw_residual_start = false;
    for (0..128) |_| {
        c.dut_set_a(dut.h, 0x7fc0_0000_7fc0_0000, 1, 1);
        c.dut_set_m_ready(dut.h, 1);
        c.dut_eval(dut.h);
        const lifecycle = c.dut_dbg_p3d_lifecycle(dut.h);
        saw_residual_start = saw_residual_start or
            lifecycle & DBG_P3D_RESIDUAL_STARTED != 0;
        if (c.dut_a_ready(dut.h) != 0 or c.dut_m_valid(dut.h) != 0)
            return error.P3dDownActivationFaultExposedStream;
        if (lifecycle & DBG_P3D_CLEANUP != 0) {
            entered_cleanup = true;
            break;
        }
        dut.step();
    }

    c.dut_set_down_activation_error(dut.h, 0);
    c.dut_set_a(dut.h, 0, 0, 0);
    if (!saw_residual_start) return error.P3dDownActivationDidNotStartResidual;
    if (!entered_cleanup) return error.P3dDownActivationFaultDidNotTrigger;
    try waitP3dTerminal(
        dut,
        true,
        SCRATCH_ERROR_SECTION | SCRATCH_ERROR_RESIDUAL,
    );
    const norm_status = try axiRead(dut, REG_NORM_STATUS);
    if (norm_status & (NORM_DONE | RESIDUAL_DONE | RESIDUAL_ERROR) !=
        (NORM_DONE | RESIDUAL_DONE | RESIDUAL_ERROR) or
        norm_status & (NORM_GAMMA_VALID | NORM_ERROR) != 0 or
        try axiRead(dut, REG_NORM_ERROR) != 0 or
        try axiRead(dut, REG_RESIDUAL_ERROR) == 0)
        return error.P3dDownActivationFaultStatusMismatch;
}

fn expectFfnResult(stream: []const u8, expected: []const f32, rows: usize, tokens: usize) !void {
    if (stream.len != rows * tokens * @sizeOf(f32) or expected.len != rows * tokens)
        return error.FfnSectionResultShapeMismatch;
    var offset: usize = 0;
    for (0..rows / ROWS) |rb| {
        for (0..tokens) |token| {
            for (0..ROWS) |lane| {
                const got = std.mem.readInt(u32, stream[offset..][0..4], .little);
                offset += 4;
                const want: u32 = @bitCast(expected[token * rows + rb * ROWS + lane]);
                if (got != want) return error.FfnSectionResultMismatch;
            }
        }
    }
}

fn expectFfnChangedFromValues(
    stream: []const u8,
    baseline: []const f32,
    rows: usize,
    tokens: usize,
) !void {
    if (stream.len != rows * tokens * @sizeOf(f32) or
        baseline.len != rows * tokens)
        return error.FfnSectionResultShapeMismatch;
    var changed = false;
    var offset: usize = 0;
    for (0..rows / ROWS) |rb| {
        for (0..tokens) |token| {
            for (0..ROWS) |lane| {
                const got = std.mem.readInt(u32, stream[offset..][0..4], .little);
                offset += 4;
                const original: u32 =
                    @bitCast(baseline[token * rows + rb * ROWS + lane]);
                changed = changed or got != original;
            }
        }
    }
    if (!changed) return error.P3dNonzeroWeightsWerePassthrough;
}

fn expectFfnChangedFromStream(stream: []const u8, baseline: []const u8) !void {
    if (stream.len != baseline.len) return error.FfnSectionResultShapeMismatch;
    if (std.mem.eql(u8, stream, baseline))
        return error.P3dNonzeroWeightsWerePassthrough;
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
    configureScratch(&dut, SCRATCH_MODE_DDR, SCRATCH_ROLE_X0, ffn_dim, tokens);
    try abortLegacyAtPendingQ8Config(&dut);

    // No reset: the aborted pending pulse must not leak into this accepted
    // section, whose legacy Q8 configuration remains exactly one cycle wide.
    configureScratch(&dut, SCRATCH_MODE_DDR, SCRATCH_ROLE_X0, ffn_dim, tokens);
    try beginLegacySectionChecked(&dut);
    if (try axiRead(&dut, REG_SCRATCH_STATUS) & SCRATCH_SECTION_ACTIVE == 0)
        return error.SectionBeginRejected;
    try runScratchOnlyProjection(&dut, ffn_dim, q1_blocks, tokens, SCRATCH_ROLE_X1, ACT_RAW_LOAD, 0xF15F_0001, port_views[0], &raw_beats);
    const gate_run = try runStreamingGateProjection(&dut, ffn_dim, tokens, 0xF15F_0001, port_views[1]);
    try rejectDuplicateGateAtWaitDown(&dut, ffn_dim, tokens, 0xF15F_0001);
    const down = try runInternalDown(
        a,
        &dut,
        model_dim,
        ffn_dim,
        tokens,
        port_views[2],
        SCRATCH_ERROR_CONFIG,
    );
    defer a.free(down.stream);
    try expectFfnResult(down.stream, &expected, model_dim, tokens);
    const expected_records = tokens * ffn_dim / shared_layout.q8_block;
    if (gate_run.scratch_reads != ffn_dim / 8 * tokens or
        gate_run.swiglu_fires != ffn_dim * tokens or
        gate_run.record_fires != expected_records or
        gate_run.capture_fires != expected_records * 5 or
        down.replay_fires != expected_records * 5)
        return error.FfnSectionTraversalMismatch;
    std.debug.print(
        "decode_top v16 FFN section 128x128x2: UP A={d}, GATE reads/y/records/capture={d}/{d}/{d}/{d}, WAIT_DOWN duplicate rejected, replay={d}, DOWN cycles={d}, bit-exact\n",
        .{ raw_beats.len, gate_run.scratch_reads, gate_run.swiglu_fires, gate_run.record_fires, gate_run.capture_fires, down.replay_fires, down.cycles },
    );

    // Exercise the complete retained-owner lifecycle on the same DUT. First
    // establish a fresh section and real UP contents, then abort GATE with a
    // pairer scratch response outstanding. No reset occurs before the exact
    // section below, so stale ownership or buffered records corrupt the result.
    configureScratch(&dut, SCRATCH_MODE_DDR, SCRATCH_ROLE_X0, ffn_dim, tokens);
    try beginLegacySectionChecked(&dut);
    if (try axiRead(&dut, REG_SCRATCH_STATUS) & SCRATCH_SECTION_ACTIVE == 0)
        return error.AbortSectionBeginRejected;
    try runScratchOnlyProjection(&dut, ffn_dim, q1_blocks, tokens, SCRATCH_ROLE_X1, ACT_RAW_LOAD, 0xF15F_A807, port_views[0], &raw_beats);
    try abortStreamingGateWithOutstandingRead(&dut, ffn_dim, tokens, 0xF15F_A807, port_views[1]);

    configureScratch(&dut, SCRATCH_MODE_DDR, SCRATCH_ROLE_X0, ffn_dim, tokens);
    try beginLegacySectionChecked(&dut);
    const restart_status = try axiRead(&dut, REG_SCRATCH_STATUS);
    if (restart_status & SCRATCH_SECTION_ACTIVE == 0 or
        restart_status & (SCRATCH_SECTION_DONE | SCRATCH_ANY_ERROR |
            SCRATCH_CONSUMER_BUSY | SCRATCH_CONSUMER_DONE |
            SCRATCH_X0_VALID | SCRATCH_X1_VALID) != 0 or
        try axiRead(&dut, REG_SCRATCH_ERROR) != 0)
        return error.FfnRestartSectionBeginFailed;
    try runScratchOnlyProjection(&dut, ffn_dim, q1_blocks, tokens, SCRATCH_ROLE_X1, ACT_RAW_LOAD, 0xF15F_A808, port_views[0], &raw_beats);
    const restarted_gate = try runStreamingGateProjection(&dut, ffn_dim, tokens, 0xF15F_A808, port_views[1]);
    const restarted_down = try runInternalDown(
        a,
        &dut,
        model_dim,
        ffn_dim,
        tokens,
        port_views[2],
        0,
    );
    defer a.free(restarted_down.stream);
    try expectFfnResult(restarted_down.stream, &expected, model_dim, tokens);
    if (restarted_gate.scratch_reads != ffn_dim / 8 * tokens or
        restarted_gate.swiglu_fires != ffn_dim * tokens or
        restarted_gate.record_fires != expected_records or
        restarted_gate.capture_fires != expected_records * 5 or
        restarted_down.replay_fires != expected_records * 5)
        return error.FfnRestartTraversalMismatch;
    std.debug.print(
        "decode_top v16 GATE Q8 fault: error bits5/6 set, outstanding owner/orphan drained; no-reset restart completed bit-exact\n",
        .{},
    );

    // FINISH is exercised first so the following complete UP/GATE setup is a
    // no-reset restart from the exact natural-done/registered-abort boundary.
    for ([_]u32{ KERNEL_ST_FINISH, KERNEL_ST_EMIT }, 0..) |target, index| {
        configureScratch(&dut, SCRATCH_MODE_DDR, SCRATCH_ROLE_X0, ffn_dim, tokens);
        try beginLegacySectionChecked(&dut);
        try runScratchOnlyProjection(
            &dut,
            ffn_dim,
            q1_blocks,
            tokens,
            SCRATCH_ROLE_X1,
            ACT_RAW_LOAD,
            0xF15F_A900 + @as(u32, @intCast(index)),
            port_views[0],
            &raw_beats,
        );
        _ = try runStreamingGateProjection(
            &dut,
            ffn_dim,
            tokens,
            0xF15F_A900 + @as(u32, @intCast(index)),
            port_views[1],
        );
        try abortInternalDownAtKernelState(
            &dut,
            model_dim,
            ffn_dim,
            tokens,
            port_views[2],
            target,
        );
    }
    std.debug.print(
        "decode_top r7 legacy DOWN: ST_FINISH clean publication blocked; " ++
            "no-reset restart and stalled EMIT abort quarantined\n",
        .{},
    );
}

fn runP3dSectionCase(a: std.mem.Allocator) !void {
    const model_rows: usize = 128;
    const ffn_rows: usize = 128;
    const tokens: usize = 1;
    const q1_blocks: usize = 1;

    var weight_bits: [3][model_rows]u128 = undefined;
    var weight_scales: [3][model_rows]f16 = undefined;
    const nonzero_scale: f16 = @floatCast(@as(f32, 1.0 / 128.0));
    for (&weight_bits) |*projection| {
        for (projection) |*bits| bits.* = ~@as(u128, 0);
    }
    for (&weight_scales) |*projection| {
        for (projection) |*scale| scale.* = nonzero_scale;
    }
    var port_storage: [3][PORTS][]u8 = undefined;
    var port_views: [3][PORTS][]const u8 = undefined;
    for (0..3) |projection| {
        for (0..PORTS) |port| {
            port_storage[projection][port] = try a.alloc(
                u8,
                weightPortBytes(model_rows / ROWS, q1_blocks, WEIGHT_FMT_BINARY),
            );
            port_views[projection][port] = port_storage[projection][port];
        }
        packWeightPorts(
            model_rows,
            q1_blocks,
            &weight_bits[projection],
            &weight_scales[projection],
            port_storage[projection],
        );
    }
    defer {
        for (&port_storage) |*projection| {
            for (projection) |storage| a.free(storage);
        }
    }

    var gamma = [_]f32{1.0} ** model_rows;
    var gamma_beats: [model_rows / 2]u64 = undefined;
    packRawF32(&gamma, &gamma_beats);
    var residual: [tokens * model_rows]f32 = undefined;
    for (&residual, 0..) |*value, index| {
        const centered: i32 = @as(i32, @intCast(index % 31)) - 15;
        value.* = @as(f32, @floatFromInt(centered)) * 0.03125;
    }
    var residual_beats: [residual.len / 2]u64 = undefined;
    packRawF32(&residual, &residual_beats);

    var dut = Dut.init();
    defer dut.deinit();
    reset(&dut);
    try loadP3dGamma(&dut, model_rows, &gamma_beats);
    try abortGammaReplacementAtPending(&dut, model_rows);
    // No reset: the cancelled replacement must leave no stale pending pulse or
    // public VALID, and a complete reload must own S_AXIS normally.
    try loadP3dGamma(&dut, model_rows, &gamma_beats);

    // Acceptance snapshots the section, but an abort in the following launch
    // cycle must prevent every RMS/Q8 leaf handshake and still restart cleanly.
    try abortP3dImmediatelyAfterBegin(
        &dut,
        model_rows,
        ffn_rows,
        tokens,
    );
    try loadP3dGamma(&dut, model_rows, &gamma_beats);

    // Leave the shared Q8 ingress in its real legacy/raw framing-error state.
    // The delayed P3d launch must clear it atomically when claiming RMS
    // ownership, without a device reset or another Q8 launch in between.
    const stale_q8 = try runProjection(
        a,
        &dut,
        model_rows,
        model_rows,
        q1_blocks,
        tokens,
        WEIGHT_FMT_BINARY,
        ACT_RAW_LOAD,
        0xF15F_D000,
        port_views[0],
        &residual_beats,
        1,
        .early_last,
        false,
        true,
    );
    defer a.free(stale_q8.result);
    if (stale_q8.quant_status != QUANT_FRAME)
        return error.P3dStaleQ8SetupDidNotFault;
    var zero = [_]u32{0} ** ROWS_PER_PORT;
    for (0..PORTS) |port| c.dut_set_w(dut.h, @intCast(port), &zero, 0);
    if (try axiRead(&dut, REG_QUANT_STATUS) != QUANT_FRAME)
        return error.P3dStaleQ8DidNotPersist;

    try beginP3dSection(&dut, model_rows, ffn_rows, tokens, false);
    const stale_restart = c.dut_dbg_p3d_lifecycle(dut.h);
    if (stale_restart & (DBG_P3D_ACTIVE | DBG_P3D_Q8_OWNER_MASK) !=
        (DBG_P3D_ACTIVE | DBG_P3D_Q8_OWNER_RMS) or
        stale_restart & DBG_P3D_CLEANUP != 0 or
        try axiRead(&dut, REG_QUANT_STATUS) != 0)
        return error.P3dStaleQ8WasAttributedToNewSection;
    c.dut_arm_ownerless_response(
        dut.h,
        OWNERLESS_ROUTE_P3D_RMS,
        1,
    );
    try sendP3dResidual(&dut, &residual_beats);
    const rms_q8_records: u32 =
        @intCast(tokens * ffn_rows / shared_layout.q8_block);
    dut.armRmsQ8Monitor(rms_q8_records);
    try runScratchOnlyProjection(
        &dut,
        ffn_rows,
        q1_blocks,
        tokens,
        SCRATCH_ROLE_X1,
        ACT_RAW_LOAD,
        0xF15F_D001,
        port_views[0],
        &.{},
    );
    try verifyP3dRmsQ8Accounting(&dut, rms_q8_records);
    try expectOwnerlessResponse(&dut, OWNERLESS_ROUTE_P3D_RMS, true);
    c.dut_arm_ownerless_response(dut.h, OWNERLESS_ROUTE_PAIRER, 0);
    const gate = try runStreamingGateProjection(
        &dut,
        ffn_rows,
        tokens,
        0xF15F_D001,
        port_views[1],
    );
    try expectOwnerlessResponse(&dut, OWNERLESS_ROUTE_PAIRER, false);
    const external = try runP3dDown(
        a,
        &dut,
        model_rows,
        ffn_rows,
        tokens,
        port_views[2],
    );
    defer a.free(external.stream);
    try expectFfnChangedFromValues(
        external.stream,
        &residual,
        model_rows,
        tokens,
    );

    // The clean section reseals the updated residual, so the next begin can
    // validate and consume R without opening S_AXIS_ACTS.
    try beginP3dSection(&dut, model_rows, ffn_rows, tokens, true);
    c.dut_set_a(dut.h, 0x7fc0_0000_7fc0_0000, 1, 1);
    for (0..32) |_| {
        c.dut_eval(dut.h);
        if (c.dut_a_ready(dut.h) != 0)
            return error.P3dResidentAcceptedResidual;
        dut.step();
    }
    c.dut_set_a(dut.h, 0, 0, 0);
    try runScratchOnlyProjection(
        &dut,
        ffn_rows,
        q1_blocks,
        tokens,
        SCRATCH_ROLE_X1,
        ACT_RAW_LOAD,
        0xF15F_D002,
        port_views[0],
        &.{},
    );
    const resident_gate = try runStreamingGateProjection(
        &dut,
        ffn_rows,
        tokens,
        0xF15F_D002,
        port_views[1],
    );
    const resident = try runP3dDown(
        a,
        &dut,
        model_rows,
        ffn_rows,
        tokens,
        port_views[2],
    );
    defer a.free(resident.stream);
    try expectFfnChangedFromStream(resident.stream, external.stream);

    const expected_records = tokens * ffn_rows / shared_layout.q8_block;
    if (gate.record_fires != expected_records or
        resident_gate.record_fires != expected_records or
        external.replay_fires != expected_records * 5 or
        resident.replay_fires != expected_records * 5)
        return error.P3dTraversalMismatch;

    try verifyGammaOverlapRejection(&dut, model_rows, &gamma_beats);

    // Keep the later fault/restart checks numerically simple. The two clean
    // sections above already proved all three nonzero projections contribute.
    for (&weight_bits) |*projection| {
        for (projection) |*bits| bits.* = 0;
    }
    for (&weight_scales) |*projection| {
        for (projection) |*scale| scale.* = 0;
    }
    for (0..3) |projection| {
        packWeightPorts(
            model_rows,
            q1_blocks,
            &weight_bits[projection],
            &weight_scales[projection],
            port_storage[projection],
        );
    }

    // Wrong residual-child activity before the DOWN phase is diagnosed as a
    // norm/controller fault. The same-edge public boundary is quarantined and
    // the accepted gamma reload proves cleanup is immediately restart-safe.
    try beginP3dSection(&dut, model_rows, ffn_rows, tokens, false);
    try faultP3dInactiveNormOwner(&dut);
    try loadP3dGamma(&dut, model_rows, &gamma_beats);

    // Reach the registered DOWN launch boundary, then collide it with stale
    // RMS-child activity. The internal launch is tentative, while all public
    // effects are blocked and the next edge diagnoses a residual fault.
    const inactive_launch_epoch: u32 = 0xF15F_FA21;
    try beginP3dSection(&dut, model_rows, ffn_rows, tokens, false);
    try sendP3dResidual(&dut, &residual_beats);
    try runScratchOnlyProjection(
        &dut,
        ffn_rows,
        q1_blocks,
        tokens,
        SCRATCH_ROLE_X1,
        ACT_RAW_LOAD,
        inactive_launch_epoch,
        port_views[0],
        &.{},
    );
    _ = try runStreamingGateProjection(
        &dut,
        ffn_rows,
        tokens,
        inactive_launch_epoch,
        port_views[1],
    );
    try faultP3dInactiveResidualAtLaunch(
        &dut,
        model_rows,
        ffn_rows,
        tokens,
        inactive_launch_epoch,
    );
    try loadP3dGamma(&dut, model_rows, &gamma_beats);

    // Repeat through a live final residual beat. Wrong RMS-child activity may
    // let an internal doomed beat retire, but must suppress the external final
    // transfer and finish with residual attribution before the next restart.
    const inactive_final_epoch: u32 = 0xF15F_FA22;
    try beginP3dSection(&dut, model_rows, ffn_rows, tokens, false);
    try sendP3dResidual(&dut, &residual_beats);
    try runScratchOnlyProjection(
        &dut,
        ffn_rows,
        q1_blocks,
        tokens,
        SCRATCH_ROLE_X1,
        ACT_RAW_LOAD,
        inactive_final_epoch,
        port_views[0],
        &.{},
    );
    _ = try runStreamingGateProjection(
        &dut,
        ffn_rows,
        tokens,
        inactive_final_epoch,
        port_views[1],
    );
    try faultP3dInactiveResidualAtFinalOutput(
        &dut,
        model_rows,
        ffn_rows,
        tokens,
        port_views[2],
    );
    try loadP3dGamma(&dut, model_rows, &gamma_beats);

    // Gamma framing is fail-closed, and an invalid table cannot begin a P3d
    // section or open any stream.
    try rejectEarlyP3dGammaFrame(&dut, model_rows, gamma_beats[0]);
    axiWrite(&dut, REG_NORM_EPS, 0x3586_37bd);
    configureScratch(&dut, SCRATCH_MODE_DDR, SCRATCH_ROLE_X0, ffn_rows, tokens);
    axiWrite(&dut, REG_SCRATCH_CTRL, SCRATCH_CTRL_SECTION_BEGIN);
    var scratch_status = try axiRead(&dut, REG_SCRATCH_STATUS);
    if (scratch_status & (SCRATCH_SECTION_DONE | SCRATCH_ANY_ERROR) !=
        (SCRATCH_SECTION_DONE | SCRATCH_ANY_ERROR) or
        scratch_status & SCRATCH_SECTION_ACTIVE != 0 or
        try axiRead(&dut, REG_SCRATCH_ERROR) & SCRATCH_ERROR_CONFIG == 0 or
        try axiRead(&dut, REG_NORM_STATUS) & (NORM_DONE | NORM_ERROR) !=
            (NORM_DONE | NORM_ERROR) or
        c.dut_a_ready(dut.h) != 0)
        return error.P3dInvalidGammaBeginDidNotFailClosed;
    try loadP3dGamma(&dut, model_rows, &gamma_beats);

    // The section's external R frame is exact: an early TLAST faults RMS,
    // invalidates both the tentative R seal and gamma, and drains to idle.
    try beginP3dSection(&dut, model_rows, ffn_rows, tokens, false);
    try rejectEarlyP3dResidualFrame(&dut, residual_beats[0]);
    try loadP3dGamma(&dut, model_rows, &gamma_beats);

    // Exercise both cancellation windows around the RMS accounting register:
    // before it captures the raw completion and after Q is visible but before
    // the qualified event can update count/done/seal on the following edge.
    try beginP3dSection(&dut, model_rows, ffn_rows, tokens, false);
    try abortP3dAtRmsRecord(
        &dut,
        &residual_beats,
        ffn_rows,
        q1_blocks,
        tokens,
        0xF15F_C001,
        port_views[0],
        false,
    );
    try loadP3dGamma(&dut, model_rows, &gamma_beats);
    try beginP3dSection(&dut, model_rows, ffn_rows, tokens, false);
    try abortP3dAtRmsRecord(
        &dut,
        &residual_beats,
        ffn_rows,
        q1_blocks,
        tokens,
        0xF15F_C002,
        port_views[0],
        true,
    );
    try loadP3dGamma(&dut, model_rows, &gamma_beats);

    // Both destinations of the sole Q8 ingress retain the terminal diagnostic
    // across automatic P3d kill/cleanup, then clear it on the next gamma load.
    try beginP3dSection(&dut, model_rows, ffn_rows, tokens, false);
    const rms_restart = c.dut_dbg_p3d_q8_accounting(dut.h);
    if (rms_restart & (DBG_RMS_Q8_ACCEPT | DBG_RMS_Q8_FIRE |
        DBG_RMS_Q8_DONE | DBG_RMS_Q8_SEALED) != 0 or
        dbgRmsQ8Count(rms_restart) != 0)
        return error.P3dRmsQ8RestartRetainedStaleEvent;
    try faultP3dRmsQ8(&dut, &residual_beats);
    try loadP3dGamma(&dut, model_rows, &gamma_beats);

    // Inject a real child response error at the shared scratch boundary.
    try beginP3dSection(&dut, model_rows, ffn_rows, tokens, false);
    c.dut_set_p3d_scratch_error(dut.h, 1);
    try sendP3dResidual(&dut, &residual_beats);
    var rms_fault = false;
    for (0..16384) |_| {
        c.dut_eval(dut.h);
        if (c.dut_m_valid(dut.h) != 0)
            return error.P3dRmsScratchFaultExposedOutput;
        if (c.dut_dbg_p3d_lifecycle(dut.h) & DBG_P3D_CLEANUP != 0) {
            rms_fault = true;
            break;
        }
        dut.step();
    }
    c.dut_set_p3d_scratch_error(dut.h, 0);
    if (!rms_fault) return error.P3dRmsScratchFaultDidNotTrigger;
    var cut_control = c.dut_dbg_control_boundaries(dut.h);
    if (c.dut_dbg_p3d_lifecycle(dut.h) & DBG_P3D_KILL == 0 or
        cut_control & DBG_CTRL_SCRATCH_R_READY == 0 or
        cut_control & (DBG_CTRL_RMS_KILL_Q | DBG_CTRL_RMS_ABORT |
            DBG_CTRL_SCRATCH_WRITER_ABORT |
            DBG_CTRL_SCRATCH_R_ABORT) != 0)
        return error.P3dRmsKillWasNotDelayedAtFaultBoundary;
    dut.step();
    cut_control = c.dut_dbg_control_boundaries(dut.h);
    if (c.dut_dbg_p3d_lifecycle(dut.h) & DBG_P3D_KILL != 0 or
        cut_control & (DBG_CTRL_RMS_KILL_Q | DBG_CTRL_RMS_ABORT |
            DBG_CTRL_SCRATCH_WRITER_ABORT | DBG_CTRL_SCRATCH_R_ABORT |
            DBG_CTRL_RMS_ABORT_DRAIN_READY) !=
            (DBG_CTRL_RMS_KILL_Q | DBG_CTRL_RMS_ABORT |
                DBG_CTRL_SCRATCH_WRITER_ABORT | DBG_CTRL_SCRATCH_R_ABORT |
                DBG_CTRL_RMS_ABORT_DRAIN_READY) or
        cut_control & (DBG_CTRL_SCRATCH_R_READY |
            DBG_CTRL_SCRATCH_R_WRITE) != 0)
        return error.P3dRmsKillDelayedBoundaryMismatch;
    try waitP3dTerminal(
        &dut,
        true,
        SCRATCH_ERROR_SECTION | SCRATCH_ERROR_RMS,
    );
    if (try axiRead(&dut, REG_NORM_STATUS) & (NORM_DONE | NORM_ERROR) !=
        (NORM_DONE | NORM_ERROR) or
        try axiRead(&dut, REG_NORM_ERROR) == 0)
        return error.P3dRmsScratchFaultStatusMismatch;
    try loadP3dGamma(&dut, model_rows, &gamma_beats);

    // Fault the reused quantizer after a complete RMS+UP milestone.
    try beginP3dSection(&dut, model_rows, ffn_rows, tokens, false);
    try sendP3dResidual(&dut, &residual_beats);
    try runScratchOnlyProjection(
        &dut,
        ffn_rows,
        q1_blocks,
        tokens,
        SCRATCH_ROLE_X1,
        ACT_RAW_LOAD,
        0xF15F_FA01,
        port_views[0],
        &.{},
    );
    try faultP3dGateQ8(&dut, ffn_rows, tokens, 0xF15F_FA01);
    try loadP3dGamma(&dut, model_rows, &gamma_beats);

    // Reach DOWN cleanly, then make the first residual input non-finite. The
    // raw DOWN beat and the leaf fault cycle are both suppressed from M_AXIS.
    try beginP3dSection(&dut, model_rows, ffn_rows, tokens, false);
    try sendP3dResidual(&dut, &residual_beats);
    try runScratchOnlyProjection(
        &dut,
        ffn_rows,
        q1_blocks,
        tokens,
        SCRATCH_ROLE_X1,
        ACT_RAW_LOAD,
        0xF15F_FA02,
        port_views[0],
        &.{},
    );
    _ = try runStreamingGateProjection(
        &dut,
        ffn_rows,
        tokens,
        0xF15F_FA02,
        port_views[1],
    );
    try faultP3dResidual(
        &dut,
        model_rows,
        ffn_rows,
        tokens,
        port_views[2],
    );
    try loadP3dGamma(&dut, model_rows, &gamma_beats);

    // Fault DOWN activation before the residual leaf can diagnose arithmetic.
    // The controller must still publish a terminal residual error, and raw DOWN
    // data must remain quarantined from M_AXIS.
    try beginP3dSection(&dut, model_rows, ffn_rows, tokens, false);
    try sendP3dResidual(&dut, &residual_beats);
    try runScratchOnlyProjection(
        &dut,
        ffn_rows,
        q1_blocks,
        tokens,
        SCRATCH_ROLE_X1,
        ACT_RAW_LOAD,
        0xF15F_FA04,
        port_views[0],
        &.{},
    );
    _ = try runStreamingGateProjection(
        &dut,
        ffn_rows,
        tokens,
        0xF15F_FA04,
        port_views[1],
    );
    try faultP3dDownActivation(&dut, model_rows, ffn_rows, tokens);
    try loadP3dGamma(&dut, model_rows, &gamma_beats);

    // Hold a committed residual beat at the external boundary, then abort via
    // AXI. The beat must stay stable before the strobe and disappear in the
    // strobe cycle without a transfer.
    try beginP3dSection(&dut, model_rows, ffn_rows, tokens, false);
    try sendP3dResidual(&dut, &residual_beats);
    try runScratchOnlyProjection(
        &dut,
        ffn_rows,
        q1_blocks,
        tokens,
        SCRATCH_ROLE_X1,
        ACT_RAW_LOAD,
        0xF15F_FA05,
        port_views[0],
        &.{},
    );
    _ = try runStreamingGateProjection(
        &dut,
        ffn_rows,
        tokens,
        0xF15F_FA05,
        port_views[1],
    );
    try abortP3dAtStalledResidualOutput(
        &dut,
        model_rows,
        ffn_rows,
        tokens,
        port_views[2],
    );
    try loadP3dGamma(&dut, model_rows, &gamma_beats);

    // Hold a real RMS response at the untagged scratch boundary while ABORT is
    // accepted. Outer and P3d subownership must survive until response drain.
    try beginP3dSection(&dut, model_rows, ffn_rows, tokens, false);
    c.dut_set_p3d_read_response_hold(dut.h, 1);
    try sendP3dResidual(&dut, &residual_beats);
    try abortP3dWithRetainedRmsResponse(&dut);
    try loadP3dGamma(&dut, model_rows, &gamma_beats);

    // No reset after any fault or the retained-response abort. A final exact
    // external run proves all leaves, owners, seals, and Q8 state restart cleanly.
    try beginP3dSection(&dut, model_rows, ffn_rows, tokens, false);
    try sendP3dResidual(&dut, &residual_beats);
    try runScratchOnlyProjection(
        &dut,
        ffn_rows,
        q1_blocks,
        tokens,
        SCRATCH_ROLE_X1,
        ACT_RAW_LOAD,
        0xF15F_FA03,
        port_views[0],
        &.{},
    );
    _ = try runStreamingGateProjection(
        &dut,
        ffn_rows,
        tokens,
        0xF15F_FA03,
        port_views[1],
    );
    const restarted = try runP3dDown(
        a,
        &dut,
        model_rows,
        ffn_rows,
        tokens,
        port_views[2],
    );
    defer a.free(restarted.stream);
    try expectFfnResult(restarted.stream, &residual, model_rows, tokens);
    scratch_status = try axiRead(&dut, REG_SCRATCH_STATUS);
    if (scratch_status & (SCRATCH_SECTION_DONE | SCRATCH_R_VALID) !=
        (SCRATCH_SECTION_DONE | SCRATCH_R_VALID))
        return error.P3dNoResetRestartDidNotResealR;
    std.debug.print(
        "decode_top v17 P3d 128x128x1: delayed-launch and stalled-output aborts; nonzero external/resident; gamma exclusion; RMS/GATE Q8 and section faults; no-reset restart exact\n",
        .{},
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
        if (try axiRead(&dut, 0x04) != 17) return error.ScratchVersionMismatch;

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

        c.dut_arm_ownerless_response(dut.h, OWNERLESS_ROUTE_DRAIN, 1);
        try drainScratch(a, &dut, SCRATCH_ROLE_X1, rows, cols, tee_up.result);
        try expectOwnerlessResponse(&dut, OWNERLESS_ROUTE_DRAIN, true);
        c.dut_arm_ownerless_response(dut.h, OWNERLESS_ROUTE_DRAIN, 0);
        try drainScratch(a, &dut, SCRATCH_ROLE_X0, rows, cols, tee_gate.result);
        try expectOwnerlessResponse(&dut, OWNERLESS_ROUTE_DRAIN, false);

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

    try verifyAbortDominatesCoencodedStarts();
    try verifyInternalModeRequiresSection();
    try verifyKernelAbortBoundaries();
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
    try runP3dSectionCase(a);
    std.debug.print("all decode_top cosim cases passed (decode_top === windowedFixedOutput, bit-exact)\n", .{});
}
