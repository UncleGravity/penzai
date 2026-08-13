//! PL Q1A8 matmul backend: drives the fabric kernel for a wire matmul command.
//!
//! Weights are streamed straight from their resident XRT BO range. Primitive
//! matmuls use the canonical host quantizer and a packed-Q8 staging region. The
//! atomic group path instead streams FP32 once through the exact PL quantizer,
//! then reuses the resident native-Q8 activation for its second projection.
//! Results DMA into a staging region and copy into the destination range.
//!
//! The v8 kernel uses four contiguous weight-port streams: port N stores rows
//! `N*4..N*4+3` of each 16-row block. The RTL zips those
//! streams into one 512-bit beat.

const std = @import("std");
const shared = @import("shared");
const regmap = @import("regmap");
const regwin = @import("regwin.zig");
const dma_mod = @import("dma.zig");
const gather = @import("gather.zig");
const seq = @import("seq.zig");
const seq_ctrl_mod = @import("seq_ctrl.zig");
const profile = @import("../profile.zig");

const wire = shared.wire;
const layout = shared.layout;
const section = shared.section;
const capabilities = shared.capabilities;

// The AXI-Lite base addresses, staging reservations, and COLS_MAX all come from
// the generated bitstream manifest (fpga/regmap/matmul.zig) — the same source
// build.tcl assigns addresses from and the RTL takes COLS_MAX from — so the host
// can never drift from the loaded gateware. Local aliases keep the call sites
// below unchanged. dma_w[0] also owns result S2MM; all weight ports use MM2S.
const dma_w_bases = regmap.addr.dma_w;
const dma_a_base: i64 = regmap.addr.dma_a; // acts MM2S
const kernel_base: i64 = regmap.addr.kernel; // kernel AXI-Lite

// Staging capacities, reserved once from the heap at init. Generous vs any real
// shape; a matmul that would exceed them falls back to PS.
const acts_staging_cap: usize = regmap.caps.acts_staging_bytes; // COLS_MAX columns of acts
const result_staging_cap: usize = regmap.caps.result_staging_bytes; // num_rb * COLS_MAX * result_bytes_per_rb

// Columns multiplied per kernel run. Matches gemm_kernel COLS_MAX in the loaded
// bitstream (both derive from caps.cols_max). Prefill (cols>1) is tiled into
// groups of this many.
const mc_cols_max: usize = regmap.caps.cols_max;
const result_bytes_per_rb = gather.result_bytes_per_rb; // single source in gather.zig

// Errors this op can raise: the DMA substrate's, this kernel's, plus heap failures.
// SeqDispatch = the seq.v arm refused or failed a run (details already printed); never
// silently retried over MMIO — a failed replay means the stream or executor needs looking at.
const KernelError = regwin.Error || error{ KernelTimeout, ScratchTimeout, BadId, BadVersion, BadRows, BadWeightPorts, BadClock, ActivationState, ActivationQuantization, ScratchState };
pub const Error = dma_mod.Error || KernelError || error{ HeapFailure, OutOfMemory, SeqDispatch, ScratchMismatch };

// ---- The matmul kernel AXI-Lite driver (a tenant on the PL substrate) -------
// Moved out of the former mmio.zig: the run signature, identity/shape checks, and
// counter accessors are matmul-specific. Register offsets come from the generated
// `regmap` module — no hand-duplicated constants.

const CTRL_START: u32 = 1 << 0;
const STATUS_DONE: u32 = 1 << 1;
const ACT_STATE_VALID: u32 = 1 << 0;
const ACT_STATE_ERROR: u32 = 1 << 1;
const SCRATCH_CTRL_DRAIN_START: u32 = 1 << 0;
const SCRATCH_CTRL_ABORT: u32 = 1 << 1;
const SCRATCH_WRITER_BUSY: u32 = 1 << 0;
const SCRATCH_WRITER_DONE: u32 = 1 << 1;
const SCRATCH_DRAIN_BUSY: u32 = 1 << 2;
const SCRATCH_DRAIN_DONE: u32 = 1 << 3;
const SCRATCH_ANY_ERROR: u32 = 1 << 4;

const ActivationMode = enum(u32) {
    packed_load = 0,
    reuse = 1,
    raw_f32_load = 2,
};

const ScratchMode = enum(u32) {
    ddr_only = 0,
    tee = 1,
    drain = 2,
};

/// Minimum kernel VERSION the driver accepts. v9 is the fixed-window gemm kernel (104-bit
/// accumulator, EMIN baked in as a format constant — no EMIN register). The driver no longer
/// writes a window floor, so an older v8 kernel (whose EMIN register resets to 0 → garbage
/// logits) is incompatible and must fall back to PS. Policy (the oldest compatible kernel),
/// distinct from the current build's VERSION reset — so it is not sourced from the regmap.
pub const min_version: u32 = 9;
pub const version_with_counters: u32 = 5;
/// First build carrying seq_top on sc_ctrl (0xA0200000). Gates the PENZAI_SEQ arm: poking the
/// seq window on an older bitstream is an unmapped sc_ctrl access (a bus fault), so the driver
/// refuses rather than trusts the env var.
pub const version_with_seq: u32 = 10;
pub const version_with_ternary: u32 = 11;
pub const version_with_logical_rows: u32 = 12;
pub const version_with_activation_reuse: u32 = 13;
pub const version_with_scratch: u32 = 14;
/// Identity/shape the driver requires of the loaded kernel, sourced from the regmap
/// reset column (the gateware's self-described values) so the runtime check and the
/// bitstream can never disagree.
pub const expected_id: u32 = regmap.resetOf("ID");
pub const expected_rows: u32 = regmap.resetOf("ROWS");
pub const expected_weight_ports: u32 = regmap.resetOf("WEIGHT_PORTS");

comptime {
    // The minimum we accept must not exceed what this build self-describes.
    std.debug.assert(min_version <= regmap.resetOf("VERSION"));
}

pub const Kernel = struct {
    win: regwin.RegWindow,
    version: u32,
    clk_hz: u32,

    pub fn open(base: i64) KernelError!Kernel {
        var win = try regwin.RegWindow.mapWindow(base);
        errdefer win.deinit();
        const id = win.rd(regmap.offsetOf("ID"));
        if (id != expected_id) return error.BadId;
        const version = win.rd(regmap.offsetOf("VERSION"));
        if (version < min_version) return error.BadVersion;
        if (win.rd(regmap.offsetOf("ROWS")) != expected_rows) return error.BadRows;
        if (win.rd(regmap.offsetOf("WEIGHT_PORTS")) != expected_weight_ports) return error.BadWeightPorts;
        const clk_hz = win.rd(regmap.offsetOf("CLK_HZ"));
        if (clk_hz == 0) return error.BadClock;
        return .{ .win = win, .version = version, .clk_hz = clk_hz };
    }

    pub fn deinit(self: *Kernel) void {
        self.win.deinit();
    }

    pub fn hasCounters(self: Kernel) bool {
        return self.version >= version_with_counters;
    }

    pub fn hasActivationReuse(self: Kernel) bool {
        return self.version >= version_with_activation_reuse;
    }

    pub fn hasScratch(self: Kernel) bool {
        return self.version >= version_with_scratch;
    }

    /// Fabric clock in MHz, self-described by the bitstream (CLK_HZ register).
    pub fn clkMhz(self: Kernel) f64 {
        return @as(f64, @floatFromInt(self.clk_hz)) / 1_000_000.0;
    }

    /// Set dims then strobe start. done_latched clears on the strobe.
    pub fn hasLogicalRows(self: Kernel) bool {
        return self.version >= version_with_logical_rows;
    }

    pub fn run(
        self: *Kernel,
        num_q1_blocks: u32,
        num_rowblocks: u32,
        num_cols: u32,
        logical_rows: ?u32,
        weight_fmt: wire.WeightFormat,
    ) void {
        self.runWithActivation(
            num_q1_blocks,
            num_rowblocks,
            num_cols,
            logical_rows,
            weight_fmt,
            .packed_load,
            0,
        );
    }

    pub fn runWithActivation(
        self: *Kernel,
        num_q1_blocks: u32,
        num_rowblocks: u32,
        num_cols: u32,
        logical_rows: ?u32,
        weight_fmt: wire.WeightFormat,
        activation_mode: ActivationMode,
        activation_epoch: u32,
    ) void {
        self.win.wr(regmap.offsetOf("NUM_Q1_BLOCKS"), num_q1_blocks);
        self.win.wr(regmap.offsetOf("NUM_ROWBLOCKS"), num_rowblocks);
        self.win.wr(regmap.offsetOf("NUM_COLS"), num_cols);
        if (logical_rows) |rows| self.win.wr(regmap.offsetOf("NUM_ROWS"), rows);
        self.win.wr(regmap.offsetOf("WEIGHT_FMT"), @intFromEnum(weight_fmt));
        if (self.hasActivationReuse()) {
            self.win.wr(regmap.offsetOf("ACT_MODE"), @intFromEnum(activation_mode));
            self.win.wr(regmap.offsetOf("ACT_EPOCH"), activation_epoch);
        } else {
            std.debug.assert(activation_mode == .packed_load);
        }
        self.win.wr(regmap.offsetOf("CTRL"), CTRL_START);
    }

    pub fn waitDone(self: *Kernel) KernelError!void {
        var i: usize = 0;
        while (i < regwin.wait_limit) : (i += 1) {
            if (self.win.rd(regmap.offsetOf("STATUS")) & STATUS_DONE != 0) {
                if (self.hasActivationReuse() and
                    self.win.rd(regmap.offsetOf("ACT_STATE")) & ACT_STATE_ERROR != 0)
                {
                    if (self.quantStatus() != 0) return error.ActivationQuantization;
                    return error.ActivationState;
                }
                return;
            }
        }
        return error.KernelTimeout;
    }

    pub fn residentMatches(self: Kernel, epoch: u32, q1_blocks: u32, cols: u32) bool {
        if (!self.hasActivationReuse()) return false;
        if (self.win.rd(regmap.offsetOf("ACT_STATE")) & ACT_STATE_VALID == 0) return false;
        return self.win.rd(regmap.offsetOf("LOADED_EPOCH")) == epoch and
            self.win.rd(regmap.offsetOf("LOADED_Q1_BLOCKS")) == q1_blocks and
            self.win.rd(regmap.offsetOf("LOADED_COLS")) == cols;
    }

    pub fn quantStatus(self: Kernel) u32 {
        if (!self.hasActivationReuse()) return 0;
        return self.win.rd(regmap.offsetOf("QUANT_STATUS"));
    }

    pub fn selectPackedActivation(self: *Kernel) void {
        if (self.hasActivationReuse()) self.win.wr(regmap.offsetOf("ACT_MODE"), @intFromEnum(ActivationMode.packed_load));
    }

    pub fn selectDdrResult(self: *Kernel) void {
        if (self.hasScratch()) self.win.wr(regmap.offsetOf("SCRATCH_MODE"), @intFromEnum(ScratchMode.ddr_only));
    }

    pub fn configureScratchTee(self: *Kernel, role: section.F32Role, rows: u32, tokens: u32) void {
        std.debug.assert(self.hasScratch());
        std.debug.assert(role == .x0 or role == .x1);
        self.configureScratch(.tee, role, rows, tokens);
    }

    pub fn configureScratchDrain(self: *Kernel, role: section.F32Role, rows: u32, tokens: u32) void {
        std.debug.assert(self.hasScratch());
        self.configureScratch(.drain, role, rows, tokens);
    }

    fn configureScratch(self: *Kernel, mode: ScratchMode, role: section.F32Role, rows: u32, tokens: u32) void {
        self.win.wr(regmap.offsetOf("SCRATCH_ROLE"), @intFromEnum(role));
        self.win.wr(regmap.offsetOf("SCRATCH_ROWS"), rows);
        self.win.wr(regmap.offsetOf("SCRATCH_TOKENS"), tokens);
        self.win.wr(regmap.offsetOf("SCRATCH_MODE"), @intFromEnum(mode));
    }

    pub fn startScratchDrain(self: *Kernel) void {
        std.debug.assert(self.hasScratch());
        self.win.wr(regmap.offsetOf("SCRATCH_CTRL"), SCRATCH_CTRL_DRAIN_START);
    }

    pub fn waitScratchWriterDone(self: *Kernel, role: section.F32Role) KernelError!void {
        try self.waitScratchDone(SCRATCH_WRITER_DONE, scratchRoleValidMask(role));
    }

    pub fn waitScratchDrainDone(self: *Kernel) KernelError!void {
        try self.waitScratchDone(SCRATCH_DRAIN_DONE, 0);
    }

    fn waitScratchDone(self: *Kernel, done_mask: u32, required_valid_mask: u32) KernelError!void {
        var i: usize = 0;
        while (i < regwin.wait_limit) : (i += 1) {
            const status = self.win.rd(regmap.offsetOf("SCRATCH_STATUS"));
            if (status & SCRATCH_ANY_ERROR != 0) {
                std.debug.print("pl: scratch failed status=0x{x} error=0x{x}\n", .{
                    status,
                    self.win.rd(regmap.offsetOf("SCRATCH_ERROR")),
                });
                return error.ScratchState;
            }
            if (status & done_mask != 0) {
                if (status & required_valid_mask != required_valid_mask) return error.ScratchState;
                return;
            }
        }
        return error.ScratchTimeout;
    }

    pub fn finishScratchDiagnostic(self: *Kernel) void {
        if (!self.hasScratch()) return;
        const status = self.win.rd(regmap.offsetOf("SCRATCH_STATUS"));
        if (status & (SCRATCH_WRITER_BUSY | SCRATCH_DRAIN_BUSY) != 0) {
            self.win.wr(regmap.offsetOf("SCRATCH_CTRL"), SCRATCH_CTRL_ABORT);
        }
        self.selectDdrResult();
    }

    pub fn cycles(self: Kernel) u32 {
        return self.win.rd(regmap.offsetOf("CYCLES"));
    }

    pub fn wStall(self: Kernel) u32 {
        return self.win.rd(regmap.offsetOf("W_STALL"));
    }
    pub fn aStall(self: Kernel) u32 {
        return self.win.rd(regmap.offsetOf("A_STALL"));
    }
    pub fn rStall(self: Kernel) u32 {
        return self.win.rd(regmap.offsetOf("R_STALL"));
    }
    pub fn wBeats(self: Kernel) u32 {
        return self.win.rd(regmap.offsetOf("W_BEATS"));
    }
    pub fn aBeats(self: Kernel) u32 {
        return self.win.rd(regmap.offsetOf("A_BEATS"));
    }
    pub fn rBeats(self: Kernel) u32 {
        return self.win.rd(regmap.offsetOf("R_BEATS"));
    }
};

pub fn capabilityInfo(kernel: Kernel) capabilities.EngineInfo {
    return .{
        .id = expected_id,
        .version = kernel.version,
        .clock_hz = kernel.clk_hz,
        .dim0 = expected_rows,
        .dim1 = expected_weight_ports,
        .dim2 = @intCast(mc_cols_max),
        .dim3 = @intCast(layout.max_q1_blocks * layout.q1_block),
    };
}

pub fn formatMask(kernel: Kernel) u32 {
    var mask = capabilities.Format.weight_q1_0 |
        capabilities.Format.activation_q8_0 | capabilities.Format.io_f32;
    if (kernel.version >= version_with_ternary) mask |= capabilities.Format.weight_q2_0_g64;
    return mask;
}

// ---- seq.v run builder ------------------------------------------------------------------------

/// Every DMA window the matmul stream may program — what seq.validate scans for transfer
/// records. Comptime-derived from the same regmap the MMIO drivers open.
pub const seq_dma_bases: [layout.weight_ports + 1]u32 = blk: {
    var bases: [layout.weight_ports + 1]u32 = undefined;
    for (dma_w_bases, 0..) |base, i| bases[i] = @intCast(base);
    bases[layout.weight_ports] = @intCast(dma_a_base);
    break :blk bases;
};

/// Maximum entries for one column group. Kernel v12 adds NUM_ROWS to the legacy
/// 48-entry program; older bitstreams omit it.
pub const seq_entries_per_op: usize = 49;

/// Push one column-group's register program onto `b` — identical in order and value to the
/// MMIO pokes in tryMatmul step 2 (pinned by the golden test below). seq.v replays it with no
/// PS in the loop; seq.validate gates it against the op's own buffer ranges before submit.
pub fn buildOp(
    b: *seq.Builder,
    weight_phys: [layout.weight_ports]u64,
    weight_port_bytes: usize,
    acts_phys: u64,
    act_total: usize,
    result_phys: u64,
    result_bytes: usize,
    q1_blocks: u32,
    num_rb: u32,
    group: u32,
    logical_rows: ?u32,
    weight_fmt: wire.WeightFormat,
) void {
    const kb: u32 = @intCast(kernel_base);
    dma_mod.record.reset(b, seq_dma_bases[0]);
    for (seq_dma_bases[1..layout.weight_ports]) |base| dma_mod.record.resetMm2s(b, base);
    dma_mod.record.resetMm2s(b, seq_dma_bases[layout.weight_ports]);
    dma_mod.record.startWriteToDdr(b, seq_dma_bases[0], result_phys, @intCast(result_bytes));
    for (seq_dma_bases[0..layout.weight_ports], 0..) |base, port| {
        dma_mod.record.startReadFromDdr(b, base, weight_phys[port], @intCast(weight_port_bytes));
    }
    dma_mod.record.startReadFromDdr(b, seq_dma_bases[layout.weight_ports], acts_phys, @intCast(act_total));
    b.write(kb + regmap.offsetOf("NUM_Q1_BLOCKS"), q1_blocks);
    b.write(kb + regmap.offsetOf("NUM_ROWBLOCKS"), num_rb);
    b.write(kb + regmap.offsetOf("NUM_COLS"), group);
    if (logical_rows) |rows| b.write(kb + regmap.offsetOf("NUM_ROWS"), rows);
    b.write(kb + regmap.offsetOf("WEIGHT_FMT"), @intFromEnum(weight_fmt));
    b.write(kb + regmap.offsetOf("CTRL"), CTRL_START);
    b.wait(kb + regmap.offsetOf("STATUS"), STATUS_DONE, STATUS_DONE);
    for (seq_dma_bases[0..layout.weight_ports]) |base| dma_mod.record.waitReadDone(b, base);
    dma_mod.record.waitReadDone(b, seq_dma_bases[layout.weight_ports]);
    dma_mod.record.waitWriteDone(b, seq_dma_bases[0]);
}

const Counters = struct {
    cycles: u64 = 0,
    w_stall: u64 = 0,
    a_stall: u64 = 0,
    r_stall: u64 = 0,
    w_beats: u64 = 0,
    a_beats: u64 = 0,
    r_beats: u64 = 0,
};

const ResultPlan = struct {
    direct: bool,
    dma_bytes: usize,
    logical_rows: ?u32,
};

fn resultPlan(rows: usize, group: usize, num_rb: usize, has_logical_rows: bool) ResultPlan {
    const padded_rows = num_rb * layout.rows_per_block;
    const direct = group == 1 and (rows == padded_rows or has_logical_rows);
    return .{
        .direct = direct,
        .dma_bytes = if (direct) rows * @sizeOf(f32) else padded_rows * group * @sizeOf(f32),
        // Staged multi-column results retain their fixed rowblock stride.
        .logical_rows = if (has_logical_rows) @intCast(if (direct) rows else padded_rows) else null,
    };
}

fn scratchDiagnosticEligible(kernel_version: u32, requested: bool, rows: usize) bool {
    return requested and
        kernel_version >= version_with_scratch and
        rows != 0 and
        rows <= section.ffn_dim_max and
        rows % layout.rows_per_block == 0;
}

fn groupTileCols(remaining: usize, scratch_diagnostic: bool) usize {
    const limit = if (scratch_diagnostic)
        @min(mc_cols_max, @as(usize, section.query_tile_max))
    else
        mc_cols_max;
    return @min(limit, remaining);
}

fn scratchRoleForProjection(projection_index: usize) section.F32Role {
    std.debug.assert(projection_index < 2);
    // llama.cpp builds Qwen's UP projection first and GATE second. The section
    // schedule names those physical roles X1 and X0 respectively.
    return if (projection_index == 0) .x1 else .x0;
}

fn scratchRoleValidMask(role: section.F32Role) u32 {
    return @as(u32, 1) << @intCast(5 + @intFromEnum(role));
}

fn scratchDrainBytes(rows: usize, tokens: usize) usize {
    return rows * tokens * @sizeOf(f32);
}

fn firstByteMismatch(a: []const u8, b: []const u8) ?usize {
    if (a.len != b.len) return @min(a.len, b.len);
    for (a, b, 0..) |a_byte, b_byte, i| {
        if (a_byte != b_byte) return i;
    }
    return null;
}

fn envEnabled(comptime name: [:0]const u8) bool {
    const raw = std.c.getenv(name.ptr) orelse return false;
    const value = std.mem.span(raw);
    return value.len != 0 and !std.mem.eql(u8, value, "0");
}

/// Generic over the heap so the runtime composes it only for heaps with physical
/// addressing (XRT); the fake heap never instantiates it.
pub fn Backend(comptime Heap: type) type {
    return struct {
        const Self = @This();

        allocator: std.mem.Allocator,
        kernel: Kernel,
        dma_w: [layout.weight_ports]dma_mod.Dma,
        dma_a: dma_mod.Dma,
        acts_staging: wire.TensorRange,
        result_staging: wire.TensorRange,
        // seq.v dispatch (opt-in: PENZAI_SEQ + a bitstream >= version_with_seq). The per-op
        // register dance is loaded into seq_top's command BRAM and replayed in fabric instead
        // of poked over MMIO. null = MMIO dispatch.
        seq_ctrl: ?seq_ctrl_mod.SeqCtrl = null,
        seq_entries: [seq_entries_per_op]seq.Entry = undefined,
        // Reusable per-call scratch, grown on demand (steady state: no realloc).
        column: []f32 = &.{},
        quants: []i8 = &.{},
        act_scales: []f16 = &.{},
        next_activation_epoch: u32 = 1,
        scratch_verify: bool = false,

        pub fn init(allocator: std.mem.Allocator, heap: *Heap) Error!Self {
            // The resident weight packing splits each 16-row block across this
            // many DMA ports; the address table must expose exactly that many.
            comptime std.debug.assert(dma_w_bases.len == layout.weight_ports);
            var kernel = try Kernel.open(kernel_base);
            errdefer kernel.deinit();
            // Userspace can restart without reconfiguring PL. Restore the
            // legacy-safe mode before any optional seq.v command can run.
            kernel.selectPackedActivation();
            kernel.finishScratchDiagnostic();

            const scratch_requested = envEnabled("PENZAI_PL_SCRATCH_VERIFY");
            const scratch_verify = scratch_requested and kernel.hasScratch();
            if (scratch_verify) {
                std.debug.print("pl: grouped matmul scratch tee verification ENABLED\n", .{});
            } else if (scratch_requested) {
                std.debug.print("pl: PENZAI_PL_SCRATCH_VERIFY set but kernel v{d} < v{d}; diagnostic disabled\n", .{
                    kernel.version,
                    version_with_scratch,
                });
            }

            // The gemm window floor is baked into the kernel (decode_top.EMIN_FLOOR), so there
            // is nothing to write here — the 104-bit accumulator covers the full f16 range with
            // no per-model calibration.
            var dma_w: [layout.weight_ports]dma_mod.Dma = undefined;
            var opened: usize = 0;
            errdefer for (dma_w[0..opened]) |*dma| dma.deinit();
            for (&dma_w, dma_w_bases[0..]) |*dma, base| {
                dma.* = try dma_mod.Dma.open(base);
                opened += 1;
            }
            var dma_a = try dma_mod.Dma.open(dma_a_base);
            errdefer dma_a.deinit();

            const acts_staging = heap.allocate(acts_staging_cap, 4096) catch return error.HeapFailure;
            const result_staging = heap.allocate(result_staging_cap, 4096) catch return error.HeapFailure;

            // Opt-in seq.v dispatch, version-gated: an older bitstream has nothing at the seq
            // window and a poke there is an unmapped sc_ctrl access. Refuse loudly, run MMIO.
            var sc: ?seq_ctrl_mod.SeqCtrl = null;
            if (std.c.getenv("PENZAI_SEQ") != null) {
                if (kernel.version >= version_with_seq) {
                    sc = try seq_ctrl_mod.SeqCtrl.open(seq.ctrl.base);
                    std.debug.print("pl: seq.v dispatch ENABLED (matmul)\n", .{});
                } else {
                    std.debug.print("pl: PENZAI_SEQ set but kernel v{d} < v{d} (no seq_top in this bitstream) — MMIO dispatch\n", .{ kernel.version, version_with_seq });
                }
            }

            return .{
                .allocator = allocator,
                .kernel = kernel,
                .dma_w = dma_w,
                .dma_a = dma_a,
                .acts_staging = acts_staging,
                .result_staging = result_staging,
                .seq_ctrl = sc,
                .scratch_verify = scratch_verify,
            };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.column);
            self.allocator.free(self.quants);
            self.allocator.free(self.act_scales);
            if (self.seq_ctrl) |*sc| sc.deinit();
            self.dma_a.deinit();
            for (&self.dma_w) |*dma| dma.deinit();
            self.kernel.deinit();
            self.* = undefined;
        }

        pub fn hasCounters(self: *const Self) bool {
            return self.kernel.hasCounters();
        }

        /// Run `mm` on the PL if its shape is supported. Detailed timing and
        /// counter reads are structurally absent when `ctx` is null.
        pub fn tryMatmul(self: *Self, heap: *Heap, mm: wire.MatmulQ1A8, ctx: ?*profile.ProfileContext) Error!?profile.MatmulExecution {
            if (mm.weight_fmt == .w158a8 and self.kernel.version < version_with_ternary) return null;
            const rows: usize = mm.rows;
            const cols: usize = mm.cols;
            const k: usize = mm.k;
            if (rows == 0 or cols == 0 or k == 0 or k % layout.q1_block != 0) return null;

            const q1_blocks = k / layout.q1_block;
            // The kernel accumulates all of K in one exact 104-bit-window pass; its acts
            // BRAM holds at most layout.max_q1_blocks blocks. A larger K would alias in the
            // BRAM, so decline it → the runtime runs the software matmul.
            if (q1_blocks > layout.max_q1_blocks) return null;
            const num_rb = layout.rowblocksFor(rows);
            const bytes_per_port_block = switch (mm.weight_fmt) {
                .w1a8 => layout.packed_per_port_q1_block,
                .w158a8 => layout.ternary_packed_per_port_block,
            };
            const weight_port_bytes = num_rb * q1_blocks * bytes_per_port_block;
            const weight_bytes = weight_port_bytes * layout.weight_ports;
            const act_stream_bytes = q1_blocks * layout.acts_per_q1_block; // per column
            const dst_bytes = rows * cols * @sizeOf(f32);

            // Shapes must match the resident packing and the staging windows must
            // hold one COLS_MAX group of acts and results.
            if (mm.weights.nbytes != weight_bytes) return null;
            if (mm.acts.nbytes != k * cols * @sizeOf(f32)) return null;
            if (mm.dst.nbytes != dst_bytes) return null;
            if (mc_cols_max * act_stream_bytes > acts_staging_cap) return null;
            if (num_rb * mc_cols_max * result_bytes_per_rb > result_staging_cap) return null;

            const wrapper_start = profile.begin(ctx);
            try self.ensureScratch(k);
            const q8_blocks = q1_blocks * layout.q8_subblocks;
            const weights_phys = heap.deviceAddress(mm.weights) catch return error.HeapFailure;
            var weight_phys: [layout.weight_ports]u64 = undefined;
            for (&weight_phys, 0..) |*phys, port| {
                phys.* = weights_phys + @as(u64, @intCast(port * weight_port_bytes));
            }
            const acts_bytes = heap.bytes(mm.acts) catch return error.HeapFailure; // col-major k*cols f32
            const dst_buf = heap.bytes(mm.dst) catch return error.HeapFailure; // col-major rows*cols f32

            var counters: Counters = .{};
            var result = profile.MatmulExecution{ .path = .direct };
            var last = wrapper_start;

            var col0: usize = 0;
            while (col0 < cols) {
                const group = @min(mc_cols_max, cols - col0);
                const act_total = group * act_stream_bytes;
                const result_plan = resultPlan(rows, group, num_rb, self.kernel.hasLogicalRows());
                if (!result_plan.direct) result.path = .staged;
                const acts_dma = subRange(self.acts_staging, act_total);
                const result_dma = if (result_plan.direct)
                    offsetRange(mm.dst, col0 * rows * @sizeOf(f32), result_plan.dma_bytes)
                else
                    subRange(self.result_staging, result_plan.dma_bytes);
                const acts_staging_buf = heap.bytes(acts_dma) catch return error.HeapFailure;

                // 1. Quantize + pack each column of the group (acts are col-major,
                // so each column is a contiguous k*f32 block; LE -> direct copy).
                for (0..group) |c| {
                    const src = ((col0 + c) * k) * @sizeOf(f32);
                    @memcpy(std.mem.sliceAsBytes(self.column[0..k]), acts_bytes[src..][0 .. k * @sizeOf(f32)]);
                    layout.quantizeQ8_0Simd(self.column[0..k], self.quants[0..k], self.act_scales[0..q8_blocks]) catch return error.HeapFailure;
                    packActs(q1_blocks, self.quants[0..k], self.act_scales[0..q8_blocks], acts_staging_buf[c * act_stream_bytes ..][0..act_stream_bytes]);
                }
                result.quantize_pack_ns +|= profile.lap(ctx, &last);
                heap.syncToDevice(acts_dma) catch return error.HeapFailure;
                result.sync_to_ns +|= profile.lap(ctx, &last);

                const acts_phys = heap.deviceAddress(acts_dma) catch return error.HeapFailure;
                const result_phys = heap.deviceAddress(result_dma) catch return error.HeapFailure;

                // 2. Program DMAs (arm result first), run the kernel, wait. Weights
                // stream from their resident range once for the whole group. Two arms,
                // same register program: seq.v replays it in fabric, or the PS pokes it.
                if (self.seq_ctrl) |*sc| {
                    var b = seq.Builder.init(&self.seq_entries);
                    buildOp(&b, weight_phys, weight_port_bytes, acts_phys, act_total, result_phys, result_plan.dma_bytes, @intCast(q1_blocks), @intCast(num_rb), @intCast(group), result_plan.logical_rows, mm.weight_fmt);
                    std.debug.assert(!b.overflow);
                    // The gate that keeps a bad stream from ever reaching a DMA: every
                    // transfer must land inside this op's own buffers.
                    const allowed = [_]seq.Range{
                        .{ .lo = weights_phys, .hi = weights_phys + weight_bytes },
                        .{ .lo = acts_phys, .hi = acts_phys + act_total },
                        .{ .lo = result_phys, .hi = result_phys + result_plan.dma_bytes },
                    };
                    seq.validate(b.entries(), &seq_dma_bases, &allowed) catch |e| {
                        std.debug.print("pl: seq stream REJECTED ({s}) — not submitting\n", .{@errorName(e)});
                        return error.SeqDispatch;
                    };
                    sc.load(0, b.entries()) catch return error.SeqDispatch;
                    sc.run(0, b.count()) catch return error.SeqDispatch;
                    result.setup_ns +|= profile.lap(ctx, &last);
                    sc.waitDone() catch |e| {
                        std.debug.print("pl: seq run FAILED ({s}) at entry {d}; aborting executor\n", .{ @errorName(e), sc.errIndex() });
                        sc.abort();
                        return error.SeqDispatch;
                    };
                    result.wait_ns +|= profile.lap(ctx, &last);
                } else {
                    try self.dma_w[0].reset();
                    for (self.dma_w[1..]) |*dma| try dma.resetMm2s();
                    try self.dma_a.resetMm2s();
                    try self.dma_w[0].startWriteToDdr(result_phys, result_plan.dma_bytes);
                    for (&self.dma_w, 0..) |*dma, port| {
                        try dma.startReadFromDdr(weight_phys[port], weight_port_bytes);
                    }
                    try self.dma_a.startReadFromDdr(acts_phys, act_total);
                    self.kernel.run(@intCast(q1_blocks), @intCast(num_rb), @intCast(group), result_plan.logical_rows, mm.weight_fmt);
                    result.setup_ns +|= profile.lap(ctx, &last);
                    self.kernel.waitDone() catch |err| {
                        self.abortProjectionDmas();
                        return err;
                    };
                    for (&self.dma_w) |*dma| try dma.waitReadDone();
                    try self.dma_a.waitReadDone();
                    try self.dma_w[0].waitWriteDone();
                    result.wait_ns +|= profile.lap(ctx, &last);
                }

                result.kernel_runs +|= 1;
                if (ctx != null) accumulateCounters(&counters, self.readCounters());

                // 3. Gather results: stream is [rowblock][col][row],
                // rowblock-lane-major; place into the col-major destination,
                // dropping pad rows.
                heap.syncFromDevice(result_dma) catch return error.HeapFailure;
                result.sync_from_ns +|= profile.lap(ctx, &last);
                if (!result_plan.direct) {
                    const result_buf = heap.bytes(result_dma) catch return error.HeapFailure;
                    gather.gatherResults(dst_buf, result_buf, rows, group, col0, num_rb);
                    result.result_layout_ns +|= profile.lap(ctx, &last);
                }
                col0 += group;
            }

            if (ctx) |active| result.wrapper_ns = shared.profiling.elapsed(wrapper_start, active.now());
            result.cycles = counters.cycles;
            result.w_stall_cycles = counters.w_stall;
            result.a_stall_cycles = counters.a_stall;
            result.r_stall_cycles = counters.r_stall;
            result.w_beats = counters.w_beats;
            result.a_beats = counters.a_beats;
            result.r_beats = counters.r_beats;
            return result;
        }

        /// Execute the atomic gate/up projection group. Each column tile enters
        /// as raw FP32 and is quantized in PL once; the second projection reuses
        /// the exact native-Q8 payload already resident in gemm_kernel.
        pub fn tryMatmulGroup2(
            self: *Self,
            heap: *Heap,
            group_op: wire.MatmulQ1A8Group2,
            ctx: ?*profile.ProfileContext,
        ) Error!?profile.MatmulExecution {
            if (!self.kernel.hasActivationReuse()) return null;

            const first = group_op.projections[0];
            const second = group_op.projections[1];
            if (first.rows == 0 or group_op.cols == 0 or group_op.k == 0 or
                first.rows != second.rows or first.weight_fmt != second.weight_fmt)
            {
                return null;
            }
            if (groupRangesOverlap(group_op)) return null;
            if (first.weight_fmt == .w158a8 and self.kernel.version < version_with_ternary) return null;

            const rows: usize = first.rows;
            const cols: usize = group_op.cols;
            const k: usize = group_op.k;
            if (k % layout.q1_block != 0) return null;
            const q1_blocks = k / layout.q1_block;
            if (q1_blocks > layout.max_q1_blocks) return null;
            const num_rb = layout.rowblocksFor(rows);
            const bytes_per_port_block = switch (first.weight_fmt) {
                .w1a8 => layout.packed_per_port_q1_block,
                .w158a8 => layout.ternary_packed_per_port_block,
            };
            const weight_port_bytes = num_rb * q1_blocks * bytes_per_port_block;
            const weight_bytes = weight_port_bytes * layout.weight_ports;
            const dst_bytes = rows * cols * @sizeOf(f32);
            if (first.weights.nbytes != weight_bytes or second.weights.nbytes != weight_bytes) return null;
            if (group_op.acts.nbytes != k * cols * @sizeOf(f32)) return null;
            if (first.dst.nbytes != dst_bytes or second.dst.nbytes != dst_bytes) return null;
            if (num_rb * mc_cols_max * result_bytes_per_rb > result_staging_cap) return null;
            const scratch_diagnostic = scratchDiagnosticEligible(self.kernel.version, self.scratch_verify, rows);

            const wrapper_start = profile.begin(ctx);
            const projections = group_op.projections;
            const dst_bufs = [2][]u8{
                heap.bytes(first.dst) catch return error.HeapFailure,
                heap.bytes(second.dst) catch return error.HeapFailure,
            };
            var weight_phys: [2][layout.weight_ports]u64 = undefined;
            for (projections, 0..) |projection, projection_index| {
                const base = heap.deviceAddress(projection.weights) catch return error.HeapFailure;
                for (&weight_phys[projection_index], 0..) |*phys, port| {
                    phys.* = base + @as(u64, @intCast(port * weight_port_bytes));
                }
            }

            var counters: Counters = .{};
            var result = profile.MatmulExecution{ .path = .direct };
            var last = wrapper_start;
            // Later primitive commands assume both legacy modes. Restore them
            // even if a grouped DMA, scratch drain, or comparison fails.
            defer {
                self.kernel.selectPackedActivation();
                if (scratch_diagnostic) self.kernel.finishScratchDiagnostic();
            }

            var col0: usize = 0;
            while (col0 < cols) {
                const tile_cols = groupTileCols(cols - col0, scratch_diagnostic);
                const act_total = tile_cols * k * @sizeOf(f32);
                const result_plan = resultPlan(rows, tile_cols, num_rb, self.kernel.hasLogicalRows());
                if (!result_plan.direct) result.path = .staged;
                const acts_dma = offsetRange(group_op.acts, col0 * k * @sizeOf(f32), act_total);
                heap.syncToDevice(acts_dma) catch return error.HeapFailure;
                result.sync_to_ns +|= profile.lap(ctx, &last);
                const acts_phys = heap.deviceAddress(acts_dma) catch return error.HeapFailure;
                const epoch = self.takeActivationEpoch();

                for (projections, 0..) |projection, projection_index| {
                    const result_dma = if (result_plan.direct)
                        offsetRange(projection.dst, col0 * rows * @sizeOf(f32), result_plan.dma_bytes)
                    else
                        subRange(self.result_staging, result_plan.dma_bytes);
                    const result_phys = heap.deviceAddress(result_dma) catch return error.HeapFailure;
                    const activation_mode: ActivationMode = if (projection_index == 0) .raw_f32_load else .reuse;

                    if (scratch_diagnostic) {
                        self.kernel.configureScratchTee(
                            scratchRoleForProjection(projection_index),
                            @intCast(rows),
                            @intCast(tile_cols),
                        );
                    }

                    // Check reuse before arming any mover. An impossible resident
                    // mismatch therefore cannot strand an active DMA channel.
                    if (activation_mode == .reuse and
                        !self.kernel.residentMatches(epoch, @intCast(q1_blocks), @intCast(tile_cols)))
                    {
                        return error.ActivationState;
                    }

                    try self.dma_w[0].reset();
                    for (self.dma_w[1..]) |*dma| try dma.resetMm2s();
                    if (activation_mode == .raw_f32_load) try self.dma_a.resetMm2s();
                    var dmas_armed = true;
                    errdefer if (dmas_armed) {
                        if (scratch_diagnostic) self.kernel.finishScratchDiagnostic();
                        self.abortProjectionDmas();
                    };
                    try self.dma_w[0].startWriteToDdr(result_phys, result_plan.dma_bytes);
                    for (&self.dma_w, 0..) |*dma, port| {
                        try dma.startReadFromDdr(weight_phys[projection_index][port], weight_port_bytes);
                    }
                    if (activation_mode == .raw_f32_load) {
                        try self.dma_a.startReadFromDdr(acts_phys, act_total);
                    }
                    self.kernel.runWithActivation(
                        @intCast(q1_blocks),
                        @intCast(num_rb),
                        @intCast(tile_cols),
                        result_plan.logical_rows,
                        projection.weight_fmt,
                        activation_mode,
                        epoch,
                    );
                    result.setup_ns +|= profile.lap(ctx, &last);
                    try self.kernel.waitDone();
                    if (scratch_diagnostic) {
                        try self.kernel.waitScratchWriterDone(scratchRoleForProjection(projection_index));
                    }
                    for (&self.dma_w) |*dma| try dma.waitReadDone();
                    if (activation_mode == .raw_f32_load) try self.dma_a.waitReadDone();
                    try self.dma_w[0].waitWriteDone();
                    dmas_armed = false;
                    result.wait_ns +|= profile.lap(ctx, &last);

                    result.kernel_runs +|= 1;
                    if (ctx != null) accumulateCounters(&counters, self.readCounters());
                    heap.syncFromDevice(result_dma) catch return error.HeapFailure;
                    result.sync_from_ns +|= profile.lap(ctx, &last);
                    if (!result_plan.direct) {
                        const result_buf = heap.bytes(result_dma) catch return error.HeapFailure;
                        gather.gatherResults(dst_bufs[projection_index], result_buf, rows, tile_cols, col0, num_rb);
                        result.result_layout_ns +|= profile.lap(ctx, &last);
                    }
                }
                if (scratch_diagnostic) {
                    try self.verifyScratchTile(heap, dst_bufs, rows, tile_cols, col0);
                    // The opt-in drain is outside the GEMM profile contract. Keep
                    // its wall time in wrapper_ns, but do not charge it to the
                    // next tile's sync_to segment.
                    _ = profile.lap(ctx, &last);
                }
                col0 += tile_cols;
            }

            if (ctx) |active| result.wrapper_ns = shared.profiling.elapsed(wrapper_start, active.now());
            result.cycles = counters.cycles;
            result.w_stall_cycles = counters.w_stall;
            result.a_stall_cycles = counters.a_stall;
            result.r_stall_cycles = counters.r_stall;
            result.w_beats = counters.w_beats;
            result.a_beats = counters.a_beats;
            result.r_beats = counters.r_beats;
            return result;
        }

        fn verifyScratchTile(
            self: *Self,
            heap: *Heap,
            dst_bufs: [2][]u8,
            rows: usize,
            tile_cols: usize,
            col0: usize,
        ) Error!void {
            const nbytes = scratchDrainBytes(rows, tile_cols);
            const drain_dma = subRange(self.result_staging, nbytes);
            const drain_phys = heap.deviceAddress(drain_dma) catch return error.HeapFailure;

            for (dst_bufs, 0..) |dst, projection_index| {
                self.kernel.configureScratchDrain(
                    scratchRoleForProjection(projection_index),
                    @intCast(rows),
                    @intCast(tile_cols),
                );

                try self.dma_w[0].resetS2mm();
                var dma_armed = true;
                errdefer if (dma_armed) {
                    self.kernel.finishScratchDiagnostic();
                    self.dma_w[0].resetS2mm() catch {};
                };
                try self.dma_w[0].startWriteToDdr(drain_phys, nbytes);
                self.kernel.startScratchDrain();
                try self.kernel.waitScratchDrainDone();
                try self.dma_w[0].waitWriteDone();
                dma_armed = false;

                heap.syncFromDevice(drain_dma) catch return error.HeapFailure;
                const got = heap.bytes(drain_dma) catch return error.HeapFailure;
                const expected_offset = col0 * rows * @sizeOf(f32);
                const expected = dst[expected_offset..][0..nbytes];
                if (firstByteMismatch(expected, got)) |offset| {
                    std.debug.print(
                        "pl: scratch mismatch projection={d} role={s} col0={d} byte={d} expected=0x{x:0>2} got=0x{x:0>2}\n",
                        .{
                            projection_index,
                            @tagName(scratchRoleForProjection(projection_index)),
                            col0,
                            offset,
                            if (offset < expected.len) expected[offset] else 0,
                            if (offset < got.len) got[offset] else 0,
                        },
                    );
                    return error.ScratchMismatch;
                }
            }
        }

        fn takeActivationEpoch(self: *Self) u32 {
            const epoch = if (self.next_activation_epoch == 0) 1 else self.next_activation_epoch;
            self.next_activation_epoch +%= 1;
            if (self.next_activation_epoch == 0) self.next_activation_epoch = 1;
            return epoch;
        }

        fn abortProjectionDmas(self: *Self) void {
            self.dma_w[0].reset() catch {};
            for (self.dma_w[1..]) |*dma| dma.resetMm2s() catch {};
            self.dma_a.resetMm2s() catch {};
        }

        fn readCounters(self: *Self) Counters {
            const cycles: u64 = self.kernel.cycles();
            if (!self.kernel.hasCounters()) {
                return .{ .cycles = cycles };
            }
            return .{
                .cycles = cycles,
                .w_stall = self.kernel.wStall(),
                .a_stall = self.kernel.aStall(),
                .r_stall = self.kernel.rStall(),
                .w_beats = self.kernel.wBeats(),
                .a_beats = self.kernel.aBeats(),
                .r_beats = self.kernel.rBeats(),
            };
        }

        fn ensureScratch(self: *Self, k: usize) Error!void {
            if (self.column.len >= k) return;
            self.allocator.free(self.column);
            self.allocator.free(self.quants);
            self.allocator.free(self.act_scales);
            self.column = self.allocator.alloc(f32, k) catch return error.OutOfMemory;
            self.quants = self.allocator.alloc(i8, k) catch return error.OutOfMemory;
            self.act_scales = self.allocator.alloc(f16, k / layout.q8_block) catch return error.OutOfMemory;
        }
    };
}

fn subRange(staging: wire.TensorRange, nbytes: usize) wire.TensorRange {
    return .{ .handle = staging.handle, .offset = 0, .nbytes = nbytes };
}

fn offsetRange(base: wire.TensorRange, offset: usize, nbytes: usize) wire.TensorRange {
    return .{ .handle = base.handle, .offset = base.offset + @as(u64, @intCast(offset)), .nbytes = @intCast(nbytes) };
}

fn groupRangesOverlap(group: wire.MatmulQ1A8Group2) bool {
    const first = group.projections[0];
    const second = group.projections[1];
    return rangesOverlap(first.dst, second.dst) or
        rangesOverlap(first.dst, group.acts) or rangesOverlap(second.dst, group.acts) or
        rangesOverlap(first.dst, first.weights) or rangesOverlap(first.dst, second.weights) or
        rangesOverlap(second.dst, first.weights) or rangesOverlap(second.dst, second.weights);
}

fn rangesOverlap(a: wire.TensorRange, b: wire.TensorRange) bool {
    if (a.handle != b.handle or a.nbytes == 0 or b.nbytes == 0) return false;
    const a_end = std.math.add(u64, a.offset, a.nbytes) catch return true;
    const b_end = std.math.add(u64, b.offset, b.nbytes) catch return true;
    return a.offset < b_end and b.offset < a_end;
}

/// Sum the per-group hardware counters into the matmul's total.
fn accumulateCounters(dst: *Counters, src: Counters) void {
    dst.cycles += src.cycles;
    dst.w_stall += src.w_stall;
    dst.a_stall += src.a_stall;
    dst.r_stall += src.r_stall;
    dst.w_beats += src.w_beats;
    dst.a_beats += src.a_beats;
    dst.r_beats += src.r_beats;
}

/// Pack the activation stream: per Q1 block, per Q8 sub-block, 4 int8 beats then
/// 1 scale beat (fp16 in the low 16 bits). One column, broadcast to all
/// rowblocks by the fabric. Mirrors shared/q1a8 act layout (160 B/Q1 block).
fn packActs(q1_blocks: usize, quants: []const i8, scales: []const f16, out: []u8) void {
    var off: usize = 0;
    for (0..q1_blocks) |blk| {
        for (0..layout.q8_subblocks) |sub| {
            const q8 = blk * layout.q8_subblocks + sub;
            const acts = quants[q8 * layout.q8_block ..][0..layout.q8_block];
            for (0..layout.q8_block / layout.beat_bytes) |beat| {
                var word: u64 = 0;
                for (0..layout.beat_bytes) |byte| {
                    const u: u64 = @as(u8, @bitCast(acts[beat * layout.beat_bytes + byte]));
                    word |= u << @intCast(byte * 8);
                }
                std.mem.writeInt(u64, out[off..][0..8], word, .little);
                off += 8;
            }
            std.mem.writeInt(u64, out[off..][0..8], @as(u16, @bitCast(scales[q8])), .little);
            off += 8;
        }
    }
}

test "packActs fills the expected stream length" {
    const q1_blocks = 3;
    var quants: [q1_blocks * layout.q1_block]i8 = undefined;
    for (&quants, 0..) |*q, i| q.* = @intCast(@as(i32, @intCast(i % 251)) - 125);
    var scales: [q1_blocks * layout.q8_subblocks]f16 = undefined;
    for (&scales, 0..) |*s, i| s.* = @floatCast(@as(f32, @floatFromInt(i + 1)) * 0.01);
    var out: [q1_blocks * layout.acts_per_q1_block]u8 = undefined;
    packActs(q1_blocks, &quants, &scales, &out);
    // Last scale beat carries the final sub-block's fp16 scale in its low 16 bits.
    const last = std.mem.readInt(u64, out[out.len - 8 ..][0..8], .little);
    try std.testing.expectEqual(@as(u16, @bitCast(scales[scales.len - 1])), @as(u16, @truncate(last)));
}

test "buildOp: exact per-col-group register program (bit-parity with the MMIO path)" {
    var buf: [seq_entries_per_op]seq.Entry = undefined;
    var b = seq.Builder.init(&buf);
    const wp = [_]u64{ 0x8_0000_0000, 0x8_0000_0100, 0x8_0000_0200, 0x8_0000_0300 };
    buildOp(&b, wp, 0x100, 0x8_1000_0000, 0x200, 0x8_2000_0000, 0x40, 7, 2, 1, 31, .w1a8);
    try std.testing.expect(!b.overflow);
    try std.testing.expectEqual(@as(u32, seq_entries_per_op), b.count());

    const E = seq.Entry;
    const W = struct {
        fn wr(addr: u32, val: u32) E {
            return .{ .tag = 0, .addr = addr, .a = val, .b = 0 };
        }
        fn wt(addr: u32, mask: u32, exp: u32) E {
            return .{ .tag = 1, .addr = addr, .a = mask, .b = exp };
        }
    };
    // Xilinx AXI-DMA constants, hand-written here so this test cross-checks dma.zig's privates.
    const RESET: u32 = 1 << 2;
    const RS: u32 = 1 << 0;
    const IDLE: u32 = 1 << 1;
    const w0: u32 = 0xA000_0000;
    const w1: u32 = 0xA001_0000;
    const w2: u32 = 0xA002_0000;
    const w3: u32 = 0xA003_0000;
    const da: u32 = 0xA004_0000;
    const kb: u32 = 0xA005_0000;
    const want = [_]E{
        // dma_w0.reset() (both channels), then MM2S-only resets: w1..w3 + acts
        W.wr(w0 + 0x00, RESET),                                                         W.wr(w0 + 0x30, RESET),                         W.wt(w0 + 0x00, RESET, 0),                                      W.wt(w0 + 0x30, RESET, 0),
        W.wr(w1 + 0x00, RESET),                                                         W.wt(w1 + 0x00, RESET, 0),                      W.wr(w2 + 0x00, RESET),                                         W.wt(w2 + 0x00, RESET, 0),
        W.wr(w3 + 0x00, RESET),                                                         W.wt(w3 + 0x00, RESET, 0),                      W.wr(da + 0x00, RESET),                                         W.wt(da + 0x00, RESET, 0),
        // arm the result S2MM FIRST
        W.wr(w0 + 0x30, RS),                                                            W.wr(w0 + 0x48, 0x2000_0000),                   W.wr(w0 + 0x4C, 0x8),                                           W.wr(w0 + 0x58, 0x40),
        // weight MM2S x4
        W.wr(w0 + 0x00, RS),                                                            W.wr(w0 + 0x18, 0x0000_0000),                   W.wr(w0 + 0x1C, 0x8),                                           W.wr(w0 + 0x28, 0x100),
        W.wr(w1 + 0x00, RS),                                                            W.wr(w1 + 0x18, 0x0000_0100),                   W.wr(w1 + 0x1C, 0x8),                                           W.wr(w1 + 0x28, 0x100),
        W.wr(w2 + 0x00, RS),                                                            W.wr(w2 + 0x18, 0x0000_0200),                   W.wr(w2 + 0x1C, 0x8),                                           W.wr(w2 + 0x28, 0x100),
        W.wr(w3 + 0x00, RS),                                                            W.wr(w3 + 0x18, 0x0000_0300),                   W.wr(w3 + 0x1C, 0x8),                                           W.wr(w3 + 0x28, 0x100),
        // acts MM2S
        W.wr(da + 0x00, RS),                                                            W.wr(da + 0x18, 0x1000_0000),                   W.wr(da + 0x1C, 0x8),                                           W.wr(da + 0x28, 0x200),
        // kernel config + start
        W.wr(kb + regmap.offsetOf("NUM_Q1_BLOCKS"), 7),                                 W.wr(kb + regmap.offsetOf("NUM_ROWBLOCKS"), 2), W.wr(kb + regmap.offsetOf("NUM_COLS"), 1),                      W.wr(kb + regmap.offsetOf("NUM_ROWS"), 31),
        W.wr(kb + regmap.offsetOf("WEIGHT_FMT"), @intFromEnum(wire.WeightFormat.w1a8)), W.wr(kb + regmap.offsetOf("CTRL"), CTRL_START),
        // waits: kernel done, then every mover idle (read x5, write x1)
        W.wt(kb + regmap.offsetOf("STATUS"), STATUS_DONE, STATUS_DONE), W.wt(w0 + 0x04, IDLE, IDLE),
        W.wt(w1 + 0x04, IDLE, IDLE),                                                    W.wt(w2 + 0x04, IDLE, IDLE),                    W.wt(w3 + 0x04, IDLE, IDLE),                                    W.wt(da + 0x04, IDLE, IDLE),
        W.wt(w0 + 0x34, IDLE, IDLE),
    };
    try std.testing.expectEqualSlices(E, &want, b.entries());
}

test "buildOp stream passes validation against its own ranges, fails without the result range" {
    var buf: [seq_entries_per_op]seq.Entry = undefined;
    var b = seq.Builder.init(&buf);
    const wp = [_]u64{ 0x8_0000_0000, 0x8_0000_0100, 0x8_0000_0200, 0x8_0000_0300 };
    buildOp(&b, wp, 0x100, 0x8_1000_0000, 0x200, 0x8_2000_0000, 0x40, 7, 2, 1, 31, .w1a8);
    const weights = seq.Range{ .lo = 0x8_0000_0000, .hi = 0x8_0000_0400 };
    const acts = seq.Range{ .lo = 0x8_1000_0000, .hi = 0x8_1000_0200 };
    const result = seq.Range{ .lo = 0x8_2000_0000, .hi = 0x8_2000_0040 };
    try seq.validate(b.entries(), &seq_dma_bases, &.{ weights, acts, result });
    try std.testing.expectError(error.S2mmOutOfRange, seq.validate(b.entries(), &seq_dma_bases, &.{ weights, acts }));
}

test "buildOp programs ternary weight format" {
    var buf: [seq_entries_per_op]seq.Entry = undefined;
    var b = seq.Builder.init(&buf);
    const wp = [_]u64{ 0x8_0000_0000, 0x8_0000_0080, 0x8_0000_0100, 0x8_0000_0180 };
    buildOp(&b, wp, 0x80, 0x8_1000_0000, 0xa0, 0x8_2000_0000, 0x40, 1, 1, 1, 16, .w158a8);

    const weight_fmt_addr: u32 = 0xA005_0000 + regmap.offsetOf("WEIGHT_FMT");
    var found = false;
    for (b.entries()) |entry| {
        if (entry.tag == 0 and entry.addr == weight_fmt_addr) {
            try std.testing.expectEqual(@as(u32, @intFromEnum(wire.WeightFormat.w158a8)), entry.a);
            found = true;
        }
    }
    try std.testing.expect(found);
}

test "result plan makes partial single-column outputs direct only on v12" {
    const rows: usize = 151_669;
    const num_rb = layout.rowblocksFor(rows);

    const legacy = resultPlan(rows, 1, num_rb, false);
    try std.testing.expect(!legacy.direct);
    try std.testing.expectEqual(@as(usize, 151_680 * @sizeOf(f32)), legacy.dma_bytes);
    try std.testing.expectEqual(@as(?u32, null), legacy.logical_rows);

    const direct = resultPlan(rows, 1, num_rb, true);
    try std.testing.expect(direct.direct);
    try std.testing.expectEqual(rows * @sizeOf(f32), direct.dma_bytes);
    try std.testing.expectEqual(@as(?u32, 151_669), direct.logical_rows);

    const staged = resultPlan(rows, 8, num_rb, true);
    try std.testing.expect(!staged.direct);
    try std.testing.expectEqual(@as(usize, 151_680 * 8 * @sizeOf(f32)), staged.dma_bytes);
    try std.testing.expectEqual(@as(?u32, 151_680), staged.logical_rows);
}

test "scratch diagnostic is v14-only and requires complete GEMM rowblocks" {
    try std.testing.expect(!scratchDiagnosticEligible(version_with_scratch - 1, true, 6144));
    try std.testing.expect(!scratchDiagnosticEligible(version_with_scratch, false, 6144));
    try std.testing.expect(scratchDiagnosticEligible(version_with_scratch, true, 6144));
    try std.testing.expect(scratchDiagnosticEligible(version_with_scratch, true, section.ffn_dim_max));
    try std.testing.expect(!scratchDiagnosticEligible(version_with_scratch, true, 0));
    try std.testing.expect(!scratchDiagnosticEligible(version_with_scratch, true, 40));
    try std.testing.expect(!scratchDiagnosticEligible(version_with_scratch, true, section.ffn_dim_max + layout.rows_per_block));
}

test "scratch group tiles and projection roles match the Qwen FFN schedule" {
    try std.testing.expectEqual(@as(usize, 8), groupTileCols(12, false));
    try std.testing.expectEqual(@as(usize, section.query_tile_max), groupTileCols(12, true));
    try std.testing.expectEqual(@as(usize, 3), groupTileCols(3, true));
    try std.testing.expectEqual(section.F32Role.x1, scratchRoleForProjection(0));
    try std.testing.expectEqual(section.F32Role.x0, scratchRoleForProjection(1));
    try std.testing.expectEqual(@as(u32, 1 << 6), scratchRoleValidMask(.x0));
    try std.testing.expectEqual(@as(u32, 1 << 7), scratchRoleValidMask(.x1));
    try std.testing.expectEqual(@as(usize, 6144 * 4 * @sizeOf(f32)), scratchDrainBytes(6144, 4));
}

test "scratch drain comparison is exact" {
    const same = [_]u8{ 1, 2, 3, 4, 5 };
    try std.testing.expectEqual(@as(?usize, null), firstByteMismatch(&same, &same));
    try std.testing.expectEqual(@as(?usize, 2), firstByteMismatch(&same, &.{ 1, 2, 9, 4, 5 }));
    try std.testing.expectEqual(@as(?usize, 3), firstByteMismatch(&.{ 1, 2, 3 }, &.{ 1, 2, 3, 4 }));
}

test "scratch drain stream is byte-identical to a col-major DDR tile" {
    const rows: usize = 6144;
    const tokens: usize = section.query_tile_max;
    const groups = rows / section.f32_rows_per_group;
    for (0..tokens) |token| {
        for (0..groups) |group| {
            for (0..section.f32_banks_per_role) |bank| {
                const drain_offset = ((token * groups + group) * section.f32_banks_per_role + bank) * section.f32_word_bytes;
                const even_row = group * section.f32_rows_per_group + bank * section.f32_values_per_word;
                const ddr_offset = (token * rows + even_row) * @sizeOf(f32);
                try std.testing.expectEqual(ddr_offset, drain_offset);

                const location = try section.f32Location(.x1, @intCast(token), @intCast(even_row));
                try std.testing.expectEqual(@as(u2, @intCast(bank)), location.bank);
                try std.testing.expectEqual(
                    @as(u32, @intCast(token)) * section.f32GroupsPerToken(.x1) + @as(u32, @intCast(group)),
                    location.address,
                );
            }
        }
    }
}

test "legacy sequencer program omits unsupported logical-row register" {
    var buf: [seq_entries_per_op]seq.Entry = undefined;
    var b = seq.Builder.init(&buf);
    const wp = [_]u64{ 0x8_0000_0000, 0x8_0000_0100, 0x8_0000_0200, 0x8_0000_0300 };
    buildOp(&b, wp, 0x100, 0x8_1000_0000, 0x200, 0x8_2000_0000, 0x40, 7, 2, 1, null, .w1a8);
    try std.testing.expect(!b.overflow);
    try std.testing.expectEqual(@as(u32, seq_entries_per_op - 1), b.count());
    for (b.entries()) |entry| {
        try std.testing.expect(entry.addr != @as(u32, @intCast(kernel_base)) + regmap.offsetOf("NUM_ROWS"));
    }
}
