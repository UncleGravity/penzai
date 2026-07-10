//! PL Q1A8 matmul backend: drives the fabric kernel for a wire matmul command.
//!
//! Weights are streamed straight from their resident XRT BO range. Activations
//! are quantized with the *canonical* quantizer (`shared.layout.quantizeQ8_0`, the
//! same round-nearest-even the PS oracle uses, so PL and PS feed bit-identical
//! int8), packed into a staging region, and DMA'd in. Results DMA into a staging
//! region and copy into the destination range.
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

// Columns multiplied per kernel run. Matches matmul_kernel COLS_MAX in the loaded
// bitstream (both derive from caps.cols_max). Prefill (cols>1) is tiled into
// groups of this many.
const mc_cols_max: usize = regmap.caps.cols_max;
const result_bytes_per_rb = gather.result_bytes_per_rb; // single source in gather.zig

// Errors this op can raise: the DMA substrate's, this kernel's, plus heap failures.
// SeqDispatch = the seq.v arm refused or failed a run (details already printed); never
// silently retried over MMIO — a failed replay means the stream or executor needs looking at.
const KernelError = regwin.Error || error{ KernelTimeout, BadId, BadVersion, BadRows, BadWeightPorts, BadClock };
pub const Error = dma_mod.Error || KernelError || error{ HeapFailure, OutOfMemory, SeqDispatch };

// ---- The matmul kernel AXI-Lite driver (a tenant on the PL substrate) -------
// Moved out of the former mmio.zig: the run signature, identity/shape checks, and
// counter accessors are matmul-specific. Register offsets come from the generated
// `regmap` module — no hand-duplicated constants.

const CTRL_START: u32 = 1 << 0;
const STATUS_DONE: u32 = 1 << 1;

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

    /// Fabric clock in MHz, self-described by the bitstream (CLK_HZ register).
    pub fn clkMhz(self: Kernel) f64 {
        return @as(f64, @floatFromInt(self.clk_hz)) / 1_000_000.0;
    }

    /// Set dims then strobe start. done_latched clears on the strobe.
    pub fn run(self: *Kernel, num_q1_blocks: u32, num_rowblocks: u32, num_cols: u32) void {
        self.win.wr(regmap.offsetOf("NUM_Q1_BLOCKS"), num_q1_blocks);
        self.win.wr(regmap.offsetOf("NUM_ROWBLOCKS"), num_rowblocks);
        self.win.wr(regmap.offsetOf("NUM_COLS"), num_cols);
        self.win.wr(regmap.offsetOf("CTRL"), CTRL_START);
    }

    pub fn waitDone(self: *Kernel) KernelError!void {
        var i: usize = 0;
        while (i < regwin.wait_limit) : (i += 1) {
            if (self.win.rd(regmap.offsetOf("STATUS")) & STATUS_DONE != 0) return;
        }
        return error.KernelTimeout;
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

// ---- seq.v run builder ------------------------------------------------------------------------

/// Every DMA window the matmul stream may program — what seq.validate scans for transfer
/// records. Comptime-derived from the same regmap the MMIO drivers open.
pub const seq_dma_bases: [layout.weight_ports + 1]u32 = blk: {
    var bases: [layout.weight_ports + 1]u32 = undefined;
    for (dma_w_bases, 0..) |base, i| bases[i] = @intCast(base);
    bases[layout.weight_ports] = @intCast(dma_a_base);
    break :blk bases;
};

/// Entries one column-group programs: dma_w0 reset (4) + 4x MM2S-only resets (8) + S2MM arm (4)
/// + 5x MM2S start (20) + kernel config/start (4) + kernel done + 5x MM2S idle + S2MM idle (7).
pub const seq_entries_per_op: usize = 47;

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
    b.write(kb + regmap.offsetOf("CTRL"), CTRL_START);
    b.wait(kb + regmap.offsetOf("STATUS"), STATUS_DONE, STATUS_DONE);
    for (seq_dma_bases[0..layout.weight_ports]) |base| dma_mod.record.waitReadDone(b, base);
    dma_mod.record.waitReadDone(b, seq_dma_bases[layout.weight_ports]);
    dma_mod.record.waitWriteDone(b, seq_dma_bases[0]);
}

/// Temporary per-segment timing accumulator for localizing per-call overhead.
const Seg = struct {
    calls: u64 = 0,
    quant_ns: u64 = 0,
    sync_to_ns: u64 = 0,
    setup_ns: u64 = 0,
    wait_ns: u64 = 0,
    sync_from_ns: u64 = 0,
    copy_ns: u64 = 0,
    cycles: u64 = 0,
};
const seg_report_interval: u64 = 200;

fn lapNs(io: ?std.Io, last: *u64) u64 {
    const now = shared.profiling.nowNs(io);
    const d = if (now >= last.*) now - last.* else 0;
    last.* = now;
    return d;
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
        seg: Seg = .{},

        pub fn init(allocator: std.mem.Allocator, heap: *Heap) Error!Self {
            // The resident weight packing splits each 16-row block across this
            // many DMA ports; the address table must expose exactly that many.
            comptime std.debug.assert(dma_w_bases.len == layout.weight_ports);
            var kernel = try Kernel.open(kernel_base);
            errdefer kernel.deinit();

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

        /// Run `mm` on the PL if its shape is supported; return counters on
        /// success, null to defer to the PS kernel. Errors are hardware/heap
        /// failures the runtime should surface, not silent fallbacks.
        pub fn tryMatmul(self: *Self, heap: *Heap, mm: wire.MatmulQ1A8, io: ?std.Io) Error!?profile.PlCounters {
            if (mm.weight_fmt != .w1a8) return null;
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
            const weight_port_bytes = num_rb * q1_blocks * layout.packed_per_port_q1_block;
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

            try self.ensureScratch(k);
            const q8_blocks = q1_blocks * layout.q8_subblocks;
            const weights_phys = heap.deviceAddress(mm.weights) catch return error.HeapFailure;
            var weight_phys: [layout.weight_ports]u64 = undefined;
            for (&weight_phys, 0..) |*phys, port| {
                phys.* = weights_phys + @as(u64, @intCast(port * weight_port_bytes));
            }
            const acts_bytes = heap.bytes(mm.acts) catch return error.HeapFailure; // col-major k*cols f32
            const dst_buf = heap.bytes(mm.dst) catch return error.HeapFailure; // col-major rows*cols f32

            var counters: profile.PlCounters = .{};
            var seg: Seg = .{};
            var last = shared.profiling.nowNs(io);

            var col0: usize = 0;
            while (col0 < cols) {
                const group = @min(mc_cols_max, cols - col0);
                const act_total = group * act_stream_bytes;
                const result_bytes = num_rb * group * result_bytes_per_rb;
                const direct_result = group == 1 and rows % layout.rows_per_block == 0;
                const acts_dma = subRange(self.acts_staging, act_total);
                const result_dma = if (direct_result)
                    offsetRange(mm.dst, col0 * rows * @sizeOf(f32), result_bytes)
                else
                    subRange(self.result_staging, result_bytes);
                const acts_staging_buf = heap.bytes(acts_dma) catch return error.HeapFailure;

                // 1. Quantize + pack each column of the group (acts are col-major,
                // so each column is a contiguous k*f32 block; LE -> direct copy).
                for (0..group) |c| {
                    const src = ((col0 + c) * k) * @sizeOf(f32);
                    @memcpy(std.mem.sliceAsBytes(self.column[0..k]), acts_bytes[src..][0 .. k * @sizeOf(f32)]);
                    layout.quantizeQ8_0Simd(self.column[0..k], self.quants[0..k], self.act_scales[0..q8_blocks]) catch return error.HeapFailure;
                    packActs(q1_blocks, self.quants[0..k], self.act_scales[0..q8_blocks], acts_staging_buf[c * act_stream_bytes ..][0..act_stream_bytes]);
                }
                seg.quant_ns += lapNs(io, &last);
                heap.syncToDevice(acts_dma) catch return error.HeapFailure;
                seg.sync_to_ns += lapNs(io, &last);

                const acts_phys = heap.deviceAddress(acts_dma) catch return error.HeapFailure;
                const result_phys = heap.deviceAddress(result_dma) catch return error.HeapFailure;

                // 2. Program DMAs (arm result first), run the kernel, wait. Weights
                // stream from their resident range once for the whole group. Two arms,
                // same register program: seq.v replays it in fabric, or the PS pokes it.
                if (self.seq_ctrl) |*sc| {
                    var b = seq.Builder.init(&self.seq_entries);
                    buildOp(&b, weight_phys, weight_port_bytes, acts_phys, act_total, result_phys, result_bytes, @intCast(q1_blocks), @intCast(num_rb), @intCast(group));
                    std.debug.assert(!b.overflow);
                    // The gate that keeps a bad stream from ever reaching a DMA: every
                    // transfer must land inside this op's own buffers.
                    const allowed = [_]seq.Range{
                        .{ .lo = weights_phys, .hi = weights_phys + weight_bytes },
                        .{ .lo = acts_phys, .hi = acts_phys + act_total },
                        .{ .lo = result_phys, .hi = result_phys + result_bytes },
                    };
                    seq.validate(b.entries(), &seq_dma_bases, &allowed) catch |e| {
                        std.debug.print("pl: seq stream REJECTED ({s}) — not submitting\n", .{@errorName(e)});
                        return error.SeqDispatch;
                    };
                    sc.load(0, b.entries()) catch return error.SeqDispatch;
                    sc.run(0, b.count()) catch return error.SeqDispatch;
                    seg.setup_ns += lapNs(io, &last);
                    sc.waitDone() catch |e| {
                        std.debug.print("pl: seq run FAILED ({s}) at entry {d}; aborting executor\n", .{ @errorName(e), sc.errIndex() });
                        sc.abort();
                        return error.SeqDispatch;
                    };
                    seg.wait_ns += lapNs(io, &last);
                } else {
                    try self.dma_w[0].reset();
                    for (self.dma_w[1..]) |*dma| try dma.resetMm2s();
                    try self.dma_a.resetMm2s();
                    try self.dma_w[0].startWriteToDdr(result_phys, result_bytes);
                    for (&self.dma_w, 0..) |*dma, port| {
                        try dma.startReadFromDdr(weight_phys[port], weight_port_bytes);
                    }
                    try self.dma_a.startReadFromDdr(acts_phys, act_total);
                    self.kernel.run(@intCast(q1_blocks), @intCast(num_rb), @intCast(group));
                    seg.setup_ns += lapNs(io, &last);
                    try self.kernel.waitDone();
                    for (&self.dma_w) |*dma| try dma.waitReadDone();
                    try self.dma_a.waitReadDone();
                    try self.dma_w[0].waitWriteDone();
                    seg.wait_ns += lapNs(io, &last);
                }

                accumulateCounters(&counters, self.readCounters());

                // 3. Gather results: stream is [rowblock][col][row],
                // rowblock-lane-major; place into the col-major destination,
                // dropping pad rows.
                heap.syncFromDevice(result_dma) catch return error.HeapFailure;
                seg.sync_from_ns += lapNs(io, &last);
                if (!direct_result) {
                    const result_buf = heap.bytes(result_dma) catch return error.HeapFailure;
                    gather.gatherResults(dst_buf, result_buf, rows, group, col0, num_rb);
                    seg.copy_ns += lapNs(io, &last);
                }
                col0 += group;
            }

            self.seg.calls += 1;
            self.seg.quant_ns += seg.quant_ns;
            self.seg.sync_to_ns += seg.sync_to_ns;
            self.seg.setup_ns += seg.setup_ns;
            self.seg.wait_ns += seg.wait_ns;
            self.seg.sync_from_ns += seg.sync_from_ns;
            self.seg.copy_ns += seg.copy_ns;
            self.seg.cycles += counters.cycles;
            if (self.seg.calls % seg_report_interval == 0) self.flushSeg();

            return counters;
        }

        /// Print mean per-segment times for the last window of PL calls to stderr,
        /// then reset. A temporary diagnostic for localizing per-call overhead.
        fn flushSeg(self: *Self) void {
            const s = self.seg;
            if (s.calls == 0) return;
            const n: f64 = @floatFromInt(s.calls);
            const us = struct {
                fn f(ns: u64, count: f64) f64 {
                    return @as(f64, @floatFromInt(ns)) / count / 1000.0;
                }
            }.f;
            std.debug.print("pl seg us/call (n={d}): quant={d:.1} sync_to={d:.1} setup={d:.1} wait={d:.1} sync_from={d:.1} copy={d:.1} | kernel_compute={d:.1}\n", .{
                s.calls,
                us(s.quant_ns, n),
                us(s.sync_to_ns, n),
                us(s.setup_ns, n),
                us(s.wait_ns, n),
                us(s.sync_from_ns, n),
                us(s.copy_ns, n),
                @as(f64, @floatFromInt(s.cycles)) / n / self.kernel.clkMhz(), // cycles -> us
            });
            self.seg = .{};
        }

        fn readCounters(self: *Self) profile.PlCounters {
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

/// Sum the per-group hardware counters into the matmul's total.
fn accumulateCounters(dst: *profile.PlCounters, src: profile.PlCounters) void {
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
    buildOp(&b, wp, 0x100, 0x8_1000_0000, 0x200, 0x8_2000_0000, 0x40, 7, 2, 1);
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
        W.wr(w0 + 0x00, RESET), W.wr(w0 + 0x30, RESET), W.wt(w0 + 0x00, RESET, 0), W.wt(w0 + 0x30, RESET, 0),
        W.wr(w1 + 0x00, RESET), W.wt(w1 + 0x00, RESET, 0),
        W.wr(w2 + 0x00, RESET), W.wt(w2 + 0x00, RESET, 0),
        W.wr(w3 + 0x00, RESET), W.wt(w3 + 0x00, RESET, 0),
        W.wr(da + 0x00, RESET), W.wt(da + 0x00, RESET, 0),
        // arm the result S2MM FIRST
        W.wr(w0 + 0x30, RS), W.wr(w0 + 0x48, 0x2000_0000), W.wr(w0 + 0x4C, 0x8), W.wr(w0 + 0x58, 0x40),
        // weight MM2S x4
        W.wr(w0 + 0x00, RS), W.wr(w0 + 0x18, 0x0000_0000), W.wr(w0 + 0x1C, 0x8), W.wr(w0 + 0x28, 0x100),
        W.wr(w1 + 0x00, RS), W.wr(w1 + 0x18, 0x0000_0100), W.wr(w1 + 0x1C, 0x8), W.wr(w1 + 0x28, 0x100),
        W.wr(w2 + 0x00, RS), W.wr(w2 + 0x18, 0x0000_0200), W.wr(w2 + 0x1C, 0x8), W.wr(w2 + 0x28, 0x100),
        W.wr(w3 + 0x00, RS), W.wr(w3 + 0x18, 0x0000_0300), W.wr(w3 + 0x1C, 0x8), W.wr(w3 + 0x28, 0x100),
        // acts MM2S
        W.wr(da + 0x00, RS), W.wr(da + 0x18, 0x1000_0000), W.wr(da + 0x1C, 0x8), W.wr(da + 0x28, 0x200),
        // kernel config + start
        W.wr(kb + regmap.offsetOf("NUM_Q1_BLOCKS"), 7),
        W.wr(kb + regmap.offsetOf("NUM_ROWBLOCKS"), 2),
        W.wr(kb + regmap.offsetOf("NUM_COLS"), 1),
        W.wr(kb + regmap.offsetOf("CTRL"), CTRL_START),
        // waits: kernel done, then every mover idle (read x5, write x1)
        W.wt(kb + regmap.offsetOf("STATUS"), STATUS_DONE, STATUS_DONE),
        W.wt(w0 + 0x04, IDLE, IDLE), W.wt(w1 + 0x04, IDLE, IDLE),
        W.wt(w2 + 0x04, IDLE, IDLE), W.wt(w3 + 0x04, IDLE, IDLE),
        W.wt(da + 0x04, IDLE, IDLE),
        W.wt(w0 + 0x34, IDLE, IDLE),
    };
    try std.testing.expectEqualSlices(E, &want, b.entries());
}

test "buildOp stream passes validation against its own ranges, fails without the result range" {
    var buf: [seq_entries_per_op]seq.Entry = undefined;
    var b = seq.Builder.init(&buf);
    const wp = [_]u64{ 0x8_0000_0000, 0x8_0000_0100, 0x8_0000_0200, 0x8_0000_0300 };
    buildOp(&b, wp, 0x100, 0x8_1000_0000, 0x200, 0x8_2000_0000, 0x40, 7, 2, 1);
    const weights = seq.Range{ .lo = 0x8_0000_0000, .hi = 0x8_0000_0400 };
    const acts = seq.Range{ .lo = 0x8_1000_0000, .hi = 0x8_1000_0200 };
    const result = seq.Range{ .lo = 0x8_2000_0000, .hi = 0x8_2000_0040 };
    try seq.validate(b.entries(), &seq_dma_bases, &.{ weights, acts, result });
    try std.testing.expectError(error.S2mmOutOfRange, seq.validate(b.entries(), &seq_dma_bases, &.{ weights, acts }));
}
