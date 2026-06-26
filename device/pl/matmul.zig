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
const bus_mod = @import("bus.zig");
const seq = @import("seq.zig");
const seq_ctrl_mod = @import("seq_ctrl.zig");
const dma_mod = @import("dma.zig");
const gather = @import("gather.zig");
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

// seq.v descriptor buffer: generous headroom over one op (~47 entries × 16B ≈ 0.7 KB) so a future
// batched run (Q/K/V etc.) fits too. Read by seq.v over the coherent HPC0 port (no flush).
const seq_desc_cap: usize = 64 * 1024;

// Errors this op can raise: the DMA substrate's, this kernel's, plus heap failures.
const KernelError = regwin.Error || error{ KernelTimeout, BadId, BadVersion, BadRows, BadWeightPorts, BadClock };
pub const Error = dma_mod.Error || KernelError || error{ HeapFailure, OutOfMemory };

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
    bus: bus_mod.Bus,
    version: u32,
    clk_hz: u32,

    pub fn open(base: i64) KernelError!Kernel {
        var b = try bus_mod.Bus.mmio(@intCast(base));
        errdefer b.deinit();
        const id = b.rd(regmap.offsetOf("ID"));
        if (id != expected_id) return error.BadId;
        const version = b.rd(regmap.offsetOf("VERSION"));
        if (version < min_version) return error.BadVersion;
        if (b.rd(regmap.offsetOf("ROWS")) != expected_rows) return error.BadRows;
        if (b.rd(regmap.offsetOf("WEIGHT_PORTS")) != expected_weight_ports) return error.BadWeightPorts;
        const clk_hz = b.rd(regmap.offsetOf("CLK_HZ"));
        if (clk_hz == 0) return error.BadClock;
        return .{ .bus = b, .version = version, .clk_hz = clk_hz };
    }

    /// Record-backed: `run`/`waitDone` append descriptor entries instead of poking MMIO. The
    /// init/counter reads aren't used in record mode (the MMIO kernel already validated the
    /// bitstream at startup), so version/clk_hz are unset.
    pub fn openRecord(base: u32, rec: *seq.Recorder) Kernel {
        return .{ .bus = bus_mod.Bus.record(base, rec), .version = 0, .clk_hz = 0 };
    }

    pub fn deinit(self: *Kernel) void {
        self.bus.deinit();
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
        self.bus.wr(regmap.offsetOf("NUM_Q1_BLOCKS"), num_q1_blocks);
        self.bus.wr(regmap.offsetOf("NUM_ROWBLOCKS"), num_rowblocks);
        self.bus.wr(regmap.offsetOf("NUM_COLS"), num_cols);
        self.bus.wr(regmap.offsetOf("CTRL"), CTRL_START);
    }

    pub fn waitDone(self: *Kernel) KernelError!void {
        if (self.bus.isRecording()) {
            self.bus.recordWait(regmap.offsetOf("STATUS"), STATUS_DONE, STATUS_DONE);
            return;
        }
        var i: usize = 0;
        while (i < regwin.wait_limit) : (i += 1) {
            if (self.bus.rd(regmap.offsetOf("STATUS")) & STATUS_DONE != 0) return;
        }
        return error.KernelTimeout;
    }

    pub fn cycles(self: Kernel) u32 {
        return self.bus.rd(regmap.offsetOf("CYCLES"));
    }

    pub fn wStall(self: Kernel) u32 {
        return self.bus.rd(regmap.offsetOf("W_STALL"));
    }
    pub fn aStall(self: Kernel) u32 {
        return self.bus.rd(regmap.offsetOf("A_STALL"));
    }
    pub fn rStall(self: Kernel) u32 {
        return self.bus.rd(regmap.offsetOf("R_STALL"));
    }
    pub fn wBeats(self: Kernel) u32 {
        return self.bus.rd(regmap.offsetOf("W_BEATS"));
    }
    pub fn aBeats(self: Kernel) u32 {
        return self.bus.rd(regmap.offsetOf("A_BEATS"));
    }
    pub fn rBeats(self: Kernel) u32 {
        return self.bus.rd(regmap.offsetOf("R_BEATS"));
    }
};

/// The per-column-group register dance: reset the movers, arm the S2MM result, start the four
/// weight reads + the acts read, strobe the kernel. ONE source of truth — tryMatmul runs it on
/// MMIO dmas/kernel (silicon), the seq.v builder runs the same on record-mode dmas/kernel to emit
/// the descriptor. Split from programWaits so tryMatmul can time setup vs wait separately.
fn programSetup(
    dma_w: []dma_mod.Dma,
    dma_a: *dma_mod.Dma,
    kernel: *Kernel,
    weight_phys: []const u64,
    weight_port_bytes: usize,
    acts_phys: u64,
    act_total: usize,
    result_phys: u64,
    result_bytes: usize,
    num_q1_blocks: u32,
    num_rowblocks: u32,
    num_cols: u32,
) Error!void {
    try dma_w[0].reset();
    for (dma_w[1..]) |*dma| try dma.resetMm2s();
    try dma_a.resetMm2s();
    try dma_w[0].startWriteToDdr(result_phys, result_bytes);
    for (dma_w, 0..) |*dma, port| try dma.startReadFromDdr(weight_phys[port], weight_port_bytes);
    try dma_a.startReadFromDdr(acts_phys, act_total);
    kernel.run(num_q1_blocks, num_rowblocks, num_cols);
}

/// Wait the kernel done + every mover idle (or, on a record bus, emit the matching WAIT entries).
fn programWaits(dma_w: []dma_mod.Dma, dma_a: *dma_mod.Dma, kernel: *Kernel) Error!void {
    try kernel.waitDone();
    for (dma_w) |*dma| try dma.waitReadDone();
    try dma_a.waitReadDone();
    try dma_w[0].waitWriteDone();
}

/// One matmul column-group's addresses + shape — the variable inputs to programSetup.
pub const RunOp = struct {
    weight_phys: [layout.weight_ports]u64,
    weight_port_bytes: usize,
    acts_phys: u64,
    act_total: usize,
    result_phys: u64,
    result_bytes: usize,
    num_q1_blocks: u32,
    num_rowblocks: u32,
    num_cols: u32,
};

/// Build a seq.v descriptor for a RUN of consecutive matmul ops (the run-builder, increment 3.3).
/// The ops must be independent — no PS op between them, e.g. the Q/K/V projections off one normed
/// input (quant once, results to distinct ranges). Records each op's programSetup/programWaits into
/// `rec` (so the stream is bit-identical to what tryMatmul pokes per op), then an END marker. seq.v
/// replays the whole run with no PS in the inner loop.
pub fn recordRun(rec: *seq.Recorder, ops: []const RunOp) Error!void {
    var dma_w: [layout.weight_ports]dma_mod.Dma = undefined;
    for (&dma_w, dma_w_bases[0..]) |*d, base| d.* = dma_mod.Dma.openRecord(@intCast(base), rec);
    var dma_a = dma_mod.Dma.openRecord(@intCast(dma_a_base), rec);
    var kernel = Kernel.openRecord(@intCast(kernel_base), rec);
    for (ops) |op| {
        try programSetup(dma_w[0..], &dma_a, &kernel, op.weight_phys[0..], op.weight_port_bytes, op.acts_phys, op.act_total, op.result_phys, op.result_bytes, op.num_q1_blocks, op.num_rowblocks, op.num_cols);
        try programWaits(dma_w[0..], &dma_a, &kernel);
    }
    rec.end();
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
        // seq.v dispatch (opt-in via PENZAI_SEQ; the bitstream must carry seq_top). The per-op
        // register dance is recorded into desc_bo and replayed by seq.v in fabric instead of by the
        // PS over MMIO. desc_bo/seq_ctrl/desc_phys are valid only when seq_enabled.
        seq_enabled: bool = false,
        seq_ctrl: seq_ctrl_mod.SeqCtrl = undefined,
        desc_bo: wire.TensorRange = undefined,
        desc_phys: u64 = 0,
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

            // Opt-in seq.v dispatch: PENZAI_SEQ set + a bitstream carrying seq_top (else MMIO).
            var self_seq_enabled = false;
            var self_seq_ctrl: seq_ctrl_mod.SeqCtrl = undefined;
            var self_desc_bo: wire.TensorRange = undefined;
            var self_desc_phys: u64 = 0;
            if (std.c.getenv("PENZAI_SEQ") != null) {
                self_seq_ctrl = try seq_ctrl_mod.SeqCtrl.open(seq.ctrl.base);
                self_desc_bo = heap.allocate(seq_desc_cap, 4096) catch return error.HeapFailure;
                self_desc_phys = heap.deviceAddress(self_desc_bo) catch return error.HeapFailure;
                self_seq_enabled = true;
                std.debug.print("pl: seq.v dispatch ENABLED (desc @ 0x{x})\n", .{self_desc_phys});
            }

            return .{
                .allocator = allocator,
                .kernel = kernel,
                .dma_w = dma_w,
                .dma_a = dma_a,
                .acts_staging = acts_staging,
                .result_staging = result_staging,
                .seq_enabled = self_seq_enabled,
                .seq_ctrl = self_seq_ctrl,
                .desc_bo = self_desc_bo,
                .desc_phys = self_desc_phys,
            };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.column);
            self.allocator.free(self.quants);
            self.allocator.free(self.act_scales);
            if (self.seq_enabled) self.seq_ctrl.deinit();
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

                // 2. Program DMAs (arm result first), run the kernel, wait. Weights stream from
                // their resident range once for the whole group. The register sequence lives in
                // programSetup/programWaits — ONE source of truth, so the seq.v descriptor builder
                // (record-mode dmas/kernel) emits exactly this stream (docs/plan-seq-impl.md).
                if (self.seq_enabled) {
                    // Record this op's register dance into the descriptor BO and let seq.v replay it
                    // (1 op/run for now; batching Q/K/V into one run is the next step). seq.v reads
                    // the descriptor over coherent HPC0 — no flush.
                    const desc_bytes: []align(@alignOf(seq.Entry)) u8 =
                        @alignCast(heap.bytes(self.desc_bo) catch return error.HeapFailure);
                    var rec = seq.Recorder.init(std.mem.bytesAsSlice(seq.Entry, desc_bytes));
                    try recordRun(&rec, &.{.{
                        .weight_phys = weight_phys,
                        .weight_port_bytes = weight_port_bytes,
                        .acts_phys = acts_phys,
                        .act_total = act_total,
                        .result_phys = result_phys,
                        .result_bytes = result_bytes,
                        .num_q1_blocks = @intCast(q1_blocks),
                        .num_rowblocks = @intCast(num_rb),
                        .num_cols = @intCast(group),
                    }});
                    self.seq_ctrl.run(self.desc_phys, rec.count());
                    seg.setup_ns += lapNs(io, &last);
                    self.seq_ctrl.waitDone() catch |e| {
                        std.debug.print("pl: seq.v run failed ({s}) errIndex={d}\n", .{ @errorName(e), self.seq_ctrl.errIndex() });
                        return error.KernelTimeout;
                    };
                    seg.wait_ns += lapNs(io, &last);
                } else {
                    try programSetup(self.dma_w[0..], &self.dma_a, &self.kernel, weight_phys[0..], weight_port_bytes, acts_phys, act_total, result_phys, result_bytes, @intCast(q1_blocks), @intCast(num_rb), @intCast(group));
                    seg.setup_ns += lapNs(io, &last);
                    try programWaits(self.dma_w[0..], &self.dma_a, &self.kernel);
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

test "kernel record: run + waitDone emit the config writes + done poll" {
    var buf: [8]seq.Entry = undefined;
    var rec = seq.Recorder.init(&buf);
    const B: u32 = @intCast(kernel_base);
    var k = Kernel.openRecord(B, &rec);
    k.run(3, 5, 1);
    try k.waitDone();
    const e = rec.entries();
    try std.testing.expectEqual(@as(usize, 5), e.len);
    try std.testing.expectEqual(seq.Entry{ .tag = 0, .addr = B + regmap.offsetOf("NUM_Q1_BLOCKS"), .a = 3, .b = 0 }, e[0]);
    try std.testing.expectEqual(seq.Entry{ .tag = 0, .addr = B + regmap.offsetOf("NUM_ROWBLOCKS"), .a = 5, .b = 0 }, e[1]);
    try std.testing.expectEqual(seq.Entry{ .tag = 0, .addr = B + regmap.offsetOf("NUM_COLS"), .a = 1, .b = 0 }, e[2]);
    try std.testing.expectEqual(seq.Entry{ .tag = 0, .addr = B + regmap.offsetOf("CTRL"), .a = CTRL_START, .b = 0 }, e[3]);
    try std.testing.expectEqual(seq.Entry{ .tag = 1, .addr = B + regmap.offsetOf("STATUS"), .a = STATUS_DONE, .b = STATUS_DONE }, e[4]);
}

test "full matmul-op descriptor: programSetup/programWaits record the whole register stream" {
    const P = layout.weight_ports;
    var buf: [128]seq.Entry = undefined;
    var rec = seq.Recorder.init(&buf);
    var dma_w: [P]dma_mod.Dma = undefined;
    for (&dma_w, dma_w_bases[0..]) |*d, base| d.* = dma_mod.Dma.openRecord(@intCast(base), &rec);
    var dma_a = dma_mod.Dma.openRecord(@intCast(dma_a_base), &rec);
    var kernel = Kernel.openRecord(@intCast(kernel_base), &rec);

    var weight_phys: [P]u64 = undefined;
    for (&weight_phys, 0..) |*w, i| w.* = 0x2_0000_0000 + @as(u64, @intCast(i)) * 0x1000;
    try programSetup(dma_w[0..], &dma_a, &kernel, weight_phys[0..], 512, 0x4_0000_0000, 256, 0x8_0000_0000, 128, 7, 1, 1);
    try programWaits(dma_w[0..], &dma_a, &kernel);

    try std.testing.expect(!rec.overflow);
    // setup: dma_w0 reset(2wr+2wait) + (P-1) resetMm2s(1wr+1wait) + dma_a resetMm2s(2)
    //        + dma_w0 startWrite(4) + P*startRead(4) + dma_a startRead(4) + kernel run(4)
    // waits: kernel done(1) + P readDone + dma_a readDone(1) + dma_w0 writeDone(1)
    const setup = 4 + (P - 1) * 2 + 2 + 4 + P * 4 + 4 + 4;
    const waits = 1 + P + 1 + 1;
    try std.testing.expectEqual(@as(u32, @intCast(setup + waits)), rec.count());

    // The kernel strobe (4 writes) sits just before the wait block; the first wait is kernel-done.
    const e = rec.entries();
    const KB: u32 = @intCast(kernel_base);
    try std.testing.expectEqual(seq.Entry{ .tag = 0, .addr = KB + regmap.offsetOf("CTRL"), .a = CTRL_START, .b = 0 }, e[setup - 1]);
    try std.testing.expectEqual(seq.Entry{ .tag = 1, .addr = KB + regmap.offsetOf("STATUS"), .a = STATUS_DONE, .b = STATUS_DONE }, e[setup]);
}

test "recordRun: a multi-op run records each op's stream then one END" {
    const P = layout.weight_ports;
    var buf: [256]seq.Entry = undefined;
    var rec = seq.Recorder.init(&buf);

    var op0: RunOp = .{
        .weight_phys = undefined, .weight_port_bytes = 512, .acts_phys = 0x4_0000_0000,
        .act_total = 256, .result_phys = 0x9_0000_0000, .result_bytes = 128,
        .num_q1_blocks = 7, .num_rowblocks = 1, .num_cols = 1,
    };
    for (&op0.weight_phys, 0..) |*w, i| w.* = 0x2_0000_0000 + @as(u64, @intCast(i)) * 0x1000;
    var op1 = op0;
    op1.acts_phys = 0x4_0000_1000; // same input acts in a real Q/K/V batch; distinct result range
    op1.result_phys = 0x9_0000_1000;

    try recordRun(&rec, &.{ op0, op1 });
    try std.testing.expect(!rec.overflow);

    const setup = 4 + (P - 1) * 2 + 2 + 4 + P * 4 + 4 + 4;
    const waits = 1 + P + 1 + 1;
    const per_op = setup + waits;
    try std.testing.expectEqual(@as(u32, @intCast(2 * per_op + 1)), rec.count());

    const e = rec.entries();
    const KB: u32 = @intCast(kernel_base);
    // each op's kernel CTRL strobe is the last write of its setup block.
    try std.testing.expectEqual(seq.Entry{ .tag = 0, .addr = KB + regmap.offsetOf("CTRL"), .a = CTRL_START, .b = 0 }, e[setup - 1]);
    try std.testing.expectEqual(seq.Entry{ .tag = 0, .addr = KB + regmap.offsetOf("CTRL"), .a = CTRL_START, .b = 0 }, e[per_op + setup - 1]);
    // run ends with END.
    try std.testing.expectEqual(@as(u32, 2), e[e.len - 1].tag);
}
