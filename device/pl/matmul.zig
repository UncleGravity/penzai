//! PL Q1A8 matmul backend: drives the fabric kernel for a wire matmul command.
//!
//! Weights are streamed straight from their resident XRT BO range. Activations
//! are quantized with the *canonical* quantizer (`shared.q1a8.quantizeQ8_0`, the
//! same round-nearest-even the PS oracle uses, so PL and PS feed bit-identical
//! int8), packed into a staging region, and DMA'd in. Results DMA into a staging
//! region and copy into the destination range.
//!
//! Scope (narrow-first, against the loaded v4 bitstream): single-column decode
//! matmuls only. Anything else returns null so the runtime falls back to PS.

const std = @import("std");
const shared = @import("shared");
const mmio = @import("mmio.zig");
const gather = @import("gather.zig");
const profile = @import("../profile.zig");

const wire = shared.wire;
const q1a8 = shared.q1a8;

// AXI-Lite base addresses, matching the loaded bitstream's address map
// (experiments/kr260-q1a8-matmul-bringup/fpga/build.tcl + src/config.zig).
const dma_w_base: i64 = 0xA000_0000; // weights MM2S + results S2MM
const dma_a_base: i64 = 0xA001_0000; // acts MM2S
const kernel_base: i64 = 0xA002_0000; // kernel AXI-Lite

// Staging capacities, reserved once from the heap at init. Generous vs any real
// shape; a matmul that would exceed them falls back to PS.
const acts_staging_cap: usize = 256 * 1024; // COLS_MAX columns of acts
const result_staging_cap: usize = 8 * 1024 * 1024; // num_rb * COLS_MAX * 32

// Columns multiplied per kernel run. Must match q1a8_kernel_mc COLS_MAX in the
// loaded bitstream. Prefill matmuls (cols>1) are tiled into groups of this many.
const mc_cols_max: usize = 8;
const result_bytes_per_rb = gather.result_bytes_per_rb; // 32; single source in gather.zig

// Fabric clock of the loaded bitstream (variant w256-f125). Must match the
// deployed bitstream; used only to convert the cycle counter to time for the
// `pl seg` diagnostic. TODO: expose as a CLK_MHZ regmap register so penzaid
// self-describes the clock (as it already does VERSION) and the v7 ~300 MHz
// bump can't silently re-introduce the drift this replaced.
const pl_clk_mhz: f64 = 125.0;

pub const Error = mmio.Error || error{ HeapFailure, OutOfMemory };

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
        kernel: mmio.Kernel,
        dma_w: mmio.Dma,
        dma_a: mmio.Dma,
        acts_staging: wire.TensorRange,
        result_staging: wire.TensorRange,
        // Reusable per-call scratch, grown on demand (steady state: no realloc).
        column: []f32 = &.{},
        quants: []i8 = &.{},
        act_scales: []f16 = &.{},
        seg: Seg = .{},

        pub fn init(allocator: std.mem.Allocator, heap: *Heap) Error!Self {
            var kernel = try mmio.Kernel.open(kernel_base);
            errdefer kernel.deinit();
            var dma_w = try mmio.Dma.open(dma_w_base);
            errdefer dma_w.deinit();
            var dma_a = try mmio.Dma.open(dma_a_base);
            errdefer dma_a.deinit();

            const acts_staging = heap.allocate(acts_staging_cap, 4096) catch return error.HeapFailure;
            const result_staging = heap.allocate(result_staging_cap, 4096) catch return error.HeapFailure;

            return .{
                .allocator = allocator,
                .kernel = kernel,
                .dma_w = dma_w,
                .dma_a = dma_a,
                .acts_staging = acts_staging,
                .result_staging = result_staging,
            };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.column);
            self.allocator.free(self.quants);
            self.allocator.free(self.act_scales);
            self.dma_a.deinit();
            self.dma_w.deinit();
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
            if (rows == 0 or cols == 0 or k == 0 or k % q1a8.q1_block != 0) return null;

            const q1_blocks = k / q1a8.q1_block;
            const num_rb = q1a8.rowblocksFor(rows);
            const weight_bytes = num_rb * q1_blocks * q1a8.packed_per_q1_block; // wide layout
            const act_stream_bytes = q1_blocks * q1a8.acts_per_q1_block; // per column
            const dst_bytes = rows * cols * @sizeOf(f32);

            // Shapes must match the resident packing and the staging windows must
            // hold one COLS_MAX group of acts and results.
            if (mm.weights.nbytes != weight_bytes) return null;
            if (mm.acts.nbytes != k * cols * @sizeOf(f32)) return null;
            if (mm.dst.nbytes != dst_bytes) return null;
            if (mc_cols_max * act_stream_bytes > acts_staging_cap) return null;
            if (num_rb * mc_cols_max * result_bytes_per_rb > result_staging_cap) return null;

            try self.ensureScratch(k);
            const q8_blocks = q1_blocks * q1a8.q8_subblocks;
            const weights_phys = heap.deviceAddress(mm.weights) catch return error.HeapFailure;
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
                const direct_result = group == 1 and rows % q1a8.rows_per_block == 0;
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
                    q1a8.quantizeQ8_0Simd(self.column[0..k], self.quants[0..k], self.act_scales[0..q8_blocks]) catch return error.HeapFailure;
                    packActs(q1_blocks, self.quants[0..k], self.act_scales[0..q8_blocks], acts_staging_buf[c * act_stream_bytes ..][0..act_stream_bytes]);
                }
                seg.quant_ns += lapNs(io, &last);
                heap.syncToDevice(acts_dma) catch return error.HeapFailure;
                seg.sync_to_ns += lapNs(io, &last);

                const acts_phys = heap.deviceAddress(acts_dma) catch return error.HeapFailure;
                const result_phys = heap.deviceAddress(result_dma) catch return error.HeapFailure;

                // 2. Program DMAs (arm result first), run the kernel, wait. Weights
                // stream from their resident range once for the whole group.
                try self.dma_w.reset();
                try self.dma_a.resetMm2s();
                try self.dma_w.startWriteToDdr(result_phys, result_bytes);
                try self.dma_w.startReadFromDdr(weights_phys, weight_bytes);
                try self.dma_a.startReadFromDdr(acts_phys, act_total);
                self.kernel.run(@intCast(q1_blocks), @intCast(num_rb), @intCast(group));
                seg.setup_ns += lapNs(io, &last);
                try self.kernel.waitDone();
                try self.dma_w.waitWriteDone();
                seg.wait_ns += lapNs(io, &last);

                accumulateCounters(&counters, self.readCounters());

                // 3. Gather results: stream is [rowblock][col][row], 8 fp32/rb
                // lane-major; place into the col-major destination, dropping pad rows.
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
                @as(f64, @floatFromInt(s.cycles)) / n / pl_clk_mhz, // cycles -> us
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
            self.act_scales = self.allocator.alloc(f16, k / q1a8.q8_block) catch return error.OutOfMemory;
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
        for (0..q1a8.q8_subblocks) |sub| {
            const q8 = blk * q1a8.q8_subblocks + sub;
            const acts = quants[q8 * q1a8.q8_block ..][0..q1a8.q8_block];
            for (0..q1a8.q8_block / q1a8.beat_bytes) |beat| {
                var word: u64 = 0;
                for (0..q1a8.beat_bytes) |byte| {
                    const u: u64 = @as(u8, @bitCast(acts[beat * q1a8.beat_bytes + byte]));
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
    var quants: [q1_blocks * q1a8.q1_block]i8 = undefined;
    for (&quants, 0..) |*q, i| q.* = @intCast(@as(i32, @intCast(i % 251)) - 125);
    var scales: [q1_blocks * q1a8.q8_subblocks]f16 = undefined;
    for (&scales, 0..) |*s, i| s.* = @floatCast(@as(f32, @floatFromInt(i + 1)) * 0.01);
    var out: [q1_blocks * q1a8.acts_per_q1_block]u8 = undefined;
    packActs(q1_blocks, &quants, &scales, &out);
    // Last scale beat carries the final sub-block's fp16 scale in its low 16 bits.
    const last = std.mem.readInt(u64, out[out.len - 8 ..][0..8], .little);
    try std.testing.expectEqual(@as(u16, @bitCast(scales[scales.len - 1])), @as(u16, @truncate(last)));
}
