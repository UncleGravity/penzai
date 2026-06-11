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
const acts_staging_cap: usize = 256 * 1024; // q1_blocks up to ~1638 (k up to ~209k)
const result_staging_cap: usize = 8 * 1024 * 1024; // rowblocks up to 262144

pub const Error = mmio.Error || error{ HeapFailure, OutOfMemory };

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
        pub fn tryMatmul(self: *Self, heap: *Heap, mm: wire.MatmulQ1A8) Error!?profile.PlCounters {
            if (mm.weight_fmt != .w1a8) return null;
            if (mm.cols != 1) return null; // decode only, for now
            const rows: usize = mm.rows;
            const k: usize = mm.k;
            if (rows == 0 or k == 0 or k % q1a8.q1_block != 0) return null;

            const q1_blocks = k / q1a8.q1_block;
            const rowblocks = q1a8.rowblocksFor(rows);
            const weight_bytes = rowblocks * q1_blocks * q1a8.packed_per_q1_block;
            const act_stream_bytes = q1_blocks * q1a8.acts_per_q1_block;
            const result_bytes = rowblocks * (q1a8.rows_per_block / 2) * q1a8.beat_bytes;
            const dst_bytes = rows * @sizeOf(f32);

            // Shape must match the resident packing, and fit the staging windows.
            if (mm.weights.nbytes != weight_bytes) return null;
            if (mm.acts.nbytes != k * @sizeOf(f32)) return null;
            if (mm.dst.nbytes != dst_bytes) return null;
            if (act_stream_bytes > acts_staging_cap or result_bytes > result_staging_cap) return null;

            try self.ensureScratch(k);

            // 1. Read the activation column and quantize with the canonical path.
            const acts_bytes = heap.bytes(mm.acts) catch return error.HeapFailure;
            for (0..k) |i| {
                self.column[i] = @bitCast(std.mem.readInt(u32, acts_bytes[i * 4 ..][0..4], .little));
            }
            const q8_blocks = q1_blocks * q1a8.q8_subblocks;
            q1a8.quantizeQ8_0(self.column[0..k], self.quants[0..k], self.act_scales[0..q8_blocks]) catch return error.HeapFailure;

            // 2. Pack the activation stream into staging and push it to the device.
            const acts_dma = subRange(self.acts_staging, act_stream_bytes);
            const acts_staging_buf = heap.bytes(acts_dma) catch return error.HeapFailure;
            packActs(q1_blocks, self.quants[0..k], self.act_scales[0..q8_blocks], acts_staging_buf);
            heap.syncToDevice(acts_dma) catch return error.HeapFailure;

            const result_dma = subRange(self.result_staging, result_bytes);
            const weights_phys = heap.deviceAddress(mm.weights) catch return error.HeapFailure;
            const acts_phys = heap.deviceAddress(acts_dma) catch return error.HeapFailure;
            const result_phys = heap.deviceAddress(result_dma) catch return error.HeapFailure;

            // 3. Program DMAs (arm result first), run the kernel, wait.
            try self.dma_w.reset();
            try self.dma_a.resetMm2s();
            try self.dma_w.startWriteToDdr(result_phys, result_bytes);
            try self.dma_w.startReadFromDdr(weights_phys, weight_bytes);
            try self.dma_a.startReadFromDdr(acts_phys, act_stream_bytes);
            self.kernel.run(@intCast(q1_blocks), @intCast(rowblocks));
            try self.kernel.waitDone();
            try self.dma_w.waitWriteDone();

            const counters = self.readCounters();

            // 4. Pull results back and copy into the destination.
            heap.syncFromDevice(result_dma) catch return error.HeapFailure;
            const result_buf = heap.bytes(result_dma) catch return error.HeapFailure;
            const dst_buf = heap.bytes(mm.dst) catch return error.HeapFailure;
            // Single-column result stream is row-major fp32; the first dst_bytes
            // are exactly the destination (padding rows sit past dst_bytes).
            @memcpy(dst_buf[0..dst_bytes], result_buf[0..dst_bytes]);

            return counters;
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
