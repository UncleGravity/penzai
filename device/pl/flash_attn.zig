//! PL flash-attention backend (v2, kv-major): drives the flash_top fabric kernel for a
//! wire flash_attn_f32 command. A tenant on the PL substrate (regwin/dma), beside
//! pl/matmul.zig.
//!
//! The kernel consumes the KV cache in its NATIVE layout (kv-position-major,
//! n_head_kv heads, GQA done in hardware), so this tenant **DMAs Q/K/V/mask straight
//! from the resident ggml tensors** — no gather, no GQA replication, no input staging.
//! That gather was ~89% of a v1 decode call. The mask gives the real KV extent (clamp
//! the −∞ padding); O comes back 8-wide packed and scatters (or DMAs straight) to dst.
//! Decode-only (single token, contiguous cache); anything else defers to the PS oracle.

const std = @import("std");
const shared = @import("shared");
const flash_regmap = @import("flash_regmap");
const regwin = @import("regwin.zig");
const dma_mod = @import("dma.zig");
const feed = @import("flash_feed.zig");
const profile = @import("../profile.zig");

const wire = shared.wire;
const capabilities = shared.capabilities;

const KernelError = regwin.Error || error{ KernelTimeout, BadId, BadVersion, BadLanes, BadClock };
pub const Error = dma_mod.Error || KernelError || error{ HeapFailure, OutOfMemory };

const addr = flash_regmap.addr;
const caps = flash_regmap.caps;

const CTRL_START: u32 = 1 << 0;
const STATUS_DONE: u32 = 1 << 1;

/// Oldest kernel VERSION the driver accepts. v3 retains the kv-major v2 datapath and
/// raises its query-head capacity to 32. Older bitstreams must fall back to PS rather
/// than let the driver submit shapes that alias their smaller on-chip pools.
pub const min_version: u32 = 3;
pub const expected_id: u32 = flash_regmap.resetOf("ID");
const expected_lanes: u32 = flash_regmap.resetOf("LANES");

comptime {
    std.debug.assert(min_version <= flash_regmap.resetOf("VERSION"));
}

/// Per-run hardware counters from the flash kernel (Q/K/V/O streams). Read only
/// for aggregate profiling and folded into the typed command outcome.
const FlashCounters = struct {
    cycles: u64 = 0,
    q_beats: u64 = 0,
    k_beats: u64 = 0,
    k_stall: u64 = 0,
    v_beats: u64 = 0,
    v_stall: u64 = 0,
    o_beats: u64 = 0,
    o_stall: u64 = 0,
};

/// The flash kernel AXI-Lite driver — this op's tenant on the substrate.
pub const Kernel = struct {
    win: regwin.RegWindow,
    version: u32,
    clk_hz: u32,

    pub fn open(base: i64) KernelError!Kernel {
        var win = try regwin.RegWindow.mapWindow(base);
        errdefer win.deinit();
        if (win.rd(flash_regmap.offsetOf("ID")) != expected_id) return error.BadId;
        const version = win.rd(flash_regmap.offsetOf("VERSION"));
        if (version < min_version) return error.BadVersion;
        if (win.rd(flash_regmap.offsetOf("LANES")) != expected_lanes) return error.BadLanes;
        const clk_hz = win.rd(flash_regmap.offsetOf("CLK_HZ"));
        if (clk_hz == 0) return error.BadClock;
        return .{ .win = win, .version = version, .clk_hz = clk_hz };
    }

    pub fn deinit(self: *Kernel) void {
        self.win.deinit();
    }

    pub fn run(self: *Kernel, hdq: u32, hdv: u32, nh: u32, nhkv: u32, ratio: u32, nkv: u32, ntok: u32, scale_bits: u32) void {
        self.win.wr(flash_regmap.offsetOf("HEAD_DIM_Q"), hdq);
        self.win.wr(flash_regmap.offsetOf("HEAD_DIM_V"), hdv);
        self.win.wr(flash_regmap.offsetOf("N_HEADS"), nh);
        self.win.wr(flash_regmap.offsetOf("N_HEAD_KV"), nhkv);
        self.win.wr(flash_regmap.offsetOf("HEAD_RATIO"), ratio);
        self.win.wr(flash_regmap.offsetOf("N_KV"), nkv);
        self.win.wr(flash_regmap.offsetOf("N_TOKENS"), ntok);
        self.win.wr(flash_regmap.offsetOf("SCALE"), scale_bits);
        self.win.wr(flash_regmap.offsetOf("CTRL"), CTRL_START);
    }

    pub fn waitDone(self: *Kernel) KernelError!void {
        var i: usize = 0;
        while (i < regwin.wait_limit) : (i += 1) {
            if (self.win.rd(flash_regmap.offsetOf("STATUS")) & STATUS_DONE != 0) return;
        }
        return error.KernelTimeout;
    }

    /// Fabric clock in MHz, self-described by the bitstream (CLK_HZ register).
    pub fn clkMhz(self: Kernel) f64 {
        return @as(f64, @floatFromInt(self.clk_hz)) / 1_000_000.0;
    }

    fn readCounters(self: *Kernel) FlashCounters {
        return .{
            .cycles = self.win.rd(flash_regmap.offsetOf("CYCLES")),
            .q_beats = self.win.rd(flash_regmap.offsetOf("Q_BEATS")),
            .k_beats = self.win.rd(flash_regmap.offsetOf("K_BEATS")),
            .k_stall = self.win.rd(flash_regmap.offsetOf("K_STALL")),
            .v_beats = self.win.rd(flash_regmap.offsetOf("V_BEATS")),
            .v_stall = self.win.rd(flash_regmap.offsetOf("V_STALL")),
            .o_beats = self.win.rd(flash_regmap.offsetOf("O_BEATS")),
            .o_stall = self.win.rd(flash_regmap.offsetOf("O_STALL")),
        };
    }
};

pub fn capabilityInfo(kernel: Kernel) capabilities.EngineInfo {
    return .{
        .id = expected_id,
        .version = kernel.version,
        .clock_hz = kernel.clk_hz,
        .dim0 = caps.lanes,
        .dim1 = caps.head_dim_max,
        .dim2 = @intCast(caps.max_heads),
        .dim3 = @intCast(caps.max_head_kv),
    };
}

pub fn formatMask(_: Kernel) u32 {
    return capabilities.Format.io_f32 | capabilities.Format.kv_f16;
}

/// Generic over the heap so the runtime composes it only for heaps with physical
/// addressing (XRT); the fake heap never instantiates it.
pub fn Backend(comptime Heap: type) type {
    return struct {
        const Self = @This();

        allocator: std.mem.Allocator,
        kernel: Kernel,
        dma_q: dma_mod.Dma,
        dma_k: dma_mod.Dma,
        dma_v: dma_mod.Dma,
        dma_mask: dma_mod.Dma,
        dma_o: dma_mod.Dma,
        o_staging: wire.TensorRange,
        logged_reject: bool = false,

        pub fn init(allocator: std.mem.Allocator, heap: *Heap) Error!Self {
            var kernel = try Kernel.open(addr.kernel);
            errdefer kernel.deinit();
            var dma_q = try dma_mod.Dma.open(addr.dma_q);
            errdefer dma_q.deinit();
            var dma_k = try dma_mod.Dma.open(addr.dma_k);
            errdefer dma_k.deinit();
            var dma_v = try dma_mod.Dma.open(addr.dma_v);
            errdefer dma_v.deinit();
            var dma_mask = try dma_mod.Dma.open(addr.dma_mask);
            errdefer dma_mask.deinit();
            var dma_o = try dma_mod.Dma.open(addr.dma_o);
            errdefer dma_o.deinit();

            // The only host reservation: the packed-O DMA sink (used when dst is not
            // contiguous). Q/K/V/mask DMA straight from the resident tensors.
            const o_staging = heap.allocate(caps.o_staging_bytes, 4096) catch return error.HeapFailure;

            return .{
                .allocator = allocator,
                .kernel = kernel,
                .dma_q = dma_q,
                .dma_k = dma_k,
                .dma_v = dma_v,
                .dma_mask = dma_mask,
                .dma_o = dma_o,
                .o_staging = o_staging,
            };
        }

        pub fn deinit(self: *Self) void {
            self.dma_o.deinit();
            self.dma_mask.deinit();
            self.dma_v.deinit();
            self.dma_k.deinit();
            self.dma_q.deinit();
            self.kernel.deinit();
            self.* = undefined;
        }

        pub fn hasCounters(self: *const Self) bool {
            _ = self;
            return true;
        }

        fn shapeOf(attn: wire.FlashAttnF32) feed.Shape {
            return .{
                .head_dim_q = attn.head_dim_q,
                .head_dim_v = attn.head_dim_v,
                .n_heads = attn.n_heads,
                .n_head_kv = attn.n_head_kv,
                .n_kv = attn.n_kv,
                .n_tokens = attn.n_tokens,
                .q_nb1 = @intCast(attn.q_nb1),
                .q_nb2 = @intCast(attn.q_nb2),
                .k_nb1 = @intCast(attn.k_nb1),
                .k_nb2 = @intCast(attn.k_nb2),
                .v_nb1 = @intCast(attn.v_nb1),
                .v_nb2 = @intCast(attn.v_nb2),
                .mask_nb1 = @intCast(attn.mask_nb1),
                .dst_nb1 = @intCast(attn.dst_nb1),
                .dst_nb2 = @intCast(attn.dst_nb2),
            };
        }

        /// Run `attn` on the PL if its shape is supported. Detailed timing and
        /// counter reads are structurally absent when `ctx` is null.
        pub fn tryFlashAttn(self: *Self, heap: *Heap, attn: wire.FlashAttnF32, ctx: ?*profile.ProfileContext) Error!?profile.FlashExecution {
            var s = shapeOf(attn);
            // Unsupported shape → defer to PS (not an error).
            if (s.head_dim_q == 0 or s.head_dim_v == 0 or s.n_heads == 0 or s.n_head_kv == 0 or
                s.n_kv == 0 or s.n_tokens == 0) return null;
            if (!attn.has_mask) return null; // the mask gives kv_hi and feeds the kernel's skip
            if (!s.beatAligned()) return null;
            if (s.head_dim_q > caps.head_dim_max or s.head_dim_v > caps.head_dim_max) return null;
            if (s.n_heads > caps.max_heads or s.n_head_kv > caps.max_head_kv) return null;
            if (!s.directDmaCapable()) {
                // One-shot bring-up diagnostic: if a decode op falls back to PS, this
                // says which stride assumption missed (no `flash seg` line ⇒ PS path).
                if (ctx != null and !self.logged_reject) {
                    self.logged_reject = true;
                    const h: usize = @sizeOf(f16);
                    const f: usize = @sizeOf(f32);
                    std.debug.print("pl flash: directDmaCapable rejected → PS (n_tok={d} k_nb1={d}/exp{d} k_nb2={d}/exp{d} v_nb1={d}/exp{d} v_nb2={d}/exp{d} q_nb2={d}/exp{d})\n", .{
                        s.n_tokens,
                        s.k_nb1,
                        s.n_head_kv * s.head_dim_q * h,
                        s.k_nb2,
                        s.head_dim_q * h,
                        s.v_nb1,
                        s.n_head_kv * s.head_dim_v * h,
                        s.v_nb2,
                        s.head_dim_v * h,
                        s.q_nb2,
                        s.head_dim_q * f,
                    });
                }
                return null; // native packed cache + single token only
            }
            if (s.oBytes() > caps.o_staging_bytes) return null;

            const wrapper_start = profile.begin(ctx);
            var last = wrapper_start;
            var result = profile.FlashExecution{
                .path = .direct,
                .kernel_runs = 1,
                .requested_n_kv = attn.n_kv,
                .requested_qkv_pairs = @as(u64, attn.n_tokens) *| @as(u64, attn.n_kv),
            };

            // Real KV extent from the mask → clamp the −∞ padding (shrinks the K/V DMA
            // and the kernel walk; bit-identical, the dropped tail is what the kernel
            // already skips). Then size each native stream.
            const mask_data = heap.read(attn.mask) catch return error.HeapFailure;
            const mask_analysis = feed.analyzeMask(s, mask_data, ctx != null);
            const valid_kv_hi = mask_analysis.valid_extent;
            const kv_hi = mask_analysis.processed_extent;
            s.n_kv = kv_hi;
            const q_bytes = s.qStreamBytes();
            const k_bytes = s.kStreamBytes(kv_hi);
            const v_bytes = s.vStreamBytes(kv_hi);
            const mask_bytes = s.maskStreamBytes(kv_hi);
            const o_bytes = s.oBytes();
            // O straight to dst when it's contiguous f32 (the common decode case, one
            // token, heads adjacent), else into the packed staging + scatter.
            const dst_contig = s.dst_nb1 == s.head_dim_v * @sizeOf(f32);
            if (!dst_contig) result.path = .staged;
            const o_range = if (dst_contig) srcRange(attn.dst, o_bytes) else subRange(self.o_staging, o_bytes);
            result.valid_n_kv = @intCast(valid_kv_hi);
            result.processed_n_kv = @intCast(kv_hi);
            result.valid_qkv_pairs = @intCast(mask_analysis.valid_pairs);
            result.processed_qkv_pairs = @as(u64, attn.n_tokens) *| @as(u64, kv_hi);
            result.q_bytes = q_bytes;
            result.k_bytes = k_bytes;
            result.v_bytes = v_bytes;
            result.mask_bytes = mask_bytes;
            result.o_bytes = o_bytes;
            result.prepare_ns +|= profile.lap(ctx, &last);

            // Flush the A53's writes to the resident sources so the PL DMA reads current
            // DDR (the KV cache was just written by set_rows; the mask was uploaded).
            const q_src = srcRange(attn.q, q_bytes);
            const k_src = srcRange(attn.k, k_bytes);
            const v_src = srcRange(attn.v, v_bytes);
            const mask_src = srcRange(attn.mask, mask_bytes);
            heap.syncToDevice(q_src) catch return error.HeapFailure;
            heap.syncToDevice(k_src) catch return error.HeapFailure;
            heap.syncToDevice(v_src) catch return error.HeapFailure;
            heap.syncToDevice(mask_src) catch return error.HeapFailure;
            result.sync_to_ns +|= profile.lap(ctx, &last);

            const q_phys = heap.deviceAddress(q_src) catch return error.HeapFailure;
            const k_phys = heap.deviceAddress(k_src) catch return error.HeapFailure;
            const v_phys = heap.deviceAddress(v_src) catch return error.HeapFailure;
            const mask_phys = heap.deviceAddress(mask_src) catch return error.HeapFailure;
            const o_phys = heap.deviceAddress(o_range) catch return error.HeapFailure;

            // Arm the output capture first, then the input feeds, then strobe start.
            try self.dma_o.resetS2mm();
            try self.dma_q.resetMm2s();
            try self.dma_k.resetMm2s();
            try self.dma_v.resetMm2s();
            try self.dma_mask.resetMm2s();
            try self.dma_o.startWriteToDdr(o_phys, o_bytes);
            try self.dma_q.startReadFromDdr(q_phys, q_bytes);
            try self.dma_k.startReadFromDdr(k_phys, k_bytes);
            try self.dma_v.startReadFromDdr(v_phys, v_bytes);
            try self.dma_mask.startReadFromDdr(mask_phys, mask_bytes);

            self.kernel.run(attn.head_dim_q, attn.head_dim_v, attn.n_heads, attn.n_head_kv, @intCast(s.headRatio()), @intCast(kv_hi), attn.n_tokens, @bitCast(attn.scale));
            result.setup_ns +|= profile.lap(ctx, &last);
            try self.kernel.waitDone();
            try self.dma_q.waitReadDone();
            try self.dma_k.waitReadDone();
            try self.dma_v.waitReadDone();
            try self.dma_mask.waitReadDone();
            try self.dma_o.waitWriteDone();
            result.wait_ns +|= profile.lap(ctx, &last);

            const counters = if (ctx != null) self.kernel.readCounters() else FlashCounters{};

            // Make the DMA's O writes visible to the A53, then (only if dst wasn't the
            // DMA target) scatter the packed result into the strided destination.
            heap.syncFromDevice(o_range) catch return error.HeapFailure;
            result.sync_from_ns +|= profile.lap(ctx, &last);
            if (!dst_contig) {
                const o_packed = heap.read(o_range) catch return error.HeapFailure;
                const dst = heap.bytes(attn.dst) catch return error.HeapFailure;
                feed.scatterO(s, o_packed, dst);
                result.result_layout_ns +|= profile.lap(ctx, &last);
            }
            if (ctx) |active| result.wrapper_ns = shared.profiling.elapsed(wrapper_start, active.now());
            result.cycles = counters.cycles;
            result.q_beats = counters.q_beats;
            result.k_beats = counters.k_beats;
            result.k_stall_cycles = counters.k_stall;
            result.v_beats = counters.v_beats;
            result.v_stall_cycles = counters.v_stall;
            result.o_beats = counters.o_beats;
            result.o_stall_cycles = counters.o_stall;
            return result;
        }
    };
}

fn subRange(staging: wire.TensorRange, nbytes: usize) wire.TensorRange {
    return .{ .handle = staging.handle, .offset = 0, .nbytes = nbytes };
}

/// A prefix of a resident source tensor (same handle/offset, fewer bytes) — what the
/// direct DMA reads and what we flush before it.
fn srcRange(base: wire.TensorRange, nbytes: usize) wire.TensorRange {
    return .{ .handle = base.handle, .offset = base.offset, .nbytes = nbytes };
}
