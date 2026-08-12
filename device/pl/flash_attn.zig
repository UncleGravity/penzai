//! PL flash-attention backend (v4, kv-major): drives the flash_top fabric kernel for a
//! wire flash_attn_f32 command. A tenant on the PL substrate (regwin/dma), beside
//! pl/matmul.zig.
//!
//! Decode retains v3's direct Q/K/V/mask path. For multi-token commands, v4 gathers
//! Q and transposes the mask in tiles of at most four queries while K/V remain direct
//! and are each streamed only once per tile. Unsupported layouts defer to the PS oracle
//! before any DMA starts.

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
pub const Error = dma_mod.Error || KernelError || feed.FeedError || error{ HeapFailure, OutOfMemory };

const addr = flash_regmap.addr;
const caps = flash_regmap.caps;

const CTRL_START: u32 = 1 << 0;
const STATUS_DONE: u32 = 1 << 1;

/// Oldest kernel VERSION the driver accepts. v3 retains the kv-major v2 datapath and
/// raises its query-head capacity to 32. Older bitstreams must fall back to PS rather
/// than let the driver submit shapes that alias their smaller on-chip pools.
pub const min_version: u32 = 3;
pub const query_blocked_version: u32 = 4;
pub const expected_id: u32 = flash_regmap.resetOf("ID");
const expected_lanes: u32 = flash_regmap.resetOf("LANES");

comptime {
    std.debug.assert(min_version <= flash_regmap.resetOf("VERSION"));
    std.debug.assert(feed.QUERY_TILE_MAX == caps.query_tile_max);
    std.debug.assert(feed.CONTEXT_MAX == caps.context_max);
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
        q_staging: wire.TensorRange,
        mask_staging: wire.TensorRange,
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

            // Decode uses the resident Q/mask ranges directly. Query-blocked prefill
            // gathers bounded Q/mask tiles into these persistent DMA sources.
            const q_staging = heap.allocate(caps.q_staging_bytes, 4096) catch return error.HeapFailure;
            errdefer heap.free(q_staging.handle) catch {};
            const mask_staging = heap.allocate(caps.mask_staging_bytes, 4096) catch return error.HeapFailure;
            errdefer heap.free(mask_staging.handle) catch {};
            const o_staging = heap.allocate(caps.o_staging_bytes, 4096) catch return error.HeapFailure;
            errdefer heap.free(o_staging.handle) catch {};

            return .{
                .allocator = allocator,
                .kernel = kernel,
                .dma_q = dma_q,
                .dma_k = dma_k,
                .dma_v = dma_v,
                .dma_mask = dma_mask,
                .dma_o = dma_o,
                .q_staging = q_staging,
                .mask_staging = mask_staging,
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

        fn shapeOf(attn: wire.FlashAttnF32) ?feed.Shape {
            return .{
                .head_dim_q = std.math.cast(usize, attn.head_dim_q) orelse return null,
                .head_dim_v = std.math.cast(usize, attn.head_dim_v) orelse return null,
                .n_heads = std.math.cast(usize, attn.n_heads) orelse return null,
                .n_head_kv = std.math.cast(usize, attn.n_head_kv) orelse return null,
                .n_kv = std.math.cast(usize, attn.n_kv) orelse return null,
                .n_tokens = std.math.cast(usize, attn.n_tokens) orelse return null,
                .q_nb1 = std.math.cast(usize, attn.q_nb1) orelse return null,
                .q_nb2 = std.math.cast(usize, attn.q_nb2) orelse return null,
                .k_nb1 = std.math.cast(usize, attn.k_nb1) orelse return null,
                .k_nb2 = std.math.cast(usize, attn.k_nb2) orelse return null,
                .v_nb1 = std.math.cast(usize, attn.v_nb1) orelse return null,
                .v_nb2 = std.math.cast(usize, attn.v_nb2) orelse return null,
                .mask_nb1 = std.math.cast(usize, attn.mask_nb1) orelse return null,
                .dst_nb1 = std.math.cast(usize, attn.dst_nb1) orelse return null,
                .dst_nb2 = std.math.cast(usize, attn.dst_nb2) orelse return null,
            };
        }

        /// Run `attn` on the PL if its shape is supported. Detailed timing and
        /// counter reads are structurally absent when `ctx` is null.
        pub fn tryFlashAttn(self: *Self, heap: *Heap, attn: wire.FlashAttnF32, ctx: ?*profile.ProfileContext) Error!?profile.FlashExecution {
            const s = shapeOf(attn) orelse return null;
            // Unsupported shape → defer to PS (not an error).
            if (s.head_dim_q == 0 or s.head_dim_v == 0 or s.n_heads == 0 or s.n_head_kv == 0 or
                s.n_kv == 0 or s.n_tokens == 0) return null;
            if (!attn.has_mask) return null; // the mask gives kv_hi and feeds the kernel's skip
            if (!s.beatAligned()) return null;
            if (!std.math.isFinite(attn.scale)) return null;
            if (s.head_dim_q > caps.head_dim_max or s.head_dim_v > caps.head_dim_max) return null;
            if (s.n_heads > caps.max_heads or s.n_head_kv > caps.max_head_kv) return null;
            if (s.n_kv > caps.context_max) return null;
            const spans = feed.requiredSpans(s, true) orelse return null;
            if (!rangesCover(attn, spans)) return null;

            if (s.n_tokens == 1 and s.directDmaCapable())
                return try self.runDirect(heap, attn, s, ctx);
            if (queryBlockedSupported(self.kernel.version, s)) {
                return try self.runQueryBlocked(heap, attn, s, ctx);
            }

            if (ctx != null and !self.logged_reject) {
                self.logged_reject = true;
                std.debug.print("pl flash: unsupported v{d} shape/layout -> PS (n_tok={d} n_kv={d} q_nb1/2={d}/{d} k_nb1/2={d}/{d} mask_nb1={d} dst_nb1/2={d}/{d})\n", .{
                    self.kernel.version,
                    s.n_tokens,
                    s.n_kv,
                    s.q_nb1,
                    s.q_nb2,
                    s.k_nb1,
                    s.k_nb2,
                    s.mask_nb1,
                    s.dst_nb1,
                    s.dst_nb2,
                });
            }
            return null;
        }

        fn runDirect(self: *Self, heap: *Heap, attn: wire.FlashAttnF32, original: feed.Shape, ctx: ?*profile.ProfileContext) Error!profile.FlashExecution {
            var s = original;
            if (s.oBytes() > caps.o_staging_bytes) return error.InvalidLength;

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
            result.setup_ns +|= profile.lap(ctx, &last);

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

        fn runQueryBlocked(self: *Self, heap: *Heap, attn: wire.FlashAttnF32, s: feed.Shape, ctx: ?*profile.ProfileContext) Error!profile.FlashExecution {
            const wrapper_start = profile.begin(ctx);
            var last = wrapper_start;
            var result = profile.FlashExecution{
                .path = .staged,
                .kernel_runs = @intCast(s.tileCount()),
                .requested_n_kv = attn.n_kv,
                .requested_qkv_pairs = @as(u64, attn.n_tokens) *| @as(u64, attn.n_kv),
            };

            const q_data = heap.read(attn.q) catch return error.HeapFailure;
            const mask_data = heap.read(attn.mask) catch return error.HeapFailure;

            // Determine the largest K/V prefix once, then flush that resident prefix
            // once. Individual tiles may submit shorter DMA transfers.
            const max_processed_extent = maxProcessedExtent(s, mask_data);
            result.prepare_ns +|= profile.lap(ctx, &last);

            const k_sync = srcRange(attn.k, s.kStreamBytes(max_processed_extent));
            const v_sync = srcRange(attn.v, s.vStreamBytes(max_processed_extent));
            heap.syncToDevice(k_sync) catch return error.HeapFailure;
            heap.syncToDevice(v_sync) catch return error.HeapFailure;
            result.sync_to_ns +|= profile.lap(ctx, &last);

            var tile_index: usize = 0;
            while (s.queryTile(tile_index)) |tile| : (tile_index += 1) {
                const q_bytes = s.qTileBytes(tile);
                const o_bytes = s.oTileBytes(tile);
                const q_range = subRange(self.q_staging, q_bytes);
                const q_staged = heap.bytes(q_range) catch return error.HeapFailure;
                _ = try feed.packQTile(s, q_data, tile, q_staged);

                // Pack the full mask tile to derive its extent, then submit only the
                // KV prefix the kernel will actually walk.
                const full_mask_bytes = s.maskTileStreamBytes(tile, s.n_kv);
                const full_mask_range = subRange(self.mask_staging, full_mask_bytes);
                const mask_staged = heap.bytes(full_mask_range) catch return error.HeapFailure;
                const mask_analysis = try feed.packMaskTile(s, mask_data, tile, mask_staged, ctx != null);
                const kv_hi = mask_analysis.processed_extent;
                const mask_bytes = s.maskTileStreamBytes(tile, kv_hi);
                const mask_range = subRange(self.mask_staging, mask_bytes);
                const k_bytes = s.kStreamBytes(kv_hi);
                const v_bytes = s.vStreamBytes(kv_hi);
                const k_range = srcRange(attn.k, k_bytes);
                const v_range = srcRange(attn.v, v_bytes);
                const o_range = offsetRange(attn.dst, tile.token_start * s.dst_nb2, o_bytes);

                result.valid_n_kv = @max(result.valid_n_kv, @as(u32, @intCast(mask_analysis.valid_extent)));
                result.processed_n_kv = @max(result.processed_n_kv, @as(u32, @intCast(kv_hi)));
                result.valid_qkv_pairs +|= @intCast(mask_analysis.valid_pairs);
                result.processed_qkv_pairs +|= @as(u64, @intCast(tile.token_count)) *| @as(u64, @intCast(kv_hi));
                result.q_bytes +|= q_bytes;
                result.k_bytes +|= k_bytes;
                result.v_bytes +|= v_bytes;
                result.mask_bytes +|= mask_bytes;
                result.o_bytes +|= o_bytes;
                result.prepare_ns +|= profile.lap(ctx, &last);

                heap.syncToDevice(q_range) catch return error.HeapFailure;
                heap.syncToDevice(mask_range) catch return error.HeapFailure;
                result.sync_to_ns +|= profile.lap(ctx, &last);

                const q_phys = heap.deviceAddress(q_range) catch return error.HeapFailure;
                const k_phys = heap.deviceAddress(k_range) catch return error.HeapFailure;
                const v_phys = heap.deviceAddress(v_range) catch return error.HeapFailure;
                const mask_phys = heap.deviceAddress(mask_range) catch return error.HeapFailure;
                const o_phys = heap.deviceAddress(o_range) catch return error.HeapFailure;

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
                self.kernel.run(
                    attn.head_dim_q,
                    attn.head_dim_v,
                    attn.n_heads,
                    attn.n_head_kv,
                    @intCast(s.headRatio()),
                    @intCast(kv_hi),
                    @intCast(tile.token_count),
                    @bitCast(attn.scale),
                );
                result.setup_ns +|= profile.lap(ctx, &last);

                try self.kernel.waitDone();
                try self.dma_q.waitReadDone();
                try self.dma_k.waitReadDone();
                try self.dma_v.waitReadDone();
                try self.dma_mask.waitReadDone();
                try self.dma_o.waitWriteDone();
                result.wait_ns +|= profile.lap(ctx, &last);

                if (ctx != null) addCounters(&result, self.kernel.readCounters());
                result.setup_ns +|= profile.lap(ctx, &last);

                heap.syncFromDevice(o_range) catch return error.HeapFailure;
                result.sync_from_ns +|= profile.lap(ctx, &last);
            }

            if (ctx) |active| result.wrapper_ns = shared.profiling.elapsed(wrapper_start, active.now());
            return result;
        }
    };
}

fn rangesCover(attn: wire.FlashAttnF32, spans: feed.RequiredSpans) bool {
    return rangeCovers(attn.q, spans.q) and rangeCovers(attn.k, spans.k) and
        rangeCovers(attn.v, spans.v) and rangeCovers(attn.mask, spans.mask) and
        rangeCovers(attn.dst, spans.dst);
}

fn queryBlockedSupported(version: u32, shape: feed.Shape) bool {
    // Kernel versions are cumulative within this engine ID. A future incompatible
    // stream contract must use a new ID rather than silently reinterpreting v4.
    return version >= query_blocked_version and
        shape.queryBlockedDmaCapable() and shape.packedDst();
}

fn maxProcessedExtent(shape: feed.Shape, mask_data: []const u8) usize {
    var max_extent: usize = 0;
    var tile_index: usize = 0;
    while (shape.queryTile(tile_index)) |tile| : (tile_index += 1) {
        var tile_shape = shape;
        tile_shape.n_tokens = tile.token_count;
        const mask_offset = tile.token_start * shape.mask_nb1;
        max_extent = @max(max_extent, feed.analyzeMask(
            tile_shape,
            mask_data[mask_offset..],
            false,
        ).processed_extent);
    }
    return max_extent;
}

fn rangeCovers(range: wire.TensorRange, required: usize) bool {
    if (required > range.nbytes) return false;
    _ = std.math.add(u64, range.offset, range.nbytes) catch return false;
    return true;
}

fn addCounters(result: *profile.FlashExecution, counters: FlashCounters) void {
    result.cycles +|= counters.cycles;
    result.q_beats +|= counters.q_beats;
    result.k_beats +|= counters.k_beats;
    result.k_stall_cycles +|= counters.k_stall;
    result.v_beats +|= counters.v_beats;
    result.v_stall_cycles +|= counters.v_stall;
    result.o_beats +|= counters.o_beats;
    result.o_stall_cycles +|= counters.o_stall;
}

fn subRange(staging: wire.TensorRange, nbytes: usize) wire.TensorRange {
    return .{ .handle = staging.handle, .offset = 0, .nbytes = nbytes };
}

/// A prefix of a resident source tensor (same handle/offset, fewer bytes) — what the
/// direct DMA reads and what we flush before it.
fn srcRange(base: wire.TensorRange, nbytes: usize) wire.TensorRange {
    return .{ .handle = base.handle, .offset = base.offset, .nbytes = nbytes };
}

fn offsetRange(base: wire.TensorRange, relative_offset: usize, nbytes: usize) wire.TensorRange {
    return .{
        .handle = base.handle,
        .offset = base.offset + relative_offset,
        .nbytes = nbytes,
    };
}

test "range preflight checks declared lengths and offset overflow" {
    const enough = wire.TensorRange{ .handle = 1, .offset = 64, .nbytes = 128 };
    try std.testing.expect(rangeCovers(enough, 128));
    try std.testing.expect(!rangeCovers(enough, 129));
    const overflow = wire.TensorRange{ .handle = 1, .offset = std.math.maxInt(u64) - 7, .nbytes = 8 };
    try std.testing.expect(!rangeCovers(overflow, 1));
}

test "query blocking requires v4 while v3 remains decode-only" {
    const shape = feed.Shape{
        .head_dim_q = 128,
        .head_dim_v = 128,
        .n_heads = 16,
        .n_head_kv = 8,
        .n_kv = 256,
        .n_tokens = 4,
        .q_nb1 = 128 * @sizeOf(f32),
        .q_nb2 = 4 * 128 * @sizeOf(f32),
        .k_nb1 = 8 * 128 * @sizeOf(f16),
        .k_nb2 = 128 * @sizeOf(f16),
        .v_nb1 = 8 * 128 * @sizeOf(f16),
        .v_nb2 = 128 * @sizeOf(f16),
        .mask_nb1 = 256 * @sizeOf(f16),
        .dst_nb1 = 128 * @sizeOf(f32),
        .dst_nb2 = 16 * 128 * @sizeOf(f32),
    };
    try std.testing.expect(!queryBlockedSupported(3, shape));
    try std.testing.expect(queryBlockedSupported(4, shape));
}

test "K V sync extent covers an all-masked tile" {
    const s = feed.Shape{
        .head_dim_q = 8,
        .head_dim_v = 8,
        .n_heads = 2,
        .n_head_kv = 1,
        .n_kv = 8,
        .n_tokens = 8,
        .q_nb1 = 8 * @sizeOf(f32),
        .q_nb2 = 8 * 8 * @sizeOf(f32),
        .k_nb1 = 8 * @sizeOf(f16),
        .k_nb2 = 8 * @sizeOf(f16),
        .v_nb1 = 8 * @sizeOf(f16),
        .v_nb2 = 8 * @sizeOf(f16),
        .mask_nb1 = 8 * @sizeOf(f16),
        .dst_nb1 = 8 * @sizeOf(f32),
        .dst_nb2 = 2 * 8 * @sizeOf(f32),
    };
    var mask = [_]u16{feed.f16_neg_inf} ** (8 * 8);
    // The first tile has a short finite prefix; the second is entirely masked and
    // therefore performs the kernel's defensive full walk.
    mask[0] = 0;
    try std.testing.expectEqual(s.n_kv, maxProcessedExtent(s, std.mem.sliceAsBytes(mask[0..])));
}

test "flash counters aggregate across query tiles" {
    var execution = profile.FlashExecution{ .path = .staged };
    addCounters(&execution, .{ .cycles = 100, .q_beats = 8, .k_beats = 16, .k_stall = 1, .v_beats = 16, .v_stall = 2, .o_beats = 8, .o_stall = 3 });
    addCounters(&execution, .{ .cycles = 200, .q_beats = 4, .k_beats = 16, .v_beats = 16, .o_beats = 4 });
    try std.testing.expectEqual(@as(u64, 300), execution.cycles);
    try std.testing.expectEqual(@as(u64, 12), execution.q_beats);
    try std.testing.expectEqual(@as(u64, 32), execution.k_beats);
    try std.testing.expectEqual(@as(u64, 32), execution.v_beats);
    try std.testing.expectEqual(@as(u64, 12), execution.o_beats);
    try std.testing.expectEqual(@as(u64, 1), execution.k_stall_cycles);
    try std.testing.expectEqual(@as(u64, 2), execution.v_stall_cycles);
    try std.testing.expectEqual(@as(u64, 3), execution.o_stall_cycles);
}
