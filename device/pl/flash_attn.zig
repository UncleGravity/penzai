//! PL flash-attention backend: drives the flash_top fabric kernel for a wire
//! flash_attn_f32 command. A second tenant on the PL substrate (regwin/dma), beside
//! pl/matmul.zig.
//!
//! Q/K/V/mask are gathered (flash_feed) from their ggml-strided ranges into the
//! kernel's contiguous consumption order — GQA replicated, kv-major within a head —
//! DMA'd in; O DMAs out and scatters back to the strided destination. v1 materializes
//! the (GQA-replicated) streams into staging, so it supports the shapes that fit the
//! reservations and otherwise returns null to defer to the PS oracle. The kernel is
//! correctness-first (sequential); the feed and the kernel both get faster later.

const std = @import("std");
const shared = @import("shared");
const flash_regmap = @import("flash_regmap");
const regwin = @import("regwin.zig");
const dma_mod = @import("dma.zig");
const feed = @import("flash_feed.zig");

const wire = shared.wire;

const KernelError = regwin.Error || error{ KernelTimeout, BadId, BadVersion, BadLanes, BadClock };
pub const Error = dma_mod.Error || KernelError || error{ HeapFailure, OutOfMemory };

const addr = flash_regmap.addr;
const caps = flash_regmap.caps;

const CTRL_START: u32 = 1 << 0;
const STATUS_DONE: u32 = 1 << 1;

const min_version: u32 = 1;
const expected_id: u32 = flash_regmap.resetOf("ID");
const expected_lanes: u32 = flash_regmap.resetOf("LANES");

/// Per-run hardware counters from the flash kernel (Q/K/V/O streams). Returned to the
/// runtime; profiler wiring is a later step.
pub const FlashCounters = struct {
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

    pub fn run(self: *Kernel, hdq: u32, hdv: u32, nh: u32, nkv: u32, ntok: u32, scale_bits: u32) void {
        self.win.wr(flash_regmap.offsetOf("HEAD_DIM_Q"), hdq);
        self.win.wr(flash_regmap.offsetOf("HEAD_DIM_V"), hdv);
        self.win.wr(flash_regmap.offsetOf("N_HEADS"), nh);
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
        k_staging: wire.TensorRange,
        v_staging: wire.TensorRange,
        mask_staging: wire.TensorRange,
        o_staging: wire.TensorRange,

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

            const q_staging = heap.allocate(caps.q_staging_bytes, 4096) catch return error.HeapFailure;
            const k_staging = heap.allocate(caps.k_staging_bytes, 4096) catch return error.HeapFailure;
            const v_staging = heap.allocate(caps.v_staging_bytes, 4096) catch return error.HeapFailure;
            const mask_staging = heap.allocate(caps.mask_staging_bytes, 4096) catch return error.HeapFailure;
            const o_staging = heap.allocate(caps.o_staging_bytes, 4096) catch return error.HeapFailure;

            return .{
                .allocator = allocator,
                .kernel = kernel,
                .dma_q = dma_q,
                .dma_k = dma_k,
                .dma_v = dma_v,
                .dma_mask = dma_mask,
                .dma_o = dma_o,
                .q_staging = q_staging,
                .k_staging = k_staging,
                .v_staging = v_staging,
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

        /// Run `attn` on the PL if its shape is supported; null defers to the PS
        /// kernel. Errors are hardware/heap failures the runtime should surface.
        pub fn tryFlashAttn(self: *Self, heap: *Heap, attn: wire.FlashAttnF32, io: ?std.Io) Error!?FlashCounters {
            _ = io;
            const s = shapeOf(attn);
            // Unsupported shape → defer to PS (not an error).
            if (s.head_dim_q == 0 or s.head_dim_v == 0 or s.n_heads == 0 or s.n_head_kv == 0 or
                s.n_kv == 0 or s.n_tokens == 0) return null;
            if (!s.beatAligned()) return null;
            if (s.head_dim_q > caps.head_dim_max or s.head_dim_v > caps.head_dim_max) return null;
            if (s.qBytes() > caps.q_staging_bytes or s.kBytes() > caps.k_staging_bytes or
                s.vBytes() > caps.v_staging_bytes or s.maskBytes() > caps.mask_staging_bytes or
                s.oBytes() > caps.o_staging_bytes) return null;

            const q_data = heap.read(attn.q) catch return error.HeapFailure;
            const k_data = heap.read(attn.k) catch return error.HeapFailure;
            const v_data = heap.read(attn.v) catch return error.HeapFailure;
            const mask_data: ?[]const u8 = if (attn.has_mask)
                (heap.read(attn.mask) catch return error.HeapFailure)
            else
                null;

            // Gather into the staging buffers (the kernel's consumption order).
            const q_sub = subRange(self.q_staging, s.qBytes());
            const k_sub = subRange(self.k_staging, s.kBytes());
            const v_sub = subRange(self.v_staging, s.vBytes());
            const mask_sub = subRange(self.mask_staging, s.maskBytes());
            const o_sub = subRange(self.o_staging, s.oBytes());
            feed.gatherQ(s, q_data, heap.bytes(q_sub) catch return error.HeapFailure);
            feed.gatherK(s, k_data, heap.bytes(k_sub) catch return error.HeapFailure);
            feed.gatherV(s, v_data, heap.bytes(v_sub) catch return error.HeapFailure);
            feed.gatherMask(s, mask_data, heap.bytes(mask_sub) catch return error.HeapFailure);
            heap.syncToDevice(q_sub) catch return error.HeapFailure;
            heap.syncToDevice(k_sub) catch return error.HeapFailure;
            heap.syncToDevice(v_sub) catch return error.HeapFailure;
            heap.syncToDevice(mask_sub) catch return error.HeapFailure;

            const q_phys = heap.deviceAddress(q_sub) catch return error.HeapFailure;
            const k_phys = heap.deviceAddress(k_sub) catch return error.HeapFailure;
            const v_phys = heap.deviceAddress(v_sub) catch return error.HeapFailure;
            const mask_phys = heap.deviceAddress(mask_sub) catch return error.HeapFailure;
            const o_phys = heap.deviceAddress(o_sub) catch return error.HeapFailure;

            // Arm the output capture first, then the input feeds, then strobe start.
            try self.dma_o.resetS2mm();
            try self.dma_q.resetMm2s();
            try self.dma_k.resetMm2s();
            try self.dma_v.resetMm2s();
            try self.dma_mask.resetMm2s();
            try self.dma_o.startWriteToDdr(o_phys, s.oBytes());
            try self.dma_q.startReadFromDdr(q_phys, s.qBytes());
            try self.dma_k.startReadFromDdr(k_phys, s.kBytes());
            try self.dma_v.startReadFromDdr(v_phys, s.vBytes());
            try self.dma_mask.startReadFromDdr(mask_phys, s.maskBytes());

            self.kernel.run(attn.head_dim_q, attn.head_dim_v, attn.n_heads, attn.n_kv, attn.n_tokens, @bitCast(attn.scale));
            try self.kernel.waitDone();
            try self.dma_q.waitReadDone();
            try self.dma_k.waitReadDone();
            try self.dma_v.waitReadDone();
            try self.dma_mask.waitReadDone();
            try self.dma_o.waitWriteDone();

            const counters = self.kernel.readCounters();

            heap.syncFromDevice(o_sub) catch return error.HeapFailure;
            const o_beats = heap.read(o_sub) catch return error.HeapFailure;
            const dst = heap.bytes(attn.dst) catch return error.HeapFailure;
            feed.scatterO(s, o_beats, dst);

            return counters;
        }
    };
}

fn subRange(staging: wire.TensorRange, nbytes: usize) wire.TensorRange {
    return .{ .handle = staging.handle, .offset = 0, .nbytes = nbytes };
}
