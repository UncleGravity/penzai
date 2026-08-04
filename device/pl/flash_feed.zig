//! Pure feed helpers for the kv-major PL flash tenant. v2 DMAs Q/K/V/mask **directly
//! from the resident ggml tensors** in their native layout — no gather, no GQA
//! replication (that was ~89% of a v1 decode call). The kernel consumes the cache
//! kv-position-major and does GQA in hardware, so all this module owns is: the masked
//! KV-extent scan (clamp the −∞ padding), the native-layout contiguity check that
//! gates the direct-DMA path, and the packed-O scatter back to the strided dst.
//!
//! Kernel stream geometry: Q/O are f32, K/V are f16. The kernel reads Q in 8×f32 beats,
//! K/V in 8×f16 beats, mask as one f16, and emits O 8×f32 per 256-bit beat (packed).

const std = @import("std");

pub const LANES: usize = 8;

pub const Shape = struct {
    head_dim_q: usize,
    head_dim_v: usize,
    n_heads: usize,
    n_head_kv: usize,
    n_kv: usize,
    n_tokens: usize,
    q_nb1: usize,
    q_nb2: usize,
    k_nb1: usize,
    k_nb2: usize,
    v_nb1: usize,
    v_nb2: usize,
    mask_nb1: usize,
    dst_nb1: usize,
    dst_nb2: usize,

    pub fn headRatio(s: Shape) usize {
        return s.n_heads / s.n_head_kv;
    }
    /// The kernel requires head dims to be a whole number of 8-lane beats, and GQA to
    /// divide evenly.
    pub fn beatAligned(s: Shape) bool {
        return s.head_dim_q % LANES == 0 and s.head_dim_v % LANES == 0 and
            s.n_heads % s.n_head_kv == 0;
    }

    /// Can the kernel consume Q/K/V straight from the resident tensors with one
    /// contiguous DMA each (the no-repack fast path)? True when this is a single-token
    /// (decode) call and the cache has no inter-{kv,head} padding — the K/V layout is
    /// `[head_dim, n_head_kv, n_kv]` packed (kv-position outer, kv-heads adjacent), Q
    /// is `[head_dim, n_heads]` packed. Bonsai GQA satisfies this; anything else (a
    /// padded/strided cache, multi-token prefill) defers to the PS oracle.
    pub fn directDmaCapable(s: Shape) bool {
        const h: usize = @sizeOf(f16);
        const f: usize = @sizeOf(f32);
        return s.n_tokens == 1 and
            s.k_nb2 == s.head_dim_q * h and
            s.k_nb1 == s.n_head_kv * s.head_dim_q * h and
            s.v_nb2 == s.head_dim_v * h and
            s.v_nb1 == s.n_head_kv * s.head_dim_v * h and
            s.q_nb2 == s.head_dim_q * f; // n_tokens=1 ⇒ heads are contiguous
    }

    // ---- direct-DMA stream byte lengths (kv_hi = the real masked extent) ----
    /// Q for the one token: all heads, contiguous (head·head_dim_q f32).
    pub fn qStreamBytes(s: Shape) usize {
        return s.n_heads * s.head_dim_q * @sizeOf(f32);
    }
    /// K for kv 0..kv_hi: a contiguous prefix of the cache (kv is the outer dim).
    pub fn kStreamBytes(s: Shape, kv_hi: usize) usize {
        return kv_hi * s.k_nb1;
    }
    pub fn vStreamBytes(s: Shape, kv_hi: usize) usize {
        return kv_hi * s.v_nb1;
    }
    /// One mask value per kv (this token), contiguous f16.
    pub fn maskStreamBytes(_: Shape, kv_hi: usize) usize {
        return kv_hi * @sizeOf(f16);
    }
    /// Packed O: 8 f32 per beat, head_dim_v/8 beats per head — i.e. the dense output.
    pub fn oBytes(s: Shape) usize {
        return s.n_tokens * s.n_heads * s.head_dim_v * @sizeOf(f32);
    }
};

/// f16 −∞ bit pattern. flash_kernel.v skips a kv whose mask is exactly this
/// (F16_NEG_INF), so the real extent is the last kv that is *not* it.
pub const f16_neg_inf: u16 = 0xFC00;

pub const MaskAnalysis = struct {
    valid_extent: usize,
    processed_extent: usize,
    valid_pairs: usize,
};

/// Analyze the mask for kernel configuration and, when requested, profiling.
/// Profiling uses a forward scan to count finite entries and find the extent in
/// one pass. The unprofiled path searches backward and stops at the first finite
/// entry because valid_pairs is not consumed. The current kernel cannot accept
/// N_KV=0, so an all-masked row retains its defensive full walk.
pub fn analyzeMask(s: Shape, mask_data: ?[]const u8, collect_pairs: bool) MaskAnalysis {
    const m = mask_data orelse return .{
        .valid_extent = s.n_kv,
        .processed_extent = s.n_kv,
        .valid_pairs = if (collect_pairs) s.n_tokens * s.n_kv else 0,
    };

    if (!collect_pairs) {
        var hi: usize = 0;
        for (0..s.n_tokens) |t| {
            var kv = s.n_kv;
            while (kv > hi) : (kv -= 1) {
                const bits = std.mem.readInt(u16, m[t * s.mask_nb1 + (kv - 1) * @sizeOf(f16) ..][0..2], .little);
                if (bits != f16_neg_inf) {
                    hi = kv;
                    break;
                }
            }
        }
        return .{
            .valid_extent = hi,
            .processed_extent = if (hi == 0) s.n_kv else hi,
            .valid_pairs = 0,
        };
    }

    var hi: usize = 0;
    var count: usize = 0;
    for (0..s.n_tokens) |t| {
        for (0..s.n_kv) |kv| {
            const bits = std.mem.readInt(u16, m[t * s.mask_nb1 + kv * @sizeOf(f16) ..][0..2], .little);
            if (bits == f16_neg_inf) continue;
            hi = @max(hi, kv + 1);
            count += 1;
        }
    }
    return .{
        .valid_extent = hi,
        .processed_extent = if (hi == 0) s.n_kv else hi,
        .valid_pairs = count,
    };
}

/// Scatter the kernel's packed O (dense f32 in (token, head, d) order — the 8-wide
/// emit, read back as a flat f32 array) into the strided destination:
/// dst[head*dst_nb1 + token*dst_nb2 + d*4].
pub fn scatterO(s: Shape, o_f32: []const u8, dst: []u8) void {
    var idx: usize = 0;
    for (0..s.n_tokens) |t| for (0..s.n_heads) |h| {
        const base = h * s.dst_nb1 + t * s.dst_nb2;
        for (0..s.head_dim_v) |d| {
            @memcpy(dst[base + d * @sizeOf(f32) ..][0..4], o_f32[idx * @sizeOf(f32) ..][0..4]);
            idx += 1;
        }
    };
}

// ---------------------------------------------------------------------------- tests

fn f16bits(v: f32) u16 {
    return @bitCast(@as(f16, @floatCast(v)));
}

test "mask analysis separates finite entries from the processed extent" {
    const base: Shape = .{
        .head_dim_q = 8,
        .head_dim_v = 8,
        .n_heads = 1,
        .n_head_kv = 1,
        .n_kv = 8,
        .n_tokens = 1,
        .q_nb1 = 0,
        .q_nb2 = 0,
        .k_nb1 = 0,
        .k_nb2 = 0,
        .v_nb1 = 0,
        .v_nb2 = 0,
        .mask_nb1 = 8 * @sizeOf(f16),
        .dst_nb1 = 0,
        .dst_nb2 = 0,
    };
    // Causal decode: finite prefix [0,3), −∞ tail → extent 3.
    var row = [_]u16{ 0, 0, 0, f16_neg_inf, f16_neg_inf, f16_neg_inf, f16_neg_inf, f16_neg_inf };
    var analysis = analyzeMask(base, std.mem.sliceAsBytes(row[0..]), true);
    try std.testing.expectEqualDeep(MaskAnalysis{ .valid_extent = 3, .processed_extent = 3, .valid_pairs = 3 }, analysis);
    // A finite entry above a masked hole still bounds the extent (last real + 1).
    row = .{ 0, f16_neg_inf, 0, f16_neg_inf, f16_neg_inf, f16_neg_inf, f16_neg_inf, f16_neg_inf };
    analysis = analyzeMask(base, std.mem.sliceAsBytes(row[0..]), true);
    try std.testing.expectEqualDeep(MaskAnalysis{ .valid_extent = 3, .processed_extent = 3, .valid_pairs = 2 }, analysis);
    try std.testing.expectEqualDeep(
        MaskAnalysis{ .valid_extent = 3, .processed_extent = 3, .valid_pairs = 0 },
        analyzeMask(base, std.mem.sliceAsBytes(row[0..]), false),
    );
    // All masked → defensive fallback to the full padded extent.
    @memset(row[0..], f16_neg_inf);
    analysis = analyzeMask(base, std.mem.sliceAsBytes(row[0..]), true);
    try std.testing.expectEqualDeep(MaskAnalysis{ .valid_extent = 0, .processed_extent = 8, .valid_pairs = 0 }, analysis);
    try std.testing.expectEqualDeep(MaskAnalysis{ .valid_extent = 0, .processed_extent = 8, .valid_pairs = 0 }, analyzeMask(base, std.mem.sliceAsBytes(row[0..]), false));
    // Null mask → the whole extent is real.
    try std.testing.expectEqualDeep(MaskAnalysis{ .valid_extent = 8, .processed_extent = 8, .valid_pairs = 8 }, analyzeMask(base, null, true));
    try std.testing.expectEqualDeep(MaskAnalysis{ .valid_extent = 8, .processed_extent = 8, .valid_pairs = 0 }, analyzeMask(base, null, false));
    // Two tokens: the bound is the max real extent across rows (t0→2, t1→5).
    var two = base;
    two.n_tokens = 2;
    const rows = [_]u16{
        0, 0, f16_neg_inf, f16_neg_inf, f16_neg_inf, f16_neg_inf, f16_neg_inf, f16_neg_inf,
        0, 0, 0,           0,           0,           f16_neg_inf, f16_neg_inf, f16_neg_inf,
    };
    try std.testing.expectEqualDeep(MaskAnalysis{ .valid_extent = 5, .processed_extent = 5, .valid_pairs = 7 }, analyzeMask(two, std.mem.sliceAsBytes(rows[0..]), true));
    try std.testing.expectEqualDeep(MaskAnalysis{ .valid_extent = 5, .processed_extent = 5, .valid_pairs = 0 }, analyzeMask(two, std.mem.sliceAsBytes(rows[0..]), false));
}

test "directDmaCapable accepts the packed Bonsai decode layout, rejects padding/prefill" {
    // Bonsai decode: [head_dim, n_head_kv, n_kv] packed, Q [head_dim, n_heads] packed.
    const ok: Shape = .{
        .head_dim_q = 128,
        .head_dim_v = 128,
        .n_heads = 16,
        .n_head_kv = 8,
        .n_kv = 256,
        .n_tokens = 1,
        .q_nb1 = 128 * 4,
        .q_nb2 = 128 * 4,
        .k_nb1 = 8 * 128 * 2,
        .k_nb2 = 128 * 2,
        .v_nb1 = 8 * 128 * 2,
        .v_nb2 = 128 * 2,
        .mask_nb1 = 256 * 2,
        .dst_nb1 = 128 * 4,
        .dst_nb2 = 16 * 128 * 4,
    };
    try std.testing.expect(ok.directDmaCapable());
    var prefill = ok; // multi-token Q is strided per head → not direct
    prefill.n_tokens = 5;
    prefill.q_nb2 = 128 * 5 * 4;
    try std.testing.expect(!prefill.directDmaCapable());
    var padded = ok; // inter-kv padding in the cache → not contiguous
    padded.k_nb1 = 8 * 128 * 2 + 64;
    try std.testing.expect(!padded.directDmaCapable());
}

test "scatterO places dense f32 into the strided destination" {
    const a = std.testing.allocator;
    // head_dim_v=4, n_heads=2, n_tokens=1; dst [hd_v, n_heads, n_tokens] strided.
    const s: Shape = .{
        .head_dim_q = 8,
        .head_dim_v = 4,
        .n_heads = 2,
        .n_head_kv = 1,
        .n_kv = 1,
        .n_tokens = 1,
        .q_nb1 = 0,
        .q_nb2 = 0,
        .k_nb1 = 0,
        .k_nb2 = 0,
        .v_nb1 = 0,
        .v_nb2 = 0,
        .mask_nb1 = 0,
        .dst_nb1 = 4 * 4,
        .dst_nb2 = 2 * 4 * 4,
    };
    // packed O: dense f32 in (h, d) order; tag = h*100 + d.
    const o = try a.alloc(u8, s.oBytes());
    defer a.free(o);
    var idx: usize = 0;
    for (0..s.n_heads) |h| for (0..s.head_dim_v) |d| {
        std.mem.writeInt(u32, o[idx * 4 ..][0..4], @intCast(h * 100 + d), .little);
        idx += 1;
    };
    const dst = try a.alloc(u8, s.n_heads * s.head_dim_v * 4);
    defer a.free(dst);
    scatterO(s, o, dst);
    for (0..s.n_heads) |h| for (0..s.head_dim_v) |d| {
        const got = std.mem.readInt(u32, dst[h * s.dst_nb1 + d * 4 ..][0..4], .little);
        try std.testing.expectEqual(@as(u32, @intCast(h * 100 + d)), got);
    };
}
