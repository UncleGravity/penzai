//! Pure feed helpers for the kv-major PL flash tenant. Decode DMAs Q/K/V/mask
//! directly from the resident ggml tensors. Query-blocked prefill retains direct
//! K/V DMA, but gathers one bounded Q tile and transposes its row-major mask into
//! the kernel's KV-major stream order.
//!
//! Kernel stream geometry: Q/O are f32, K/V are f16. The kernel reads Q in 8×f32 beats,
//! K/V in 8×f16 beats, mask as one f16, and emits O 8×f32 per 256-bit beat (packed).

const std = @import("std");
const shared = @import("shared");

pub const LANES: usize = 8;
pub const QUERY_TILE_MAX: usize = shared.section.query_tile_max;
pub const CONTEXT_MAX: usize = shared.section.context_max;

pub const FeedError = error{ InvalidShape, InvalidLength };

pub const QueryTile = struct {
    token_start: usize,
    token_count: usize,
};

pub const RequiredSpans = struct {
    q: usize,
    k: usize,
    v: usize,
    mask: usize,
    dst: usize,
};

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

    /// Initial v4 query-blocked eligibility is deliberately exact: accept the
    /// layouts emitted by the supported ggml graph and let every padded or unusual
    /// view retain the PS implementation. Q is head-major/token-inner in resident
    /// memory and is gathered into query-major order by `packQTile`; K/V already
    /// have the KV-major physical order consumed by the kernel.
    pub fn queryBlockedDmaCapable(s: Shape) bool {
        if (s.n_tokens < 2 or s.n_tokens > CONTEXT_MAX or s.n_kv > CONTEXT_MAX) return false;
        const h = @sizeOf(f16);
        const f = @sizeOf(f32);
        const q_row = checkedMul(s.head_dim_q, f) orelse return false;
        const q_head = checkedMul(s.n_tokens, q_row) orelse return false;
        const k_row = checkedMul(s.head_dim_q, h) orelse return false;
        const k_kv = checkedMul(s.n_head_kv, k_row) orelse return false;
        const v_row = checkedMul(s.head_dim_v, h) orelse return false;
        const v_kv = checkedMul(s.n_head_kv, v_row) orelse return false;
        const mask_row = checkedMul(s.n_kv, h) orelse return false;
        return s.q_nb1 == q_row and s.q_nb2 == q_head and
            s.k_nb2 == k_row and s.k_nb1 == k_kv and
            s.v_nb2 == v_row and s.v_nb1 == v_kv and
            s.mask_nb1 == mask_row;
    }

    pub fn packedDst(s: Shape) bool {
        const row = checkedMul(s.head_dim_v, @sizeOf(f32)) orelse return false;
        const token = checkedMul(s.n_heads, row) orelse return false;
        return s.dst_nb1 == row and s.dst_nb2 == token;
    }

    pub fn tileCount(s: Shape) usize {
        return std.math.divCeil(usize, s.n_tokens, QUERY_TILE_MAX) catch unreachable;
    }

    pub fn queryTile(s: Shape, index: usize) ?QueryTile {
        const start = checkedMul(index, QUERY_TILE_MAX) orelse return null;
        if (start >= s.n_tokens) return null;
        return .{ .token_start = start, .token_count = @min(QUERY_TILE_MAX, s.n_tokens - start) };
    }

    // ---- direct-DMA stream byte lengths (kv_hi = the real masked extent) ----
    /// Q for the one token: all heads, contiguous (head·head_dim_q f32).
    pub fn qStreamBytes(s: Shape) usize {
        return s.n_tokens * s.n_heads * s.head_dim_q * @sizeOf(f32);
    }
    pub fn qTileBytes(s: Shape, tile: QueryTile) usize {
        return tile.token_count * s.n_heads * s.head_dim_q * @sizeOf(f32);
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
    pub fn maskTileStreamBytes(_: Shape, tile: QueryTile, kv_hi: usize) usize {
        return tile.token_count * kv_hi * @sizeOf(f16);
    }
    /// Packed O: 8 f32 per beat, head_dim_v/8 beats per head — i.e. the dense output.
    pub fn oBytes(s: Shape) usize {
        return s.n_tokens * s.n_heads * s.head_dim_v * @sizeOf(f32);
    }
    pub fn oTileBytes(s: Shape, tile: QueryTile) usize {
        return tile.token_count * s.n_heads * s.head_dim_v * @sizeOf(f32);
    }
};

/// Validate all strided tensor spans before any helper indexes a mapped range.
/// Returning null means the PL path must decline before starting a DMA; the PS
/// implementation remains responsible for reporting malformed commands.
pub fn requiredSpans(s: Shape, has_mask: bool) ?RequiredSpans {
    if (s.head_dim_q == 0 or s.head_dim_v == 0 or s.n_heads == 0 or
        s.n_head_kv == 0 or s.n_kv == 0 or s.n_tokens == 0) return null;
    const q_row = checkedMul(s.head_dim_q, @sizeOf(f32)) orelse return null;
    const k_row = checkedMul(s.head_dim_q, @sizeOf(f16)) orelse return null;
    const v_row = checkedMul(s.head_dim_v, @sizeOf(f16)) orelse return null;
    const dst_row = checkedMul(s.head_dim_v, @sizeOf(f32)) orelse return null;
    const mask_row = checkedMul(s.n_kv, @sizeOf(f16)) orelse return null;
    return .{
        .q = stridedSpan(q_row, s.n_tokens, s.n_heads, s.q_nb1, s.q_nb2) orelse return null,
        .k = stridedSpan(k_row, s.n_kv, s.n_head_kv, s.k_nb1, s.k_nb2) orelse return null,
        .v = stridedSpan(v_row, s.n_kv, s.n_head_kv, s.v_nb1, s.v_nb2) orelse return null,
        .mask = if (has_mask) stridedSpan(mask_row, s.n_tokens, 1, s.mask_nb1, mask_row) orelse return null else 0,
        .dst = dstSpan(s, dst_row) orelse return null,
    };
}

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

/// Gather one Q tile from ggml's packed `[head][token][dimension]` layout into the
/// kernel stream order `[token][head][dimension]`.
pub fn packQTile(s: Shape, q_data: []const u8, tile: QueryTile, out: []u8) FeedError!usize {
    if (tile.token_count == 0 or tile.token_count > QUERY_TILE_MAX or
        tile.token_start + tile.token_count > s.n_tokens) return error.InvalidShape;
    const spans = requiredSpans(s, true) orelse return error.InvalidShape;
    if (q_data.len < spans.q) return error.InvalidLength;
    const row_bytes = checkedMul(s.head_dim_q, @sizeOf(f32)) orelse return error.InvalidShape;
    const want = s.qTileBytes(tile);
    if (out.len < want) return error.InvalidLength;

    var cursor: usize = 0;
    for (0..tile.token_count) |local_token| {
        const token = tile.token_start + local_token;
        for (0..s.n_heads) |head| {
            const src = head * s.q_nb2 + token * s.q_nb1;
            @memcpy(out[cursor..][0..row_bytes], q_data[src..][0..row_bytes]);
            cursor += row_bytes;
        }
    }
    return cursor;
}

/// Transpose one F16 mask tile from resident `[query][kv]` rows into the v4
/// stream order `[kv][query]`, while deriving the exact tile extent and optional
/// valid-pair count in the same pass. The full tile is staged so the caller can
/// DMA only the prefix ending at `processed_extent`.
pub fn packMaskTile(
    s: Shape,
    mask_data: []const u8,
    tile: QueryTile,
    out: []u8,
    collect_pairs: bool,
) FeedError!MaskAnalysis {
    if (tile.token_count == 0 or tile.token_count > QUERY_TILE_MAX or
        tile.token_start + tile.token_count > s.n_tokens) return error.InvalidShape;
    const spans = requiredSpans(s, true) orelse return error.InvalidShape;
    if (mask_data.len < spans.mask) return error.InvalidLength;
    const want = checkedMul(checkedMul(tile.token_count, s.n_kv) orelse return error.InvalidShape, @sizeOf(f16)) orelse return error.InvalidShape;
    if (out.len < want) return error.InvalidLength;

    var hi: usize = 0;
    var count: usize = 0;
    for (0..s.n_kv) |kv| {
        for (0..tile.token_count) |local_token| {
            const token = tile.token_start + local_token;
            const src = token * s.mask_nb1 + kv * @sizeOf(f16);
            const dst = (kv * tile.token_count + local_token) * @sizeOf(f16);
            const bits = std.mem.readInt(u16, mask_data[src..][0..2], .little);
            std.mem.writeInt(u16, out[dst..][0..2], bits, .little);
            if (bits != f16_neg_inf) {
                hi = kv + 1;
                if (collect_pairs) count += 1;
            }
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

pub fn scatterOTile(s: Shape, tile: QueryTile, o_f32: []const u8, dst: []u8) FeedError!void {
    if (tile.token_count == 0 or tile.token_count > QUERY_TILE_MAX or
        tile.token_start + tile.token_count > s.n_tokens) return error.InvalidShape;
    const dst_row = checkedMul(s.head_dim_v, @sizeOf(f32)) orelse return error.InvalidShape;
    const dst_span = dstSpan(s, dst_row) orelse return error.InvalidShape;
    const want = s.oTileBytes(tile);
    if (o_f32.len < want or dst.len < dst_span) return error.InvalidLength;
    var idx: usize = 0;
    for (tile.token_start..tile.token_start + tile.token_count) |t| for (0..s.n_heads) |h| {
        const base = h * s.dst_nb1 + t * s.dst_nb2;
        for (0..s.head_dim_v) |d| {
            @memcpy(dst[base + d * @sizeOf(f32) ..][0..4], o_f32[idx * @sizeOf(f32) ..][0..4]);
            idx += 1;
        }
    };
}

fn stridedSpan(row_bytes: usize, ne1: usize, ne2: usize, nb1: usize, nb2: usize) ?usize {
    if (row_bytes == 0 or ne1 == 0 or ne2 == 0 or nb1 < row_bytes or nb2 < row_bytes) return null;
    var span = row_bytes;
    span = checkedAdd(span, checkedMul(ne1 - 1, nb1) orelse return null) orelse return null;
    span = checkedAdd(span, checkedMul(ne2 - 1, nb2) orelse return null) orelse return null;
    return span;
}

fn dstSpan(s: Shape, row_bytes: usize) ?usize {
    return stridedSpan(row_bytes, s.n_heads, s.n_tokens, s.dst_nb1, s.dst_nb2);
}

fn checkedAdd(a: usize, b: usize) ?usize {
    return std.math.add(usize, a, b) catch null;
}

fn checkedMul(a: usize, b: usize) ?usize {
    return std.math.mul(usize, a, b) catch null;
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

fn blockedTestShape(n_tokens: usize, n_kv: usize) Shape {
    const hdq: usize = 8;
    const hdv: usize = 8;
    const heads: usize = 2;
    const kv_heads: usize = 1;
    return .{
        .head_dim_q = hdq,
        .head_dim_v = hdv,
        .n_heads = heads,
        .n_head_kv = kv_heads,
        .n_kv = n_kv,
        .n_tokens = n_tokens,
        .q_nb1 = hdq * @sizeOf(f32),
        .q_nb2 = n_tokens * hdq * @sizeOf(f32),
        .k_nb1 = kv_heads * hdq * @sizeOf(f16),
        .k_nb2 = hdq * @sizeOf(f16),
        .v_nb1 = kv_heads * hdv * @sizeOf(f16),
        .v_nb2 = hdv * @sizeOf(f16),
        .mask_nb1 = n_kv * @sizeOf(f16),
        .dst_nb1 = hdv * @sizeOf(f32),
        .dst_nb2 = heads * hdv * @sizeOf(f32),
    };
}

test "query-blocked eligibility and tile plan are bounded and exact" {
    var s = blockedTestShape(16, 256);
    try std.testing.expect(s.queryBlockedDmaCapable());
    try std.testing.expect(s.packedDst());
    try std.testing.expectEqual(@as(usize, 4), s.tileCount());
    try std.testing.expectEqualDeep(QueryTile{ .token_start = 0, .token_count = 4 }, s.queryTile(0).?);
    try std.testing.expectEqualDeep(QueryTile{ .token_start = 12, .token_count = 4 }, s.queryTile(3).?);
    try std.testing.expect(s.queryTile(4) == null);

    s.n_tokens = 5;
    s.q_nb2 = s.n_tokens * s.head_dim_q * @sizeOf(f32);
    try std.testing.expectEqual(@as(usize, 2), s.tileCount());
    try std.testing.expectEqualDeep(QueryTile{ .token_start = 4, .token_count = 1 }, s.queryTile(1).?);

    var padded = s;
    padded.q_nb2 += 64;
    try std.testing.expect(!padded.queryBlockedDmaCapable());
    var wrong_mask = s;
    wrong_mask.mask_nb1 += 2;
    try std.testing.expect(!wrong_mask.queryBlockedDmaCapable());
    var too_long = s;
    too_long.n_kv = CONTEXT_MAX + 1;
    try std.testing.expect(!too_long.queryBlockedDmaCapable());

    const cases = [_]struct { tokens: usize, tiles: usize }{
        .{ .tokens = 1, .tiles = 1 },
        .{ .tokens = 2, .tiles = 1 },
        .{ .tokens = 4, .tiles = 1 },
        .{ .tokens = 5, .tiles = 2 },
        .{ .tokens = 16, .tiles = 4 },
    };
    for (cases) |case| {
        const shape = blockedTestShape(case.tokens, 256);
        try std.testing.expectEqual(case.tiles, shape.tileCount());
    }
}

test "K and V bytes are independent of query count within a tile" {
    const base = blockedTestShape(4, 256);
    for (1..QUERY_TILE_MAX + 1) |queries| {
        var s = base;
        s.n_tokens = queries;
        try std.testing.expectEqual(@as(usize, 256 * s.n_head_kv * s.head_dim_q * @sizeOf(f16)), s.kStreamBytes(256));
        try std.testing.expectEqual(@as(usize, 256 * s.n_head_kv * s.head_dim_v * @sizeOf(f16)), s.vStreamBytes(256));
    }
}

test "Q gather converts head-major source into query-major tile stream" {
    const s = blockedTestShape(5, 6);
    const spans = requiredSpans(s, true).?;
    const q = try std.testing.allocator.alloc(u8, spans.q);
    defer std.testing.allocator.free(q);
    @memset(q, 0);
    const row_bytes = s.head_dim_q * @sizeOf(f32);
    for (0..s.n_heads) |head| for (0..s.n_tokens) |token| {
        @memset(q[head * s.q_nb2 + token * s.q_nb1 ..][0..row_bytes], @intCast(head * 16 + token));
    };

    const tile = s.queryTile(1).?;
    var staged: [QUERY_TILE_MAX * 2 * 8 * @sizeOf(f32)]u8 = undefined;
    const n = try packQTile(s, q, tile, &staged);
    try std.testing.expectEqual(s.qTileBytes(tile), n);
    // Final partial tile contains token 4, then heads 0 and 1.
    try std.testing.expectEqual(@as(u8, 4), staged[0]);
    try std.testing.expectEqual(@as(u8, 20), staged[row_bytes]);
}

test "mask packing is KV-major and preserves causal, holes, and all-masked rows" {
    const s = blockedTestShape(5, 6);
    var mask: [5 * 6]u16 = undefined;
    for (0..s.n_tokens) |token| for (0..s.n_kv) |kv| {
        mask[token * s.n_kv + kv] = if (kv <= token) @intCast(token * 16 + kv) else f16_neg_inf;
    };
    // An interior hole must not shorten the extent.
    mask[3 * s.n_kv + 1] = f16_neg_inf;
    var staged: [QUERY_TILE_MAX * 6]u16 = undefined;
    const tile = QueryTile{ .token_start = 1, .token_count = 4 };
    const analysis = try packMaskTile(s, std.mem.sliceAsBytes(mask[0..]), tile, std.mem.sliceAsBytes(staged[0..]), true);
    try std.testing.expectEqual(@as(usize, 5), analysis.valid_extent);
    try std.testing.expectEqual(@as(usize, 5), analysis.processed_extent);
    try std.testing.expectEqual(@as(usize, 13), analysis.valid_pairs);
    for (0..s.n_kv) |kv| for (0..tile.token_count) |local| {
        try std.testing.expectEqual(mask[(tile.token_start + local) * s.n_kv + kv], staged[kv * tile.token_count + local]);
    };

    @memset(mask[0..], f16_neg_inf);
    const all_masked = try packMaskTile(s, std.mem.sliceAsBytes(mask[0..]), tile, std.mem.sliceAsBytes(staged[0..]), true);
    try std.testing.expectEqualDeep(MaskAnalysis{ .valid_extent = 0, .processed_extent = 6, .valid_pairs = 0 }, all_masked);
}

test "required spans reject overflow and short staging buffers" {
    var s = blockedTestShape(5, 6);
    const spans = requiredSpans(s, true).?;
    const q = try std.testing.allocator.alloc(u8, spans.q);
    defer std.testing.allocator.free(q);
    var short: [1]u8 = undefined;
    try std.testing.expectError(error.InvalidLength, packQTile(s, q, s.queryTile(0).?, &short));

    s.q_nb2 = std.math.maxInt(usize);
    try std.testing.expect(requiredSpans(s, true) == null);
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

test "scatterOTile applies the token base for a partial tile" {
    var s = blockedTestShape(5, 1);
    s.dst_nb1 += 8;
    s.dst_nb2 = s.n_heads * s.dst_nb1 + 16;
    const spans = requiredSpans(s, true).?;
    const tile = QueryTile{ .token_start = 4, .token_count = 1 };
    const staged = try std.testing.allocator.alloc(u8, s.oTileBytes(tile));
    defer std.testing.allocator.free(staged);
    const dst = try std.testing.allocator.alloc(u8, spans.dst);
    defer std.testing.allocator.free(dst);
    @memset(dst, 0xA5);
    for (0..s.n_heads) |head| for (0..s.head_dim_v) |d| {
        const index = head * s.head_dim_v + d;
        std.mem.writeInt(u32, staged[index * 4 ..][0..4], @intCast(1000 + index), .little);
    };
    try scatterOTile(s, tile, staged, dst);
    for (0..s.n_heads) |head| for (0..s.head_dim_v) |d| {
        const got = std.mem.readInt(u32, dst[4 * s.dst_nb2 + head * s.dst_nb1 + d * 4 ..][0..4], .little);
        try std.testing.expectEqual(@as(u32, @intCast(1000 + head * s.head_dim_v + d)), got);
    };
}
