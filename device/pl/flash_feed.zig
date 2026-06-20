//! Pure feed preparation for the PL flash tenant: gather the ggml-strided Q/K/V/mask
//! tensors into the contiguous order the flash kernel consumes them (per token, per
//! head: a Q row, then per kv a mask + a K row + a V row), and scatter the kernel's O
//! beats back into the strided destination. GQA lives here — the kernel is GQA-
//! agnostic, so the K/V rows are replicated per query head sharing a kv head. This is
//! the substantive, board-independent logic, tested in isolation; the DMA + register
//! orchestration around it (flash_attn.zig) is mechanical and mirrors pl/matmul.zig.
//!
//! Kernel stream geometry: Q/O are f32, K/V are f16; the kernel reads Q in 8×f32
//! beats, K/V in 8×f16 beats, mask as one f16, and emits O one f32 per 256-bit beat
//! (lane 0). `head_dim_q`/`head_dim_v` must be multiples of 8 (the LANES width).

const std = @import("std");

pub const LANES: usize = 8;
pub const o_beat_bytes: usize = 32; // 256-bit O beat; the f32 sits in lane 0

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
    /// The kernel requires head dims to be a whole number of 8-lane beats.
    pub fn beatAligned(s: Shape) bool {
        return s.head_dim_q % LANES == 0 and s.head_dim_v % LANES == 0 and
            s.n_heads % s.n_head_kv == 0;
    }
    pub fn qBytes(s: Shape) usize {
        return s.n_tokens * s.n_heads * s.head_dim_q * @sizeOf(f32);
    }
    pub fn kBytes(s: Shape) usize {
        return s.n_tokens * s.n_heads * s.n_kv * s.head_dim_q * @sizeOf(f16);
    }
    pub fn vBytes(s: Shape) usize {
        return s.n_tokens * s.n_heads * s.n_kv * s.head_dim_v * @sizeOf(f16);
    }
    pub fn maskBytes(s: Shape) usize {
        return s.n_tokens * s.n_heads * s.n_kv * @sizeOf(f16);
    }
    pub fn oBytes(s: Shape) usize {
        return s.n_tokens * s.n_heads * s.head_dim_v * o_beat_bytes;
    }
};

/// Q row per (token, head), contiguous (head_dim_q f32 each), in (token, head) order.
pub fn gatherQ(s: Shape, q_data: []const u8, out: []u8) void {
    const row = s.head_dim_q * @sizeOf(f32);
    var off: usize = 0;
    for (0..s.n_tokens) |t| for (0..s.n_heads) |h| {
        const base = t * s.q_nb1 + h * s.q_nb2;
        @memcpy(out[off..][0..row], q_data[base..][0..row]);
        off += row;
    };
}

/// K row per (token, head, kv): K[kv][kv_head(h)], replicated per query head (GQA).
pub fn gatherK(s: Shape, k_data: []const u8, out: []u8) void {
    const row = s.head_dim_q * @sizeOf(f16);
    const ratio = s.headRatio();
    var off: usize = 0;
    for (0..s.n_tokens) |_| for (0..s.n_heads) |h| {
        const kv_head = h / ratio;
        for (0..s.n_kv) |kv| {
            const base = kv * s.k_nb1 + kv_head * s.k_nb2;
            @memcpy(out[off..][0..row], k_data[base..][0..row]);
            off += row;
        }
    };
}

/// V row per (token, head, kv): V[kv][kv_head(h)], replicated per query head (GQA).
pub fn gatherV(s: Shape, v_data: []const u8, out: []u8) void {
    const row = s.head_dim_v * @sizeOf(f16);
    const ratio = s.headRatio();
    var off: usize = 0;
    for (0..s.n_tokens) |_| for (0..s.n_heads) |h| {
        const kv_head = h / ratio;
        for (0..s.n_kv) |kv| {
            const base = kv * s.v_nb1 + kv_head * s.v_nb2;
            @memcpy(out[off..][0..row], v_data[base..][0..row]);
            off += row;
        }
    };
}

/// Mask f16 per (token, head, kv): mask[token][kv], replicated per head. Null mask →
/// all-zero (finite, so the kernel processes every kv).
pub fn gatherMask(s: Shape, mask_data: ?[]const u8, out: []u8) void {
    var off: usize = 0;
    for (0..s.n_tokens) |t| for (0..s.n_heads) |_| {
        for (0..s.n_kv) |kv| {
            const bits: u16 = if (mask_data) |m|
                std.mem.readInt(u16, m[t * s.mask_nb1 + kv * @sizeOf(f16) ..][0..2], .little)
            else
                0;
            std.mem.writeInt(u16, out[off..][0..2], bits, .little);
            off += @sizeOf(f16);
        }
    };
}

/// Scatter the kernel's O beats (one f32 in lane 0 of each 256-bit beat, emitted in
/// (token, head, d) order) into the strided destination: dst[head*dst_nb1 +
/// token*dst_nb2 + d*4].
pub fn scatterO(s: Shape, o_beats: []const u8, dst: []u8) void {
    var idx: usize = 0;
    for (0..s.n_tokens) |t| for (0..s.n_heads) |h| {
        const base = h * s.dst_nb1 + t * s.dst_nb2;
        for (0..s.head_dim_v) |d| {
            @memcpy(dst[base + d * @sizeOf(f32) ..][0..4], o_beats[idx * o_beat_bytes ..][0..4]);
            idx += 1;
        }
    };
}

// ---------------------------------------------------------------------------- tests

fn f16bits(v: f32) u16 {
    return @bitCast(@as(f16, @floatCast(v)));
}

test "gatherQ lays rows out in (token, head) order" {
    const a = std.testing.allocator;
    // head_dim_q=8, n_heads=2, n_tokens=2; ggml layout [hd, n_tok, n_heads].
    const s: Shape = .{
        .head_dim_q = 8, .head_dim_v = 8, .n_heads = 2, .n_head_kv = 1,
        .n_kv = 1, .n_tokens = 2,
        .q_nb1 = 8 * 4, .q_nb2 = 2 * 8 * 4, // nb1 = stride/token, nb2 = stride/head
        .k_nb1 = 0, .k_nb2 = 0, .v_nb1 = 0, .v_nb2 = 0, .mask_nb1 = 0, .dst_nb1 = 0, .dst_nb2 = 0,
    };
    const q = try a.alloc(u8, s.n_tokens * s.n_heads * s.head_dim_q * 4);
    defer a.free(q);
    // q[(token*nh+head)*hd + d] layout via the strides; mark each f32 with a tag.
    for (0..s.n_tokens) |t| for (0..s.n_heads) |h| for (0..s.head_dim_q) |d| {
        const off = t * s.q_nb1 + h * s.q_nb2 + d * 4;
        std.mem.writeInt(u32, q[off..][0..4], @intCast((t * 10 + h) * 100 + d), .little);
    };
    const out = try a.alloc(u8, s.qBytes());
    defer a.free(out);
    gatherQ(s, q, out);
    // out is contiguous (t,h,d). Check a few.
    var i: usize = 0;
    for (0..s.n_tokens) |t| for (0..s.n_heads) |h| for (0..s.head_dim_q) |d| {
        const got = std.mem.readInt(u32, out[i * 4 ..][0..4], .little);
        try std.testing.expectEqual(@as(u32, @intCast((t * 10 + h) * 100 + d)), got);
        i += 1;
    };
}

test "gatherK/V replicate the shared kv head across query heads (GQA)" {
    const a = std.testing.allocator;
    // n_heads=4, n_head_kv=2 -> ratio 2: heads 0,1 share kv_head 0; heads 2,3 share 1.
    const s: Shape = .{
        .head_dim_q = 8, .head_dim_v = 8, .n_heads = 4, .n_head_kv = 2,
        .n_kv = 3, .n_tokens = 1,
        .q_nb1 = 0, .q_nb2 = 0,
        .k_nb1 = 8 * 2, .k_nb2 = 3 * 8 * 2, // [hd, n_kv, n_head_kv]
        .v_nb1 = 8 * 2, .v_nb2 = 3 * 8 * 2,
        .mask_nb1 = 0, .dst_nb1 = 0, .dst_nb2 = 0,
    };
    const k = try a.alloc(u8, s.n_kv * s.n_head_kv * s.head_dim_q * 2);
    defer a.free(k);
    for (0..s.n_kv) |kv| for (0..s.n_head_kv) |kh| for (0..s.head_dim_q) |d| {
        const off = kv * s.k_nb1 + kh * s.k_nb2 + d * 2;
        std.mem.writeInt(u16, k[off..][0..2], @intCast((kv * 10 + kh) * 8 + d), .little);
    };
    const out = try a.alloc(u8, s.kBytes());
    defer a.free(out);
    gatherK(s, k, out);
    // out order: (h, kv, d). head h uses kv_head h/2.
    var i: usize = 0;
    for (0..s.n_heads) |h| {
        const kh = h / s.headRatio();
        for (0..s.n_kv) |kv| for (0..s.head_dim_q) |d| {
            const got = std.mem.readInt(u16, out[i * 2 ..][0..2], .little);
            try std.testing.expectEqual(@as(u16, @intCast((kv * 10 + kh) * 8 + d)), got);
            i += 1;
        };
    }
}

test "gatherMask replicates per head; null mask is all-zero" {
    const a = std.testing.allocator;
    const s: Shape = .{
        .head_dim_q = 8, .head_dim_v = 8, .n_heads = 2, .n_head_kv = 1,
        .n_kv = 3, .n_tokens = 1,
        .q_nb1 = 0, .q_nb2 = 0, .k_nb1 = 0, .k_nb2 = 0, .v_nb1 = 0, .v_nb2 = 0,
        .mask_nb1 = 3 * 2, .dst_nb1 = 0, .dst_nb2 = 0,
    };
    var mask: [3]u16 = .{ f16bits(0), f16bits(-std.math.inf(f32)), f16bits(0) };
    const out = try a.alloc(u8, s.maskBytes());
    defer a.free(out);
    gatherMask(s, std.mem.sliceAsBytes(mask[0..]), out);
    // (h, kv): both heads see the same mask row.
    for (0..s.n_heads) |h| for (0..s.n_kv) |kv| {
        const got = std.mem.readInt(u16, out[(h * s.n_kv + kv) * 2 ..][0..2], .little);
        try std.testing.expectEqual(mask[kv], got);
    };
    gatherMask(s, null, out);
    for (out) |b| try std.testing.expectEqual(@as(u8, 0), b);
}

test "scatterO extracts lane 0 into the strided destination" {
    const a = std.testing.allocator;
    const s: Shape = .{
        .head_dim_q = 8, .head_dim_v = 4, .n_heads = 2, .n_head_kv = 1,
        .n_kv = 1, .n_tokens = 1,
        .q_nb1 = 0, .q_nb2 = 0, .k_nb1 = 0, .k_nb2 = 0, .v_nb1 = 0, .v_nb2 = 0, .mask_nb1 = 0,
        .dst_nb1 = 4 * 4, .dst_nb2 = 2 * 4 * 4, // dst [hd_v, n_heads, n_tokens]
    };
    const o = try a.alloc(u8, s.oBytes());
    defer a.free(o);
    @memset(o, 0);
    // beat idx = (t*nh+h)*hd_v + d ; lane 0 = a tag.
    var idx: usize = 0;
    for (0..s.n_tokens) |t| for (0..s.n_heads) |h| for (0..s.head_dim_v) |d| {
        std.mem.writeInt(u32, o[idx * o_beat_bytes ..][0..4], @intCast((t * 2 + h) * 100 + d), .little);
        idx += 1;
    };
    const dst = try a.alloc(u8, s.n_tokens * s.n_heads * s.head_dim_v * 4);
    defer a.free(dst);
    scatterO(s, o, dst);
    for (0..s.n_tokens) |t| for (0..s.n_heads) |h| for (0..s.head_dim_v) |d| {
        const got = std.mem.readInt(u32, dst[h * s.dst_nb1 + t * s.dst_nb2 + d * 4 ..][0..4], .little);
        try std.testing.expectEqual(@as(u32, @intCast((t * 2 + h) * 100 + d)), got);
    };
}
