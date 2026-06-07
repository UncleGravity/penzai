//! Wire packing for the Q1A8 fabric: the AXIS weight stream, the AXIS
//! activation stream, and result unpack. This is the host↔RTL packing
//! contract (plan §16). Layout ported from the old `proto/q1a8_layout.py`;
//! the roundtrip test pins pack/unpack agreement.

const std = @import("std");
const q1a8 = @import("q1a8");

pub fn weightBytes(num_rowblocks: usize, q1_blocks: usize) usize {
    return num_rowblocks * q1_blocks * q1a8.WEIGHT_BYTES_PER_BLOCK;
}
pub fn actBytes(q1_blocks: usize) usize {
    return q1_blocks * q1a8.ACT_BYTES_PER_BLOCK;
}
pub fn resultBytes(num_rowblocks: usize) usize {
    return num_rowblocks * q1a8.RESULT_BYTES_PER_ROWBLOCK;
}

fn put(out: []u8, off: *usize, word: u64) void {
    std.mem.writeInt(u64, out[off.*..][0..8], word, .little);
    off.* += 8;
}
fn get(buf: []const u8, off: *usize) u64 {
    const w = std.mem.readInt(u64, buf[off.*..][0..8], .little);
    off.* += 8;
    return w;
}

/// Pack weights into the AXIS stream: rowblock-major, then Q1-block-major.
/// `weight_bits`/`weight_scales` are row-major (row*q1_blocks + blk), as in
/// matmul_ref.Problem. `rows` must be a multiple of ROWS.
pub fn packWeights(
    rows: usize,
    q1_blocks: usize,
    weight_bits: []const u128,
    weight_scales: []const f16,
    out: []u8,
) void {
    const num_rowblocks = rows / q1a8.ROWS;
    std.debug.assert(out.len == weightBytes(num_rowblocks, q1_blocks));
    var off: usize = 0;
    var rb: usize = 0;
    while (rb < num_rowblocks) : (rb += 1) {
        var blk: usize = 0;
        while (blk < q1_blocks) : (blk += 1) {
            // scale beats: 4 fp16 lanes per beat
            var sbeat: usize = 0;
            while (sbeat < q1a8.SCALE_BEATS) : (sbeat += 1) {
                var word: u64 = 0;
                var local: usize = 0;
                while (local < 4) : (local += 1) {
                    const lane = sbeat * 4 + local;
                    if (lane >= q1a8.ROWS) continue;
                    const s = weight_scales[(rb * q1a8.ROWS + lane) * q1_blocks + blk];
                    const sb: u64 = @as(u16, @bitCast(s));
                    word |= sb << @intCast(local * 16);
                }
                put(out, &off, word);
            }
            // wbits beats: per Q8 sub-block, 2 rows (u32) per beat
            var sub: usize = 0;
            while (sub < q1a8.Q8_SUBBLOCKS) : (sub += 1) {
                var wbeat: usize = 0;
                while (wbeat < q1a8.WBITS_BEATS) : (wbeat += 1) {
                    var word: u64 = 0;
                    var local: usize = 0;
                    while (local < 2) : (local += 1) {
                        const lane = wbeat * 2 + local;
                        if (lane >= q1a8.ROWS) continue;
                        const bits = weight_bits[(rb * q1a8.ROWS + lane) * q1_blocks + blk];
                        const wb: u64 = @truncate(bits >> @intCast(sub * 32));
                        word |= (wb & 0xFFFF_FFFF) << @intCast(local * 32);
                    }
                    put(out, &off, word);
                }
            }
        }
    }
}

// ---- Wide weight layout (M6): one 256-bit (ROWS*32) beat per scale/subblock --
pub const WIDE_BEAT_BYTES: usize = q1a8.ROWS * 4; // ROWS*32 bits

pub fn weightBytesWide(num_rowblocks: usize, q1_blocks: usize) usize {
    return num_rowblocks * q1_blocks * (1 + q1a8.Q8_SUBBLOCKS) * WIDE_BEAT_BYTES;
}

/// Pack weights for q1a8_kernel_wide: per q1block per rowblock, one scale beat
/// (ROWS fp16 in the low bits) then 4 wbit beats (ROWS×32 bits, one subblock).
pub fn packWeightsWide(
    rows: usize,
    q1_blocks: usize,
    weight_bits: []const u128,
    weight_scales: []const f16,
    out: []u8,
) void {
    const num_rb = rows / q1a8.ROWS;
    std.debug.assert(out.len == weightBytesWide(num_rb, q1_blocks));
    var off: usize = 0;
    var rb: usize = 0;
    while (rb < num_rb) : (rb += 1) {
        var blk: usize = 0;
        while (blk < q1_blocks) : (blk += 1) {
            @memset(out[off..][0..WIDE_BEAT_BYTES], 0); // scale beat (low bits used)
            for (0..q1a8.ROWS) |lane| {
                const s = weight_scales[(rb * q1a8.ROWS + lane) * q1_blocks + blk];
                std.mem.writeInt(u16, out[off + lane * 2 ..][0..2], @bitCast(s), .little);
            }
            off += WIDE_BEAT_BYTES;
            var sub: usize = 0;
            while (sub < q1a8.Q8_SUBBLOCKS) : (sub += 1) {
                for (0..q1a8.ROWS) |lane| {
                    const bits = weight_bits[(rb * q1a8.ROWS + lane) * q1_blocks + blk];
                    const w: u32 = @truncate(bits >> @intCast(sub * 32));
                    std.mem.writeInt(u32, out[off + lane * 4 ..][0..4], w, .little);
                }
                off += WIDE_BEAT_BYTES;
            }
        }
    }
}

/// Inverse of packWeights, for the roundtrip gate.
pub fn unpackWeights(
    rows: usize,
    q1_blocks: usize,
    bytes: []const u8,
    out_bits: []u128,
    out_scales: []f16,
) void {
    const num_rowblocks = rows / q1a8.ROWS;
    std.debug.assert(bytes.len == weightBytes(num_rowblocks, q1_blocks));
    var off: usize = 0;
    var rb: usize = 0;
    while (rb < num_rowblocks) : (rb += 1) {
        var blk: usize = 0;
        while (blk < q1_blocks) : (blk += 1) {
            var sbeat: usize = 0;
            while (sbeat < q1a8.SCALE_BEATS) : (sbeat += 1) {
                const word = get(bytes, &off);
                var local: usize = 0;
                while (local < 4) : (local += 1) {
                    const lane = sbeat * 4 + local;
                    if (lane >= q1a8.ROWS) continue;
                    const sb: u16 = @truncate(word >> @intCast(local * 16));
                    out_scales[(rb * q1a8.ROWS + lane) * q1_blocks + blk] = @bitCast(sb);
                }
            }
            // accumulate bits per (rowblock-lane); zero them first this block
            var lane: usize = 0;
            while (lane < q1a8.ROWS) : (lane += 1)
                out_bits[(rb * q1a8.ROWS + lane) * q1_blocks + blk] = 0;
            var sub: usize = 0;
            while (sub < q1a8.Q8_SUBBLOCKS) : (sub += 1) {
                var wbeat: usize = 0;
                while (wbeat < q1a8.WBITS_BEATS) : (wbeat += 1) {
                    const word = get(bytes, &off);
                    var local: usize = 0;
                    while (local < 2) : (local += 1) {
                        const l = wbeat * 2 + local;
                        if (l >= q1a8.ROWS) continue;
                        const wb: u128 = @as(u32, @truncate(word >> @intCast(local * 32)));
                        out_bits[(rb * q1a8.ROWS + l) * q1_blocks + blk] |= wb << @intCast(sub * 32);
                    }
                }
            }
        }
    }
}

/// Quantize one float activation column to Q8_0 (per-Q8_BLOCK fp16 scale).
/// `column.len` must be q1_blocks * Q1_BLOCK; outputs sized accordingly.
pub fn quantizeActs(column: []const f32, out_quants: []i8, out_scales: []f16) void {
    const nblocks = column.len / q1a8.Q8_BLOCK;
    std.debug.assert(out_quants.len == column.len);
    std.debug.assert(out_scales.len == nblocks);
    var b: usize = 0;
    while (b < nblocks) : (b += 1) {
        const base = b * q1a8.Q8_BLOCK;
        var amax: f32 = 0;
        for (column[base..][0..q1a8.Q8_BLOCK]) |v| amax = @max(amax, @abs(v));
        if (amax == 0) {
            out_scales[b] = 0;
            @memset(out_quants[base..][0..q1a8.Q8_BLOCK], 0);
            continue;
        }
        const scale: f16 = @floatCast(amax / 127.0);
        out_scales[b] = scale;
        const inv: f32 = 1.0 / @as(f32, @floatCast(scale));
        for (column[base..][0..q1a8.Q8_BLOCK], 0..) |v, i| {
            const q = std.math.round(v * inv);
            out_quants[base + i] = @intFromFloat(std.math.clamp(q, -128, 127));
        }
    }
}

/// Pack the activation stream: per Q1 block, per Q8 sub-block, 4 int8 beats
/// then 1 scale beat. One column, broadcast to all rowblocks by the fabric.
pub fn packActs(q1_blocks: usize, quants: []const i8, scales: []const f16, out: []u8) void {
    std.debug.assert(out.len == actBytes(q1_blocks));
    var off: usize = 0;
    var blk: usize = 0;
    while (blk < q1_blocks) : (blk += 1) {
        var sub: usize = 0;
        while (sub < q1a8.Q8_SUBBLOCKS) : (sub += 1) {
            const q8 = blk * q1a8.Q8_SUBBLOCKS + sub;
            const acts = quants[q8 * q1a8.Q8_BLOCK ..][0..q1a8.Q8_BLOCK];
            var beat: usize = 0;
            while (beat < q1a8.Q8_BLOCK / q1a8.BEAT_BYTES) : (beat += 1) {
                var word: u64 = 0;
                var byte: usize = 0;
                while (byte < q1a8.BEAT_BYTES) : (byte += 1) {
                    const u: u64 = @as(u8, @bitCast(acts[beat * q1a8.BEAT_BYTES + byte]));
                    word |= u << @intCast(byte * 8);
                }
                put(out, &off, word);
            }
            put(out, &off, @as(u16, @bitCast(scales[q8]))); // scale in low 16b
        }
    }
}

/// Unpack the result stream into per-row fp32 (lane-major, 2 fp32/beat).
pub fn unpackResults(num_rowblocks: usize, bytes: []const u8, out: []f32) void {
    std.debug.assert(bytes.len == resultBytes(num_rowblocks));
    std.debug.assert(out.len == num_rowblocks * q1a8.ROWS);
    var off: usize = 0;
    var rb: usize = 0;
    while (rb < num_rowblocks) : (rb += 1) {
        var beat: usize = 0;
        while (beat < q1a8.ROWS / 2) : (beat += 1) {
            const word = get(bytes, &off);
            const lane = rb * q1a8.ROWS + beat * 2;
            out[lane] = @bitCast(@as(u32, @truncate(word)));
            out[lane + 1] = @bitCast(@as(u32, @truncate(word >> 32)));
        }
    }
}

const testing = std.testing;

test "weight pack/unpack roundtrip" {
    const rows = q1a8.ROWS * 2; // 2 rowblocks
    const blocks = 5;
    var prng = std.Random.DefaultPrng.init(0xC0FFEE);
    const rnd = prng.random();

    var bits: [rows * blocks]u128 = undefined;
    var scales: [rows * blocks]f16 = undefined;
    for (&bits) |*b| b.* = (@as(u128, rnd.int(u64)) << 64) | rnd.int(u64);
    for (&scales) |*s| s.* = @floatCast(rnd.float(f32));

    var buf: [weightBytes(2, blocks)]u8 = undefined;
    packWeights(rows, blocks, &bits, &scales, &buf);

    var bits2: [rows * blocks]u128 = undefined;
    var scales2: [rows * blocks]f16 = undefined;
    unpackWeights(rows, blocks, &buf, &bits2, &scales2);
    try testing.expectEqualSlices(u128, &bits, &bits2);
    try testing.expectEqualSlices(f16, &scales, &scales2);
}

test "quantizeActs scale and reconstruction" {
    var col: [q1a8.Q1_BLOCK]f32 = undefined;
    for (&col, 0..) |*v, i| v.* = @floatFromInt(@as(i32, @intCast(i)) - 64);
    var q: [q1a8.Q1_BLOCK]i8 = undefined;
    var s: [q1a8.Q8_SUBBLOCKS]f16 = undefined;
    quantizeActs(&col, &q, &s);
    // dequant should track the input within one quantization step
    for (col, 0..) |v, i| {
        const dq = @as(f32, @floatFromInt(q[i])) * @as(f32, @floatCast(s[i / q1a8.Q8_BLOCK]));
        try testing.expectApproxEqAbs(v, dq, @as(f32, @floatCast(s[i / q1a8.Q8_BLOCK])) + 1e-3);
    }
}

test "result pack/unpack roundtrip" {
    const num_rowblocks = 2;
    var vals: [num_rowblocks * q1a8.ROWS]f32 = undefined;
    for (&vals, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i)) * 1.25 - 3.0;
    var buf: [resultBytes(num_rowblocks)]u8 = undefined;
    var off: usize = 0;
    for (0..num_rowblocks) |rb| {
        for (0..q1a8.ROWS / 2) |beat| {
            const lo: u64 = @as(u32, @bitCast(vals[rb * q1a8.ROWS + beat * 2]));
            const hi: u64 = @as(u32, @bitCast(vals[rb * q1a8.ROWS + beat * 2 + 1]));
            put(&buf, &off, lo | (hi << 32));
        }
    }
    var out: [num_rowblocks * q1a8.ROWS]f32 = undefined;
    unpackResults(num_rowblocks, &buf, &out);
    try testing.expectEqualSlices(f32, &vals, &out);
}
