const std = @import("std");
const builtin = @import("builtin");
const shared = @import("shared");

const q1a8 = shared.q1a8;

pub const available = builtin.cpu.arch == .aarch64;

const max_workers: usize = 4;
const min_parallel_work_units: usize = 512;

const MatmulError = error{
    InvalidShape,
    InvalidLength,
    OutOfMemory,
};

const Vec8i8 = @Vector(8, i8);
const Vec8i16 = @Vector(8, i16);

const sign_lut = makeSignLut();

const Q8Vectors = struct {
    q0: Vec8i8,
    q1: Vec8i8,
    q2: Vec8i8,
    q3: Vec8i8,
};

const WorkerContext = struct {
    packed_weights: []const u8,
    dst_f32: []u8,
    rows: usize,
    col: usize,
    q1_blocks: usize,
    packed_per_rowblock: usize,
    quants: []const i8,
    act_scales: []const f32,
    rb_start: usize,
    rb_end: usize,
};

comptime {
    if (q1a8.rows_per_block != 8) @compileError("q1a8 SIMD path assumes 8-row blocks");
    if (q1a8.q8_block != 32) @compileError("q1a8 SIMD path assumes 32-wide q8 blocks");
}

pub fn runQ1A8(
    allocator: std.mem.Allocator,
    packed_weights: []const u8,
    acts_f32: []const u8,
    dst_f32: []u8,
    rows: u32,
    cols: u32,
    k: u32,
) MatmulError!void {
    if (comptime !available) @compileError("q1a8 SIMD path is AArch64-only");

    const rows_usize: usize = @intCast(rows);
    const cols_usize: usize = @intCast(cols);
    const k_usize: usize = @intCast(k);
    if (rows_usize == 0 or cols_usize == 0) return error.InvalidShape;
    const q1_blocks = q1a8.blocksPerRow(k_usize) catch return error.InvalidShape;
    if (packed_weights.len != (q1a8.packedWeightBytes(rows_usize, k_usize) catch return error.InvalidShape)) {
        return error.InvalidLength;
    }
    if (acts_f32.len != (q1a8.actsF32Bytes(cols_usize, k_usize) catch return error.InvalidShape)) {
        return error.InvalidLength;
    }
    if (dst_f32.len != (q1a8.outputF32Bytes(rows_usize, cols_usize) catch return error.InvalidShape)) {
        return error.InvalidLength;
    }

    const quants = try allocator.alloc(i8, k_usize);
    defer allocator.free(quants);
    const q8_blocks = q1_blocks * q1a8.q8_subblocks;
    const act_scales = try allocator.alloc(f32, q8_blocks);
    defer allocator.free(act_scales);
    const column = try allocator.alloc(f32, k_usize);
    defer allocator.free(column);

    const rowblocks = q1a8.rowblocksFor(rows_usize);
    const packed_per_rowblock = q1_blocks * q1a8.packed_per_q1_block;

    for (0..cols_usize) |col| {
        for (0..k_usize) |i| {
            column[i] = readF32(acts_f32, (col * k_usize + i) * @sizeOf(f32));
        }
        q1a8.quantizeQ8_0F32Scales(column, quants, act_scales) catch return error.InvalidShape;

        const workers = chooseWorkerCount(rowblocks, q1_blocks);
        const ctx = WorkerContext{
            .packed_weights = packed_weights,
            .dst_f32 = dst_f32,
            .rows = rows_usize,
            .col = col,
            .q1_blocks = q1_blocks,
            .packed_per_rowblock = packed_per_rowblock,
            .quants = quants,
            .act_scales = act_scales,
            .rb_start = 0,
            .rb_end = rowblocks,
        };
        if (workers == 1) {
            computeRowblockRange(&ctx);
        } else {
            runColumnParallel(ctx, rowblocks, workers);
        }
    }
}

fn chooseWorkerCount(rowblocks: usize, q1_blocks: usize) usize {
    if (comptime builtin.single_threaded) return 1;
    const work_units = rowblocks * q1_blocks;
    if (work_units < min_parallel_work_units) return 1;
    return @min(max_workers, rowblocks);
}

fn runColumnParallel(base_ctx: WorkerContext, rowblocks: usize, workers: usize) void {
    var contexts: [max_workers]WorkerContext = undefined;
    var threads: [max_workers - 1]std.Thread = undefined;
    var spawned: usize = 0;

    for (0..workers) |worker| {
        const range = workerRange(rowblocks, workers, worker);
        contexts[worker] = base_ctx;
        contexts[worker].rb_start = range.start;
        contexts[worker].rb_end = range.end;
    }

    for (1..workers) |worker| {
        threads[spawned] = std.Thread.spawn(.{}, workerMain, .{&contexts[worker]}) catch {
            for (threads[0..spawned]) |thread| thread.join();
            computeRowblockRange(&base_ctx);
            return;
        };
        spawned += 1;
    }

    computeRowblockRange(&contexts[0]);
    for (threads[0..spawned]) |thread| thread.join();
}

fn workerRange(rowblocks: usize, workers: usize, worker: usize) struct { start: usize, end: usize } {
    const base = rowblocks / workers;
    const extra = rowblocks % workers;
    const start = worker * base + @min(worker, extra);
    const count = base + @intFromBool(worker < extra);
    return .{ .start = start, .end = start + count };
}

fn workerMain(ctx: *const WorkerContext) void {
    computeRowblockRange(ctx);
}

fn computeRowblockRange(ctx: *const WorkerContext) void {
    for (ctx.rb_start..ctx.rb_end) |rb| {
        const row_start = rb * q1a8.rows_per_block;
        const row_count = @min(q1a8.rows_per_block, ctx.rows - row_start);
        const rb_base = rb * ctx.packed_per_rowblock;
        var accs = [_]f32{0} ** q1a8.rows_per_block;

        for (0..ctx.q1_blocks) |q1| {
            const block_base = rb_base + q1 * q1a8.packed_per_q1_block;
            const weight_scales = loadWeightScales(ctx.packed_weights, block_base, row_count);

            for (0..q1a8.q8_subblocks) |sub| {
                const q8 = q1 * q1a8.q8_subblocks + sub;
                const act_scale = ctx.act_scales[q8];
                if (act_scale == 0) continue;

                const bits_base = block_base + q1a8.scales_bytes + sub * q1a8.wbits_bytes;
                const quants_base = q1 * q1a8.q1_block + sub * q1a8.q8_block;
                const q8_vecs = loadQ8Vectors(ctx.quants[quants_base..][0..q1a8.q8_block]);
                for (0..row_count) |lane| {
                    const weight_scale = weight_scales[lane];
                    if (weight_scale == 0) continue;

                    const bits = readU32(ctx.packed_weights, bits_base + lane * @sizeOf(u32));
                    const sum = dotQ8(bits, q8_vecs);
                    accs[lane] = @mulAdd(f32, @as(f32, @floatFromInt(sum)), weight_scale * act_scale, accs[lane]);
                }
            }
        }

        for (0..row_count) |lane| {
            writeF32(ctx.dst_f32, (ctx.col * ctx.rows + row_start + lane) * @sizeOf(f32), accs[lane]);
        }
    }
}

fn makeSignLut() [256]Vec8i16 {
    @setEvalBranchQuota(20_000);
    var table: [256]Vec8i16 = undefined;
    for (0..256) |mask| {
        var signs: [8]i16 = undefined;
        inline for (0..8) |i| {
            signs[i] = if (((mask >> i) & 1) != 0) 1 else -1;
        }
        table[mask] = signs;
    }
    return table;
}

fn loadQ8Vectors(quants: *const [q1a8.q8_block]i8) Q8Vectors {
    return .{
        .q0 = quants[0..8].*,
        .q1 = quants[8..16].*,
        .q2 = quants[16..24].*,
        .q3 = quants[24..32].*,
    };
}

inline fn dot8(mask: u8, q: Vec8i8) i16 {
    const q16: Vec8i16 = @intCast(q);
    return @reduce(.Add, q16 * sign_lut[mask]);
}

inline fn dotQ8(bits: u32, q: Q8Vectors) i32 {
    return @as(i32, dot8(@truncate(bits), q.q0)) +
        @as(i32, dot8(@truncate(bits >> 8), q.q1)) +
        @as(i32, dot8(@truncate(bits >> 16), q.q2)) +
        @as(i32, dot8(@truncate(bits >> 24), q.q3));
}

fn loadWeightScales(bytes: []const u8, block_base: usize, row_count: usize) [q1a8.rows_per_block]f32 {
    var scales = [_]f32{0} ** q1a8.rows_per_block;
    for (0..row_count) |lane| {
        scales[lane] = @floatCast(readF16(bytes, block_base + lane * @sizeOf(f16)));
    }
    return scales;
}

fn readF32(bytes: []const u8, offset: usize) f32 {
    return @bitCast(std.mem.readInt(u32, bytes[offset..][0..4], .little));
}

fn readF16(bytes: []const u8, offset: usize) f16 {
    return @bitCast(std.mem.readInt(u16, bytes[offset..][0..2], .little));
}

fn readU32(bytes: []const u8, offset: usize) u32 {
    return std.mem.readInt(u32, bytes[offset..][0..4], .little);
}

fn writeF32(bytes: []u8, offset: usize, value: f32) void {
    std.mem.writeInt(u32, bytes[offset..][0..4], @bitCast(value), .little);
}
