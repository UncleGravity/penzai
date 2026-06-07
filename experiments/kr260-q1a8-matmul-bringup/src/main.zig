//! Laptop self-test (M0–M1): build a random Q1A8 problem, exercise the packer
//! roundtrip and the reference oracle, and print a summary. No hardware.
//! The board runner (M4) and Verilator cosim (M2) are added later; see README.

const std = @import("std");
const q1a8 = @import("q1a8");
const ref = @import("matmul_ref");
const pack = @import("pack");

pub fn main() !void {
    const rows = q1a8.ROWS * 3; // 3 rowblocks
    const blocks = 4; // K = 512
    const num_rowblocks = rows / q1a8.ROWS;

    var gpa = std.heap.DebugAllocator(.{}){};
    defer _ = gpa.deinit();
    const a = gpa.allocator();

    const bits = try a.alloc(u128, rows * blocks);
    defer a.free(bits);
    const wscales = try a.alloc(f16, rows * blocks);
    defer a.free(wscales);
    const column = try a.alloc(f32, blocks * q1a8.Q1_BLOCK);
    defer a.free(column);

    var prng = std.Random.DefaultPrng.init(1);
    const rnd = prng.random();
    for (bits) |*b| b.* = (@as(u128, rnd.int(u64)) << 64) | rnd.int(u64);
    for (wscales) |*s| s.* = @floatCast(0.01 + rnd.float(f32) * 0.1);
    for (column) |*v| v.* = (rnd.float(f32) - 0.5) * 4.0;

    const aquants = try a.alloc(i8, column.len);
    defer a.free(aquants);
    const ascales = try a.alloc(f16, blocks * q1a8.Q8_SUBBLOCKS);
    defer a.free(ascales);
    pack.quantizeActs(column, aquants, ascales);

    // pack weights, then unpack and confirm the contract roundtrips
    const wbuf = try a.alloc(u8, pack.weightBytes(num_rowblocks, blocks));
    defer a.free(wbuf);
    pack.packWeights(rows, blocks, bits, wscales, wbuf);
    const bits2 = try a.alloc(u128, rows * blocks);
    defer a.free(bits2);
    const wscales2 = try a.alloc(f16, rows * blocks);
    defer a.free(wscales2);
    pack.unpackWeights(rows, blocks, wbuf, bits2, wscales2);
    if (!std.mem.eql(u128, bits, bits2)) return error.WeightRoundtripMismatch;

    const abuf = try a.alloc(u8, pack.actBytes(blocks));
    defer a.free(abuf);
    pack.packActs(blocks, aquants, ascales, abuf);

    const p: ref.Problem = .{
        .rows = rows,
        .q1_blocks = blocks,
        .weight_bits = bits,
        .weight_scales = wscales,
        .act_quants = aquants,
        .act_scales = ascales,
    };
    const out = try a.alloc(f32, rows);
    defer a.free(out);
    ref.scaledOutput(p, out);

    std.debug.print(
        "case=selftest ok=1 rows={d} q1_blocks={d} weight_bytes={d} act_bytes={d} out[0]={d:.4} out[last]={d:.4}\n",
        .{ rows, blocks, wbuf.len, abuf.len, out[0], out[rows - 1] },
    );
}
