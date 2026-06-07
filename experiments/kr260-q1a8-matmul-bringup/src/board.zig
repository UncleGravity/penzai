//! M4 board runner: run a Q1A8 matmul on the KR260 fabric and check the result
//! against matmul_ref. One XRT BO holds packed weights, packed acts, and the
//! result region; two AXI DMAs feed the kernel; we strobe it over AXI-Lite,
//! wait, read back, and diff. Output is line-oriented key=value text.
//!
//! Run as root on the board (needs /dev/mem + the loaded q1a8-matmul overlay).

const std = @import("std");
const config = @import("config");
const xrtmod = @import("xrt");
const dmamod = @import("dma");
const mmio = @import("mmio");
const q1a8 = @import("q1a8");
const pack = @import("pack");
const ref = @import("matmul_ref");

const Ctx = struct {
    xrt: *xrtmod.Xrt,
    bo: xrtmod.BufferHandle,
    mem: []u8,
    phys: u64,
    dma_w: *dmamod.Dma,
    dma_a: *dmamod.Dma,
    kernel: *mmio.Kernel,
};

fn dumpHex(label: []const u8, bytes: []const u8) void {
    std.debug.print("{s}", .{label});
    for (bytes, 0..) |x, i| {
        if (i % 16 == 0) std.debug.print("\n  ", .{});
        std.debug.print("{x:0>2} ", .{x});
    }
    std.debug.print("\n", .{});
}

fn runCase(a: std.mem.Allocator, ctx: *Ctx, rows: usize, blocks: usize, seed: u64, verbose: bool) !void {
    const num_rb = rows / q1a8.ROWS;

    const bits = try a.alloc(u128, rows * blocks);
    defer a.free(bits);
    const wscales = try a.alloc(f16, rows * blocks);
    defer a.free(wscales);
    const column = try a.alloc(f32, blocks * q1a8.Q1_BLOCK);
    defer a.free(column);

    var prng = std.Random.DefaultPrng.init(seed);
    const rnd = prng.random();
    for (bits) |*b| b.* = (@as(u128, rnd.int(u64)) << 64) | rnd.int(u64);
    for (wscales) |*s| s.* = @floatCast(0.01 + rnd.float(f32) * 0.2);
    for (column) |*v| v.* = (rnd.float(f32) - 0.5) * 4.0;

    const aquants = try a.alloc(i8, column.len);
    defer a.free(aquants);
    const ascales = try a.alloc(f16, blocks * q1a8.Q8_SUBBLOCKS);
    defer a.free(ascales);
    pack.quantizeActs(column, aquants, ascales);

    const w_bytes = pack.weightBytes(num_rb, blocks);
    const a_bytes = pack.actBytes(blocks);
    const r_bytes = pack.resultBytes(num_rb);

    // Pack directly into the mapped BO regions, then push to device.
    pack.packWeights(rows, blocks, bits, wscales, ctx.mem[config.weights_offset..][0..w_bytes]);
    pack.packActs(blocks, aquants, ascales, ctx.mem[config.acts_offset..][0..a_bytes]);
    @memset(ctx.mem[config.results_offset..][0..r_bytes], 0);
    _ = ctx.xrt.boSync(ctx.bo, xrtmod.sync_to_device, ctx.mem.len, 0);

    // Program DMAs: arm the result sink first, then the two source streams.
    try ctx.dma_w.reset();
    try ctx.dma_a.resetMm2s();
    try ctx.dma_w.startWriteToDdr(ctx.phys + config.results_offset, r_bytes);
    try ctx.dma_w.startReadFromDdr(ctx.phys + config.weights_offset, w_bytes);
    try ctx.dma_a.startReadFromDdr(ctx.phys + config.acts_offset, a_bytes);

    // Kick the kernel and wait for it + the result DMA.
    ctx.kernel.run(@intCast(blocks), @intCast(num_rb));
    ctx.kernel.waitDone() catch |e| {
        ctx.dma_w.dumpStatus("weights+results");
        ctx.dma_a.dumpStatus("acts");
        return e;
    };
    try ctx.dma_w.waitWriteDone();

    _ = ctx.xrt.boSync(ctx.bo, xrtmod.sync_from_device, ctx.mem.len, 0);
    const got = try a.alloc(f32, rows);
    defer a.free(got);
    pack.unpackResults(num_rb, ctx.mem[config.results_offset..][0..r_bytes], got);

    const expected = try a.alloc(f32, rows);
    defer a.free(expected);
    ref.scaledOutput(.{
        .rows = rows,
        .q1_blocks = blocks,
        .weight_bits = bits,
        .weight_scales = wscales,
        .act_quants = aquants,
        .act_scales = ascales,
    }, expected);

    var max_rel: f32 = 0;
    for (got, expected, 0..) |g, e, i| {
        const rel = @abs(g - e) / @max(@abs(e), 1.0);
        max_rel = @max(max_rel, rel);
        if (rel > 0.02)
            std.debug.print("  mismatch row={d} got={d:.5} exp={d:.5} rel={d:.4}\n", .{ i, g, e, rel });
    }
    if (verbose) {
        // Weights region read back from DDR (whole-BO sync invalidated it): if
        // this matches the packer, the to-device path is coherent and the kernel
        // saw correct input. Results region: raw DMA output vs expected layout.
        dumpHex("DBG weights[0..48] in DDR (expect packer bytes):", ctx.mem[config.weights_offset..][0..@min(w_bytes, 48)]);
        dumpHex("DBG results raw bytes from DDR (got):", ctx.mem[config.results_offset..][0..r_bytes]);
        const exp_bytes = try a.alloc(u8, r_bytes);
        defer a.free(exp_bytes);
        var off: usize = 0;
        for (0..num_rb) |rb| {
            for (0..q1a8.ROWS / 2) |beat| {
                const lo: u64 = @as(u32, @bitCast(expected[rb * q1a8.ROWS + beat * 2]));
                const hi: u64 = @as(u32, @bitCast(expected[rb * q1a8.ROWS + beat * 2 + 1]));
                std.mem.writeInt(u64, exp_bytes[off..][0..8], lo | (hi << 32), .little);
                off += 8;
            }
        }
        dumpHex("DBG results expected bytes:", exp_bytes);
    }

    const cyc = ctx.kernel.cycles();
    const macs = rows * blocks * q1a8.Q1_BLOCK;
    const bpc = @as(f64, @floatFromInt(macs)) / @as(f64, @floatFromInt(@max(cyc, 1)));
    std.debug.print(
        "case=matmul rows={d} blocks={d} ok={d} max_rel={d:.4} cycles={d} MAC/cycle={d:.1}\n",
        .{ rows, blocks, @intFromBool(max_rel <= 0.02), max_rel, cyc, bpc },
    );
    if (max_rel > 0.02) return error.ResultMismatch;
}

pub fn main() !void {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer _ = gpa.deinit();
    const a = gpa.allocator();

    var xrt = try xrtmod.Xrt.open();
    defer xrt.close();
    const dev = xrt.deviceOpen(0);
    if (dev == null) return error.XrtDeviceOpen;
    defer _ = xrt.deviceClose(dev);

    const bo = xrt.boAlloc(dev, config.default_bo_size, xrtmod.flags_normal, xrtmod.group_default);
    if (bo == null) return error.BoAlloc;
    defer _ = xrt.boFree(bo);
    const map = xrt.boMap(bo);
    if (map == null) return error.BoMap;
    const mem = @as([*]u8, @ptrCast(map.?))[0..config.default_bo_size];
    const phys = xrt.boAddress(bo);

    var dma_w = try dmamod.Dma.openDevice(config.dma_w_base);
    defer dma_w.closeDevice();
    var dma_a = try dmamod.Dma.openDevice(config.dma_a_base);
    defer dma_a.closeDevice();
    var kernel = try mmio.Kernel.openDevice();
    defer kernel.closeDevice();

    const kid = kernel.id();
    const kver = kernel.version();
    std.debug.print(
        "case=info bo_phys=0x{x} id=0x{x:0>8} version=0x{x:0>8} rows={d} id_ok={d} ver_ok={d}\n",
        .{ phys, kid, kver, kernel.rows(), @intFromBool(kid == mmio.ID_VALUE), @intFromBool(kver == mmio.VERSION_VALUE) },
    );
    if (kid != mmio.ID_VALUE) return error.BadKernelId;

    var ctx: Ctx = .{
        .xrt = &xrt,
        .bo = bo,
        .mem = mem,
        .phys = phys,
        .dma_w = &dma_w,
        .dma_a = &dma_a,
        .kernel = &kernel,
    };

    const Case = struct { rows: usize, blocks: usize };
    const cases = [_]Case{
        .{ .rows = 8, .blocks = 1 },
        .{ .rows = 8, .blocks = 4 },
        .{ .rows = 24, .blocks = 2 },
        .{ .rows = 64, .blocks = 16 },
    };
    var ok = true;
    for (cases, 0..) |cs, i| {
        // flip the last arg to true to dump raw weight/result bytes for a case
        runCase(a, &ctx, cs.rows, cs.blocks, 0x1000 + i, false) catch |e| {
            std.debug.print("case=matmul rows={d} blocks={d} ok=0 err={s}\n", .{ cs.rows, cs.blocks, @errorName(e) });
            ok = false;
        };
    }
    std.debug.print("case=summary ok={d}\n", .{@intFromBool(ok)});
    if (!ok) std.process.exit(1);
}
