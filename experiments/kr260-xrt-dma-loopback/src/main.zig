//! KR260 XRT BO + AXI DMA loopback verifier and bandwidth sweep.

const std = @import("std");
const config = @import("config.zig");
const dma = @import("dma.zig");
const sizes = @import("sizes.zig");
const xrt = @import("xrt.zig");

const Options = struct {
    bo_bytes: ?usize = null,
    size_bytes: ?usize = null,
    max_bytes: usize = config.default_max_transfer,
    chunk_bytes: usize = config.default_chunk_size,
    iters: usize = config.default_stress_iters,
};

const Board = struct {
    x: xrt.Xrt,
    dev: xrt.DeviceHandle,

    fn open() !Board {
        var x = xrt.Xrt.open() catch |err| {
            std.debug.print("xrt: failed to load libxrt_coreutil.so.2: {s}\n", .{@errorName(err)});
            return err;
        };
        errdefer x.close();

        const dev = x.deviceOpen(0);
        if (dev == null) {
            std.debug.print("xrt: device open FAIL (zocl/app loaded? render group?)\n", .{});
            return error.DeviceOpen;
        }

        std.debug.print("xrt: device open OK\n", .{});
        return .{ .x = x, .dev = dev };
    }

    fn close(self: *Board) void {
        _ = self.x.deviceClose(self.dev);
        self.x.close();
    }
};

const MappedBo = struct {
    handle: xrt.BufferHandle,
    phys: u64,
    mem: [*]u8,
    size: usize,

    fn alloc(board: *Board, size: usize) !MappedBo {
        const bo = board.x.boAlloc(board.dev, size, xrt.flags_normal, xrt.group_default) orelse {
            std.debug.print("bo: alloc FAIL size_bytes={d} size_mib={d}\n", .{ size, sizes.mib(size) });
            return error.BoAlloc;
        };
        errdefer _ = board.x.boFree(bo);

        const mem: [*]u8 = @ptrCast(board.x.boMap(bo) orelse {
            std.debug.print("bo: map FAIL size_bytes={d}\n", .{size});
            return error.BoMap;
        });

        const phys = board.x.boAddress(bo);
        std.debug.print("bo: alloc/map PASS phys=0x{x} size_bytes={d} size_mib={d}\n", .{ phys, size, sizes.mib(size) });
        return .{ .handle = bo, .phys = phys, .mem = mem, .size = size };
    }

    fn free(self: *MappedBo, board: *Board) void {
        _ = board.x.boFree(self.handle);
        self.* = undefined;
    }
};

const RunResult = struct {
    chunks: usize,
    dma_ns: u64,
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;

    var args = try std.process.Args.Iterator.initAllocator(init.minimal.args, allocator);
    defer args.deinit();
    _ = args.next();

    const command = args.next() orelse "verify";
    if (std.mem.eql(u8, command, "help") or std.mem.eql(u8, command, "--help") or
        std.mem.eql(u8, command, "-h"))
    {
        printUsage();
        return;
    }

    const opts = parseOptions(&args) catch |err| {
        std.debug.print("error=invalid_args detail={s}\n", .{@errorName(err)});
        printUsage();
        return err;
    };
    try validateOptions(opts);

    if (std.mem.eql(u8, command, "verify")) {
        try runVerify(init.io, opts);
    } else if (std.mem.eql(u8, command, "sweep")) {
        try runSweep(init.io, opts);
    } else if (std.mem.eql(u8, command, "stress")) {
        try runStress(init.io, opts);
    } else {
        std.debug.print("error=unknown_command command={s}\n", .{command});
        printUsage();
        return error.UnknownCommand;
    }
}

fn printUsage() void {
    std.debug.print(
        \\usage: kr260-xrt-dma-loopback <command> [options]
        \\
        \\commands:
        \\  verify   small correctness smoke (default)
        \\  sweep    correctness + bandwidth over transfer sizes
        \\  stress   repeated transfer at one size
        \\
        \\options:
        \\  --bo SIZE      backing BO size for sweep/stress, default 768MiB
        \\  --size SIZE    transfer size for verify/stress
        \\  --max SIZE     max sweep transfer, default 384MiB
        \\  --chunk SIZE   DMA chunk size, default 32MiB
        \\  --iters N      stress iterations, default 10
        \\
        \\SIZE accepts B, KiB, MiB, GiB suffixes.
        \\
    , .{});
}

fn parseOptions(args: *std.process.Args.Iterator) !Options {
    var opts: Options = .{};
    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "--bo")) {
            opts.bo_bytes = try sizes.parse(args.next() orelse return error.MissingValue);
        } else if (std.mem.startsWith(u8, arg, "--bo=")) {
            opts.bo_bytes = try sizes.parse(arg["--bo=".len..]);
        } else if (std.mem.eql(u8, arg, "--size")) {
            opts.size_bytes = try sizes.parse(args.next() orelse return error.MissingValue);
        } else if (std.mem.startsWith(u8, arg, "--size=")) {
            opts.size_bytes = try sizes.parse(arg["--size=".len..]);
        } else if (std.mem.eql(u8, arg, "--max")) {
            opts.max_bytes = try sizes.parse(args.next() orelse return error.MissingValue);
        } else if (std.mem.startsWith(u8, arg, "--max=")) {
            opts.max_bytes = try sizes.parse(arg["--max=".len..]);
        } else if (std.mem.eql(u8, arg, "--chunk")) {
            opts.chunk_bytes = try sizes.parse(args.next() orelse return error.MissingValue);
        } else if (std.mem.startsWith(u8, arg, "--chunk=")) {
            opts.chunk_bytes = try sizes.parse(arg["--chunk=".len..]);
        } else if (std.mem.eql(u8, arg, "--iters")) {
            opts.iters = try parseCount(args.next() orelse return error.MissingValue);
        } else if (std.mem.startsWith(u8, arg, "--iters=")) {
            opts.iters = try parseCount(arg["--iters=".len..]);
        } else {
            return error.UnknownOption;
        }
    }
    return opts;
}

fn parseCount(text: []const u8) !usize {
    const value = try std.fmt.parseInt(usize, text, 10);
    if (value == 0) return error.InvalidCount;
    return value;
}

fn validateOptions(opts: Options) !void {
    if (opts.bo_bytes != null and opts.bo_bytes.? == 0) return error.InvalidSize;
    if (opts.size_bytes != null and opts.size_bytes.? == 0) return error.InvalidSize;
    if (opts.max_bytes == 0 or opts.chunk_bytes == 0 or opts.iters == 0) return error.InvalidSize;
}

fn runVerify(io: std.Io, opts: Options) !void {
    const transfer_size = opts.size_bytes orelse config.smoke_transfer_size;
    const bo_size = opts.bo_bytes orelse @max(config.smoke_bo_size, alignUp(transfer_size * 2, sizes.MiB));

    var board = try Board.open();
    defer board.close();
    var d = try openDma();
    defer d.closeDevice();

    var bo = try MappedBo.alloc(&board, bo_size);
    defer bo.free(&board);

    const result = try runTransfer(io, &board, &d, &bo, transfer_size, opts.chunk_bytes, "verify", null);
    printCase("verify", transfer_size, opts.chunk_bytes, result);
    std.debug.print("ALL PASS\n", .{});
}

fn runSweep(io: std.Io, opts: Options) !void {
    const bo_size = opts.bo_bytes orelse config.default_bo_size;
    const max_transfer = @min(opts.max_bytes, bo_size / 2);

    var board = try Board.open();
    defer board.close();
    var d = try openDma();
    defer d.closeDevice();

    var bo = try MappedBo.alloc(&board, bo_size);
    defer bo.free(&board);

    const sweep_sizes = [_]usize{
        4 * sizes.KiB,
        64 * sizes.KiB,
        1 * sizes.MiB,
        8 * sizes.MiB,
        32 * sizes.MiB,
        64 * sizes.MiB,
        128 * sizes.MiB,
        256 * sizes.MiB,
        384 * sizes.MiB,
    };

    for (sweep_sizes) |transfer_size| {
        if (transfer_size > max_transfer) continue;
        const result = try runTransfer(io, &board, &d, &bo, transfer_size, opts.chunk_bytes, "sweep", null);
        printCase("sweep", transfer_size, opts.chunk_bytes, result);
    }
    std.debug.print("ALL PASS\n", .{});
}

fn runStress(io: std.Io, opts: Options) !void {
    const transfer_size = opts.size_bytes orelse config.default_stress_size;
    const bo_size = opts.bo_bytes orelse config.default_bo_size;
    if (transfer_size > bo_size / 2) return error.TransferExceedsBoWindow;

    var board = try Board.open();
    defer board.close();
    var d = try openDma();
    defer d.closeDevice();

    var bo = try MappedBo.alloc(&board, bo_size);
    defer bo.free(&board);

    var min_ns: u64 = std.math.maxInt(u64);
    var max_ns: u64 = 0;
    var total_ns: u128 = 0;

    var iter: usize = 0;
    while (iter < opts.iters) : (iter += 1) {
        const result = try runTransfer(io, &board, &d, &bo, transfer_size, opts.chunk_bytes, "stress", iter);
        printCase("stress", transfer_size, opts.chunk_bytes, result);
        min_ns = @min(min_ns, result.dma_ns);
        max_ns = @max(max_ns, result.dma_ns);
        total_ns += result.dma_ns;
    }

    const avg_ns: u64 = @intCast(total_ns / opts.iters);
    std.debug.print(
        "case=stress_summary size_bytes={d} size_mib={d} iters={d} min_us={d} avg_us={d} max_us={d} avg_payload_mib_s={d} avg_ddr_mib_s={d}\n",
        .{
            transfer_size,
            sizes.mib(transfer_size),
            opts.iters,
            nsToUs(min_ns),
            nsToUs(avg_ns),
            nsToUs(max_ns),
            mibPerSec(transfer_size, avg_ns),
            mibPerSec(transfer_size * 2, avg_ns),
        },
    );
    std.debug.print("ALL PASS\n", .{});
}

fn openDma() !dma.Dma {
    var d = dma.Dma.openDevice() catch |err| {
        std.debug.print("dma: /dev/mem map FAIL (run as root?): {s}\n", .{@errorName(err)});
        return err;
    };
    d.dumpStatus("open");
    return d;
}

fn runTransfer(
    io: std.Io,
    board: *Board,
    d: *dma.Dma,
    bo: *MappedBo,
    transfer_size: usize,
    chunk_size: usize,
    case_name: []const u8,
    maybe_iter: ?usize,
) !RunResult {
    if (transfer_size == 0 or transfer_size > bo.size / 2) return error.TransferExceedsBoWindow;

    const src_off: usize = 0;
    const dst_off: usize = bo.size / 2;
    const src = bo.mem[src_off..][0..transfer_size];
    const dst = bo.mem[dst_off..][0..transfer_size];
    const seed: u8 = if (maybe_iter) |iter| @truncate(iter + 1) else 1;

    fillPattern(src, seed);
    @memset(dst, 0);

    if (board.x.boSync(bo.handle, xrt.sync_to_device, transfer_size, src_off) != 0) return error.BoSync;
    if (board.x.boSync(bo.handle, xrt.sync_to_device, transfer_size, dst_off) != 0) return error.BoSync;

    const start = nowNs(io);
    const chunks = try d.loopbackChunked(bo.phys + src_off, bo.phys + dst_off, transfer_size, chunk_size);
    const dma_ns = elapsedNs(io, start);

    if (board.x.boSync(bo.handle, xrt.sync_from_device, transfer_size, dst_off) != 0) return error.BoSync;
    if (!std.mem.eql(u8, src, dst)) {
        const mismatch = firstMismatch(src, dst) orelse 0;
        std.debug.print(
            "case={s} ok=0 error=mismatch offset={d} src=0x{x} dst=0x{x}\n",
            .{ case_name, mismatch, src[mismatch], dst[mismatch] },
        );
        return error.Mismatch;
    }

    return .{ .chunks = chunks, .dma_ns = dma_ns };
}

fn printCase(case_name: []const u8, transfer_size: usize, chunk_size: usize, result: RunResult) void {
    std.debug.print(
        "case={s} size_bytes={d} size_kib={d} size_mib={d} chunk_bytes={d} chunk_mib={d} chunks={d} ok=1 dma_us={d} payload_mib_s={d} ddr_mib_s={d}\n",
        .{
            case_name,
            transfer_size,
            sizes.kib(transfer_size),
            sizes.mib(transfer_size),
            chunk_size,
            sizes.mib(chunk_size),
            result.chunks,
            nsToUs(result.dma_ns),
            mibPerSec(transfer_size, result.dma_ns),
            mibPerSec(transfer_size * 2, result.dma_ns),
        },
    );
}

fn fillPattern(buf: []u8, seed: u8) void {
    for (buf, 0..) |*byte, i| {
        byte.* = pattern(i, seed);
    }
}

fn firstMismatch(a: []const u8, b: []const u8) ?usize {
    const len = @min(a.len, b.len);
    for (0..len) |i| {
        if (a[i] != b[i]) return i;
    }
    if (a.len != b.len) return len;
    return null;
}

inline fn pattern(i: usize, seed: u8) u8 {
    return @truncate(i *% 7 +% @as(usize, seed));
}

fn alignUp(value: usize, alignment: usize) usize {
    return std.mem.alignForward(usize, value, alignment);
}

fn nowNs(io: std.Io) u64 {
    const ns = std.Io.Timestamp.now(io, .awake).nanoseconds;
    if (ns <= 0) return 0;
    return @intCast(ns);
}

fn elapsedNs(io: std.Io, start: u64) u64 {
    const now = nowNs(io);
    if (now <= start) return 0;
    return now - start;
}

fn nsToUs(ns: u64) u64 {
    return ns / 1000;
}

fn mibPerSec(bytes: usize, elapsed_ns: u64) u64 {
    if (elapsed_ns == 0) return 0;
    const numerator = @as(u128, bytes) * 1_000_000_000;
    const denominator = @as(u128, elapsed_ns) * sizes.MiB;
    return @intCast(numerator / denominator);
}
