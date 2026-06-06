//! KR260 HP0 128-bit DDR bandwidth benchmark.

const std = @import("std");
const build_options = @import("build_options");
const config = @import("config.zig");
const dma = @import("dma.zig");
const regs = @import("regs.zig");
const sizes = @import("sizes.zig");
const xrt = @import("xrt.zig");

const Options = struct {
    bo_bytes: ?usize = null,
    size_bytes: ?usize = null,
    chunk_bytes: usize = config.default_chunk_size,
    seed: u8 = config.default_seed,
};

const Board = struct {
    x: xrt.Xrt,
    dev: xrt.DeviceHandle,

    fn open() !Board {
        var x = try xrt.Xrt.open();
        errdefer x.close();

        const dev = x.deviceOpen(0);
        if (dev == null) return error.DeviceOpen;
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
            return error.BoAlloc;
        };
        errdefer _ = board.x.boFree(bo);

        const mem: [*]u8 = @ptrCast(board.x.boMap(bo) orelse return error.BoMap);
        return .{
            .handle = bo,
            .phys = board.x.boAddress(bo),
            .mem = mem,
            .size = size,
        };
    }

    fn free(self: *MappedBo, board: *Board) void {
        _ = board.x.boFree(self.handle);
        self.* = undefined;
    }
};

const TransferResult = struct {
    elapsed_ns: u64,
    chunks: usize,
    cycles: u64,
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;

    var args = try std.process.Args.Iterator.initAllocator(init.minimal.args, allocator);
    defer args.deinit();
    _ = args.next();

    const command = args.next() orelse "run";
    if (std.mem.eql(u8, command, "help") or std.mem.eql(u8, command, "--help") or
        std.mem.eql(u8, command, "-h"))
    {
        printUsage();
        return;
    }

    const opts = parseOptions(&args) catch |err| {
        std.debug.print("case=args ok=0 error={s}\n", .{@errorName(err)});
        printUsage();
        return err;
    };

    if (std.mem.eql(u8, command, "run")) {
        try runBenchmark(init.io, opts, false);
    } else if (std.mem.eql(u8, command, "verify")) {
        try runBenchmark(init.io, opts, true);
    } else {
        std.debug.print("case=args ok=0 error=unknown_command command={s}\n", .{command});
        printUsage();
        return error.UnknownCommand;
    }
}

fn printUsage() void {
    std.debug.print(
        \\usage: kr260-xrt-ddr-bandwidth-singleport <command> [options]
        \\
        \\commands:
        \\  run      fixed 384MiB write/read benchmark (default)
        \\  verify   small 4KiB write/read smoke
        \\
        \\options:
        \\  --bo SIZE       backing BO size, default 768MiB for run
        \\  --size SIZE     transfer size, default 384MiB for run
        \\  --chunk SIZE    DMA chunk size, default 32MiB
        \\  --seed N        byte pattern seed, default 1
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
        } else if (std.mem.eql(u8, arg, "--chunk")) {
            opts.chunk_bytes = try sizes.parse(args.next() orelse return error.MissingValue);
        } else if (std.mem.startsWith(u8, arg, "--chunk=")) {
            opts.chunk_bytes = try sizes.parse(arg["--chunk=".len..]);
        } else if (std.mem.eql(u8, arg, "--seed")) {
            opts.seed = try parseSeed(args.next() orelse return error.MissingValue);
        } else if (std.mem.startsWith(u8, arg, "--seed=")) {
            opts.seed = try parseSeed(arg["--seed=".len..]);
        } else {
            return error.UnknownOption;
        }
    }
    return opts;
}

fn parseSeed(text: []const u8) !u8 {
    return std.fmt.parseInt(u8, text, 0);
}

fn runBenchmark(io: std.Io, opts: Options, smoke: bool) !void {
    const transfer_size = opts.size_bytes orelse if (smoke) config.smoke_transfer_size else config.default_transfer_size;
    const bo_size = opts.bo_bytes orelse if (smoke) @max(config.smoke_bo_size, alignUp(transfer_size, sizes.MiB)) else config.default_bo_size;
    try validateRunOptions(bo_size, transfer_size, opts.chunk_bytes);

    var board = Board.open() catch |err| {
        std.debug.print("case=info variant={s} xrt_device_open=0 error={s}\n", .{ build_options.variant, @errorName(err) });
        return err;
    };
    defer board.close();

    var d = dma.Dma.openDevice() catch |err| {
        std.debug.print("case=info variant={s} dma_open=0 error={s}\n", .{ build_options.variant, @errorName(err) });
        return err;
    };
    defer d.closeDevice();

    var engine = regs.Engine.openDevice() catch |err| {
        std.debug.print("case=info variant={s} regs_open=0 error={s}\n", .{ build_options.variant, @errorName(err) });
        return err;
    };
    defer engine.closeDevice();

    var bo = MappedBo.alloc(&board, bo_size) catch |err| {
        std.debug.print("case=info variant={s} bo_alloc=0 bo_bytes={d} error={s}\n", .{ build_options.variant, bo_size, @errorName(err) });
        return err;
    };
    defer bo.free(&board);

    std.debug.print(
        "case=info variant={s} xrt_device_open=1 dma_open=1 regs_open=1 bo_bytes={d} bo_mib={d} bo_phys=0x{x} size_bytes={d} size_mib={d} chunk_bytes={d} chunk_mib={d} seed={d}\n",
        .{
            build_options.variant,
            bo.size,
            sizes.mib(bo.size),
            bo.phys,
            transfer_size,
            sizes.mib(transfer_size),
            opts.chunk_bytes,
            sizes.mib(opts.chunk_bytes),
            opts.seed,
        },
    );

    const write_result = try runWrite(io, &board, &d, &engine, &bo, transfer_size, opts.chunk_bytes, opts.seed);
    printTransfer("write", transfer_size, write_result, false);

    const read_result = try runRead(io, &board, &d, &engine, &bo, transfer_size, opts.chunk_bytes, opts.seed);
    printTransfer("read", transfer_size, read_result, false);

    std.debug.print("case=summary variant={s} ok=1\n", .{build_options.variant});
}

fn validateRunOptions(bo_size: usize, transfer_size: usize, chunk_size: usize) !void {
    if (bo_size == 0 or transfer_size == 0 or chunk_size == 0) return error.InvalidSize;
    if (transfer_size > bo_size) return error.TransferExceedsBo;
    if (transfer_size % config.data_width_bytes != 0) return error.UnalignedTransfer;
    if (chunk_size % config.data_width_bytes != 0) return error.UnalignedChunk;
    if (chunk_size > config.max_dma_transfer) return error.ChunkExceedsDmaLengthWidth;
}

fn runWrite(
    io: std.Io,
    board: *Board,
    d: *dma.Dma,
    engine: *regs.Engine,
    bo: *MappedBo,
    transfer_size: usize,
    chunk_size: usize,
    seed: u8,
) !TransferResult {
    const dst = bo.mem[0..transfer_size];
    @memset(dst, 0);
    if (board.x.boSync(bo.handle, xrt.sync_to_device, transfer_size, 0) != 0) return error.BoSync;

    try d.reset();
    var chunks: usize = 0;
    var cycles: u64 = 0;
    var offset: usize = 0;
    const start = nowNs(io);
    while (offset < transfer_size) {
        const len = @min(chunk_size, transfer_size - offset);
        try d.startWriteToDdr(bo.phys + @as(u64, @intCast(offset)), len);
        try engine.startWrite(len, offset, seed);
        try d.waitWriteDone();
        try engine.waitWriteDone();
        cycles += engine.snapshot().gen_cycles;
        offset += len;
        chunks += 1;
    }
    const elapsed_ns = elapsedNs(io, start);

    if (board.x.boSync(bo.handle, xrt.sync_from_device, transfer_size, 0) != 0) return error.BoSync;
    if (firstPatternMismatch(dst, seed)) |mismatch| {
        const expected = pattern(mismatch, seed);
        const actual = dst[mismatch];
        std.debug.print(
            "case=write variant={s} ok=0 error=mismatch index={d} expected=0x{x:0>2} actual=0x{x:0>2}\n",
            .{ build_options.variant, mismatch, expected, actual },
        );
        return error.Mismatch;
    }

    return .{ .elapsed_ns = elapsed_ns, .chunks = chunks, .cycles = cycles };
}

fn runRead(
    io: std.Io,
    board: *Board,
    d: *dma.Dma,
    engine: *regs.Engine,
    bo: *MappedBo,
    transfer_size: usize,
    chunk_size: usize,
    seed: u8,
) !TransferResult {
    const src = bo.mem[0..transfer_size];
    fillPattern(src, seed);
    if (board.x.boSync(bo.handle, xrt.sync_to_device, transfer_size, 0) != 0) return error.BoSync;

    try d.reset();
    var chunks: usize = 0;
    var cycles: u64 = 0;
    var offset: usize = 0;
    const start = nowNs(io);
    while (offset < transfer_size) {
        const len = @min(chunk_size, transfer_size - offset);
        try engine.startRead(len, offset, seed);
        try d.startReadFromDdr(bo.phys + @as(u64, @intCast(offset)), len);
        try d.waitReadDone();
        engine.waitReadDone() catch |err| {
            const snap = engine.snapshot();
            if (err == error.CheckerMismatch) {
                std.debug.print(
                    "case=read variant={s} ok=0 error=mismatch index={d} expected=0x{x:0>2} actual=0x{x:0>2} bytes_checked={d}\n",
                    .{ build_options.variant, snap.first_error_index, snap.expected, snap.actual, snap.bytes_checked },
                );
            }
            return err;
        };
        const snap = engine.snapshot();
        if (snap.bytes_checked != len) {
            std.debug.print(
                "case=read variant={s} ok=0 error=short_check offset={d} expected_bytes={d} actual_bytes={d} status=0x{x:0>8}\n",
                .{ build_options.variant, offset, len, snap.bytes_checked, snap.status.raw },
            );
            return error.ShortCheck;
        }
        cycles += snap.check_cycles;
        offset += len;
        chunks += 1;
    }
    const elapsed_ns = elapsedNs(io, start);

    return .{ .elapsed_ns = elapsed_ns, .chunks = chunks, .cycles = cycles };
}

fn printTransfer(test_name: []const u8, transfer_size: usize, result: TransferResult, ddr_double: bool) void {
    const ddr_bytes = if (ddr_double) transfer_size * 2 else transfer_size;
    std.debug.print(
        "case={s} variant={s} size_bytes={d} size_mib={d} chunks={d} ok=1 elapsed_us={d} mib_s={d} ddr_mib_s={d} pl_cycles={d} inferred_clk_mhz_x100={d}\n",
        .{
            test_name,
            build_options.variant,
            transfer_size,
            sizes.mib(transfer_size),
            result.chunks,
            nsToUs(result.elapsed_ns),
            mibPerSec(transfer_size, result.elapsed_ns),
            mibPerSec(ddr_bytes, result.elapsed_ns),
            result.cycles,
            inferredMhzX100(result.cycles, result.elapsed_ns),
        },
    );
}

fn fillPattern(buf: []u8, seed: u8) void {
    for (buf, 0..) |*byte, i| {
        byte.* = pattern(i, seed);
    }
}

fn firstPatternMismatch(buf: []const u8, seed: u8) ?usize {
    for (buf, 0..) |byte, i| {
        if (byte != pattern(i, seed)) return i;
    }
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

fn inferredMhzX100(cycles: u64, elapsed_ns: u64) u64 {
    if (elapsed_ns == 0) return 0;
    return @intCast((@as(u128, cycles) * 100_000) / elapsed_ns);
}
