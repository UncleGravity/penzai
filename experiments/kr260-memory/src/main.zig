//! KR260 XRT BO memory capacity and sync probe.

const std = @import("std");
const meminfo = @import("meminfo.zig");
const sizes = @import("sizes.zig");
const xrt = @import("xrt.zig");

const page_stride = 4096;

const Options = struct {
    min_bytes: usize = 32 * sizes.MiB,
    max_bytes: usize = 768 * sizes.MiB,
    step_bytes: usize = 32 * sizes.MiB,
    chunk_bytes: usize = 32 * sizes.MiB,
    large_bytes: usize = 256 * sizes.MiB,
    max_chunks: usize = 64,
    iters: usize = 3,
    touch: bool = true,
    sync_after_alloc: bool = false,
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

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;

    var args = try std.process.Args.Iterator.initAllocator(init.minimal.args, allocator);
    defer args.deinit();
    _ = args.next();

    const command = args.next() orelse {
        printUsage();
        return;
    };

    if (std.mem.eql(u8, command, "help") or std.mem.eql(u8, command, "--help") or
        std.mem.eql(u8, command, "-h"))
    {
        printUsage();
        return;
    }

    var opts = parseOptions(&args) catch |err| {
        std.debug.print("error=invalid_args detail={s}\n", .{@errorName(err)});
        printUsage();
        return err;
    };
    validateOptions(&opts) catch |err| {
        std.debug.print("error=invalid_options detail={s}\n", .{@errorName(err)});
        return err;
    };

    if (std.mem.eql(u8, command, "info")) {
        try runInfo(init.io);
    } else if (std.mem.eql(u8, command, "single-sweep")) {
        try runSingleSweep(init.io, opts);
    } else if (std.mem.eql(u8, command, "chunked")) {
        try runChunked(allocator, init.io, opts);
    } else if (std.mem.eql(u8, command, "sync-sweep")) {
        try runSyncSweep(init.io, opts);
    } else if (std.mem.eql(u8, command, "fragment")) {
        try runFragment(allocator, init.io, opts);
    } else if (std.mem.eql(u8, command, "all")) {
        try runInfo(init.io);
        try runSingleSweep(init.io, opts);
        try runChunked(allocator, init.io, opts);
        try runSyncSweep(init.io, opts);
        try runFragment(allocator, init.io, opts);
    } else {
        std.debug.print("error=unknown_command command={s}\n", .{command});
        printUsage();
        return error.UnknownCommand;
    }
}

fn printUsage() void {
    std.debug.print(
        \\usage: kr260-memory <command> [options]
        \\
        \\commands:
        \\  info          print XRT/device and /proc/meminfo facts
        \\  single-sweep  allocate one BO at a time from --min to --max
        \\  chunked       allocate repeated --chunk BOs until failure
        \\  sync-sweep    time xrtBOSync to/from device over fixed sizes
        \\  fragment      allocate mixed BOs, free alternating BOs, retry large BO sizes
        \\  all           run all commands with the selected options
        \\
        \\options:
        \\  --min SIZE        default 32MiB
        \\  --max SIZE        default 768MiB
        \\  --step SIZE       default 32MiB
        \\  --chunk SIZE      default 32MiB
        \\  --large SIZE      default 256MiB
        \\  --max-chunks N    default 64
        \\  --iters N         default 3 for sync-sweep
        \\  --touch           sparse-touch mapped BOs during allocation probes (default)
        \\  --no-touch
        \\  --sync            run xrtBOSync after allocation probe touches
        \\  --no-sync         default
        \\
        \\SIZE accepts B, KiB, MiB, GiB suffixes, for example 768MiB.
        \\
    , .{});
}

fn parseOptions(args: *std.process.Args.Iterator) !Options {
    var opts: Options = .{};
    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "--touch")) {
            opts.touch = true;
        } else if (std.mem.eql(u8, arg, "--no-touch")) {
            opts.touch = false;
        } else if (std.mem.eql(u8, arg, "--sync")) {
            opts.sync_after_alloc = true;
        } else if (std.mem.eql(u8, arg, "--no-sync")) {
            opts.sync_after_alloc = false;
        } else if (std.mem.eql(u8, arg, "--min")) {
            opts.min_bytes = try sizes.parse(args.next() orelse return error.MissingValue);
        } else if (std.mem.startsWith(u8, arg, "--min=")) {
            opts.min_bytes = try sizes.parse(arg["--min=".len..]);
        } else if (std.mem.eql(u8, arg, "--max")) {
            opts.max_bytes = try sizes.parse(args.next() orelse return error.MissingValue);
        } else if (std.mem.startsWith(u8, arg, "--max=")) {
            opts.max_bytes = try sizes.parse(arg["--max=".len..]);
        } else if (std.mem.eql(u8, arg, "--step")) {
            opts.step_bytes = try sizes.parse(args.next() orelse return error.MissingValue);
        } else if (std.mem.startsWith(u8, arg, "--step=")) {
            opts.step_bytes = try sizes.parse(arg["--step=".len..]);
        } else if (std.mem.eql(u8, arg, "--chunk")) {
            opts.chunk_bytes = try sizes.parse(args.next() orelse return error.MissingValue);
        } else if (std.mem.startsWith(u8, arg, "--chunk=")) {
            opts.chunk_bytes = try sizes.parse(arg["--chunk=".len..]);
        } else if (std.mem.eql(u8, arg, "--large")) {
            opts.large_bytes = try sizes.parse(args.next() orelse return error.MissingValue);
        } else if (std.mem.startsWith(u8, arg, "--large=")) {
            opts.large_bytes = try sizes.parse(arg["--large=".len..]);
        } else if (std.mem.eql(u8, arg, "--max-chunks")) {
            opts.max_chunks = try parseCount(args.next() orelse return error.MissingValue);
        } else if (std.mem.startsWith(u8, arg, "--max-chunks=")) {
            opts.max_chunks = try parseCount(arg["--max-chunks=".len..]);
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

fn validateOptions(opts: *Options) !void {
    if (opts.min_bytes == 0 or opts.max_bytes == 0 or opts.step_bytes == 0 or
        opts.chunk_bytes == 0 or opts.large_bytes == 0)
        return error.InvalidSize;
    if (opts.min_bytes > opts.max_bytes) return error.InvalidRange;
}

fn runInfo(io: std.Io) !void {
    printMeminfo(io, "info");

    {
        var board = Board.open() catch |err| {
            std.debug.print("case=info xrt_device_open=0 error={s}\n", .{@errorName(err)});
            return err;
        };
        defer board.close();

        std.debug.print("case=info xrt_device_open=1\n", .{});
    }
    printMeminfo(io, "info_after_close");
}

fn printMeminfo(io: std.Io, label: []const u8) void {
    const info = meminfo.read(io) catch |err| {
        std.debug.print("case=meminfo label={s} ok=0 error={s}\n", .{ label, @errorName(err) });
        return;
    };
    std.debug.print(
        "case=meminfo label={s} ok=1 mem_total_bytes={d} mem_available_bytes={d} cma_total_bytes={d} cma_free_bytes={d} buffers_bytes={d} cached_bytes={d}\n",
        .{
            label,
            optU64(info.mem_total),
            optU64(info.mem_available),
            optU64(info.cma_total),
            optU64(info.cma_free),
            optU64(info.buffers),
            optU64(info.cached),
        },
    );
}

fn optU64(value: ?u64) u64 {
    return value orelse 0;
}

fn runSingleSweep(io: std.Io, opts: Options) !void {
    printMeminfo(io, "single_before");
    {
        var board = try Board.open();
        defer board.close();

        var size = opts.min_bytes;
        while (size <= opts.max_bytes) : (size += opts.step_bytes) {
            const ok = try attemptAlloc(io, &board, "single", size, opts.touch, opts.sync_after_alloc);
            if (!ok) break;
            if (opts.max_bytes - size < opts.step_bytes) break;
        }
        printMeminfo(io, "single_after");
    }
    printMeminfo(io, "single_after_close");
}

fn attemptAlloc(
    io: std.Io,
    board: *Board,
    case_name: []const u8,
    size: usize,
    touch: bool,
    do_sync: bool,
) !bool {
    const alloc_start = nowNs(io);
    const bo = board.x.boAlloc(board.dev, size, xrt.flags_normal, xrt.group_default) orelse {
        const alloc_ns = elapsedNs(io, alloc_start);
        printAllocFailure(case_name, size, "bo_alloc", alloc_ns);
        return false;
    };
    defer _ = board.x.boFree(bo);
    const alloc_ns = elapsedNs(io, alloc_start);

    const phys = board.x.boAddress(bo);
    var map_ns: u64 = 0;
    var touch_ns: u64 = 0;
    var sync_to_ns: u64 = 0;
    var sync_from_ns: u64 = 0;

    if (touch or do_sync) {
        const map_start = nowNs(io);
        const mem: [*]u8 = @ptrCast(board.x.boMap(bo) orelse {
            printAllocFailure(case_name, size, "bo_map", alloc_ns);
            return false;
        });
        map_ns = elapsedNs(io, map_start);

        if (touch) {
            const touch_start = nowNs(io);
            touchSparse(mem, size, 0x51);
            touch_ns = elapsedNs(io, touch_start);
        }

        if (do_sync) {
            var sync_start = nowNs(io);
            if (board.x.boSync(bo, xrt.sync_to_device, size, 0) != 0) {
                printAllocFailure(case_name, size, "bo_sync_to_device", alloc_ns);
                return false;
            }
            sync_to_ns = elapsedNs(io, sync_start);

            sync_start = nowNs(io);
            if (board.x.boSync(bo, xrt.sync_from_device, size, 0) != 0) {
                printAllocFailure(case_name, size, "bo_sync_from_device", alloc_ns);
                return false;
            }
            sync_from_ns = elapsedNs(io, sync_start);
        }
    }

    std.debug.print(
        "case={s} size_bytes={d} size_kib={d} size_mib={d} ok=1 phys=0x{x} alloc_us={d} map_us={d} touch_us={d} sync_to_us={d} sync_from_us={d}\n",
        .{
            case_name,
            size,
            sizes.kib(size),
            sizes.mib(size),
            phys,
            nsToUs(alloc_ns),
            nsToUs(map_ns),
            nsToUs(touch_ns),
            nsToUs(sync_to_ns),
            nsToUs(sync_from_ns),
        },
    );
    return true;
}

fn printAllocFailure(case_name: []const u8, size: usize, err: []const u8, elapsed_ns: u64) void {
    std.debug.print(
        "case={s} size_bytes={d} size_kib={d} size_mib={d} ok=0 error={s} elapsed_us={d}\n",
        .{ case_name, size, sizes.kib(size), sizes.mib(size), err, nsToUs(elapsed_ns) },
    );
}

fn runChunked(allocator: std.mem.Allocator, io: std.Io, opts: Options) !void {
    printMeminfo(io, "chunked_before");
    {
        var board = try Board.open();
        defer board.close();

        const handles = try allocator.alloc(xrt.BufferHandle, opts.max_chunks);
        defer allocator.free(handles);
        @memset(handles, null);

        var count: usize = 0;
        var total: usize = 0;
        while (count < opts.max_chunks) : (count += 1) {
            const alloc_start = nowNs(io);
            const bo = board.x.boAlloc(board.dev, opts.chunk_bytes, xrt.flags_normal, xrt.group_default) orelse {
                std.debug.print(
                    "case=chunked event=alloc index={d} ok=0 error=bo_alloc chunk_bytes={d} total_bytes={d} total_mib={d} elapsed_us={d}\n",
                    .{ count, opts.chunk_bytes, total, sizes.mib(total), nsToUs(elapsedNs(io, alloc_start)) },
                );
                break;
            };
            handles[count] = bo;
            total += opts.chunk_bytes;

            var touch_ns: u64 = 0;
            if (opts.touch) {
                const mem: [*]u8 = @ptrCast(board.x.boMap(bo) orelse {
                    std.debug.print("case=chunked event=map index={d} ok=0 error=bo_map\n", .{count});
                    break;
                });
                const touch_start = nowNs(io);
                touchSparse(mem, opts.chunk_bytes, @truncate(count + 1));
                touch_ns = elapsedNs(io, touch_start);
            }

            std.debug.print(
                "case=chunked event=alloc index={d} ok=1 chunk_bytes={d} chunk_mib={d} total_bytes={d} total_mib={d} phys=0x{x} alloc_us={d} touch_us={d}\n",
                .{
                    count,
                    opts.chunk_bytes,
                    sizes.mib(opts.chunk_bytes),
                    total,
                    sizes.mib(total),
                    board.x.boAddress(bo),
                    nsToUs(elapsedNs(io, alloc_start)),
                    nsToUs(touch_ns),
                },
            );
        }

        const allocated = countHandles(handles);
        std.debug.print(
            "case=chunked event=summary chunks={d} total_bytes={d} total_mib={d}\n",
            .{ allocated, allocated * opts.chunk_bytes, sizes.mib(allocated * opts.chunk_bytes) },
        );

        freeHandles(&board, handles);
        printMeminfo(io, "chunked_after");
    }
    printMeminfo(io, "chunked_after_close");
}

fn countHandles(handles: []xrt.BufferHandle) usize {
    var count: usize = 0;
    for (handles) |handle| {
        if (handle != null) count += 1;
    }
    return count;
}

fn freeHandles(board: *Board, handles: []xrt.BufferHandle) void {
    for (handles) |*handle| {
        if (handle.*) |bo| {
            _ = board.x.boFree(bo);
            handle.* = null;
        }
    }
}

fn runSyncSweep(io: std.Io, opts: Options) !void {
    printMeminfo(io, "sync_before");
    {
        var board = try Board.open();
        defer board.close();

        const sync_sizes = [_]usize{
            4 * sizes.KiB,
            64 * sizes.KiB,
            1 * sizes.MiB,
            8 * sizes.MiB,
            32 * sizes.MiB,
            128 * sizes.MiB,
            256 * sizes.MiB,
            512 * sizes.MiB,
            768 * sizes.MiB,
        };

        for (sync_sizes) |size| {
            if (size > opts.max_bytes) continue;
            const bo = board.x.boAlloc(board.dev, size, xrt.flags_normal, xrt.group_default) orelse {
                printAllocFailure("sync_alloc", size, "bo_alloc", 0);
                break;
            };
            defer _ = board.x.boFree(bo);

            const mem: [*]u8 = @ptrCast(board.x.boMap(bo) orelse {
                printAllocFailure("sync_alloc", size, "bo_map", 0);
                break;
            });
            @memset(mem[0..size], @as(u8, 0xA5));

            const to_ns = timeRepeatedSync(io, &board, bo, xrt.sync_to_device, size, opts.iters) catch |err| {
                std.debug.print("case=sync direction=to_device size_bytes={d} ok=0 error={s}\n", .{ size, @errorName(err) });
                continue;
            };
            const from_ns = timeRepeatedSync(io, &board, bo, xrt.sync_from_device, size, opts.iters) catch |err| {
                std.debug.print("case=sync direction=from_device size_bytes={d} ok=0 error={s}\n", .{ size, @errorName(err) });
                continue;
            };

            printSyncResult("to_device", size, opts.iters, to_ns);
            printSyncResult("from_device", size, opts.iters, from_ns);
        }
        printMeminfo(io, "sync_after");
    }
    printMeminfo(io, "sync_after_close");
}

fn timeRepeatedSync(
    io: std.Io,
    board: *Board,
    bo: xrt.BufferHandle,
    direction: c_int,
    size: usize,
    iters: usize,
) !u64 {
    const start = nowNs(io);
    var i: usize = 0;
    while (i < iters) : (i += 1) {
        if (board.x.boSync(bo, direction, size, 0) != 0) return error.BoSync;
    }
    return elapsedNs(io, start);
}

fn printSyncResult(direction: []const u8, size: usize, iters: usize, elapsed_ns: u64) void {
    const mib_s = mibPerSec(size, iters, elapsed_ns);
    std.debug.print(
        "case=sync direction={s} size_bytes={d} size_kib={d} size_mib={d} iters={d} elapsed_us={d} mib_s={d}\n",
        .{ direction, size, sizes.kib(size), sizes.mib(size), iters, nsToUs(elapsed_ns), mib_s },
    );
}

fn runFragment(allocator: std.mem.Allocator, io: std.Io, opts: Options) !void {
    printMeminfo(io, "fragment_before");
    {
        var board = try Board.open();
        defer board.close();

        const handles = try allocator.alloc(xrt.BufferHandle, opts.max_chunks);
        defer allocator.free(handles);
        @memset(handles, null);

        var sizes_by_index = try allocator.alloc(usize, opts.max_chunks);
        defer allocator.free(sizes_by_index);
        @memset(sizes_by_index, 0);

        const pattern = [_]usize{
            opts.chunk_bytes,
            @max(opts.chunk_bytes / 2, sizes.MiB),
            opts.chunk_bytes * 2,
            @max(opts.chunk_bytes / 4, sizes.MiB),
        };

        var i: usize = 0;
        while (i < opts.max_chunks) : (i += 1) {
            const size = pattern[i % pattern.len];
            const bo = board.x.boAlloc(board.dev, size, xrt.flags_normal, xrt.group_default) orelse {
                std.debug.print("case=fragment event=fill index={d} ok=0 error=bo_alloc size_bytes={d}\n", .{ i, size });
                break;
            };
            handles[i] = bo;
            sizes_by_index[i] = size;
            if (opts.touch) {
                const mem: [*]u8 = @ptrCast(board.x.boMap(bo) orelse {
                    std.debug.print("case=fragment event=fill index={d} ok=0 error=bo_map size_bytes={d}\n", .{ i, size });
                    break;
                });
                touchSparse(mem, size, @truncate(i + 3));
            }
            std.debug.print(
                "case=fragment event=fill index={d} ok=1 size_bytes={d} size_mib={d} phys=0x{x}\n",
                .{ i, size, sizes.mib(size), board.x.boAddress(bo) },
            );
        }

        var freed_bytes: usize = 0;
        for (handles, 0..) |*handle, index| {
            if (index % 2 == 0 and handle.* != null) {
                _ = board.x.boFree(handle.*);
                handle.* = null;
                freed_bytes += sizes_by_index[index];
            }
        }
        std.debug.print("case=fragment event=free_alternating freed_bytes={d} freed_mib={d}\n", .{ freed_bytes, sizes.mib(freed_bytes) });

        try runFragmentRetries(io, &board, opts);

        freeHandles(&board, handles);
        printMeminfo(io, "fragment_after");
    }
    printMeminfo(io, "fragment_after_close");
}

fn runFragmentRetries(io: std.Io, board: *Board, opts: Options) !void {
    const retry_sizes = [_]usize{
        64 * sizes.MiB,
        128 * sizes.MiB,
        256 * sizes.MiB,
        512 * sizes.MiB,
    };

    var saw_large = false;
    for (retry_sizes) |retry_size| {
        if (retry_size == opts.large_bytes) saw_large = true;
        _ = try attemptAlloc(io, board, "fragment_retry", retry_size, opts.touch, opts.sync_after_alloc);
    }
    if (!saw_large) {
        _ = try attemptAlloc(io, board, "fragment_retry", opts.large_bytes, opts.touch, opts.sync_after_alloc);
    }
}

fn touchSparse(mem: [*]u8, size: usize, seed: u8) void {
    if (size == 0) return;
    var offset: usize = 0;
    while (offset < size) : (offset += page_stride) {
        mem[offset] = seed +% @as(u8, @truncate(offset / page_stride));
    }
    mem[size - 1] = seed ^ 0xFF;
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

fn mibPerSec(size: usize, iters: usize, elapsed_ns: u64) u64 {
    if (elapsed_ns == 0) return 0;
    const total_bytes = @as(u128, size) * @as(u128, iters);
    const denom = @as(u128, elapsed_ns) * sizes.MiB;
    return @intCast((total_bytes * 1_000_000_000) / denom);
}
