//! KR260 four-HP-port 128-bit DDR bandwidth benchmark.

const std = @import("std");
const build_options = @import("build_options");
const config = @import("config.zig");
const dma = @import("dma.zig");
const regs = @import("regs.zig");
const sizes = @import("sizes.zig");
const xrt = @import("xrt.zig");

const RunMode = enum {
    both,
    read,
    write,

    fn includesRead(self: RunMode) bool {
        return self == .both or self == .read;
    }

    fn includesWrite(self: RunMode) bool {
        return self == .both or self == .write;
    }

    fn label(self: RunMode) []const u8 {
        return switch (self) {
            .both => "both",
            .read => "read",
            .write => "write",
        };
    }
};

const Options = struct {
    ports: usize = config.lane_count,
    ports_set: bool = false,
    active: ?ActiveSet = null,
    offsets: ?OffsetList = null,
    bo_bytes: ?usize = null,
    size_bytes: ?usize = null,
    chunk_bytes: usize = config.default_chunk_size,
    repeat: usize = 1,
    mode: RunMode = .both,
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

const Lane = struct {
    index: usize,
    d: dma.Dma,
    engine: regs.Engine,

    fn cfg(self: Lane) config.LaneConfig {
        return config.lanes[self.index];
    }

    fn close(self: *Lane) void {
        self.engine.closeDevice();
        self.d.closeDevice();
    }
};

const ActiveSet = struct {
    indices: [config.lane_count]usize = [_]usize{0} ** config.lane_count,
    count: usize = 0,

    fn first(count: usize) ActiveSet {
        var active: ActiveSet = .{};
        var i: usize = 0;
        while (i < count) : (i += 1) active.indices[i] = i;
        active.count = count;
        return active;
    }

    fn single(index: usize) ActiveSet {
        var active: ActiveSet = .{};
        active.indices[0] = index;
        active.count = 1;
        return active;
    }

    fn add(self: *ActiveSet, index: usize) !void {
        if (self.count >= config.lane_count) return error.TooManyLanes;
        if (self.contains(index)) return error.DuplicateLane;
        self.indices[self.count] = index;
        self.count += 1;
    }

    fn contains(self: *const ActiveSet, index: usize) bool {
        for (self.slice()) |existing| {
            if (existing == index) return true;
        }
        return false;
    }

    fn slice(self: *const ActiveSet) []const usize {
        return self.indices[0..self.count];
    }
};

const OffsetList = struct {
    values: [config.lane_count]usize = [_]usize{0} ** config.lane_count,
    count: usize = 0,
};

const RunLayout = struct {
    active: ActiveSet,
    offsets: [config.lane_count]usize,
};

const LaneResult = struct {
    chunks: usize = 0,
    cycles: u64 = 0,
    bytes_checked: u64 = 0,
};

const TransferResult = struct {
    elapsed_ns: u64,
    lanes: [config.lane_count]LaneResult,
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
        \\usage: kr260-xrt-ddr-bandwidth-multiport <command> [options]
        \\
        \\commands:
        \\  run      per-port and aggregate HP0-HP3 benchmark (default)
        \\  verify   small per-port and aggregate smoke
        \\
        \\options:
        \\  --ports N       active HP ports from hp0 upward, default 4
        \\  --active LIST   comma-separated lanes, e.g. hp0,hp3; overrides default run shape
        \\  --offsets LIST  comma-separated BO offsets for active lanes, e.g. 0MiB,384MiB
        \\  --bo SIZE       backing BO size, default 768MiB for run
        \\  --size SIZE     per-port transfer size, default 192MiB for run
        \\  --chunk SIZE    DMA chunk size, default 32MiB
        \\  --read-only     run only DDR -> PL checker cases
        \\  --write-only    run only PL generator -> DDR cases
        \\  --repeat N      repeat selected benchmark cases, default 1
        \\
        \\SIZE accepts B, KiB, MiB, GiB suffixes.
        \\
    , .{});
}

fn parseOptions(args: *std.process.Args.Iterator) !Options {
    var opts: Options = .{};
    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "--ports")) {
            opts.ports_set = true;
            opts.ports = try parsePorts(args.next() orelse return error.MissingValue);
        } else if (std.mem.startsWith(u8, arg, "--ports=")) {
            opts.ports_set = true;
            opts.ports = try parsePorts(arg["--ports=".len..]);
        } else if (std.mem.eql(u8, arg, "--active")) {
            opts.active = try parseActive(args.next() orelse return error.MissingValue);
        } else if (std.mem.startsWith(u8, arg, "--active=")) {
            opts.active = try parseActive(arg["--active=".len..]);
        } else if (std.mem.eql(u8, arg, "--offsets")) {
            opts.offsets = try parseOffsets(args.next() orelse return error.MissingValue);
        } else if (std.mem.startsWith(u8, arg, "--offsets=")) {
            opts.offsets = try parseOffsets(arg["--offsets=".len..]);
        } else if (std.mem.eql(u8, arg, "--bo")) {
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
        } else if (std.mem.eql(u8, arg, "--read-only")) {
            if (opts.mode == .write) return error.ModeConflict;
            opts.mode = .read;
        } else if (std.mem.eql(u8, arg, "--write-only")) {
            if (opts.mode == .read) return error.ModeConflict;
            opts.mode = .write;
        } else if (std.mem.eql(u8, arg, "--repeat")) {
            opts.repeat = try parseRepeat(args.next() orelse return error.MissingValue);
        } else if (std.mem.startsWith(u8, arg, "--repeat=")) {
            opts.repeat = try parseRepeat(arg["--repeat=".len..]);
        } else {
            return error.UnknownOption;
        }
    }

    if (opts.active != null and opts.ports_set) return error.ActiveConflictsWithPorts;
    return opts;
}

fn parsePorts(text: []const u8) !usize {
    const value = try std.fmt.parseInt(usize, text, 10);
    if (value == 0 or value > config.lane_count) return error.InvalidPortCount;
    return value;
}

fn parseRepeat(text: []const u8) !usize {
    const value = try std.fmt.parseInt(usize, text, 10);
    if (value == 0) return error.InvalidRepeat;
    return value;
}

fn parseActive(text: []const u8) !ActiveSet {
    const trimmed = std.mem.trim(u8, text, " \t\r\n");
    if (trimmed.len == 0) return error.InvalidActiveList;
    if (std.ascii.eqlIgnoreCase(trimmed, "all")) return ActiveSet.first(config.lane_count);

    var active: ActiveSet = .{};
    var parts = std.mem.splitScalar(u8, trimmed, ',');
    while (parts.next()) |part| {
        const name = std.mem.trim(u8, part, " \t\r\n");
        if (name.len == 0) return error.InvalidActiveList;
        try active.add(try parseLaneName(name));
    }
    if (active.count == 0) return error.InvalidActiveList;
    return active;
}

fn parseLaneName(text: []const u8) !usize {
    for (config.lanes, 0..) |lane, index| {
        if (std.ascii.eqlIgnoreCase(text, lane.name)) return index;
    }

    if (text.len == 1 and text[0] >= '0' and text[0] <= '3') {
        return text[0] - '0';
    }

    return error.InvalidLaneName;
}

fn parseOffsets(text: []const u8) !OffsetList {
    const trimmed = std.mem.trim(u8, text, " \t\r\n");
    if (trimmed.len == 0) return error.InvalidOffsetList;

    var offsets: OffsetList = .{};
    var parts = std.mem.splitScalar(u8, trimmed, ',');
    while (parts.next()) |part| {
        if (offsets.count >= config.lane_count) return error.TooManyOffsets;
        const item = std.mem.trim(u8, part, " \t\r\n");
        if (item.len == 0) return error.InvalidOffsetList;
        offsets.values[offsets.count] = try sizes.parse(item);
        offsets.count += 1;
    }
    if (offsets.count == 0) return error.InvalidOffsetList;
    return offsets;
}

fn runBenchmark(io: std.Io, opts: Options, smoke: bool) !void {
    const transfer_size = opts.size_bytes orelse if (smoke) config.smoke_transfer_size else config.default_transfer_size;
    const default_active = opts.active orelse ActiveSet.first(opts.ports);
    const default_layout = try makeLayout(default_active, transfer_size, opts.offsets);
    const default_bo = if (smoke) @max(config.smoke_bo_size, alignUp(layoutEnd(default_layout, transfer_size), sizes.MiB)) else config.default_bo_size;
    const bo_size = opts.bo_bytes orelse default_bo;
    try validateRunOptions(default_layout, bo_size, transfer_size, opts.chunk_bytes);

    var board = Board.open() catch |err| {
        std.debug.print("case=info variant={s} xrt_device_open=0 error={s}\n", .{ build_options.variant, @errorName(err) });
        return err;
    };
    defer board.close();

    var lane_storage: [config.lane_count]?Lane = [_]?Lane{null} ** config.lane_count;
    try openLanes(&lane_storage, default_active);
    defer closeLanes(&lane_storage);

    var bo = MappedBo.alloc(&board, bo_size) catch |err| {
        std.debug.print("case=info variant={s} bo_alloc=0 bo_bytes={d} error={s}\n", .{ build_options.variant, bo_size, @errorName(err) });
        return err;
    };
    defer bo.free(&board);

    std.debug.print(
        "case=info variant={s} ports={d} active=",
        .{ build_options.variant, default_active.count },
    );
    printActive(default_active);
    std.debug.print(
        " mode={s} repeat={d} xrt_device_open=1 lanes_open=1 bo_bytes={d} bo_mib={d} bo_phys=0x{x} size_bytes_each={d} size_mib_each={d} chunk_bytes={d} chunk_mib={d} offsets=",
        .{
            opts.mode.label(),
            opts.repeat,
            bo.size,
            sizes.mib(bo.size),
            bo.phys,
            transfer_size,
            sizes.mib(transfer_size),
            opts.chunk_bytes,
            sizes.mib(opts.chunk_bytes),
        },
    );
    printOffsets(default_layout);
    std.debug.print("\n", .{});

    var iter: usize = 0;
    while (iter < opts.repeat) : (iter += 1) {
        if (opts.active != null) {
            try runCase(io, &board, &lane_storage, &bo, default_layout, transfer_size, opts.chunk_bytes, opts.mode, iter, opts.repeat);
        } else {
            var lane_index: usize = 0;
            while (lane_index < opts.ports) : (lane_index += 1) {
                const layout = try makeLayout(ActiveSet.single(lane_index), transfer_size, null);
                try runCase(io, &board, &lane_storage, &bo, layout, transfer_size, opts.chunk_bytes, opts.mode, iter, opts.repeat);
            }

            if (opts.ports > 1) {
                const layout = try makeLayout(ActiveSet.first(opts.ports), transfer_size, opts.offsets);
                try runCase(io, &board, &lane_storage, &bo, layout, transfer_size, opts.chunk_bytes, opts.mode, iter, opts.repeat);
            }
        }
    }

    std.debug.print("case=summary variant={s} ports={d} active=", .{ build_options.variant, default_active.count });
    printActive(default_active);
    std.debug.print(" mode={s} repeat={d} ok=1\n", .{ opts.mode.label(), opts.repeat });
}

fn runCase(
    io: std.Io,
    board: *Board,
    lane_storage: *[config.lane_count]?Lane,
    bo: *MappedBo,
    layout: RunLayout,
    transfer_size: usize,
    chunk_size: usize,
    mode: RunMode,
    iter: usize,
    repeat: usize,
) !void {
    if (mode.includesWrite()) {
        const write_result = try runWrite(io, board, lane_storage, bo, layout, transfer_size, chunk_size);
        printTransfer("write", layout, transfer_size, write_result, iter, repeat);
    }

    if (mode.includesRead()) {
        const read_result = try runRead(io, board, lane_storage, bo, layout, transfer_size, chunk_size);
        printTransfer("read", layout, transfer_size, read_result, iter, repeat);
    }
}

fn makeLayout(active: ActiveSet, transfer_size: usize, maybe_offsets: ?OffsetList) !RunLayout {
    var offsets = [_]usize{0} ** config.lane_count;
    for (active.slice()) |idx| {
        offsets[idx] = defaultLaneOffset(idx, transfer_size);
    }

    if (maybe_offsets) |custom| {
        if (custom.count == active.count) {
            for (active.slice(), 0..) |idx, pos| {
                offsets[idx] = custom.values[pos];
            }
        } else if (custom.count == config.lane_count) {
            for (active.slice()) |idx| {
                offsets[idx] = custom.values[idx];
            }
        } else {
            return error.InvalidOffsetCount;
        }
    }

    return .{
        .active = active,
        .offsets = offsets,
    };
}

fn layoutEnd(layout: RunLayout, transfer_size: usize) usize {
    var end: usize = 0;
    for (layout.active.slice()) |idx| {
        end = @max(end, layout.offsets[idx] + transfer_size);
    }
    return end;
}

fn openLanes(storage: *[config.lane_count]?Lane, active: ActiveSet) !void {
    errdefer closeLanes(storage);

    for (active.slice()) |i| {
        const cfg = config.lanes[i];
        var d = dma.Dma.openDevice(cfg.dma_base) catch |err| {
            std.debug.print("case=info variant={s} lane={s} dma_open=0 base=0x{x} error={s}\n", .{ build_options.variant, cfg.name, @as(u64, @intCast(cfg.dma_base)), @errorName(err) });
            return err;
        };

        const engine = regs.Engine.openDevice(cfg.regs_base) catch |err| {
            d.closeDevice();
            std.debug.print("case=info variant={s} lane={s} regs_open=0 base=0x{x} error={s}\n", .{ build_options.variant, cfg.name, @as(u64, @intCast(cfg.regs_base)), @errorName(err) });
            return err;
        };

        storage[i] = .{
            .index = i,
            .d = d,
            .engine = engine,
        };
    }
}

fn closeLanes(storage: *[config.lane_count]?Lane) void {
    var i: usize = config.lane_count;
    while (i > 0) {
        i -= 1;
        if (storage[i]) |*lane| lane.close();
    }
}

fn laneAt(storage: *[config.lane_count]?Lane, index: usize) !*Lane {
    if (storage[index]) |*lane| return lane;
    return error.LaneNotOpen;
}

fn validateRunOptions(layout: RunLayout, bo_size: usize, transfer_size: usize, chunk_size: usize) !void {
    if (layout.active.count == 0 or layout.active.count > config.lane_count) return error.InvalidPortCount;
    if (bo_size == 0 or transfer_size == 0 or chunk_size == 0) return error.InvalidSize;
    if (transfer_size % config.data_width_bytes != 0) return error.UnalignedTransfer;
    if (chunk_size % config.data_width_bytes != 0) return error.UnalignedChunk;
    if (chunk_size > config.max_dma_transfer) return error.ChunkExceedsDmaLengthWidth;

    for (layout.active.slice(), 0..) |idx, pos| {
        const offset = layout.offsets[idx];
        if (offset % config.data_width_bytes != 0) return error.UnalignedOffset;
        const end = std.math.add(usize, offset, transfer_size) catch return error.TransferExceedsBo;
        if (end > bo_size) return error.TransferExceedsBo;

        for (layout.active.slice()[0..pos]) |prev_idx| {
            const prev_offset = layout.offsets[prev_idx];
            const prev_end = prev_offset + transfer_size;
            if (offset < prev_end and prev_offset < end) return error.OverlappingWindows;
        }
    }
}

fn runWrite(
    io: std.Io,
    board: *Board,
    lane_storage: *[config.lane_count]?Lane,
    bo: *MappedBo,
    layout: RunLayout,
    transfer_size: usize,
    chunk_size: usize,
) !TransferResult {
    var result: TransferResult = .{
        .elapsed_ns = 0,
        .lanes = [_]LaneResult{.{}} ** config.lane_count,
    };

    for (layout.active.slice()) |idx| {
        const lane = try laneAt(lane_storage, idx);
        const base = laneBoOffset(layout, idx);
        const dst = bo.mem[base .. base + transfer_size];
        @memset(dst, 0);
        if (board.x.boSync(bo.handle, xrt.sync_to_device, transfer_size, base) != 0) return error.BoSync;
        try lane.d.reset();
    }

    var offset: usize = 0;
    const start = nowNs(io);
    while (offset < transfer_size) {
        const len = @min(chunk_size, transfer_size - offset);

        for (layout.active.slice()) |idx| {
            const lane = try laneAt(lane_storage, idx);
            const phys = bo.phys + @as(u64, @intCast(laneBoOffset(layout, idx) + offset));
            try lane.d.startWriteToDdr(phys, len);
        }
        for (layout.active.slice()) |idx| {
            const lane = try laneAt(lane_storage, idx);
            try lane.engine.startWrite(len, offset, lane.cfg().seed);
        }
        for (layout.active.slice()) |idx| {
            const lane = try laneAt(lane_storage, idx);
            lane.d.waitWriteDone() catch |err| {
                printDmaFailure("write", lane);
                return err;
            };
        }
        for (layout.active.slice()) |idx| {
            const lane = try laneAt(lane_storage, idx);
            lane.engine.waitWriteDone() catch |err| {
                printEngineFailure("write", lane);
                return err;
            };
            result.lanes[idx].cycles += lane.engine.snapshot().gen_cycles;
            result.lanes[idx].chunks += 1;
        }

        offset += len;
    }
    result.elapsed_ns = elapsedNs(io, start);

    for (layout.active.slice()) |idx| {
        const lane = try laneAt(lane_storage, idx);
        const cfg = lane.cfg();
        const base = laneBoOffset(layout, idx);
        if (board.x.boSync(bo.handle, xrt.sync_from_device, transfer_size, base) != 0) return error.BoSync;

        const dst = bo.mem[base .. base + transfer_size];
        if (firstPatternMismatch(dst, cfg.seed)) |mismatch| {
            const expected = pattern(mismatch, cfg.seed);
            const actual = dst[mismatch];
            const ds = lane.d.snapshot();
            const snap = lane.engine.snapshot();
            std.debug.print(
                "case=write variant={s} lane={s} ok=0 error=mismatch index={d} expected=0x{x:0>2} actual=0x{x:0>2} dma_mm2s=0x{x:0>8} dma_s2mm=0x{x:0>8} engine_status=0x{x:0>8}\n",
                .{ build_options.variant, cfg.name, mismatch, expected, actual, ds.mm2s, ds.s2mm, snap.status.raw },
            );
            return error.Mismatch;
        }
        result.lanes[idx].bytes_checked = @intCast(transfer_size);
    }

    return result;
}

fn runRead(
    io: std.Io,
    board: *Board,
    lane_storage: *[config.lane_count]?Lane,
    bo: *MappedBo,
    layout: RunLayout,
    transfer_size: usize,
    chunk_size: usize,
) !TransferResult {
    var result: TransferResult = .{
        .elapsed_ns = 0,
        .lanes = [_]LaneResult{.{}} ** config.lane_count,
    };

    for (layout.active.slice()) |idx| {
        const lane = try laneAt(lane_storage, idx);
        const cfg = lane.cfg();
        const base = laneBoOffset(layout, idx);
        const src = bo.mem[base .. base + transfer_size];
        fillPattern(src, cfg.seed);
        if (board.x.boSync(bo.handle, xrt.sync_to_device, transfer_size, base) != 0) return error.BoSync;
        try lane.d.reset();
    }

    var offset: usize = 0;
    const start = nowNs(io);
    while (offset < transfer_size) {
        const len = @min(chunk_size, transfer_size - offset);

        for (layout.active.slice()) |idx| {
            const lane = try laneAt(lane_storage, idx);
            try lane.engine.startRead(len, offset, lane.cfg().seed);
        }
        for (layout.active.slice()) |idx| {
            const lane = try laneAt(lane_storage, idx);
            const phys = bo.phys + @as(u64, @intCast(laneBoOffset(layout, idx) + offset));
            try lane.d.startReadFromDdr(phys, len);
        }
        for (layout.active.slice()) |idx| {
            const lane = try laneAt(lane_storage, idx);
            lane.d.waitReadDone() catch |err| {
                printDmaFailure("read", lane);
                return err;
            };
        }
        for (layout.active.slice()) |idx| {
            const lane = try laneAt(lane_storage, idx);
            lane.engine.waitReadDone() catch |err| {
                const snap = lane.engine.snapshot();
                if (err == error.CheckerMismatch) {
                    const cfg = lane.cfg();
                    const ds = lane.d.snapshot();
                    const engine_cfg = lane.engine.configSnapshot();
                    const lane_base = laneBoOffset(layout, idx);
                    const phys = bo.phys + @as(u64, @intCast(lane_base + offset));
                    const transfer_error_index: usize = @intCast(@min(snap.first_error_index, @as(u64, @intCast(transfer_size - 1))));
                    const cpu_byte = bo.mem[lane_base + transfer_error_index];
                    const host_expected = pattern(transfer_error_index, cfg.seed);
                    std.debug.print(
                        "case=read variant={s} lane={s} ok=0 error=mismatch chunk_offset={d} chunk_len={d} index={d} transfer_index={d} lane_offset={d} phys=0x{x} seed=0x{x:0>2} checker_expected=0x{x:0>2} checker_actual=0x{x:0>2} host_expected=0x{x:0>2} cpu_byte=0x{x:0>2} bytes_checked={d} dma_mm2s=0x{x:0>8} dma_s2mm=0x{x:0>8} engine_status=0x{x:0>8} cfg_length_lo=0x{x:0>8} cfg_length_hi=0x{x:0>8} cfg_base_lo=0x{x:0>8} cfg_seed=0x{x:0>2}\n",
                        .{
                            build_options.variant,
                            cfg.name,
                            offset,
                            len,
                            snap.first_error_index,
                            transfer_error_index,
                            lane_base,
                            phys,
                            cfg.seed,
                            snap.expected,
                            snap.actual,
                            host_expected,
                            cpu_byte,
                            snap.bytes_checked,
                            ds.mm2s,
                            ds.s2mm,
                            snap.status.raw,
                            engine_cfg.length_lo,
                            engine_cfg.length_hi,
                            engine_cfg.base_lo,
                            engine_cfg.seed,
                        },
                    );
                } else {
                    printEngineFailure("read", lane);
                }
                return err;
            };

            const snap = lane.engine.snapshot();
            if (snap.bytes_checked != @as(u64, @intCast(len))) {
                const ds = lane.d.snapshot();
                std.debug.print(
                    "case=read variant={s} lane={s} ok=0 error=short_check offset={d} expected_bytes={d} actual_bytes={d} dma_mm2s=0x{x:0>8} dma_s2mm=0x{x:0>8} engine_status=0x{x:0>8}\n",
                    .{ build_options.variant, lane.cfg().name, offset, len, snap.bytes_checked, ds.mm2s, ds.s2mm, snap.status.raw },
                );
                return error.ShortCheck;
            }

            result.lanes[idx].cycles += snap.check_cycles;
            result.lanes[idx].bytes_checked += snap.bytes_checked;
            result.lanes[idx].chunks += 1;
        }

        offset += len;
    }
    result.elapsed_ns = elapsedNs(io, start);

    return result;
}

fn printTransfer(test_name: []const u8, layout: RunLayout, transfer_size: usize, result: TransferResult, iter: usize, repeat: usize) void {
    const aggregate_bytes = transfer_size * layout.active.count;

    std.debug.print(
        "case={s} variant={s} ports={d} active=",
        .{ test_name, build_options.variant, layout.active.count },
    );
    printActive(layout.active);
    if (repeat > 1) std.debug.print(" iter={d}", .{iter + 1});
    std.debug.print(
        " offsets=",
        .{},
    );
    printOffsets(layout);
    std.debug.print(
        " size_bytes_each={d} size_mib_each={d} ok=1 elapsed_us={d} aggregate_mib_s={d}",
        .{
            transfer_size,
            sizes.mib(transfer_size),
            nsToUs(result.elapsed_ns),
            mibPerSec(aggregate_bytes, result.elapsed_ns),
        },
    );

    for (layout.active.slice()) |idx| {
        const lane_result = result.lanes[idx];
        std.debug.print(
            " lane{d}_mib_s={d} lane{d}_chunks={d} lane{d}_pl_cycles={d} lane{d}_inferred_clk_mhz_x100={d} lane{d}_bytes_checked={d}",
            .{
                idx,
                mibPerSec(transfer_size, result.elapsed_ns),
                idx,
                lane_result.chunks,
                idx,
                lane_result.cycles,
                idx,
                inferredMhzX100(lane_result.cycles, result.elapsed_ns),
                idx,
                lane_result.bytes_checked,
            },
        );
    }
    std.debug.print("\n", .{});
}

fn printActive(active: ActiveSet) void {
    for (active.slice(), 0..) |idx, pos| {
        if (pos != 0) std.debug.print(",", .{});
        std.debug.print("{s}", .{config.lanes[idx].name});
    }
}

fn printOffsets(layout: RunLayout) void {
    for (layout.active.slice(), 0..) |idx, pos| {
        if (pos != 0) std.debug.print(",", .{});
        const offset = layout.offsets[idx];
        if (offset % sizes.MiB == 0) {
            std.debug.print("{s}:{d}MiB", .{ config.lanes[idx].name, sizes.mib(offset) });
        } else {
            std.debug.print("{s}:{d}B", .{ config.lanes[idx].name, offset });
        }
    }
}

fn printDmaFailure(test_name: []const u8, lane: *Lane) void {
    const ds = lane.d.snapshot();
    std.debug.print(
        "case={s} variant={s} lane={s} ok=0 error=dma dma_mm2s=0x{x:0>8} dma_s2mm=0x{x:0>8}\n",
        .{ test_name, build_options.variant, lane.cfg().name, ds.mm2s, ds.s2mm },
    );
}

fn printEngineFailure(test_name: []const u8, lane: *Lane) void {
    const snap = lane.engine.snapshot();
    std.debug.print(
        "case={s} variant={s} lane={s} ok=0 error=engine status=0x{x:0>8} gen_cycles={d} check_cycles={d} bytes_checked={d}\n",
        .{ test_name, build_options.variant, lane.cfg().name, snap.status.raw, snap.gen_cycles, snap.check_cycles, snap.bytes_checked },
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

fn laneBoOffset(layout: RunLayout, lane_index: usize) usize {
    return layout.offsets[lane_index];
}

fn defaultLaneOffset(lane_index: usize, transfer_size: usize) usize {
    if (transfer_size == config.default_transfer_size) return config.lanes[lane_index].bo_offset;
    return lane_index * transfer_size;
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
