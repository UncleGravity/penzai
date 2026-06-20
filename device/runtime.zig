const std = @import("std");
const build_options = @import("build_options");
const shared = @import("shared");
const profile = @import("profile.zig");
const heap_mod = @import("mem/heap.zig");
const ps_activations = @import("ps/activations.zig");
const ps_elemwise = @import("ps/elemwise.zig");
const ps_flash_attn = @import("ps/flash_attn.zig");
const ps_matmul_q1a8 = @import("ps/matmul_q1a8.zig");
const ps_rmsnorm = @import("ps/rmsnorm.zig");
const ps_rows = @import("ps/rows.zig");
const ps_rope = @import("ps/rope.zig");
const ps_select = @import("ps/select.zig");
const ps_pad = @import("ps/pad.zig");
const ps_softmax = @import("ps/softmax.zig");
const pl_matmul = @import("pl/matmul.zig");
const pl_flash = @import("pl/flash_attn.zig");

const wire = shared.wire;
const profiling = shared.profiling;

pub const RuntimeError = error{
    InvalidRequest,
    OutOfMemory,
    UnknownHandle,
    OutOfBounds,
    UnsupportedOp,
    BackendFailure,
};

pub const Runtime = RuntimeFor(heap_mod.Heap);

pub fn RuntimeFor(comptime Heap: type) type {
    return struct {
        const Self = @This();

        // The PL matmul backend exists only for heaps with physical addressing
        // (XRT on the board); the fake heap has none, so it stays PS-only.
        const pl_supported = @hasDecl(Heap, "deviceAddress");
        const PlBackend = if (pl_supported) pl_matmul.Backend(Heap) else void;
        const FlashBackend = if (pl_supported) pl_flash.Backend(Heap) else void;

        allocator: std.mem.Allocator,
        heap: Heap,
        pl: ?PlBackend = null,
        flash: ?FlashBackend = null,
        pl_verify: bool = false,
        /// PL counters from the most recent matmul, consumed by the profiler.
        pl_last: ?profile.PlCounters = null,

        pub fn init(allocator: std.mem.Allocator, heap_size: usize) !Self {
            var self: Self = .{
                .allocator = allocator,
                .heap = try Heap.init(allocator, heap_size),
            };
            if (comptime pl_supported) {
                self.pl_verify = plVerifyEnabled();
                // Only probe op backends whose IP the resident bitstream actually has.
                // A /dev/mem read of an unmapped PL address is a bus fault (DECERR ->
                // SError -> SIGBUS), NOT a catchable Zig error — so the else-branch
                // fallback can't save us; we must avoid touching the address at all.
                if (plOpEnabled("matmul")) {
                    if (PlBackend.init(allocator, &self.heap)) |backend| {
                        self.pl = backend;
                        std.debug.print("pl: q1a8 kernel ready (version {d}, counters {}, verify {})\n", .{
                            backend.kernel.version, backend.hasCounters(), self.pl_verify,
                        });
                    } else |err| {
                        std.debug.print("pl: matmul unavailable ({s}); falling back to PS matmul\n", .{@errorName(err)});
                    }
                }
                if (plOpEnabled("flash")) {
                    if (FlashBackend.init(allocator, &self.heap)) |backend| {
                        self.flash = backend;
                        std.debug.print("pl: flash kernel ready\n", .{});
                    } else |err| {
                        std.debug.print("pl: flash unavailable ({s}); falling back to PS flash\n", .{@errorName(err)});
                    }
                }
            }
            return self;
        }

        pub fn deinit(self: *Self) void {
            if (comptime pl_supported) {
                if (self.pl) |*backend| backend.deinit();
                if (self.flash) |*backend| backend.deinit();
            }
            self.heap.deinit();
            self.* = undefined;
        }

        pub fn dispatch(self: *Self, request: wire.Request, io: ?std.Io) RuntimeError!DispatchResult {
            return switch (request) {
                .hello => |request_id| .{ .meta = ok(request_id) },
                .alloc => |req| blk: {
                    const range = self.heap.allocate(req.nbytes, req.alignment) catch |err| switch (err) {
                        error.OutOfMemory => return error.OutOfMemory,
                        error.InvalidAlignment => return error.InvalidRequest,
                        else => return mapHeapError(err),
                    };
                    break :blk .{ .meta = .{
                        .request_id = req.request_id,
                        .status = .ok,
                        .handle = range.handle,
                        .nbytes = range.nbytes,
                    } };
                },
                .free => |req| blk: {
                    self.heap.free(req.handle) catch |err| return mapHeapError(err);
                    break :blk .{ .meta = ok(req.request_id) };
                },
                .upload => |req| blk: {
                    self.heap.write(req.range, req.bytes) catch |err| return mapHeapError(err);
                    break :blk .{ .meta = .{
                        .request_id = req.request_id,
                        .status = .ok,
                        .nbytes = req.bytes.len,
                    } };
                },
                .fill => |req| blk: {
                    self.heap.fill(req.range, req.value) catch |err| return mapHeapError(err);
                    break :blk .{ .meta = .{
                        .request_id = req.request_id,
                        .status = .ok,
                        .nbytes = req.range.nbytes,
                    } };
                },
                .download => |req| blk: {
                    const bytes = self.heap.read(req.range) catch |err| return mapHeapError(err);
                    break :blk .{ .meta = .{
                        .request_id = req.request_id,
                        .status = .ok,
                        .nbytes = bytes.len,
                    }, .payload = bytes };
                },
                .run_graph => |req| blk: {
                    try self.applyPreloads(req.preload_bytes);
                    if (build_options.enable_profiling and req.tier != .off) {
                        const profiled = try self.runProfiled(io, req.command_bytes);
                        break :blk .{ .meta = .{
                            .request_id = req.request_id,
                            .status = .ok,
                            .value0 = profiled.command_count,
                            .nbytes = profiled.payload.len,
                        }, .payload = profiled.payload, .owns_payload = true };
                    }
                    const commands = wire.decodeCommandBuffer(self.allocator, req.command_bytes) catch |err| switch (err) {
                        error.OutOfMemory => return error.OutOfMemory,
                        else => return error.InvalidRequest,
                    };
                    defer self.allocator.free(commands);
                    for (commands) |command| try self.execute(command, io);
                    break :blk .{ .meta = .{
                        .request_id = req.request_id,
                        .status = .ok,
                        .value0 = commands.len,
                    } };
                },
            };
        }

        fn applyPreloads(self: *Self, preload_bytes: []const u8) RuntimeError!void {
            var it: wire.PreloadIterator = .{ .bytes = preload_bytes };
            while (it.next() catch return error.InvalidRequest) |entry| {
                self.heap.write(entry.range, entry.bytes) catch |err| return mapHeapError(err);
            }
        }

        const ProfiledRun = struct { payload: []u8, command_count: usize };

        /// Execute a graph while timing each command and accruing a profile.
        /// Decode + execution stay here (runtime's job); the byte/span/aggregate
        /// bookkeeping and payload encoding live in `profile.Collector`.
        fn runProfiled(self: *Self, io: ?std.Io, command_bytes: []const u8) RuntimeError!ProfiledRun {
            const request_start_ns = profiling.nowNs(io);
            const commands = wire.decodeCommandBuffer(self.allocator, command_bytes) catch |err| switch (err) {
                error.OutOfMemory => return error.OutOfMemory,
                else => return error.InvalidRequest,
            };
            defer self.allocator.free(commands);
            const decode_done_ns = profiling.nowNs(io);

            var collector: profile.Collector = .{};

            const execute_start_ns = profiling.nowNs(io);
            for (commands) |command| {
                const start_ns = profiling.nowNs(io);
                self.pl_last = null;
                try self.execute(command, io);
                const end_ns = profiling.nowNs(io);
                collector.record(command, start_ns, end_ns, self.pl_last);
            }
            const execute_done_ns = profiling.nowNs(io);

            const fclk_hz: u32 = if (comptime pl_supported)
                (if (self.pl) |b| b.kernel.clk_hz else 0)
            else
                0;
            const payload = collector.encode(self.allocator, .{
                .device_total_ns = profiling.elapsed(request_start_ns, execute_done_ns),
                .decode_ns = profiling.elapsed(request_start_ns, decode_done_ns),
                .execute_ns = profiling.elapsed(execute_start_ns, execute_done_ns),
                .command_count = @intCast(commands.len),
                .device_fclk_hz = fclk_hz,
            }) catch return error.OutOfMemory;
            return .{ .payload = payload, .command_count = commands.len };
        }

        /// Debug cross-check (PENZAI_PL_VERIFY=1): run the PS oracle into scratch
        /// and compare against the PL result already in `dst`. Logs mismatches
        /// beyond the documented fp tolerance; never changes results.
        fn verifyMatmul(self: *Self, mm: wire.MatmulQ1A8) void {
            const weights = self.heap.read(mm.weights) catch return;
            const acts = self.heap.read(mm.acts) catch return;
            const dst = self.heap.bytes(mm.dst) catch return; // holds the PL result
            const scratch = self.allocator.alloc(u8, dst.len) catch return;
            defer self.allocator.free(scratch);
            ps_matmul_q1a8.runQ1A8(self.allocator, weights, acts, scratch, mm.rows, mm.cols, mm.k) catch return;
            var max_abs: f32 = 0;
            var max_rel: f32 = 0;
            var nbad: usize = 0;
            const n = dst.len / @sizeOf(f32);
            for (0..n) |i| {
                const a: f32 = @bitCast(std.mem.readInt(u32, dst[i * 4 ..][0..4], .little));
                const b: f32 = @bitCast(std.mem.readInt(u32, scratch[i * 4 ..][0..4], .little));
                const d = @abs(a - b);
                const rel = d / @max(@abs(b), 1e-6);
                max_abs = @max(max_abs, d);
                max_rel = @max(max_rel, rel);
                if (rel > 0.02 and d > 1e-3) nbad += 1;
            }
            if (nbad > 0) std.debug.print("pl verify rows={d} k={d}: {d}/{d} mismatch (max_abs={d:.4} max_rel={d:.4})\n", .{ mm.rows, mm.k, nbad, n, max_abs, max_rel });
        }

        /// Debug cross-check (PENZAI_PL_VERIFY=1): run the PS flash oracle into
        /// scratch and compare against the PL result already in `dst`. The PL
        /// kernel uses LUT-based exp/recip and a different fp reduction order, so
        /// the tolerance is tighter than matmul's (rel > 0.01 vs 0.02): the
        /// silicon run showed byte-identical output, so the real max_rel is well
        /// under 1%, and a tighter gate catches numerical regressions from the
        /// upcoming pipelined kernel rewrite. Logs mismatches; never changes
        /// results.
        fn verifyFlash(self: *Self, attn: wire.FlashAttnF32) void {
            const q = self.heap.read(attn.q) catch return;
            const k = self.heap.read(attn.k) catch return;
            const v = self.heap.read(attn.v) catch return;
            const mask: ?[]const u8 = if (attn.has_mask)
                (self.heap.read(attn.mask) catch return)
            else
                null;
            const dst = self.heap.bytes(attn.dst) catch return; // holds the PL result
            const scratch = self.allocator.alloc(u8, dst.len) catch return;
            defer self.allocator.free(scratch);
            ps_flash_attn.runBytes(q, k, v, mask, scratch, .{
                .head_dim_q = attn.head_dim_q,
                .head_dim_v = attn.head_dim_v,
                .n_heads = attn.n_heads,
                .n_head_kv = attn.n_head_kv,
                .n_kv = attn.n_kv,
                .n_tokens = attn.n_tokens,
                .scale = attn.scale,
                .q_nb1 = attn.q_nb1,
                .q_nb2 = attn.q_nb2,
                .k_nb1 = attn.k_nb1,
                .k_nb2 = attn.k_nb2,
                .v_nb1 = attn.v_nb1,
                .v_nb2 = attn.v_nb2,
                .mask_nb1 = attn.mask_nb1,
                .dst_nb1 = attn.dst_nb1,
                .dst_nb2 = attn.dst_nb2,
            }) catch return;
            var max_abs: f32 = 0;
            var max_rel: f32 = 0;
            var nbad: usize = 0;
            const n = dst.len / @sizeOf(f32);
            for (0..n) |i| {
                const a: f32 = @bitCast(std.mem.readInt(u32, dst[i * 4 ..][0..4], .little));
                const b: f32 = @bitCast(std.mem.readInt(u32, scratch[i * 4 ..][0..4], .little));
                const d = @abs(a - b);
                const rel = d / @max(@abs(b), 1e-6);
                max_abs = @max(max_abs, d);
                max_rel = @max(max_rel, rel);
                if (rel > 0.01 and d > 1e-3) nbad += 1;
            }
            if (nbad > 0) std.debug.print("pl verify flash hdq={d} nkv={d} ntok={d}: {d}/{d} mismatch (max_abs={d:.4} max_rel={d:.4})\n", .{ attn.head_dim_q, attn.n_kv, attn.n_tokens, nbad, n, max_abs, max_rel });
        }

        fn plVerifyEnabled() bool {
            const raw = std.c.getenv("PENZAI_PL_VERIFY") orelse return false;
            const v = std.mem.span(raw);
            return v.len != 0 and !std.mem.eql(u8, v, "0");
        }

        /// Which op backends to probe on /dev/mem. The KR260 holds one bitstream at a
        /// time, so probing an op whose IP is absent faults fatally (see init). Set
        /// `PENZAI_PL_OPS` to match the resident bitstream:
        ///   unset/"matmul" → matmul only (pre-flash default; the matmul bitstream)
        ///   "flash"        → flash only (the flash bitstream)
        ///   "matmul,flash" / "all" → both (a combined bitstream)
        ///   "none"         → neither (pure PS)
        fn plOpEnabled(op: []const u8) bool {
            const raw = std.c.getenv("PENZAI_PL_OPS") orelse return std.mem.eql(u8, op, "matmul");
            const v = std.mem.span(raw);
            if (std.mem.eql(u8, v, "all")) return true;
            if (std.mem.eql(u8, v, "none")) return false;
            var it = std.mem.tokenizeScalar(u8, v, ',');
            while (it.next()) |tok| {
                if (std.mem.eql(u8, std.mem.trim(u8, tok, " \t"), op)) return true;
            }
            return false;
        }

        fn execute(self: *Self, command: wire.Command, io: ?std.Io) RuntimeError!void {
            switch (command) {
                .copy => |copy| {
                    const src = self.heap.read(copy.src) catch |err| return mapHeapError(err);
                    self.heap.write(copy.dst, src) catch |err| return mapHeapError(err);
                },
                .cpy_f32_to_f16 => |copy| {
                    const src = self.heap.read(copy.src) catch |err| return mapHeapError(err);
                    const dst = self.heap.bytes(copy.dst) catch |err| return mapHeapError(err);
                    ps_elemwise.f32ToF16Bytes(src, dst) catch |err| return mapKernelError(err);
                },
                .matmul_q1a8 => |matmul| {
                    if (comptime pl_supported) {
                        if (self.pl) |*backend| {
                            const maybe = backend.tryMatmul(&self.heap, matmul, io) catch |err| {
                                std.debug.print("pl matmul failed: {s}\n", .{@errorName(err)});
                                return error.BackendFailure;
                            };
                            if (maybe) |counters| {
                                self.pl_last = counters;
                                if (self.pl_verify) self.verifyMatmul(matmul);
                                return;
                            }
                        }
                    }
                    const weights = self.heap.read(matmul.weights) catch |err| return mapHeapError(err);
                    const acts = self.heap.read(matmul.acts) catch |err| return mapHeapError(err);
                    const dst = self.heap.bytes(matmul.dst) catch |err| return mapHeapError(err);
                    ps_matmul_q1a8.runQ1A8(self.allocator, weights, acts, dst, matmul.rows, matmul.cols, matmul.k) catch |err| switch (err) {
                        error.OutOfMemory => return error.OutOfMemory,
                        else => return error.InvalidRequest,
                    };
                },
                .rmsnorm => |rmsnorm| {
                    const input = self.heap.read(rmsnorm.input) catch |err| return mapHeapError(err);
                    const dst = self.heap.bytes(rmsnorm.dst) catch |err| return mapHeapError(err);
                    ps_rmsnorm.runBytes(input, dst, rmsnorm.rows, rmsnorm.cols, rmsnorm.eps) catch |err| return mapKernelError(err);
                },
                .rope => |rope| {
                    const input = self.heap.read(rope.input) catch |err| return mapHeapError(err);
                    const positions = self.heap.read(rope.positions) catch |err| return mapHeapError(err);
                    const dst = self.heap.bytes(rope.dst) catch |err| return mapHeapError(err);
                    ps_rope.applyBytes(input, positions, dst, .{
                        .head_dim = rope.head_dim,
                        .n_heads = rope.n_heads,
                        .n_tokens = rope.n_tokens,
                        .n_dims = rope.n_dims,
                        .mode = ropeMode(rope.mode),
                        .n_ctx_orig = rope.n_ctx_orig,
                        .freq_base = rope.freq_base,
                        .freq_scale = rope.freq_scale,
                        .ext_factor = rope.ext_factor,
                        .attn_factor = rope.attn_factor,
                        .beta_fast = rope.beta_fast,
                        .beta_slow = rope.beta_slow,
                    }) catch |err| return mapKernelError(err);
                },
                .softmax => |softmax| {
                    const src = self.heap.read(softmax.src) catch |err| return mapHeapError(err);
                    const dst = self.heap.bytes(softmax.dst) catch |err| return mapHeapError(err);
                    ps_softmax.runBytes(src, dst) catch |err| return mapKernelError(err);
                },
                .silu => |silu| {
                    const src = self.heap.read(silu.src) catch |err| return mapHeapError(err);
                    const dst = self.heap.bytes(silu.dst) catch |err| return mapHeapError(err);
                    ps_activations.siluBytes(src, dst) catch |err| return mapKernelError(err);
                },
                .swiglu => |swiglu| {
                    const gate = self.heap.read(swiglu.lhs) catch |err| return mapHeapError(err);
                    const up = self.heap.read(swiglu.rhs) catch |err| return mapHeapError(err);
                    const dst = self.heap.bytes(swiglu.dst) catch |err| return mapHeapError(err);
                    ps_activations.swigluBytes(gate, up, dst) catch |err| return mapKernelError(err);
                },
                .add_f32 => |add| {
                    const lhs = self.heap.read(add.lhs) catch |err| return mapHeapError(err);
                    const rhs = self.heap.read(add.rhs) catch |err| return mapHeapError(err);
                    const dst = self.heap.bytes(add.dst) catch |err| return mapHeapError(err);
                    ps_elemwise.add2dBytes(lhs, rhs, dst, add.rows, add.cols, rhsRowBroadcast(add.mode)) catch |err| return mapKernelError(err);
                },
                .mul_f32 => |mul| {
                    const lhs = self.heap.read(mul.lhs) catch |err| return mapHeapError(err);
                    const rhs = self.heap.read(mul.rhs) catch |err| return mapHeapError(err);
                    const dst = self.heap.bytes(mul.dst) catch |err| return mapHeapError(err);
                    ps_elemwise.mul2dBytes(lhs, rhs, dst, mul.rows, mul.cols, rhsRowBroadcast(mul.mode)) catch |err| return mapKernelError(err);
                },
                .scale_f32 => |scale| {
                    const src = self.heap.read(scale.src) catch |err| return mapHeapError(err);
                    const dst = self.heap.bytes(scale.dst) catch |err| return mapHeapError(err);
                    ps_elemwise.scaleBytes(src, scale.scale, dst) catch |err| return mapKernelError(err);
                },
                .add_scaled_f32 => |add_scaled| {
                    const lhs = self.heap.read(add_scaled.lhs) catch |err| return mapHeapError(err);
                    const rhs = self.heap.read(add_scaled.rhs) catch |err| return mapHeapError(err);
                    const dst = self.heap.bytes(add_scaled.dst) catch |err| return mapHeapError(err);
                    ps_elemwise.addScaledBytes(lhs, rhs, add_scaled.rhs_scale, dst) catch |err| return mapKernelError(err);
                },
                .set_rows => |set_rows| {
                    const src = self.heap.read(set_rows.src) catch |err| return mapHeapError(err);
                    const indices = self.heap.read(set_rows.indices) catch |err| return mapHeapError(err);
                    const dst = self.heap.bytes(set_rows.dst) catch |err| return mapHeapError(err);
                    ps_rows.setRowsF32ToF16Bytes(set_rows, src, indices, dst) catch |err| return mapKernelError(err);
                },
                .get_rows => |get_rows| {
                    const src = self.heap.read(get_rows.src) catch |err| return mapHeapError(err);
                    const indices = self.heap.read(get_rows.indices) catch |err| return mapHeapError(err);
                    const dst = self.heap.bytes(get_rows.dst) catch |err| return mapHeapError(err);
                    ps_rows.getRowsF32Bytes(get_rows, src, indices, dst) catch |err| return mapKernelError(err);
                },
                .flash_attn_f32 => |attn| {
                    if (comptime pl_supported) {
                        if (self.flash) |*backend| {
                            const maybe = backend.tryFlashAttn(&self.heap, attn, io) catch |err| {
                                std.debug.print("pl flash failed: {s}\n", .{@errorName(err)});
                                return error.BackendFailure;
                            };
                            if (maybe) |_| {
                                if (self.pl_verify) self.verifyFlash(attn);
                                return;
                            }
                        }
                    }
                    const q = self.heap.read(attn.q) catch |err| return mapHeapError(err);
                    const k = self.heap.read(attn.k) catch |err| return mapHeapError(err);
                    const v = self.heap.read(attn.v) catch |err| return mapHeapError(err);
                    const mask = if (attn.has_mask) self.heap.read(attn.mask) catch |err| return mapHeapError(err) else null;
                    const dst = self.heap.bytes(attn.dst) catch |err| return mapHeapError(err);
                    ps_flash_attn.runBytes(q, k, v, mask, dst, .{
                        .head_dim_q = attn.head_dim_q,
                        .head_dim_v = attn.head_dim_v,
                        .n_heads = attn.n_heads,
                        .n_head_kv = attn.n_head_kv,
                        .n_kv = attn.n_kv,
                        .n_tokens = attn.n_tokens,
                        .scale = attn.scale,
                        .q_nb1 = attn.q_nb1,
                        .q_nb2 = attn.q_nb2,
                        .k_nb1 = attn.k_nb1,
                        .k_nb2 = attn.k_nb2,
                        .v_nb1 = attn.v_nb1,
                        .v_nb2 = attn.v_nb2,
                        .mask_nb1 = attn.mask_nb1,
                        .dst_nb1 = attn.dst_nb1,
                        .dst_nb2 = attn.dst_nb2,
                    }) catch |err| return mapKernelError(err);
                },
                .argmax => |argmax| {
                    const src = self.heap.read(argmax.src) catch |err| return mapHeapError(err);
                    const dst = self.heap.bytes(argmax.dst) catch |err| return mapHeapError(err);
                    ps_select.argmaxF32Bytes(src, dst, argmax.rows, argmax.cols) catch |err| return mapKernelError(err);
                },
                .pad => |pad| {
                    const src = self.heap.read(pad.src) catch |err| return mapHeapError(err);
                    const dst = self.heap.bytes(pad.dst) catch |err| return mapHeapError(err);
                    ps_pad.padZeroTailBytes(src, dst) catch |err| return mapKernelError(err);
                },
            }
        }
    };
}

pub const DispatchResult = struct {
    meta: wire.ResponseMeta,
    payload: []const u8 = &.{},
    owns_payload: bool = false,

    pub fn deinit(self: *DispatchResult, allocator: std.mem.Allocator) void {
        if (self.owns_payload) allocator.free(@constCast(self.payload));
        self.* = undefined;
    }
};

pub fn errorCode(err: RuntimeError) wire.ErrorCode {
    return switch (err) {
        error.InvalidRequest => .invalid_request,
        error.OutOfMemory => .out_of_memory,
        error.UnknownHandle => .unknown_handle,
        error.OutOfBounds => .out_of_bounds,
        error.UnsupportedOp => .unsupported_op,
        error.BackendFailure => .backend_failure,
    };
}

fn ok(request_id: u64) wire.ResponseMeta {
    return .{ .request_id = request_id, .status = .ok };
}

fn rhsRowBroadcast(mode: wire.BinaryF32Mode) bool {
    return switch (mode) {
        .same_shape => false,
        .rhs_row_broadcast => true,
    };
}

fn ropeMode(mode: wire.RopeMode) ps_rope.Mode {
    return switch (mode) {
        .normal => .normal,
        .neox => .neox,
    };
}

fn mapHeapError(err: anyerror) RuntimeError {
    return switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        error.UnknownHandle => error.UnknownHandle,
        error.OutOfBounds => error.OutOfBounds,
        error.InvalidAlignment => error.InvalidRequest,
        error.BackendFailure => error.BackendFailure,
        else => error.BackendFailure,
    };
}

fn mapKernelError(err: anyerror) RuntimeError {
    return switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        else => error.InvalidRequest,
    };
}

test "runtime dispatches ps f32 command variants" {
    var runtime = try Runtime.init(std.testing.allocator, 4096);
    defer runtime.deinit();

    const a = try tensor(&runtime, 2);
    const b = try tensor(&runtime, 2);
    const get_rows_src = try tensor(&runtime, 4);
    const flash_q = try tensor(&runtime, 2);
    const flash_k = try rawTensor(&runtime, 4 * @sizeOf(f16), @alignOf(f16));
    const flash_v = try rawTensor(&runtime, 4 * @sizeOf(f16), @alignOf(f16));
    const out_rms = try tensor(&runtime, 2);
    const out_rope = try tensor(&runtime, 2);
    const out_softmax = try tensor(&runtime, 2);
    const out_silu = try tensor(&runtime, 2);
    const out_swiglu = try tensor(&runtime, 2);
    const out_add = try tensor(&runtime, 2);
    const out_mul = try tensor(&runtime, 2);
    const out_scale = try tensor(&runtime, 2);
    const out_add_scaled = try tensor(&runtime, 2);
    const out_flash = try tensor(&runtime, 2);
    const indices = try rawTensor(&runtime, @sizeOf(i32), @alignOf(i32));
    const out_rows = try rawTensor(&runtime, 4 * @sizeOf(f16), @alignOf(f16));
    const out_get_rows = try tensor(&runtime, 2);
    const out_pad = try tensor(&runtime, 4);
    const out_argmax = try rawTensor(&runtime, @sizeOf(i32), @alignOf(i32));

    try writeTensor(&runtime, a, &.{ 3, 4 });
    try writeTensor(&runtime, b, &.{ 1, 2 });
    try writeTensor(&runtime, get_rows_src, &.{ 1, 2, 3, 4 });
    try writeTensor(&runtime, flash_q, &.{ 1, 0 });
    try writeF16Tensor(&runtime, flash_k, &.{ 1, 0, 0, 1 });
    try writeF16Tensor(&runtime, flash_v, &.{ 10, 20, 30, 40 });
    try writeI32Tensor(&runtime, indices, &.{1});

    const commands = [_]wire.Command{
        .{ .rmsnorm = .{ .input = a, .dst = out_rms, .rows = 2, .cols = 1, .eps = 0 } },
        .{ .rope = .{
            .input = b,
            .positions = indices,
            .dst = out_rope,
            .head_dim = 2,
            .n_heads = 1,
            .n_tokens = 1,
            .n_dims = 2,
            .mode = .normal,
            .n_ctx_orig = 0,
            .freq_base = 10000,
            .freq_scale = 1,
            .ext_factor = 0,
            .attn_factor = 1,
            .beta_fast = 0,
            .beta_slow = 0,
        } },
        .{ .softmax = .{ .src = b, .dst = out_softmax } },
        .{ .silu = .{ .src = b, .dst = out_silu } },
        .{ .swiglu = .{ .lhs = b, .rhs = a, .dst = out_swiglu } },
        .{ .add_f32 = .{ .lhs = a, .rhs = b, .dst = out_add, .rows = 2, .cols = 1, .mode = .same_shape } },
        .{ .mul_f32 = .{ .lhs = a, .rhs = b, .dst = out_mul, .rows = 2, .cols = 1, .mode = .same_shape } },
        .{ .scale_f32 = .{ .src = b, .dst = out_scale, .scale = 0.5 } },
        .{ .add_scaled_f32 = .{ .lhs = a, .rhs = b, .dst = out_add_scaled, .rhs_scale = 0.5 } },
        .{ .set_rows = .{
            .src = a,
            .indices = indices,
            .dst = out_rows,
            .index_type = .i32,
            .head_dim = 2,
            .ne01 = 1,
            .ne02 = 1,
            .ne03 = 1,
            .ne11 = 1,
            .ne12 = 1,
            .src_nb1 = 2 * @sizeOf(f32),
            .src_nb2 = 2 * @sizeOf(f32),
            .src_nb3 = 2 * @sizeOf(f32),
            .indices_nb1 = @sizeOf(i32),
            .indices_nb2 = @sizeOf(i32),
            .dst_nb1 = 2 * @sizeOf(f16),
            .dst_nb2 = 4 * @sizeOf(f16),
            .dst_nb3 = 4 * @sizeOf(f16),
        } },
        .{ .get_rows = .{
            .src = get_rows_src,
            .indices = indices,
            .dst = out_get_rows,
            .src_type = .f32,
            .row_width = 2,
            .src_rows = 2,
            .ne10 = 1,
            .ne11 = 1,
            .ne12 = 1,
            .src_nb1 = 2 * @sizeOf(f32),
            .src_nb2 = 2 * @sizeOf(f32),
            .src_nb3 = 2 * @sizeOf(f32),
            .indices_nb1 = @sizeOf(i32),
            .indices_nb2 = @sizeOf(i32),
            .dst_nb1 = 2 * @sizeOf(f32),
            .dst_nb2 = 2 * @sizeOf(f32),
            .dst_nb3 = 2 * @sizeOf(f32),
        } },
        .{ .flash_attn_f32 = .{
            .q = flash_q,
            .k = flash_k,
            .v = flash_v,
            .mask = .{ .handle = 0, .offset = 0, .nbytes = 0 },
            .dst = out_flash,
            .has_mask = false,
            .head_dim_q = 2,
            .head_dim_v = 2,
            .n_heads = 1,
            .n_head_kv = 1,
            .n_kv = 2,
            .n_tokens = 1,
            .scale = 1,
            .q_nb1 = 2 * @sizeOf(f32),
            .q_nb2 = 2 * @sizeOf(f32),
            .k_nb1 = 2 * @sizeOf(f16),
            .k_nb2 = 4 * @sizeOf(f16),
            .v_nb1 = 2 * @sizeOf(f16),
            .v_nb2 = 4 * @sizeOf(f16),
            .mask_nb1 = 0,
            .dst_nb1 = 2 * @sizeOf(f32),
            .dst_nb2 = 2 * @sizeOf(f32),
        } },
        .{ .pad = .{ .src = b, .dst = out_pad } },
        .{ .argmax = .{ .src = get_rows_src, .dst = out_argmax, .rows = 1, .cols = 4 } },
    };
    var command_bytes: [2048]u8 = undefined;
    const command_len = try wire.encodeCommandBuffer(&commands, &command_bytes);

    const result = try runtime.dispatch(.{ .run_graph = .{
        .request_id = 99,
        .command_bytes = command_bytes[0..command_len],
    } }, std.testing.io);
    try std.testing.expectEqual(wire.Status.ok, result.meta.status);
    try std.testing.expectEqual(@as(u64, commands.len), result.meta.value0);

    try expectTensor(&runtime, out_rms, &.{ 0.84852815, 1.1313709 });
    try expectTensor(&runtime, out_rope, &.{ -1.1426396, 1.9220756 });
    try expectTensor(&runtime, out_softmax, &.{ 0.26894143, 0.7310586 });
    try expectTensor(&runtime, out_silu, &.{ 0.7310586, 1.761594 });
    try expectTensor(&runtime, out_swiglu, &.{ 2.1931758, 7.046376 });
    try expectTensor(&runtime, out_add, &.{ 4, 6 });
    try expectTensor(&runtime, out_mul, &.{ 3, 8 });
    try expectTensor(&runtime, out_scale, &.{ 0.5, 1 });
    try expectTensor(&runtime, out_add_scaled, &.{ 3.5, 5 });
    try expectF16TensorAt(&runtime, out_rows, 2, &.{ 3, 4 });
    try expectTensor(&runtime, out_get_rows, &.{ 3, 4 });
    const w0 = @exp(@as(f32, 1)) / (@exp(@as(f32, 1)) + 1.0);
    const w1 = 1.0 - w0;
    try expectTensorApprox(&runtime, out_flash, &.{ w0 * 10 + w1 * 30, w0 * 20 + w1 * 40 }, 0.00001);
    try expectTensor(&runtime, out_pad, &.{ 1, 2, 0, 0 });
    try expectI32(&runtime, out_argmax, 3);
}

fn expectI32(runtime: *Runtime, range: wire.TensorRange, expected: i32) !void {
    const bytes = try runtime.heap.read(range);
    try std.testing.expectEqual(@as(usize, @sizeOf(i32)), bytes.len);
    try std.testing.expectEqual(expected, std.mem.readInt(i32, bytes[0..4], .little));
}

fn tensor(runtime: *Runtime, len: usize) !wire.TensorRange {
    return runtime.heap.allocate(len * @sizeOf(f32), @alignOf(f32));
}

fn rawTensor(runtime: *Runtime, len: usize, tensor_alignment: u32) !wire.TensorRange {
    return runtime.heap.allocate(len, tensor_alignment);
}

fn writeTensor(runtime: *Runtime, range: wire.TensorRange, values: []const f32) !void {
    const bytes = try runtime.heap.bytes(range);
    try std.testing.expectEqual(values.len * @sizeOf(f32), bytes.len);
    for (values, 0..) |value, i| {
        writeF32(bytes, i, value);
    }
}

fn writeI32Tensor(runtime: *Runtime, range: wire.TensorRange, values: []const i32) !void {
    const bytes = try runtime.heap.bytes(range);
    try std.testing.expectEqual(values.len * @sizeOf(i32), bytes.len);
    for (values, 0..) |value, i| {
        std.mem.writeInt(i32, bytes[i * @sizeOf(i32) ..][0..4], value, .little);
    }
}

fn writeF16Tensor(runtime: *Runtime, range: wire.TensorRange, values: []const f16) !void {
    const bytes = try runtime.heap.bytes(range);
    try std.testing.expectEqual(values.len * @sizeOf(f16), bytes.len);
    for (values, 0..) |value, i| {
        std.mem.writeInt(u16, bytes[i * @sizeOf(f16) ..][0..2], @bitCast(value), .little);
    }
}

fn expectTensor(runtime: *Runtime, range: wire.TensorRange, expected: []const f32) !void {
    try expectTensorApprox(runtime, range, expected, 0.000001);
}

fn expectTensorApprox(runtime: *Runtime, range: wire.TensorRange, expected: []const f32, tolerance: f32) !void {
    const bytes = try runtime.heap.read(range);
    try std.testing.expectEqual(expected.len * @sizeOf(f32), bytes.len);
    for (expected, 0..) |value, i| {
        try std.testing.expect(@abs(value - readF32(bytes, i)) <= tolerance);
    }
}

fn expectF16TensorAt(runtime: *Runtime, range: wire.TensorRange, start_index: usize, expected: []const f32) !void {
    const bytes = try runtime.heap.read(range);
    for (expected, 0..) |value, i| {
        const index = start_index + i;
        const half: f16 = @bitCast(std.mem.readInt(u16, bytes[index * @sizeOf(f16) ..][0..2], .little));
        try std.testing.expect(@abs(value - @as(f32, @floatCast(half))) <= 0.000001);
    }
}

fn writeF32(bytes: []u8, index: usize, value: f32) void {
    std.mem.writeInt(u32, bytes[index * @sizeOf(f32) ..][0..4], @bitCast(value), .little);
}

fn readF32(bytes: []const u8, index: usize) f32 {
    return @bitCast(std.mem.readInt(u32, bytes[index * @sizeOf(f32) ..][0..4], .little));
}
