//! Lifecycle, ordering, and exact-numeric cosim for the weighted RMSNorm source.

const std = @import("std");
const section = @import("shared_section");
const c = @cImport(@cInclude("shim.h"));

const GammaStatus = section.RmsNormGammaStatus;
const RunStatus = section.RmsNormWeightedSourceStatus;

const zero_lanes = [_]u64{ 0, 0, 0, 0 };

const Scalar = struct {
    data: u32,
    last: bool,
    status: u2,
};

const Request = struct {
    role: u2,
    token: u3,
    group: u11,
};

const PendingResponse = struct {
    request: Request,
    request_cycle: usize,
    due_cycle: usize,
    inject_error: bool,
};

const Dut = struct {
    handle: *c.Dut,

    fn init() Dut {
        return .{ .handle = c.dut_new().? };
    }

    fn deinit(self: *Dut) void {
        c.dut_free(self.handle);
    }

    fn eval(self: *Dut) void {
        c.dut_eval(self.handle);
    }

    fn step(self: *Dut) void {
        c.dut_set_clk(self.handle, 1);
        self.eval();
        c.dut_set_clk(self.handle, 0);
        self.eval();
    }

    fn clearInputs(self: *Dut) void {
        c.dut_set_abort(self.handle, 0);
        c.dut_set_gamma_config(self.handle, 0, 0);
        c.dut_set_gamma_stream(self.handle, 0, 0, 0, 0);
        c.dut_set_run_config(self.handle, 0, 0, 0);
        c.dut_set_inverse(self.handle, 0, 0, 0, 0);
        c.dut_set_read_request_ready(self.handle, 0);
        c.dut_set_read_response(self.handle, 0, &zero_lanes, 0);
        c.dut_set_scalar_ready(self.handle, 0);
        self.eval();
    }

    fn reset(self: *Dut) void {
        c.dut_set_clk(self.handle, 0);
        c.dut_set_rst_n(self.handle, 0);
        self.clearInputs();
        for (0..5) |_| self.step();
        c.dut_set_rst_n(self.handle, 1);
        self.step();
    }
};

const GammaOptions = struct {
    stalls: bool = true,
    early_last_word: ?usize = null,
    partial_keep_word: ?usize = null,
    omit_final_last: bool = false,
    nonfinite_scalar: ?usize = null,
    expected_status: ?u4 = null,
};

const GammaStats = struct {
    cycles: usize,
    words: usize,
};

const RunOptions = struct {
    stalls: bool = true,
    wrong_inverse_token: ?usize = null,
    early_inverse_final: ?usize = null,
    omit_inverse_final: bool = false,
    inverse_override_index: ?usize = null,
    inverse_override_bits: u32 = 0,
    scratch_error_request: ?usize = null,
    residual_override_index: ?usize = null,
    residual_override_bits: u32 = 0,
    hold_first_scalar_cycles: usize = 0,
    fixed_response_delay: ?usize = null,
    measure_cadence: bool = false,
    expected_status: ?u9 = null,
    expected_requests: ?usize = null,
    expected_scalars: ?usize = null,
};

const RunStats = struct {
    cycles: usize,
    inverses: usize,
    requests: usize,
    responses: usize,
    scalars: usize,
    request_interval_min: ?usize,
    request_interval_max: ?usize,
    scalar_interval_min: ?usize,
    scalar_interval_max: ?usize,
    boundary_interval_min: ?usize,
    boundary_interval_max: ?usize,
};

fn recordInterval(minimum: *?usize, maximum: *?usize, interval: usize) void {
    minimum.* = if (minimum.*) |prior| @min(prior, interval) else interval;
    maximum.* = if (maximum.*) |prior| @max(prior, interval) else interval;
}

fn gammaStatus(dut: *Dut) u4 {
    return @truncate(c.dut_gamma_status(dut.handle));
}

fn runStatus(dut: *Dut) u9 {
    return @truncate(c.dut_status(dut.handle));
}

fn readScalar(dut: *Dut) Scalar {
    return .{
        .data = c.dut_scalar_data(dut.handle),
        .last = c.dut_scalar_last(dut.handle) != 0,
        .status = @truncate(c.dut_scalar_status(dut.handle)),
    };
}

fn readRequest(dut: *Dut) Request {
    return .{
        .role = @truncate(c.dut_read_request_role(dut.handle)),
        .token = @truncate(c.dut_read_request_token(dut.handle)),
        .group = @truncate(c.dut_read_request_group(dut.handle)),
    };
}

fn isFinite(bits: u32) bool {
    return ((bits >> 23) & 0xff) != 0xff;
}

fn gammaWord(gamma: []const u32, word: usize, nonfinite_scalar: ?usize) u64 {
    var lo = gamma[word * 2];
    var hi = gamma[word * 2 + 1];
    if (nonfinite_scalar == word * 2) lo = 0x7fc1_2345;
    if (nonfinite_scalar == word * 2 + 1) hi = 0xff80_0000;
    return @as(u64, lo) | (@as(u64, hi) << 32);
}

fn configureGamma(dut: *Dut, rows: u32) !void {
    try std.testing.expect(c.dut_gamma_config_ready(dut.handle) != 0);
    c.dut_set_gamma_config(dut.handle, 1, @intCast(rows));
    dut.eval();
    try std.testing.expect(c.dut_gamma_config_ready(dut.handle) != 0);
    dut.step();
    c.dut_set_gamma_config(dut.handle, 0, 0);
    dut.eval();
}

fn loadGamma(
    dut: *Dut,
    gamma: []const u32,
    rows: u32,
    options: GammaOptions,
    seed: u64,
) !GammaStats {
    if (options.expected_status == null)
        try std.testing.expectEqual(@as(usize, rows), gamma.len);
    try configureGamma(dut, rows);

    var prng = std.Random.DefaultPrng.init(seed);
    const random = prng.random();
    const total_words = if (rows >= 2) @as(usize, rows) / 2 else 0;
    var word_index: usize = 0;
    var presenting = false;
    var held_data: u64 = 0;
    var held_keep: u8 = 0;
    var held_last = false;
    var cycle: usize = 0;

    while (c.dut_gamma_done(dut.handle) == 0) : (cycle += 1) {
        if (cycle > total_words * 16 + 256) return error.GammaTimeout;
        if (!presenting and word_index < total_words and
            (!options.stalls or random.uintLessThan(u8, 4) != 0))
        {
            presenting = true;
            held_data = gammaWord(gamma, word_index, options.nonfinite_scalar);
            held_keep = if (options.partial_keep_word == word_index) 0x0f else 0xff;
            held_last = options.early_last_word == word_index or
                (!options.omit_final_last and word_index + 1 == total_words);
        }
        c.dut_set_gamma_stream(
            dut.handle,
            @intFromBool(presenting),
            held_data,
            held_keep,
            @intFromBool(held_last),
        );
        c.dut_set_inverse(dut.handle, 0, 0, 0, 0);
        c.dut_set_read_request_ready(dut.handle, 0);
        c.dut_set_read_response(dut.handle, 0, &zero_lanes, 0);
        c.dut_set_scalar_ready(dut.handle, 0);
        dut.eval();

        const fire = presenting and c.dut_gamma_stream_ready(dut.handle) != 0;
        if (c.dut_gamma_busy(dut.handle) != 0) {
            try std.testing.expect(c.dut_gamma_valid(dut.handle) == 0);
            try std.testing.expect(c.dut_run_config_ready(dut.handle) == 0);
        }
        dut.step();
        if (fire) {
            word_index += 1;
            presenting = false;
        }
    }

    if (options.expected_status) |expected_status| {
        try std.testing.expect(c.dut_gamma_error(dut.handle) != 0);
        try std.testing.expectEqual(expected_status, gammaStatus(dut));
        try std.testing.expect(c.dut_gamma_valid(dut.handle) == 0);
    } else {
        try std.testing.expect(c.dut_gamma_error(dut.handle) == 0);
        try std.testing.expectEqual(@as(u4, 0), gammaStatus(dut));
        try std.testing.expectEqual(total_words, word_index);
        try std.testing.expect(c.dut_gamma_valid(dut.handle) != 0);
    }
    try std.testing.expect(c.dut_gamma_busy(dut.handle) == 0);
    try std.testing.expect(c.dut_busy(dut.handle) == 0);

    c.dut_set_gamma_stream(dut.handle, 0, 0, 0, 0);
    dut.step();
    try std.testing.expect(c.dut_gamma_done(dut.handle) == 0);
    if (options.expected_status) |expected_status| {
        try std.testing.expect(c.dut_gamma_error(dut.handle) != 0);
        try std.testing.expectEqual(expected_status, gammaStatus(dut));
    } else {
        try std.testing.expect(c.dut_gamma_error(dut.handle) == 0);
        try std.testing.expectEqual(@as(u4, 0), gammaStatus(dut));
    }
    return .{ .cycles = cycle + 1, .words = word_index };
}

fn configureRun(dut: *Dut, rows: u32, tokens: u8) !void {
    try std.testing.expect(c.dut_run_config_ready(dut.handle) != 0);
    c.dut_set_run_config(dut.handle, 1, @intCast(rows), tokens);
    dut.eval();
    try std.testing.expect(c.dut_run_config_ready(dut.handle) != 0);
    dut.step();
    c.dut_set_run_config(dut.handle, 0, 0, 0);
    dut.eval();
}

fn residualBits(token: usize, row: usize, options: RunOptions) u32 {
    const flat = token * 4096 + row;
    if (options.residual_override_index == flat)
        return options.residual_override_bits;
    if ((row + token * 7) % 127 == 0)
        return @as(u32, @intFromBool(((row / 127) & 1) != 0)) << 31;
    if ((row + token * 11) % 251 == 0)
        return (@as(u32, @intFromBool((token & 1) != 0)) << 31) |
            @as(u32, @intCast(1 + (row % 0x007f_ffff)));
    const exponent: u32 = @intCast(112 + ((row * 13 + token * 17) % 28));
    const fraction: u32 = @intCast((row *% 0x12d687 +% token *% 0x34567) & 0x007f_ffff);
    return (@as(u32, @intFromBool((row + token) % 3 == 0)) << 31) |
        (exponent << 23) | fraction;
}

fn responseFor(request: Request, options: RunOptions) [4]u64 {
    var lanes: [4]u64 = undefined;
    for (0..4) |lane| {
        const row = @as(usize, request.group) * 8 + lane * 2;
        const lo = residualBits(request.token, row, options);
        const hi = residualBits(request.token, row + 1, options);
        lanes[lane] = @as(u64, lo) | (@as(u64, hi) << 32);
    }
    return lanes;
}

fn expectedScalar(
    gamma: []const u32,
    inverses: []const u32,
    rows: usize,
    scalar_index: usize,
    options: RunOptions,
) !Scalar {
    const token = scalar_index / rows;
    const row = scalar_index % rows;
    const weighted = section.rmsNormWeightedBits(
        residualBits(token, row, options),
        inverses[token],
        gamma[row],
    );
    if (weighted.mul1_status != 0) return error.OracleMul1Fault;
    if (weighted.mul2_status != 0) return error.OracleMul2Fault;
    return .{
        .data = weighted.bits,
        .last = row % 32 == 31,
        .status = 0,
    };
}

fn runSource(
    dut: *Dut,
    gamma: []const u32,
    inverses: []const u32,
    rows: u32,
    tokens: u8,
    options: RunOptions,
    seed: u64,
) !RunStats {
    const row_count: usize = rows;
    const token_count: usize = tokens;
    if (options.expected_status == null) {
        try std.testing.expectEqual(row_count, gamma.len);
        try std.testing.expectEqual(token_count, inverses.len);
    }
    try configureRun(dut, rows, tokens);

    var prng = std.Random.DefaultPrng.init(seed);
    const random = prng.random();
    var inverse_index: usize = 0;
    var inverse_presenting = false;
    var inverse_token: u8 = 0;
    var inverse_data: u32 = 0;
    var inverse_final = false;

    var pending: ?PendingResponse = null;
    var response_active = false;
    var response_lanes = zero_lanes;
    var response_error = false;
    var response_request_cycle: usize = 0;
    var request_count: usize = 0;
    var response_count: usize = 0;
    var scalar_count: usize = 0;
    var held_request: ?Request = null;
    var held_scalar: ?Scalar = null;
    var held_scalar_cycles: usize = 0;
    var release_scalars = options.hold_first_scalar_cycles == 0;
    var last_request_cycle: ?usize = null;
    var last_scalar_cycle: ?usize = null;
    var request_interval_min: ?usize = null;
    var request_interval_max: ?usize = null;
    var scalar_interval_min: ?usize = null;
    var scalar_interval_max: ?usize = null;
    var boundary_interval_min: ?usize = null;
    var boundary_interval_max: ?usize = null;
    var cycle: usize = 0;

    while (c.dut_done(dut.handle) == 0) : (cycle += 1) {
        if (cycle > row_count * @max(token_count, 1) * 40 + 8192)
            return error.RunTimeout;

        if (!inverse_presenting and inverse_index < token_count and
            (!options.stalls or random.uintLessThan(u8, 4) != 0))
        {
            inverse_presenting = true;
            inverse_token = if (options.wrong_inverse_token == inverse_index)
                @intCast((inverse_index + 1) & 3)
            else
                @intCast(inverse_index);
            inverse_data = if (options.inverse_override_index == inverse_index)
                options.inverse_override_bits
            else
                inverses[inverse_index];
            inverse_final = options.early_inverse_final == inverse_index or
                (!options.omit_inverse_final and inverse_index + 1 == token_count);
        }
        c.dut_set_inverse(
            dut.handle,
            @intFromBool(inverse_presenting),
            inverse_token,
            inverse_data,
            @intFromBool(inverse_final),
        );

        if (!response_active) {
            if (pending) |owner| {
                if (cycle >= owner.due_cycle) {
                    response_lanes = responseFor(owner.request, options);
                    response_error = owner.inject_error;
                    response_request_cycle = owner.request_cycle;
                    response_active = true;
                    pending = null;
                }
            }
        }
        c.dut_set_read_response(
            dut.handle,
            @intFromBool(response_active),
            &response_lanes,
            @intFromBool(response_error),
        );
        const request_ready = pending == null and !response_active and
            (!options.stalls or random.uintLessThan(u8, 4) != 0);
        c.dut_set_read_request_ready(dut.handle, @intFromBool(request_ready));
        const random_scalar_ready = !options.stalls or
            random.uintLessThan(u8, 3) != 0;
        const scalar_ready = release_scalars and random_scalar_ready;
        c.dut_set_scalar_ready(dut.handle, @intFromBool(scalar_ready));
        dut.eval();

        const inverse_fire = inverse_presenting and
            c.dut_inverse_ready(dut.handle) != 0;
        const request_valid = c.dut_read_request_valid(dut.handle) != 0;
        const request_fire = request_valid and request_ready;
        const response_fire = response_active and
            c.dut_read_response_ready(dut.handle) != 0;
        const scalar_valid = c.dut_scalar_valid(dut.handle) != 0;
        const scalar_fire = scalar_valid and scalar_ready;

        if (pending != null or response_active)
            try std.testing.expect(!request_valid);
        if (request_valid) {
            const current = readRequest(dut);
            if (!request_ready) {
                if (held_request) |prior| try std.testing.expectEqual(prior, current);
                held_request = current;
            } else {
                held_request = null;
            }
        } else {
            held_request = null;
        }

        if (response_active)
            try std.testing.expect(c.dut_read_response_ready(dut.handle) != 0);

        if (scalar_valid) {
            const current = readScalar(dut);
            if (!scalar_ready) {
                if (held_scalar) |prior| try std.testing.expectEqual(prior, current);
                held_scalar = current;
            } else {
                held_scalar = null;
            }
            if (!release_scalars) {
                held_scalar_cycles += 1;
                if (held_scalar_cycles >= options.hold_first_scalar_cycles)
                    release_scalars = true;
            }
        } else {
            held_scalar = null;
        }

        if (request_fire) {
            const current = readRequest(dut);
            const groups_per_token = row_count / 8;
            try std.testing.expectEqual(@as(u2, 0), current.role);
            try std.testing.expectEqual(
                @as(u3, @intCast(request_count / groups_per_token)),
                current.token,
            );
            try std.testing.expectEqual(
                @as(u11, @intCast(request_count % groups_per_token)),
                current.group,
            );
            pending = .{
                .request = current,
                .request_cycle = cycle,
                .due_cycle = cycle + (options.fixed_response_delay orelse
                    2 + random.uintLessThan(usize, 6)),
                .inject_error = options.scratch_error_request == request_count,
            };
            if (options.measure_cadence) {
                if (last_request_cycle) |last|
                    recordInterval(
                        &request_interval_min,
                        &request_interval_max,
                        cycle - last,
                    );
                last_request_cycle = cycle;
            }
            request_count += 1;
        }

        if (scalar_fire) {
            try std.testing.expect(options.expected_status == null);
            try std.testing.expectEqual(
                try expectedScalar(gamma, inverses, row_count, scalar_count, options),
                readScalar(dut),
            );
            if (options.measure_cadence) {
                if (last_scalar_cycle) |last| {
                    if (scalar_count % 8 == 0) {
                        recordInterval(
                            &boundary_interval_min,
                            &boundary_interval_max,
                            cycle - last,
                        );
                    } else {
                        recordInterval(
                            &scalar_interval_min,
                            &scalar_interval_max,
                            cycle - last,
                        );
                    }
                }
                last_scalar_cycle = cycle;
            }
            scalar_count += 1;
        }

        dut.step();
        if (inverse_fire) {
            inverse_index += 1;
            inverse_presenting = false;
        }
        if (response_fire) {
            if (options.fixed_response_delay) |delay|
                try std.testing.expectEqual(delay, cycle - response_request_cycle);
            response_active = false;
            response_error = false;
            response_count += 1;
        }
    }

    try std.testing.expect(c.dut_busy(dut.handle) == 0);
    try std.testing.expect(c.dut_scalar_valid(dut.handle) == 0);
    if (options.expected_status) |expected_status| {
        try std.testing.expect(c.dut_error(dut.handle) != 0);
        try std.testing.expectEqual(expected_status, runStatus(dut));
        try std.testing.expectEqual(options.expected_requests orelse 0, request_count);
        try std.testing.expectEqual(options.expected_requests orelse 0, response_count);
        try std.testing.expectEqual(options.expected_scalars orelse 0, scalar_count);
        try std.testing.expect(c.dut_gamma_valid(dut.handle) == 0);
    } else {
        try std.testing.expect(c.dut_error(dut.handle) == 0);
        try std.testing.expectEqual(@as(u9, 0), runStatus(dut));
        try std.testing.expectEqual(token_count, inverse_index);
        try std.testing.expectEqual(row_count / 8 * token_count, request_count);
        try std.testing.expectEqual(request_count, response_count);
        try std.testing.expectEqual(row_count * token_count, scalar_count);
        try std.testing.expect(c.dut_gamma_valid(dut.handle) != 0);
    }

    dut.clearInputs();
    dut.step();
    try std.testing.expect(c.dut_done(dut.handle) == 0);
    if (options.expected_status) |expected_status| {
        try std.testing.expect(c.dut_error(dut.handle) != 0);
        try std.testing.expectEqual(expected_status, runStatus(dut));
    } else {
        try std.testing.expect(c.dut_error(dut.handle) == 0);
        try std.testing.expectEqual(@as(u9, 0), runStatus(dut));
    }
    return .{
        .cycles = cycle + 1,
        .inverses = inverse_index,
        .requests = request_count,
        .responses = response_count,
        .scalars = scalar_count,
        .request_interval_min = request_interval_min,
        .request_interval_max = request_interval_max,
        .scalar_interval_min = scalar_interval_min,
        .scalar_interval_max = scalar_interval_max,
        .boundary_interval_min = boundary_interval_min,
        .boundary_interval_max = boundary_interval_max,
    };
}

fn makeGamma(allocator: std.mem.Allocator, rows: usize, salt: u32) ![]u32 {
    const gamma = try allocator.alloc(u32, rows);
    for (gamma, 0..) |*bits, row| {
        const row32: u32 = @intCast(row);
        bits.* = switch (row % 257) {
            0 => @as(u32, @intFromBool(((row / 257) & 1) != 0)) << 31,
            1 => 0x0000_0001,
            2 => 0x807f_ffff,
            else => blk: {
                const exponent = 124 + ((row32 +% salt) % 7);
                const fraction = (row32 *% 0x45d9f3b +% salt *% 0x119de1f) &
                    0x007f_ffff;
                break :blk (@as(u32, @intFromBool((row32 +% salt) % 5 == 0)) << 31) |
                    (exponent << 23) | fraction;
            },
        };
        std.debug.assert(isFinite(bits.*));
    }
    return gamma;
}

fn makeInverses(tokens: usize, salt: u32) [4]u32 {
    var inverses = [_]u32{0} ** 4;
    for (0..tokens) |token| {
        const fraction = (salt *% 0x45d9f3b +% @as(u32, @intCast(token)) *%
            0x119de1f) & 0x007f_ffff;
        inverses[token] = (@as(u32, @intCast(125 + token)) << 23) | fraction;
    }
    return inverses;
}

fn reloadGamma(dut: *Dut, gamma: []const u32, seed: u64) !void {
    _ = try loadGamma(dut, gamma, @intCast(gamma.len), .{}, seed);
}

fn verifyGammaFaults(
    dut: *Dut,
    gamma: []const u32,
    replacement_gamma: []const u32,
    inverses: []const u32,
) !void {
    try std.testing.expectEqual(gamma.len, replacement_gamma.len);
    try std.testing.expect(!std.mem.eql(u32, gamma, replacement_gamma));

    try reloadGamma(dut, gamma, 0x1000);
    _ = try loadGamma(dut, gamma, 120, .{
        .expected_status = GammaStatus.bad_cfg,
    }, 0x1001);
    _ = try loadGamma(dut, gamma, 128, .{
        .early_last_word = 0,
        .expected_status = GammaStatus.frame,
    }, 0x1002);
    _ = try loadGamma(dut, gamma, 128, .{
        .partial_keep_word = 1,
        .expected_status = GammaStatus.frame,
    }, 0x1003);
    _ = try loadGamma(dut, gamma, 128, .{
        .omit_final_last = true,
        .expected_status = GammaStatus.frame,
    }, 0x1004);
    _ = try loadGamma(dut, gamma, 128, .{
        .nonfinite_scalar = 0,
        .expected_status = GammaStatus.nonfinite,
    }, 0x1005);
    _ = try loadGamma(dut, gamma, 128, .{
        .nonfinite_scalar = 3,
        .expected_status = GammaStatus.nonfinite,
    }, 0x1006);

    // A late malformed replacement may overwrite tentative RAM words, but it
    // cannot leave a readable table. Reject a run, then prove a different full
    // reload is the exact table observed by every emitted scalar.
    try reloadGamma(dut, gamma, 0x1010);
    _ = try loadGamma(dut, replacement_gamma, 128, .{
        .early_last_word = gamma.len / 2 - 2,
        .expected_status = GammaStatus.frame,
    }, 0x1011);
    _ = try runSource(dut, replacement_gamma, inverses, 128, 1, .{
        .expected_status = RunStatus.gamma,
        .expected_requests = 0,
        .expected_scalars = 0,
    }, 0x1012);
    try std.testing.expect(c.dut_read_request_valid(dut.handle) == 0);
    try reloadGamma(dut, replacement_gamma, 0x1013);
    _ = try runSource(dut, replacement_gamma, inverses, 128, 1, .{}, 0x1014);

    _ = try loadGamma(dut, gamma, 128, .{
        .omit_final_last = true,
        .expected_status = GammaStatus.frame,
    }, 0x1015);
    _ = try runSource(dut, gamma, inverses, 128, 1, .{
        .expected_status = RunStatus.gamma,
        .expected_requests = 0,
        .expected_scalars = 0,
    }, 0x1016);
    try std.testing.expect(c.dut_read_request_valid(dut.handle) == 0);
    try reloadGamma(dut, gamma, 0x1017);
    _ = try runSource(dut, gamma, inverses, 128, 1, .{}, 0x1018);
}

fn verifyRunFaults(dut: *Dut, gamma: []u32) !void {
    const inverses4 = makeInverses(4, 0x219);

    try reloadGamma(dut, gamma, 0x2000);
    _ = try runSource(dut, gamma, inverses4[0..1], 64, 1, .{
        .expected_status = RunStatus.bad_cfg,
    }, 0x2001);

    try reloadGamma(dut, gamma, 0x2010);
    _ = try runSource(dut, gamma, inverses4[0..0], 128, 0, .{
        .expected_status = RunStatus.bad_cfg,
    }, 0x2011);

    try reloadGamma(dut, gamma, 0x2020);
    _ = try runSource(dut, gamma, inverses4[0..1], 4096, 1, .{
        .expected_status = RunStatus.gamma,
    }, 0x2021);

    try reloadGamma(dut, gamma, 0x2030);
    _ = try runSource(dut, gamma, inverses4[0..4], 128, 4, .{
        .wrong_inverse_token = 0,
        .expected_status = RunStatus.inverse_frame,
    }, 0x2031);

    try reloadGamma(dut, gamma, 0x2040);
    _ = try runSource(dut, gamma, inverses4[0..4], 128, 4, .{
        .early_inverse_final = 0,
        .expected_status = RunStatus.inverse_frame,
    }, 0x2041);

    try reloadGamma(dut, gamma, 0x2050);
    _ = try runSource(dut, gamma, inverses4[0..1], 128, 1, .{
        .omit_inverse_final = true,
        .expected_status = RunStatus.inverse_frame,
    }, 0x2051);

    const bad_inverses = [_]u32{
        0x0000_0000, // +0
        0x8000_0000, // -0
        0x0000_0001, // positive subnormal
        0x3f80_0000 | 0x8000_0000, // negative normal
        0x7f80_0000, // +infinity
        0x7fc1_2345, // NaN
    };
    for (bad_inverses, 0..) |bad_inverse, index| {
        try reloadGamma(dut, gamma, 0x2060 + index * 2);
        _ = try runSource(dut, gamma, inverses4[0..1], 128, 1, .{
            .inverse_override_index = 0,
            .inverse_override_bits = bad_inverse,
            .expected_status = RunStatus.inverse_frame,
        }, 0x2061 + index * 2);
    }

    try reloadGamma(dut, gamma, 0x2070);
    _ = try runSource(dut, gamma, inverses4[0..1], 128, 1, .{
        .scratch_error_request = 0,
        .expected_status = RunStatus.scratch,
        .expected_requests = 1,
    }, 0x2071);

    try reloadGamma(dut, gamma, 0x2080);
    _ = try runSource(dut, gamma, inverses4[0..1], 128, 1, .{
        .residual_override_index = 0,
        .residual_override_bits = 0x7f80_0000,
        .expected_status = RunStatus.mul1(section.RmsNormMulStatus.nonfinite_input),
        .expected_requests = 1,
    }, 0x2081);

    const inverse_two = [_]u32{0x4000_0000};
    try reloadGamma(dut, gamma, 0x2090);
    _ = try runSource(dut, gamma, inverse_two[0..], 128, 1, .{
        .residual_override_index = 0,
        .residual_override_bits = 0x7f7f_ffff,
        .expected_status = RunStatus.mul1(section.RmsNormMulStatus.overflow),
        .expected_requests = 1,
    }, 0x2091);

    const saved_gamma0 = gamma[0];
    gamma[0] = 0x7f7f_ffff;
    const inverse_one = [_]u32{0x3f80_0000};
    try reloadGamma(dut, gamma, 0x20a0);
    _ = try runSource(dut, gamma, inverse_one[0..], 128, 1, .{
        .residual_override_index = 0,
        .residual_override_bits = 0x4000_0000,
        .expected_status = RunStatus.mul2(section.RmsNormMulStatus.overflow),
        .expected_requests = 1,
    }, 0x20a1);
    gamma[0] = saved_gamma0;

    // Every terminal fault invalidates the resident table. Reloading and
    // publishing a complete run without reset proves the common restart path.
    try reloadGamma(dut, gamma, 0x20f0);
    _ = try runSource(
        dut,
        gamma,
        inverses4[0..1],
        128,
        1,
        .{ .hold_first_scalar_cycles = 4 },
        0x20f1,
    );
}

fn sendInverses(dut: *Dut, inverses: []const u32) !void {
    for (inverses, 0..) |bits, index| {
        var cycles: usize = 0;
        while (true) : (cycles += 1) {
            if (cycles > 64) return error.InverseTimeout;
            c.dut_set_inverse(
                dut.handle,
                1,
                @intCast(index),
                bits,
                @intFromBool(index + 1 == inverses.len),
            );
            c.dut_set_read_request_ready(dut.handle, 0);
            c.dut_set_read_response(dut.handle, 0, &zero_lanes, 0);
            c.dut_set_scalar_ready(dut.handle, 0);
            dut.eval();
            const fire = c.dut_inverse_ready(dut.handle) != 0;
            dut.step();
            if (fire) break;
        }
    }
    c.dut_set_inverse(dut.handle, 0, 0, 0, 0);
    dut.eval();
}

fn acceptRequest(dut: *Dut, expected_index: usize, rows: usize) !Request {
    var cycles: usize = 0;
    while (true) : (cycles += 1) {
        if (cycles > 64) return error.RequestTimeout;
        c.dut_set_read_request_ready(dut.handle, 1);
        c.dut_set_read_response(dut.handle, 0, &zero_lanes, 0);
        c.dut_set_scalar_ready(dut.handle, 0);
        dut.eval();
        if (c.dut_read_request_valid(dut.handle) != 0) {
            const request = readRequest(dut);
            const groups_per_token = rows / 8;
            try std.testing.expectEqual(@as(u2, 0), request.role);
            try std.testing.expectEqual(
                @as(u3, @intCast(expected_index / groups_per_token)),
                request.token,
            );
            try std.testing.expectEqual(
                @as(u11, @intCast(expected_index % groups_per_token)),
                request.group,
            );
            dut.step();
            c.dut_set_read_request_ready(dut.handle, 0);
            dut.eval();
            return request;
        }
        dut.step();
    }
}

fn deliverResponse(dut: *Dut, request: Request, options: RunOptions) !void {
    var lanes = responseFor(request, options);
    c.dut_set_read_response(dut.handle, 1, &lanes, 0);
    dut.eval();
    try std.testing.expect(c.dut_read_response_ready(dut.handle) != 0);
    dut.step();
    c.dut_set_read_response(dut.handle, 0, &zero_lanes, 0);
    dut.eval();
}

fn expectAbortIdle(dut: *Dut, max_cycles: usize) !void {
    c.dut_set_abort(dut.handle, 1);
    dut.eval();
    try std.testing.expect(c.dut_gamma_stream_ready(dut.handle) == 0);
    try std.testing.expect(c.dut_scalar_valid(dut.handle) == 0);
    dut.step();
    c.dut_set_abort(dut.handle, 0);
    dut.eval();

    var cycles: usize = 0;
    while (c.dut_busy(dut.handle) != 0 or
        c.dut_gamma_busy(dut.handle) != 0) : (cycles += 1)
    {
        if (cycles > max_cycles) return error.AbortTimeout;
        try std.testing.expect(c.dut_done(dut.handle) == 0);
        try std.testing.expect(c.dut_error(dut.handle) == 0);
        try std.testing.expect(c.dut_gamma_done(dut.handle) == 0);
        try std.testing.expect(c.dut_gamma_error(dut.handle) == 0);
        try std.testing.expect(c.dut_scalar_valid(dut.handle) == 0);
        dut.step();
    }
    try std.testing.expect(c.dut_done(dut.handle) == 0);
    try std.testing.expect(c.dut_error(dut.handle) == 0);
    try std.testing.expectEqual(@as(u9, 0), runStatus(dut));
    try std.testing.expect(c.dut_gamma_done(dut.handle) == 0);
    try std.testing.expect(c.dut_gamma_error(dut.handle) == 0);
    try std.testing.expectEqual(@as(u4, 0), gammaStatus(dut));
    try std.testing.expect(c.dut_gamma_valid(dut.handle) == 0);
    try std.testing.expect(c.dut_scalar_valid(dut.handle) == 0);
    dut.clearInputs();
}

fn restartAfterAbort(
    dut: *Dut,
    gamma: []const u32,
    inverses: []const u32,
    seed: u64,
) !void {
    try reloadGamma(dut, gamma, seed);
    _ = try runSource(dut, gamma, inverses, @intCast(gamma.len), 1, .{}, seed ^ 1);
}

fn verifyAbortGammaConfig(
    dut: *Dut,
    gamma: []const u32,
    inverses: []const u32,
) !void {
    try reloadGamma(dut, gamma, 0x3000);
    try configureGamma(dut, @intCast(gamma.len));
    try std.testing.expect(c.dut_gamma_busy(dut.handle) != 0);
    try std.testing.expect(c.dut_gamma_valid(dut.handle) == 0);
    try expectAbortIdle(dut, 16);
    try restartAfterAbort(dut, gamma, inverses, 0x3001);
}

fn verifyAbortGammaLoad(
    dut: *Dut,
    gamma: []const u32,
    inverses: []const u32,
) !void {
    try configureGamma(dut, @intCast(gamma.len));
    for (0..3) |word| {
        c.dut_set_gamma_stream(
            dut.handle,
            1,
            gammaWord(gamma, word, null),
            0xff,
            0,
        );
        dut.eval();
        try std.testing.expect(c.dut_gamma_stream_ready(dut.handle) != 0);
        dut.step();
    }
    // Present the next word without clocking it. Abort must withdraw ready
    // combinationally, so this beat is never accepted or written.
    c.dut_set_gamma_stream(
        dut.handle,
        1,
        gammaWord(gamma, 3, null),
        0xff,
        0,
    );
    dut.eval();
    try std.testing.expect(c.dut_gamma_stream_ready(dut.handle) != 0);
    try expectAbortIdle(dut, 16);
    try restartAfterAbort(dut, gamma, inverses, 0x3011);
}

fn verifyAbortScales(
    dut: *Dut,
    gamma: []const u32,
    inverses: []const u32,
) !void {
    try configureRun(dut, @intCast(gamma.len), 4);
    try sendInverses(dut, inverses[0..1]);
    try std.testing.expect(c.dut_busy(dut.handle) != 0);
    try expectAbortIdle(dut, 16);
    try restartAfterAbort(dut, gamma, inverses[0..1], 0x3021);
}

fn checkPostAbortIdle(dut: *Dut) !void {
    var cycles: usize = 0;
    while (c.dut_busy(dut.handle) != 0 or
        c.dut_gamma_busy(dut.handle) != 0) : (cycles += 1)
    {
        if (cycles > 32) return error.AbortTimeout;
        try std.testing.expect(c.dut_done(dut.handle) == 0);
        try std.testing.expect(c.dut_error(dut.handle) == 0);
        try std.testing.expect(c.dut_scalar_valid(dut.handle) == 0);
        dut.step();
    }
    try std.testing.expect(c.dut_done(dut.handle) == 0);
    try std.testing.expect(c.dut_error(dut.handle) == 0);
    try std.testing.expectEqual(@as(u9, 0), runStatus(dut));
    try std.testing.expect(c.dut_gamma_done(dut.handle) == 0);
    try std.testing.expect(c.dut_gamma_error(dut.handle) == 0);
    try std.testing.expectEqual(@as(u4, 0), gammaStatus(dut));
    try std.testing.expect(c.dut_gamma_valid(dut.handle) == 0);
    try std.testing.expect(c.dut_scalar_valid(dut.handle) == 0);
    dut.clearInputs();
}

fn verifyAbortOutstanding(
    dut: *Dut,
    gamma: []const u32,
    inverses: []const u32,
) !void {
    try configureRun(dut, @intCast(gamma.len), 1);
    try sendInverses(dut, inverses);
    const request = try acceptRequest(dut, 0, gamma.len);

    c.dut_set_abort(dut.handle, 1);
    dut.eval();
    try std.testing.expect(c.dut_busy(dut.handle) != 0);
    try std.testing.expect(c.dut_scalar_valid(dut.handle) == 0);
    dut.step();
    c.dut_set_abort(dut.handle, 0);
    dut.eval();
    try std.testing.expect(c.dut_busy(dut.handle) != 0);
    try std.testing.expect(c.dut_done(dut.handle) == 0);
    try std.testing.expect(c.dut_read_response_ready(dut.handle) != 0);

    // The accepted untagged owner is retired, but its payload is discarded.
    var lanes = responseFor(request, .{});
    c.dut_set_read_response(dut.handle, 1, &lanes, 0);
    dut.eval();
    try std.testing.expect(c.dut_read_response_ready(dut.handle) != 0);
    dut.step();
    c.dut_set_read_response(dut.handle, 0, &zero_lanes, 0);
    try checkPostAbortIdle(dut);
    try restartAfterAbort(dut, gamma, inverses, 0x3031);
}

fn verifyAbortMultiplier(
    dut: *Dut,
    gamma: []const u32,
    inverses: []const u32,
    cycles_after_response: usize,
    seed: u64,
) !void {
    try configureRun(dut, @intCast(gamma.len), 1);
    try sendInverses(dut, inverses);
    const request = try acceptRequest(dut, 0, gamma.len);
    try deliverResponse(dut, request, .{});
    for (0..cycles_after_response) |_| {
        c.dut_set_scalar_ready(dut.handle, 0);
        dut.eval();
        try std.testing.expect(c.dut_busy(dut.handle) != 0);
        try std.testing.expect(c.dut_scalar_valid(dut.handle) == 0);
        dut.step();
    }
    try expectAbortIdle(dut, 32);
    try restartAfterAbort(dut, gamma, inverses, seed);
}

fn verifyAbortStalledScalar(
    dut: *Dut,
    gamma: []const u32,
    inverses: []const u32,
) !void {
    try configureRun(dut, @intCast(gamma.len), 1);
    try sendInverses(dut, inverses);
    const request = try acceptRequest(dut, 0, gamma.len);
    try deliverResponse(dut, request, .{});
    c.dut_set_scalar_ready(dut.handle, 0);
    var cycles: usize = 0;
    while (c.dut_scalar_valid(dut.handle) == 0) : (cycles += 1) {
        if (cycles > 64) return error.ScalarTimeout;
        try std.testing.expect(c.dut_busy(dut.handle) != 0);
        dut.step();
    }
    const held = readScalar(dut);
    for (0..3) |_| {
        dut.step();
        try std.testing.expect(c.dut_scalar_valid(dut.handle) != 0);
        try std.testing.expectEqual(held, readScalar(dut));
        try std.testing.expect(c.dut_done(dut.handle) == 0);
    }
    try expectAbortIdle(dut, 32);
    try restartAfterAbort(dut, gamma, inverses, 0x3061);
}

fn verifyAborts(dut: *Dut, gamma: []const u32) !void {
    const all_inverses = makeInverses(4, 0x331);
    const inverses = all_inverses[0..1];
    try verifyAbortGammaConfig(dut, gamma, inverses);
    try verifyAbortGammaLoad(dut, gamma, inverses);
    try verifyAbortScales(dut, gamma, all_inverses[0..4]);
    try verifyAbortOutstanding(dut, gamma, inverses);
    // Six-cycle multiplier latency makes these offsets land in MUL1 and MUL2.
    try verifyAbortMultiplier(dut, gamma, inverses, 2, 0x3041);
    try verifyAbortMultiplier(dut, gamma, inverses, 10, 0x3051);
    try verifyAbortStalledScalar(dut, gamma, inverses);
}

pub fn main() !void {
    var dut = Dut.init();
    defer dut.deinit();
    dut.reset();

    const allocator = std.heap.page_allocator;
    const gamma128 = try makeGamma(allocator, 128, 0x41);
    defer allocator.free(gamma128);
    const gamma4096 = try makeGamma(allocator, 4096, 0x83);
    defer allocator.free(gamma4096);
    const inverses = makeInverses(4, 0x51a);

    try verifyGammaFaults(
        &dut,
        gamma128,
        gamma4096[0..gamma128.len],
        inverses[0..1],
    );

    const gamma128_stats = try loadGamma(&dut, gamma128, 128, .{}, 0x4100);
    const compact_one = try runSource(
        &dut,
        gamma128,
        inverses[0..1],
        128,
        1,
        .{ .hold_first_scalar_cycles = 5 },
        0x4101,
    );
    // A second run reuses the exact committed table without a gamma reload.
    const compact_four = try runSource(
        &dut,
        gamma128,
        inverses[0..4],
        128,
        4,
        .{},
        0x4102,
    );
    const cadence = try runSource(
        &dut,
        gamma128,
        inverses[0..1],
        128,
        1,
        .{
            .stalls = false,
            .fixed_response_delay = 2,
            .measure_cadence = true,
        },
        0x4103,
    );
    try std.testing.expectEqual(
        cadence.request_interval_min.?,
        cadence.request_interval_max.?,
    );
    try std.testing.expectEqual(
        cadence.scalar_interval_min.?,
        cadence.scalar_interval_max.?,
    );
    try std.testing.expectEqual(
        cadence.boundary_interval_min.?,
        cadence.boundary_interval_max.?,
    );
    const cadence_group_interval = cadence.request_interval_max.?;
    const cadence_scalar_interval = cadence.scalar_interval_max.?;
    const cadence_boundary_interval = cadence.boundary_interval_max.?;
    try std.testing.expectEqual(@as(usize, 115), cadence_group_interval);
    try std.testing.expectEqual(@as(usize, 14), cadence_scalar_interval);
    try std.testing.expectEqual(@as(usize, 17), cadence_boundary_interval);
    const proposed_group_cap: usize = 116;
    try std.testing.expect(cadence_group_interval <= proposed_group_cap);
    const cadence_target_result = if (cadence_group_interval <= proposed_group_cap)
        "PASS"
    else
        "MISS";
    const cadence_target_excess = cadence_group_interval -| proposed_group_cap;

    try verifyRunFaults(&dut, gamma128);
    try verifyAborts(&dut, gamma128);

    const gamma4096_stats = try loadGamma(&dut, gamma4096, 4096, .{}, 0x4200);
    const full_one = try runSource(
        &dut,
        gamma4096,
        inverses[0..1],
        4096,
        1,
        .{},
        0x4201,
    );
    const full_four = try runSource(
        &dut,
        gamma4096,
        inverses[0..4],
        4096,
        4,
        .{},
        0x4202,
    );

    std.debug.print(
        "section_rmsnorm_weighted_source cosim:\n" ++
            "  gamma words/cycles: 128={d}/{d}, 4096={d}/{d}\n" ++
            "  rows/tokens scalars/cycles: 128/1={d}/{d}, 128/4={d}/{d}\n" ++
            "  rows/tokens scalars/cycles: 4096/1={d}/{d}, 4096/4={d}/{d}\n" ++
            "  no-stall cadence: group={d}, scalar={d}, boundary={d} cycles\n" ++
            "  proposed group target<=116: {s}, excess={d} cycles\n" ++
            "  exact two-step RNE, reuse, stalls, faults, abort/drain/restart: passed\n\n",
        .{
            gamma128_stats.words,
            gamma128_stats.cycles,
            gamma4096_stats.words,
            gamma4096_stats.cycles,
            compact_one.scalars,
            compact_one.cycles,
            compact_four.scalars,
            compact_four.cycles,
            full_one.scalars,
            full_one.cycles,
            full_four.scalars,
            full_four.cycles,
            cadence_group_interval,
            cadence_scalar_interval,
            cadence_boundary_interval,
            cadence_target_result,
            cadence_target_excess,
        },
    );
}
