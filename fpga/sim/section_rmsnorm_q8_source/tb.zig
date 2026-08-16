//! End-to-end cosim for weighted RMSNorm into the canonical native-Q8 stream.

const std = @import("std");
const section = @import("shared_section");
const layout = section.layout;
const c = @cImport(@cInclude("shim.h"));

const rows_set = [_]usize{ 128, 256, 512, 1024, 2048, 4096 };
const zero_lanes = [_]u64{ 0, 0, 0, 0 };

const AggregateStatus = section.RmsNormQ8SourceStatus;
const WeightedStatus = section.RmsNormWeightedSourceStatus;
const weighted_bad_cfg = AggregateStatus.weighted(WeightedStatus.bad_cfg);
const weighted_gamma = AggregateStatus.weighted(WeightedStatus.gamma);
const weighted_inverse = AggregateStatus.weighted(WeightedStatus.inverse_frame);
const weighted_scratch = AggregateStatus.weighted(WeightedStatus.scratch);
const q8_scale = AggregateStatus.q8(1 << 1);
const q8_arith = AggregateStatus.q8(1 << 3);

const Request = struct {
    role: u2,
    token: u3,
    group: u11,
};

const NativeBeat = struct {
    data: u64,
    last: bool,
    token: u2,
    block: u9,
};

const ExpectedRecord = struct {
    data: [5]u64,
    token: u2,
    block: u9,
};

const PendingResponse = struct {
    request: Request,
    due_cycle: usize,
    inject_error: bool,
};

const ResidualMode = enum {
    normal,
    overflow_second_block,
    tiny_second_block,
};

const RunOptions = struct {
    stalls: bool = true,
    fixed_response_delay: ?usize = null,
    hold_each_beat: bool = false,
    measure_cadence: bool = false,
    wrong_inverse_token: ?usize = null,
    scratch_error_request: ?usize = null,
    residual_override_index: ?usize = null,
    residual_override_bits: u32 = 0,
    residual_mode: ResidualMode = .normal,
    expected_status: ?u16 = null,
    expected_requests: ?usize = null,
    expected_scalars: ?usize = null,
    expected_records: ?usize = null,
};

const RunStats = struct {
    cycles: usize,
    requests: usize,
    responses: usize,
    scalars: usize,
    records: usize,
    beats: usize,
    max_record_cycles: usize,
    final_flush_cycles: usize,
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
        c.dut_set_output_ready(self.handle, 0);
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

fn gammaStatus(dut: *Dut) u4 {
    return @truncate(c.dut_gamma_status(dut.handle));
}

fn runStatus(dut: *Dut) u16 {
    return c.dut_status(dut.handle);
}

fn readRequest(dut: *Dut) Request {
    return .{
        .role = @truncate(c.dut_read_request_role(dut.handle)),
        .token = @truncate(c.dut_read_request_token(dut.handle)),
        .group = @truncate(c.dut_read_request_group(dut.handle)),
    };
}

fn readBeat(dut: *Dut) NativeBeat {
    return .{
        .data = c.dut_output_data(dut.handle),
        .last = c.dut_output_last(dut.handle) != 0,
        .token = @truncate(c.dut_output_token(dut.handle)),
        .block = @truncate(c.dut_output_block(dut.handle)),
    };
}

fn makeGamma(allocator: std.mem.Allocator, rows: usize) ![]u32 {
    const gamma = try allocator.alloc(u32, rows);
    for (gamma, 0..) |*bits, row| {
        const sign: u32 = @as(u32, @intFromBool(row % 17 == 0)) << 31;
        const exponent: u32 = @intCast(124 + (row % 5));
        const fraction: u32 = @intCast((row *% 0x15a4e3 +% 0x224466) & 0x007f_ffff);
        bits.* = sign | (exponent << 23) | fraction;
    }
    return gamma;
}

fn makeOnes(allocator: std.mem.Allocator, rows: usize) ![]u32 {
    const gamma = try allocator.alloc(u32, rows);
    @memset(gamma, 0x3f80_0000);
    return gamma;
}

fn makeInverses(tokens: usize) [4]u32 {
    var result = [_]u32{0} ** 4;
    const values = [_]u32{ 0x3f20_0000, 0x3f58_0000, 0x3f90_0000, 0x3fc0_0000 };
    for (0..tokens) |token| result[token] = values[token];
    return result;
}

fn gammaWord(gamma: []const u32, word: usize, nonfinite_scalar: ?usize) u64 {
    var lo = gamma[word * 2];
    var hi = gamma[word * 2 + 1];
    if (nonfinite_scalar == word * 2) lo = 0x7fc1_2345;
    if (nonfinite_scalar == word * 2 + 1) hi = 0xff80_0000;
    return @as(u64, lo) | (@as(u64, hi) << 32);
}

fn configureGamma(dut: *Dut, rows: usize) !void {
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
    rows: usize,
    seed: u64,
    expected_status: ?u4,
    early_last_word: ?usize,
    nonfinite_scalar: ?usize,
) !usize {
    try configureGamma(dut, rows);
    var prng = std.Random.DefaultPrng.init(seed);
    const random = prng.random();
    const words = rows / 2;
    var word: usize = 0;
    var presenting = false;
    var held_data: u64 = 0;
    var held_last = false;
    var cycles: usize = 0;

    while (c.dut_gamma_done(dut.handle) == 0) : (cycles += 1) {
        if (cycles > words * 12 + 64) return error.GammaTimeout;
        if (!presenting and word < words and random.uintLessThan(u8, 4) != 0) {
            presenting = true;
            held_data = gammaWord(gamma, word, nonfinite_scalar);
            held_last = early_last_word == word or
                (early_last_word == null and word + 1 == words);
        }
        c.dut_set_gamma_stream(
            dut.handle,
            @intFromBool(presenting),
            held_data,
            0xff,
            @intFromBool(held_last),
        );
        c.dut_set_output_ready(dut.handle, 0);
        dut.eval();
        const fire = presenting and c.dut_gamma_stream_ready(dut.handle) != 0;
        if (c.dut_gamma_busy(dut.handle) != 0) {
            try std.testing.expect(c.dut_gamma_valid(dut.handle) == 0);
            try std.testing.expect(c.dut_run_config_ready(dut.handle) == 0);
        }
        dut.step();
        if (fire) {
            word += 1;
            presenting = false;
        }
    }

    if (expected_status) |status| {
        try std.testing.expect(c.dut_gamma_error(dut.handle) != 0);
        try std.testing.expectEqual(status, gammaStatus(dut));
        try std.testing.expect(c.dut_gamma_valid(dut.handle) == 0);
    } else {
        try std.testing.expect(c.dut_gamma_error(dut.handle) == 0);
        try std.testing.expectEqual(@as(u4, 0), gammaStatus(dut));
        try std.testing.expectEqual(words, word);
        try std.testing.expect(c.dut_gamma_valid(dut.handle) != 0);
    }
    try std.testing.expect(c.dut_gamma_busy(dut.handle) == 0);
    dut.clearInputs();
    dut.step();
    try std.testing.expect(c.dut_gamma_done(dut.handle) == 0);
    return cycles + 1;
}

fn reloadGamma(dut: *Dut, gamma: []const u32, seed: u64) !void {
    _ = try loadGamma(dut, gamma, gamma.len, seed, null, null, null);
}

fn configureRun(dut: *Dut, rows: usize, tokens: usize) !void {
    try std.testing.expect(c.dut_run_config_ready(dut.handle) != 0);
    c.dut_set_run_config(dut.handle, 1, @intCast(rows), @intCast(tokens));
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
    if (token == 0 and row >= 32 and row < 64) switch (options.residual_mode) {
        .overflow_second_block => return if (row == 32) 0x4cbe_bc20 else 0,
        .tiny_second_block => return 0x0080_0000,
        .normal => {},
    };
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

fn expectedRecord(
    gamma: []const u32,
    inverses: []const u32,
    rows: usize,
    record_index: usize,
    options: RunOptions,
) !ExpectedRecord {
    const blocks_per_token = rows / 32;
    const token = record_index / blocks_per_token;
    const block_index = record_index % blocks_per_token;
    var values: [layout.q8_block]f32 = undefined;
    for (&values, 0..) |*value, lane| {
        const row = block_index * layout.q8_block + lane;
        const weighted = section.rmsNormWeightedBits(
            residualBits(token, row, options),
            inverses[token],
            gamma[row],
        );
        if (weighted.mul1_status != 0) return error.OracleMul1Fault;
        if (weighted.mul2_status != 0) return error.OracleMul2Fault;
        value.* = @bitCast(weighted.bits);
    }

    var quants: [layout.q8_block]i8 = undefined;
    var scales: [1]f16 = undefined;
    try layout.quantizeQ8_0(&values, &quants, &scales);
    var data = [_]u64{0} ** 5;
    for (0..4) |beat| {
        for (0..8) |lane| {
            const byte: u8 = @bitCast(quants[beat * 8 + lane]);
            data[beat] |= @as(u64, byte) << @as(u6, @intCast(lane * 8));
        }
    }
    data[4] = @as(u16, @bitCast(scales[0]));
    return .{
        .data = data,
        .token = @intCast(token),
        .block = @intCast(block_index),
    };
}

fn checkBeat(
    actual: NativeBeat,
    expected: ExpectedRecord,
    beat_index: usize,
) !void {
    if (actual.data != expected.data[beat_index]) {
        std.debug.print(
            "native data mismatch token={d} block={d} beat={d}: got=0x{x:0>16} expected=0x{x:0>16}\n",
            .{ expected.token, expected.block, beat_index, actual.data, expected.data[beat_index] },
        );
        return error.NativeDataMismatch;
    }
    try std.testing.expectEqual(expected.token, actual.token);
    try std.testing.expectEqual(expected.block, actual.block);
    try std.testing.expectEqual(beat_index == 4, actual.last);
}

fn runSource(
    dut: *Dut,
    gamma: []const u32,
    inverses: []const u32,
    rows: usize,
    tokens: usize,
    options: RunOptions,
    seed: u64,
) !RunStats {
    try configureRun(dut, rows, tokens);
    var prng = std.Random.DefaultPrng.init(seed);
    const random = prng.random();
    var inverse_index: usize = 0;
    var inverse_presenting = false;
    var pending: ?PendingResponse = null;
    var response_active = false;
    var response_lanes = zero_lanes;
    var response_error = false;
    var request_count: usize = 0;
    var response_count: usize = 0;
    var scalar_count: usize = 0;
    var beat_count: usize = 0;
    var record_count: usize = 0;
    var held_request: ?Request = null;
    var held_beat: ?NativeBeat = null;
    var forced_holds = [_]usize{2} ** 5;
    var expected_record: ?ExpectedRecord = null;
    var last_record_cycle: usize = 0;
    var max_record_cycles: usize = 0;
    var source_done_cycle: ?usize = null;
    var final_flush_cycles: usize = 0;
    var cycle: usize = 0;

    while (c.dut_done(dut.handle) == 0) : (cycle += 1) {
        if (cycle > rows * @max(tokens, 1) * 48 + 16_384)
            return error.RunTimeout;

        if (!inverse_presenting and inverse_index < tokens and
            (!options.stalls or random.uintLessThan(u8, 4) != 0))
            inverse_presenting = true;
        const inverse_token = if (options.wrong_inverse_token == inverse_index)
            (inverse_index + 1) & 3
        else
            inverse_index;
        c.dut_set_inverse(
            dut.handle,
            @intFromBool(inverse_presenting),
            @intCast(inverse_token),
            if (inverse_index < inverses.len) inverses[inverse_index] else 0,
            @intFromBool(inverse_index + 1 == tokens),
        );

        if (!response_active) {
            if (pending) |owner| {
                if (cycle >= owner.due_cycle) {
                    response_lanes = responseFor(owner.request, options);
                    response_error = owner.inject_error;
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

        var output_ready = !options.stalls or random.uintLessThan(u8, 4) != 0;
        c.dut_set_output_ready(dut.handle, @intFromBool(output_ready));
        dut.eval();
        const output_valid = c.dut_output_valid(dut.handle) != 0;
        const native_beat_index = beat_count % 5;
        if (output_valid and options.hold_each_beat and record_count == 0 and
            forced_holds[native_beat_index] != 0)
        {
            output_ready = false;
            forced_holds[native_beat_index] -= 1;
            c.dut_set_output_ready(dut.handle, 0);
            dut.eval();
        }

        const inverse_fire = inverse_presenting and
            c.dut_inverse_ready(dut.handle) != 0;
        const request_valid = c.dut_read_request_valid(dut.handle) != 0;
        const request_fire = request_valid and request_ready;
        const response_fire = response_active and
            c.dut_read_response_ready(dut.handle) != 0;
        const output_fire = output_valid and output_ready;
        const scalar_fire = c.dut_debug_scalar_fire(dut.handle) != 0;

        if (c.dut_debug_source_done(dut.handle) != 0 and source_done_cycle == null)
            source_done_cycle = cycle;

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

        if (output_valid) {
            const current = readBeat(dut);
            if (expected_record == null)
                expected_record = try expectedRecord(
                    gamma,
                    inverses,
                    rows,
                    record_count,
                    options,
                );
            try checkBeat(current, expected_record.?, native_beat_index);
            if (!output_ready) {
                if (held_beat) |prior| try std.testing.expectEqual(prior, current);
                held_beat = current;
            } else {
                held_beat = null;
            }
        } else {
            held_beat = null;
        }

        if (request_fire) {
            const request = readRequest(dut);
            const groups_per_token = rows / 8;
            try std.testing.expectEqual(@as(u2, 0), request.role);
            try std.testing.expectEqual(
                @as(u3, @intCast(request_count / groups_per_token)),
                request.token,
            );
            try std.testing.expectEqual(
                @as(u11, @intCast(request_count % groups_per_token)),
                request.group,
            );
            pending = .{
                .request = request,
                .due_cycle = cycle + (options.fixed_response_delay orelse
                    2 + random.uintLessThan(usize, 6)),
                .inject_error = options.scratch_error_request == request_count,
            };
            request_count += 1;
        }

        if (scalar_fire) scalar_count += 1;
        if (output_fire) {
            beat_count += 1;
            if (native_beat_index == 4) {
                record_count += 1;
                expected_record = null;
                const interval = cycle - last_record_cycle;
                max_record_cycles = @max(max_record_cycles, interval);
                last_record_cycle = cycle;
                if (options.measure_cadence)
                    try std.testing.expect(interval <= 750);
                if (source_done_cycle) |source_cycle| {
                    if (record_count == rows / 32 * tokens) {
                        final_flush_cycles = cycle - source_cycle;
                        if (options.measure_cadence)
                            try std.testing.expect(final_flush_cycles <= 281);
                    }
                }
            }
        }

        dut.step();
        if (inverse_fire) {
            inverse_index += 1;
            inverse_presenting = false;
        }
        if (response_fire) {
            response_active = false;
            response_error = false;
            response_count += 1;
        }
    }

    try std.testing.expect(c.dut_busy(dut.handle) == 0);
    try std.testing.expect(c.dut_output_valid(dut.handle) == 0);
    const expected_error = options.expected_status != null;
    try std.testing.expectEqual(expected_error, c.dut_error(dut.handle) != 0);
    try std.testing.expectEqual(options.expected_status orelse 0, runStatus(dut));
    if (expected_error) {
        try std.testing.expect(c.dut_gamma_valid(dut.handle) == 0);
    } else {
        try std.testing.expect(c.dut_gamma_valid(dut.handle) != 0);
        try std.testing.expectEqual(tokens, inverse_index);
    }
    try std.testing.expectEqual(
        options.expected_requests orelse (rows / 8 * tokens),
        request_count,
    );
    try std.testing.expectEqual(request_count, response_count);
    try std.testing.expectEqual(
        options.expected_scalars orelse (rows * tokens),
        scalar_count,
    );
    const expected_records = options.expected_records orelse (rows / 32 * tokens);
    try std.testing.expectEqual(expected_records, record_count);
    try std.testing.expectEqual(expected_records * 5, beat_count);
    if (options.hold_each_beat)
        try std.testing.expectEqualSlices(usize, &([_]usize{0} ** 5), &forced_holds);
    if (options.measure_cadence) {
        try std.testing.expect(source_done_cycle != null);
        try std.testing.expect(final_flush_cycles <= 281);
    }

    dut.clearInputs();
    dut.step();
    try std.testing.expect(c.dut_done(dut.handle) == 0);
    try std.testing.expect(c.dut_output_valid(dut.handle) == 0);
    return .{
        .cycles = cycle + 1,
        .requests = request_count,
        .responses = response_count,
        .scalars = scalar_count,
        .records = record_count,
        .beats = beat_count,
        .max_record_cycles = max_record_cycles,
        .final_flush_cycles = final_flush_cycles,
    };
}

fn sendInverses(dut: *Dut, inverses: []const u32) !void {
    for (inverses, 0..) |bits, token| {
        var cycles: usize = 0;
        while (true) : (cycles += 1) {
            if (cycles > 64) return error.InverseTimeout;
            c.dut_set_inverse(
                dut.handle,
                1,
                @intCast(token),
                bits,
                @intFromBool(token + 1 == inverses.len),
            );
            dut.eval();
            const fire = c.dut_inverse_ready(dut.handle) != 0;
            dut.step();
            if (fire) break;
        }
    }
    c.dut_set_inverse(dut.handle, 0, 0, 0, 0);
    dut.eval();
}

fn verifySimultaneousGammaPriority(dut: *Dut, gamma: []const u32) !void {
    try std.testing.expect(c.dut_gamma_config_ready(dut.handle) != 0);
    try std.testing.expect(c.dut_run_config_ready(dut.handle) != 0);
    c.dut_set_gamma_config(dut.handle, 1, @intCast(gamma.len));
    c.dut_set_run_config(dut.handle, 1, @intCast(gamma.len), 1);
    dut.eval();
    try std.testing.expect(c.dut_gamma_config_ready(dut.handle) != 0);
    try std.testing.expect(c.dut_run_config_ready(dut.handle) == 0);
    dut.step();
    c.dut_set_gamma_config(dut.handle, 0, 0);
    c.dut_set_run_config(dut.handle, 0, 0, 0);

    for (0..gamma.len / 2) |word| {
        c.dut_set_gamma_stream(
            dut.handle,
            1,
            gammaWord(gamma, word, null),
            0xff,
            @intFromBool(word + 1 == gamma.len / 2),
        );
        dut.eval();
        try std.testing.expect(c.dut_gamma_stream_ready(dut.handle) != 0);
        try std.testing.expect(c.dut_busy(dut.handle) == 0);
        try std.testing.expect(c.dut_done(dut.handle) == 0);
        try std.testing.expect(c.dut_error(dut.handle) == 0);
        try std.testing.expectEqual(@as(u16, 0), runStatus(dut));
        dut.step();
    }
    c.dut_set_gamma_stream(dut.handle, 0, 0, 0, 0);
    dut.eval();
    try std.testing.expect(c.dut_gamma_done(dut.handle) != 0);
    try std.testing.expect(c.dut_gamma_error(dut.handle) == 0);
    try std.testing.expect(c.dut_gamma_valid(dut.handle) != 0);
    try std.testing.expect(c.dut_busy(dut.handle) == 0);
    try std.testing.expect(c.dut_done(dut.handle) == 0);
    try std.testing.expect(c.dut_error(dut.handle) == 0);
    try std.testing.expectEqual(@as(u16, 0), runStatus(dut));
    dut.step();
    try std.testing.expect(c.dut_gamma_done(dut.handle) == 0);
    try std.testing.expect(c.dut_run_config_ready(dut.handle) != 0);
}

const AbortTarget = enum {
    outstanding_read,
    mul1,
    mul2,
    q8_wait,
    beat0,
    beat1,
    beat2,
    beat3,
    beat4,
    q8_drain,
    final_edge,
};

fn targetBeat(target: AbortTarget) ?usize {
    return switch (target) {
        .beat0 => 0,
        .beat1 => 1,
        .beat2 => 2,
        .beat3 => 3,
        .beat4 => 4,
        else => null,
    };
}

fn abortReached(dut: *Dut, target: AbortTarget, owner_pending: bool) bool {
    return switch (target) {
        .outstanding_read => owner_pending and c.dut_debug_source_state(dut.handle) == 4,
        .mul1 => c.dut_debug_source_state(dut.handle) == 6,
        .mul2 => c.dut_debug_source_state(dut.handle) == 8,
        .q8_wait => c.dut_debug_q8_state(dut.handle) == 4,
        .q8_drain => c.dut_debug_wrapper_state(dut.handle) == 2 and
            c.dut_debug_q8_state(dut.handle) == 4,
        else => false,
    };
}

fn finishSilentAbort(dut: *Dut) !void {
    c.dut_set_abort(dut.handle, 1);
    dut.eval();
    try std.testing.expect(c.dut_output_valid(dut.handle) == 0);
    try std.testing.expect(c.dut_done(dut.handle) == 0);
    try std.testing.expect(c.dut_error(dut.handle) == 0);
    try std.testing.expectEqual(@as(u16, 0), runStatus(dut));
    dut.step();
    c.dut_set_abort(dut.handle, 0);
    dut.eval();

    var cycles: usize = 0;
    while (c.dut_busy(dut.handle) != 0 or
        c.dut_gamma_busy(dut.handle) != 0) : (cycles += 1)
    {
        if (cycles > 64) return error.AbortCleanupTimeout;
        try std.testing.expect(c.dut_output_valid(dut.handle) == 0);
        try std.testing.expect(c.dut_done(dut.handle) == 0);
        try std.testing.expect(c.dut_error(dut.handle) == 0);
        try std.testing.expectEqual(@as(u16, 0), runStatus(dut));
        try std.testing.expect(c.dut_gamma_done(dut.handle) == 0);
        try std.testing.expect(c.dut_gamma_error(dut.handle) == 0);
        dut.step();
    }

    try std.testing.expect(c.dut_done(dut.handle) == 0);
    try std.testing.expect(c.dut_error(dut.handle) == 0);
    try std.testing.expectEqual(@as(u16, 0), runStatus(dut));
    try std.testing.expect(c.dut_gamma_done(dut.handle) == 0);
    try std.testing.expect(c.dut_gamma_error(dut.handle) == 0);
    try std.testing.expectEqual(@as(u4, 0), gammaStatus(dut));
    try std.testing.expect(c.dut_gamma_valid(dut.handle) == 0);
    try std.testing.expect(c.dut_output_valid(dut.handle) == 0);
    dut.clearInputs();
}

fn verifyAbortGammaLoad(dut: *Dut, gamma: []const u32) !void {
    try configureGamma(dut, gamma.len);
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
        try std.testing.expect(c.dut_gamma_busy(dut.handle) != 0);
        try std.testing.expect(c.dut_gamma_valid(dut.handle) == 0);
        dut.step();
    }
    c.dut_set_gamma_stream(dut.handle, 0, 0, 0, 0);
    dut.eval();
    try std.testing.expect(c.dut_gamma_busy(dut.handle) != 0);
    try finishSilentAbort(dut);

    try reloadGamma(dut, gamma, 0x3200);
    const inverse = makeInverses(1);
    _ = try runSource(dut, gamma, inverse[0..1], 128, 1, .{}, 0x3201);
}

fn verifyAbortInverseIntake(dut: *Dut, gamma: []const u32) !void {
    const inverses = makeInverses(4);
    try configureRun(dut, 128, 4);
    c.dut_set_inverse(dut.handle, 1, 0, inverses[0], 0);
    dut.eval();
    try std.testing.expect(c.dut_inverse_ready(dut.handle) != 0);
    try std.testing.expect(c.dut_busy(dut.handle) != 0);
    dut.step();
    c.dut_set_inverse(dut.handle, 0, 0, 0, 0);
    dut.eval();
    try std.testing.expect(c.dut_inverse_ready(dut.handle) != 0);
    try std.testing.expect(c.dut_busy(dut.handle) != 0);
    try finishSilentAbort(dut);

    try reloadGamma(dut, gamma, 0x3210);
    _ = try runSource(dut, gamma, inverses[0..1], 128, 1, .{}, 0x3211);
}

fn verifyAbortPhase(
    dut: *Dut,
    gamma: []const u32,
    target: AbortTarget,
    seed: u64,
) !void {
    try reloadGamma(dut, gamma, seed);
    const inverse = makeInverses(1);
    try configureRun(dut, 128, 1);
    try sendInverses(dut, inverse[0..1]);

    const options = RunOptions{ .stalls = false, .fixed_response_delay = 2 };
    var pending: ?PendingResponse = null;
    var response_active = false;
    var response_lanes = zero_lanes;
    var request_count: usize = 0;
    var beat_count: usize = 0;
    var expected_record: ?ExpectedRecord = null;
    var reached = false;
    var cycle: usize = 0;

    while (!reached) : (cycle += 1) {
        if (cycle > 128 * 48 + 4096) return error.AbortPhaseTimeout;

        if (!response_active) {
            if (pending) |owner| {
                if (target != .outstanding_read and cycle >= owner.due_cycle) {
                    response_lanes = responseFor(owner.request, options);
                    response_active = true;
                    pending = null;
                }
            }
        }
        c.dut_set_read_response(
            dut.handle,
            @intFromBool(response_active),
            &response_lanes,
            0,
        );
        c.dut_set_read_request_ready(dut.handle, @intFromBool(pending == null and !response_active));
        c.dut_set_output_ready(dut.handle, 1);
        dut.eval();

        const request_fire = c.dut_read_request_valid(dut.handle) != 0 and
            pending == null and !response_active;
        const response_fire = response_active and
            c.dut_read_response_ready(dut.handle) != 0;
        const output_valid = c.dut_output_valid(dut.handle) != 0;
        const beat_index = beat_count % 5;
        const final_beat = beat_count == (128 / 32) * 5 - 1;

        if (output_valid) {
            const current = readBeat(dut);
            if (expected_record == null)
                expected_record = try expectedRecord(
                    gamma,
                    inverse[0..1],
                    128,
                    beat_count / 5,
                    options,
                );
            try checkBeat(current, expected_record.?, beat_index);

            const selected_beat = if (targetBeat(target)) |wanted|
                beat_index == wanted
            else
                false;
            if (selected_beat or (target == .final_edge and final_beat)) {
                c.dut_set_output_ready(dut.handle, 0);
                dut.eval();
                const held = readBeat(dut);
                for (0..3) |_| {
                    try std.testing.expect(c.dut_output_valid(dut.handle) != 0);
                    try std.testing.expectEqual(held, readBeat(dut));
                    dut.step();
                }
                reached = true;
            }
        }

        if (!reached and abortReached(dut, target, pending != null))
            reached = true;

        if (request_fire) {
            const request = readRequest(dut);
            try std.testing.expectEqual(@as(u2, 0), request.role);
            try std.testing.expectEqual(@as(u3, 0), request.token);
            try std.testing.expectEqual(@as(u11, @intCast(request_count)), request.group);
            pending = .{
                .request = request,
                .due_cycle = cycle + 2,
                .inject_error = false,
            };
            request_count += 1;
        }

        if (reached) break;
        if (output_valid) {
            beat_count += 1;
            if (beat_index == 4) expected_record = null;
        }
        dut.step();
        if (response_fire) response_active = false;
    }

    try std.testing.expect(reached);
    const was_final_edge = target == .final_edge;
    c.dut_set_output_ready(dut.handle, @intFromBool(was_final_edge));
    c.dut_set_abort(dut.handle, 1);
    dut.eval();
    try std.testing.expect(c.dut_output_valid(dut.handle) == 0);
    try std.testing.expect(c.dut_done(dut.handle) == 0);
    try std.testing.expect(c.dut_error(dut.handle) == 0);
    dut.step();
    c.dut_set_abort(dut.handle, 0);

    // Any accepted untagged scratch owner must be drained after the abort. The
    // payload is deliberately arbitrary because cleanup must discard it.
    if (pending) |owner| {
        response_lanes = responseFor(owner.request, options);
        response_active = true;
        pending = null;
    }
    var cleanup_cycles: usize = 0;
    while (c.dut_busy(dut.handle) != 0) : (cleanup_cycles += 1) {
        if (cleanup_cycles > 64) return error.AbortCleanupTimeout;
        c.dut_set_read_request_ready(dut.handle, 0);
        c.dut_set_read_response(
            dut.handle,
            @intFromBool(response_active),
            &response_lanes,
            0,
        );
        c.dut_set_output_ready(dut.handle, 1);
        dut.eval();
        try std.testing.expect(c.dut_output_valid(dut.handle) == 0);
        try std.testing.expect(c.dut_done(dut.handle) == 0);
        try std.testing.expect(c.dut_error(dut.handle) == 0);
        try std.testing.expectEqual(@as(u16, 0), runStatus(dut));
        const drain_fire = response_active and
            c.dut_read_response_ready(dut.handle) != 0;
        dut.step();
        if (drain_fire) response_active = false;
    }

    try std.testing.expect(!response_active);
    try std.testing.expect(c.dut_done(dut.handle) == 0);
    try std.testing.expect(c.dut_error(dut.handle) == 0);
    try std.testing.expectEqual(@as(u16, 0), runStatus(dut));
    try std.testing.expect(c.dut_gamma_valid(dut.handle) == 0);
    try std.testing.expect(c.dut_output_valid(dut.handle) == 0);
    dut.clearInputs();
}

fn verifyAborts(dut: *Dut, gamma: []const u32) !void {
    try verifyAbortGammaLoad(dut, gamma);
    try verifyAbortInverseIntake(dut, gamma);

    const targets = [_]AbortTarget{
        .outstanding_read,
        .mul1,
        .mul2,
        .q8_wait,
        .beat0,
        .beat1,
        .beat2,
        .beat3,
        .beat4,
        .q8_drain,
        .final_edge,
    };
    for (targets, 0..) |target, index|
        try verifyAbortPhase(dut, gamma, target, 0x3000 + index);

    // The final run proves every abort path permits a no-reset gamma reload and
    // clean restart; all preceding targets also reloaded without reset.
    try reloadGamma(dut, gamma, 0x30ff);
    const inverse = makeInverses(1);
    _ = try runSource(dut, gamma, inverse[0..1], 128, 1, .{}, 0x31ff);
}

fn verifyGammaFaults(dut: *Dut, gamma: []const u32) !void {
    _ = try loadGamma(dut, gamma, 64, 0x1100, section.RmsNormGammaStatus.bad_cfg, null, null);
    _ = try loadGamma(dut, gamma, 128, 0x1101, section.RmsNormGammaStatus.frame, 2, null);
    _ = try loadGamma(dut, gamma, 128, 0x1102, section.RmsNormGammaStatus.nonfinite, null, 5);
    try reloadGamma(dut, gamma, 0x1103);
}

fn verifyRunFaults(dut: *Dut, gamma: []const u32, ones: []u32) !void {
    const inverses = makeInverses(4);

    _ = try runSource(dut, gamma, inverses[0..1], 64, 1, .{
        .expected_status = weighted_bad_cfg,
        .expected_requests = 0,
        .expected_scalars = 0,
        .expected_records = 0,
    }, 0x2100);

    try reloadGamma(dut, gamma, 0x2101);
    _ = try runSource(dut, gamma, inverses[0..1], 256, 1, .{
        .expected_status = weighted_gamma,
        .expected_requests = 0,
        .expected_scalars = 0,
        .expected_records = 0,
    }, 0x2102);

    try reloadGamma(dut, gamma, 0x2103);
    _ = try runSource(dut, gamma, inverses[0..1], 128, 1, .{
        .wrong_inverse_token = 0,
        .expected_status = weighted_inverse,
        .expected_requests = 0,
        .expected_scalars = 0,
        .expected_records = 0,
    }, 0x2104);

    try reloadGamma(dut, gamma, 0x2105);
    _ = try runSource(dut, gamma, inverses[0..1], 128, 1, .{
        .scratch_error_request = 8,
        .expected_status = weighted_scratch,
        .expected_requests = 9,
        .expected_scalars = 64,
        .expected_records = 1,
    }, 0x2106);

    try reloadGamma(dut, gamma, 0x2107);
    _ = try runSource(dut, gamma, inverses[0..1], 128, 1, .{
        .residual_override_index = 64,
        .residual_override_bits = 0x7f80_0000,
        .expected_status = @as(u16, section.RmsNormWeightedSourceStatus.mul1(
            section.RmsNormMulStatus.nonfinite_input,
        )),
        .expected_requests = 9,
        .expected_scalars = 64,
        .expected_records = 1,
    }, 0x2108);

    const saved_gamma = ones[64];
    ones[64] = 0x7f7f_ffff;
    try reloadGamma(dut, ones, 0x2109);
    const one_inverse = [_]u32{0x3f80_0000};
    _ = try runSource(dut, ones, &one_inverse, 128, 1, .{
        .residual_override_index = 64,
        .residual_override_bits = 0x4000_0000,
        .expected_status = @as(u16, section.RmsNormWeightedSourceStatus.mul2(
            section.RmsNormMulStatus.overflow,
        )),
        .expected_requests = 9,
        .expected_scalars = 64,
        .expected_records = 1,
    }, 0x210a);
    ones[64] = saved_gamma;

    try reloadGamma(dut, ones, 0x210b);
    _ = try runSource(dut, ones, &one_inverse, 128, 1, .{
        .residual_mode = .overflow_second_block,
        .expected_status = q8_scale,
        .expected_requests = 9,
        .expected_scalars = 64,
        .expected_records = 1,
    }, 0x210c);

    try reloadGamma(dut, ones, 0x210d);
    _ = try runSource(dut, ones, &one_inverse, 128, 1, .{
        .residual_mode = .tiny_second_block,
        .expected_status = q8_arith,
        .expected_requests = 9,
        .expected_scalars = 64,
        .expected_records = 1,
    }, 0x210e);

    try reloadGamma(dut, gamma, 0x210f);
    _ = try runSource(dut, gamma, inverses[0..1], 128, 1, .{}, 0x2110);
}

pub fn main() !void {
    var dut = Dut.init();
    defer dut.deinit();
    dut.reset();

    const allocator = std.heap.page_allocator;
    const gamma128 = try makeGamma(allocator, 128);
    defer allocator.free(gamma128);
    const ones128 = try makeOnes(allocator, 128);
    defer allocator.free(ones128);

    try verifyGammaFaults(&dut, gamma128);
    try verifySimultaneousGammaPriority(&dut, gamma128);
    try verifyRunFaults(&dut, gamma128, ones128);
    try verifyAborts(&dut, gamma128);

    var total_scalars: usize = 0;
    var total_records: usize = 0;
    var total_beats: usize = 0;
    var total_cycles: usize = 0;
    var cadence: RunStats = undefined;

    for (rows_set, 0..) |rows, row_index| {
        const gamma = try makeGamma(allocator, rows);
        defer allocator.free(gamma);
        try reloadGamma(&dut, gamma, 0x4000 + row_index);
        const inverses = makeInverses(4);
        for (1..5) |tokens| {
            const stats = try runSource(
                &dut,
                gamma,
                inverses[0..tokens],
                rows,
                tokens,
                .{ .hold_each_beat = rows == 128 and tokens == 1 },
                0x5000 + row_index * 8 + tokens,
            );
            total_scalars += stats.scalars;
            total_records += stats.records;
            total_beats += stats.beats;
            total_cycles += stats.cycles;
        }
        if (rows == 128) {
            cadence = try runSource(
                &dut,
                gamma,
                inverses[0..1],
                rows,
                1,
                .{
                    .stalls = false,
                    .fixed_response_delay = 2,
                    .measure_cadence = true,
                },
                0x5cad,
            );
        }
    }

    std.debug.print(
        "section_rmsnorm_q8_source cosim:\n" ++
            "  legal shapes=24, exact scalars={d}, records={d}, native beats={d}\n" ++
            "  randomized lifecycle cycles={d}, cadence max={d}/record, final flush={d}\n" ++
            "  priority, reuse, tags, TLAST, faults, quarantine, abort phases=13: passed\n\n",
        .{
            total_scalars,
            total_records,
            total_beats,
            total_cycles,
            cadence.max_record_cycles,
            cadence.final_flush_cycles,
        },
    );
}
