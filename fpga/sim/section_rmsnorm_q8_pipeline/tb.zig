//! End-to-end cosim for the autonomous scratch-backed RMSNorm-to-Q8 pipeline.
//!
//! The software scratch model observes the direct residual writes, then serves
//! both reduction and weighted-source replay through the shared tagged read
//! port. Clean output is checked bit-for-bit against the fixed inverse-RMS,
//! weighted F32, and canonical Q8_0 software oracles.

const std = @import("std");
const section = @import("shared_section");
const layout = section.layout;
const c = @cImport(@cInclude("shim.h"));

const eps_1e_6: u32 = 0x3586_37bd;
const rows_set = [_]usize{ 128, 256, 512, 1024, 2048, 4096 };
const zero_lanes = [_]u64{ 0, 0, 0, 0 };

fn reduceStatus(raw: u13) u30 {
    return section.RmsNormQ8PipelineStatus.reduce(raw);
}

fn sourceStatus(raw: u16) u30 {
    return section.RmsNormQ8PipelineStatus.source(raw);
}

const reduce_loader = reduceStatus(section.RmsNormReduceStatus.frontend(
    section.RmsNormFrontendStatus.loader,
));
const reduce_scratch = reduceStatus(section.RmsNormReduceStatus.frontend(
    section.RmsNormFrontendStatus.scratch,
));
const reduce_inverse = reduceStatus(section.RmsNormReduceStatus.inverse(
    section.RmsNormInvStatus.arithmetic,
));
const source_gamma = sourceStatus(section.RmsNormQ8SourceStatus.weighted(
    section.RmsNormWeightedSourceStatus.gamma,
));
const source_scratch = sourceStatus(section.RmsNormQ8SourceStatus.weighted(
    section.RmsNormWeightedSourceStatus.scratch,
));
const source_q8_scale = sourceStatus(section.RmsNormQ8SourceStatus.q8(1 << 1));

const Scratch = struct {
    banks: [4][2048]u64 = .{.{0} ** 2048} ** 4,

    fn write(self: *Scratch, bank: u8, address: u16, data: u64) !void {
        try std.testing.expect(bank < 4);
        try std.testing.expect(address < 2048);
        self.banks[bank][address] = data;
    }

    fn read(self: *const Scratch, token: u8, group: u16) ![4]u64 {
        try std.testing.expect(token < 4);
        try std.testing.expect(group < 512);
        const address = @as(usize, token) * 512 + group;
        return .{
            self.banks[0][address],
            self.banks[1][address],
            self.banks[2][address],
            self.banks[3][address],
        };
    }
};

const Request = struct {
    role: u2,
    token: u3,
    group: u11,
};

const PendingResponse = struct {
    request: Request,
    due_cycle: usize,
    inject_error: bool,
};

const NativeBeat = struct {
    data: u64,
    last: bool,
    token: u2,
    block: u9,
};

const ScratchWrite = struct {
    bank: u2,
    address: u14,
    data: u64,
};

const ExpectedRecord = struct {
    data: [5]u64,
    token: u2,
    block: u9,
};

const RunOptions = struct {
    stalls: bool = true,
    fixed_response_delay: ?usize = null,
    hold_first_record_beats: bool = false,
    early_last_word: ?usize = null,
    partial_keep_word: ?usize = null,
    omit_final_last: bool = false,
    write_error_word: ?usize = null,
    read_error_request: ?usize = null,
    expected_status: ?u30 = null,
    expected_writes: ?usize = null,
    expected_requests: ?usize = null,
    expected_records: ?usize = null,
    oracle_valid: bool = true,
};

const RunStats = struct {
    cycles: usize,
    writes: usize,
    requests: usize,
    responses: usize,
    records: usize,
    beats: usize,
    final_output_fires: usize,
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
        c.dut_set_run_config(self.handle, 0, 0, 0, 0);
        c.dut_set_residual_stream(self.handle, 0, 0, 0, 0);
        c.dut_set_r_write_sink(self.handle, 0, 0);
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

fn pipelineStatus(dut: *Dut) u30 {
    return @truncate(c.dut_status(dut.handle));
}

fn gammaStatus(dut: *Dut) u4 {
    return @truncate(c.dut_gamma_status(dut.handle));
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
        const fraction: u32 = @intCast(
            (row *% 0x15a4e3 +% 0x224466) & 0x007f_ffff,
        );
        bits.* = sign | (exponent << 23) | fraction;
    }
    return gamma;
}

fn makeOnes(allocator: std.mem.Allocator, rows: usize) ![]u32 {
    const gamma = try allocator.alloc(u32, rows);
    @memset(gamma, 0x3f80_0000);
    return gamma;
}

fn makeResiduals(
    allocator: std.mem.Allocator,
    rows: usize,
    tokens: usize,
) ![]u32 {
    const values = try allocator.alloc(u32, rows * tokens);
    for (0..tokens) |token| {
        for (0..rows) |row| {
            const exponent: u8 = @intCast(90 + ((row * 13 + token * 7) % 32));
            const fraction: u23 = @truncate(
                row *% 0x12d687 +% token *% 0x34567,
            );
            const sign: u32 = @intFromBool((row + token) % 3 == 0);
            values[token * rows + row] = (sign << 31) |
                (@as(u32, exponent) << 23) | fraction;
        }
    }
    return values;
}

fn gammaWord(gamma: []const u32, word: usize, nonfinite: ?usize) u64 {
    var lo = gamma[word * 2];
    var hi = gamma[word * 2 + 1];
    if (nonfinite == word * 2) lo = 0x7fc1_2345;
    if (nonfinite == word * 2 + 1) hi = 0xff80_0000;
    return @as(u64, lo) | (@as(u64, hi) << 32);
}

fn residualWord(values: []const u32, word: usize) u64 {
    return @as(u64, values[word * 2]) |
        (@as(u64, values[word * 2 + 1]) << 32);
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
    partial_keep_word: ?usize,
    nonfinite_scalar: ?usize,
) !usize {
    try configureGamma(dut, rows);
    var prng = std.Random.DefaultPrng.init(seed);
    const random = prng.random();
    const words = rows / 2;
    var word: usize = 0;
    var presenting = false;
    var held_data: u64 = 0;
    var held_keep: u8 = 0xff;
    var held_last = false;
    var cycles: usize = 0;

    while (c.dut_gamma_done(dut.handle) == 0) : (cycles += 1) {
        if (cycles > words * 12 + 128) return error.GammaTimeout;
        if (!presenting and word < words and random.uintLessThan(u8, 4) != 0) {
            presenting = true;
            held_data = gammaWord(gamma, word, nonfinite_scalar);
            held_keep = if (partial_keep_word == word) 0x0f else 0xff;
            held_last = early_last_word == word or
                (early_last_word == null and word + 1 == words);
        }
        c.dut_set_gamma_stream(
            dut.handle,
            @intFromBool(presenting),
            held_data,
            held_keep,
            @intFromBool(held_last),
        );
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
    _ = try loadGamma(dut, gamma, gamma.len, seed, null, null, null, null);
}

fn configureRun(
    dut: *Dut,
    rows: usize,
    tokens: usize,
    eps: u32,
) !void {
    try std.testing.expect(c.dut_run_config_ready(dut.handle) != 0);
    c.dut_set_run_config(
        dut.handle,
        1,
        @intCast(rows),
        @intCast(tokens),
        eps,
    );
    dut.eval();
    try std.testing.expect(c.dut_run_config_ready(dut.handle) != 0);
    dut.step();
    c.dut_set_run_config(dut.handle, 0, 0, 0, 0);
    dut.eval();
}

fn expectRejectedStart(
    dut: *Dut,
    rows: usize,
    tokens: usize,
    residual_data: u64,
    expected_status: u30,
) !void {
    try std.testing.expect(c.dut_run_config_ready(dut.handle) != 0);
    c.dut_set_run_config(
        dut.handle,
        1,
        @intCast(rows),
        @intCast(tokens),
        eps_1e_6,
    );
    c.dut_set_residual_stream(dut.handle, 1, residual_data, 0xff, 0);
    c.dut_set_r_write_sink(dut.handle, 1, 0);
    c.dut_set_read_request_ready(dut.handle, 1);
    c.dut_set_output_ready(dut.handle, 1);
    dut.eval();

    try std.testing.expect(c.dut_run_config_ready(dut.handle) != 0);
    try std.testing.expect(c.dut_residual_stream_ready(dut.handle) == 0);
    try std.testing.expect(c.dut_r_write_valid(dut.handle) == 0);
    try std.testing.expect(c.dut_read_request_valid(dut.handle) == 0);
    try std.testing.expect(c.dut_output_valid(dut.handle) == 0);
    try std.testing.expect(c.dut_debug_output_fire(dut.handle) == 0);
    try std.testing.expect(c.dut_debug_final_output_fire(dut.handle) == 0);
    try std.testing.expect(c.dut_busy(dut.handle) == 0);
    try std.testing.expect(c.dut_done(dut.handle) == 0);
    dut.step();
    c.dut_set_run_config(dut.handle, 0, 0, 0, 0);
    dut.eval();

    try std.testing.expectEqual(@as(u8, 3), c.dut_debug_state(dut.handle));
    try std.testing.expect(c.dut_residual_stream_ready(dut.handle) == 0);
    try std.testing.expect(c.dut_r_write_valid(dut.handle) == 0);
    try std.testing.expect(c.dut_read_request_valid(dut.handle) == 0);
    try std.testing.expect(c.dut_output_valid(dut.handle) == 0);
    try std.testing.expect(c.dut_debug_output_fire(dut.handle) == 0);
    try std.testing.expect(c.dut_debug_final_output_fire(dut.handle) == 0);
    try std.testing.expect(c.dut_done(dut.handle) == 0);
    try std.testing.expect(c.dut_error(dut.handle) == 0);
    try std.testing.expectEqual(@as(u30, 0), pipelineStatus(dut));

    var cycles: usize = 0;
    while (c.dut_done(dut.handle) == 0) : (cycles += 1) {
        if (cycles > 128) return error.RejectedStartTimeout;
        dut.eval();
        try std.testing.expect(c.dut_residual_stream_ready(dut.handle) == 0);
        try std.testing.expect(c.dut_r_write_valid(dut.handle) == 0);
        try std.testing.expect(c.dut_read_request_valid(dut.handle) == 0);
        try std.testing.expect(c.dut_output_valid(dut.handle) == 0);
        try std.testing.expect(c.dut_debug_output_fire(dut.handle) == 0);
        try std.testing.expect(c.dut_debug_final_output_fire(dut.handle) == 0);
        dut.step();
    }

    try std.testing.expect(c.dut_busy(dut.handle) == 0);
    try std.testing.expect(c.dut_error(dut.handle) != 0);
    try std.testing.expectEqual(expected_status, pipelineStatus(dut));
    try std.testing.expect(c.dut_gamma_valid(dut.handle) == 0);
    try std.testing.expect(c.dut_residual_stream_ready(dut.handle) == 0);
    try std.testing.expect(c.dut_r_write_valid(dut.handle) == 0);
    try std.testing.expect(c.dut_read_request_valid(dut.handle) == 0);
    try std.testing.expect(c.dut_output_valid(dut.handle) == 0);
    try std.testing.expect(c.dut_debug_final_output_fire(dut.handle) == 0);
    dut.clearInputs();
    dut.step();
    try std.testing.expect(c.dut_done(dut.handle) == 0);
}

fn verifyRejectedStarts(
    dut: *Dut,
    gamma: []const u32,
    residual_data: u64,
) !void {
    try std.testing.expect(c.dut_gamma_valid(dut.handle) == 0);
    try expectRejectedStart(dut, 128, 1, residual_data, source_gamma);

    try reloadGamma(dut, gamma, 0x1000);
    try std.testing.expect(c.dut_gamma_valid(dut.handle) != 0);
    try expectRejectedStart(dut, 256, 1, residual_data, source_gamma);

    try reloadGamma(dut, gamma, 0x1001);
}

fn computeInverses(
    values: []const u32,
    rows: usize,
    tokens: usize,
    eps: u32,
) ![4]u32 {
    var result = [_]u32{0} ** 4;
    for (0..tokens) |token| {
        const token_values = values[token * rows ..][0..rows];
        const scan = try section.rmsNormMaxExp(token_values, @intCast(rows));
        if (scan.subnormal_warning) return error.OracleSubnormal;
        const sum = try section.rmsNormSumsqFixed(
            token_values,
            @intCast(rows),
            scan.max_exp,
        );
        result[token] = (try section.rmsNormInvFixed(
            sum.sum_sq,
            @intCast(rows),
            scan.max_exp,
            eps,
        )).inv_rms_bits;
    }
    return result;
}

fn expectedRecord(
    values: []const u32,
    gamma: []const u32,
    inverses: []const u32,
    rows: usize,
    record_index: usize,
) !ExpectedRecord {
    const blocks_per_token = rows / layout.q8_block;
    const token = record_index / blocks_per_token;
    const block = record_index % blocks_per_token;
    var normalized: [layout.q8_block]f32 = undefined;
    for (&normalized, 0..) |*value, lane| {
        const row = block * layout.q8_block + lane;
        const weighted = section.rmsNormWeightedBits(
            values[token * rows + row],
            inverses[token],
            gamma[row],
        );
        if (weighted.mul1_status != 0) return error.OracleMul1Fault;
        if (weighted.mul2_status != 0) return error.OracleMul2Fault;
        value.* = @bitCast(weighted.bits);
    }

    var quants: [layout.q8_block]i8 = undefined;
    var scales: [1]f16 = undefined;
    try layout.quantizeQ8_0(&normalized, &quants, &scales);
    var data = [_]u64{0} ** 5;
    for (0..4) |beat| {
        for (0..8) |lane| {
            const byte: u8 = @bitCast(quants[beat * 8 + lane]);
            data[beat] |= @as(u64, byte) <<
                @as(u6, @intCast(lane * 8));
        }
    }
    data[4] = @as(u16, @bitCast(scales[0]));
    return .{
        .data = data,
        .token = @intCast(token),
        .block = @intCast(block),
    };
}

fn checkBeat(
    actual: NativeBeat,
    expected: ExpectedRecord,
    beat_index: usize,
) !void {
    if (actual.data != expected.data[beat_index]) {
        std.debug.print(
            "native mismatch token={d} block={d} beat={d}: " ++
                "got=0x{x:0>16} expected=0x{x:0>16}\n",
            .{
                expected.token,
                expected.block,
                beat_index,
                actual.data,
                expected.data[beat_index],
            },
        );
        return error.NativeDataMismatch;
    }
    try std.testing.expectEqual(expected.token, actual.token);
    try std.testing.expectEqual(expected.block, actual.block);
    try std.testing.expectEqual(beat_index == 4, actual.last);
}

fn runPipeline(
    dut: *Dut,
    scratch: *Scratch,
    values: []const u32,
    gamma: []const u32,
    rows: usize,
    tokens: usize,
    eps: u32,
    options: RunOptions,
    seed: u64,
) !RunStats {
    const expected_inverses = if (options.oracle_valid)
        try computeInverses(values, rows, tokens, eps)
    else
        [_]u32{0} ** 4;
    try configureRun(dut, rows, tokens, eps);

    var prng = std.Random.DefaultPrng.init(seed);
    const random = prng.random();
    const total_words = values.len / 2;
    const reduce_requests = rows / 8 * tokens;
    var word: usize = 0;
    var presenting = false;
    var held_word: u64 = 0;
    var held_keep: u8 = 0xff;
    var held_last = false;
    var pending: ?PendingResponse = null;
    var response_active = false;
    var response_lanes = zero_lanes;
    var response_error = false;
    var request_count: usize = 0;
    var response_count: usize = 0;
    var write_count: usize = 0;
    var beat_count: usize = 0;
    var record_count: usize = 0;
    var final_output_fire_count: usize = 0;
    var held_request: ?Request = null;
    var held_write: ?ScratchWrite = null;
    var held_beat: ?NativeBeat = null;
    var expected_record: ?ExpectedRecord = null;
    var forced_holds = [_]usize{2} ** 5;
    var cycle: usize = 0;

    while (c.dut_done(dut.handle) == 0) : (cycle += 1) {
        if (cycle > rows * @max(tokens, 1) * 160 + 32_768) {
            std.debug.print(
                "pipeline timeout state={d} owner={d} busy={d} " ++
                    "reduce={d}/{d}/{d} source={d}/{d}/{d} " ++
                    "word={d} req={d} rsp={d} pending={} active={} status=0x{x}\n",
                .{
                    c.dut_debug_state(dut.handle),
                    c.dut_debug_read_owner(dut.handle),
                    c.dut_busy(dut.handle),
                    c.dut_debug_reduce_busy(dut.handle),
                    c.dut_debug_reduce_done(dut.handle),
                    c.dut_debug_reduce_error(dut.handle),
                    c.dut_debug_source_busy(dut.handle),
                    c.dut_debug_source_done(dut.handle),
                    c.dut_debug_source_error(dut.handle),
                    word,
                    request_count,
                    response_count,
                    pending != null,
                    response_active,
                    pipelineStatus(dut),
                },
            );
            return error.PipelineTimeout;
        }

        if (!presenting and word < total_words and
            (!options.stalls or random.uintLessThan(u8, 4) != 0))
        {
            presenting = true;
            held_word = residualWord(values, word);
            held_keep = if (options.partial_keep_word == word) 0x0f else 0xff;
            held_last = options.early_last_word == word or
                (options.early_last_word == null and
                    !options.omit_final_last and word + 1 == total_words);
        }
        c.dut_set_residual_stream(
            dut.handle,
            @intFromBool(presenting),
            held_word,
            held_keep,
            @intFromBool(held_last),
        );

        const write_ready = !options.stalls or random.uintLessThan(u8, 4) != 0;
        c.dut_set_r_write_sink(
            dut.handle,
            @intFromBool(write_ready),
            @intFromBool(options.write_error_word == word),
        );

        if (!response_active) {
            if (pending) |owner| {
                if (cycle >= owner.due_cycle) {
                    response_lanes = try scratch.read(
                        owner.request.token,
                        owner.request.group,
                    );
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
        const beat_index = beat_count % 5;
        if (output_valid and options.hold_first_record_beats and
            record_count == 0 and forced_holds[beat_index] != 0)
        {
            output_ready = false;
            forced_holds[beat_index] -= 1;
            c.dut_set_output_ready(dut.handle, 0);
            dut.eval();
        }

        const stream_fire = presenting and
            c.dut_residual_stream_ready(dut.handle) != 0;
        const write_fire = c.dut_r_write_valid(dut.handle) != 0 and write_ready;
        const request_valid = c.dut_read_request_valid(dut.handle) != 0;
        const request_fire = request_valid and request_ready;
        const response_fire = response_active and
            c.dut_read_response_ready(dut.handle) != 0;
        const output_fire = c.dut_output_valid(dut.handle) != 0 and output_ready;
        const final_output_fire = c.dut_debug_final_output_fire(dut.handle) != 0;

        try std.testing.expectEqual(
            output_fire,
            c.dut_debug_output_fire(dut.handle) != 0,
        );
        const expected_final_output_fire = if (output_fire) blk: {
            const beat = readBeat(dut);
            break :blk beat.last and
                @as(usize, beat.token) + 1 == tokens and
                @as(usize, beat.block) + 1 == rows / 32;
        } else false;
        try std.testing.expectEqual(
            expected_final_output_fire,
            final_output_fire,
        );
        if (final_output_fire) {
            const beat = readBeat(dut);
            try std.testing.expect(output_fire);
            try std.testing.expect(beat.last);
            try std.testing.expectEqual(tokens - 1, @as(usize, beat.token));
            try std.testing.expectEqual(rows / 32 - 1, @as(usize, beat.block));
            final_output_fire_count += 1;
        }

        try std.testing.expectEqual(stream_fire, write_fire);
        if (c.dut_r_write_valid(dut.handle) != 0) {
            const current = ScratchWrite{
                .bank = @truncate(c.dut_r_write_bank(dut.handle)),
                .address = @truncate(c.dut_r_write_address(dut.handle)),
                .data = c.dut_r_write_data(dut.handle),
            };
            if (!write_ready) {
                if (held_write) |prior|
                    try std.testing.expectEqual(prior, current);
                held_write = current;
            } else {
                held_write = null;
            }
        } else {
            held_write = null;
        }
        if (write_fire) {
            const token_word_count = rows / 2;
            const token = word / token_word_count;
            const token_word = word % token_word_count;
            try std.testing.expectEqual(
                @as(u8, @intCast(token_word % 4)),
                c.dut_r_write_bank(dut.handle),
            );
            try std.testing.expectEqual(
                @as(u16, @intCast(token * 512 + token_word / 4)),
                c.dut_r_write_address(dut.handle),
            );
            try std.testing.expectEqual(
                held_word,
                c.dut_r_write_data(dut.handle),
            );
            try scratch.write(
                c.dut_r_write_bank(dut.handle),
                c.dut_r_write_address(dut.handle),
                c.dut_r_write_data(dut.handle),
            );
            write_count += 1;
        }

        if (pending != null or response_active)
            try std.testing.expect(!request_valid);
        if (request_valid) {
            const current = readRequest(dut);
            if (!request_ready) {
                if (held_request) |prior|
                    try std.testing.expectEqual(prior, current);
                held_request = current;
            } else {
                held_request = null;
            }
        } else {
            held_request = null;
        }

        if (request_fire) {
            const request = readRequest(dut);
            const local = if (request_count < reduce_requests)
                request_count
            else
                request_count - reduce_requests;
            const groups_per_token = rows / 8;
            try std.testing.expectEqual(@as(u2, 0), request.role);
            try std.testing.expectEqual(
                @as(u3, @intCast(local / groups_per_token)),
                request.token,
            );
            try std.testing.expectEqual(
                @as(u11, @intCast(local % groups_per_token)),
                request.group,
            );
            pending = .{
                .request = request,
                .due_cycle = cycle + (options.fixed_response_delay orelse
                    2 + random.uintLessThan(usize, 6)),
                .inject_error = options.read_error_request == request_count,
            };
            request_count += 1;
        }

        if (output_valid) {
            if (expected_record == null)
                expected_record = try expectedRecord(
                    values,
                    gamma,
                    expected_inverses[0..tokens],
                    rows,
                    record_count,
                );
            try checkBeat(readBeat(dut), expected_record.?, beat_index);
            if (!output_ready) {
                if (held_beat) |prior|
                    try std.testing.expectEqual(prior, readBeat(dut));
                held_beat = readBeat(dut);
            } else {
                held_beat = null;
            }
        } else {
            held_beat = null;
        }

        if (output_fire) {
            beat_count += 1;
            if (beat_index == 4) {
                record_count += 1;
                expected_record = null;
            }
        }

        dut.step();
        if (stream_fire) {
            word += 1;
            presenting = false;
        }
        if (response_fire) {
            response_active = false;
            response_error = false;
            response_count += 1;
        }
    }

    try std.testing.expect(c.dut_busy(dut.handle) == 0);
    try std.testing.expect(c.dut_output_valid(dut.handle) == 0);
    try std.testing.expectEqual(options.expected_status != null, c.dut_error(dut.handle) != 0);
    try std.testing.expectEqual(options.expected_status orelse 0, pipelineStatus(dut));
    try std.testing.expectEqual(options.expected_writes orelse total_words, write_count);
    try std.testing.expectEqual(
        options.expected_requests orelse reduce_requests * 2,
        request_count,
    );
    try std.testing.expectEqual(request_count, response_count);
    const expected_records = options.expected_records orelse rows / 32 * tokens;
    try std.testing.expectEqual(expected_records, record_count);
    try std.testing.expectEqual(expected_records * 5, beat_count);
    try std.testing.expectEqual(
        @as(usize, @intFromBool(options.expected_status == null)),
        final_output_fire_count,
    );
    if (options.hold_first_record_beats)
        try std.testing.expectEqualSlices(usize, &([_]usize{0} ** 5), &forced_holds);
    if (options.expected_status != null)
        try std.testing.expect(c.dut_gamma_valid(dut.handle) == 0)
    else
        try std.testing.expect(c.dut_gamma_valid(dut.handle) != 0);

    dut.clearInputs();
    dut.step();
    try std.testing.expect(c.dut_done(dut.handle) == 0);
    try std.testing.expect(c.dut_output_valid(dut.handle) == 0);
    return .{
        .cycles = cycle + 1,
        .writes = write_count,
        .requests = request_count,
        .responses = response_count,
        .records = record_count,
        .beats = beat_count,
        .final_output_fires = final_output_fire_count,
    };
}

fn verifyGammaFaults(dut: *Dut, gamma: []const u32) !void {
    _ = try loadGamma(
        dut,
        gamma,
        64,
        0x1100,
        section.RmsNormGammaStatus.bad_cfg,
        null,
        null,
        null,
    );
    _ = try loadGamma(
        dut,
        gamma,
        128,
        0x1101,
        section.RmsNormGammaStatus.frame,
        2,
        null,
        null,
    );
    _ = try loadGamma(
        dut,
        gamma,
        128,
        0x1102,
        section.RmsNormGammaStatus.frame,
        null,
        3,
        null,
    );
    _ = try loadGamma(
        dut,
        gamma,
        128,
        0x1103,
        section.RmsNormGammaStatus.nonfinite,
        null,
        null,
        5,
    );
    try reloadGamma(dut, gamma, 0x1104);
}

fn verifyRunFaults(
    dut: *Dut,
    scratch: *Scratch,
    gamma: []const u32,
) !void {
    const allocator = std.heap.page_allocator;
    const values = try makeResiduals(allocator, 128, 2);
    defer allocator.free(values);

    _ = try runPipeline(dut, scratch, values[0..128], gamma, 128, 1, eps_1e_6, .{
        .early_last_word = 2,
        .expected_status = reduce_loader,
        .expected_writes = 3,
        .expected_requests = 0,
        .expected_records = 0,
        .oracle_valid = false,
    }, 0x2100);
    try reloadGamma(dut, gamma, 0x2101);

    _ = try runPipeline(dut, scratch, values[0..128], gamma, 128, 1, eps_1e_6, .{
        .partial_keep_word = 3,
        .expected_status = reduce_loader,
        .expected_writes = 4,
        .expected_requests = 0,
        .expected_records = 0,
        .oracle_valid = false,
    }, 0x2102);
    try reloadGamma(dut, gamma, 0x2103);

    _ = try runPipeline(dut, scratch, values[0..128], gamma, 128, 1, eps_1e_6, .{
        .write_error_word = 3,
        .expected_status = reduce_loader,
        .expected_writes = 4,
        .expected_requests = 0,
        .expected_records = 0,
        .oracle_valid = false,
    }, 0x2104);
    try reloadGamma(dut, gamma, 0x2105);

    _ = try runPipeline(dut, scratch, values[0..128], gamma, 128, 1, eps_1e_6, .{
        .omit_final_last = true,
        .expected_status = reduce_loader,
        .expected_requests = 0,
        .expected_records = 0,
        .oracle_valid = false,
    }, 0x2106);
    try reloadGamma(dut, gamma, 0x2107);

    _ = try runPipeline(dut, scratch, values[0..128], gamma, 128, 1, 0, .{
        .expected_status = reduceStatus(section.RmsNormReduceStatus.bad_cfg),
        .expected_writes = 0,
        .expected_requests = 0,
        .expected_records = 0,
        .oracle_valid = false,
    }, 0x2108);
    try reloadGamma(dut, gamma, 0x2109);

    _ = try runPipeline(dut, scratch, values[0..128], gamma, 128, 1, eps_1e_6, .{
        .read_error_request = 0,
        .expected_status = reduce_scratch,
        .expected_requests = 1,
        .expected_records = 0,
        .oracle_valid = false,
    }, 0x210a);
    try reloadGamma(dut, gamma, 0x210b);

    var inverse_fault = try makeOnes(allocator, 256);
    defer allocator.free(inverse_fault);
    @memset(inverse_fault[128..256], 0x7f7f_ffff);
    _ = try runPipeline(dut, scratch, inverse_fault, gamma, 128, 2, eps_1e_6, .{
        .expected_status = reduce_inverse,
        .expected_requests = 32,
        .expected_records = 0,
        .oracle_valid = false,
    }, 0x210c);
    try reloadGamma(dut, gamma, 0x210d);

    const source_fault_request = 128 / 8 + 8;
    _ = try runPipeline(dut, scratch, values[0..128], gamma, 128, 1, eps_1e_6, .{
        .read_error_request = source_fault_request,
        .expected_status = source_scratch,
        .expected_requests = source_fault_request + 1,
        .expected_records = 1,
    }, 0x210e);
    try reloadGamma(dut, gamma, 0x210f);

    const values256 = try makeResiduals(allocator, 256, 1);
    defer allocator.free(values256);
    _ = try runPipeline(dut, scratch, values256, gamma, 256, 1, eps_1e_6, .{
        .expected_status = source_gamma,
        .expected_writes = 0,
        .expected_requests = 0,
        .expected_records = 0,
    }, 0x2110);
    try reloadGamma(dut, gamma, 0x2111);

    const q8_gamma = try makeOnes(allocator, 128);
    defer allocator.free(q8_gamma);
    @memset(q8_gamma[32..64], 0x4f80_0000);
    try reloadGamma(dut, q8_gamma, 0x2112);
    _ = try runPipeline(dut, scratch, values[0..128], q8_gamma, 128, 1, eps_1e_6, .{
        .expected_status = source_q8_scale,
        .expected_requests = 25,
        .expected_records = 1,
    }, 0x2113);
    try reloadGamma(dut, gamma, 0x2114);

    _ = try runPipeline(
        dut,
        scratch,
        values[0..128],
        gamma,
        128,
        1,
        eps_1e_6,
        .{},
        0x2115,
    );
}

const AbortTarget = enum {
    residual_load,
    reduction_read,
    inverse,
    source_read,
    source_mul,
    q8,
    output,
};

fn targetReached(
    dut: *Dut,
    target: AbortTarget,
    words: usize,
    pending: bool,
) bool {
    return switch (target) {
        .residual_load => words >= 3,
        .reduction_read => pending and c.dut_debug_read_owner(dut.handle) == 1,
        .inverse => c.dut_debug_reduce_inverse_state(dut.handle) != 0,
        .source_read => pending and c.dut_debug_read_owner(dut.handle) == 2,
        .source_mul => c.dut_debug_weighted_state(dut.handle) == 6 or
            c.dut_debug_weighted_state(dut.handle) == 8,
        .q8 => c.dut_debug_q8_quantizer_state(dut.handle) != 0,
        .output => c.dut_output_valid(dut.handle) != 0,
    };
}

fn abortPipelinePhase(
    dut: *Dut,
    scratch: *Scratch,
    values: []const u32,
    gamma: []const u32,
    target: AbortTarget,
) !void {
    try configureRun(dut, 128, 1, eps_1e_6);
    var word: usize = 0;
    var pending: ?Request = null;
    var response_active = false;
    var response_lanes = zero_lanes;
    var cycles: usize = 0;

    while (true) : (cycles += 1) {
        if (cycles > 32_768) {
            std.debug.print(
                "abort target timeout {s}: parent={d} owner={d} " ++
                    "reduce={d}/{d} frontend={d} inv={d} " ++
                    "source={d}/{d} weighted={d} q8={d}/{d} " ++
                    "word={d} pending={} response={} output={d}\n",
                .{
                    @tagName(target),
                    c.dut_debug_state(dut.handle),
                    c.dut_debug_read_owner(dut.handle),
                    c.dut_debug_reduce_state(dut.handle),
                    c.dut_debug_reduce_busy(dut.handle),
                    c.dut_debug_reduce_frontend_state(dut.handle),
                    c.dut_debug_reduce_inverse_state(dut.handle),
                    c.dut_debug_source_state(dut.handle),
                    c.dut_debug_source_busy(dut.handle),
                    c.dut_debug_weighted_state(dut.handle),
                    c.dut_debug_q8_state(dut.handle),
                    c.dut_debug_q8_quantizer_state(dut.handle),
                    word,
                    pending != null,
                    response_active,
                    c.dut_output_valid(dut.handle),
                },
            );
            return error.AbortTargetTimeout;
        }
        if (targetReached(dut, target, word, pending != null)) break;
        const stream_valid = word < values.len / 2;
        c.dut_set_residual_stream(
            dut.handle,
            @intFromBool(stream_valid),
            if (stream_valid) residualWord(values, word) else 0,
            0xff,
            @intFromBool(stream_valid and word + 1 == values.len / 2),
        );
        c.dut_set_r_write_sink(dut.handle, 1, 0);

        if (pending) |request| {
            if (!response_active) {
                response_lanes = try scratch.read(request.token, request.group);
                response_active = true;
                pending = null;
            }
        }
        c.dut_set_read_response(
            dut.handle,
            @intFromBool(response_active),
            &response_lanes,
            0,
        );
        const request_ready = pending == null and !response_active;
        c.dut_set_read_request_ready(dut.handle, @intFromBool(request_ready));
        c.dut_set_output_ready(dut.handle, @intFromBool(target != .output));
        dut.eval();

        const stream_fire = stream_valid and
            c.dut_residual_stream_ready(dut.handle) != 0;
        const write_fire = c.dut_r_write_valid(dut.handle) != 0;
        try std.testing.expectEqual(stream_fire, write_fire);
        if (write_fire) try scratch.write(
            c.dut_r_write_bank(dut.handle),
            c.dut_r_write_address(dut.handle),
            c.dut_r_write_data(dut.handle),
        );
        if (c.dut_read_request_valid(dut.handle) != 0 and request_ready)
            pending = readRequest(dut);
        const response_fire = response_active and
            c.dut_read_response_ready(dut.handle) != 0;

        if (targetReached(dut, target, word, pending != null)) break;
        dut.step();
        if (stream_fire) word += 1;
        if (response_fire) response_active = false;
    }

    c.dut_set_residual_stream(dut.handle, 0, 0, 0, 0);
    c.dut_set_r_write_sink(dut.handle, 0, 0);
    c.dut_set_read_request_ready(dut.handle, 0);
    c.dut_set_output_ready(dut.handle, 0);
    c.dut_set_abort(dut.handle, 1);
    dut.eval();
    try std.testing.expect(c.dut_output_valid(dut.handle) == 0);
    try std.testing.expect(c.dut_done(dut.handle) == 0);
    try std.testing.expect(c.dut_error(dut.handle) == 0);
    try std.testing.expectEqual(@as(u30, 0), pipelineStatus(dut));
    dut.step();
    c.dut_set_abort(dut.handle, 0);

    if (pending) |request| {
        response_lanes = try scratch.read(request.token, request.group);
        response_active = true;
        pending = null;
    }
    var cleanup: usize = 0;
    while (c.dut_busy(dut.handle) != 0 or c.dut_gamma_busy(dut.handle) != 0) : (cleanup += 1) {
        if (cleanup > 512) return error.AbortCleanupTimeout;
        c.dut_set_read_response(
            dut.handle,
            @intFromBool(response_active),
            &response_lanes,
            0,
        );
        dut.eval();
        try std.testing.expect(c.dut_output_valid(dut.handle) == 0);
        try std.testing.expect(c.dut_done(dut.handle) == 0);
        try std.testing.expect(c.dut_error(dut.handle) == 0);
        try std.testing.expectEqual(@as(u30, 0), pipelineStatus(dut));
        const drain_fire = response_active and
            c.dut_read_response_ready(dut.handle) != 0;
        dut.step();
        if (drain_fire) response_active = false;
    }
    try std.testing.expect(!response_active);
    try std.testing.expect(c.dut_gamma_valid(dut.handle) == 0);
    try std.testing.expect(c.dut_done(dut.handle) == 0);
    try std.testing.expect(c.dut_error(dut.handle) == 0);
    try std.testing.expectEqual(@as(u30, 0), pipelineStatus(dut));
    dut.clearInputs();

    try reloadGamma(
        dut,
        gamma,
        0x3000 + @as(u64, @intFromEnum(target)),
    );
    _ = try runPipeline(
        dut,
        scratch,
        values,
        gamma,
        128,
        1,
        eps_1e_6,
        .{},
        0x3100 + @as(u64, @intFromEnum(target)),
    );
}

fn abortGammaLoad(dut: *Dut, gamma: []const u32) !void {
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
        dut.step();
    }
    c.dut_set_gamma_stream(dut.handle, 0, 0, 0, 0);
    c.dut_set_abort(dut.handle, 1);
    dut.eval();
    try std.testing.expect(c.dut_gamma_done(dut.handle) == 0);
    try std.testing.expect(c.dut_gamma_error(dut.handle) == 0);
    dut.step();
    c.dut_set_abort(dut.handle, 0);
    dut.eval();
    for (0..32) |_| {
        if (c.dut_gamma_busy(dut.handle) == 0 and
            c.dut_gamma_config_ready(dut.handle) != 0) break;
        dut.step();
    }
    try std.testing.expect(c.dut_gamma_busy(dut.handle) == 0);
    try std.testing.expect(c.dut_gamma_config_ready(dut.handle) != 0);
    try std.testing.expect(c.dut_gamma_valid(dut.handle) == 0);
    try std.testing.expect(c.dut_gamma_done(dut.handle) == 0);
    try std.testing.expect(c.dut_gamma_error(dut.handle) == 0);
    dut.clearInputs();
}

fn verifyAborts(
    dut: *Dut,
    scratch: *Scratch,
    gamma: []const u32,
    values: []const u32,
) !void {
    try abortGammaLoad(dut, gamma);
    try reloadGamma(dut, gamma, 0x3200);
    const targets = [_]AbortTarget{
        .residual_load,
        .reduction_read,
        .inverse,
        .source_read,
        .source_mul,
        .q8,
        .output,
    };
    for (targets) |target|
        try abortPipelinePhase(dut, scratch, values, gamma, target);
}

pub fn main() !void {
    var dut = Dut.init();
    defer dut.deinit();
    dut.reset();
    var scratch: Scratch = .{};

    const allocator = std.heap.page_allocator;
    const gamma128 = try makeGamma(allocator, 128);
    defer allocator.free(gamma128);
    const values128 = try makeResiduals(allocator, 128, 1);
    defer allocator.free(values128);

    try verifyRejectedStarts(&dut, gamma128, residualWord(values128, 0));
    try verifyGammaFaults(&dut, gamma128);
    try verifyRunFaults(&dut, &scratch, gamma128);
    try verifyAborts(&dut, &scratch, gamma128, values128);

    var legal_shapes: usize = 0;
    var total_cycles: usize = 0;
    var total_writes: usize = 0;
    var total_requests: usize = 0;
    var total_records: usize = 0;
    var four_token_final_fires: usize = 0;
    for (rows_set, 0..) |rows, row_index| {
        const gamma = try makeGamma(allocator, rows);
        defer allocator.free(gamma);
        try reloadGamma(&dut, gamma, 0x4000 + row_index);
        for (1..5) |tokens| {
            const values = try makeResiduals(allocator, rows, tokens);
            defer allocator.free(values);
            const stats = try runPipeline(
                &dut,
                &scratch,
                values,
                gamma,
                rows,
                tokens,
                eps_1e_6,
                .{ .hold_first_record_beats = rows == 128 and tokens == 1 },
                0x5000 + row_index * 8 + tokens,
            );
            legal_shapes += 1;
            total_cycles += stats.cycles;
            total_writes += stats.writes;
            total_requests += stats.requests;
            total_records += stats.records;
            if (tokens == 4)
                four_token_final_fires += stats.final_output_fires;
        }
    }
    try std.testing.expectEqual(rows_set.len, four_token_final_fires);

    std.debug.print(
        "section_rmsnorm_q8_pipeline cosim:\n" ++
            "  legal shapes={d}, cycles={d}, R writes={d}, shared reads={d}\n" ++
            "  exact native records={d}, four-token final hooks={d}, " ++
            "gamma/residual framing, reduction/source " ++
            "faults, prefix quarantine, abort phases=8, no-reset restart: passed\n\n",
        .{
            legal_shapes,
            total_cycles,
            total_writes,
            total_requests,
            total_records,
            four_token_final_fires,
        },
    );
}
