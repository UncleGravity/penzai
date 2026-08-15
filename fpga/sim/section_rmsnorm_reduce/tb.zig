//! End-to-end cosim for `section_rmsnorm_reduce`.
//!
//! Raw token-major F32 words pass through the direct R scratch load, replay,
//! fixed reduction, and inverse-RMS leaves. A cycle-accurate scratch model
//! checks ownership and retained-response cleanup while the software oracle
//! checks every scalar bit.

const std = @import("std");
const section = @import("shared_section");
const c = @cImport(@cInclude("shim.h"));

const eps_1e_6: u32 = 0x3586_37bd;
const zero_lanes = [_]u64{ 0, 0, 0, 0 };

const Result = struct {
    token: u2,
    inv_rms: u32,
    final: bool,
};

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

const PendingRead = struct {
    token: u8,
    group: u16,
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

    fn reset(self: *Dut) void {
        c.dut_set_clk(self.handle, 0);
        c.dut_set_rst_n(self.handle, 0);
        c.dut_set_config(self.handle, 0, 0, 0, 0);
        c.dut_set_abort(self.handle, 0);
        c.dut_set_stream(self.handle, 0, 0, 0, 0);
        c.dut_set_r_write_sink(self.handle, 0, 0);
        c.dut_set_read_request_ready(self.handle, 0);
        c.dut_set_read_response(self.handle, 0, &zero_lanes, 0);
        c.dut_set_result_ready(self.handle, 0);
        self.eval();
        for (0..5) |_| self.step();
        c.dut_set_rst_n(self.handle, 1);
        self.step();
    }
};

const RunOptions = struct {
    stalls: bool = true,
    hold_first_result_cycles: usize = 0,
    early_last_word: ?usize = null,
    partial_keep_word: ?usize = null,
    omit_final_last: bool = false,
    late_final_last: bool = false,
    write_error_word: ?usize = null,
    read_error_request: ?usize = null,
    expected_status: ?u13 = null,
};

const RunStats = struct {
    cycles: usize,
    writes: usize,
    reads: usize,
    results: usize,
};

fn wordAt(values: []const u32, word: usize) u64 {
    return @as(u64, values[word * 2]) |
        (@as(u64, values[word * 2 + 1]) << 32);
}

fn readResult(dut: *Dut) Result {
    return .{
        .token = @truncate(c.dut_result_token(dut.handle)),
        .inv_rms = c.dut_result_inv_rms(dut.handle),
        .final = c.dut_result_final(dut.handle) != 0,
    };
}

fn readStatus(dut: *Dut) u13 {
    return @truncate(c.dut_status(dut.handle));
}

fn clearInputs(dut: *Dut) void {
    c.dut_set_stream(dut.handle, 0, 0, 0, 0);
    c.dut_set_r_write_sink(dut.handle, 0, 0);
    c.dut_set_read_request_ready(dut.handle, 0);
    c.dut_set_read_response(dut.handle, 0, &zero_lanes, 0);
    c.dut_set_result_ready(dut.handle, 0);
    dut.eval();
}

fn configure(dut: *Dut, rows: u32, tokens: u3, eps: u32) !void {
    try std.testing.expect(c.dut_config_ready(dut.handle) != 0);
    c.dut_set_config(dut.handle, 1, @intCast(rows), tokens, eps);
    dut.eval();
    try std.testing.expect(c.dut_config_ready(dut.handle) != 0);
    dut.step();
    c.dut_set_config(dut.handle, 0, 0, 0, 0);
    dut.eval();
}

fn run(
    dut: *Dut,
    scratch: *Scratch,
    values: []const u32,
    rows: u32,
    tokens: u3,
    eps: u32,
    options: RunOptions,
    seed: u64,
) !RunStats {
    const token_count: usize = tokens;
    try std.testing.expectEqual(@as(usize, rows) * token_count, values.len);

    var expected: [4]u32 = undefined;
    if (options.expected_status == null) {
        for (0..token_count) |token| {
            const token_values = values[token * rows ..][0..rows];
            const scan = try section.rmsNormMaxExp(token_values, rows);
            try std.testing.expect(!scan.subnormal_warning);
            const sum = try section.rmsNormSumsqFixed(
                token_values,
                rows,
                scan.max_exp,
            );
            expected[token] = (try section.rmsNormInvFixed(
                sum.sum_sq,
                rows,
                scan.max_exp,
                eps,
            )).inv_rms_bits;
        }
    }

    try configure(dut, rows, tokens, eps);
    try std.testing.expect(c.dut_busy(dut.handle) != 0);

    var prng = std.Random.DefaultPrng.init(seed);
    const rnd = prng.random();
    const total_words = values.len / 2;
    const total_reads = values.len / 8;
    var word_index: usize = 0;
    var presenting = false;
    var pending: ?PendingRead = null;
    var response_active = false;
    var response_lanes = zero_lanes;
    var response_error = false;
    var request_count: usize = 0;
    var write_count: usize = 0;
    var result_count: usize = 0;
    var held_result: ?Result = null;
    var held_result_cycles: usize = 0;
    var release_results = options.hold_first_result_cycles == 0;
    var cycle: usize = 0;

    while (true) : (cycle += 1) {
        if (cycle > values.len * 20 + 8192) return error.StreamTimeout;

        const offered_words = total_words + @intFromBool(options.late_final_last);
        if (!presenting and word_index < offered_words)
            presenting = !options.stalls or rnd.uintLessThan(u8, 4) != 0;
        const stream_data = if (presenting and word_index < total_words)
            wordAt(values, word_index)
        else
            0;
        const natural_last = presenting and !options.omit_final_last and
            !options.late_final_last and word_index + 1 == total_words;
        const delayed_last = presenting and options.late_final_last and
            word_index == total_words;
        const stream_last = natural_last or
            delayed_last or
            (presenting and options.early_last_word == word_index);
        const stream_keep: u8 = if (presenting and
            options.partial_keep_word == word_index) 0x0f else 0xff;
        c.dut_set_stream(
            dut.handle,
            @intFromBool(presenting),
            stream_data,
            stream_keep,
            @intFromBool(stream_last),
        );

        const write_ready = !options.stalls or rnd.uintLessThan(u8, 4) != 0;
        const inject_write_error = options.write_error_word == word_index;
        c.dut_set_r_write_sink(
            dut.handle,
            @intFromBool(write_ready),
            @intFromBool(inject_write_error),
        );

        if (!response_active) {
            if (pending) |read| {
                if (cycle >= read.due_cycle) {
                    response_lanes = try scratch.read(read.token, read.group);
                    response_error = read.inject_error;
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
            (!options.stalls or rnd.uintLessThan(u8, 4) != 0);
        c.dut_set_read_request_ready(dut.handle, @intFromBool(request_ready));
        const random_result_ready = !options.stalls or
            rnd.uintLessThan(u8, 3) != 0;
        const result_ready = release_results and random_result_ready;
        c.dut_set_result_ready(dut.handle, @intFromBool(result_ready));
        dut.eval();

        const stream_fire = presenting and c.dut_stream_ready(dut.handle) != 0;
        const write_valid = c.dut_r_write_valid(dut.handle) != 0;
        const write_fire = write_valid and write_ready;
        try std.testing.expectEqual(stream_fire, write_fire);
        if (write_valid) {
            const expected_bank: u8 = @intCast(word_index % 4);
            const token = word_index / (rows / 2);
            const token_word = word_index % (rows / 2);
            const expected_address: u16 = @intCast(token * 512 + token_word / 4);
            try std.testing.expectEqual(expected_bank, c.dut_r_write_bank(dut.handle));
            try std.testing.expectEqual(expected_address, c.dut_r_write_address(dut.handle));
            try std.testing.expectEqual(stream_data, c.dut_r_write_data(dut.handle));
        }

        const request_valid = c.dut_read_request_valid(dut.handle) != 0;
        const request_fire = request_valid and request_ready;
        const response_fire = response_active and
            c.dut_read_response_ready(dut.handle) != 0;
        const result_valid = c.dut_result_valid(dut.handle) != 0;
        const result_fire = result_valid and result_ready;

        if (options.expected_status != null)
            try std.testing.expect(!result_valid);
        if (result_valid) {
            // Publication is atomic: no scalar appears until replay is retired.
            try std.testing.expectEqual(total_words, word_index);
            try std.testing.expectEqual(total_reads, request_count);
            try std.testing.expect(pending == null and !response_active);
        }
        if (result_valid and !release_results) {
            held_result_cycles += 1;
            if (held_result_cycles >= options.hold_first_result_cycles)
                release_results = true;
        }
        if (result_valid and !result_ready) {
            const current = readResult(dut);
            if (held_result) |prior| try std.testing.expectEqual(prior, current);
            held_result = current;
            try std.testing.expect(c.dut_done(dut.handle) == 0);
            try std.testing.expect(c.dut_busy(dut.handle) != 0);
        } else {
            held_result = null;
        }

        if (write_fire and !inject_write_error) {
            try scratch.write(
                c.dut_r_write_bank(dut.handle),
                c.dut_r_write_address(dut.handle),
                c.dut_r_write_data(dut.handle),
            );
            write_count += 1;
        }
        if (request_fire) {
            const token = c.dut_read_request_token(dut.handle);
            const group = c.dut_read_request_group(dut.handle);
            try std.testing.expectEqual(
                @as(u8, @intCast(request_count / (rows / 8))),
                token,
            );
            try std.testing.expectEqual(
                @as(u16, @intCast(request_count % (rows / 8))),
                group,
            );
            pending = .{
                .token = token,
                .group = group,
                .due_cycle = cycle + 2 + rnd.uintLessThan(usize, 5),
                .inject_error = options.read_error_request == request_count,
            };
            request_count += 1;
        }
        if (result_fire) {
            const got = readResult(dut);
            try std.testing.expect(options.expected_status == null);
            try std.testing.expectEqual(@as(u2, @intCast(result_count)), got.token);
            try std.testing.expectEqual(expected[result_count], got.inv_rms);
            try std.testing.expectEqual(result_count + 1 == token_count, got.final);
            result_count += 1;
        }

        dut.step();
        if (stream_fire) {
            word_index += 1;
            presenting = false;
        }
        if (response_fire) {
            response_active = false;
            response_error = false;
        }

        if (c.dut_done(dut.handle) != 0 and c.dut_busy(dut.handle) == 0) {
            if (options.expected_status) |expected_status| {
                try std.testing.expect(c.dut_error(dut.handle) != 0);
                try std.testing.expectEqual(expected_status, readStatus(dut));
                try std.testing.expectEqual(@as(usize, 0), result_count);
            } else {
                if (c.dut_error(dut.handle) != 0 or readStatus(dut) != 0) {
                    std.debug.print(
                        "reduce unexpected terminal error: status=0x{x}, " ++
                            "cycles={d}, words={d}/{d}, reads={d}/{d}, results={d}\n",
                        .{
                            readStatus(dut), cycle + 1,   word_index,   total_words,
                            request_count,   total_reads, result_count,
                        },
                    );
                }
                try std.testing.expect(c.dut_error(dut.handle) == 0);
                try std.testing.expectEqual(@as(u13, 0), readStatus(dut));
                try std.testing.expectEqual(total_words, write_count);
                try std.testing.expectEqual(total_reads, request_count);
                try std.testing.expectEqual(token_count, result_count);
            }
            break;
        }
    }

    clearInputs(dut);
    return .{
        .cycles = cycle + 1,
        .writes = write_count,
        .reads = request_count,
        .results = result_count,
    };
}

fn makeValues(allocator: std.mem.Allocator, rows: usize, tokens: usize) ![]u32 {
    const values = try allocator.alloc(u32, rows * tokens);
    for (0..tokens) |token| {
        for (0..rows) |row| {
            const exponent: u8 = @intCast(90 + ((row * 13 + token * 7) % 32));
            const mantissa: u23 = @truncate(row *% 0x12d687 +% token *% 0x34567);
            const sign: u32 = @intFromBool((row + token) % 3 == 0);
            values[token * rows + row] = (sign << 31) |
                (@as(u32, exponent) << 23) | mantissa;
        }
    }
    return values;
}

fn expectConfigFault(
    dut: *Dut,
    rows: u32,
    tokens: u3,
    eps: u32,
) !void {
    try configure(dut, rows, tokens, eps);
    try std.testing.expect(c.dut_done(dut.handle) != 0);
    try std.testing.expect(c.dut_error(dut.handle) != 0);
    try std.testing.expect(c.dut_busy(dut.handle) == 0);
    try std.testing.expect(c.dut_result_valid(dut.handle) == 0);
    try std.testing.expectEqual(section.RmsNormReduceStatus.bad_cfg, readStatus(dut));
    dut.step();
}

fn verifyConfigFaults(dut: *Dut) !void {
    try expectConfigFault(dut, 7, 1, eps_1e_6);
    try expectConfigFault(dut, 24, 1, eps_1e_6);
    try expectConfigFault(dut, 4096, 0, eps_1e_6);
    try expectConfigFault(dut, 4096, 5, eps_1e_6);
    try expectConfigFault(dut, 4096, 1, 0);
    try expectConfigFault(dut, 4096, 1, 0x0000_0001);
    try expectConfigFault(dut, 4096, 1, 0xbf80_0000);
    try expectConfigFault(dut, 4096, 1, 0x7f80_0000);
}

fn testAbortOutstanding(
    dut: *Dut,
    scratch: *Scratch,
    values: []const u32,
) !void {
    try configure(dut, 8, 1, eps_1e_6);
    var word: usize = 0;
    var pending: ?PendingRead = null;
    var cycle: usize = 0;
    while (pending == null) : (cycle += 1) {
        if (cycle > 128) return error.StreamTimeout;
        const valid = word < 4;
        c.dut_set_stream(
            dut.handle,
            @intFromBool(valid),
            if (valid) wordAt(values, word) else 0,
            0xff,
            @intFromBool(valid and word == 3),
        );
        c.dut_set_r_write_sink(dut.handle, 1, 0);
        c.dut_set_read_request_ready(dut.handle, 1);
        c.dut_set_read_response(dut.handle, 0, &zero_lanes, 0);
        c.dut_set_result_ready(dut.handle, 1);
        dut.eval();
        const stream_fire = valid and c.dut_stream_ready(dut.handle) != 0;
        if (stream_fire) {
            try scratch.write(
                c.dut_r_write_bank(dut.handle),
                c.dut_r_write_address(dut.handle),
                c.dut_r_write_data(dut.handle),
            );
        }
        if (c.dut_read_request_valid(dut.handle) != 0) {
            pending = .{
                .token = c.dut_read_request_token(dut.handle),
                .group = c.dut_read_request_group(dut.handle),
                .due_cycle = cycle + 8,
                .inject_error = false,
            };
        }
        dut.step();
        if (stream_fire) word += 1;
    }

    c.dut_set_stream(dut.handle, 0, 0, 0, 0);
    c.dut_set_abort(dut.handle, 1);
    c.dut_set_read_request_ready(dut.handle, 0);
    dut.eval();
    try std.testing.expect(c.dut_busy(dut.handle) != 0);
    dut.step();
    c.dut_set_abort(dut.handle, 0);
    dut.eval();
    try std.testing.expect(c.dut_busy(dut.handle) != 0);
    try std.testing.expect(c.dut_done(dut.handle) == 0);
    try std.testing.expect(c.dut_result_valid(dut.handle) == 0);

    const read = pending.?;
    var response = try scratch.read(read.token, read.group);
    while (cycle < read.due_cycle) : (cycle += 1) {
        c.dut_set_read_response(dut.handle, 0, &zero_lanes, 0);
        dut.step();
        try std.testing.expect(c.dut_busy(dut.handle) != 0);
    }
    c.dut_set_read_response(dut.handle, 1, &response, 0);
    dut.eval();
    try std.testing.expect(c.dut_read_response_ready(dut.handle) != 0);
    dut.step();
    c.dut_set_read_response(dut.handle, 0, &zero_lanes, 0);
    for (0..4) |_| {
        dut.eval();
        if (c.dut_busy(dut.handle) == 0) break;
        dut.step();
    }
    try std.testing.expect(c.dut_busy(dut.handle) == 0);
    try std.testing.expect(c.dut_done(dut.handle) == 0);
    try std.testing.expect(c.dut_error(dut.handle) == 0);
    try std.testing.expectEqual(@as(u13, 0), readStatus(dut));
    try std.testing.expect(c.dut_result_valid(dut.handle) == 0);
    clearInputs(dut);
}

fn streamConfiguredValues(
    dut: *Dut,
    scratch: *Scratch,
    values: []const u32,
    rows: u32,
) !void {
    const total_words = values.len / 2;
    var word: usize = 0;
    var cycles: usize = 0;
    while (word < total_words) : (cycles += 1) {
        if (cycles > total_words * 16 + 128) return error.StreamTimeout;
        const data = wordAt(values, word);
        c.dut_set_stream(
            dut.handle,
            1,
            data,
            0xff,
            @intFromBool(word + 1 == total_words),
        );
        c.dut_set_r_write_sink(dut.handle, 1, 0);
        c.dut_set_read_request_ready(dut.handle, 0);
        c.dut_set_read_response(dut.handle, 0, &zero_lanes, 0);
        c.dut_set_result_ready(dut.handle, 0);
        dut.eval();

        const fire = c.dut_stream_ready(dut.handle) != 0;
        try std.testing.expectEqual(fire, c.dut_r_write_valid(dut.handle) != 0);
        if (fire) {
            const expected_bank: u8 = @intCast(word % 4);
            const token = word / (rows / 2);
            const token_word = word % (rows / 2);
            const expected_address: u16 = @intCast(token * 512 + token_word / 4);
            try std.testing.expectEqual(expected_bank, c.dut_r_write_bank(dut.handle));
            try std.testing.expectEqual(expected_address, c.dut_r_write_address(dut.handle));
            try std.testing.expectEqual(data, c.dut_r_write_data(dut.handle));
            try scratch.write(
                c.dut_r_write_bank(dut.handle),
                c.dut_r_write_address(dut.handle),
                c.dut_r_write_data(dut.handle),
            );
        }
        dut.step();
        try std.testing.expect(c.dut_busy(dut.handle) != 0);
        try std.testing.expect(c.dut_done(dut.handle) == 0);
        try std.testing.expect(c.dut_error(dut.handle) == 0);
        try std.testing.expectEqual(@as(u13, 0), readStatus(dut));
        try std.testing.expect(c.dut_result_valid(dut.handle) == 0);
        if (fire) word += 1;
    }
    c.dut_set_stream(dut.handle, 0, 0, 0, 0);
    c.dut_set_r_write_sink(dut.handle, 0, 0);
    dut.eval();
}

fn serviceRead(
    dut: *Dut,
    scratch: *Scratch,
    expected_token: u8,
    expected_group: u16,
) !void {
    c.dut_set_read_request_ready(dut.handle, 0);
    c.dut_set_read_response(dut.handle, 0, &zero_lanes, 0);
    c.dut_set_result_ready(dut.handle, 0);
    dut.eval();
    var cycles: usize = 0;
    while (c.dut_read_request_valid(dut.handle) == 0) : (cycles += 1) {
        if (cycles > 256) return error.ReadRequestTimeout;
        try std.testing.expect(c.dut_busy(dut.handle) != 0);
        try std.testing.expect(c.dut_done(dut.handle) == 0);
        try std.testing.expect(c.dut_error(dut.handle) == 0);
        try std.testing.expectEqual(@as(u13, 0), readStatus(dut));
        try std.testing.expect(c.dut_result_valid(dut.handle) == 0);
        dut.step();
    }
    try std.testing.expectEqual(expected_token, c.dut_read_request_token(dut.handle));
    try std.testing.expectEqual(expected_group, c.dut_read_request_group(dut.handle));
    c.dut_set_read_request_ready(dut.handle, 1);
    dut.eval();
    try std.testing.expect(c.dut_read_request_valid(dut.handle) != 0);
    dut.step();
    c.dut_set_read_request_ready(dut.handle, 0);

    var response = try scratch.read(expected_token, expected_group);
    c.dut_set_read_response(dut.handle, 1, &response, 0);
    dut.eval();
    cycles = 0;
    while (c.dut_read_response_ready(dut.handle) == 0) : (cycles += 1) {
        if (cycles > 32) return error.ReadResponseTimeout;
        dut.step();
    }
    dut.step();
    c.dut_set_read_response(dut.handle, 0, &zero_lanes, 0);
    dut.eval();
}

fn abortToIdle(dut: *Dut) !void {
    c.dut_set_abort(dut.handle, 1);
    dut.eval();
    try std.testing.expect(c.dut_busy(dut.handle) != 0);
    try std.testing.expect(c.dut_result_valid(dut.handle) == 0);
    dut.step();
    c.dut_set_abort(dut.handle, 0);
    dut.eval();

    var cycles: usize = 0;
    while (c.dut_busy(dut.handle) != 0) : (cycles += 1) {
        if (cycles > 8) return error.AbortCleanupTimeout;
        try std.testing.expect(c.dut_done(dut.handle) == 0);
        try std.testing.expect(c.dut_error(dut.handle) == 0);
        try std.testing.expectEqual(@as(u13, 0), readStatus(dut));
        try std.testing.expect(c.dut_result_valid(dut.handle) == 0);
        dut.step();
    }
    try std.testing.expect(c.dut_done(dut.handle) == 0);
    try std.testing.expect(c.dut_error(dut.handle) == 0);
    try std.testing.expectEqual(@as(u13, 0), readStatus(dut));
    try std.testing.expect(c.dut_result_valid(dut.handle) == 0);
    try std.testing.expect(c.dut_config_ready(dut.handle) != 0);
    clearInputs(dut);
}

fn testAbortInverseActive(
    dut: *Dut,
    scratch: *Scratch,
    values: []const u32,
) !void {
    try configure(dut, 8, 2, eps_1e_6);
    try streamConfiguredValues(dut, scratch, values, 8);
    try serviceRead(dut, scratch, 0, 0);

    // A second-token request can only appear after token zero's reduction
    // record has handshaken into the serialized inverse leaf. Keep it
    // unaccepted so this abort has no retained scratch owner to drain.
    var cycles: usize = 0;
    while (c.dut_read_request_valid(dut.handle) == 0) : (cycles += 1) {
        if (cycles > 256) return error.BridgeTimeout;
        try std.testing.expect(c.dut_result_valid(dut.handle) == 0);
        dut.step();
    }
    try std.testing.expectEqual(@as(u8, 1), c.dut_read_request_token(dut.handle));
    try std.testing.expectEqual(@as(u16, 0), c.dut_read_request_group(dut.handle));
    try std.testing.expect(c.dut_read_response_ready(dut.handle) == 0);
    try abortToIdle(dut);
}

fn testAbortStalledResult(
    dut: *Dut,
    scratch: *Scratch,
    values: []const u32,
) !void {
    try configure(dut, 8, 2, eps_1e_6);
    try streamConfiguredValues(dut, scratch, values, 8);
    try serviceRead(dut, scratch, 0, 0);
    try serviceRead(dut, scratch, 1, 0);

    c.dut_set_result_ready(dut.handle, 0);
    dut.eval();
    var cycles: usize = 0;
    while (c.dut_result_valid(dut.handle) == 0) : (cycles += 1) {
        if (cycles > 256) return error.ResultTimeout;
        try std.testing.expect(c.dut_busy(dut.handle) != 0);
        try std.testing.expect(c.dut_done(dut.handle) == 0);
        try std.testing.expect(c.dut_error(dut.handle) == 0);
        try std.testing.expectEqual(@as(u13, 0), readStatus(dut));
        dut.step();
    }
    const held = readResult(dut);
    try std.testing.expectEqual(@as(u2, 0), held.token);
    try std.testing.expect(!held.final);
    for (0..3) |_| {
        dut.step();
        try std.testing.expect(c.dut_result_valid(dut.handle) != 0);
        try std.testing.expectEqual(held, readResult(dut));
        try std.testing.expect(c.dut_done(dut.handle) == 0);
        try std.testing.expect(c.dut_busy(dut.handle) != 0);
    }
    try abortToIdle(dut);
}

pub fn main() !void {
    var dut = Dut.init();
    defer dut.deinit();
    dut.reset();
    var scratch: Scratch = .{};

    try verifyConfigFaults(&dut);

    const allocator = std.heap.page_allocator;
    const compact = try makeValues(allocator, 128, 4);
    defer allocator.free(compact);
    // Exercise the exact zero reduction and epsilon-only inverse path.
    for (compact[3 * 128 .. 4 * 128], 0..) |*value, row|
        value.* = @as(u32, @intFromBool((row & 1) != 0)) << 31;
    const compact_stats = try run(
        &dut,
        &scratch,
        compact,
        128,
        4,
        eps_1e_6,
        .{ .hold_first_result_cycles = 5 },
        0x6171_8e22,
    );

    const full = try makeValues(allocator, 4096, 4);
    defer allocator.free(full);
    const full_stats = try run(
        &dut,
        &scratch,
        full,
        4096,
        4,
        eps_1e_6,
        .{},
        0x8f21_b453,
    );

    var fault_values = try makeValues(allocator, 8, 2);
    defer allocator.free(fault_values);
    const front_loader = section.RmsNormReduceStatus.frontend(
        section.RmsNormFrontendStatus.loader,
    );
    const front_maxexp = section.RmsNormReduceStatus.frontend(
        section.RmsNormFrontendStatus.max_exp,
    );
    const front_subnormal = section.RmsNormReduceStatus.frontend(
        section.RmsNormFrontendStatus.subnormal,
    );
    const front_scratch = section.RmsNormReduceStatus.frontend(
        section.RmsNormFrontendStatus.scratch,
    );
    _ = try run(&dut, &scratch, fault_values, 8, 2, eps_1e_6, .{
        .early_last_word = 0,
        .expected_status = front_loader,
    }, 0x1001);
    _ = try run(&dut, &scratch, fault_values, 8, 2, eps_1e_6, .{
        .partial_keep_word = 1,
        .expected_status = front_loader,
    }, 0x1002);
    _ = try run(&dut, &scratch, fault_values, 8, 2, eps_1e_6, .{
        .omit_final_last = true,
        .expected_status = front_loader,
    }, 0x1003);
    _ = try run(&dut, &scratch, fault_values, 8, 2, eps_1e_6, .{
        .late_final_last = true,
        .expected_status = front_loader,
    }, 0x1004);
    _ = try run(&dut, &scratch, fault_values, 8, 2, eps_1e_6, .{
        .write_error_word = 1,
        .expected_status = front_loader,
    }, 0x1005);

    fault_values[3] = 0x7f80_0000;
    _ = try run(&dut, &scratch, fault_values, 8, 2, eps_1e_6, .{
        .expected_status = front_maxexp,
    }, 0x1006);
    fault_values[3] = 0x0000_0001;
    _ = try run(&dut, &scratch, fault_values, 8, 2, eps_1e_6, .{
        .expected_status = front_subnormal,
    }, 0x1007);
    fault_values[3] = 0x3f80_0000;
    _ = try run(&dut, &scratch, fault_values, 8, 2, eps_1e_6, .{
        .read_error_request = 0,
        .expected_status = front_scratch,
    }, 0x1008);

    // Token zero completes numerically, but token one's mean overflows. The
    // buffered prefix must never become visible at the composed boundary.
    for (fault_values[0..8]) |*value| value.* = 0x3f80_0000;
    for (fault_values[8..16]) |*value| value.* = 0x7f7f_ffff;
    const inverse_arithmetic = section.RmsNormReduceStatus.inverse(
        section.RmsNormInvStatus.arithmetic,
    );
    const prefix_fault = try run(
        &dut,
        &scratch,
        fault_values,
        8,
        2,
        eps_1e_6,
        .{ .expected_status = inverse_arithmetic },
        0x1009,
    );
    try std.testing.expectEqual(@as(usize, 0), prefix_fault.results);
    try std.testing.expectEqual(@as(usize, 2), prefix_fault.reads);

    for (fault_values) |*value| value.* = 0x3f80_0000;
    const abort_values = fault_values[0..8];
    try testAbortOutstanding(&dut, &scratch, abort_values);
    const drain_restart = try run(
        &dut,
        &scratch,
        abort_values,
        8,
        1,
        eps_1e_6,
        .{ .hold_first_result_cycles = 3 },
        0x1010,
    );

    try testAbortInverseActive(&dut, &scratch, fault_values);
    const inverse_restart = try run(
        &dut,
        &scratch,
        abort_values,
        8,
        1,
        eps_1e_6,
        .{},
        0x1011,
    );

    try testAbortStalledResult(&dut, &scratch, fault_values);
    const emit_restart = try run(
        &dut,
        &scratch,
        abort_values,
        8,
        1,
        eps_1e_6,
        .{},
        0x1012,
    );

    std.debug.print(
        "\n  section RMSNorm reduction: 128x4={d} cycles, 4096x4={d} cycles\n" ++
            "  writes/reads/results full={d}/{d}/{d}; restarts={d}/{d}/{d} cycles\n" ++
            "  exact raw-F32 scalars, frame/child faults, atomic publication, " ++
            "owner/inverse/emit abort+restart: passed\n\n",
        .{
            compact_stats.cycles,
            full_stats.cycles,
            full_stats.writes,
            full_stats.reads,
            full_stats.results,
            drain_restart.cycles,
            inverse_restart.cycles,
            emit_restart.cycles,
        },
    );
}
