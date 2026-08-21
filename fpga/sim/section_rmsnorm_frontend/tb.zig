//! End-to-end cosim for the RMSNorm reduction frontend.
//!
//! A cycle-accurate scratch model checks every direct R location, delays the
//! untagged read response, and injects sink/read faults plus abort-with-owner.

const std = @import("std");
const section = @import("shared_section");
const c = @cImport(@cInclude("shim.h"));

const zero_lanes = [_]u64{ 0, 0, 0, 0 };

const Result = struct {
    token: u2,
    max_exp: u8,
    sum_sq: u48,
    rows: u14,
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
        for (0..4) |_| self.step();
        c.dut_set_rst_n(self.handle, 1);
        self.step();
    }
};

const RunOptions = struct {
    resident: bool = false,
    stalls: bool = true,
    early_last_word: ?usize = null,
    write_error_word: ?usize = null,
    read_error_request: ?usize = null,
    expected_status: ?u7 = null,
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

fn seedScratch(
    scratch: *Scratch,
    values: []const u32,
    rows: u32,
    tokens: u3,
) !void {
    const words_per_token = rows / 2;
    for (0..@as(usize, tokens) * words_per_token) |word| {
        const token = word / words_per_token;
        const token_word = word % words_per_token;
        try scratch.write(
            @intCast(token_word % 4),
            @intCast(token * 512 + token_word / 4),
            wordAt(values, word),
        );
    }
}

fn readResult(dut: *Dut) Result {
    return .{
        .token = @truncate(c.dut_result_token(dut.handle)),
        .max_exp = c.dut_result_max_exp(dut.handle),
        .sum_sq = @truncate(c.dut_result_sum_sq(dut.handle)),
        .rows = @truncate(c.dut_result_rows(dut.handle)),
        .final = c.dut_result_final(dut.handle) != 0,
    };
}

fn configure(dut: *Dut, rows: u32, tokens: u3, resident: bool) !void {
    try std.testing.expect(c.dut_config_ready(dut.handle) != 0);
    c.dut_set_config(
        dut.handle,
        1,
        @intCast(rows),
        tokens,
        @intFromBool(resident),
    );
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
    options: RunOptions,
    seed: u64,
) !RunStats {
    const token_count: usize = tokens;
    try std.testing.expectEqual(@as(usize, rows) * token_count, values.len);

    var expected_max: [4]u8 = .{0} ** 4;
    var expected_sum: [4]section.RmsNormSumsqResult = undefined;
    if (options.expected_status == null) {
        for (0..token_count) |token| {
            const token_values = values[token * rows ..][0..rows];
            const scan = try section.rmsNormMaxExp(token_values, rows);
            try std.testing.expect(!scan.subnormal_warning);
            expected_max[token] = scan.max_exp;
            expected_sum[token] = try section.rmsNormSumsqFixed(
                token_values,
                rows,
                scan.max_exp,
            );
        }
    }

    if (options.resident)
        try seedScratch(scratch, values, rows, tokens);

    try configure(dut, rows, tokens, options.resident);
    try std.testing.expect(c.dut_busy(dut.handle) != 0);

    var prng = std.Random.DefaultPrng.init(seed);
    const rnd = prng.random();
    const total_words = values.len / 2;
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
    var cycle: usize = 0;

    while (true) : (cycle += 1) {
        if (cycle > values.len * 20 + 4096) return error.StreamTimeout;

        if (!options.resident and !presenting and word_index < total_words)
            presenting = !options.stalls or rnd.uintLessThan(u8, 4) != 0;
        const stream_data = if (presenting) wordAt(values, word_index) else 0;
        const natural_last = presenting and word_index + 1 == total_words;
        const stream_last = natural_last or
            (presenting and options.early_last_word == word_index);
        c.dut_set_stream(
            dut.handle,
            @intFromBool(presenting),
            stream_data,
            0xff,
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
        const result_ready = !options.stalls or rnd.uintLessThan(u8, 3) != 0;
        c.dut_set_result_ready(dut.handle, @intFromBool(result_ready));
        dut.eval();

        const stream_fire = presenting and c.dut_stream_ready(dut.handle) != 0;
        const write_valid = c.dut_r_write_valid(dut.handle) != 0;
        const write_fire = write_valid and write_ready;
        try std.testing.expectEqual(stream_fire, write_fire);
        if (options.resident) {
            try std.testing.expect(c.dut_stream_ready(dut.handle) == 0);
            try std.testing.expect(!write_valid);
        }
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
        const response_fire = response_active and c.dut_read_response_ready(dut.handle) != 0;
        const result_valid = c.dut_result_valid(dut.handle) != 0;
        const result_fire = result_valid and result_ready;

        if (result_valid and !result_ready) {
            const current = readResult(dut);
            if (held_result) |prior| try std.testing.expectEqual(prior, current);
            held_result = current;
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
            const groups_per_pass = values.len / 8;
            const request_in_pass = request_count % groups_per_pass;
            try std.testing.expectEqual(
                @as(u8, @intCast(request_in_pass / (rows / 8))),
                token,
            );
            try std.testing.expectEqual(
                @as(u16, @intCast(request_in_pass % (rows / 8))),
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
            try std.testing.expectEqual(expected_max[result_count], got.max_exp);
            try std.testing.expectEqual(expected_sum[result_count].sum_sq, got.sum_sq);
            try std.testing.expectEqual(@as(u14, @intCast(rows)), got.rows);
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
                try std.testing.expectEqual(
                    expected_status,
                    c.dut_status(dut.handle),
                );
            } else {
                if (c.dut_error(dut.handle) != 0 or
                    c.dut_status(dut.handle) != 0)
                {
                    std.debug.print(
                        "frontend unexpected terminal error: status=0x{x}, " ++
                            "cycles={d}, words={d}/{d}, requests={d}, results={d}\n",
                        .{
                            c.dut_status(dut.handle), cycle + 1,
                            word_index,               total_words,
                            request_count,            result_count,
                        },
                    );
                }
                try std.testing.expect(c.dut_error(dut.handle) == 0);
                try std.testing.expectEqual(@as(u8, 0), c.dut_status(dut.handle));
                try std.testing.expectEqual(
                    if (options.resident) @as(usize, 0) else total_words,
                    write_count,
                );
                try std.testing.expectEqual(
                    (values.len / 8) *
                        (if (options.resident) @as(usize, 2) else 1),
                    request_count,
                );
                try std.testing.expectEqual(token_count, result_count);
            }
            break;
        }
    }

    c.dut_set_stream(dut.handle, 0, 0, 0, 0);
    c.dut_set_r_write_sink(dut.handle, 0, 0);
    c.dut_set_read_request_ready(dut.handle, 0);
    c.dut_set_read_response(dut.handle, 0, &zero_lanes, 0);
    c.dut_set_result_ready(dut.handle, 0);
    dut.eval();
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

fn testBadConfig(dut: *Dut) !void {
    try configure(dut, 7, 1, false);
    try std.testing.expect(c.dut_done(dut.handle) != 0);
    try std.testing.expect(c.dut_error(dut.handle) != 0);
    try std.testing.expect(c.dut_status(dut.handle) & section.RmsNormFrontendStatus.bad_cfg != 0);
    dut.step();
}

fn testAbortOutstanding(
    dut: *Dut,
    scratch: *Scratch,
    values: []const u32,
) !void {
    try configure(dut, 8, 1, false);
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
    dut.eval();
    try std.testing.expect(c.dut_busy(dut.handle) == 0);
    try std.testing.expect(c.dut_done(dut.handle) == 0);
    try std.testing.expect(c.dut_error(dut.handle) == 0);
    try std.testing.expectEqual(@as(u8, 0), c.dut_status(dut.handle));
}

pub fn main() !void {
    var dut = Dut.init();
    defer dut.deinit();
    dut.reset();
    var scratch: Scratch = .{};

    try testBadConfig(&dut);

    const allocator = std.heap.page_allocator;
    const compact = try makeValues(allocator, 128, 4);
    defer allocator.free(compact);
    const compact_stats = try run(&dut, &scratch, compact, 128, 4, .{}, 0x6171_8e22);
    const resident_stats = try run(&dut, &scratch, compact, 128, 4, .{
        .resident = true,
    }, 0x6171_8e23);

    const full = try makeValues(allocator, 4096, 4);
    defer allocator.free(full);
    const full_stats = try run(&dut, &scratch, full, 4096, 4, .{}, 0x8f21_b453);

    var fault_values = try makeValues(allocator, 8, 2);
    defer allocator.free(fault_values);
    _ = try run(&dut, &scratch, fault_values, 8, 2, .{
        .early_last_word = 0,
        .expected_status = section.RmsNormFrontendStatus.loader,
    }, 0x1001);
    _ = try run(&dut, &scratch, fault_values, 8, 2, .{
        .write_error_word = 1,
        .expected_status = section.RmsNormFrontendStatus.loader,
    }, 0x1002);

    fault_values[3] = 0x7f80_0000;
    _ = try run(&dut, &scratch, fault_values, 8, 2, .{
        .expected_status = section.RmsNormFrontendStatus.max_exp,
    }, 0x1003);
    fault_values[3] = 0x0000_0001;
    _ = try run(&dut, &scratch, fault_values, 8, 2, .{
        .expected_status = section.RmsNormFrontendStatus.subnormal,
    }, 0x1004);
    fault_values[3] = 0x3f80_0000;
    _ = try run(&dut, &scratch, fault_values, 8, 2, .{
        .read_error_request = 0,
        .expected_status = section.RmsNormFrontendStatus.scratch,
    }, 0x1005);
    _ = try run(&dut, &scratch, fault_values, 8, 2, .{
        .resident = true,
        .read_error_request = 0,
        .expected_status = section.RmsNormFrontendStatus.max_exp |
            section.RmsNormFrontendStatus.scratch,
    }, 0x1006);

    const abort_values = fault_values[0..8];
    try testAbortOutstanding(&dut, &scratch, abort_values);
    const restart_stats = try run(&dut, &scratch, abort_values, 8, 1, .{}, 0x1007);

    std.debug.print(
        "\n  section RMSNorm frontend: 128x4={d} cycles, " ++
            "resident 128x4={d} cycles, 4096x4={d} cycles\n" ++
            "  writes/reads/results full={d}/{d}/{d}; restart={d} cycles\n" ++
            "  exact scan+sumsq, stalls, loader/scan/scratch faults, abort/drain/restart: passed\n\n",
        .{
            compact_stats.cycles,
            resident_stats.cycles,
            full_stats.cycles,
            full_stats.writes,
            full_stats.reads,
            full_stats.results,
            restart_stats.cycles,
        },
    );
}
