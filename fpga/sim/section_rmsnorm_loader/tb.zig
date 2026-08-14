//! Ordering, framing, backpressure, and restart cosim for the P3d residual loader.

const std = @import("std");
const section = @import("shared_section");
const c = @cImport(@cInclude("shim.h"));

const RunStats = struct {
    cycles: usize,
    write_stalls: usize,
    group_stalls: usize,
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
        c.dut_set_config(self.handle, 0, 0, 0);
        c.dut_set_abort(self.handle, 0);
        c.dut_set_input(self.handle, 0, 0, 0, 0);
        c.dut_set_write_sink(self.handle, 0, 0);
        c.dut_set_group_ready(self.handle, 0);
        self.eval();
        for (0..4) |_| self.step();
        c.dut_set_rst_n(self.handle, 1);
        self.step();
    }
};

fn configure(dut: *Dut, rows: u32, tokens: u3) !void {
    try std.testing.expect(c.dut_config_ready(dut.handle) != 0);
    c.dut_set_config(dut.handle, 1, @intCast(rows), tokens);
    dut.eval();
    try std.testing.expect(c.dut_config_ready(dut.handle) != 0);
    dut.step();
    c.dut_set_config(dut.handle, 0, 0, 0);
    dut.eval();
}

fn wordAt(values: []const u32, ordinal: usize) u64 {
    return @as(u64, values[ordinal * 2]) |
        (@as(u64, values[ordinal * 2 + 1]) << 32);
}

fn readGroup(dut: *Dut) [8]u32 {
    var lanes: [8]u32 = undefined;
    c.dut_group_data(dut.handle, &lanes);
    return lanes;
}

fn runSuccess(
    dut: *Dut,
    values: []const u32,
    rows: u32,
    tokens: u3,
    stalls: bool,
    seed: u64,
) !RunStats {
    const token_count: usize = tokens;
    try std.testing.expectEqual(@as(usize, rows) * token_count, values.len);
    try configure(dut, rows, tokens);
    try std.testing.expect(c.dut_busy(dut.handle) != 0);

    var prng = std.Random.DefaultPrng.init(seed);
    const rnd = prng.random();
    const total_words = values.len / 2;
    const total_groups = values.len / 8;
    var sent_words: usize = 0;
    var received_groups: usize = 0;
    var presenting = false;
    var held_group: ?[8]u32 = null;
    var cycle: usize = 0;
    var write_stalls: usize = 0;
    var group_stalls: usize = 0;

    while (true) {
        if (cycle > total_words * 8 + 1024) return error.StreamTimeout;

        if (!presenting and sent_words < total_words)
            presenting = !stalls or rnd.uintLessThan(u8, 4) != 0;
        const input_word = if (presenting) wordAt(values, sent_words) else 0;
        c.dut_set_input(
            dut.handle,
            @intFromBool(presenting),
            input_word,
            if (presenting) 0xff else 0,
            @intFromBool(presenting and sent_words + 1 == total_words),
        );

        var wr_ready = !stalls or rnd.uintLessThan(u8, 4) != 0;
        if (stalls and presenting and write_stalls == 0) wr_ready = false;
        c.dut_set_write_sink(dut.handle, @intFromBool(wr_ready), 0);
        var group_ready = !stalls or rnd.uintLessThan(u8, 3) != 0;
        c.dut_set_group_ready(dut.handle, @intFromBool(group_ready));
        dut.eval();
        if (stalls and c.dut_group_valid(dut.handle) != 0 and group_stalls == 0) {
            group_ready = false;
            c.dut_set_group_ready(dut.handle, 0);
            dut.eval();
        }

        const input_ready = c.dut_input_ready(dut.handle) != 0;
        const input_fire = presenting and input_ready;
        const write_valid = c.dut_write_valid(dut.handle) != 0;
        const write_fire = write_valid and wr_ready;
        try std.testing.expectEqual(input_fire, write_fire);
        if (presenting and !input_ready) write_stalls += 1;

        if (write_valid) {
            const words_per_token = @as(usize, rows) / 2;
            const token = sent_words / words_per_token;
            const word = sent_words % words_per_token;
            const loc = try section.rmsNormResidualWordLocation(
                rows,
                tokens,
                @intCast(sent_words),
            );
            try std.testing.expectEqual(@as(u8, loc.bank), c.dut_write_bank(dut.handle));
            try std.testing.expectEqual(@as(u16, @intCast(loc.address)), c.dut_write_address(dut.handle));
            try std.testing.expectEqual(input_word, c.dut_write_data(dut.handle));
            try std.testing.expectEqual(token, @as(usize, loc.address) / section.f32GroupsPerToken(.residual));
            try std.testing.expectEqual(word % 4, @as(usize, loc.bank));
        }

        const group_valid = c.dut_group_valid(dut.handle) != 0;
        const group_fire = group_valid and group_ready;
        if (group_valid) {
            const got = readGroup(dut);
            const expected = values[received_groups * 8 ..][0..8];
            try std.testing.expectEqualSlices(u32, expected, &got);
            try std.testing.expectEqual(received_groups + 1 == total_groups, c.dut_group_last(dut.handle) != 0);
            if (!group_ready) {
                group_stalls += 1;
                if (held_group) |prior| try std.testing.expectEqual(prior, got);
                held_group = got;
            } else {
                held_group = null;
            }
        } else {
            held_group = null;
        }

        if (input_fire) {
            sent_words += 1;
            presenting = false;
        }
        if (group_fire) received_groups += 1;

        dut.step();
        cycle += 1;
        try std.testing.expect(c.dut_error(dut.handle) == 0);
        if (c.dut_done(dut.handle) != 0) break;
    }

    try std.testing.expectEqual(total_words, sent_words);
    try std.testing.expectEqual(total_groups, received_groups);
    try std.testing.expect(c.dut_busy(dut.handle) == 0);
    return .{
        .cycles = cycle,
        .write_stalls = write_stalls,
        .group_stalls = group_stalls,
    };
}

fn sendWord(
    dut: *Dut,
    data: u64,
    keep: u8,
    last: bool,
    sink_error: bool,
) !void {
    c.dut_set_input(dut.handle, 1, data, keep, @intFromBool(last));
    c.dut_set_write_sink(dut.handle, 1, @intFromBool(sink_error));
    c.dut_set_group_ready(dut.handle, 1);
    dut.eval();
    try std.testing.expect(c.dut_input_ready(dut.handle) != 0);
    try std.testing.expect(c.dut_write_valid(dut.handle) != 0);
    dut.step();
    c.dut_set_input(dut.handle, 0, 0, 0, 0);
    c.dut_set_write_sink(dut.handle, 0, 0);
    dut.eval();
}

fn expectConfigFault(dut: *Dut, rows: u32, tokens: u3) !void {
    try configure(dut, rows, tokens);
    try std.testing.expect(c.dut_done(dut.handle) != 0);
    try std.testing.expect(c.dut_error(dut.handle) != 0);
    try std.testing.expectEqual(
        section.RmsNormLoaderStatus.bad_cfg,
        @as(u4, @truncate(c.dut_status(dut.handle))),
    );
}

fn verifyFaultsAndRestart(dut: *Dut) !void {
    try expectConfigFault(dut, 7, 1);
    dut.step();
    try expectConfigFault(dut, 4104, 1);
    dut.step();
    try expectConfigFault(dut, 8, 0);
    dut.step();
    try expectConfigFault(dut, 8, 5);
    dut.step();

    // Early TLAST and partial keep are framing failures.
    try configure(dut, 8, 1);
    try sendWord(dut, 0x3f80_0000_3f80_0000, 0xff, true, false);
    try std.testing.expect(c.dut_done(dut.handle) != 0);
    try std.testing.expectEqual(
        section.RmsNormLoaderStatus.frame,
        @as(u4, @truncate(c.dut_status(dut.handle))),
    );
    dut.step();

    try configure(dut, 8, 1);
    try sendWord(dut, 0x4000_0000_3f80_0000, 0x0f, false, false);
    try std.testing.expect(c.dut_error(dut.handle) != 0);
    try std.testing.expectEqual(
        section.RmsNormLoaderStatus.frame,
        @as(u4, @truncate(c.dut_status(dut.handle))),
    );
    dut.step();

    // A direct scratch sink failure suppresses group publication.
    try configure(dut, 8, 1);
    try sendWord(dut, 0, 0xff, false, true);
    try std.testing.expect(c.dut_error(dut.handle) != 0);
    try std.testing.expectEqual(
        section.RmsNormLoaderStatus.sink,
        @as(u4, @truncate(c.dut_status(dut.handle))),
    );
    try std.testing.expect(c.dut_group_valid(dut.handle) == 0);
    dut.step();

    // Missing final TLAST is diagnosed on the fourth word.
    try configure(dut, 8, 1);
    for (0..4) |word|
        try sendWord(dut, @intCast(word), 0xff, false, false);
    try std.testing.expect(c.dut_error(dut.handle) != 0);
    try std.testing.expectEqual(
        section.RmsNormLoaderStatus.frame,
        @as(u4, @truncate(c.dut_status(dut.handle))),
    );
    dut.step();

    // Abort both while accepting input and while the final group is stalled.
    try configure(dut, 16, 1);
    c.dut_set_abort(dut.handle, 1);
    dut.step();
    c.dut_set_abort(dut.handle, 0);
    dut.eval();
    try std.testing.expect(c.dut_config_ready(dut.handle) != 0);
    try std.testing.expect(c.dut_error(dut.handle) == 0);

    try configure(dut, 8, 1);
    c.dut_set_group_ready(dut.handle, 0);
    for (0..4) |word| try sendWord(dut, @intCast(word), 0xff, word == 3, false);
    c.dut_set_group_ready(dut.handle, 0);
    dut.eval();
    try std.testing.expect(c.dut_group_valid(dut.handle) != 0);
    c.dut_set_abort(dut.handle, 1);
    dut.step();
    c.dut_set_abort(dut.handle, 0);
    dut.eval();
    try std.testing.expect(c.dut_config_ready(dut.handle) != 0);
    try std.testing.expect(c.dut_group_valid(dut.handle) == 0);

    const ones = [_]u32{0x3f80_0000} ** 8;
    const stats = try runSuccess(dut, &ones, 8, 1, false, 0);
    try std.testing.expectEqual(@as(usize, 5), stats.cycles);
}

fn fillValues(values: []u32) void {
    for (values, 0..) |*value, index| {
        const sign: u32 = if (index & 1 == 0) 0 else 0x8000_0000;
        const exponent: u32 = @intCast(1 + (index * 37) % 254);
        const mantissa: u32 = @intCast((index *% 0x45d9_f3b) & 0x007f_ffff);
        value.* = sign | (exponent << 23) | mantissa;
    }
}

pub fn main() !void {
    var dut = Dut.init();
    defer dut.deinit();
    dut.reset();

    var stalled_values: [64 * 4]u32 = undefined;
    fillValues(&stalled_values);
    const stalled = try runSuccess(&dut, &stalled_values, 64, 4, true, 0x51ca_1001);
    try std.testing.expect(stalled.write_stalls > 0);
    try std.testing.expect(stalled.group_stalls > 0);

    var max_values: [4096 * 4]u32 = undefined;
    fillValues(&max_values);
    const maximum = try runSuccess(&dut, &max_values, 4096, 4, false, 0);
    try std.testing.expectEqual(@as(usize, max_values.len / 2 + 1), maximum.cycles);

    try verifyFaultsAndRestart(&dut);

    std.debug.print(
        "section RMSNorm loader: 4096x4 in {d} cycles; exact R mapping, " ++
            "group packing, stalls, faults, and abort/restart passed\n",
        .{maximum.cycles},
    );
}
