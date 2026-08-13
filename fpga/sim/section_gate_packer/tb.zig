//! Focused cosim for the native GEMM-result to streamed-GATE group packer.
//!
//! The maximum legal shape checks every input beat, packed scalar, and inferred
//! tag under independent source/output stalls. Directed tests separate physical
//! run TLAST from logical block TLAST and cover malformed keep/framing, abort,
//! stalled-output cancellation, and clean restart.

const std = @import("std");
const section = @import("shared_section");
const c = @cImport(@cInclude("shim.h"));

comptime {
    if (section.q8_buffer_token_capacity != 4 or
        section.q8_buffer_block_capacity != 384 or
        section.ffn_pair_groups_per_block != 4)
    {
        @compileError("section_gate_packer geometry drifted from the shared oracle");
    }
}

const zero_lanes = [_]u32{0} ** 8;

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
        c.dut_set_start(self.handle, 0, 0, 0);
        c.dut_set_abort(self.handle, 0);
        c.dut_set_input(self.handle, 0, 0, 0, 0);
        c.dut_set_output_ready(self.handle, 0);
        self.eval();
        for (0..4) |_| self.step();
        c.dut_set_rst_n(self.handle, 1);
        self.step();
    }
};

const Tag = struct {
    token: u2,
    block: u9,
    group: u2,
};

const Output = struct {
    lanes: [8]u32,
    tag: Tag,
    last: bool,
};

const Stats = struct {
    cycles: usize,
    input_stalls: usize,
    output_stalls: usize,
    input_run_lasts: usize,
    output_logical_lasts: usize,
};

fn nativeTagAt(group_index: usize, tokens: u3) !Tag {
    const tag = try section.ffnPairNativeTag(tokens, @intCast(group_index));
    return .{
        .token = @intCast(tag.token),
        .block = @intCast(tag.block),
        .group = @intCast(tag.group),
    };
}

fn scalarValue(tag: Tag, lane: u3) u32 {
    return 0x4000_0000 |
        (@as(u32, tag.block) << 12) |
        (@as(u32, tag.token) << 10) |
        (@as(u32, tag.group) << 8) |
        @as(u32, lane);
}

fn groupValue(tag: Tag) [8]u32 {
    var lanes: [8]u32 = undefined;
    for (&lanes, 0..) |*lane, index|
        lane.* = scalarValue(tag, @intCast(index));
    return lanes;
}

fn beatValue(group_index: usize, beat_in_group: usize, tokens: u3) !u64 {
    const tag = try nativeTagAt(group_index, tokens);
    const lane: u3 = @intCast(beat_in_group * 2);
    return @as(u64, scalarValue(tag, lane)) |
        (@as(u64, scalarValue(tag, lane + 1)) << 32);
}

fn readOutput(dut: *Dut) Output {
    var lanes = zero_lanes;
    c.dut_output_data(dut.handle, &lanes);
    return .{
        .lanes = lanes,
        .tag = .{
            .token = @truncate(c.dut_output_token(dut.handle)),
            .block = @truncate(c.dut_output_block(dut.handle)),
            .group = @truncate(c.dut_output_group(dut.handle)),
        },
        .last = c.dut_output_last(dut.handle) != 0,
    };
}

fn startRun(dut: *Dut, tokens: u3, blocks: u9) !void {
    c.dut_set_start(dut.handle, 1, tokens, blocks);
    dut.eval();
    try std.testing.expect(c.dut_start_ready(dut.handle) != 0);
    dut.step();
    c.dut_set_start(dut.handle, 0, 0, 0);
    dut.eval();
}

fn runSuccess(dut: *Dut, tokens: u3, blocks: u9, seed: u64, ideal: bool) !Stats {
    try startRun(dut, tokens, blocks);
    try std.testing.expect(c.dut_busy(dut.handle) != 0);
    try std.testing.expect(c.dut_done(dut.handle) == 0);
    try std.testing.expect(c.dut_error(dut.handle) == 0);

    const group_count = @as(usize, tokens) * @as(usize, blocks) *
        section.ffn_pair_groups_per_block;
    const beat_count = group_count * 4;
    var input_index: usize = 0;
    var output_index: usize = 0;
    var input_presenting = false;
    var held_output: ?Output = null;
    var prng = std.Random.DefaultPrng.init(seed);
    const rnd = prng.random();
    var stats: Stats = .{
        .cycles = 0,
        .input_stalls = 0,
        .output_stalls = 0,
        .input_run_lasts = 0,
        .output_logical_lasts = 0,
    };

    while (c.dut_done(dut.handle) == 0) : (stats.cycles += 1) {
        if (stats.cycles > beat_count * 8 + 1024) return error.StreamTimeout;

        if (!input_presenting and input_index < beat_count)
            input_presenting = ideal or rnd.uintLessThan(u8, 4) != 0;
        if (input_presenting) {
            const run_last = input_index + 1 == beat_count;
            c.dut_set_input(
                dut.handle,
                1,
                try beatValue(input_index / 4, input_index % 4, tokens),
                0xff,
                @intFromBool(run_last),
            );
        } else {
            c.dut_set_input(dut.handle, 0, 0, 0, 0);
        }

        const output_ready = ideal or rnd.uintLessThan(u8, 3) != 0;
        c.dut_set_output_ready(dut.handle, @intFromBool(output_ready));
        dut.eval();

        const input_fire = input_presenting and c.dut_input_ready(dut.handle) != 0;
        if (input_presenting and !input_fire) stats.input_stalls += 1;
        if (input_fire) {
            if (input_index + 1 == beat_count) stats.input_run_lasts += 1;
            input_index += 1;
            input_presenting = false;
        }

        const output_valid = c.dut_output_valid(dut.handle) != 0;
        if (output_valid) {
            const got = readOutput(dut);
            const tag = try nativeTagAt(output_index, tokens);
            const expected: Output = .{
                .lanes = groupValue(tag),
                .tag = tag,
                .last = tag.group == 3,
            };
            try std.testing.expectEqual(expected, got);
            if (!output_ready) {
                stats.output_stalls += 1;
                if (held_output) |held| try std.testing.expectEqual(held, got);
                held_output = got;
            } else {
                if (got.last) stats.output_logical_lasts += 1;
                output_index += 1;
                held_output = null;
            }
        } else {
            held_output = null;
        }

        dut.step();
    }

    try std.testing.expectEqual(beat_count, input_index);
    try std.testing.expectEqual(group_count, output_index);
    try std.testing.expectEqual(@as(usize, 1), stats.input_run_lasts);
    try std.testing.expectEqual(
        @as(usize, tokens) * @as(usize, blocks),
        stats.output_logical_lasts,
    );
    try std.testing.expect(c.dut_busy(dut.handle) == 0);
    try std.testing.expect(c.dut_done(dut.handle) != 0);
    try std.testing.expect(c.dut_error(dut.handle) == 0);
    c.dut_set_input(dut.handle, 0, 0, 0, 0);
    c.dut_set_output_ready(dut.handle, 0);
    dut.step();
    // Completion and status are sticky until the next accepted start.
    try std.testing.expect(c.dut_done(dut.handle) != 0);
    try std.testing.expect(c.dut_error(dut.handle) == 0);
    return stats;
}

fn expectFailure(dut: *Dut) !void {
    try std.testing.expect(c.dut_busy(dut.handle) == 0);
    try std.testing.expect(c.dut_done(dut.handle) != 0);
    try std.testing.expect(c.dut_error(dut.handle) != 0);
    try std.testing.expect(c.dut_input_ready(dut.handle) == 0);
    try std.testing.expect(c.dut_output_valid(dut.handle) == 0);
    dut.step();
    try std.testing.expect(c.dut_done(dut.handle) != 0);
    try std.testing.expect(c.dut_error(dut.handle) != 0);
}

fn sendBeat(dut: *Dut, data: u64, keep: u8, last: bool) !void {
    c.dut_set_input(dut.handle, 1, data, keep, @intFromBool(last));
    var cycles: usize = 0;
    while (true) : (cycles += 1) {
        if (cycles > 32) return error.InputTimeout;
        dut.eval();
        if (c.dut_input_ready(dut.handle) != 0) {
            dut.step();
            break;
        }
        dut.step();
    }
    c.dut_set_input(dut.handle, 0, 0, 0, 0);
    dut.eval();
}

fn exerciseInvalidShapes(dut: *Dut) !void {
    const shapes = [_]struct { tokens: u3, blocks: u9 }{
        .{ .tokens = 0, .blocks = 1 },
        .{ .tokens = 5, .blocks = 1 },
        .{ .tokens = 1, .blocks = 0 },
        .{ .tokens = 1, .blocks = 385 },
    };
    for (shapes) |shape| {
        try startRun(dut, shape.tokens, shape.blocks);
        try expectFailure(dut);
    }
}

fn exerciseFramingFaults(dut: *Dut) !void {
    // Early physical TLAST must not be confused with group or block framing.
    try startRun(dut, 1, 1);
    try sendBeat(dut, try beatValue(0, 0, 1), 0xff, true);
    try expectFailure(dut);

    try startRun(dut, 1, 1);
    try sendBeat(dut, try beatValue(0, 0, 1), 0x7f, false);
    try expectFailure(dut);

    // Missing physical TLAST is rejected on the final beat even though that beat
    // also completes a logical group-three output.
    try startRun(dut, 1, 1);
    c.dut_set_output_ready(dut.handle, 1);
    for (0..15) |beat|
        try sendBeat(dut, try beatValue(beat / 4, beat % 4, 1), 0xff, false);
    try sendBeat(dut, try beatValue(3, 3, 1), 0xff, false);
    try expectFailure(dut);

    _ = try runSuccess(dut, 2, 2, 0xface, false);
}

fn exerciseAbortAndRestart(dut: *Dut) !void {
    try startRun(dut, 1, 1);
    for (0..4) |beat|
        try sendBeat(dut, try beatValue(beat / 4, beat % 4, 1), 0xff, false);
    c.dut_set_output_ready(dut.handle, 0);
    dut.eval();
    try std.testing.expect(c.dut_output_valid(dut.handle) != 0);
    const held = readOutput(dut);
    dut.step();
    try std.testing.expectEqual(held, readOutput(dut));

    c.dut_set_abort(dut.handle, 1);
    dut.eval();
    try std.testing.expect(c.dut_input_ready(dut.handle) == 0);
    try std.testing.expect(c.dut_output_valid(dut.handle) == 0);
    dut.step();
    c.dut_set_abort(dut.handle, 0);
    try expectFailure(dut);

    _ = try runSuccess(dut, 4, 3, 0xab07, false);
}

pub fn main() !void {
    var dut = Dut.init();
    defer dut.deinit();
    dut.reset();

    try exerciseInvalidShapes(&dut);
    const ideal = try runSuccess(&dut, 4, 8, 0, true);
    const full = try runSuccess(
        &dut,
        @intCast(section.q8_buffer_token_capacity),
        @intCast(section.q8_buffer_block_capacity),
        0x6a7e_2026,
        false,
    );
    try std.testing.expect(full.input_stalls != 0);
    try std.testing.expect(full.output_stalls != 0);
    try exerciseFramingFaults(&dut);
    try exerciseAbortAndRestart(&dut);

    std.debug.print(
        "section GATE packer: ideal 512 beats / 128 groups in {d} cycles; " ++
            "max 24576 beats / 6144 exact groups, stalls in/out={d}/{d}, " ++
            "physical/logical TLAST={d}/{d}, cycles={d}\n",
        .{
            ideal.cycles,
            full.input_stalls,
            full.output_stalls,
            full.input_run_lasts,
            full.output_logical_lasts,
            full.cycles,
        },
    );
}
