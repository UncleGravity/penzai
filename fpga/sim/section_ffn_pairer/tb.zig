//! Focused cosim for the P3 streamed GATE/resident-UP pairer.
//!
//! The maximum-shape sweep checks every legal tag and X1 group address under
//! independent input, request, response, and output stalls. Directed cases
//! exercise malformed tags/framing, read errors, abort with an outstanding
//! response, output cancellation, and clean restart.

const std = @import("std");
const section = @import("shared_section");
const c = @cImport(@cInclude("shim.h"));

comptime {
    if (section.q8_buffer_token_capacity != 4 or
        section.q8_buffer_block_capacity != 384 or
        section.ffn_pair_groups_per_block != 4)
    {
        @compileError("section_ffn_pairer geometry drifted from the shared oracle");
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
        c.dut_set_gate(self.handle, 0, &zero_lanes, 0, 0, 0, 0);
        c.dut_set_read_request_ready(self.handle, 0);
        c.dut_set_read_response(self.handle, 0, &zero_lanes, 0);
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

const Request = struct {
    role: u2,
    token: u3,
    group: u11,
};

const Output = struct {
    gate: u32,
    up: u32,
    last: bool,
};

const PendingResponse = struct {
    tag: Tag,
    lanes: [8]u32,
    delay: u3,
    presenting: bool = false,
};

const RunStats = struct {
    cycles: usize,
    input_stalls: usize,
    request_stalls: usize,
    response_stalls: usize,
    output_stalls: usize,
};

fn canonicalTagAt(group_index: usize, tokens: u3) !Tag {
    const tag = try section.ffnPairCanonicalTag(tokens, @intCast(group_index));
    return .{
        .token = @intCast(tag.token),
        .block = @intCast(tag.block),
        .group = @intCast(tag.group),
    };
}

fn nativeTagAt(group_index: usize, tokens: u3) !Tag {
    const tag = try section.ffnPairNativeTag(tokens, @intCast(group_index));
    return .{
        .token = @intCast(tag.token),
        .block = @intCast(tag.block),
        .group = @intCast(tag.group),
    };
}

fn gateScalar(tag: Tag, lane: u3) u32 {
    return 0x4000_0000 |
        (@as(u32, tag.block) << 12) |
        (@as(u32, tag.token) << 10) |
        (@as(u32, tag.group) << 8) |
        @as(u32, lane);
}

fn upScalar(tag: Tag, lane: u3) u32 {
    return 0x8000_0000 |
        (@as(u32, tag.block) << 12) |
        (@as(u32, tag.token) << 10) |
        (@as(u32, tag.group) << 8) |
        @as(u32, lane);
}

fn gateGroup(tag: Tag) [8]u32 {
    var lanes: [8]u32 = undefined;
    for (&lanes, 0..) |*lane, index| lane.* = gateScalar(tag, @intCast(index));
    return lanes;
}

fn upGroup(tag: Tag) [8]u32 {
    var lanes: [8]u32 = undefined;
    for (&lanes, 0..) |*lane, index| lane.* = upScalar(tag, @intCast(index));
    return lanes;
}

fn startRun(dut: *Dut, tokens: u3, blocks: u9) !void {
    c.dut_set_start(dut.handle, 1, tokens, blocks);
    dut.eval();
    try std.testing.expect(c.dut_start_ready(dut.handle) != 0);
    dut.step();
    c.dut_set_start(dut.handle, 0, 0, 0);
    dut.eval();
}

fn runConfiguredSuccess(
    dut: *Dut,
    tokens: u3,
    blocks: u9,
    seed: u64,
    ideal: bool,
) !RunStats {
    try startRun(dut, tokens, blocks);
    try std.testing.expect(c.dut_busy(dut.handle) != 0);
    try std.testing.expect(c.dut_error(dut.handle) == 0);

    const group_count: usize = @as(usize, tokens) * @as(usize, blocks) *
        section.ffn_pair_groups_per_block;
    const scalar_count = group_count * 8;
    var prng = std.Random.DefaultPrng.init(seed);
    const rnd = prng.random();
    var input_index: usize = 0;
    var request_index: usize = 0;
    var output_index: usize = 0;
    var input_presenting = false;
    var pending: ?PendingResponse = null;
    var held_request: ?Request = null;
    var held_output: ?Output = null;
    var stats: RunStats = .{
        .cycles = 0,
        .input_stalls = 0,
        .request_stalls = 0,
        .response_stalls = 0,
        .output_stalls = 0,
    };

    while (c.dut_done(dut.handle) == 0) : (stats.cycles += 1) {
        if (stats.cycles > scalar_count * 8 + 4096) return error.StreamTimeout;

        if (!input_presenting and input_index < group_count)
            input_presenting = ideal or rnd.uintLessThan(u8, 4) != 0;
        if (input_presenting) {
            const tag = try nativeTagAt(input_index, tokens);
            const lanes = gateGroup(tag);
            c.dut_set_gate(
                dut.handle,
                1,
                &lanes,
                @intFromBool(tag.group == 3),
                tag.token,
                tag.block,
                tag.group,
            );
        } else {
            c.dut_set_gate(dut.handle, 0, &zero_lanes, 0, 0, 0, 0);
        }

        const request_ready = ideal or rnd.uintLessThan(u8, 4) != 0;
        const output_ready = ideal or rnd.uintLessThan(u8, 3) != 0;
        c.dut_set_read_request_ready(dut.handle, @intFromBool(request_ready));
        c.dut_set_output_ready(dut.handle, @intFromBool(output_ready));

        var response_presented = false;
        if (pending) |rsp| {
            if (rsp.presenting) {
                c.dut_set_read_response(dut.handle, 1, &rsp.lanes, 0);
                response_presented = true;
            } else {
                c.dut_set_read_response(dut.handle, 0, &zero_lanes, 0);
            }
        } else {
            c.dut_set_read_response(dut.handle, 0, &zero_lanes, 0);
        }
        dut.eval();

        const input_fire = input_presenting and c.dut_gate_ready(dut.handle) != 0;
        if (input_presenting and !input_fire) stats.input_stalls += 1;
        if (input_fire) {
            input_index += 1;
            input_presenting = false;
        }

        const request_valid = c.dut_read_request_valid(dut.handle) != 0;
        if (request_valid) {
            const got: Request = .{
                .role = @truncate(c.dut_read_request_role(dut.handle)),
                .token = @truncate(c.dut_read_request_token(dut.handle)),
                .group = @truncate(c.dut_read_request_group(dut.handle)),
            };
            const tag = try canonicalTagAt(request_index, tokens);
            const expected_group = try section.ffnPairScratchGroup(tag.block, tag.group);
            try std.testing.expectEqual(@as(u2, 2), got.role);
            try std.testing.expectEqual(@as(u3, tag.token), got.token);
            try std.testing.expectEqual(@as(u11, @intCast(expected_group)), got.group);
            if (!request_ready) {
                stats.request_stalls += 1;
                if (held_request) |held| try std.testing.expectEqual(held, got);
                held_request = got;
            } else {
                try std.testing.expect(pending == null);
                pending = .{
                    .tag = tag,
                    .lanes = upGroup(tag),
                    .delay = if (ideal) 0 else @intCast(rnd.uintLessThan(u8, 6)),
                };
                request_index += 1;
                held_request = null;
            }
        } else {
            held_request = null;
        }

        if (pending) |*rsp| {
            if (response_presented) {
                if (c.dut_read_response_ready(dut.handle) != 0) {
                    pending = null;
                } else {
                    stats.response_stalls += 1;
                }
            } else if (rsp.delay != 0) {
                rsp.delay -= 1;
            } else {
                rsp.presenting = true;
            }
        }

        const output_valid = c.dut_output_valid(dut.handle) != 0;
        if (output_valid) {
            const got: Output = .{
                .gate = c.dut_output_gate(dut.handle),
                .up = c.dut_output_up(dut.handle),
                .last = c.dut_output_last(dut.handle) != 0,
            };
            const group_index = output_index / 8;
            const lane: u3 = @intCast(output_index % 8);
            const tag = try canonicalTagAt(group_index, tokens);
            const expected: Output = .{
                .gate = gateScalar(tag, lane),
                .up = upScalar(tag, lane),
                .last = tag.group == 3 and lane == 7,
            };
            try std.testing.expectEqual(expected, got);
            if (!output_ready) {
                stats.output_stalls += 1;
                if (held_output) |held| try std.testing.expectEqual(held, got);
                held_output = got;
            } else {
                output_index += 1;
                held_output = null;
            }
        } else {
            held_output = null;
        }

        dut.step();
    }

    try std.testing.expectEqual(group_count, input_index);
    try std.testing.expectEqual(group_count, request_index);
    try std.testing.expectEqual(scalar_count, output_index);
    try std.testing.expect(pending == null);
    try std.testing.expect(c.dut_busy(dut.handle) == 0);
    try std.testing.expect(c.dut_error(dut.handle) == 0);
    c.dut_set_gate(dut.handle, 0, &zero_lanes, 0, 0, 0, 0);
    c.dut_set_read_request_ready(dut.handle, 0);
    c.dut_set_read_response(dut.handle, 0, &zero_lanes, 0);
    c.dut_set_output_ready(dut.handle, 0);
    dut.step();
    return stats;
}

fn sendOneGroup(dut: *Dut, tag: Tag, last: bool) !void {
    const lanes = gateGroup(tag);
    c.dut_set_gate(
        dut.handle,
        1,
        &lanes,
        @intFromBool(last),
        tag.token,
        tag.block,
        tag.group,
    );
    var cycles: usize = 0;
    while (true) : (cycles += 1) {
        if (cycles > 32) return error.InputTimeout;
        dut.eval();
        if (c.dut_gate_ready(dut.handle) != 0) {
            dut.step();
            break;
        }
        dut.step();
    }
    c.dut_set_gate(dut.handle, 0, &zero_lanes, 0, 0, 0, 0);
    dut.eval();
}

fn sendOneTokenBlock(dut: *Dut, token: u2, block: u9) !void {
    inline for (0..4) |group| {
        try sendOneGroup(
            dut,
            .{ .token = token, .block = block, .group = @intCast(group) },
            group == 3,
        );
    }
}

fn waitRequest(dut: *Dut) !Request {
    c.dut_set_read_request_ready(dut.handle, 0);
    var cycles: usize = 0;
    while (c.dut_read_request_valid(dut.handle) == 0) : (cycles += 1) {
        if (cycles > 32) return error.RequestTimeout;
        dut.step();
    }
    return .{
        .role = @truncate(c.dut_read_request_role(dut.handle)),
        .token = @truncate(c.dut_read_request_token(dut.handle)),
        .group = @truncate(c.dut_read_request_group(dut.handle)),
    };
}

fn expectClosedFailure(dut: *Dut) !void {
    try std.testing.expect(c.dut_done(dut.handle) != 0);
    try std.testing.expect(c.dut_error(dut.handle) != 0);
    try std.testing.expect(c.dut_busy(dut.handle) == 0);
    try std.testing.expect(c.dut_gate_ready(dut.handle) == 0);
    try std.testing.expect(c.dut_read_request_valid(dut.handle) == 0);
    try std.testing.expect(c.dut_output_valid(dut.handle) == 0);
}

fn exerciseFaultsAndRestart(dut: *Dut) !void {
    // Invalid shape fails synchronously without opening any stream boundary.
    c.dut_set_start(dut.handle, 1, 0, 1);
    dut.eval();
    try std.testing.expect(c.dut_start_ready(dut.handle) != 0);
    dut.step();
    c.dut_set_start(dut.handle, 0, 0, 0);
    try expectClosedFailure(dut);
    dut.step();

    // Wrong first tag is consumed and terminates the run before any read issue.
    try startRun(dut, 1, 1);
    try sendOneGroup(dut, .{ .token = 0, .block = 0, .group = 1 }, false);
    try expectClosedFailure(dut);
    dut.step();

    // Correct tag with early TLAST is also a terminal framing error.
    try startRun(dut, 1, 1);
    try sendOneGroup(dut, .{ .token = 0, .block = 0, .group = 0 }, true);
    try expectClosedFailure(dut);
    dut.step();

    // Two-token ingress is native GEMM order. Canonical token-zero g2 is rejected
    // after its lower pair because token one g0 must arrive next.
    try startRun(dut, 2, 1);
    try sendOneGroup(dut, .{ .token = 0, .block = 0, .group = 0 }, false);
    try sendOneGroup(dut, .{ .token = 0, .block = 0, .group = 1 }, false);
    try sendOneGroup(dut, .{ .token = 0, .block = 0, .group = 2 }, false);
    try expectClosedFailure(dut);
    dut.step();

    // Missing TLAST on group three is the complementary terminal frame error.
    try startRun(dut, 1, 1);
    inline for (0..4) |group| {
        try sendOneGroup(
            dut,
            .{ .token = 0, .block = 0, .group = @intCast(group) },
            false,
        );
    }
    try expectClosedFailure(dut);
    dut.step();

    // A scratch error clears queued and output state and permits a clean restart.
    try startRun(dut, 1, 1);
    const tag: Tag = .{ .token = 0, .block = 0, .group = 0 };
    try sendOneTokenBlock(dut, 0, 0);
    const request = try waitRequest(dut);
    try std.testing.expectEqual(@as(u11, 0), request.group);
    c.dut_set_read_request_ready(dut.handle, 1);
    dut.eval();
    dut.step();
    c.dut_set_read_request_ready(dut.handle, 0);
    const up = upGroup(tag);
    c.dut_set_read_response(dut.handle, 1, &up, 1);
    dut.eval();
    try std.testing.expect(c.dut_read_response_ready(dut.handle) != 0);
    dut.step();
    c.dut_set_read_response(dut.handle, 0, &zero_lanes, 0);
    try expectClosedFailure(dut);
    dut.step();

    _ = try runConfiguredSuccess(dut, 1, 1, 0x1234, false);
}

fn exerciseAbortOrphanAndRestart(dut: *Dut) !void {
    try startRun(dut, 1, 1);
    const tag: Tag = .{ .token = 0, .block = 0, .group = 0 };
    try sendOneTokenBlock(dut, 0, 0);
    _ = try waitRequest(dut);

    // Accept the read request, then abort before its response appears.
    c.dut_set_read_request_ready(dut.handle, 1);
    dut.eval();
    dut.step();
    c.dut_set_read_request_ready(dut.handle, 0);
    c.dut_set_abort(dut.handle, 1);
    dut.eval();
    try std.testing.expect(c.dut_output_valid(dut.handle) == 0);
    dut.step();
    c.dut_set_abort(dut.handle, 0);
    try expectClosedFailure(dut);
    dut.step();
    try std.testing.expect(c.dut_start_ready(dut.handle) == 0);

    // The late response is drained but never emitted. Restart opens only after it.
    const up = upGroup(tag);
    c.dut_set_read_response(dut.handle, 1, &up, 0);
    dut.eval();
    try std.testing.expect(c.dut_read_response_ready(dut.handle) != 0);
    try std.testing.expect(c.dut_start_ready(dut.handle) == 0);
    dut.step();
    c.dut_set_read_response(dut.handle, 0, &zero_lanes, 0);
    dut.eval();
    try std.testing.expect(c.dut_start_ready(dut.handle) != 0);

    _ = try runConfiguredSuccess(dut, 2, 2, 0xabcd, false);
}

fn exerciseAbortStalledOutput(dut: *Dut) !void {
    try startRun(dut, 1, 1);
    const tag: Tag = .{ .token = 0, .block = 0, .group = 0 };
    try sendOneTokenBlock(dut, 0, 0);
    _ = try waitRequest(dut);
    c.dut_set_read_request_ready(dut.handle, 1);
    dut.step();
    c.dut_set_read_request_ready(dut.handle, 0);
    const up = upGroup(tag);
    c.dut_set_read_response(dut.handle, 1, &up, 0);
    dut.eval();
    try std.testing.expect(c.dut_read_response_ready(dut.handle) != 0);
    dut.step();
    c.dut_set_read_response(dut.handle, 0, &zero_lanes, 0);
    c.dut_set_output_ready(dut.handle, 0);

    var cycles: usize = 0;
    while (c.dut_output_valid(dut.handle) == 0) : (cycles += 1) {
        if (cycles > 32) return error.OutputTimeout;
        dut.step();
    }
    const held: Output = .{
        .gate = c.dut_output_gate(dut.handle),
        .up = c.dut_output_up(dut.handle),
        .last = c.dut_output_last(dut.handle) != 0,
    };
    dut.step();
    try std.testing.expectEqual(held.gate, c.dut_output_gate(dut.handle));
    try std.testing.expectEqual(held.up, c.dut_output_up(dut.handle));
    try std.testing.expectEqual(held.last, c.dut_output_last(dut.handle) != 0);

    c.dut_set_abort(dut.handle, 1);
    dut.eval();
    try std.testing.expect(c.dut_output_valid(dut.handle) == 0);
    dut.step();
    c.dut_set_abort(dut.handle, 0);
    try expectClosedFailure(dut);
    dut.step();

    _ = try runConfiguredSuccess(dut, 1, 1, 0x5555, false);
}

pub fn main() !void {
    var dut = Dut.init();
    defer dut.deinit();
    dut.reset();

    const ideal = try runConfiguredSuccess(&dut, 4, 8, 0, true);
    try std.testing.expect(ideal.cycles <= 1168);

    const full = try runConfiguredSuccess(
        &dut,
        @intCast(section.q8_buffer_token_capacity),
        @intCast(section.q8_buffer_block_capacity),
        0xf3c0_2026,
        false,
    );
    try std.testing.expect(full.input_stalls != 0);
    try std.testing.expect(full.request_stalls != 0);
    try std.testing.expect(full.response_stalls != 0);
    try std.testing.expect(full.output_stalls != 0);

    try exerciseFaultsAndRestart(&dut);
    try exerciseAbortOrphanAndRestart(&dut);
    try exerciseAbortStalledOutput(&dut);

    std.debug.print(
        "section FFN pairer: ideal 128 groups / 1024 scalars in {d} cycles; " ++
            "6144 tagged X1 reads / 49152 ordered scalars, " ++
            "stalls in/req/rsp/out={d}/{d}/{d}/{d}, cycles={d}\n",
        .{
            ideal.cycles,
            full.input_stalls,
            full.request_stalls,
            full.response_stalls,
            full.output_stalls,
            full.cycles,
        },
    );
}
