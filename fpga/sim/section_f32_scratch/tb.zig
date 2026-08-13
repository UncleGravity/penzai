//! Focused cosim for the four-bank P2 FP32 section scratch.
//!
//! The test drives the exact GEMM result order, reconstructs token-major
//! 256-bit groups through the read port, and checks role/bounds isolation,
//! malformed framing, simultaneous-port collisions, bubbles, and response
//! backpressure.  Expected locations are derived from shared/section.zig.

const std = @import("std");
const section = @import("shared_section");
const c = @cImport(@cInclude("shim.h"));

const Role = section.F32Role;

comptime {
    if (section.query_tile_max != 4 or
        section.f32_banks_per_role != 4 or
        section.f32_values_per_word != 2 or
        section.f32_rows_per_group != 8)
    {
        @compileError("section_f32_scratch is frozen to P2 section contract v1");
    }
}

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
        c.dut_set_rst_n(self.handle, 0);
        c.dut_set_write_config(self.handle, 0, 0, 0, 0);
        c.dut_set_write_abort(self.handle, 0);
        c.dut_set_write_stream(self.handle, 0, 0, 0, 0);
        c.dut_set_read_request(self.handle, 0, 0, 0, 0);
        c.dut_set_read_ready(self.handle, 0);
        c.dut_set_clk(self.handle, 0);
        self.eval();
        for (0..4) |_| self.step();
        c.dut_set_rst_n(self.handle, 1);
        self.step();
    }
};

const ReadResult = struct {
    lanes: [4]u64,
    is_error: bool,
};

fn roleCode(role: Role) u8 {
    return @intCast(@intFromEnum(role));
}

fn roleSpan(role: Role) u32 {
    return section.f32RoleSpan(role);
}

fn roleBase(role: Role) u32 {
    return section.f32RoleBase(role);
}

fn valueBits(tag: u8, role: Role, token: u32, row: u32) u32 {
    return (@as(u32, tag & 0x0f) << 28) |
        (@as(u32, roleCode(role)) << 26) |
        ((token & 0x03) << 24) |
        (row & 0x3fff);
}

fn pairWord(tag: u8, role: Role, token: u32, even_row: u32) u64 {
    return @as(u64, valueBits(tag, role, token, even_row)) |
        (@as(u64, valueBits(tag, role, token, even_row + 1)) << 32);
}

fn configure(dut: *Dut, role: Role, rows: u32, tokens: u32) !void {
    try std.testing.expect(c.dut_write_config_ready(dut.handle) != 0);
    c.dut_set_write_config(
        dut.handle,
        1,
        roleCode(role),
        @intCast(rows),
        @intCast(tokens),
    );
    dut.step();
    c.dut_set_write_config(dut.handle, 0, 0, 0, 0);
    dut.eval();
}

fn sendMappedBeat(
    dut: *Dut,
    role: Role,
    token: u32,
    even_row: u32,
    data: u64,
    keep: u8,
    last: bool,
) !void {
    try std.testing.expect(c.dut_write_stream_ready(dut.handle) != 0);
    c.dut_set_write_stream(dut.handle, 1, data, keep, @intFromBool(last));
    dut.eval();
    if (keep == 0xff) {
        const loc = try section.f32PhysicalLocation(role, token, even_row);
        try std.testing.expect(c.dut_write_commit_valid(dut.handle) != 0);
        try std.testing.expectEqual(@as(u8, loc.bank), c.dut_write_commit_bank(dut.handle));
        try std.testing.expectEqual(
            @as(u16, @intCast(loc.address)),
            c.dut_write_commit_address(dut.handle),
        );
    } else {
        try std.testing.expect(c.dut_write_commit_valid(dut.handle) == 0);
    }
    dut.step();
    c.dut_set_write_stream(dut.handle, 0, 0, 0, 0);
    dut.eval();
}

fn nextNoise(state: *u64) u64 {
    state.* = state.* *% 6364136223846793005 +% 1442695040888963407;
    return state.*;
}

fn writeRole(
    dut: *Dut,
    role: Role,
    rows: u32,
    tokens: u32,
    tag: u8,
    bubble_seed: u64,
) !usize {
    try configure(dut, role, rows, tokens);
    try std.testing.expect(c.dut_write_busy(dut.handle) != 0);
    try std.testing.expect(c.dut_write_error(dut.handle) == 0);

    var noise = bubble_seed;
    var beats: usize = 0;
    const rowblocks = (rows + 15) / 16;
    for (0..rowblocks) |rb_usize| {
        const rb: u32 = @intCast(rb_usize);
        const row_base = rb * 16;
        const rows_this = @min(@as(u32, 16), rows - row_base);
        const pairs = rows_this / 2;
        for (0..tokens) |token_usize| {
            const token: u32 = @intCast(token_usize);
            for (0..pairs) |pair_usize| {
                const pair: u32 = @intCast(pair_usize);
                // Input bubbles must not advance any address counter.
                if (nextNoise(&noise) % 7 == 0) {
                    c.dut_set_write_stream(dut.handle, 0, nextNoise(&noise), 0x0f, 1);
                    dut.step();
                    try std.testing.expect(c.dut_write_stream_ready(dut.handle) != 0);
                }
                const is_last = rb + 1 == rowblocks and
                    token + 1 == tokens and pair + 1 == pairs;
                try sendMappedBeat(
                    dut,
                    role,
                    token,
                    row_base + pair * 2,
                    pairWord(tag, role, token, row_base + pair * 2),
                    0xff,
                    is_last,
                );
                beats += 1;
            }
        }
    }

    try std.testing.expect(c.dut_write_done(dut.handle) != 0);
    try std.testing.expect(c.dut_write_busy(dut.handle) == 0);
    try std.testing.expect(c.dut_write_stream_ready(dut.handle) == 0);
    try std.testing.expect(c.dut_write_error(dut.handle) == 0);
    return beats;
}

fn readGroup(
    dut: *Dut,
    role: Role,
    token: u32,
    group: u32,
    stall_cycles: usize,
) !ReadResult {
    c.dut_set_read_ready(dut.handle, 0);
    try std.testing.expect(c.dut_read_request_ready(dut.handle) != 0);
    c.dut_set_read_request(
        dut.handle,
        1,
        roleCode(role),
        @intCast(token),
        @intCast(group),
    );
    dut.eval();
    try std.testing.expect(c.dut_read_issue_valid(dut.handle) != 0);
    const group_count = role.rowCapacity() / section.f32_rows_per_group;
    const request_bad = token >= section.query_tile_max or group >= group_count;
    const expected_address: u16 = if (request_bad) 0 else @intCast(roleBase(role) + token * group_count + group);
    try std.testing.expectEqual(expected_address, c.dut_read_issue_address(dut.handle));
    dut.step();
    c.dut_set_read_request(dut.handle, 0, 0, 0, 0);
    try std.testing.expect(c.dut_read_request_ready(dut.handle) == 0);
    try std.testing.expect(c.dut_read_response_valid(dut.handle) == 0);
    dut.step();
    dut.eval();

    try std.testing.expect(c.dut_read_response_valid(dut.handle) != 0);
    var result: ReadResult = .{
        .lanes = undefined,
        .is_error = c.dut_read_response_error(dut.handle) != 0,
    };
    for (0..4) |lane| {
        result.lanes[lane] = c.dut_read_response_lane(dut.handle, @intCast(lane));
    }

    for (0..stall_cycles) |_| {
        try std.testing.expect(c.dut_read_request_ready(dut.handle) == 0);
        dut.step();
        try std.testing.expect(c.dut_read_response_valid(dut.handle) != 0);
        try std.testing.expectEqual(result.is_error, c.dut_read_response_error(dut.handle) != 0);
        for (0..4) |lane| {
            try std.testing.expectEqual(
                result.lanes[lane],
                c.dut_read_response_lane(dut.handle, @intCast(lane)),
            );
        }
    }

    c.dut_set_read_ready(dut.handle, 1);
    dut.step();
    c.dut_set_read_ready(dut.handle, 0);
    dut.eval();
    try std.testing.expect(c.dut_read_response_valid(dut.handle) == 0);
    return result;
}

fn expectGroup(
    dut: *Dut,
    role: Role,
    token: u32,
    group: u32,
    tag: u8,
    stall_cycles: usize,
) !void {
    const got = try readGroup(dut, role, token, group, stall_cycles);
    try std.testing.expect(!got.is_error);
    for (0..4) |lane_usize| {
        const lane: u32 = @intCast(lane_usize);
        const even_row = group * section.f32_rows_per_group + lane * 2;
        const loc = try section.f32Location(role, token, even_row);
        const physical = try section.f32PhysicalLocation(role, token, even_row);
        try std.testing.expectEqual(@as(u2, @intCast(lane)), loc.bank);
        try std.testing.expectEqual(loc.bank, physical.bank);
        try std.testing.expectEqual(
            token * (role.rowCapacity() / section.f32_rows_per_group) + group,
            loc.address,
        );
        try std.testing.expectEqual(roleBase(role) + loc.address, physical.address);
        try std.testing.expectEqual(pairWord(tag, role, token, even_row), got.lanes[lane_usize]);
    }
}

fn verifyRole(dut: *Dut, role: Role, rows: u32, tokens: u32, tag: u8) !usize {
    var groups_checked: usize = 0;
    var noise: u64 = 0x9e37_79b9_7f4a_7c15 ^
        (@as(u64, roleCode(role)) << 40) ^ (@as(u64, tag) << 8);
    const groups = rows / section.f32_rows_per_group;
    for (0..tokens) |token_usize| {
        const token: u32 = @intCast(token_usize);
        for (0..groups) |group_usize| {
            const group: u32 = @intCast(group_usize);
            const stalls: usize = @intCast(nextNoise(&noise) % 7);
            try expectGroup(dut, role, token, group, tag, stalls);
            groups_checked += 1;
        }
    }
    return groups_checked;
}

fn expectBadConfig(dut: *Dut, role: Role, rows: u32, tokens: u32) !void {
    try configure(dut, role, rows, tokens);
    try std.testing.expect(c.dut_write_done(dut.handle) != 0);
    try std.testing.expect(c.dut_write_error(dut.handle) != 0);
    try std.testing.expect(c.dut_write_busy(dut.handle) == 0);
    dut.step();
    try std.testing.expect(c.dut_write_done(dut.handle) == 0);
}

fn testBounds(dut: *Dut) !void {
    try expectBadConfig(dut, .x0, 0, 1);
    try expectBadConfig(dut, .x0, 7, 1);
    try expectBadConfig(dut, .x0, section.ffn_dim_max + 8, 1);
    try expectBadConfig(dut, .residual, section.model_dim_max + 8, 1);
    try expectBadConfig(dut, .x1, 8, 0);
    try expectBadConfig(dut, .x1, 8, section.query_tile_max + 1);

    inline for (.{ Role.residual, Role.x0, Role.x1, Role.x2 }) |role| {
        const group_count = role.rowCapacity() / section.f32_rows_per_group;
        const last_ok = try readGroup(dut, role, section.query_tile_max - 1, group_count - 1, 1);
        try std.testing.expect(!last_ok.is_error);

        const bad_group = try readGroup(dut, role, 0, group_count, 2);
        try std.testing.expect(bad_group.is_error);
        try std.testing.expectEqual([_]u64{ 0, 0, 0, 0 }, bad_group.lanes);

        const bad_token = try readGroup(dut, role, section.query_tile_max, 0, 1);
        try std.testing.expect(bad_token.is_error);
        try std.testing.expectEqual([_]u64{ 0, 0, 0, 0 }, bad_token.lanes);
    }
}

fn testFraming(dut: *Dut) !void {
    // TLAST is diagnostic: consume the fixed geometry despite an early marker
    // and a missing final marker, then finish with an error rather than hanging.
    try configure(dut, .residual, 8, 1);
    for (0..4) |pair| {
        try sendMappedBeat(
            dut,
            .residual,
            0,
            @intCast(pair * 2),
            pairWord(3, .residual, 0, @intCast(pair * 2)),
            0xff,
            pair == 0,
        );
    }
    try std.testing.expect(c.dut_write_done(dut.handle) != 0);
    try std.testing.expect(c.dut_write_error(dut.handle) != 0);
    try expectGroup(dut, .residual, 0, 0, 3, 3);

    // A partial word is diagnosed and is not written.
    try configure(dut, .residual, 8, 1);
    for (0..4) |pair| {
        try sendMappedBeat(
            dut,
            .residual,
            0,
            @intCast(pair * 2),
            pairWord(4, .residual, 0, @intCast(pair * 2)),
            if (pair == 2) 0x0f else 0xff,
            pair == 3,
        );
    }
    try std.testing.expect(c.dut_write_done(dut.handle) != 0);
    try std.testing.expect(c.dut_write_error(dut.handle) != 0);
}

fn testAbort(dut: *Dut) !void {
    try configure(dut, .x1, 16, 1);
    try sendMappedBeat(dut, .x1, 0, 0, pairWord(7, .x1, 0, 0), 0xff, false);

    // Presenting a beat together with abort must not handshake or commit it.
    c.dut_set_write_stream(dut.handle, 1, pairWord(7, .x1, 0, 2), 0xff, 0);
    c.dut_set_write_abort(dut.handle, 1);
    dut.eval();
    try std.testing.expect(c.dut_write_stream_ready(dut.handle) == 0);
    try std.testing.expect(c.dut_write_commit_valid(dut.handle) == 0);
    dut.step();
    c.dut_set_write_abort(dut.handle, 0);
    c.dut_set_write_stream(dut.handle, 0, 0, 0, 0);
    dut.eval();

    try std.testing.expect(c.dut_write_done(dut.handle) != 0);
    try std.testing.expect(c.dut_write_error(dut.handle) != 0);
    try std.testing.expect(c.dut_write_busy(dut.handle) == 0);
}

fn consumeCurrentRead(dut: *Dut, stalls: usize) !ReadResult {
    try std.testing.expect(c.dut_read_response_valid(dut.handle) != 0);
    var result: ReadResult = .{
        .lanes = undefined,
        .is_error = c.dut_read_response_error(dut.handle) != 0,
    };
    for (0..4) |lane| result.lanes[lane] = c.dut_read_response_lane(dut.handle, @intCast(lane));
    for (0..stalls) |_| {
        try std.testing.expect(c.dut_read_request_ready(dut.handle) == 0);
        dut.step();
        try std.testing.expect(c.dut_read_response_valid(dut.handle) != 0);
        try std.testing.expectEqual(result.is_error, c.dut_read_response_error(dut.handle) != 0);
        for (0..4) |lane| {
            try std.testing.expectEqual(result.lanes[lane], c.dut_read_response_lane(dut.handle, @intCast(lane)));
        }
    }
    c.dut_set_read_ready(dut.handle, 1);
    dut.step();
    c.dut_set_read_ready(dut.handle, 0);
    dut.eval();
    return result;
}

fn testSimultaneousPorts(dut: *Dut) !void {
    // Different roles occupy disjoint physical address ranges, so an X0 read
    // may proceed while X2 is written.
    try configure(dut, .x2, 8, 1);
    c.dut_set_read_ready(dut.handle, 0);
    c.dut_set_read_request(dut.handle, 1, roleCode(.x0), 0, 0);
    c.dut_set_write_stream(dut.handle, 1, pairWord(5, .x2, 0, 0), 0xff, 0);
    dut.step();
    c.dut_set_read_request(dut.handle, 0, 0, 0, 0);
    c.dut_set_write_stream(dut.handle, 0, 0, 0, 0);
    dut.step();
    dut.eval();
    const cross_role = try consumeCurrentRead(dut, 4);
    try std.testing.expect(!cross_role.is_error);
    try std.testing.expectEqual(pairWord(1, .x0, 0, 0), cross_role.lanes[0]);
    for (1..4) |pair| {
        try sendMappedBeat(
            dut,
            .x2,
            0,
            @intCast(pair * 2),
            pairWord(5, .x2, 0, @intCast(pair * 2)),
            0xff,
            pair == 3,
        );
    }
    try std.testing.expect(c.dut_write_error(dut.handle) == 0);
    try expectGroup(dut, .x2, 0, 0, 5, 2);

    // Same-group read/write is a schedule error.  It returns a deterministic
    // zero response while allowing the fixed GEMM stream to finish.
    try configure(dut, .x2, 8, 1);
    c.dut_set_read_ready(dut.handle, 0);
    c.dut_set_read_request(dut.handle, 1, roleCode(.x2), 0, 0);
    c.dut_set_write_stream(dut.handle, 1, pairWord(6, .x2, 0, 0), 0xff, 0);
    dut.step();
    c.dut_set_read_request(dut.handle, 0, 0, 0, 0);
    c.dut_set_write_stream(dut.handle, 0, 0, 0, 0);
    dut.step();
    dut.eval();
    const collided = try consumeCurrentRead(dut, 3);
    try std.testing.expect(collided.is_error);
    try std.testing.expectEqual([_]u64{ 0, 0, 0, 0 }, collided.lanes);
    for (1..4) |pair| {
        try sendMappedBeat(
            dut,
            .x2,
            0,
            @intCast(pair * 2),
            pairWord(6, .x2, 0, @intCast(pair * 2)),
            0xff,
            pair == 3,
        );
    }
    try std.testing.expect(c.dut_write_error(dut.handle) == 0);
    try expectGroup(dut, .x2, 0, 0, 6, 1);

    // A write to the requested address one cycle later, when the registered
    // address reaches the memories, is the same invalid schedule and must also
    // return a deterministic error response.
    try configure(dut, .x2, 8, 1);
    c.dut_set_read_ready(dut.handle, 0);
    c.dut_set_read_request(dut.handle, 1, roleCode(.x2), 0, 0);
    dut.step();
    c.dut_set_read_request(dut.handle, 0, 0, 0, 0);
    c.dut_set_write_stream(dut.handle, 1, pairWord(8, .x2, 0, 0), 0xff, 0);
    dut.step();
    c.dut_set_write_stream(dut.handle, 0, 0, 0, 0);
    dut.eval();
    const issue_collided = try consumeCurrentRead(dut, 2);
    try std.testing.expect(issue_collided.is_error);
    try std.testing.expectEqual([_]u64{ 0, 0, 0, 0 }, issue_collided.lanes);
    for (1..4) |pair| {
        try sendMappedBeat(
            dut,
            .x2,
            0,
            @intCast(pair * 2),
            pairWord(8, .x2, 0, @intCast(pair * 2)),
            0xff,
            pair == 3,
        );
    }
    try std.testing.expect(c.dut_write_error(dut.handle) == 0);
    try expectGroup(dut, .x2, 0, 0, 8, 1);
}

pub fn main() !void {
    var dut = Dut.init();
    defer dut.deinit();
    dut.reset();

    // These cumulative bases are the shared role capacities expressed as
    // per-bank addresses.  The end address is exactly 16384 words per bank.
    try std.testing.expectEqual(@as(u32, 0), roleBase(.residual));
    try std.testing.expectEqual(@as(u32, 2048), roleBase(.x0));
    try std.testing.expectEqual(@as(u32, 8192), roleBase(.x1));
    try std.testing.expectEqual(@as(u32, 14336), roleBase(.x2));
    try std.testing.expectEqual(section.f32BankDepth(), roleBase(.x2) + roleSpan(.x2));

    try testBounds(&dut);
    try testFraming(&dut);
    try testAbort(&dut);

    // A successful half-final-rowblock shape complements the malformed framing
    // cases above.  The later full X1 pass overwrites it before isolation checks.
    var beats: usize = 0;
    var groups: usize = 0;
    beats += try writeRole(&dut, .x1, 40, section.query_tile_max, 2, 0x7123_a004);
    groups += try verifyRole(&dut, .x1, 40, section.query_tile_max, 2);

    // Full-capacity passes over all four roles prove every physical word.  The
    // last X0 commit is 8191 immediately before X1; the last X2 commit is 16383.
    beats += try writeRole(&dut, .residual, section.model_dim_max, section.query_tile_max, 8, 0x08cc_2141);
    groups += try verifyRole(&dut, .residual, section.model_dim_max, section.query_tile_max, 8);
    beats += try writeRole(&dut, .x0, section.ffn_dim_max, section.query_tile_max, 1, 0x582d_912a);
    groups += try verifyRole(&dut, .x0, section.ffn_dim_max, section.query_tile_max, 1);
    beats += try writeRole(&dut, .x1, section.ffn_dim_max, section.query_tile_max, 2, 0x7123_a004);
    groups += try verifyRole(&dut, .x1, section.ffn_dim_max, section.query_tile_max, 2);
    beats += try writeRole(&dut, .x2, section.model_dim_max, section.query_tile_max, 9, 0xb418_49a2);
    groups += try verifyRole(&dut, .x2, section.model_dim_max, section.query_tile_max, 9);

    // Neighboring roles retain their extrema after every other role has filled.
    try expectGroup(&dut, .residual, 3, section.model_dim_max / 8 - 1, 8, 5);
    try expectGroup(&dut, .x0, 3, section.ffn_dim_max / 8 - 1, 1, 5);
    try expectGroup(&dut, .x1, 0, 0, 2, 5);
    try expectGroup(&dut, .x2, 3, section.model_dim_max / 8 - 1, 9, 5);

    try testSimultaneousPorts(&dut);

    std.debug.print(
        "\n  section F32 scratch cosim: {d} GEMM beats, {d} 256-bit groups checked\n" ++
            "  X0/X1 no-transpose mapping, roles/bounds, framing, collisions, backpressure: passed\n" ++
            "  storage geometry: 4 x 16384 x 64 = 512 KiB (16 URAM288 target)\n\n",
        .{ beats, groups },
    );
}
