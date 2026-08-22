//! Scratch mapping, exact arithmetic, framing, fault, and restart cosim for the
//! standalone P3d residual-add boundary.

const std = @import("std");
const section = @import("shared_section");
const c = @cImport(@cInclude("shim.h"));

const add_latency: usize = 15;

const WrapperState = struct {
    const add_lo_request: u8 = 4;
    const add_lo_wait: u8 = 5;
    const write: u8 = 8;
    const output: u8 = 9;
};

const AddState = struct {
    const idle: u8 = 0;
    const capture: u8 = 1;
    const align16: u8 = 2;
    const align8: u8 = 3;
    const align4: u8 = 4;
    const align2: u8 = 5;
    const align1: u8 = 6;
    const add: u8 = 7;
    const norm16: u8 = 8;
    const norm8: u8 = 9;
    const norm4: u8 = 10;
    const norm2: u8 = 11;
    const norm1: u8 = 12;
    const round: u8 = 13;
    const final: u8 = 14;
    const result: u8 = 15;
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
        const zeros = [_]u64{0} ** 4;
        c.dut_set_config(self.handle, 0, 0, 0);
        c.dut_set_abort(self.handle, 0);
        c.dut_set_input(self.handle, 0, 0, 0, 0);
        c.dut_set_read_request_ready(self.handle, 0);
        c.dut_set_read_response(self.handle, 0, &zeros, 0);
        c.dut_set_write_sink(self.handle, 0, 0);
        c.dut_set_output_ready(self.handle, 0);
    }

    fn reset(self: *Dut) void {
        c.dut_set_clk(self.handle, 0);
        c.dut_set_rst_n(self.handle, 0);
        self.clearInputs();
        self.eval();
        for (0..4) |_| self.step();
        c.dut_set_rst_n(self.handle, 1);
        self.step();
    }
};

const ScratchModel = struct {
    rows: usize,
    tokens: usize,
    values: []u32,

    fn group(self: ScratchModel, token: usize, group_index: usize) ![4]u64 {
        if (token >= self.tokens or group_index >= self.rows / 8)
            return error.BadScratchRead;
        var lanes: [4]u64 = undefined;
        for (&lanes, 0..) |*word, bank| {
            const row = group_index * 8 + bank * 2;
            const base = token * self.rows + row;
            word.* = @as(u64, self.values[base]) |
                (@as(u64, self.values[base + 1]) << 32);
        }
        return lanes;
    }

    fn commit(
        self: *ScratchModel,
        bank: usize,
        address: usize,
        data: u64,
    ) !void {
        if (bank >= 4 or address >= 2048) return error.BadScratchWrite;
        const token = address / 512;
        const group_index = address % 512;
        if (token >= self.tokens or group_index >= self.rows / 8)
            return error.BadScratchWrite;
        const row = group_index * 8 + bank * 2;
        const base = token * self.rows + row;
        self.values[base] = @truncate(data);
        self.values[base + 1] = @truncate(data >> 32);
    }
};

const NativeLocation = struct {
    token: usize,
    row: usize,
    pair: usize,
    rowblock: usize,
};

fn nativeLocation(tokens: usize, ordinal: usize) NativeLocation {
    const words_per_rowblock = tokens * 8;
    const rowblock = ordinal / words_per_rowblock;
    const within = ordinal % words_per_rowblock;
    const token = within / 8;
    const pair = within % 8;
    return .{
        .token = token,
        .row = rowblock * 16 + pair * 2,
        .pair = pair,
        .rowblock = rowblock,
    };
}

fn nativeWord(values: []const u32, rows: usize, tokens: usize, ordinal: usize) u64 {
    const loc = nativeLocation(tokens, ordinal);
    const base = loc.token * rows + loc.row;
    return @as(u64, values[base]) | (@as(u64, values[base + 1]) << 32);
}

fn finiteBits(index: usize, salt: u32) u32 {
    const i: u32 = @truncate(index);
    const sign = ((i +% salt) & 1) << 31;
    const exponent = 105 + ((i *% 17 +% salt) % 40);
    const fraction = (i *% 0x45d9_f3b +% salt *% 0x119d_e1f) & 0x007f_ffff;
    return sign | (exponent << 23) | fraction;
}

fn configure(dut: *Dut, rows: u16, tokens: u8) !void {
    dut.eval();
    try std.testing.expect(c.dut_config_ready(dut.handle) != 0);
    c.dut_set_config(dut.handle, 1, rows, tokens);
    dut.eval();
    try std.testing.expect(c.dut_config_ready(dut.handle) != 0);
    dut.step();
    c.dut_set_config(dut.handle, 0, 0, 0);
    dut.eval();
}

const PendingResponse = struct {
    lanes: [4]u64,
    delay: usize,
};

const RunStats = struct {
    cycles: usize,
    reads: usize,
    writes: usize,
    outputs: usize,
    input_stalls: usize,
    read_stalls: usize,
    write_stalls: usize,
    output_stalls: usize,
};

fn runSuccess(
    allocator: std.mem.Allocator,
    dut: *Dut,
    rows: usize,
    tokens: usize,
    stalls: bool,
    seed: u64,
) !RunStats {
    const scalar_count = rows * tokens;
    const word_count = scalar_count / 2;
    const read_count = word_count;
    const residual = try allocator.alloc(u32, scalar_count);
    defer allocator.free(residual);
    const down = try allocator.alloc(u32, scalar_count);
    defer allocator.free(down);
    const expected = try allocator.alloc(u32, scalar_count);
    defer allocator.free(expected);

    for (0..scalar_count) |index| {
        residual[index] = finiteBits(index, 0x15);
        down[index] = finiteBits(index, 0x2b);
        const sum = section.residualAddRneBits(residual[index], down[index]);
        try std.testing.expectEqual(@as(u2, 0), sum.status);
        expected[index] = sum.bits;
    }

    var scratch = ScratchModel{
        .rows = rows,
        .tokens = tokens,
        .values = residual,
    };
    try configure(dut, @intCast(rows), @intCast(tokens));
    try std.testing.expect(c.dut_busy(dut.handle) != 0);

    var prng = std.Random.DefaultPrng.init(seed);
    const rnd = prng.random();
    var pending: ?PendingResponse = null;
    var presenting = false;
    var sent: usize = 0;
    var reads: usize = 0;
    var writes: usize = 0;
    var outputs: usize = 0;
    var cycle: usize = 0;
    var input_stalls: usize = 0;
    var read_stalls: usize = 0;
    var write_stalls: usize = 0;
    var output_stalls: usize = 0;
    var held_write: ?u64 = null;
    var held_output: ?u64 = null;

    while (true) {
        if (cycle > word_count * 64 + 4096) return error.StreamTimeout;

        if (!presenting and sent < word_count)
            presenting = !stalls or rnd.uintLessThan(u8, 4) != 0;
        const input_word = if (presenting)
            nativeWord(down, rows, tokens, sent)
        else
            0;
        c.dut_set_input(
            dut.handle,
            @intFromBool(presenting),
            input_word,
            if (presenting) 0xff else 0,
            @intFromBool(presenting and sent + 1 == word_count),
        );

        const req_ready = !stalls or rnd.uintLessThan(u8, 4) != 0;
        c.dut_set_read_request_ready(dut.handle, @intFromBool(req_ready));
        const zeros = [_]u64{0} ** 4;
        if (pending) |response| {
            c.dut_set_read_response(
                dut.handle,
                @intFromBool(response.delay == 0),
                &response.lanes,
                0,
            );
        } else {
            c.dut_set_read_response(dut.handle, 0, &zeros, 0);
        }

        const write_ready = !stalls or rnd.uintLessThan(u8, 4) != 0;
        const output_ready = !stalls or rnd.uintLessThan(u8, 4) != 0;
        c.dut_set_write_sink(dut.handle, @intFromBool(write_ready), 0);
        c.dut_set_output_ready(dut.handle, @intFromBool(output_ready));
        dut.eval();

        const input_ready = c.dut_input_ready(dut.handle) != 0;
        const input_fire = presenting and input_ready;
        if (presenting and !input_ready) input_stalls += 1;

        const req_valid = c.dut_read_request_valid(dut.handle) != 0;
        const req_fire = req_valid and req_ready;
        if (req_valid and !req_ready) read_stalls += 1;
        if (req_fire) {
            try std.testing.expect(pending == null);
            try std.testing.expectEqual(@as(u8, 0), c.dut_read_request_role(dut.handle));
            const loc = nativeLocation(tokens, reads);
            const expected_token = loc.token;
            const expected_group = loc.row / 8;
            try std.testing.expectEqual(
                @as(u8, @intCast(expected_token)),
                c.dut_read_request_token(dut.handle),
            );
            try std.testing.expectEqual(
                @as(u16, @intCast(expected_group)),
                c.dut_read_request_group(dut.handle),
            );
            pending = .{
                .lanes = try scratch.group(expected_token, expected_group),
                .delay = if (stalls) rnd.uintLessThan(u8, 4) else 0,
            };
            reads += 1;
        }

        const response_valid = pending != null and pending.?.delay == 0;
        const response_fire = response_valid and
            c.dut_read_response_ready(dut.handle) != 0;

        const write_valid = c.dut_write_valid(dut.handle) != 0;
        const write_fire = write_valid and write_ready;
        if (write_valid) {
            const got = c.dut_write_data(dut.handle);
            const expected_word = nativeWord(expected, rows, tokens, writes);
            try std.testing.expectEqual(expected_word, got);
            const loc = nativeLocation(tokens, writes);
            try std.testing.expectEqual(
                @as(u8, @intCast(loc.pair % 4)),
                c.dut_write_bank(dut.handle),
            );
            try std.testing.expectEqual(
                @as(u16, @intCast(loc.token * 512 + loc.row / 8)),
                c.dut_write_address(dut.handle),
            );
            if (!write_ready) {
                write_stalls += 1;
                if (held_write) |held| try std.testing.expectEqual(held, got);
                held_write = got;
            } else {
                held_write = null;
            }
        } else {
            held_write = null;
        }
        if (write_fire) {
            try scratch.commit(
                c.dut_write_bank(dut.handle),
                c.dut_write_address(dut.handle),
                c.dut_write_data(dut.handle),
            );
            writes += 1;
        }

        const output_valid = c.dut_output_valid(dut.handle) != 0;
        const output_fire = output_valid and output_ready;
        if (output_valid) {
            try std.testing.expect(outputs < writes);
            const got = c.dut_output_data(dut.handle);
            try std.testing.expectEqual(
                nativeWord(expected, rows, tokens, outputs),
                got,
            );
            try std.testing.expectEqual(@as(u8, 0xff), c.dut_output_keep(dut.handle));
            try std.testing.expectEqual(
                outputs + 1 == word_count,
                c.dut_output_last(dut.handle) != 0,
            );
            if (!output_ready) {
                output_stalls += 1;
                if (held_output) |held| try std.testing.expectEqual(held, got);
                held_output = got;
            } else {
                held_output = null;
            }
        } else {
            held_output = null;
        }
        if (output_fire) outputs += 1;

        if (input_fire) {
            sent += 1;
            presenting = false;
        }

        dut.step();
        cycle += 1;
        if (response_fire) {
            pending = null;
        } else if (pending) |*response| {
            if (response.delay != 0) response.delay -= 1;
        }

        try std.testing.expect(c.dut_error(dut.handle) == 0);
        if (c.dut_done(dut.handle) != 0) break;
    }

    try std.testing.expectEqual(word_count, sent);
    try std.testing.expectEqual(read_count, reads);
    try std.testing.expectEqual(word_count, writes);
    try std.testing.expectEqual(word_count, outputs);
    try std.testing.expect(pending == null);
    try std.testing.expect(c.dut_busy(dut.handle) == 0);
    try std.testing.expectEqualSlices(u32, expected, scratch.values);
    dut.clearInputs();
    dut.step();
    return .{
        .cycles = cycle,
        .reads = reads,
        .writes = writes,
        .outputs = outputs,
        .input_stalls = input_stalls,
        .read_stalls = read_stalls,
        .write_stalls = write_stalls,
        .output_stalls = output_stalls,
    };
}

fn waitDone(dut: *Dut) !void {
    for (0..64) |_| {
        dut.step();
        if (c.dut_done(dut.handle) != 0) return;
    }
    return error.DoneTimeout;
}

fn expectConfigFault(dut: *Dut, rows: u16, tokens: u8) !void {
    try configure(dut, rows, tokens);
    try std.testing.expect(c.dut_done(dut.handle) != 0);
    try std.testing.expect(c.dut_error(dut.handle) != 0);
    try std.testing.expectEqual(
        section.ResidualAddStatus.bad_cfg,
        @as(u7, @truncate(c.dut_status(dut.handle))),
    );
    dut.step();
}

fn primeInput(dut: *Dut, lanes: [4]u64, response_error: bool) !void {
    c.dut_set_read_request_ready(dut.handle, 1);
    dut.eval();
    try std.testing.expect(c.dut_read_request_valid(dut.handle) != 0);
    dut.step();
    c.dut_set_read_request_ready(dut.handle, 0);
    c.dut_set_read_response(
        dut.handle,
        1,
        &lanes,
        @intFromBool(response_error),
    );
    dut.eval();
    try std.testing.expect(c.dut_read_response_ready(dut.handle) != 0);
    dut.step();
    const zeros = [_]u64{0} ** 4;
    c.dut_set_read_response(dut.handle, 0, &zeros, 0);
    dut.eval();
}

fn sendFirstWord(dut: *Dut, data: u64, keep: u8, last: bool) !void {
    c.dut_set_input(dut.handle, 1, data, keep, @intFromBool(last));
    dut.eval();
    try std.testing.expect(c.dut_input_ready(dut.handle) != 0);
    dut.step();
    c.dut_set_input(dut.handle, 0, 0, 0, 0);
    dut.eval();
}

fn advanceToWrite(dut: *Dut, residual: [4]u64, down: u64) !void {
    try configure(dut, 128, 1);
    try primeInput(dut, residual, false);
    try sendFirstWord(dut, down, 0xff, false);
    for (0..2 * add_latency + 8) |_| {
        if (c.dut_write_valid(dut.handle) != 0) return;
        dut.step();
    }
    return error.WriteTimeout;
}

fn expectFault(dut: *Dut, expected: u7) !void {
    if (c.dut_done(dut.handle) == 0) try waitDone(dut);
    try std.testing.expect(c.dut_error(dut.handle) != 0);
    try std.testing.expectEqual(
        expected,
        @as(u7, @truncate(c.dut_status(dut.handle))),
    );
    dut.clearInputs();
    dut.step();
    try std.testing.expect(c.dut_config_ready(dut.handle) != 0);
}

fn verifyFaults(dut: *Dut) !void {
    const zeros = [_]u64{0} ** 4;
    try expectConfigFault(dut, 0, 1);
    try expectConfigFault(dut, 64, 1);
    try expectConfigFault(dut, 136, 1); // Divisible by eight, but not power-of-two.
    try expectConfigFault(dut, 4104, 1);
    try expectConfigFault(dut, 128, 0);
    try expectConfigFault(dut, 128, 5);

    try configure(dut, 128, 1);
    try primeInput(dut, zeros, false);
    try sendFirstWord(dut, 0, 0xff, true);
    try expectFault(dut, section.ResidualAddStatus.frame);

    try configure(dut, 128, 1);
    try primeInput(dut, zeros, false);
    try sendFirstWord(dut, 0, 0x0f, false);
    try expectFault(dut, section.ResidualAddStatus.frame);

    try configure(dut, 128, 1);
    try primeInput(dut, zeros, true);
    try expectFault(dut, section.ResidualAddStatus.scratch_read);

    try advanceToWrite(dut, zeros, 0);
    c.dut_set_write_sink(dut.handle, 1, 1);
    dut.eval();
    try std.testing.expect(c.dut_write_valid(dut.handle) != 0);
    dut.step();
    c.dut_set_write_sink(dut.handle, 0, 0);
    try expectFault(dut, section.ResidualAddStatus.scratch_write);

    var nonfinite = zeros;
    nonfinite[0] = 0x0000_0000_7f80_0000;
    try configure(dut, 128, 1);
    try primeInput(dut, nonfinite, false);
    try sendFirstWord(dut, 0, 0xff, false);
    try expectFault(
        dut,
        section.ResidualAddStatus.arithmetic(
            section.ResidualAddArithmeticStatus.nonfinite_input,
        ),
    );

    const maximums = [_]u64{
        0x7f7f_ffff_7f7f_ffff,
        0,
        0,
        0,
    };
    try configure(dut, 128, 1);
    try primeInput(dut, maximums, false);
    try sendFirstWord(dut, 0x7f7f_ffff_7f7f_ffff, 0xff, false);
    try expectFault(
        dut,
        section.ResidualAddStatus.arithmetic(
            section.ResidualAddArithmeticStatus.overflow,
        ),
    );
}

fn verifyAbortOutstandingRead(dut: *Dut) !void {
    const zeros = [_]u64{0} ** 4;
    try configure(dut, 128, 1);
    c.dut_set_read_request_ready(dut.handle, 1);
    dut.eval();
    try std.testing.expect(c.dut_read_request_valid(dut.handle) != 0);
    dut.step();
    c.dut_set_read_request_ready(dut.handle, 0);

    c.dut_set_abort(dut.handle, 1);
    dut.eval();
    try std.testing.expect(c.dut_read_request_valid(dut.handle) == 0);
    try std.testing.expect(c.dut_input_ready(dut.handle) == 0);
    dut.step();
    try std.testing.expect(c.dut_busy(dut.handle) != 0);
    try std.testing.expect(c.dut_done(dut.handle) == 0);

    c.dut_set_abort(dut.handle, 0);
    c.dut_set_read_response(dut.handle, 1, &zeros, 0);
    dut.eval();
    try std.testing.expect(c.dut_read_response_ready(dut.handle) != 0);
    dut.step();
    c.dut_set_read_response(dut.handle, 0, &zeros, 0);
    dut.step();
    try std.testing.expect(c.dut_config_ready(dut.handle) != 0);
    try std.testing.expect(c.dut_done(dut.handle) == 0);
    try std.testing.expect(c.dut_error(dut.handle) == 0);
}

fn abortCurrentRun(dut: *Dut) !void {
    c.dut_set_abort(dut.handle, 1);
    dut.eval();
    try std.testing.expect(c.dut_read_request_valid(dut.handle) == 0);
    try std.testing.expect(c.dut_input_ready(dut.handle) == 0);
    try std.testing.expect(c.dut_write_valid(dut.handle) == 0);
    try std.testing.expect(c.dut_output_valid(dut.handle) == 0);
    dut.step();

    dut.clearInputs();
    dut.eval();
    for (0..4) |_| {
        if (c.dut_config_ready(dut.handle) != 0) break;
        dut.step();
    }
    try std.testing.expect(c.dut_config_ready(dut.handle) != 0);
    try std.testing.expect(c.dut_done(dut.handle) == 0);
    try std.testing.expect(c.dut_error(dut.handle) == 0);
}

fn verifyAbortPhases(dut: *Dut) !usize {
    const zeros = [_]u64{0} ** 4;

    // Quarantine an abort in the wrapper request state before the leaf accepts.
    try configure(dut, 128, 1);
    try primeInput(dut, zeros, false);
    try sendFirstWord(dut, 0, 0xff, false);
    try std.testing.expectEqual(
        WrapperState.add_lo_request,
        c.dut_debug_state(dut.handle),
    );
    try std.testing.expectEqual(AddState.idle, c.dut_debug_add_state(dut.handle));
    try abortCurrentRun(dut);

    const add_states = [_]u8{
        AddState.capture,
        AddState.align16,
        AddState.align8,
        AddState.align4,
        AddState.align2,
        AddState.align1,
        AddState.add,
        AddState.norm16,
        AddState.norm8,
        AddState.norm4,
        AddState.norm2,
        AddState.norm1,
        AddState.round,
        AddState.final,
        AddState.result,
    };
    for (add_states) |target_state| {
        try configure(dut, 128, 1);
        try primeInput(dut, zeros, false);
        try sendFirstWord(dut, 0x3f80_0000_3f80_0000, 0xff, false);
        try std.testing.expectEqual(
            WrapperState.add_lo_request,
            c.dut_debug_state(dut.handle),
        );
        dut.step();
        try std.testing.expectEqual(
            WrapperState.add_lo_wait,
            c.dut_debug_state(dut.handle),
        );
        for (0..15) |_| {
            if (c.dut_debug_add_state(dut.handle) == target_state) break;
            dut.step();
        }
        try std.testing.expectEqual(target_state, c.dut_debug_add_state(dut.handle));
        try std.testing.expectEqual(
            WrapperState.add_lo_wait,
            c.dut_debug_state(dut.handle),
        );
        try abortCurrentRun(dut);
    }

    // A write that was ready before abort must not handshake in the abort cycle.
    try advanceToWrite(dut, zeros, 0);
    try std.testing.expectEqual(WrapperState.write, c.dut_debug_state(dut.handle));
    c.dut_set_write_sink(dut.handle, 1, 0);
    dut.eval();
    try std.testing.expect(c.dut_write_valid(dut.handle) != 0);
    try abortCurrentRun(dut);

    // Commit one tentative R word, then abort before that word can publish.
    try advanceToWrite(dut, zeros, 0);
    c.dut_set_write_sink(dut.handle, 1, 0);
    dut.eval();
    try std.testing.expect(c.dut_write_valid(dut.handle) != 0);
    dut.step();
    c.dut_set_write_sink(dut.handle, 0, 0);
    dut.eval();
    try std.testing.expectEqual(WrapperState.output, c.dut_debug_state(dut.handle));
    try std.testing.expect(c.dut_output_valid(dut.handle) != 0);
    c.dut_set_output_ready(dut.handle, 1);
    try abortCurrentRun(dut);

    return 18;
}

fn verifyOrphanCollisions(dut: *Dut) !void {
    const zeros = [_]u64{0} ** 4;

    // ST_REQ: an orphan response must suppress a same-cycle new request.
    try configure(dut, 128, 1);
    c.dut_set_read_request_ready(dut.handle, 1);
    c.dut_set_read_response(dut.handle, 1, &zeros, 0);
    dut.eval();
    try std.testing.expect(c.dut_read_request_valid(dut.handle) == 0);
    try std.testing.expect(c.dut_read_response_ready(dut.handle) != 0);
    dut.step();
    c.dut_set_read_request_ready(dut.handle, 0);
    c.dut_set_read_response(dut.handle, 0, &zeros, 0);
    try expectFault(dut, section.ResidualAddStatus.internal);

    // ST_INPUT: ingress is fail-closed in the raw orphan detection cycle.
    try configure(dut, 128, 1);
    try primeInput(dut, zeros, false);
    c.dut_set_input(dut.handle, 1, 0, 0xff, 0);
    c.dut_set_read_response(dut.handle, 1, &zeros, 0);
    dut.eval();
    try std.testing.expect(c.dut_input_ready(dut.handle) == 0);
    try std.testing.expect(c.dut_read_response_ready(dut.handle) != 0);
    try std.testing.expect(c.dut_write_valid(dut.handle) == 0);
    try std.testing.expect(c.dut_output_valid(dut.handle) == 0);
    dut.step();
    dut.clearInputs();
    try expectFault(dut, section.ResidualAddStatus.internal);

    // ST_WRITE: the already-computed word may commit tentatively on the
    // diagnosis edge, but the priority fault must prevent publication.
    try advanceToWrite(dut, zeros, 0);
    c.dut_set_write_sink(dut.handle, 1, 0);
    c.dut_set_read_response(dut.handle, 1, &zeros, 0);
    dut.eval();
    try std.testing.expect(c.dut_write_valid(dut.handle) != 0);
    try std.testing.expect(c.dut_output_valid(dut.handle) == 0);
    try std.testing.expect(c.dut_read_response_ready(dut.handle) != 0);
    dut.step();
    dut.clearInputs();
    try expectFault(dut, section.ResidualAddStatus.internal);

    // ST_OUTPUT: a previously committed word remains tentative, but it is not
    // published and the R role cannot seal.
    try advanceToWrite(dut, zeros, 0);
    c.dut_set_write_sink(dut.handle, 1, 0);
    dut.eval();
    try std.testing.expect(c.dut_write_valid(dut.handle) != 0);
    dut.step();
    c.dut_set_write_sink(dut.handle, 0, 0);
    dut.eval();
    try std.testing.expect(c.dut_output_valid(dut.handle) != 0);
    c.dut_set_output_ready(dut.handle, 1);
    c.dut_set_read_response(dut.handle, 1, &zeros, 0);
    dut.eval();
    try std.testing.expect(c.dut_output_valid(dut.handle) == 0);
    try std.testing.expect(c.dut_read_response_ready(dut.handle) != 0);
    dut.step();
    dut.clearInputs();
    try expectFault(dut, section.ResidualAddStatus.internal);

    // ST_ADD_LO_REQ: orphan detection suppresses the arithmetic request too.
    try configure(dut, 128, 1);
    try primeInput(dut, zeros, false);
    try sendFirstWord(dut, 0, 0xff, false);
    try std.testing.expectEqual(
        WrapperState.add_lo_request,
        c.dut_debug_state(dut.handle),
    );
    try std.testing.expectEqual(AddState.idle, c.dut_debug_add_state(dut.handle));
    c.dut_set_read_response(dut.handle, 1, &zeros, 0);
    dut.eval();
    try std.testing.expect(c.dut_read_response_ready(dut.handle) != 0);
    try std.testing.expect(c.dut_input_ready(dut.handle) == 0);
    try std.testing.expect(c.dut_write_valid(dut.handle) == 0);
    try std.testing.expect(c.dut_output_valid(dut.handle) == 0);
    dut.step();
    try std.testing.expectEqual(AddState.idle, c.dut_debug_add_state(dut.handle));
    dut.clearInputs();
    try expectFault(dut, section.ResidualAddStatus.internal);

    // ST_ADD_LO_WAIT/result: the orphan cycle cannot consume a held sum.
    try configure(dut, 128, 1);
    try primeInput(dut, zeros, false);
    try sendFirstWord(dut, 0, 0xff, false);
    dut.step();
    for (0..15) |_| {
        if (c.dut_debug_add_state(dut.handle) == AddState.result) break;
        dut.step();
    }
    try std.testing.expectEqual(AddState.result, c.dut_debug_add_state(dut.handle));
    try std.testing.expectEqual(
        WrapperState.add_lo_wait,
        c.dut_debug_state(dut.handle),
    );
    c.dut_set_read_response(dut.handle, 1, &zeros, 0);
    dut.eval();
    try std.testing.expect(c.dut_read_response_ready(dut.handle) != 0);
    try std.testing.expect(c.dut_write_valid(dut.handle) == 0);
    try std.testing.expect(c.dut_output_valid(dut.handle) == 0);
    dut.step();
    dut.clearInputs();
    try expectFault(dut, section.ResidualAddStatus.internal);
}

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    var dut = Dut.init();
    defer dut.deinit();
    dut.reset();

    const stalled = try runSuccess(
        allocator,
        &dut,
        128,
        4,
        true,
        0x99e1_074c_32af_5bd6,
    );
    try std.testing.expect(stalled.input_stalls > 0);
    try std.testing.expect(stalled.read_stalls > 0);
    try std.testing.expect(stalled.write_stalls > 0);
    try std.testing.expect(stalled.output_stalls > 0);

    const legal_rows = [_]usize{ 128, 256, 512, 1024, 2048, 4096 };
    var clean_cycles: usize = 0;
    var clean_reads: usize = 0;
    var clean_writes: usize = 0;
    var clean_outputs: usize = 0;
    var maximum: RunStats = undefined;
    for (legal_rows) |rows| {
        for (1..5) |tokens| {
            const clean = try runSuccess(
                allocator,
                &dut,
                rows,
                tokens,
                false,
                0,
            );
            clean_cycles += clean.cycles;
            clean_reads += clean.reads;
            clean_writes += clean.writes;
            clean_outputs += clean.outputs;
            if (rows == 4096 and tokens == 4) maximum = clean;
        }
    }
    try std.testing.expectEqual(@as(usize, 40_320), clean_reads);
    try std.testing.expectEqual(@as(usize, 40_320), clean_writes);
    try std.testing.expectEqual(clean_writes, clean_outputs);
    try std.testing.expectEqual(@as(usize, 8192), maximum.reads);
    try std.testing.expectEqual(@as(usize, 8192), maximum.writes);
    try std.testing.expectEqual(@as(usize, 8192), maximum.outputs);

    try verifyFaults(&dut);
    try verifyAbortOutstandingRead(&dut);
    const abort_phases = try verifyAbortPhases(&dut);
    try verifyOrphanCollisions(&dut);

    // The final restart proves no aborted/faulted read or arithmetic result can
    // be mistaken for a new R epoch.
    const restarted = try runSuccess(allocator, &dut, 128, 1, false, 0);

    std.debug.print(
        "section residual add: 24 clean shapes {d} cycles, " ++
            "{d} reads/{d} writes; 4096x4 {d} cycles; " ++
            "stalls i/r/w/o={d}/{d}/{d}/{d}; restart {d} cycles; " ++
            "exact RNE, native ordering, R seal, faults, orphan gating, and " ++
            "{d} abort phases, abort/drain passed\n",
        .{
            clean_cycles,
            clean_reads,
            clean_writes,
            maximum.cycles,
            stalled.input_stalls,
            stalled.read_stalls,
            stalled.write_stalls,
            stalled.output_stalls,
            restarted.cycles,
            abort_phases,
        },
    );
}
