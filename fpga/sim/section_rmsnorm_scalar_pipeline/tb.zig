//! Real-RTL cosim for the reusable weighted RMSNorm scalar pipeline.
//!
//! External mode loads R while resident mode never accepts or writes residual
//! traffic. Both modes are checked against the fixed reduction/inverse and
//! exact two-multiply weighted software oracle.

const std = @import("std");
const section = @import("shared_section");
const c = @cImport(@cInclude("shim.h"));

const eps_1e_6: u32 = 0x3586_37bd;
const rows_set = [_]usize{ 128, 256, 512, 1024, 2048, 4096 };
const zero_lanes = [_]u64{ 0, 0, 0, 0 };

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

const Pending = struct {
    request: Request,
    due: usize,
    inject_error: bool,
};

const ScalarBeat = struct {
    data: u32,
    last: bool,
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

    fn clear(self: *Dut) void {
        c.dut_set_abort(self.handle, 0);
        c.dut_set_gamma_config(self.handle, 0, 0);
        c.dut_set_gamma_stream(self.handle, 0, 0, 0, 0);
        c.dut_set_run_config(self.handle, 0, 0, 0, 0, 0);
        c.dut_set_residual_stream(self.handle, 0, 0, 0, 0);
        c.dut_set_r_write_sink(self.handle, 0, 0);
        c.dut_set_read_request_ready(self.handle, 0);
        c.dut_set_read_response(self.handle, 0, &zero_lanes, 0);
        c.dut_set_scalar_ready(self.handle, 0);
        self.eval();
    }

    fn reset(self: *Dut) void {
        c.dut_set_clk(self.handle, 0);
        c.dut_set_rst_n(self.handle, 0);
        self.clear();
        for (0..5) |_| self.step();
        c.dut_set_rst_n(self.handle, 1);
        self.step();
    }
};

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

fn wordAt(values: []const u32, word: usize) u64 {
    return @as(u64, values[word * 2]) |
        (@as(u64, values[word * 2 + 1]) << 32);
}

fn requestAt(dut: *Dut) Request {
    return .{
        .role = @truncate(c.dut_read_request_role(dut.handle)),
        .token = @truncate(c.dut_read_request_token(dut.handle)),
        .group = @truncate(c.dut_read_request_group(dut.handle)),
    };
}

fn populateResident(
    scratch: *Scratch,
    values: []const u32,
    rows: usize,
) !void {
    for (0..values.len / 2) |word| {
        const token_word = word % (rows / 2);
        const token = word / (rows / 2);
        try scratch.write(
            @intCast(token_word % 4),
            @intCast(token * 512 + token_word / 4),
            wordAt(values, word),
        );
    }
}

fn loadGamma(dut: *Dut, gamma: []const u32) !void {
    try std.testing.expect(c.dut_gamma_config_ready(dut.handle) != 0);
    c.dut_set_gamma_config(dut.handle, 1, @intCast(gamma.len));
    dut.eval();
    try std.testing.expect(c.dut_gamma_config_ready(dut.handle) != 0);
    dut.step();
    c.dut_set_gamma_config(dut.handle, 0, 0);
    for (0..gamma.len / 2) |word| {
        c.dut_set_gamma_stream(
            dut.handle,
            1,
            wordAt(gamma, word),
            0xff,
            @intFromBool(word + 1 == gamma.len / 2),
        );
        dut.eval();
        try std.testing.expect(c.dut_gamma_stream_ready(dut.handle) != 0);
        dut.step();
    }
    c.dut_set_gamma_stream(dut.handle, 0, 0, 0, 0);
    dut.eval();
    try std.testing.expect(c.dut_gamma_done(dut.handle) != 0);
    try std.testing.expect(c.dut_gamma_error(dut.handle) == 0);
    try std.testing.expectEqual(@as(u8, 0), c.dut_gamma_status(dut.handle));
    try std.testing.expect(c.dut_gamma_valid(dut.handle) != 0);
    dut.step();
}

fn configureRun(
    dut: *Dut,
    rows: usize,
    tokens: usize,
    resident: bool,
) !void {
    try std.testing.expect(c.dut_run_config_ready(dut.handle) != 0);
    c.dut_set_run_config(
        dut.handle,
        1,
        @intCast(rows),
        @intCast(tokens),
        eps_1e_6,
        @intFromBool(resident),
    );
    dut.eval();
    try std.testing.expect(c.dut_run_config_ready(dut.handle) != 0);
    dut.step();
    c.dut_set_run_config(dut.handle, 0, 0, 0, 0, 0);
    dut.eval();
}

fn inverses(
    values: []const u32,
    rows: usize,
    tokens: usize,
) ![4]u32 {
    var result = [_]u32{0} ** 4;
    for (0..tokens) |token| {
        const token_values = values[token * rows ..][0..rows];
        const scan = try section.rmsNormMaxExp(token_values, @intCast(rows));
        const sum = try section.rmsNormSumsqFixed(
            token_values,
            @intCast(rows),
            scan.max_exp,
        );
        result[token] = (try section.rmsNormInvFixed(
            sum.sum_sq,
            @intCast(rows),
            scan.max_exp,
            eps_1e_6,
        )).inv_rms_bits;
    }
    return result;
}

const RunOptions = struct {
    resident: bool,
    read_error_request: ?usize = null,
    expected_status: ?u23 = null,
};

const RunStats = struct {
    cycles: usize,
    writes: usize,
    requests: usize,
    scalars: usize,
};

fn runPipeline(
    dut: *Dut,
    scratch: *Scratch,
    values: []const u32,
    gamma: []const u32,
    rows: usize,
    tokens: usize,
    options: RunOptions,
    seed: u64,
) !RunStats {
    if (options.resident)
        try populateResident(scratch, values, rows);
    const inv = if (options.expected_status == null)
        try inverses(values, rows, tokens)
    else
        [_]u32{0} ** 4;
    try configureRun(dut, rows, tokens, options.resident);

    var prng = std.Random.DefaultPrng.init(seed);
    const random = prng.random();
    const groups = rows / 8 * tokens;
    const reduce_phase_requests = groups *
        @as(usize, if (options.resident) 2 else 1);
    var word: usize = 0;
    var presenting = false;
    var pending: ?Pending = null;
    var response_active = false;
    var response_lanes = zero_lanes;
    var response_error = false;
    var request_count: usize = 0;
    var response_count: usize = 0;
    var write_count: usize = 0;
    var scalar_count: usize = 0;
    var held_scalar: ?ScalarBeat = null;
    var cycle: usize = 0;

    while (c.dut_done(dut.handle) == 0) : (cycle += 1) {
        if (cycle > rows * tokens * 160 + 32768)
            return error.PipelineTimeout;

        if (!options.resident and !presenting and word < values.len / 2 and
            random.uintLessThan(u8, 4) != 0)
            presenting = true;
        c.dut_set_residual_stream(
            dut.handle,
            @intFromBool(presenting),
            if (presenting) wordAt(values, word) else 0,
            0xff,
            @intFromBool(presenting and word + 1 == values.len / 2),
        );
        const write_ready = random.uintLessThan(u8, 4) != 0;
        c.dut_set_r_write_sink(dut.handle, @intFromBool(write_ready), 0);

        if (!response_active) {
            if (pending) |item| {
                if (cycle >= item.due) {
                    response_lanes = try scratch.read(
                        item.request.token,
                        item.request.group,
                    );
                    response_error = item.inject_error;
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
            random.uintLessThan(u8, 4) != 0;
        c.dut_set_read_request_ready(dut.handle, @intFromBool(request_ready));
        const scalar_ready = random.uintLessThan(u8, 4) != 0;
        c.dut_set_scalar_ready(dut.handle, @intFromBool(scalar_ready));
        dut.eval();

        const stream_fire = presenting and
            c.dut_residual_stream_ready(dut.handle) != 0;
        const write_fire = c.dut_r_write_valid(dut.handle) != 0 and write_ready;
        try std.testing.expectEqual(stream_fire, write_fire);
        if (options.resident) {
            try std.testing.expect(c.dut_residual_stream_ready(dut.handle) == 0);
            try std.testing.expect(c.dut_r_write_valid(dut.handle) == 0);
        }
        if (write_fire) {
            const token_word = word % (rows / 2);
            const token = word / (rows / 2);
            try std.testing.expectEqual(
                @as(u8, @intCast(token_word % 4)),
                c.dut_r_write_bank(dut.handle),
            );
            try std.testing.expectEqual(
                @as(u16, @intCast(token * 512 + token_word / 4)),
                c.dut_r_write_address(dut.handle),
            );
            try scratch.write(
                c.dut_r_write_bank(dut.handle),
                c.dut_r_write_address(dut.handle),
                c.dut_r_write_data(dut.handle),
            );
            write_count += 1;
        }

        const request_fire = c.dut_read_request_valid(dut.handle) != 0 and
            request_ready;
        if (request_fire) {
            const request = requestAt(dut);
            try std.testing.expectEqual(@as(u2, 0), request.role);
            const local = if (request_count < reduce_phase_requests)
                request_count % groups
            else
                request_count - reduce_phase_requests;
            try std.testing.expectEqual(
                @as(u3, @intCast(local / (rows / 8))),
                request.token,
            );
            try std.testing.expectEqual(
                @as(u11, @intCast(local % (rows / 8))),
                request.group,
            );
            pending = .{
                .request = request,
                .due = cycle + 1 + random.uintLessThan(usize, 4),
                .inject_error = options.read_error_request == request_count,
            };
            request_count += 1;
        }
        const response_fire = response_active and
            c.dut_read_response_ready(dut.handle) != 0;

        const scalar_valid = c.dut_scalar_valid(dut.handle) != 0;
        if (scalar_valid) {
            const current = ScalarBeat{
                .data = c.dut_scalar_data(dut.handle),
                .last = c.dut_scalar_last(dut.handle) != 0,
            };
            if (held_scalar) |prior|
                try std.testing.expectEqual(prior, current);
            try std.testing.expectEqual(@as(u8, 0), c.dut_scalar_status(dut.handle));
            if (options.expected_status == null) {
                const token = scalar_count / rows;
                const row = scalar_count % rows;
                const expected = section.rmsNormWeightedBits(
                    values[token * rows + row],
                    inv[token],
                    gamma[row],
                );
                try std.testing.expectEqual(@as(u2, 0), expected.mul1_status);
                try std.testing.expectEqual(@as(u2, 0), expected.mul2_status);
                try std.testing.expectEqual(expected.bits, current.data);
                try std.testing.expectEqual(row % 32 == 31, current.last);
            }
            if (!scalar_ready)
                held_scalar = current
            else
                held_scalar = null;
        } else {
            held_scalar = null;
        }
        const scalar_fire = scalar_valid and scalar_ready;
        try std.testing.expectEqual(
            scalar_fire and scalar_count + 1 == rows * tokens,
            c.dut_debug_final_output_fire(dut.handle) != 0,
        );

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
        if (scalar_fire)
            scalar_count += 1;
    }

    try std.testing.expectEqual(options.expected_status != null, c.dut_error(dut.handle) != 0);
    try std.testing.expectEqual(options.expected_status orelse 0, @as(u23, @truncate(c.dut_status(dut.handle))));
    try std.testing.expectEqual(request_count, response_count);
    if (options.expected_status == null) {
        try std.testing.expectEqual(rows * tokens, scalar_count);
        try std.testing.expectEqual(groups *
            @as(usize, if (options.resident) 3 else 2), request_count);
        try std.testing.expectEqual(
            if (options.resident) 0 else values.len / 2,
            write_count,
        );
        try std.testing.expect(c.dut_gamma_valid(dut.handle) != 0);
    } else {
        try std.testing.expectEqual(@as(usize, 0), scalar_count);
        try std.testing.expect(c.dut_gamma_valid(dut.handle) == 0);
    }
    dut.clear();
    dut.step();
    return .{
        .cycles = cycle + 1,
        .writes = write_count,
        .requests = request_count,
        .scalars = scalar_count,
    };
}

fn abortResidentPhase(
    dut: *Dut,
    scratch: *Scratch,
    values: []const u32,
    gamma: []const u32,
    target_request: usize,
    expected_frontend_state: u8,
) !void {
    try populateResident(scratch, values, 128);
    try configureRun(dut, 128, 1, true);
    var pending: ?Request = null;
    var response_active = false;
    var response_lanes = zero_lanes;
    var requests: usize = 0;
    var cycles: usize = 0;

    while (pending == null or requests <= target_request) : (cycles += 1) {
        if (cycles > 4096) return error.AbortTargetTimeout;
        c.dut_set_residual_stream(dut.handle, 1, wordAt(values, 0), 0xff, 0);
        c.dut_set_r_write_sink(dut.handle, 1, 0);
        c.dut_set_scalar_ready(dut.handle, 1);
        if (response_active)
            c.dut_set_read_response(dut.handle, 1, &response_lanes, 0)
        else
            c.dut_set_read_response(dut.handle, 0, &zero_lanes, 0);
        const ready = pending == null and !response_active;
        c.dut_set_read_request_ready(dut.handle, @intFromBool(ready));
        dut.eval();
        try std.testing.expect(c.dut_residual_stream_ready(dut.handle) == 0);
        try std.testing.expect(c.dut_r_write_valid(dut.handle) == 0);
        const request_fire = ready and
            c.dut_read_request_valid(dut.handle) != 0;
        const response_fire = response_active and
            c.dut_read_response_ready(dut.handle) != 0;
        if (request_fire) {
            const request = requestAt(dut);
            requests += 1;
            pending = request;
            if (requests - 1 != target_request) {
                response_lanes = try scratch.read(request.token, request.group);
                response_active = true;
                pending = null;
            }
        }
        dut.step();
        if (response_fire)
            response_active = false;
        if (requests -| 1 == target_request and pending != null)
            break;
    }

    try std.testing.expectEqual(expected_frontend_state, c.dut_debug_frontend_state(dut.handle));
    try std.testing.expectEqual(@as(u8, 1), c.dut_debug_read_owner(dut.handle));
    c.dut_set_abort(dut.handle, 1);
    dut.eval();
    try std.testing.expect(c.dut_scalar_valid(dut.handle) == 0);
    dut.step();
    c.dut_set_abort(dut.handle, 0);
    try std.testing.expect(c.dut_busy(dut.handle) != 0);
    const retained = pending.?;
    response_lanes = try scratch.read(retained.token, retained.group);
    c.dut_set_read_response(dut.handle, 1, &response_lanes, 0);
    dut.eval();
    try std.testing.expect(c.dut_read_response_ready(dut.handle) != 0);
    dut.step();
    c.dut_set_read_response(dut.handle, 0, &zero_lanes, 0);
    for (0..256) |_| {
        if (c.dut_busy(dut.handle) == 0) break;
        dut.step();
    }
    try std.testing.expect(c.dut_busy(dut.handle) == 0);
    try std.testing.expect(c.dut_error(dut.handle) == 0);
    try std.testing.expectEqual(@as(u32, 0), c.dut_status(dut.handle));
    try std.testing.expect(c.dut_gamma_valid(dut.handle) == 0);
    dut.clear();

    try loadGamma(dut, gamma);
    _ = try runPipeline(
        dut,
        scratch,
        values,
        gamma,
        128,
        1,
        .{ .resident = true },
        0x9000 + target_request,
    );
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
    try loadGamma(&dut, gamma128);
    const resident_max_scratch = section.RmsNormScalarPipelineStatus.reduce(
        section.RmsNormReduceStatus.frontend(
            section.RmsNormFrontendStatus.max_exp |
                section.RmsNormFrontendStatus.scratch,
        ),
    );
    _ = try runPipeline(
        &dut,
        &scratch,
        values128,
        gamma128,
        128,
        1,
        .{
            .resident = true,
            .read_error_request = 0,
            .expected_status = resident_max_scratch,
        },
        0x7000,
    );
    try loadGamma(&dut, gamma128);
    const resident_sum_scratch = section.RmsNormScalarPipelineStatus.reduce(
        section.RmsNormReduceStatus.frontend(
            section.RmsNormFrontendStatus.scratch,
        ),
    );
    _ = try runPipeline(
        &dut,
        &scratch,
        values128,
        gamma128,
        128,
        1,
        .{
            .resident = true,
            .read_error_request = 16,
            .expected_status = resident_sum_scratch,
        },
        0x7001,
    );
    try loadGamma(&dut, gamma128);
    try abortResidentPhase(&dut, &scratch, values128, gamma128, 0, 2);
    try abortResidentPhase(&dut, &scratch, values128, gamma128, 16, 4);

    var shapes: usize = 0;
    var resident_shapes: usize = 0;
    var cycles: usize = 0;
    var writes: usize = 0;
    var requests: usize = 0;
    var scalars: usize = 0;
    for (rows_set, 0..) |rows, row_index| {
        const gamma = try makeGamma(allocator, rows);
        defer allocator.free(gamma);
        try loadGamma(&dut, gamma);
        for (1..5) |tokens| {
            const values = try makeResiduals(allocator, rows, tokens);
            defer allocator.free(values);
            const external = try runPipeline(
                &dut,
                &scratch,
                values,
                gamma,
                rows,
                tokens,
                .{ .resident = false },
                0xa000 + row_index * 16 + tokens,
            );
            const resident = try runPipeline(
                &dut,
                &scratch,
                values,
                gamma,
                rows,
                tokens,
                .{ .resident = true },
                0xb000 + row_index * 16 + tokens,
            );
            shapes += 1;
            resident_shapes += 1;
            cycles += external.cycles + resident.cycles;
            writes += external.writes + resident.writes;
            requests += external.requests + resident.requests;
            scalars += external.scalars + resident.scalars;
        }
    }

    std.debug.print(
        "section_rmsnorm_scalar_pipeline cosim:\n" ++
            "  external shapes={d}, resident shapes={d}, cycles={d}\n" ++
            "  R writes={d}, shared reads={d}, exact scalars={d}\n" ++
            "  resident max/sum scratch faults and retained-owner " ++
            "abort/restart: passed\n\n",
        .{ shapes, resident_shapes, cycles, writes, requests, scalars },
    );
}
