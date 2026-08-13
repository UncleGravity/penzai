//! Bit-exact cosim and PS/Q8 characterization for `section_swiglu`.

const std = @import("std");
const ref = @import("swiglu_ref");
const coeffs = @import("swiglu_coeffs");
const layout = @import("shared_layout");
const ps_activations = @import("ps_activations");
const c = @cImport(@cInclude("shim.h"));

const Case = struct {
    gate: f32,
    up: f32,
    last: bool,
};

const Payload = struct {
    data: u32,
    last: bool,
    status: u8,
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
};

fn reset(dut: *Dut) void {
    c.dut_set_clk(dut.handle, 0);
    c.dut_set_rst_n(dut.handle, 0);
    c.dut_set_abort(dut.handle, 0);
    c.dut_set_input(dut.handle, 0, 0, 0, 0);
    c.dut_set_out_ready(dut.handle, 0);
    dut.eval();
    for (0..6) |_| dut.step();
    c.dut_set_rst_n(dut.handle, 1);
    dut.step();
}

fn expectOutput(dut: *Dut, input: Case, index: usize) !void {
    const expected = ref.model(input.gate, input.up);
    const got_bits = c.dut_out_data(dut.handle);
    const got_status: u2 = @truncate(c.dut_out_status(dut.handle));
    const got_last = c.dut_out_last(dut.handle) != 0;
    if (got_bits != expected.bits or got_status != expected.status or got_last != input.last) {
        std.debug.print(
            "mismatch {d}: gate=0x{x:0>8} up=0x{x:0>8} got=0x{x:0>8}/0x{x}/{any} expected=0x{x:0>8}/0x{x}/{any}\n",
            .{ index, @as(u32, @bitCast(input.gate)), @as(u32, @bitCast(input.up)), got_bits, got_status, got_last, expected.bits, expected.status, input.last },
        );
        return error.ResultMismatch;
    }
}

fn runStream(
    dut: *Dut,
    inputs: []const Case,
    random_bubbles: bool,
    random_backpressure: bool,
    seed: u64,
) !struct { latency: usize, stalled: bool } {
    var prng = std.Random.DefaultPrng.init(seed);
    const rnd = prng.random();
    var sent: usize = 0;
    var received: usize = 0;
    var cycle: usize = 0;
    var presenting = false;
    var first_accept_cycle: ?usize = null;
    var first_output_cycle: ?usize = null;
    var stalled = false;
    var held_payload: ?Payload = null;

    while (received < inputs.len) : (cycle += 1) {
        if (cycle > inputs.len * 12 + 4096) return error.StreamTimeout;
        if (!presenting and sent < inputs.len) {
            presenting = !random_bubbles or rnd.uintLessThan(u8, 4) != 0;
        }
        if (presenting) {
            const input = inputs[sent];
            c.dut_set_input(
                dut.handle,
                1,
                @bitCast(input.gate),
                @bitCast(input.up),
                @intFromBool(input.last),
            );
        } else {
            c.dut_set_input(dut.handle, 0, 0, 0, 0);
        }
        const ready_out = !random_backpressure or rnd.uintLessThan(u8, 2) != 0;
        c.dut_set_out_ready(dut.handle, @intFromBool(ready_out));
        dut.eval();

        const input_fire = presenting and c.dut_in_ready(dut.handle) != 0;
        const output_valid = c.dut_out_valid(dut.handle) != 0;
        const output_fire = output_valid and ready_out;
        if (output_valid and !ready_out) {
            const payload: Payload = .{
                .data = c.dut_out_data(dut.handle),
                .last = c.dut_out_last(dut.handle) != 0,
                .status = c.dut_out_status(dut.handle),
            };
            if (held_payload) |held| {
                if (payload.data != held.data or payload.last != held.last or payload.status != held.status)
                    return error.OutputChangedUnderBackpressure;
            }
            held_payload = payload;
        } else {
            held_payload = null;
        }
        if (presenting and !input_fire) stalled = true;
        if (input_fire) {
            first_accept_cycle = first_accept_cycle orelse cycle;
            sent += 1;
            presenting = false;
        }
        if (output_fire) {
            first_output_cycle = first_output_cycle orelse cycle;
            try expectOutput(dut, inputs[received], received);
            received += 1;
        }
        dut.step();
    }
    c.dut_set_input(dut.handle, 0, 0, 0, 0);
    c.dut_set_out_ready(dut.handle, 0);
    return .{
        .latency = first_output_cycle.? - first_accept_cycle.?,
        .stalled = stalled,
    };
}

fn buildCases(allocator: std.mem.Allocator) ![]Case {
    var cases: std.ArrayList(Case) = .empty;
    errdefer cases.deinit(allocator);

    // One exactly representable midpoint from every ROM segment proves every
    // committed slope/intercept entry against the independent coefficient model.
    for (0..coeffs.segment_count) |index| {
        const gate = coeffs.domain_min +
            (@as(f32, @floatFromInt(index)) + 0.5) * coeffs.segment_step;
        try cases.append(allocator, .{
            .gate = gate,
            .up = @as(f32, @floatFromInt(@as(i32, @intCast(index % 17)) - 8)) / 4.0,
            .last = index % 32 == 31,
        });
    }

    const directed_gate = [_]f32{
        -std.math.inf(f32), -std.math.nan(f32), -100, -17,               -16,
        -15.999,            -8,                 -2,   -1,                -0.0,
        0.0,                1,                  2,    8,                 15.999,
        16,                 17,                 100,  std.math.inf(f32), std.math.nan(f32),
    };
    for (directed_gate, 0..) |gate, index| {
        try cases.append(allocator, .{ .gate = gate, .up = if (index % 2 == 0) 1.0 else -2.0, .last = index % 7 == 6 });
    }
    try cases.append(allocator, .{ .gate = 1, .up = std.math.inf(f32), .last = true });
    try cases.append(allocator, .{ .gate = 1, .up = std.math.nan(f32), .last = false });
    try cases.append(allocator, .{ .gate = 100, .up = std.math.floatMax(f32), .last = true });

    var prng = std.Random.DefaultPrng.init(0x5116_1a08_0020_0001);
    const rnd = prng.random();
    for (0..4096) |index| {
        const gate = (rnd.float(f32) - 0.5) * 48.0;
        const up = (rnd.float(f32) - 0.5) * 16.0;
        try cases.append(allocator, .{ .gate = gate, .up = up, .last = index % 32 == 31 });
    }
    return cases.toOwnedSlice(allocator);
}

fn testAbort(dut: *Dut) !void {
    var old: [20]Case = undefined;
    for (&old, 0..) |*item, index| item.* = .{ .gate = @floatFromInt(index + 1), .up = -3, .last = index == 19 };
    c.dut_set_out_ready(dut.handle, 0);
    var sent: usize = 0;
    while (sent < old.len) {
        const item = old[sent];
        c.dut_set_input(dut.handle, 1, @bitCast(item.gate), @bitCast(item.up), @intFromBool(item.last));
        dut.eval();
        if (c.dut_in_ready(dut.handle) != 0) sent += 1;
        dut.step();
    }
    for (0..20) |_| dut.step();
    c.dut_set_abort(dut.handle, 1);
    c.dut_set_input(dut.handle, 0, 0, 0, 0);
    dut.step();
    c.dut_set_abort(dut.handle, 0);
    dut.eval();
    if (c.dut_out_valid(dut.handle) != 0 or c.dut_in_ready(dut.handle) == 0)
        return error.AbortDidNotFlush;

    var fresh: [64]Case = undefined;
    for (&fresh, 0..) |*item, index| item.* = .{
        .gate = -4.0 + @as(f32, @floatFromInt(index)) / 16.0,
        .up = 0.25,
        .last = index % 32 == 31,
    };
    _ = try runStream(dut, &fresh, false, true, 0xa80f_1a5a);
}

fn characterizePsAndQ8() !void {
    const blocks = 1024;
    var gate: [32]f32 = undefined;
    var up: [32]f32 = undefined;
    var ps: [32]f32 = undefined;
    var pl: [32]f32 = undefined;
    var ps_q: [32]i8 = undefined;
    var pl_q: [32]i8 = undefined;
    var ps_scale: [1]f16 = undefined;
    var pl_scale: [1]f16 = undefined;
    var prng = std.Random.DefaultPrng.init(0x08f0_5116_cafe_0020);
    const rnd = prng.random();
    var max_normalized: f64 = 0;
    var q_differences: usize = 0;
    var max_q_delta: u16 = 0;
    var max_scale_code_delta: u16 = 0;

    for (0..blocks) |block_index| {
        for (0..32) |lane| {
            gate[lane] = if ((block_index + lane) % 127 == 0)
                (if (lane % 2 == 0) -20.0 else 20.0)
            else
                (rnd.float(f32) - 0.5) * 32.0;
            up[lane] = (rnd.float(f32) - 0.5) * 8.0;
            pl[lane] = @bitCast(ref.model(gate[lane], up[lane]).bits);
        }
        try ps_activations.swigluBytes(std.mem.sliceAsBytes(&gate), std.mem.sliceAsBytes(&up), std.mem.sliceAsBytes(&ps));
        for (pl, ps, up) |pl_value, ps_value, up_value| {
            const normalized = @abs(@as(f64, pl_value) - @as(f64, ps_value)) /
                @max(@abs(@as(f64, up_value)), 1.0);
            max_normalized = @max(max_normalized, normalized);
        }
        try layout.quantizeQ8_0(&pl, &pl_q, &pl_scale);
        try layout.quantizeQ8_0(&ps, &ps_q, &ps_scale);
        for (pl_q, ps_q) |pl_value, ps_value| {
            const delta: i16 = @as(i16, pl_value) - @as(i16, ps_value);
            if (delta != 0) q_differences += 1;
            max_q_delta = @max(max_q_delta, @abs(delta));
        }
        const pl_scale_bits: u16 = @bitCast(pl_scale[0]);
        const ps_scale_bits: u16 = @bitCast(ps_scale[0]);
        const scale_delta = if (pl_scale_bits >= ps_scale_bits)
            pl_scale_bits - ps_scale_bits
        else
            ps_scale_bits - pl_scale_bits;
        max_scale_code_delta = @max(max_scale_code_delta, scale_delta);
    }

    std.debug.print(
        "  PS vector comparison: max |PL-PS|/max(|up|,1)={e:.6}\n" ++
            "  canonical Q8 drift: {d}/{d} bytes, max delta={d}, max f16-scale code delta={d}\n",
        .{ max_normalized, q_differences, blocks * 32, max_q_delta, max_scale_code_delta },
    );
    if (max_normalized > 2.0e-4) return error.PsErrorBound;
    if (max_q_delta > 1 or max_scale_code_delta > 1) return error.Q8DriftBound;
}

pub fn main() !void {
    const allocator = std.heap.page_allocator;
    const inputs = try buildCases(allocator);
    defer allocator.free(inputs);

    var dut = Dut.init();
    defer dut.deinit();
    reset(&dut);

    const burst = try runStream(&dut, inputs, false, false, 0x1111);
    if (burst.latency != 15) {
        std.debug.print("unexpected fixed latency: {d}\n", .{burst.latency});
        return error.LatencyMismatch;
    }

    reset(&dut);
    const elastic = try runStream(&dut, inputs, true, true, 0xbacc_5116);
    if (!elastic.stalled) return error.CreditBackpressureNotObserved;

    reset(&dut);
    try testAbort(&dut);
    try characterizePsAndQ8();

    std.debug.print(
        "\n  section SwiGLU cosim: {d} scalars, all 1024 ROM segments, exact RTL/model bits\n" ++
            "  burst latency={d}, bubbles/backpressure/stable hold/abort/nonfinite: passed\n\n",
        .{ inputs.len, burst.latency },
    );
}
