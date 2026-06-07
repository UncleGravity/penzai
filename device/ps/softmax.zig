const std = @import("std");

pub const SoftmaxError = error{
    InvalidLength,
    InvalidInput,
};

pub fn runF32(input: []const f32, dst: []f32) SoftmaxError!void {
    if (input.len == 0 or dst.len != input.len) return error.InvalidLength;

    const max_value = try maxInput(input);
    if (!std.math.isFinite(max_value)) {
        try writeInfiniteSoftmax(input, dst, max_value);
        return;
    }

    var sum: f64 = 0;
    for (input) |x| {
        if (x != x) return error.InvalidInput;
        sum += @exp(x - max_value);
    }
    if (sum == 0 or !std.math.isFinite(sum)) return error.InvalidInput;

    const inv_sum: f32 = @floatCast(1.0 / sum);
    for (input, dst) |x, *out| {
        out.* = @exp(x - max_value) * inv_sum;
    }
}

pub fn runBytes(input: []const u8, dst: []u8) SoftmaxError!void {
    const count = try f32Count(input);
    if (dst.len != input.len) return error.InvalidLength;

    const max_value = try maxInputBytes(input, count);
    if (!std.math.isFinite(max_value)) {
        try writeInfiniteSoftmaxBytes(input, dst, count, max_value);
        return;
    }

    var sum: f64 = 0;
    for (0..count) |i| {
        const x = readF32(input, i);
        if (x != x) return error.InvalidInput;
        sum += @exp(x - max_value);
    }
    if (sum == 0 or !std.math.isFinite(sum)) return error.InvalidInput;

    const inv_sum: f32 = @floatCast(1.0 / sum);
    for (0..count) |i| {
        writeF32(dst, i, @exp(readF32(input, i) - max_value) * inv_sum);
    }
}

fn maxInput(input: []const f32) SoftmaxError!f32 {
    var max_value = input[0];
    if (max_value != max_value) return error.InvalidInput;
    for (input[1..]) |x| {
        if (x != x) return error.InvalidInput;
        max_value = @max(max_value, x);
    }
    if (!std.math.isFinite(max_value) and max_value < 0) return error.InvalidInput;
    return max_value;
}

fn maxInputBytes(input: []const u8, count: usize) SoftmaxError!f32 {
    var max_value = readF32(input, 0);
    if (max_value != max_value) return error.InvalidInput;
    for (1..count) |i| {
        const x = readF32(input, i);
        if (x != x) return error.InvalidInput;
        max_value = @max(max_value, x);
    }
    if (!std.math.isFinite(max_value) and max_value < 0) return error.InvalidInput;
    return max_value;
}

fn writeInfiniteSoftmax(input: []const f32, dst: []f32, max_value: f32) SoftmaxError!void {
    if (max_value < 0) return error.InvalidInput;

    var count: usize = 0;
    for (input) |x| {
        if (x == max_value) count += 1;
    }
    if (count == 0) return error.InvalidInput;

    const value = 1.0 / @as(f32, @floatFromInt(count));
    for (input, dst) |x, *out| {
        out.* = if (x == max_value) value else 0;
    }
}

fn writeInfiniteSoftmaxBytes(input: []const u8, dst: []u8, count: usize, max_value: f32) SoftmaxError!void {
    if (max_value < 0) return error.InvalidInput;

    var positives: usize = 0;
    for (0..count) |i| {
        if (readF32(input, i) == max_value) positives += 1;
    }
    if (positives == 0) return error.InvalidInput;

    const value = 1.0 / @as(f32, @floatFromInt(positives));
    for (0..count) |i| {
        writeF32(dst, i, if (readF32(input, i) == max_value) value else 0);
    }
}

fn f32Count(bytes: []const u8) SoftmaxError!usize {
    if (bytes.len == 0 or bytes.len % @sizeOf(f32) != 0) return error.InvalidLength;
    return bytes.len / @sizeOf(f32);
}

fn readF32(bytes: []const u8, index: usize) f32 {
    return @bitCast(std.mem.readInt(u32, bytes[index * @sizeOf(f32) ..][0..4], .little));
}

fn writeF32(bytes: []u8, index: usize, value: f32) void {
    std.mem.writeInt(u32, bytes[index * @sizeOf(f32) ..][0..4], @bitCast(value), .little);
}

fn expectApprox(expected: f32, actual: f32, tolerance: f32) !void {
    try std.testing.expect(@abs(expected - actual) <= tolerance);
}

test "softmax produces stable probabilities" {
    const input = [_]f32{ 1, 2, 3 };
    var dst: [3]f32 = undefined;

    try runF32(&input, &dst);

    try expectApprox(0.09003057, dst[0], 0.000001);
    try expectApprox(0.24472848, dst[1], 0.000001);
    try expectApprox(0.66524094, dst[2], 0.000001);
    try expectApprox(1, dst[0] + dst[1] + dst[2], 0.000001);
}

test "softmax subtracts the maximum before exponentiation" {
    const input = [_]f32{ 1000, 1000 };
    var dst: [2]f32 = undefined;

    try runF32(&input, &dst);

    try expectApprox(0.5, dst[0], 0.000001);
    try expectApprox(0.5, dst[1], 0.000001);
}

test "softmax supports negative infinity masks" {
    const input = [_]f32{ 0, -std.math.inf(f32), 0 };
    var dst: [3]f32 = undefined;

    try runF32(&input, &dst);

    try expectApprox(0.5, dst[0], 0.000001);
    try expectApprox(0, dst[1], 0.000001);
    try expectApprox(0.5, dst[2], 0.000001);
}

test "softmax byte wrapper uses little-endian f32 values" {
    var input: [8]u8 = undefined;
    var dst: [8]u8 = undefined;
    writeF32(&input, 0, 1000);
    writeF32(&input, 1, 1000);

    try runBytes(&input, &dst);

    try expectApprox(0.5, readF32(&dst, 0), 0.000001);
    try expectApprox(0.5, readF32(&dst, 1), 0.000001);
}
