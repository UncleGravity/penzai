//! Versioned inference metrics contract shared by host and daemon.

pub const schema_version: u16 = 1;

/// Version of the indexed hardware recorder schema exposed at MMIO 0x160.
pub const recorder_schema: u32 = 0x0001_0000;
pub const metric_count: u8 = 40;
pub const stage_count: u8 = 11;

pub const HardwareCapability = struct {
    pub const core_bank: u32 = 1 << 0;
    pub const full_bank: u32 = 1 << 1;
    pub const saturating: u32 = 1 << 2;
    pub const frozen_snapshot: u32 = 1 << 3;
    pub const indexed_read: u32 = 1 << 4;
};

/// Capabilities compiled into the production recorder. Runtime metrics levels
/// only select how much of this bank is returned; they do not add counters.
pub const compiled_hardware_capabilities: u32 =
    (@as(u32, metric_count) << 16) |
    (@as(u32, stage_count) << 8) |
    HardwareCapability.core_bank |
    HardwareCapability.saturating |
    HardwareCapability.frozen_snapshot |
    HardwareCapability.indexed_read;

pub const Outcome = enum(u2) {
    none = 0,
    commit = 1,
    failed = 2,
    cancel = 3,
};

/// Stable IDs for the lean recorder bank. Total cycles is u64; stage calls are
/// saturating u8; selector high-water is u3; other counters are saturating u32.
/// Indexed reads zero-extend every value to a 32-bit word. The order is ABI.
pub const MetricId = enum(u8) {
    total_cycles = 0,
    control_cycles = 1,

    embed_cycles = 2,
    attn_norm_cycles = 3,
    qkv_rope_cycles = 4,
    kv_append_cycles = 5,
    attention_cycles = 6,
    o_proj_resid_cycles = 7,
    ffn_norm_cycles = 8,
    gate_up_swiglu_q8_cycles = 9,
    down_resid_cycles = 10,
    final_norm_cycles = 11,
    lm_head_cycles = 12,

    embed_calls = 13,
    attn_norm_calls = 14,
    qkv_rope_calls = 15,
    kv_append_calls = 16,
    attention_calls = 17,
    o_proj_resid_calls = 18,
    ffn_norm_calls = 19,
    gate_up_swiglu_q8_calls = 20,
    down_resid_calls = 21,
    final_norm_calls = 22,
    lm_head_calls = 23,

    projection_weight_beats = 24,
    projection_weight_source_empty_cycles = 25,
    projection_weight_consumer_blocked_cycles = 26,
    projection_selector_full_cycles = 27,
    projection_selector_high_water = 28,
    projection_q8_requests = 29,
    projection_q8_responses = 30,
    projection_q8_request_wait_cycles = 31,
    projection_q8_response_wait_cycles = 32,
    projection_drain_cycles = 33,
    projection_bank_wait_cycles = 34,
    weight_axi_r_beats = 35,
    weight_axi_r_gap_port_cycles = 36,
    weight_zip_skew_cycles = 37,
    history_axi_r_beats = 38,
    kv_axi_w_beats = 39,
};

pub fn metricName(id: MetricId) []const u8 {
    return @tagName(id);
}

pub fn metricStorageBits(id: MetricId) u8 {
    return switch (id) {
        .total_cycles => 64,
        .embed_calls,
        .attn_norm_calls,
        .qkv_rope_calls,
        .kv_append_calls,
        .attention_calls,
        .o_proj_resid_calls,
        .ffn_norm_calls,
        .gate_up_swiglu_q8_calls,
        .down_resid_calls,
        .final_norm_calls,
        .lm_head_calls,
        => 8,
        .projection_selector_high_water => 3,
        else => 32,
    };
}

pub const Level = enum(u8) {
    none = 0,
    summary = 1,
    full = 2,
};

pub const Capability = struct {
    pub const summary: u32 = 1 << 0;
    pub const full: u32 = 1 << 1;
};

pub const Summary = struct {
    prompt_tokens: u64 = 0,
    generated_tokens: u64 = 0,
    prefill_tiles: u64 = 0,
    decode_executions: u64 = 0,
    prefill_wall_ns: u64 = 0,
    prefill_device_ns: u64 = 0,
    decode_wall_ns: u64 = 0,
    decode_device_ns: u64 = 0,
    first_token_ns: u64 = 0,
    output_first_token_ns: u64 = 0,
    engine_cycles: u64 = 0,
};

test "metrics levels have stable wire values" {
    const std = @import("std");
    try std.testing.expectEqual(@as(u8, 0), @intFromEnum(Level.none));
    try std.testing.expectEqual(@as(u8, 1), @intFromEnum(Level.summary));
    try std.testing.expectEqual(@as(u8, 2), @intFromEnum(Level.full));
}

test "hardware recorder schema has stable dense IDs" {
    const std = @import("std");
    const fields = @typeInfo(MetricId).@"enum".fields;

    try std.testing.expectEqual(@as(usize, metric_count), fields.len);
    inline for (fields, 0..) |field, index| {
        try std.testing.expectEqual(@as(u8, @intCast(index)), @as(u8, @intCast(field.value)));
    }
    try std.testing.expectEqual(@as(u32, 0x0028_0b1d), compiled_hardware_capabilities);
    try std.testing.expectEqualStrings("projection_drain_cycles", metricName(.projection_drain_cycles));
    try std.testing.expectEqual(@as(u8, 64), metricStorageBits(.total_cycles));
    try std.testing.expectEqual(@as(u8, 8), metricStorageBits(.embed_calls));
    try std.testing.expectEqual(@as(u8, 3), metricStorageBits(.projection_selector_high_water));
    try std.testing.expectEqual(@as(u8, 32), metricStorageBits(.weight_axi_r_beats));
}
