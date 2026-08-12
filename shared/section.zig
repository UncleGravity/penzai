//! Fixed P2 section scratch contract.
//!
//! Scratch is private PL storage owned by one section command. It is deliberately
//! not a `wire.TensorRange`: the host names external DDR tensors, while the section
//! controller assigns these fixed roles and addresses internally.

const std = @import("std");
const layout = @import("layout.zig");

pub const version: u16 = 1;

// V1 is the first routed point. A command may contain a longer prompt; the
// controller walks it in tiles no larger than this value.
pub const query_tile_max: u32 = 4;
pub const model_dim_max: u32 = 4096;
pub const ffn_dim_max: u32 = 12288;
pub const head_dim_max: u32 = 128;
pub const heads_max: u32 = 32;
pub const kv_heads_max: u32 = 8;
pub const context_max: u32 = 8192;

pub const f32_banks_per_role: u32 = 4;
pub const f32_values_per_word: u32 = 2;
pub const f32_word_bytes: u32 = f32_values_per_word * @sizeOf(f32);
pub const f32_rows_per_group: u32 = f32_banks_per_role * f32_values_per_word;

/// Named storage roles. These are schedule contracts, not host-addressable banks.
pub const F32Role = enum(u8) {
    residual,
    x0,
    x1,
    x2,

    pub fn rowCapacity(self: F32Role) u32 {
        return switch (self) {
            .residual, .x2 => model_dim_max,
            .x0, .x1 => ffn_dim_max,
        };
    }

    pub fn bytes(self: F32Role) u64 {
        return @as(u64, query_tile_max) * self.rowCapacity() * @sizeOf(f32);
    }
};

pub const F32Location = struct {
    bank: u2,
    address: u32,
};

pub const Error = error{
    InvalidTokenCount,
    InvalidModelDim,
    InvalidFfnDim,
    InvalidHeadDim,
    InvalidHeadCount,
    InvalidKvCount,
    InvalidRow,
    InvalidRowPair,
    InvalidSubblock,
};

/// Map an even/odd f32 row pair into four physical 64-bit banks. Eight adjacent
/// logical rows occupy one address across the four banks. GEMM's two-row result
/// beats can therefore be written directly while consumers read a 256-bit group.
pub fn f32Location(role: F32Role, token: u32, even_row: u32) Error!F32Location {
    if (token >= query_tile_max) return error.InvalidTokenCount;
    if (even_row >= role.rowCapacity()) return error.InvalidRow;
    if (even_row % f32_values_per_word != 0) return error.InvalidRowPair;
    const groups_per_token = std.math.divCeil(u32, role.rowCapacity(), f32_rows_per_group) catch unreachable;
    return .{
        .bank = @intCast((even_row % f32_rows_per_group) / f32_values_per_word),
        .address = token * groups_per_token + even_row / f32_rows_per_group,
    };
}

/// The Q8 activation is the GEMM kernel's native pair of memories: 32 int8
/// values plus one f16 scale. DDR's padded 40-byte stream is only an ingress
/// representation and is not the scratch layout.
pub const Q8Location = struct {
    address: u32,
};

pub const q8_column_capacity: u32 = 8;

pub fn q8Location(subblock: u32, token: u32) Error!Q8Location {
    if (token >= query_tile_max) return error.InvalidTokenCount;
    if (subblock >= layout.max_sub_index) return error.InvalidSubblock;
    return .{ .address = subblock * q8_column_capacity + token };
}

pub const FfnShape = struct {
    token_count: u32,
    model_dim: u32,
    ffn_dim: u32,

    pub fn validate(self: FfnShape) Error!void {
        try validateTokenCount(self.token_count);
        try validateModelDim(self.model_dim);
        if (self.ffn_dim == 0 or self.ffn_dim > ffn_dim_max or self.ffn_dim % layout.q1_block != 0)
            return error.InvalidFfnDim;
    }
};

pub const AttentionShape = struct {
    query_count: u32,
    model_dim: u32,
    head_dim_q: u32,
    head_dim_v: u32,
    n_heads: u32,
    n_head_kv: u32,
    kv_count: u32,

    pub fn validate(self: AttentionShape) Error!void {
        try validateTokenCount(self.query_count);
        try validateModelDim(self.model_dim);
        if (self.head_dim_q == 0 or self.head_dim_q > head_dim_max or self.head_dim_q % 8 != 0 or
            self.head_dim_v == 0 or self.head_dim_v > head_dim_max or self.head_dim_v % 8 != 0)
        {
            return error.InvalidHeadDim;
        }
        if (self.n_heads == 0 or self.n_heads > heads_max or
            self.model_dim != self.n_heads * self.head_dim_q or
            self.model_dim != self.n_heads * self.head_dim_v)
            return error.InvalidHeadCount;
        if (self.n_head_kv == 0 or self.n_head_kv > kv_heads_max or self.n_heads % self.n_head_kv != 0)
            return error.InvalidHeadCount;
        if (self.kv_count == 0 or self.kv_count > context_max) return error.InvalidKvCount;
    }

    pub fn newKvBytes(self: AttentionShape) Error!u64 {
        try self.validate();
        return @as(u64, self.query_count) * self.n_head_kv * (self.head_dim_q + self.head_dim_v) * @sizeOf(f16);
    }

    pub fn stateBytes(self: AttentionShape) Error!u64 {
        try self.validate();
        // Per (query, head): FP32 m and l. The output accumulator lives in X2.
        return @as(u64, self.query_count) * self.n_heads * 2 * @sizeOf(f32);
    }

    pub fn outputAccumulatorBytes(self: AttentionShape) Error!u64 {
        try self.validate();
        return @as(u64, self.query_count) * self.n_heads * self.head_dim_v * @sizeOf(f32);
    }
};

pub fn totalF32Bytes() u64 {
    return F32Role.residual.bytes() + F32Role.x0.bytes() +
        F32Role.x1.bytes() + F32Role.x2.bytes();
}

fn validateTokenCount(count: u32) Error!void {
    if (count == 0 or count > query_tile_max) return error.InvalidTokenCount;
}

fn validateModelDim(dim: u32) Error!void {
    if (dim == 0 or dim > model_dim_max or dim % layout.q1_block != 0)
        return error.InvalidModelDim;
}

test "f32 mapping accepts GEMM row-pair order without a transpose copy" {
    for (0..16) |row| {
        if (row % 2 != 0) continue;
        const loc = try f32Location(.x0, 2, @intCast(row));
        try std.testing.expectEqual(@as(u2, @intCast((row / 2) % 4)), loc.bank);
        try std.testing.expectEqual(2 * (ffn_dim_max / 8) + @as(u32, @intCast(row / 8)), loc.address);
    }
    try std.testing.expectError(error.InvalidRowPair, f32Location(.x0, 0, 1));
    try std.testing.expectError(error.InvalidTokenCount, f32Location(.x0, query_tile_max, 0));
}

test "native Q8 addresses preserve token reuse within GEMM storage" {
    try std.testing.expectEqual(@as(u32, 19), (try q8Location(2, 3)).address);
    try std.testing.expectError(error.InvalidTokenCount, q8Location(0, query_tile_max));
    try std.testing.expectError(error.InvalidSubblock, q8Location(layout.max_sub_index, 0));
}

test "v1 Bonsai section shapes and capacities are explicit" {
    try (FfnShape{ .token_count = 4, .model_dim = 4096, .ffn_dim = 12288 }).validate();
    const attention = AttentionShape{
        .query_count = 4,
        .model_dim = 4096,
        .head_dim_q = 128,
        .head_dim_v = 128,
        .n_heads = 32,
        .n_head_kv = 8,
        .kv_count = 8192,
    };
    try attention.validate();
    try std.testing.expectEqual(@as(u64, 16 * 1024), try attention.newKvBytes());
    try std.testing.expectEqual(@as(u64, 1024), try attention.stateBytes());
    try std.testing.expectEqual(F32Role.x2.bytes(), try attention.outputAccumulatorBytes());
    try std.testing.expectEqual(@as(u64, 512 * 1024), totalF32Bytes());
    try std.testing.expectError(error.InvalidTokenCount, (FfnShape{ .token_count = 5, .model_dim = 4096, .ffn_dim = 12288 }).validate());
}
