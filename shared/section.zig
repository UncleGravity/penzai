//! Fixed P2 section scratch contract.
//!
//! Scratch is private PL storage owned by one section command. It is deliberately
//! not a `wire.TensorRange`: the host names external DDR tensors, while the section
//! controller assigns these fixed roles and addresses internally.

const std = @import("std");
const layout = @import("layout.zig");

pub const version: u16 = 1;
pub const ffn_kernel_version: u32 = 15;

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

/// Number of 256-bit row groups reserved for one token in a role.  Each group
/// contributes one 64-bit word to every physical bank.
pub fn f32GroupsPerToken(role: F32Role) u32 {
    return role.rowCapacity() / f32_rows_per_group;
}

/// Per-bank address span reserved for a role across the maximum query tile.
pub fn f32RoleSpan(role: F32Role) u32 {
    return query_tile_max * f32GroupsPerToken(role);
}

/// First physical address occupied by a role in each of the four banks.
pub fn f32RoleBase(role: F32Role) u32 {
    return switch (role) {
        .residual => 0,
        .x0 => f32RoleSpan(.residual),
        .x1 => f32RoleSpan(.residual) + f32RoleSpan(.x0),
        .x2 => f32RoleSpan(.residual) + f32RoleSpan(.x0) + f32RoleSpan(.x1),
    };
}

/// Physical word depth of each 64-bit bank across all roles.
pub fn f32BankDepth() u32 {
    return f32RoleBase(.x2) + f32RoleSpan(.x2);
}

pub const Error = error{
    InvalidBank,
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
    return .{
        .bank = @intCast((even_row % f32_rows_per_group) / f32_values_per_word),
        .address = token * f32GroupsPerToken(role) + even_row / f32_rows_per_group,
    };
}

/// Map a row pair to its absolute address in the shared four-bank memory.
pub fn f32PhysicalLocation(role: F32Role, token: u32, even_row: u32) Error!F32Location {
    const local = try f32Location(role, token, even_row);
    return .{
        .bank = local.bank,
        .address = f32RoleBase(role) + local.address,
    };
}

/// The Q8 activation is the GEMM kernel's native pair of memories: 32 int8
/// values plus one f16 scale. DDR's padded 40-byte stream is only an ingress
/// representation and is not the scratch layout.
pub const Q8Location = struct {
    address: u32,
};

pub const q8_column_capacity: u32 = 8;

// P3's compact DOWN-input store holds one native Q8 record per FFN block and
// token in each ping-pong bank. The block-major physical layout accepts the
// producer schedule directly; consumers can request the same records in
// token-major order without a transpose copy.
pub const q8_buffer_bank_count: u32 = 2;
pub const q8_buffer_token_capacity: u32 = query_tile_max;
pub const q8_buffer_block_capacity: u32 = ffn_dim_max / layout.q8_block;
pub const q8_buffer_records_per_bank: u32 =
    q8_buffer_token_capacity * q8_buffer_block_capacity;
pub const q8_buffer_record_capacity: u32 =
    q8_buffer_bank_count * q8_buffer_records_per_bank;

// The streaming FFN pairer consumes one 32-value Q8 block as four adjacent
// eight-value FP32 scratch groups. Its GATE tag maps directly to the resident
// X1/UP role's role-local read group.
pub const ffn_pair_groups_per_block: u32 = layout.q8_block / f32_rows_per_group;

pub const FfnPairTag = struct {
    block: u32,
    token: u32,
    group: u32,
};

/// Decode the native GEMM arrival order. Each 32-row block arrives as a lower
/// 16-row sweep across all tokens followed by the corresponding upper sweep.
pub fn ffnPairNativeTag(tokens: u32, ordinal: u32) Error!FfnPairTag {
    if (tokens == 0 or tokens > query_tile_max) return error.InvalidTokenCount;
    const groups_per_block = tokens * ffn_pair_groups_per_block;
    const block = ordinal / groups_per_block;
    if (block >= q8_buffer_block_capacity) return error.InvalidSubblock;
    const local = ordinal % groups_per_block;
    const half_span = tokens * 2;
    const half = local / half_span;
    const within_half = local % half_span;
    return .{
        .block = block,
        .token = within_half / 2,
        .group = half * 2 + within_half % 2,
    };
}

/// Decode the canonical consumer order used for X1 reads and SwiGLU output.
pub fn ffnPairCanonicalTag(tokens: u32, ordinal: u32) Error!FfnPairTag {
    if (tokens == 0 or tokens > query_tile_max) return error.InvalidTokenCount;
    const groups_per_block = tokens * ffn_pair_groups_per_block;
    const block = ordinal / groups_per_block;
    if (block >= q8_buffer_block_capacity) return error.InvalidSubblock;
    const local = ordinal % groups_per_block;
    return .{
        .block = block,
        .token = local / ffn_pair_groups_per_block,
        .group = local % ffn_pair_groups_per_block,
    };
}

pub fn ffnPairScratchGroup(block: u32, group: u32) Error!u32 {
    if (block >= q8_buffer_block_capacity) return error.InvalidSubblock;
    if (group >= ffn_pair_groups_per_block) return error.InvalidRow;
    return block * ffn_pair_groups_per_block + group;
}

pub const Q8BufferLocation = struct {
    address: u32,
};

pub fn q8BufferLocation(bank: u32, token: u32, block: u32) Error!Q8BufferLocation {
    if (bank >= q8_buffer_bank_count) return error.InvalidBank;
    if (token >= q8_buffer_token_capacity) return error.InvalidTokenCount;
    if (block >= q8_buffer_block_capacity) return error.InvalidSubblock;
    return .{ .address = bank * q8_buffer_records_per_bank +
        block * q8_buffer_token_capacity + token };
}

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

/// Host-visible section extent. The controller tiles this full command into
/// `FfnShape` chunks, so it accepts the full context rather than one PL tile.
pub const FfnCommandShape = struct {
    token_count: u32,
    model_dim: u32,
    ffn_dim: u32,

    pub fn validate(self: FfnCommandShape) Error!void {
        if (self.token_count == 0 or self.token_count > context_max)
            return error.InvalidTokenCount;
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

test "physical f32 mapping exhaustively and uniquely fills every bank" {
    try std.testing.expectEqual(@as(u32, 0), f32RoleBase(.residual));
    try std.testing.expectEqual(@as(u32, 2048), f32RoleBase(.x0));
    try std.testing.expectEqual(@as(u32, 8192), f32RoleBase(.x1));
    try std.testing.expectEqual(@as(u32, 14336), f32RoleBase(.x2));
    try std.testing.expectEqual(@as(u32, 16384), f32BankDepth());

    const total_words: usize = @as(usize, f32_banks_per_role) * @as(usize, f32BankDepth());
    const seen = try std.testing.allocator.alloc(bool, total_words);
    defer std.testing.allocator.free(seen);
    @memset(seen, false);

    const roles = [_]F32Role{ .residual, .x0, .x1, .x2 };
    var visited: usize = 0;
    for (roles) |role| {
        try std.testing.expectEqual(
            f32RoleBase(role) + f32RoleSpan(role),
            if (role == .x2) f32BankDepth() else f32RoleBase(@enumFromInt(@intFromEnum(role) + 1)),
        );

        for (0..query_tile_max) |token| {
            for (0..role.rowCapacity() / f32_values_per_word) |pair| {
                const even_row: u32 = @intCast(pair * f32_values_per_word);
                const local = try f32Location(role, @intCast(token), even_row);
                const physical = try f32PhysicalLocation(role, @intCast(token), even_row);
                try std.testing.expectEqual(local.bank, physical.bank);
                try std.testing.expectEqual(f32RoleBase(role) + local.address, physical.address);
                try std.testing.expect(physical.address < f32BankDepth());

                const index = @as(usize, physical.bank) * @as(usize, f32BankDepth()) +
                    @as(usize, physical.address);
                try std.testing.expect(!seen[index]);
                seen[index] = true;
                visited += 1;
            }
        }

        try std.testing.expectError(error.InvalidRow, f32PhysicalLocation(role, 0, role.rowCapacity()));
        try std.testing.expectError(error.InvalidRowPair, f32PhysicalLocation(role, 0, 1));
        try std.testing.expectError(error.InvalidTokenCount, f32PhysicalLocation(role, query_tile_max, 0));
    }

    try std.testing.expectEqual(total_words, visited);
    for (seen) |word_was_mapped| try std.testing.expect(word_was_mapped);
}

test "native Q8 addresses preserve token reuse within GEMM storage" {
    try std.testing.expectEqual(@as(u32, 19), (try q8Location(2, 3)).address);
    try std.testing.expectError(error.InvalidTokenCount, q8Location(0, query_tile_max));
    try std.testing.expectError(error.InvalidSubblock, q8Location(layout.max_sub_index, 0));
}

test "P3 Q8 buffer mapping is block-major, exhaustive, and ping-ponged" {
    try std.testing.expectEqual(@as(u32, 384), q8_buffer_block_capacity);
    try std.testing.expectEqual(@as(u32, 1536), q8_buffer_records_per_bank);
    try std.testing.expectEqual(@as(u32, 3072), q8_buffer_record_capacity);
    try std.testing.expectEqual(@as(u32, 0), (try q8BufferLocation(0, 0, 0)).address);
    try std.testing.expectEqual(@as(u32, 7), (try q8BufferLocation(0, 3, 1)).address);
    try std.testing.expectEqual(@as(u32, 1536), (try q8BufferLocation(1, 0, 0)).address);
    try std.testing.expectEqual(@as(u32, 3071), (try q8BufferLocation(1, 3, 383)).address);

    var seen = [_]bool{false} ** q8_buffer_record_capacity;
    var visited: usize = 0;
    for (0..q8_buffer_bank_count) |bank| {
        for (0..q8_buffer_block_capacity) |block| {
            for (0..q8_buffer_token_capacity) |token| {
                const loc = try q8BufferLocation(
                    @intCast(bank),
                    @intCast(token),
                    @intCast(block),
                );
                try std.testing.expect(loc.address < q8_buffer_record_capacity);
                try std.testing.expect(!seen[loc.address]);
                seen[loc.address] = true;
                visited += 1;
            }
        }
    }
    try std.testing.expectEqual(@as(usize, q8_buffer_record_capacity), visited);
    for (seen) |mapped| try std.testing.expect(mapped);

    try std.testing.expectError(error.InvalidBank, q8BufferLocation(2, 0, 0));
    try std.testing.expectError(error.InvalidTokenCount, q8BufferLocation(0, 4, 0));
    try std.testing.expectError(error.InvalidSubblock, q8BufferLocation(0, 0, 384));
}

test "P3 FFN pair tags map exhaustively onto X1 scratch groups" {
    try std.testing.expectEqual(@as(u32, 4), ffn_pair_groups_per_block);

    for (0..q8_buffer_block_capacity) |block| {
        for (0..ffn_pair_groups_per_block) |group| {
            const got = try ffnPairScratchGroup(@intCast(block), @intCast(group));
            try std.testing.expectEqual(
                @as(u32, @intCast(block * ffn_pair_groups_per_block + group)),
                got,
            );
            try std.testing.expect(got < f32GroupsPerToken(.x1));
        }
    }

    try std.testing.expectError(
        error.InvalidSubblock,
        ffnPairScratchGroup(q8_buffer_block_capacity, 0),
    );
    try std.testing.expectError(
        error.InvalidRow,
        ffnPairScratchGroup(0, ffn_pair_groups_per_block),
    );
}

test "P3 FFN pair native ingress reorders into canonical token blocks" {
    for (1..query_tile_max + 1) |tokens| {
        const groups_per_block = tokens * ffn_pair_groups_per_block;
        for (0..q8_buffer_block_capacity) |block| {
            for (0..groups_per_block) |local| {
                const ordinal: u32 = @intCast(block * groups_per_block + local);
                const native = try ffnPairNativeTag(@intCast(tokens), ordinal);
                const canonical = try ffnPairCanonicalTag(@intCast(tokens), ordinal);
                const half_span = tokens * 2;
                const half = local / half_span;
                const within_half = local % half_span;

                try std.testing.expectEqual(@as(u32, @intCast(block)), native.block);
                try std.testing.expectEqual(@as(u32, @intCast(within_half / 2)), native.token);
                try std.testing.expectEqual(
                    @as(u32, @intCast(half * 2 + within_half % 2)),
                    native.group,
                );
                try std.testing.expectEqual(@as(u32, @intCast(block)), canonical.block);
                try std.testing.expectEqual(
                    @as(u32, @intCast(local / ffn_pair_groups_per_block)),
                    canonical.token,
                );
                try std.testing.expectEqual(
                    @as(u32, @intCast(local % ffn_pair_groups_per_block)),
                    canonical.group,
                );
            }
        }
    }

    try std.testing.expectError(error.InvalidTokenCount, ffnPairNativeTag(0, 0));
    try std.testing.expectError(error.InvalidTokenCount, ffnPairCanonicalTag(5, 0));
    try std.testing.expectError(
        error.InvalidSubblock,
        ffnPairNativeTag(4, q8_buffer_block_capacity * 4 * ffn_pair_groups_per_block),
    );
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

    try (FfnCommandShape{ .token_count = 1, .model_dim = 4096, .ffn_dim = 12288 }).validate();
    try (FfnCommandShape{ .token_count = 16, .model_dim = 4096, .ffn_dim = 12288 }).validate();
    try (FfnCommandShape{ .token_count = context_max, .model_dim = 4096, .ffn_dim = 12288 }).validate();
    try std.testing.expectError(error.InvalidTokenCount, (FfnCommandShape{ .token_count = context_max + 1, .model_dim = 4096, .ffn_dim = 12288 }).validate());
}
