const std = @import("std");

pub const version: u16 = 5;
pub const encoded_len: usize = 520;

const magic: u32 = 0x5041_4350; // "PCAP", little-endian on the wire.

pub const Feature = struct {
    pub const inference: u32 = 1 << 0;
    pub const metrics_summary: u32 = 1 << 1;
    pub const metrics_full: u32 = 1 << 2;
};

pub const Format = struct {
    pub const weight_q1_0: u32 = 1 << 0;
    pub const weight_q2_0_g64: u32 = 1 << 1;
    pub const activation_q8_0: u32 = 1 << 2;
    pub const io_f32: u32 = 1 << 3;
    pub const kv_f16: u32 = 1 << 4;
};

pub const IdentityFlag = struct {
    pub const git_dirty: u32 = 1 << 0;
    pub const bitstream_hash_verified: u32 = 1 << 1;
};

pub const ReceiptStatus = enum(u8) {
    missing = 0,
    loaded = 1,
    invalid = 2,
};

pub fn BoundedText(comptime capacity: usize) type {
    return struct {
        len: u16 = 0,
        bytes: [capacity]u8 = [_]u8{0} ** capacity,

        pub fn set(self: *@This(), value: []const u8) error{TooLong}!void {
            if (value.len > capacity) return error.TooLong;
            @memset(&self.bytes, 0);
            @memcpy(self.bytes[0..value.len], value);
            self.len = @intCast(value.len);
        }

        pub fn slice(self: *const @This()) []const u8 {
            return self.bytes[0..self.len];
        }
    };
}

pub const Text128 = BoundedText(128);
pub const Text64 = BoundedText(64);

/// Identity copied from the existing bitstream run manifest by deploy.sh.
pub const Receipt = struct {
    status: ReceiptStatus = .missing,
    manifest_schema: u32 = 0,
    identity_flags: u32 = 0,
    run_id: Text128 = .{},
    variant: Text64 = .{},
    git_commit: Text64 = .{},
    bitstream_sha256: Text64 = .{},
    manifest_sha256: Text64 = .{},
    source_sha256: Text64 = .{},

    pub fn invalid() Receipt {
        return .{ .status = .invalid };
    }

    pub fn parse(text: []const u8) error{ InvalidReceipt, TooLong }!Receipt {
        var receipt: Receipt = .{};
        var seen_receipt_schema = false;
        var seen_manifest_schema = false;
        var seen_run = false;
        var seen_variant = false;
        var seen_commit = false;
        var seen_git_dirty = false;
        var seen_bitstream = false;
        var seen_manifest = false;
        var seen_source = false;
        var seen_bitstream_verified = false;

        var lines = std.mem.splitScalar(u8, text, '\n');
        while (lines.next()) |raw_line| {
            const line = std.mem.trim(u8, raw_line, " \t\r");
            if (line.len == 0 or std.mem.startsWith(u8, line, "#")) continue;
            var fields = std.mem.splitScalar(u8, line, '\t');
            const key = fields.next() orelse return error.InvalidReceipt;
            const value = fields.next() orelse return error.InvalidReceipt;
            if (fields.next() != null) return error.InvalidReceipt;
            if (std.mem.eql(u8, key, "key") and std.mem.eql(u8, value, "value")) continue;

            if (std.mem.eql(u8, key, "receipt_schema_version")) {
                const receipt_schema = std.fmt.parseInt(u32, value, 10) catch return error.InvalidReceipt;
                if (receipt_schema != 1) return error.InvalidReceipt;
                seen_receipt_schema = true;
            } else if (std.mem.eql(u8, key, "manifest_schema_version")) {
                receipt.manifest_schema = std.fmt.parseInt(u32, value, 10) catch return error.InvalidReceipt;
                seen_manifest_schema = true;
            } else if (std.mem.eql(u8, key, "run_id")) {
                receipt.run_id.set(value) catch return error.TooLong;
                seen_run = true;
            } else if (std.mem.eql(u8, key, "variant")) {
                receipt.variant.set(value) catch return error.TooLong;
                seen_variant = true;
            } else if (std.mem.eql(u8, key, "git_commit")) {
                receipt.git_commit.set(value) catch return error.TooLong;
                seen_commit = true;
            } else if (std.mem.eql(u8, key, "git_dirty")) {
                if (std.mem.eql(u8, value, "1")) receipt.identity_flags |= IdentityFlag.git_dirty else if (!std.mem.eql(u8, value, "0")) return error.InvalidReceipt;
                seen_git_dirty = true;
            } else if (std.mem.eql(u8, key, "bitstream_sha256")) {
                receipt.bitstream_sha256.set(value) catch return error.TooLong;
                seen_bitstream = true;
            } else if (std.mem.eql(u8, key, "manifest_sha256")) {
                receipt.manifest_sha256.set(value) catch return error.TooLong;
                seen_manifest = true;
            } else if (std.mem.eql(u8, key, "source_sha256")) {
                receipt.source_sha256.set(value) catch return error.TooLong;
                seen_source = true;
            } else if (std.mem.eql(u8, key, "bitstream_hash_verified")) {
                if (std.mem.eql(u8, value, "1")) receipt.identity_flags |= IdentityFlag.bitstream_hash_verified else if (!std.mem.eql(u8, value, "0")) return error.InvalidReceipt;
                seen_bitstream_verified = true;
            }
        }

        if (!seen_receipt_schema or !seen_manifest_schema or !seen_run or
            !seen_variant or !seen_commit or !seen_git_dirty or
            !seen_bitstream or !seen_manifest or !seen_source or
            !seen_bitstream_verified or receipt.manifest_schema == 0)
        {
            return error.InvalidReceipt;
        }
        receipt.status = .loaded;
        return receipt;
    }
};

pub const EngineInfo = struct {
    interface_id: u32 = 0,
    interface_version: u32 = 0,
    clock_hz: u32 = 0,
    token_tile_max: u32 = 0,
    token_lanes: u32 = 0,
    model_spec_count: u32 = 0,
    context_tokens_max: u32 = 0,
    address_record_bytes: u32 = 0,
};

pub const Report = struct {
    receipt_status: ReceiptStatus = .missing,
    wire_abi: u16 = 0,
    metrics_schema: u16 = 0,
    feature_mask: u32 = 0,
    format_mask: u32 = 0,
    identity_flags: u32 = 0,
    manifest_schema: u32 = 0,
    engine: EngineInfo = .{},
    run_id: Text128 = .{},
    variant: Text64 = .{},
    git_commit: Text64 = .{},
    bitstream_sha256: Text64 = .{},
    manifest_sha256: Text64 = .{},
    source_sha256: Text64 = .{},

    pub fn applyReceipt(self: *Report, receipt: Receipt) void {
        self.receipt_status = receipt.status;
        self.identity_flags = receipt.identity_flags;
        self.manifest_schema = receipt.manifest_schema;
        self.run_id = receipt.run_id;
        self.variant = receipt.variant;
        self.git_commit = receipt.git_commit;
        self.bitstream_sha256 = receipt.bitstream_sha256;
        self.manifest_sha256 = receipt.manifest_sha256;
        self.source_sha256 = receipt.source_sha256;
    }
};

pub const DecodeError = error{
    InvalidLength,
    BadMagic,
    UnsupportedVersion,
    InvalidStatus,
    InvalidText,
};

pub fn encode(report: Report, out: []u8) error{OutputTooSmall}!usize {
    if (out.len < encoded_len) return error.OutputTooSmall;
    var cursor: usize = 0;
    putU32(out, &cursor, magic);
    putU16(out, &cursor, version);
    putU8(out, &cursor, @intFromEnum(report.receipt_status));
    putU8(out, &cursor, 0);
    putU16(out, &cursor, report.wire_abi);
    putU16(out, &cursor, report.metrics_schema);
    putU32(out, &cursor, report.feature_mask);
    putU32(out, &cursor, report.format_mask);
    putU32(out, &cursor, report.identity_flags);
    putU32(out, &cursor, report.manifest_schema);
    putEngine(out, &cursor, report.engine);
    putText(128, out, &cursor, report.run_id);
    putText(64, out, &cursor, report.variant);
    putText(64, out, &cursor, report.git_commit);
    putText(64, out, &cursor, report.bitstream_sha256);
    putText(64, out, &cursor, report.manifest_sha256);
    putText(64, out, &cursor, report.source_sha256);
    std.debug.assert(cursor == encoded_len);
    return cursor;
}

pub fn decode(bytes: []const u8) DecodeError!Report {
    if (bytes.len != encoded_len) return error.InvalidLength;
    var cursor: usize = 0;
    if (takeU32(bytes, &cursor) != magic) return error.BadMagic;
    if (takeU16(bytes, &cursor) != version) return error.UnsupportedVersion;
    const status = std.enums.fromInt(ReceiptStatus, takeU8(bytes, &cursor)) orelse return error.InvalidStatus;
    if (takeU8(bytes, &cursor) != 0) return error.InvalidLength;
    var report: Report = .{
        .receipt_status = status,
        .wire_abi = takeU16(bytes, &cursor),
        .metrics_schema = takeU16(bytes, &cursor),
        .feature_mask = takeU32(bytes, &cursor),
        .format_mask = takeU32(bytes, &cursor),
        .identity_flags = takeU32(bytes, &cursor),
        .manifest_schema = takeU32(bytes, &cursor),
        .engine = takeEngine(bytes, &cursor),
    };
    report.run_id = try takeText(128, bytes, &cursor);
    report.variant = try takeText(64, bytes, &cursor);
    report.git_commit = try takeText(64, bytes, &cursor);
    report.bitstream_sha256 = try takeText(64, bytes, &cursor);
    report.manifest_sha256 = try takeText(64, bytes, &cursor);
    report.source_sha256 = try takeText(64, bytes, &cursor);
    if (cursor != encoded_len) return error.InvalidLength;
    return report;
}

fn putEngine(out: []u8, cursor: *usize, info: EngineInfo) void {
    inline for (.{
        info.interface_id,
        info.interface_version,
        info.clock_hz,
        info.token_tile_max,
        info.token_lanes,
        info.model_spec_count,
        info.context_tokens_max,
        info.address_record_bytes,
    }) |value| putU32(out, cursor, value);
}

fn takeEngine(bytes: []const u8, cursor: *usize) EngineInfo {
    return .{
        .interface_id = takeU32(bytes, cursor),
        .interface_version = takeU32(bytes, cursor),
        .clock_hz = takeU32(bytes, cursor),
        .token_tile_max = takeU32(bytes, cursor),
        .token_lanes = takeU32(bytes, cursor),
        .model_spec_count = takeU32(bytes, cursor),
        .context_tokens_max = takeU32(bytes, cursor),
        .address_record_bytes = takeU32(bytes, cursor),
    };
}

fn putText(comptime capacity: usize, out: []u8, cursor: *usize, value: BoundedText(capacity)) void {
    putU16(out, cursor, value.len);
    @memcpy(out[cursor.*..][0..capacity], &value.bytes);
    cursor.* += capacity;
}

fn takeText(comptime capacity: usize, bytes: []const u8, cursor: *usize) DecodeError!BoundedText(capacity) {
    var value: BoundedText(capacity) = .{};
    value.len = takeU16(bytes, cursor);
    if (value.len > capacity) return error.InvalidText;
    @memcpy(&value.bytes, bytes[cursor.*..][0..capacity]);
    for (value.bytes[value.len..]) |byte| if (byte != 0) return error.InvalidText;
    cursor.* += capacity;
    return value;
}

fn putU8(out: []u8, cursor: *usize, value: u8) void {
    out[cursor.*] = value;
    cursor.* += 1;
}

fn putU16(out: []u8, cursor: *usize, value: u16) void {
    std.mem.writeInt(u16, out[cursor.*..][0..2], value, .little);
    cursor.* += 2;
}

fn putU32(out: []u8, cursor: *usize, value: u32) void {
    std.mem.writeInt(u32, out[cursor.*..][0..4], value, .little);
    cursor.* += 4;
}

fn takeU8(bytes: []const u8, cursor: *usize) u8 {
    defer cursor.* += 1;
    return bytes[cursor.*];
}

fn takeU16(bytes: []const u8, cursor: *usize) u16 {
    defer cursor.* += 2;
    return std.mem.readInt(u16, bytes[cursor.*..][0..2], .little);
}

fn takeU32(bytes: []const u8, cursor: *usize) u32 {
    defer cursor.* += 4;
    return std.mem.readInt(u32, bytes[cursor.*..][0..4], .little);
}

test "capability payload roundtrips exactly" {
    var report: Report = .{
        .receipt_status = .loaded,
        .wire_abi = 18,
        .metrics_schema = 1,
        .feature_mask = Feature.inference | Feature.metrics_summary,
        .format_mask = Format.weight_q1_0 | Format.weight_q2_0_g64 | Format.activation_q8_0,
        .identity_flags = IdentityFlag.bitstream_hash_verified,
        .manifest_schema = 1,
        .engine = .{
            .interface_id = 0xB05A_4000,
            .interface_version = 0x0001_0007,
            .clock_hz = 285_000_000,
            .token_tile_max = 8,
            .token_lanes = 4,
            .model_spec_count = 3,
            .context_tokens_max = 65_536,
            .address_record_bytes = 64,
        },
    };
    try report.run_id.set("20260804T120000Z-deadbeef-f225");
    try report.variant.set("f225");
    try report.git_commit.set("deadbeef1234");
    try report.bitstream_sha256.set("0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef");
    try report.manifest_sha256.set("1123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef");
    try report.source_sha256.set("2123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef");

    var encoded: [encoded_len]u8 = undefined;
    try std.testing.expectEqual(encoded_len, try encode(report, &encoded));
    try std.testing.expectEqualDeep(report, try decode(&encoded));

    std.mem.writeInt(u16, encoded[4..6], 1, .little);
    try std.testing.expectError(error.UnsupportedVersion, decode(&encoded));
}

test "deployment receipt requires the existing manifest identity" {
    const text = "key\tvalue\n" ++
        "receipt_schema_version\t1\n" ++
        "manifest_schema_version\t1\n" ++
        "run_id\tclean-run\n" ++
        "variant\tf225\n" ++
        "git_commit\tdeadbeef1234\n" ++
        "git_dirty\t0\n" ++
        "source_sha256\t2222\n" ++
        "manifest_sha256\t3333\n" ++
        "bitstream_sha256\t4444\n" ++
        "bitstream_hash_verified\t1\n";
    const receipt = try Receipt.parse(text);
    try std.testing.expectEqual(ReceiptStatus.loaded, receipt.status);
    try std.testing.expect(receipt.identity_flags & IdentityFlag.bitstream_hash_verified != 0);
    try std.testing.expectEqualStrings("clean-run", receipt.run_id.slice());
}

test "deployment receipt requires explicit dirty state" {
    const text = "key\tvalue\n" ++
        "receipt_schema_version\t1\n" ++
        "manifest_schema_version\t1\n" ++
        "run_id\tincomplete-run\n" ++
        "variant\tf225\n" ++
        "git_commit\tdeadbeef1234\n" ++
        "source_sha256\t2222\n" ++
        "manifest_sha256\t3333\n" ++
        "bitstream_sha256\t4444\n" ++
        "bitstream_hash_verified\t1\n";
    try std.testing.expectError(error.InvalidReceipt, Receipt.parse(text));
}
