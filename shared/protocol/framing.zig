const std = @import("std");

pub const version: u16 = 1;
pub const header_len: usize = 24;
pub const max_metadata_len: usize = 64 * 1024;
pub const max_payload_len: u64 = 64 * 1024 * 1024;

const magic = [4]u8{ 'P', 'N', 'Z', '1' };

pub const DecodeError = error{
    Truncated,
    BadMagic,
    UnsupportedVersion,
    UnsupportedFlags,
    MetadataTooLarge,
    PayloadTooLarge,
    TrailingBytes,
};

pub const EncodeError = error{
    MetadataTooLarge,
    PayloadTooLarge,
    OutputTooSmall,
};

pub const Frame = struct {
    metadata: []const u8,
    payload: []const u8,
};

pub const Header = struct {
    metadata_len: usize,
    payload_len: usize,
    total_len: usize,
};

pub fn encodedLen(metadata_len: usize, payload_len: usize) EncodeError!usize {
    if (metadata_len > max_metadata_len) return error.MetadataTooLarge;
    if (payload_len > max_payload_len) return error.PayloadTooLarge;
    return header_len + metadata_len + payload_len;
}

pub fn encode(metadata: []const u8, payload: []const u8, out: []u8) EncodeError!usize {
    const want = try encodedLen(metadata.len, payload.len);
    if (out.len < want) return error.OutputTooSmall;

    @memcpy(out[0..4], &magic);
    putU16(out, 4, version);
    putU16(out, 6, 0);
    putU32(out, 8, @intCast(metadata.len));
    putU64(out, 12, @intCast(payload.len));
    putU32(out, 20, 0);
    @memcpy(out[header_len..][0..metadata.len], metadata);
    @memcpy(out[header_len + metadata.len ..][0..payload.len], payload);
    return want;
}

pub fn decode(bytes: []const u8) DecodeError!Frame {
    const header = try decodeHeader(bytes);
    if (bytes.len < header.total_len) return error.Truncated;
    if (bytes.len != header.total_len) return error.TrailingBytes;

    const metadata_start = header_len;
    const payload_start = metadata_start + header.metadata_len;
    return .{
        .metadata = bytes[metadata_start..payload_start],
        .payload = bytes[payload_start..header.total_len],
    };
}

pub fn decodeHeader(bytes: []const u8) DecodeError!Header {
    if (bytes.len < header_len) return error.Truncated;
    if (!std.mem.eql(u8, bytes[0..4], &magic)) return error.BadMagic;
    if (getU16(bytes, 4) != version) return error.UnsupportedVersion;
    if (getU16(bytes, 6) != 0) return error.UnsupportedFlags;

    const metadata_len = getU32(bytes, 8);
    const payload_len = getU64(bytes, 12);
    if (metadata_len > max_metadata_len) return error.MetadataTooLarge;
    if (payload_len > max_payload_len) return error.PayloadTooLarge;

    const total = header_len + @as(usize, metadata_len) + @as(usize, @intCast(payload_len));
    return .{
        .metadata_len = @intCast(metadata_len),
        .payload_len = @intCast(payload_len),
        .total_len = total,
    };
}

fn putU16(out: []u8, offset: usize, value: u16) void {
    std.mem.writeInt(u16, out[offset..][0..2], value, .little);
}

fn putU32(out: []u8, offset: usize, value: u32) void {
    std.mem.writeInt(u32, out[offset..][0..4], value, .little);
}

fn putU64(out: []u8, offset: usize, value: u64) void {
    std.mem.writeInt(u64, out[offset..][0..8], value, .little);
}

fn getU16(bytes: []const u8, offset: usize) u16 {
    return std.mem.readInt(u16, bytes[offset..][0..2], .little);
}

fn getU32(bytes: []const u8, offset: usize) u32 {
    return std.mem.readInt(u32, bytes[offset..][0..4], .little);
}

fn getU64(bytes: []const u8, offset: usize) u64 {
    return std.mem.readInt(u64, bytes[offset..][0..8], .little);
}

test "frame roundtrip" {
    const meta = "metadata";
    const payload = "payload bytes";
    var buf: [128]u8 = undefined;
    const n = try encode(meta, payload, &buf);
    const frame = try decode(buf[0..n]);
    try std.testing.expectEqualSlices(u8, meta, frame.metadata);
    try std.testing.expectEqualSlices(u8, payload, frame.payload);
}

test "reject trailing bytes" {
    var buf: [64]u8 = undefined;
    const n = try encode("", "", &buf);
    try std.testing.expectError(error.TrailingBytes, decode(buf[0 .. n + 1]));
}

test "decode header without body" {
    var buf: [128]u8 = undefined;
    const n = try encode("abc", "payload", &buf);
    const header = try decodeHeader(buf[0..framing_header_len_for_test]);
    try std.testing.expectEqual(@as(usize, 3), header.metadata_len);
    try std.testing.expectEqual(@as(usize, 7), header.payload_len);
    try std.testing.expectEqual(n, header.total_len);
}

const framing_header_len_for_test = header_len;
