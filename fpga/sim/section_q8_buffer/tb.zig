//! Focused cosim for the P3 ping-ponged native-Q8 record buffer.
//!
//! Records are captured in block-major producer order and replayed in requested
//! token-major order. The test covers both full banks plus framing, duplicate,
//! incomplete-seal, pre-seal read, output-backpressure, and abort boundaries.

const std = @import("std");
const section = @import("shared_section");

const c = @cImport({
    @cInclude("shim.h");
});

comptime {
    if (section.q8_buffer_bank_count != 2 or
        section.q8_buffer_token_capacity != 4 or
        section.q8_buffer_block_capacity != 384 or
        section.q8_buffer_records_per_bank != 1536 or
        section.q8_buffer_record_capacity != 3072)
    {
        @compileError("section_q8_buffer geometry drifted from the shared oracle");
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
        c.dut_set_config(self.handle, 0, 0, 0, 0);
        c.dut_set_seal(self.handle, 0, 0);
        c.dut_set_abort(self.handle, 0, 0);
        c.dut_set_capture(self.handle, 0, 0, 0, 0, 0, 0);
        c.dut_set_read_request(self.handle, 0, 0, 0, 0);
        c.dut_set_output_ready(self.handle, 0);
        c.dut_set_clk(self.handle, 0);
        self.eval();
        for (0..4) |_| self.step();
        c.dut_set_rst_n(self.handle, 1);
        self.step();
    }
};

const RecordFault = enum {
    none,
    early_last,
    missing_last,
    tag_change,
    scale_padding,
};

fn bankMask(bank: u1) u8 {
    return @as(u8, 1) << bank;
}

fn hasBank(bits: u8, bank: u1) bool {
    return bits & bankMask(bank) != 0;
}

fn recordBeat(bank: u1, token: u2, block: u9, beat: u3) u64 {
    if (beat == 4) {
        return 0x3c00 | (@as(u64, bank) << 15) |
            (@as(u64, token) << 12) | (@as(u64, block) & 0x0fff);
    }

    var word: u64 = 0;
    for (0..8) |byte_index| {
        const lane: u8 = @intCast(@as(u32, beat) * 8 + byte_index);
        const value: u8 = @truncate(
            @as(u32, block) *% 29 + @as(u32, token) *% 67 +
                @as(u32, bank) *% 113 + @as(u32, lane) *% 11 + 3,
        );
        word |= @as(u64, value) << @intCast(byte_index * 8);
    }
    return word;
}

fn startConfig(dut: *Dut, bank: u1, tokens: u3, blocks: u9) !void {
    c.dut_set_config(dut.handle, 1, bank, tokens, blocks);
    dut.eval();
    try std.testing.expect(c.dut_config_ready(dut.handle) != 0);
    dut.step();
    c.dut_set_config(dut.handle, 0, 0, 0, 0);
    dut.eval();
}

fn waitActive(dut: *Dut, bank: u1) !void {
    var cycles: usize = 0;
    while (!hasBank(c.dut_bank_active(dut.handle), bank)) : (cycles += 1) {
        try std.testing.expect(cycles < section.q8_buffer_records_per_bank + 8);
        try std.testing.expect(hasBank(c.dut_bank_clearing(dut.handle), bank));
        try std.testing.expect(!hasBank(c.dut_bank_valid(dut.handle), bank));
        dut.step();
    }
    try std.testing.expect(cycles <= section.q8_buffer_records_per_bank);
    try std.testing.expect(!hasBank(c.dut_bank_clearing(dut.handle), bank));
    try std.testing.expect(!hasBank(c.dut_bank_error(dut.handle), bank));
    try std.testing.expectEqual(@as(u16, 0), c.dut_bank_record_count(dut.handle, bank));
}

fn configure(dut: *Dut, bank: u1, tokens: u3, blocks: u9) !void {
    try startConfig(dut, bank, tokens, blocks);
    try waitActive(dut, bank);
}

fn expectBadConfig(dut: *Dut, bank: u1, tokens: u3, blocks: u9) !void {
    try startConfig(dut, bank, tokens, blocks);
    try std.testing.expect(!hasBank(c.dut_bank_clearing(dut.handle), bank));
    try std.testing.expect(!hasBank(c.dut_bank_active(dut.handle), bank));
    try std.testing.expect(!hasBank(c.dut_bank_valid(dut.handle), bank));
    try std.testing.expect(hasBank(c.dut_bank_error(dut.handle), bank));
}

fn sendRecord(
    dut: *Dut,
    bank: u1,
    token: u2,
    block: u9,
    fault: RecordFault,
    expect_error: bool,
    add_bubble: bool,
) !void {
    for (0..5) |beat_usize| {
        const beat: u3 = @intCast(beat_usize);
        if (add_bubble and beat == 2) {
            c.dut_set_capture(dut.handle, 0, 0xfeed_face_dead_beef, 1, bank, token, block);
            dut.step();
        }

        const presented_token: u2 = if (fault == .tag_change and beat == 2)
            token +% 1
        else
            token;
        const last = switch (fault) {
            .early_last => beat == 1 or beat == 4,
            .missing_last => false,
            else => beat == 4,
        };
        var data = recordBeat(bank, token, block, beat);
        if (fault == .scale_padding and beat == 4)
            data |= 0x55aa_0000_0000_0000;

        c.dut_set_capture(
            dut.handle,
            1,
            data,
            @intFromBool(last),
            bank,
            presented_token,
            block,
        );
        dut.eval();
        try std.testing.expect(c.dut_capture_ready(dut.handle) != 0);

        if (beat == 4) {
            try std.testing.expectEqual(
                !expect_error,
                c.dut_capture_commit_valid(dut.handle) != 0,
            );
            if (!expect_error) {
                const expected = try section.q8BufferLocation(bank, token, block);
                try std.testing.expectEqual(
                    @as(u16, @intCast(expected.address)),
                    c.dut_capture_commit_address(dut.handle),
                );
            }
        }
        dut.step();
    }

    c.dut_set_capture(dut.handle, 0, 0, 0, 0, 0, 0);
    dut.eval();
    try std.testing.expect(c.dut_capture_done(dut.handle) != 0);
    try std.testing.expectEqual(expect_error, c.dut_capture_error(dut.handle) != 0);
}

fn seal(dut: *Dut, bank: u1, expect_error: bool) !void {
    c.dut_set_seal(dut.handle, 1, bank);
    dut.eval();
    try std.testing.expect(c.dut_seal_ready(dut.handle) != 0);
    dut.step();
    c.dut_set_seal(dut.handle, 0, 0);
    dut.eval();
    try std.testing.expect(c.dut_seal_done(dut.handle) != 0);
    try std.testing.expectEqual(expect_error, c.dut_seal_error(dut.handle) != 0);
    try std.testing.expectEqual(!expect_error, hasBank(c.dut_bank_valid(dut.handle), bank));
    try std.testing.expect(!hasBank(c.dut_bank_active(dut.handle), bank));
    try std.testing.expectEqual(expect_error, hasBank(c.dut_bank_error(dut.handle), bank));
}

fn readRecord(
    dut: *Dut,
    bank: u1,
    token: u2,
    block: u9,
    expect_error: bool,
    stall_seed: usize,
) !void {
    c.dut_set_output_ready(dut.handle, 0);
    c.dut_set_read_request(dut.handle, 1, bank, token, block);
    dut.eval();
    try std.testing.expect(c.dut_read_request_ready(dut.handle) != 0);
    try std.testing.expect(c.dut_read_issue_valid(dut.handle) != 0);
    const expected_address: u16 = if (block < section.q8_buffer_block_capacity)
        @intCast((try section.q8BufferLocation(bank, token, block)).address)
    else
        0;
    try std.testing.expectEqual(expected_address, c.dut_read_issue_address(dut.handle));
    dut.step();
    c.dut_set_read_request(dut.handle, 0, 0, 0, 0);
    dut.eval();

    var wait_cycles: usize = 0;
    while (c.dut_output_valid(dut.handle) == 0) : (wait_cycles += 1) {
        try std.testing.expect(wait_cycles < 4);
        dut.step();
    }

    for (0..5) |beat_usize| {
        const beat: u3 = @intCast(beat_usize);
        try std.testing.expectEqual(@as(u1, bank), @as(u1, @intCast(c.dut_output_bank(dut.handle))));
        try std.testing.expectEqual(@as(u8, token), c.dut_output_token(dut.handle));
        try std.testing.expectEqual(@as(u16, block), c.dut_output_block(dut.handle));
        try std.testing.expectEqual(beat == 4, c.dut_output_last(dut.handle) != 0);
        try std.testing.expectEqual(expect_error, c.dut_output_error(dut.handle) != 0);
        const expected_data: u64 = if (expect_error) 0 else recordBeat(bank, token, block, beat);
        try std.testing.expectEqual(expected_data, c.dut_output_data(dut.handle));

        const stalls = (stall_seed + beat_usize * 3) % 4;
        const stable_data = c.dut_output_data(dut.handle);
        for (0..stalls) |_| {
            try std.testing.expect(c.dut_read_request_ready(dut.handle) == 0);
            dut.step();
            try std.testing.expect(c.dut_output_valid(dut.handle) != 0);
            try std.testing.expectEqual(stable_data, c.dut_output_data(dut.handle));
            try std.testing.expectEqual(beat == 4, c.dut_output_last(dut.handle) != 0);
            try std.testing.expectEqual(expect_error, c.dut_output_error(dut.handle) != 0);
        }

        c.dut_set_output_ready(dut.handle, 1);
        dut.step();
        c.dut_set_output_ready(dut.handle, 0);
        dut.eval();
    }
    try std.testing.expect(c.dut_output_valid(dut.handle) == 0);
}

fn testErrors(dut: *Dut) !void {
    try expectBadConfig(dut, 0, 0, 1);
    try expectBadConfig(dut, 1, 5, 1);
    try expectBadConfig(dut, 0, 1, 0);
    try expectBadConfig(dut, 1, 1, 385);

    // A pre-seal request is accepted but cannot observe partial storage.
    try configure(dut, 0, 2, 2);
    try sendRecord(dut, 0, 0, 0, .none, false, true);
    try readRecord(dut, 0, 0, 0, true, 3);
    try sendRecord(dut, 0, 1, 0, .none, false, false);
    try sendRecord(dut, 0, 0, 1, .none, false, false);
    try seal(dut, 0, true); // token 1 / block 1 is missing

    // Duplicate tags never increment the exact completion count twice.
    try configure(dut, 0, 1, 2);
    try sendRecord(dut, 0, 0, 0, .none, false, false);
    try sendRecord(dut, 0, 0, 0, .none, true, false);
    try std.testing.expectEqual(@as(u16, 1), c.dut_bank_record_count(dut.handle, 0));
    try seal(dut, 0, true);

    inline for (.{
        RecordFault.early_last,
        RecordFault.missing_last,
        RecordFault.tag_change,
        RecordFault.scale_padding,
    }) |fault| {
        try configure(dut, 0, 1, 1);
        try sendRecord(dut, 0, 0, 0, fault, true, true);
        try std.testing.expectEqual(@as(u16, 0), c.dut_bank_record_count(dut.handle, 0));
        try seal(dut, 0, true);
    }

    // A tag outside the configured set is consumed and diagnosed fail-closed.
    try configure(dut, 0, 1, 1);
    try sendRecord(dut, 0, 0, 1, .none, true, false);
    try seal(dut, 0, true);

    // Abort suppresses a same-cycle beat, discards a partial record, and a new
    // configuration clears sticky error before capture resumes.
    try configure(dut, 0, 1, 1);
    for (0..2) |beat_usize| {
        const beat: u3 = @intCast(beat_usize);
        c.dut_set_capture(dut.handle, 1, recordBeat(0, 0, 0, beat), 0, 0, 0, 0);
        dut.eval();
        try std.testing.expect(c.dut_capture_ready(dut.handle) != 0);
        dut.step();
    }
    c.dut_set_abort(dut.handle, 1, 0);
    c.dut_set_capture(dut.handle, 1, recordBeat(0, 0, 0, 2), 0, 0, 0, 0);
    dut.eval();
    try std.testing.expect(c.dut_capture_ready(dut.handle) == 0);
    try std.testing.expect(c.dut_capture_commit_valid(dut.handle) == 0);
    dut.step();
    c.dut_set_abort(dut.handle, 0, 0);
    c.dut_set_capture(dut.handle, 0, 0, 0, 0, 0, 0);
    dut.eval();
    try std.testing.expect(hasBank(c.dut_bank_error(dut.handle), 0));
    try std.testing.expect(!hasBank(c.dut_bank_active(dut.handle), 0));
    try std.testing.expect(!hasBank(c.dut_bank_valid(dut.handle), 0));
}

fn testAbortReplay(dut: *Dut) !void {
    try configure(dut, 0, 1, 1);
    try sendRecord(dut, 0, 0, 0, .none, false, false);
    try seal(dut, 0, false);

    c.dut_set_output_ready(dut.handle, 0);
    c.dut_set_read_request(dut.handle, 1, 0, 0, 0);
    dut.eval();
    try std.testing.expect(c.dut_read_request_ready(dut.handle) != 0);
    dut.step();
    c.dut_set_read_request(dut.handle, 0, 0, 0, 0);
    while (c.dut_output_valid(dut.handle) == 0) dut.step();
    const stalled = c.dut_output_data(dut.handle);
    dut.step();
    try std.testing.expectEqual(stalled, c.dut_output_data(dut.handle));

    c.dut_set_abort(dut.handle, 1, 0);
    dut.eval();
    try std.testing.expect(c.dut_output_valid(dut.handle) == 0);
    dut.step();
    c.dut_set_abort(dut.handle, 0, 0);
    dut.eval();
    try std.testing.expect(!hasBank(c.dut_bank_valid(dut.handle), 0));
    try std.testing.expect(hasBank(c.dut_bank_error(dut.handle), 0));
}

fn testFullCapacity(dut: *Dut) !void {
    // Both duplicate maps clear concurrently. One global capture stream then
    // alternates banks while retaining independent counts and ownership.
    try startConfig(dut, 0, 4, 384);
    try startConfig(dut, 1, 4, 384);
    try waitActive(dut, 0);
    try waitActive(dut, 1);

    var records_written: usize = 0;
    for (0..section.q8_buffer_block_capacity) |block_usize| {
        const block: u9 = @intCast(block_usize);
        for (0..section.q8_buffer_token_capacity) |token_usize| {
            const token: u2 = @intCast(token_usize);
            const bubble = (block_usize * 5 + token_usize) % 97 == 0;
            try sendRecord(dut, 0, token, block, .none, false, bubble);
            // Reverse block order in bank one to prove tags, rather than source
            // sequence, own the physical mapping.
            const reverse_block: u9 = @intCast(section.q8_buffer_block_capacity - 1 - block_usize);
            try sendRecord(dut, 1, token, reverse_block, .none, false, !bubble and block_usize % 131 == 0);
            records_written += 2;
        }
    }
    try std.testing.expectEqual(@as(u16, 1536), c.dut_bank_record_count(dut.handle, 0));
    try std.testing.expectEqual(@as(u16, 1536), c.dut_bank_record_count(dut.handle, 1));
    try seal(dut, 0, false);
    try seal(dut, 1, false);

    var records_read: usize = 0;
    for (0..section.q8_buffer_token_capacity) |token_usize| {
        const token: u2 = @intCast(token_usize);
        for (0..section.q8_buffer_block_capacity) |block_usize| {
            const block: u9 = @intCast(block_usize);
            try readRecord(dut, 0, token, block, false, block_usize + token_usize);
            try readRecord(dut, 1, token, block, false, block_usize * 3 + token_usize);
            records_read += 2;
        }
    }
    try std.testing.expectEqual(records_written, records_read);

    std.debug.print(
        "\n  section Q8 buffer cosim: {d} full-capacity records written and replayed\n" ++
            "  block-major tagged capture -> token-major replay, exact seal, stalls: passed\n" ++
            "  storage geometry: 3072 x 256-bit payload + 3072 x f16 scale\n\n",
        .{records_read},
    );
}

pub fn main() !void {
    var dut = Dut.init();
    defer dut.deinit();
    dut.reset();

    try std.testing.expectEqual(@as(u32, 0), (try section.q8BufferLocation(0, 0, 0)).address);
    try std.testing.expectEqual(@as(u32, 1536), (try section.q8BufferLocation(1, 0, 0)).address);
    try std.testing.expectEqual(@as(u32, 3071), (try section.q8BufferLocation(1, 3, 383)).address);

    try testErrors(&dut);
    try testAbortReplay(&dut);
    try testFullCapacity(&dut);
}
