const std = @import("std");

pub const RegionError = error{
    OutOfMemory,
    UnknownHandle,
    OutOfBounds,
    InvalidAlignment,
};

/// A returned allocation: a stable `handle` plus the physical byte `offset`
/// into the backing store where its bytes live.
pub const Allocation = struct {
    handle: u64,
    offset: u64,
};

/// Offset bookkeeping for a single linear backing store of `size` bytes,
/// shared by the in-memory and XRT (board DRAM) heaps. It owns no
/// bytes — only the map from handles to byte ranges — so the same policy is
/// tested once and reused by both backends.
///
/// Two structures, cleanly separated:
///   * `free_list` — the unallocated extents, kept sorted by offset and
///     coalesced, so freed space is always reused and adjacent gaps merge.  A
///     real free, not the bump allocator's no-op.
///   * `used` — the live allocations keyed by handle, the source of truth for
///     bounds checks and offset resolution.
///
/// First-fit over `free`; O(n) in the number of live/free extents, which stays
/// small because model provisioning uses a handful of coarse buffers.
pub const Regions = struct {
    const Self = @This();

    const Extent = struct { offset: u64, len: u64 };
    const Slot = struct { handle: u64, offset: u64, len: u64 };

    allocator: std.mem.Allocator,
    size: u64,
    free_list: std.ArrayList(Extent),
    used: std.ArrayList(Slot),
    next_handle: u64,

    pub fn init(allocator: std.mem.Allocator, size: usize) !Self {
        var free_list: std.ArrayList(Extent) = .empty;
        errdefer free_list.deinit(allocator);
        if (size > 0) try free_list.append(allocator, .{ .offset = 0, .len = size });
        return .{
            .allocator = allocator,
            .size = size,
            .free_list = free_list,
            .used = .empty,
            .next_handle = 1,
        };
    }

    pub fn deinit(self: *Self) void {
        self.free_list.deinit(self.allocator);
        self.used.deinit(self.allocator);
        self.* = undefined;
    }

    /// First-fit allocation.  All list mutations reserve capacity up front so a
    /// failed reservation returns `OutOfMemory` without touching the maps.
    pub fn allocate(self: *Self, nbytes: u64, alignment: u32) RegionError!Allocation {
        if (alignment == 0 or !std.math.isPowerOfTwo(alignment)) return error.InvalidAlignment;
        self.used.ensureUnusedCapacity(self.allocator, 1) catch return error.OutOfMemory;

        // Zero-length allocations own no bytes; hand out a handle at offset 0
        // (never dereferenced) without disturbing the free list.
        if (nbytes == 0) return self.recordAssumeCapacity(0, 0);

        // Carving one extent removes it and re-inserts up to two pieces
        // (alignment pad in front, remainder behind), a net growth of one.
        self.free_list.ensureUnusedCapacity(self.allocator, 2) catch return error.OutOfMemory;

        for (self.free_list.items, 0..) |extent, i| {
            const start = alignForward(extent.offset, alignment);
            const pad = start - extent.offset;
            if (pad > extent.len or nbytes > extent.len - pad) continue;

            const tail_offset = start + nbytes;
            const tail_len = extent.offset + extent.len - tail_offset;

            _ = self.free_list.orderedRemove(i);
            // Insert the higher-offset remainder first so the lower-offset pad
            // lands ahead of it and the list stays sorted by offset.
            if (tail_len > 0) self.free_list.insertAssumeCapacity(i, .{ .offset = tail_offset, .len = tail_len });
            if (pad > 0) self.free_list.insertAssumeCapacity(i, .{ .offset = extent.offset, .len = pad });

            return self.recordAssumeCapacity(start, nbytes);
        }
        return error.OutOfMemory;
    }

    pub fn free(self: *Self, handle: u64) RegionError!void {
        const idx = self.slotIndex(handle) orelse return error.UnknownHandle;
        const slot = self.used.items[idx];
        // Coalescing inserts at most one net extent; reserve before mutating so
        // the slot is never dropped without its bytes being returned.
        if (slot.len != 0) self.free_list.ensureUnusedCapacity(self.allocator, 1) catch return error.OutOfMemory;
        _ = self.used.swapRemove(idx);
        if (slot.len != 0) self.insertFreeCoalescingAssumeCapacity(.{ .offset = slot.offset, .len = slot.len });
    }

    /// Resolve `handle`'s sub-range to an absolute byte offset, bounds-checked
    /// against the allocation's length.
    pub fn absolute(self: *Self, handle: u64, offset: u64, nbytes: u64) RegionError!u64 {
        const s = self.slotOf(handle) orelse return error.UnknownHandle;
        if (offset > s.len or nbytes > s.len - offset) return error.OutOfBounds;
        return s.offset + offset;
    }

    fn recordAssumeCapacity(self: *Self, offset: u64, len: u64) Allocation {
        const handle = self.next_handle;
        self.next_handle += 1;
        self.used.appendAssumeCapacity(.{ .handle = handle, .offset = offset, .len = len });
        return .{ .handle = handle, .offset = offset };
    }

    /// Insert a freed extent, merging with adjacent free neighbours so the list
    /// stays sorted, non-overlapping, and gap-free between touching extents.
    fn insertFreeCoalescingAssumeCapacity(self: *Self, ext: Extent) void {
        var block = ext;
        var pos: usize = 0;
        while (pos < self.free_list.items.len and self.free_list.items[pos].offset < block.offset) : (pos += 1) {}

        if (pos > 0) {
            const left = self.free_list.items[pos - 1];
            if (left.offset + left.len == block.offset) {
                block = .{ .offset = left.offset, .len = left.len + block.len };
                _ = self.free_list.orderedRemove(pos - 1);
                pos -= 1;
            }
        }
        if (pos < self.free_list.items.len) {
            const right = self.free_list.items[pos];
            if (block.offset + block.len == right.offset) {
                block.len += right.len;
                _ = self.free_list.orderedRemove(pos);
            }
        }
        self.free_list.insertAssumeCapacity(pos, block);
    }

    fn slotIndex(self: *Self, handle: u64) ?usize {
        for (self.used.items, 0..) |s, i| {
            if (s.handle == handle) return i;
        }
        return null;
    }

    fn slotOf(self: *Self, handle: u64) ?Slot {
        return if (self.slotIndex(handle)) |i| self.used.items[i] else null;
    }
};

fn alignForward(value: u64, alignment: u32) u64 {
    const mask = @as(u64, alignment) - 1;
    return (value + mask) & ~mask;
}

// ===========================  Tests  ===========================

const testing = std.testing;

/// Total free bytes across the free list — the invariant `size - live`.
fn freeBytes(r: *Regions) u64 {
    var total: u64 = 0;
    for (r.free_list.items) |e| total += e.len;
    return total;
}

/// Assert the free list is sorted, non-overlapping, and fully coalesced (no two
/// listed extents touch — those would have merged).
fn assertCoalesced(r: *Regions) !void {
    var prev_end: ?u64 = null;
    for (r.free_list.items) |e| {
        try testing.expect(e.len > 0);
        if (prev_end) |end| try testing.expect(e.offset > end); // strict gap between extents
        prev_end = e.offset + e.len;
    }
}

test "alloc/free reclaims space" {
    var r = try Regions.init(testing.allocator, 1024);
    defer r.deinit();

    const a = try r.allocate(256, 64);
    const b = try r.allocate(256, 64);
    try testing.expectEqual(@as(u64, 512), freeBytes(&r));

    try r.free(a.handle);
    try r.free(b.handle);
    try testing.expectEqual(@as(u64, 1024), freeBytes(&r));
    // Fully reclaimed and merged back into one extent.
    try testing.expectEqual(@as(usize, 1), r.free_list.items.len);
    try assertCoalesced(&r);
}

test "grow-after-free fits where the bump allocator would OOM" {
    var r = try Regions.init(testing.allocator, 1024);
    defer r.deinit();

    // Persistent buffer up front, then a compute buffer that grows across
    // free/realloc cycles — the llama.cpp gallocr pattern.
    const weights = try r.allocate(512, 64);
    var compute = try r.allocate(256, 64);
    var want: u64 = 384;
    while (want <= 512) : (want += 128) {
        try r.free(compute.handle);
        compute = try r.allocate(want, 64); // reuses the coalesced tail, no OOM
        try assertCoalesced(&r);
    }
    _ = weights;
    try testing.expectEqual(@as(u64, 512 + 512), r.size); // sanity
}

test "coalesces both neighbours" {
    var r = try Regions.init(testing.allocator, 300);
    defer r.deinit();

    const a = try r.allocate(100, 1);
    const b = try r.allocate(100, 1);
    const c = try r.allocate(100, 1);
    try testing.expectEqual(@as(usize, 0), r.free_list.items.len);

    try r.free(a.handle);
    try r.free(c.handle);
    try testing.expectEqual(@as(usize, 2), r.free_list.items.len); // two disjoint gaps
    try r.free(b.handle); // the middle merges left and right into one
    try testing.expectEqual(@as(usize, 1), r.free_list.items.len);
    try testing.expectEqual(@as(u64, 300), freeBytes(&r));
    try assertCoalesced(&r);
}

test "alignment padding is reclaimable free space" {
    var r = try Regions.init(testing.allocator, 1024);
    defer r.deinit();

    _ = try r.allocate(1, 1); // occupy [0,1)
    const a = try r.allocate(16, 64); // must start at 64, leaving [1,64) free
    try testing.expectEqual(@as(u64, 64), a.offset);
    try assertCoalesced(&r);

    // The 63-byte pad is real free space: a small allocation lands in it.
    const pad = try r.allocate(16, 1);
    try testing.expect(pad.offset >= 1 and pad.offset < 64);
}

test "out of memory when no extent fits" {
    var r = try Regions.init(testing.allocator, 128);
    defer r.deinit();

    _ = try r.allocate(100, 1);
    try testing.expectError(error.OutOfMemory, r.allocate(64, 1));
    // Freeing makes room again.
}

test "bounds and handle validation" {
    var r = try Regions.init(testing.allocator, 128);
    defer r.deinit();

    const a = try r.allocate(32, 8);
    try testing.expectEqual(a.offset, try r.absolute(a.handle, 0, 32));
    try testing.expectEqual(a.offset + 8, try r.absolute(a.handle, 8, 8));
    try testing.expectError(error.OutOfBounds, r.absolute(a.handle, 8, 32));
    try testing.expectError(error.UnknownHandle, r.absolute(9999, 0, 0));

    try r.free(a.handle);
    try testing.expectError(error.UnknownHandle, r.free(a.handle)); // double free
    try testing.expectError(error.UnknownHandle, r.absolute(a.handle, 0, 1));
}

test "zero-length allocation gets a handle without consuming space" {
    var r = try Regions.init(testing.allocator, 64);
    defer r.deinit();

    const z = try r.allocate(0, 64);
    try testing.expect(z.handle != 0);
    try testing.expectEqual(@as(u64, 64), freeBytes(&r));
    try testing.expectEqual(@as(u64, z.offset), try r.absolute(z.handle, 0, 0));
    try r.free(z.handle);
}

test "invalid alignment rejected" {
    var r = try Regions.init(testing.allocator, 64);
    defer r.deinit();
    try testing.expectError(error.InvalidAlignment, r.allocate(8, 0));
    try testing.expectError(error.InvalidAlignment, r.allocate(8, 3));
}

test "handles are never reused" {
    var r = try Regions.init(testing.allocator, 128);
    defer r.deinit();

    const a = try r.allocate(16, 1);
    try r.free(a.handle);
    const b = try r.allocate(16, 1);
    try testing.expect(b.handle != a.handle);
}
