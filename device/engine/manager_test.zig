const std = @import("std");
const shared = @import("shared");
const manager_mod = @import("manager.zig");

const model_spec = shared.engine.model_spec;
const rpc = shared.engine.rpc;

const FakeHeap = struct {
    const Slot = struct { handle: u64, address: u64, bytes: u64, live: bool };

    next_handle: u64 = 1,
    next_address: u64 = 0x1000_0000,
    slots: [8]Slot = undefined,
    slot_count: usize = 0,
    fail_free: bool = false,

    fn add(self: *FakeHeap, bytes: u64, alignment: u32) !shared.wire.TensorRange {
        if (self.slot_count == self.slots.len) return error.OutOfMemory;
        const mask = @as(u64, alignment) - 1;
        const address = (self.next_address + mask) & ~mask;
        const handle = self.next_handle;
        self.next_handle += 1;
        self.next_address = address + bytes + 0x10000;
        self.slots[self.slot_count] = .{
            .handle = handle,
            .address = address,
            .bytes = bytes,
            .live = true,
        };
        self.slot_count += 1;
        return .{ .handle = handle, .offset = 0, .nbytes = bytes };
    }

    pub fn allocate(self: *FakeHeap, bytes: u64, alignment: u32) !shared.wire.TensorRange {
        if (alignment == 0 or !std.math.isPowerOfTwo(alignment))
            return error.InvalidAlignment;
        return self.add(bytes, alignment);
    }

    pub fn free(self: *FakeHeap, handle: u64) !void {
        if (self.fail_free) return error.BackendFailure;
        for (self.slots[0..self.slot_count]) |*slot| {
            if (slot.handle == handle and slot.live) {
                slot.live = false;
                return;
            }
        }
        return error.UnknownHandle;
    }

    pub fn deviceAddress(self: *FakeHeap, range: shared.wire.TensorRange) !u64 {
        for (self.slots[0..self.slot_count]) |slot| {
            if (slot.handle != range.handle or !slot.live) continue;
            if (range.offset > slot.bytes or range.nbytes > slot.bytes - range.offset)
                return error.OutOfBounds;
            return slot.address + range.offset;
        }
        return error.UnknownHandle;
    }

    fn live(self: *const FakeHeap, handle: u64) bool {
        for (self.slots[0..self.slot_count]) |slot|
            if (slot.handle == handle) return slot.live;
        return false;
    }
};

test "manager derives immutable image addresses and owns session KV" {
    const Manager = manager_mod.Manager(2, 4);
    var manager: Manager = .{};
    var heap: FakeHeap = .{};
    const plan = try model_spec.planModelImage(.bonsai_1_7b, .q1_0, 0);
    const image = try heap.add(plan.image_bytes, model_spec.ddr_alignment);

    try manager.install(&heap, .{
        .image = image,
        .model_hash = 0x1234_5678,
        .model_slot = 0,
        .spec_id = .bonsai_1_7b,
        .weight_format = .q1_0,
    });
    try std.testing.expect(manager.ownsHandle(image.handle));
    const model = try manager.registry.model(0);
    const image_base = try heap.deviceAddress(image);
    try std.testing.expectEqual(image_base, model.header.embedding);
    try std.testing.expectEqual(
        image_base + plan.rope_table,
        model.header.rope_table,
    );

    const opened = try manager.open(&heap, .{
        .model_slot = 0,
        .session_slot = 1,
        .capacity_tokens = 32,
        .epoch = 7,
    });
    const kv = try manager.sessionRange(1);
    try std.testing.expect(manager.ownsHandle(kv.handle));
    try std.testing.expectEqual(try heap.deviceAddress(kv), opened.kv_base);
    try std.testing.expectError(error.ModelInUse, manager.uninstall(0));

    try manager.close(&heap, 1);
    try std.testing.expect(!heap.live(kv.handle));
    try std.testing.expect(!manager.ownsHandle(kv.handle));
    const unpublished = try manager.uninstall(0);
    try std.testing.expectEqual(image, unpublished);
    try std.testing.expect(!manager.ownsHandle(image.handle));
    try std.testing.expect(heap.live(image.handle));
}

test "manager failures do not leak KV or publish bad images" {
    const Manager = manager_mod.Manager(1, 1);
    var manager: Manager = .{};
    var heap: FakeHeap = .{};
    const plan = try model_spec.planModelImage(.bonsai_4b, .q2_0, 0);
    const wrong = try heap.add(plan.image_bytes - 64, model_spec.ddr_alignment);
    try std.testing.expectError(error.InvalidImageRange, manager.install(&heap, .{
        .image = wrong,
        .model_hash = 1,
        .model_slot = 0,
        .spec_id = .bonsai_4b,
        .weight_format = .q2_0,
    }));
    try std.testing.expect(!manager.ownsHandle(wrong.handle));

    const image = try heap.add(plan.image_bytes, model_spec.ddr_alignment);
    try manager.install(&heap, rpc.ModelInstall{
        .image = image,
        .model_hash = 2,
        .model_slot = 0,
        .spec_id = .bonsai_4b,
        .weight_format = .q2_0,
    });
    const live_before = heap.slot_count;
    try std.testing.expectError(error.InvalidKvCapacity, manager.open(&heap, .{
        .model_slot = 0,
        .session_slot = 0,
        .capacity_tokens = model_spec.bonsai_4b.context_length + 1,
        .epoch = 1,
    }));
    try std.testing.expectEqual(live_before, heap.slot_count);
}

test "failed KV release leaves session published and protected" {
    const Manager = manager_mod.Manager(1, 1);
    var manager: Manager = .{};
    var heap: FakeHeap = .{};
    const plan = try model_spec.planModelImage(.bonsai_1_7b, .q1_0, 0);
    const image = try heap.add(plan.image_bytes, model_spec.ddr_alignment);
    try manager.install(&heap, .{
        .image = image,
        .model_hash = 9,
        .model_slot = 0,
        .spec_id = .bonsai_1_7b,
        .weight_format = .q1_0,
    });
    _ = try manager.open(&heap, .{
        .model_slot = 0,
        .session_slot = 0,
        .capacity_tokens = 8,
        .epoch = 1,
    });
    const kv = try manager.sessionRange(0);
    heap.fail_free = true;
    try std.testing.expectError(error.HeapFailure, manager.close(&heap, 0));
    try std.testing.expect(manager.ownsHandle(kv.handle));
    _ = try manager.session(0);

    heap.fail_free = false;
    try manager.close(&heap, 0);
    try std.testing.expect(!manager.ownsHandle(kv.handle));
}
