//! Runtime ownership for resident model images and session KV arenas.

const std = @import("std");
const shared = @import("shared");
const state_mod = @import("state.zig");

const model_spec = shared.engine.model_spec;
const rpc = shared.engine.rpc;
const wire = shared.wire;

pub const Error = state_mod.Error || model_spec.SpecError || error{
    InvalidImageRange,
    HeapFailure,
    OutOfMemory,
};

pub fn Manager(
    comptime model_slot_count: usize,
    comptime session_slot_count: usize,
) type {
    return struct {
        const Self = @This();
        const Registry = state_mod.Registry(model_slot_count, session_slot_count);

        registry: Registry = Registry.init(),
        model_ranges: [model_slot_count]?wire.TensorRange = @splat(null),
        session_ranges: [session_slot_count]?wire.TensorRange = @splat(null),

        pub fn ownsHandle(self: *const Self, handle: u64) bool {
            if (handle == 0) return false;
            for (self.model_ranges) |range|
                if (range != null and range.?.handle == handle) return true;
            for (self.session_ranges) |range|
                if (range != null and range.?.handle == handle) return true;
            return false;
        }

        pub fn install(
            self: *Self,
            heap: anytype,
            request: rpc.ModelInstall,
        ) Error!void {
            const model_index: usize = request.model_slot;
            if (model_index >= model_slot_count) return error.InvalidModelSlot;
            if (request.image.handle == 0) return error.InvalidImageRange;

            const expected = try model_spec.planModelImage(
                request.spec_id,
                request.weight_format,
                0,
            );
            if (request.image.nbytes != expected.image_bytes)
                return error.InvalidImageRange;
            const base = heap.deviceAddress(request.image) catch
                return error.HeapFailure;
            const plan = try model_spec.planModelImage(
                request.spec_id,
                request.weight_format,
                base,
            );
            const spec = plan.spec();
            const install_request = state_mod.ModelInstall{
                .header = .{
                    .model_hash = request.model_hash,
                    .image_layout_hash = state_mod.image_layout_hash,
                    .embedding = plan.embedding,
                    .lm_head = plan.lm_head,
                    .output_norm = plan.output_norm,
                    .rope_table = plan.rope_table,
                    .interface_version = state_mod.interface_version,
                    .model_slot = request.model_slot,
                    .spec_id = @intFromEnum(request.spec_id),
                    .layer_count = spec.layers,
                    .weight_format = @intFromEnum(request.weight_format),
                },
                .layers = plan.layers(),
            };
            _ = try self.registry.install(install_request);
            self.model_ranges[model_index] = request.image;
        }

        /// Unpublish the image. The caller may free its heap handle only after
        /// this succeeds; images remain host-owned because the install RPC can
        /// name a subrange of a larger allocation.
        pub fn uninstall(self: *Self, model_slot: u16) Error!wire.TensorRange {
            const model_index: usize = model_slot;
            if (model_index >= model_slot_count) return error.InvalidModelSlot;
            try self.registry.uninstall(model_slot);
            const range = self.model_ranges[model_index] orelse
                return error.InvalidImageRange;
            self.model_ranges[model_index] = null;
            return range;
        }

        pub fn open(
            self: *Self,
            heap: anytype,
            request: rpc.OpenSession,
        ) Error!state_mod.SessionSnapshot {
            const session_index: usize = request.session_slot;
            if (session_index >= session_slot_count)
                return error.InvalidSessionSlot;
            if (self.session_ranges[session_index] != null)
                return error.SessionSlotOccupied;
            const model_record = try self.registry.model(request.model_slot);
            const spec = model_record.spec();
            if (request.epoch == 0) return error.InvalidSessionEpoch;
            if (request.capacity_tokens == 0 or
                request.capacity_tokens > spec.context_length)
                return error.InvalidKvCapacity;
            const bytes = try spec.kvCacheBytes(request.capacity_tokens);
            const alignment: u32 = @intCast(try spec.kvRecordBytes());
            const range = heap.allocate(bytes, alignment) catch |err| switch (err) {
                error.OutOfMemory => return error.OutOfMemory,
                else => return error.HeapFailure,
            };
            errdefer heap.free(range.handle) catch {};
            const kv_base = heap.deviceAddress(range) catch
                return error.HeapFailure;
            const snapshot = try self.registry.open(.{
                .kv_base = kv_base,
                .kv_capacity_tokens = request.capacity_tokens,
                .epoch = request.epoch,
                .model_slot = request.model_slot,
                .session_slot = request.session_slot,
            });
            self.session_ranges[session_index] = range;
            return snapshot;
        }

        pub fn close(
            self: *Self,
            heap: anytype,
            session_slot: u16,
        ) Error!void {
            const session_index: usize = session_slot;
            if (session_index >= session_slot_count)
                return error.InvalidSessionSlot;
            const range = self.session_ranges[session_index] orelse
                return error.HeapFailure;
            const snapshot = try self.registry.session(session_slot);
            if (snapshot.in_flight != 0 or
                self.registry.active_session_slot == session_slot)
                return error.SessionBusy;
            // Free can fail, so keep the registry publication intact until the
            // heap confirms release.  With the preflight above, close cannot
            // fail in this single-threaded manager.
            heap.free(range.handle) catch return error.HeapFailure;
            self.registry.close(session_slot) catch unreachable;
            self.session_ranges[session_index] = null;
        }

        pub fn sessionRange(
            self: *const Self,
            session_slot: u16,
        ) Error!wire.TensorRange {
            const session_index: usize = session_slot;
            if (session_index >= session_slot_count)
                return error.InvalidSessionSlot;
            return self.session_ranges[session_index] orelse
                error.SessionSlotEmpty;
        }

        pub fn model(
            self: *const Self,
            model_slot: u16,
        ) Error!*const state_mod.ModelRecord {
            return self.registry.model(model_slot);
        }

        pub fn session(
            self: *const Self,
            session_slot: u16,
        ) Error!state_mod.SessionSnapshot {
            return self.registry.session(session_slot);
        }

        pub fn begin(
            self: *Self,
            command: shared.engine.command.ExecuteTile,
        ) Error!state_mod.ExecutionLease {
            return self.registry.begin(command);
        }

        pub fn commit(
            self: *Self,
            request: state_mod.CommitRequest,
        ) Error!state_mod.SessionSnapshot {
            return self.registry.commit(request);
        }

        pub fn abort(
            self: *Self,
            request: state_mod.AbortRequest,
        ) Error!state_mod.SessionSnapshot {
            return self.registry.abort(request);
        }

        pub fn reset(
            self: *Self,
            request: state_mod.ResetSessionRequest,
        ) Error!state_mod.SessionSnapshot {
            return self.registry.reset(request);
        }
    };
}

pub const DefaultManager = Manager(
    state_mod.default_model_slots,
    state_mod.default_session_slots,
);

test {
    _ = std;
}
