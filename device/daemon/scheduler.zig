//! Transactional model and session scheduling for the inference engine.

const std = @import("std");
const engine = @import("engine");
const shared = @import("shared");

const capabilities = shared.capabilities;
const rpc = shared.engine.rpc;

pub const Error = engine.manager.Error || error{
    UnsupportedOp,
    BackendFailure,
};

pub fn SchedulerFor(comptime HardwareDriver: type) type {
    const has_hardware = HardwareDriver != void;

    return struct {
        const Self = @This();

        pub const inference_available = has_hardware;

        hardware: if (has_hardware) HardwareDriver else void,
        manager: engine.manager.DefaultManager = .{},
        quarantined: bool = false,

        pub fn init(
            hardware: if (has_hardware) HardwareDriver else void,
        ) Self {
            return .{ .hardware = hardware };
        }

        pub fn deinit(self: *Self) void {
            if (comptime has_hardware) self.hardware.deinit();
            self.* = undefined;
        }

        pub fn ownsHandle(self: *const Self, handle: u64) bool {
            return self.manager.ownsHandle(handle);
        }

        pub fn addCapabilities(
            self: *const Self,
            report: *capabilities.Report,
        ) void {
            if (comptime has_hardware) {
                report.feature_mask = capabilities.Feature.inference |
                    self.hardware.metricsFeatureMask();
                report.format_mask = engine.driver.formatMask(self.hardware);
                report.engine = engine.driver.capabilityInfo(self.hardware);
            }
        }

        pub fn installModel(
            self: *Self,
            heap: anytype,
            install: rpc.ModelInstall,
        ) Error!void {
            try self.requireHardware();
            heap.syncToDevice(install.image) catch return error.BackendFailure;
            try self.manager.install(heap, install);
            errdefer if (!self.quarantined and
                !self.hardware.isQuarantined())
            {
                _ = self.manager.uninstall(install.model_slot) catch {};
            };

            const model = self.manager.model(install.model_slot) catch
                return error.BackendFailure;
            self.hardware.programModel(model) catch |err| {
                self.captureHardwareFailure("model install", err);
                return error.BackendFailure;
            };
        }

        pub fn uninstallModel(
            self: *Self,
            model_slot: u16,
        ) Error!void {
            try self.requireHardware();
            _ = try self.manager.uninstall(model_slot);
            self.hardware.invalidateModel(model_slot);
        }

        pub fn openSession(
            self: *Self,
            heap: anytype,
            request: rpc.OpenSession,
        ) Error!void {
            try self.requireHardware();
            _ = try self.manager.open(heap, request);
        }

        pub fn closeSession(
            self: *Self,
            heap: anytype,
            session_slot: u16,
        ) Error!void {
            try self.requireHardware();
            try self.manager.close(heap, session_slot);
        }

        pub fn resetSession(
            self: *Self,
            request: rpc.ResetSession,
        ) Error!void {
            try self.requireHardware();
            _ = try self.manager.reset(.{
                .next_epoch = request.next_epoch,
                .session_slot = request.session_slot,
            });
        }

        pub fn execute(
            self: *Self,
            request: shared.engine.command.ExecuteTile,
        ) Error!rpc.ExecuteResult {
            try self.requireHardware();

            const lease = try self.manager.begin(request);
            var lease_active = true;
            defer if (lease_active and
                !self.quarantined and
                !self.hardware.isQuarantined())
            {
                _ = self.manager.abort(.{
                    .request_id = request.request_id,
                    .session_slot = request.session_slot,
                }) catch {};
            };

            const model = self.manager.model(request.model_slot) catch
                return error.BackendFailure;
            const completion = self.hardware.execute(
                request,
                lease,
                model,
            ) catch |err| {
                self.captureHardwareFailure("execute", err);
                return error.BackendFailure;
            };
            const snapshot = self.manager.commit(.{
                .request_id = request.request_id,
                .session_slot = request.session_slot,
                .completed_layers = model.header.layer_count,
                .kv_writes_fenced = 1,
            }) catch return error.BackendFailure;
            lease_active = false;

            const has_token = completion.token != null;
            return .{
                .session_slot = request.session_slot,
                .flags = if (has_token) rpc.ResultFlags.has_token else 0,
                .epoch = snapshot.epoch,
                .committed_tokens = snapshot.committed_tokens,
                .token_id = completion.token orelse 0,
                .logit = if (has_token) completion.logit else 0,
                .status = completion.status,
                .cycles = completion.cycles,
                .metrics_level = request.metrics_level,
                .metrics_snapshot = completion.metrics_snapshot,
            };
        }

        fn requireHardware(self: *Self) Error!void {
            if (comptime !has_hardware) return error.UnsupportedOp;
            if (self.quarantined) return error.BackendFailure;
            if (self.hardware.isQuarantined()) {
                self.quarantined = true;
                return error.BackendFailure;
            }
        }

        fn captureHardwareFailure(
            self: *Self,
            operation: []const u8,
            err: anyerror,
        ) void {
            if (self.hardware.isQuarantined()) self.quarantined = true;
            std.debug.print(
                "inference engine {s} failed: {s}\n",
                .{ operation, @errorName(err) },
            );
        }
    };
}

test {
    _ = engine;
}
