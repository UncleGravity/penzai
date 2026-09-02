//! Production AXI-Lite driver for the resident inference engine.
//!
//! Model configuration and execution are mailbox transactions. The driver never
//! acknowledges a successful EXEC until every expected field has been copied
//! and validated; failed transactions are run-cleared before returning.

const std = @import("std");
const shared = @import("shared");
const engine_regmap = @import("engine_regmap");
const regwin = @import("regwin.zig");
const state = @import("state.zig");

const capabilities = shared.capabilities;
const engine_command = shared.engine.command;
const engine_metrics = shared.engine.metrics;
const model_spec = shared.engine.model_spec;
const rpc = shared.engine.rpc;

pub const expected_id: u32 = 0xB05A_4000;
pub const expected_version: u32 = 0x0001_0007;
pub const expected_layout_hash: u64 = state.image_layout_hash;
pub const expected_axi_masters: u32 = 7;

comptime {
    std.debug.assert(expected_id == engine_regmap.resetOf("ID"));
    std.debug.assert(expected_version == engine_regmap.resetOf("VERSION"));
    std.debug.assert(@as(u32, @truncate(expected_layout_hash)) ==
        engine_regmap.resetOf("LAYOUT_HASH_LO"));
    std.debug.assert(@as(u32, @truncate(expected_layout_hash >> 32)) ==
        engine_regmap.resetOf("LAYOUT_HASH_HI"));
    std.debug.assert(expected_axi_masters ==
        engine_regmap.resetOf("AXI_MASTERS"));
}

pub const DriverError = error{
    BadId,
    BadVersion,
    BadLayout,
    BadClock,
    BadAxiMasters,
    InvalidLease,
    EngineBusy,
    StaleEvent,
    DriverTimeout,
    FrontendFailure,
    ProtocolFailure,
    ModelConfigurationFailure,
    RuntimeFailure,
    RecoveryFailure,
    ErrorTagMismatch,
    CommitMismatch,
    ResultMismatch,
    BadMetricsSchema,
    BadMetricsCapabilities,
    MetricsUnsupported,
    MetricsMismatch,
};

pub const Error = regwin.Error || DriverError;

const Ctrl = struct {
    const model_clear: u32 = 1 << 0;
    const model_begin: u32 = 1 << 1;
    const model_layer: u32 = 1 << 2;
    const model_seal: u32 = 1 << 3;
    const execute: u32 = 1 << 4;
    const run_clear: u32 = 1 << 5;
    const ack_commit: u32 = 1 << 6;
    const ack_error: u32 = 1 << 7;
    const ack_result: u32 = 1 << 8;
    const ack_model_error: u32 = 1 << 9;
    const ack_frontend_error: u32 = 1 << 10;
    const ack_metrics: u32 = 1 << 11;
};

const Status = struct {
    const busy: u32 = 1 << 0;
    const run_clear: u32 = 1 << 1;
    const clear_done: u32 = 1 << 2;
    const model_loading: u32 = 1 << 3;
    const model_sealed: u32 = 1 << 4;
    const model_op: u32 = 1 << 5;
    const exec_pending: u32 = 1 << 6;
    const commit_pending: u32 = 1 << 7;
    const error_pending: u32 = 1 << 8;
    const result_pending: u32 = 1 << 9;
    const model_error_pending: u32 = 1 << 10;
    const protocol_error: u32 = 1 << 11;
    const frontend_error_pending: u32 = 1 << 12;
    const cmd_ready: u32 = 1 << 13;
    const model_clear_ready: u32 = 1 << 14;
    const model_begin_ready: u32 = 1 << 15;
    const model_layer_ready: u32 = 1 << 16;
    const model_seal_ready: u32 = 1 << 17;
    const metrics_pending: u32 = 1 << 18;

    const mailbox_pending = commit_pending | error_pending | result_pending |
        model_error_pending | frontend_error_pending | metrics_pending;
};

const MetricsStatus = rpc.MetricsStatus;

pub const Diagnostic = struct {
    frontend_code: u8 = 0,
    model_code: u8 = 0,
    model_layer: u8 = 0,
    model_word: u8 = 0,
    runtime_tag: u32 = 0,
    runtime_code: u16 = 0,
    runtime_detail: u8 = 0,
    runtime_layer: u8 = 0,
    runtime_stage: u8 = 0,
    result_status: u8 = 0,
};

pub const Completion = struct {
    token: ?u32,
    logit: f32,
    status: u32,
    cycles: u64,
    committed_tokens: u32,
    metrics_snapshot: ?rpc.MetricsSnapshot,
};

const ActiveModel = struct {
    model_slot: u16,
    spec_id: u16,
    model_hash: u64,

    fn fromModel(model: *const state.ModelRecord) ActiveModel {
        return .{
            .model_slot = model.header.model_slot,
            .spec_id = model.header.spec_id,
            .model_hash = model.header.model_hash,
        };
    }
};

const ModelLoadOutcome = enum { cleared, begun, layer_written, sealed };

/// Generic register-window implementation.  The real `Driver` uses volatile
/// `/dev/mem`; tests provide a deterministic in-memory window.
pub fn DriverFor(comptime Window: type, comptime poll_limit: usize) type {
    return struct {
        const Self = @This();

        win: Window,
        version: u32,
        clk_hz: u32,
        active_model: ?ActiveModel = null,
        next_tag: u32 = 1,
        metrics_capabilities: u32,
        last_diagnostic: Diagnostic = .{},
        quarantined: bool = false,

        pub fn open(base: i64) Error!Self {
            var win = try Window.mapWindow(base);
            errdefer win.deinit();
            var self = try initWindow(win);
            try self.clearRun();
            return self;
        }

        pub fn initWindow(win: Window) DriverError!Self {
            if (win.rd(engine_regmap.offsetOf("ID")) != expected_id)
                return error.BadId;
            if (win.rd(engine_regmap.offsetOf("VERSION")) != expected_version)
                return error.BadVersion;
            const layout_hash = @as(u64, win.rd(
                engine_regmap.offsetOf("LAYOUT_HASH_LO"),
            )) | (@as(u64, win.rd(
                engine_regmap.offsetOf("LAYOUT_HASH_HI"),
            )) << 32);
            if (layout_hash != expected_layout_hash) return error.BadLayout;
            if (win.rd(engine_regmap.offsetOf("AXI_MASTERS")) !=
                expected_axi_masters) return error.BadAxiMasters;
            const clk_hz = win.rd(engine_regmap.offsetOf("CLK_HZ"));
            if (clk_hz == 0) return error.BadClock;
            var self: Self = .{
                .win = win,
                .version = expected_version,
                .clk_hz = clk_hz,
                .metrics_capabilities = win.rd(engine_regmap.offsetOf("METRICS_CAPABILITIES")),
            };
            try self.validateMetricsLevel(.none);
            return self;
        }

        pub fn deinit(self: *Self) void {
            self.win.deinit();
            self.* = undefined;
        }

        pub fn metricsFeatureMask(self: *const Self) u32 {
            _ = self;
            return capabilities.Feature.metrics_summary;
        }

        pub fn invalidateModel(self: *Self, model_slot: u16) void {
            if (self.active_model) |active| {
                if (active.model_slot == model_slot) self.active_model = null;
            }
        }

        pub fn isQuarantined(self: *const Self) bool {
            return self.quarantined;
        }

        /// Load every immutable address word and seal the spec.  Any rejected
        /// word invalidates the software cache and leaves the engine run-cleared.
        pub fn programModel(
            self: *Self,
            model: *const state.ModelRecord,
        ) DriverError!void {
            if (self.quarantined) return error.RecoveryFailure;
            return self.programModelActive(model) catch |err| {
                if (!self.recover()) return error.RecoveryFailure;
                return err;
            };
        }

        fn programModelActive(
            self: *Self,
            model: *const state.ModelRecord,
        ) DriverError!void {
            self.last_diagnostic = .{};
            self.active_model = null;

            const spec = model.spec();
            if (model.header.interface_version != expected_version or
                model.header.image_layout_hash != expected_layout_hash or
                model.header.model_hash == 0 or
                model.header.layer_count != spec.layers or
                model.layers().len != spec.layers or
                spec.key_head_dim != spec.value_head_dim)
                return error.InvalidLease;

            try self.issueModelOp(
                Ctrl.model_clear,
                Status.model_clear_ready,
                .cleared,
            );

            self.wr("MODEL_SPEC_ID", model.header.spec_id);
            self.wr64(
                "MODEL_SPEC_HASH_LO",
                "MODEL_SPEC_HASH_HI",
                model.header.model_hash,
            );
            self.wr("MODEL_SPEC_LAYER_COUNT", spec.layers);
            self.wr("MODEL_SPEC_HIDDEN_BLOCKS", spec.model_dim / 32);
            self.wr("MODEL_SPEC_FFN_BLOCKS", spec.ffn_dim / 32);
            self.wr("MODEL_SPEC_Q_HEADS", spec.query_heads);
            self.wr("MODEL_SPEC_KV_HEADS", spec.kv_heads);
            self.wr("MODEL_SPEC_HEAD_DIM", spec.key_head_dim);
            self.wr("MODEL_SPEC_WEIGHT_FMT", model.header.weight_format);
            self.wr("MODEL_SPEC_CONTEXT_LIMIT", spec.context_length);
            self.wr("MODEL_SPEC_VOCAB_ROWS", spec.vocab_size);
            self.wr64(
                "MODEL_SPEC_EMBED_ADDR_LO",
                "MODEL_SPEC_EMBED_ADDR_HI",
                model.header.embedding,
            );
            self.wr64(
                "MODEL_SPEC_LM_ADDR_LO",
                "MODEL_SPEC_LM_ADDR_HI",
                model.header.lm_head,
            );
            self.wr64(
                "MODEL_SPEC_FINAL_ADDR_LO",
                "MODEL_SPEC_FINAL_ADDR_HI",
                model.header.output_norm,
            );
            self.wr64(
                "MODEL_SPEC_ROPE_ADDR_LO",
                "MODEL_SPEC_ROPE_ADDR_HI",
                model.header.rope_table,
            );
            try self.issueModelOp(
                Ctrl.model_begin,
                Status.model_begin_ready,
                .begun,
            );

            for (model.layers(), 0..) |layer, layer_index| {
                const words = [_]u64{
                    layer.fused_qkv,
                    layer.attention_output,
                    layer.fused_gate_up,
                    layer.ffn_down,
                    layer.attention_norm,
                    layer.attention_q_norm,
                    layer.attention_k_norm,
                    layer.ffn_norm,
                };
                for (words, 0..) |word, word_index| {
                    self.wr("MODEL_SPEC_LAYER_INDEX", layer_index);
                    self.wr("MODEL_SPEC_LAYER_WORD", word_index);
                    self.wr64(
                        "MODEL_SPEC_LAYER_DATA_LO",
                        "MODEL_SPEC_LAYER_DATA_HI",
                        word,
                    );
                    try self.issueModelOp(
                        Ctrl.model_layer,
                        Status.model_layer_ready,
                        .layer_written,
                    );
                }
            }
            try self.issueModelOp(
                Ctrl.model_seal,
                Status.model_seal_ready,
                .sealed,
            );

            const status_value = try self.cleanStatus();
            if (status_value & Status.model_sealed == 0 or
                self.rd("ACTIVE_MODEL_SPEC_ID") != model.header.spec_id or
                self.rd64("ACTIVE_MODEL_SPEC_HASH_LO", "ACTIVE_MODEL_SPEC_HASH_HI") !=
                    model.header.model_hash)
                return error.ModelConfigurationFailure;
            self.active_model = ActiveModel.fromModel(model);
        }

        pub fn ensureModel(
            self: *Self,
            model: *const state.ModelRecord,
        ) DriverError!void {
            if (self.quarantined) return error.RecoveryFailure;
            const wanted = ActiveModel.fromModel(model);
            const status_value = self.status();
            if (self.active_model != null and
                std.meta.eql(self.active_model.?, wanted) and
                status_value & (Status.model_sealed | Status.mailbox_pending |
                    Status.protocol_error | Status.busy | Status.run_clear |
                    Status.exec_pending | Status.model_op) ==
                    Status.model_sealed and
                self.rd("ACTIVE_MODEL_SPEC_ID") == wanted.spec_id and
                self.rd64("ACTIVE_MODEL_SPEC_HASH_LO", "ACTIVE_MODEL_SPEC_HASH_HI") ==
                    wanted.model_hash)
                return;
            try self.programModel(model);
        }

        /// Execute one decode/prefill tile against an already accepted daemon
        /// lease.  The returned completion is safe for session publication.
        pub fn execute(
            self: *Self,
            request: engine_command.ExecuteTile,
            lease: state.ExecutionLease,
            model: *const state.ModelRecord,
        ) DriverError!Completion {
            if (self.quarantined) return error.RecoveryFailure;
            try self.ensureModel(model);
            return self.executeActive(request, lease, model) catch |err| {
                if (!self.recover()) return error.RecoveryFailure;
                return err;
            };
        }

        pub fn clearRun(self: *Self) DriverError!void {
            self.control(Ctrl.run_clear);
            var count: usize = 0;
            while (count < poll_limit) : (count += 1) {
                const value = self.status();
                if (value & (Status.busy | Status.run_clear |
                    Status.exec_pending | Status.model_op) == 0 and
                    value & Status.clear_done != 0)
                {
                    return self.drainMailboxes();
                }
            }
            return error.DriverTimeout;
        }

        fn executeActive(
            self: *Self,
            request: engine_command.ExecuteTile,
            lease: state.ExecutionLease,
            model: *const state.ModelRecord,
        ) DriverError!Completion {
            self.last_diagnostic = .{};
            try validateLease(request, lease, model);
            try self.validateMetricsLevel(request.metrics_level);
            try self.waitCommandReady();

            const tag = self.allocateTag();
            const emit_logits = request.flags & engine_command.Flags.emit_logits != 0;
            const lane_mask: u8 = @intCast(
                (@as(u16, 1) << @intCast(request.valid_tokens)) - 1,
            );
            self.wr("CMD_TAG", tag);
            self.wr("CMD_MODEL_SPEC_ID", model.header.spec_id);
            self.wr64(
                "CMD_MODEL_SPEC_HASH_LO",
                "CMD_MODEL_SPEC_HASH_HI",
                model.header.model_hash,
            );
            self.wr(
                "CMD_SHAPE",
                @as(u32, request.valid_tokens) |
                    (@as(u32, lane_mask) << 8) |
                    (@as(u32, @intFromBool(emit_logits)) << 16),
            );
            self.wr("CMD_POSITION_BASE", request.first_position);
            self.wr64("CMD_KV_BASE_LO", "CMD_KV_BASE_HI", lease.kv_base);
            self.wr("CMD_KV_CAPACITY", lease.kv_capacity_tokens);
            inline for (0..model_spec.token_tile_max) |index| {
                self.win.wr(
                    engine_regmap.offsetOf(std.fmt.comptimePrint(
                        "CMD_TOKEN{d}",
                        .{index},
                    )),
                    request.token_ids[index],
                );
            }
            self.control(Ctrl.execute);
            return self.waitCompletion(tag, request, lease, model, emit_logits);
        }

        fn waitCompletion(
            self: *Self,
            tag: u32,
            request: engine_command.ExecuteTile,
            lease: state.ExecutionLease,
            model: *const state.ModelRecord,
            emit_logits: bool,
        ) DriverError!Completion {
            var have_commit = false;
            var have_result = false;
            var have_metrics = false;
            var token: u32 = 0;
            var logit: f32 = 0;
            var metrics_snapshot: ?rpc.MetricsSnapshot = null;
            var count: usize = 0;
            while (count < poll_limit) : (count += 1) {
                const value = self.status();
                try self.checkFatalStatus(value);
                if (value & Status.error_pending != 0) {
                    self.captureRuntimeDiagnostic();
                    if (self.last_diagnostic.runtime_tag != tag)
                        return error.ErrorTagMismatch;
                    return error.RuntimeFailure;
                }
                if (value & Status.commit_pending != 0 and !have_commit) {
                    try self.validateCommit(tag, request, lease, model, emit_logits);
                    have_commit = true;
                }
                if (value & Status.result_pending != 0 and !have_result) {
                    const result = try self.validateResult(model, emit_logits);
                    token = result.token;
                    logit = result.logit;
                    have_result = true;
                }
                if (value & Status.metrics_pending != 0 and !have_metrics) {
                    metrics_snapshot = try self.readMetricsSnapshot(
                        tag,
                        request.metrics_level,
                        .commit,
                    );
                    have_metrics = true;
                }
                if (have_commit and have_metrics and (!emit_logits or have_result)) {
                    if (!emit_logits and value & Status.result_pending != 0)
                        return error.ResultMismatch;
                    const ack = Ctrl.ack_commit |
                        (if (emit_logits) Ctrl.ack_result else 0) |
                        Ctrl.ack_metrics;
                    self.control(ack);
                    try self.waitMailboxesEmpty();
                    return .{
                        .token = if (emit_logits) token else null,
                        .logit = if (emit_logits) logit else 0,
                        .status = 0,
                        .cycles = if (metrics_snapshot) |snapshot|
                            snapshot.values[@intFromEnum(engine_metrics.MetricId.total_cycles)]
                        else
                            self.rd64("CYCLES_LO", "CYCLES_HI"),
                        .committed_tokens = lease.pending_end,
                        .metrics_snapshot = metrics_snapshot,
                    };
                }
            }
            return error.DriverTimeout;
        }

        fn validateCommit(
            self: *Self,
            tag: u32,
            request: engine_command.ExecuteTile,
            lease: state.ExecutionLease,
            model: *const state.ModelRecord,
            emit_logits: bool,
        ) DriverError!void {
            const info = self.rd("COMMIT_INFO");
            const token_count: u8 = @truncate(info & 0xf);
            const kv_length: u32 = (info >> 4) & 0x1ffff;
            const logits_valid = info & (1 << 21) != 0;
            if (self.rd("COMMIT_TAG") != tag or
                self.rd("COMMIT_MODEL_SPEC_ID") != model.header.spec_id or
                self.rd64("COMMIT_MODEL_SPEC_HASH_LO", "COMMIT_MODEL_SPEC_HASH_HI") !=
                    model.header.model_hash or
                token_count != request.valid_tokens or
                kv_length != lease.pending_end or
                logits_valid != emit_logits)
                return error.CommitMismatch;
        }

        const Result = struct { token: u32, logit: f32 };

        fn validateResult(
            self: *Self,
            model: *const state.ModelRecord,
            emit_logits: bool,
        ) DriverError!Result {
            const value = self.rd("RESULT_TOKEN_STATUS");
            const token = value & 0x3ffff;
            const result_status: u8 = @truncate((value >> 18) & 0xff);
            const result_error = value & (1 << 26) != 0;
            const logit: f32 = @bitCast(self.rd("RESULT_LOGIT"));
            self.last_diagnostic.result_status = result_status;
            if (!emit_logits or result_error or result_status != 0 or
                token >= model.spec().vocab_size or
                !std.math.isFinite(logit))
                return error.ResultMismatch;
            return .{ .token = token, .logit = logit };
        }

        fn issueModelOp(
            self: *Self,
            command: u32,
            ready: u32,
            outcome: ModelLoadOutcome,
        ) DriverError!void {
            var count: usize = 0;
            while (count < poll_limit) : (count += 1) {
                const value = try self.cleanStatus();
                if (value & (Status.busy | Status.run_clear |
                    Status.exec_pending | Status.model_op) == 0 and
                    value & ready != 0)
                    break;
            } else return error.DriverTimeout;

            self.control(command);
            count = 0;
            while (count < poll_limit) : (count += 1) {
                const value = try self.cleanStatus();
                if (value & Status.model_op != 0) continue;
                const accepted = switch (outcome) {
                    .cleared => value & (Status.model_loading |
                        Status.model_sealed) == 0,
                    .begun, .layer_written => value & Status.model_loading != 0 and
                        value & Status.model_sealed == 0,
                    .sealed => value & Status.model_sealed != 0 and
                        value & Status.model_loading == 0,
                };
                if (accepted) return;
            }
            return error.DriverTimeout;
        }

        fn waitCommandReady(self: *Self) DriverError!void {
            var count: usize = 0;
            while (count < poll_limit) : (count += 1) {
                const value = try self.cleanStatus();
                if (value & Status.cmd_ready != 0 and
                    value & (Status.busy | Status.run_clear |
                        Status.exec_pending | Status.model_op) == 0)
                    return;
            }
            return error.DriverTimeout;
        }

        fn cleanStatus(self: *Self) DriverError!u32 {
            const value = self.status();
            try self.checkFatalStatus(value);
            if (value & (Status.commit_pending | Status.error_pending |
                Status.result_pending | Status.metrics_pending) != 0)
                return error.StaleEvent;
            return value;
        }

        fn checkFatalStatus(self: *Self, value: u32) DriverError!void {
            if (value & Status.frontend_error_pending != 0) {
                self.last_diagnostic.frontend_code = @truncate(
                    self.rd("FRONTEND_ERROR"),
                );
                return error.FrontendFailure;
            }
            if (value & Status.protocol_error != 0)
                return error.ProtocolFailure;
            if (value & Status.model_error_pending != 0) {
                const word = self.rd("MODEL_SPEC_ERROR");
                self.last_diagnostic.model_code = @truncate(word & 0xff);
                self.last_diagnostic.model_layer = @truncate((word >> 8) & 0x3f);
                self.last_diagnostic.model_word = @truncate((word >> 16) & 0x7);
                return error.ModelConfigurationFailure;
            }
        }

        fn captureRuntimeDiagnostic(self: *Self) void {
            const code_detail = self.rd("ERROR_CODE_DETAIL");
            const location = self.rd("ERROR_LOCATION");
            self.last_diagnostic.runtime_tag = self.rd("ERROR_TAG");
            self.last_diagnostic.runtime_code = @truncate(code_detail & 0xffff);
            self.last_diagnostic.runtime_detail = @truncate((code_detail >> 16) & 0xff);
            self.last_diagnostic.runtime_layer = @truncate(location & 0x3f);
            self.last_diagnostic.runtime_stage = @truncate((location >> 8) & 0x1f);
        }

        fn waitMailboxesEmpty(self: *Self) DriverError!void {
            var count: usize = 0;
            while (count < poll_limit) : (count += 1) {
                const value = self.status();
                if (value & Status.mailbox_pending == 0) return;
            }
            return error.DriverTimeout;
        }

        fn drainMailboxes(self: *Self) DriverError!void {
            var count: usize = 0;
            while (count < poll_limit) : (count += 1) {
                const value = self.status();
                if (value & Status.mailbox_pending == 0) return;
                if (value & Status.metrics_pending != 0)
                    _ = try self.readMetricsSnapshot(null, .none, null);
                self.ackPending(value);
            }
            return error.DriverTimeout;
        }

        fn recover(self: *Self) bool {
            if (self.quarantined) return false;
            // Model configuration operations are one-entry pulses. Give an accepted pulse
            // time to retire before issuing RUN_CLEAR, which the wrapper rejects
            // while a spec operation is live.
            var count: usize = 0;
            while (count < poll_limit and
                self.status() & Status.model_op != 0) : (count += 1)
            {}
            self.control(Ctrl.run_clear);
            count = 0;
            while (count < poll_limit) : (count += 1) {
                const value = self.status();
                if (value & (Status.busy | Status.run_clear |
                    Status.exec_pending | Status.model_op) == 0 and
                    value & Status.clear_done != 0)
                {
                    self.drainMailboxes() catch {
                        self.quarantined = true;
                        return false;
                    };
                    return true;
                }
            }
            self.quarantined = true;
            return false;
        }

        fn ackPending(self: *Self, value: u32) void {
            var ack: u32 = 0;
            if (value & Status.commit_pending != 0) ack |= Ctrl.ack_commit;
            if (value & Status.error_pending != 0) ack |= Ctrl.ack_error;
            if (value & Status.result_pending != 0) ack |= Ctrl.ack_result;
            if (value & Status.model_error_pending != 0)
                ack |= Ctrl.ack_model_error;
            if (value & Status.frontend_error_pending != 0)
                ack |= Ctrl.ack_frontend_error;
            if (value & Status.metrics_pending != 0) ack |= Ctrl.ack_metrics;
            if (ack != 0) self.control(ack);
        }

        fn validateMetricsLevel(
            self: *Self,
            level: engine_metrics.Level,
        ) DriverError!void {
            if (self.rd("METRICS_SCHEMA") != engine_metrics.recorder_schema)
                return error.BadMetricsSchema;
            const actual = self.rd("METRICS_CAPABILITIES");
            if (!rpc.validMetricsCapabilities(actual))
                return error.BadMetricsCapabilities;
            self.metrics_capabilities = actual;
            if (level == .full)
                return error.MetricsUnsupported;
        }

        fn readMetricsSnapshot(
            self: *Self,
            expected_tag: ?u32,
            level: engine_metrics.Level,
            expected_outcome: ?engine_metrics.Outcome,
        ) DriverError!?rpc.MetricsSnapshot {
            try self.validateMetricsLevel(level);
            const status_value = self.rd("METRICS_STATUS");
            const tag = self.rd("METRICS_TAG");
            rpc.validateMetricsEnvelope(
                engine_metrics.recorder_schema,
                self.metrics_capabilities,
                status_value,
                tag,
                expected_outcome,
            ) catch return error.MetricsMismatch;
            if (expected_tag != null and tag != expected_tag.?)
                return error.MetricsMismatch;

            if (level == .none) return null;

            var snapshot: rpc.MetricsSnapshot = .{
                .schema = engine_metrics.recorder_schema,
                .capabilities = self.metrics_capabilities,
                .status = status_value,
                .tag = tag,
                .overflow = .{
                    self.rd("METRICS_OVERFLOW0"),
                    self.rd("METRICS_OVERFLOW1"),
                    self.rd("METRICS_OVERFLOW2"),
                    self.rd("METRICS_OVERFLOW3"),
                },
                .values = undefined,
            };
            for (&snapshot.values, 0..) |*metric_value, index| {
                self.wr("METRICS_INDEX", index);
                const lo = self.rd("METRICS_DATA_LO");
                const hi = self.rd("METRICS_DATA_HI");
                if (index != @intFromEnum(engine_metrics.MetricId.total_cycles) and hi != 0)
                    return error.MetricsMismatch;
                metric_value.* = @as(u64, lo) | (@as(u64, hi) << 32);
            }
            rpc.validateMetricsSnapshot(
                snapshot,
                self.rd64("CYCLES_LO", "CYCLES_HI"),
            ) catch return error.MetricsMismatch;
            return snapshot;
        }

        fn allocateTag(self: *Self) u32 {
            const tag = self.next_tag;
            self.next_tag +%= 1;
            if (self.next_tag == 0) self.next_tag = 1;
            return tag;
        }

        inline fn status(self: *Self) u32 {
            return self.rd("STATUS");
        }

        inline fn control(self: *Self, value: u32) void {
            self.wr("CTRL", value);
        }

        inline fn rd(self: *Self, comptime name: []const u8) u32 {
            return self.win.rd(engine_regmap.offsetOf(name));
        }

        inline fn wr(
            self: *Self,
            comptime name: []const u8,
            value: anytype,
        ) void {
            self.win.wr(engine_regmap.offsetOf(name), @intCast(value));
        }

        inline fn rd64(
            self: *Self,
            comptime lo: []const u8,
            comptime hi: []const u8,
        ) u64 {
            return @as(u64, self.rd(lo)) | (@as(u64, self.rd(hi)) << 32);
        }

        inline fn wr64(
            self: *Self,
            comptime lo: []const u8,
            comptime hi: []const u8,
            value: u64,
        ) void {
            self.wr(lo, @as(u32, @truncate(value)));
            self.wr(hi, @as(u32, @truncate(value >> 32)));
        }
    };
}

pub const Driver = DriverFor(regwin.RegWindow, regwin.wait_limit);

pub fn capabilityInfo(kernel: Driver) capabilities.EngineInfo {
    return .{
        .interface_id = expected_id,
        .interface_version = kernel.version,
        .clock_hz = kernel.clk_hz,
        .token_tile_max = model_spec.token_tile_max,
        .token_lanes = model_spec.physical_token_lanes,
        .model_spec_count = model_spec.all_specs.len,
        .context_tokens_max = model_spec.bonsai_8b.context_length,
        .address_record_bytes = model_spec.LayerAddresses.encoded_bytes,
    };
}

pub fn formatMask(_: Driver) u32 {
    return capabilities.Format.weight_q1_0 |
        capabilities.Format.weight_q2_0_g64 |
        capabilities.Format.activation_q8_0 |
        capabilities.Format.io_f32 |
        capabilities.Format.kv_f16;
}

fn validateLease(
    request: engine_command.ExecuteTile,
    lease: state.ExecutionLease,
    model: *const state.ModelRecord,
) DriverError!void {
    const pending_end = std.math.add(
        u32,
        request.first_position,
        request.valid_tokens,
    ) catch return error.InvalidLease;
    if (request.request_id != lease.request_id or
        request.model_slot != lease.model_slot or
        request.session_slot != lease.session_slot or
        request.session_epoch != lease.session_epoch or
        request.first_position != lease.first_position or
        request.valid_tokens != lease.valid_tokens or
        request.flags != lease.flags or
        request.first_position != lease.committed_tokens or
        pending_end != lease.pending_end or
        lease.spec_id != model.header.spec_id or
        lease.model_hash != model.header.model_hash or
        lease.model_slot != model.header.model_slot or
        lease.contract_version != model_spec.contract_version or
        lease.weight_format != model.header.weight_format or
        lease.kv_base == 0 or lease.kv_base & 0xfff != 0 or
        lease.kv_capacity_tokens == 0 or
        lease.kv_capacity_tokens > 0x1ffff or
        lease.kv_capacity_tokens > model.spec().context_length or
        pending_end > lease.kv_capacity_tokens)
        return error.InvalidLease;
}

const FakeMmio = struct {
    regs: [0x200 / 4]u32 = @splat(0),
    loading: bool = false,
    sealed: bool = false,
    commit_pending: bool = false,
    error_pending: bool = false,
    result_pending: bool = false,
    metrics_pending: bool = false,
    model_error_pending: bool = false,
    frontend_error_pending: bool = false,
    layer_writes: u32 = 0,
    execute_count: u32 = 0,
    metrics_index_writes: u32 = 0,
    run_clear_count: u32 = 0,
    corrupt_commit_tag: bool = false,
    fail_runtime: bool = false,
    clear_never_completes: bool = false,
    next_result_token: u32 = 42,
    next_result_logit: f32 = 1.25,
    metric_values: [engine_metrics.metric_count]u64 = @splat(0),

    fn init() FakeMmio {
        var self: FakeMmio = .{};
        self.put("ID", expected_id);
        self.put("VERSION", expected_version);
        self.put("LAYOUT_HASH_LO", @truncate(expected_layout_hash));
        self.put("LAYOUT_HASH_HI", @truncate(expected_layout_hash >> 32));
        self.put("CLK_HZ", 285_000_000);
        self.put("AXI_MASTERS", expected_axi_masters);
        self.put("METRICS_SCHEMA", engine_metrics.recorder_schema);
        self.put("METRICS_CAPABILITIES", engine_metrics.compiled_hardware_capabilities);
        self.put("METRICS_STATUS", MetricsStatus.enabled);
        return self;
    }

    fn status(self: *const FakeMmio) u32 {
        var value: u32 = Status.model_clear_ready;
        if (!self.clear_never_completes) value |= Status.clear_done;
        if (self.loading) value |= Status.model_loading |
            Status.model_layer_ready | Status.model_seal_ready;
        if (self.sealed) value |= Status.model_sealed | Status.cmd_ready;
        if (!self.loading and !self.sealed) value |= Status.model_begin_ready;
        if (self.commit_pending) value |= Status.commit_pending;
        if (self.error_pending) value |= Status.error_pending;
        if (self.result_pending) value |= Status.result_pending;
        if (self.metrics_pending) value |= Status.metrics_pending;
        if (self.model_error_pending) value |= Status.model_error_pending;
        if (self.frontend_error_pending) value |= Status.frontend_error_pending;
        return value;
    }

    fn rd(self: *FakeMmio, offset: u32) u32 {
        if (offset == engine_regmap.offsetOf("STATUS")) return self.status();
        if (offset == engine_regmap.offsetOf("METRICS_DATA_LO") or
            offset == engine_regmap.offsetOf("METRICS_DATA_HI"))
        {
            const index = self.get("METRICS_INDEX");
            if (index >= engine_metrics.metric_count) return 0;
            const value = self.metric_values[index];
            return if (offset == engine_regmap.offsetOf("METRICS_DATA_LO"))
                @truncate(value)
            else
                @truncate(value >> 32);
        }
        return self.regs[offset / 4];
    }

    fn wr(self: *FakeMmio, offset: u32, value: u32) void {
        if (offset != engine_regmap.offsetOf("CTRL")) {
            if (offset == engine_regmap.offsetOf("METRICS_INDEX"))
                self.metrics_index_writes += 1;
            self.regs[offset / 4] = value;
            return;
        }
        if (value & Ctrl.ack_commit != 0) self.commit_pending = false;
        if (value & Ctrl.ack_error != 0) self.error_pending = false;
        if (value & Ctrl.ack_result != 0) self.result_pending = false;
        if (value & Ctrl.ack_metrics != 0) {
            self.metrics_pending = false;
            self.put("METRICS_STATUS", MetricsStatus.enabled);
        }
        if (value & Ctrl.ack_model_error != 0)
            self.model_error_pending = false;
        if (value & Ctrl.ack_frontend_error != 0)
            self.frontend_error_pending = false;
        if (value & Ctrl.model_clear != 0) {
            self.loading = false;
            self.sealed = false;
            self.layer_writes = 0;
        } else if (value & Ctrl.model_begin != 0) {
            self.loading = true;
        } else if (value & Ctrl.model_layer != 0) {
            self.layer_writes += 1;
        } else if (value & Ctrl.model_seal != 0) {
            const expected = self.get("MODEL_SPEC_LAYER_COUNT") *
                model_spec.LayerAddresses.count;
            if (self.layer_writes != expected) {
                self.loading = false;
                self.model_error_pending = true;
                self.put("MODEL_SPEC_ERROR", 3);
            } else {
                self.loading = false;
                self.sealed = true;
                self.put("ACTIVE_MODEL_SPEC_ID", self.get("MODEL_SPEC_ID"));
                self.put("ACTIVE_MODEL_SPEC_HASH_LO", self.get("MODEL_SPEC_HASH_LO"));
                self.put("ACTIVE_MODEL_SPEC_HASH_HI", self.get("MODEL_SPEC_HASH_HI"));
            }
        } else if (value & Ctrl.execute != 0) {
            self.execute_count += 1;
            const shape = self.get("CMD_SHAPE");
            const count = shape & 0xf;
            const end = self.get("CMD_POSITION_BASE") + count;
            const capacity = self.get("CMD_KV_CAPACITY");
            if (capacity == 0 or capacity > 0x1ffff or
                capacity > self.get("MODEL_SPEC_CONTEXT_LIMIT") or
                end > capacity)
            {
                self.frontend_error_pending = true;
                self.put("FRONTEND_ERROR", 1);
                return;
            }
            if (self.fail_runtime) {
                self.error_pending = true;
                self.put("ERROR_TAG", self.get("CMD_TAG"));
                self.put("ERROR_CODE_DETAIL", 7);
            } else {
                const tag = self.get("CMD_TAG") +
                    @as(u32, @intFromBool(self.corrupt_commit_tag));
                self.put("COMMIT_TAG", tag);
                self.put("COMMIT_MODEL_SPEC_ID", self.get("CMD_MODEL_SPEC_ID"));
                self.put("COMMIT_MODEL_SPEC_HASH_LO", self.get("CMD_MODEL_SPEC_HASH_LO"));
                self.put("COMMIT_MODEL_SPEC_HASH_HI", self.get("CMD_MODEL_SPEC_HASH_HI"));
                self.put("COMMIT_INFO", count | (end << 4) |
                    (shape & (1 << 16)) << 5);
                self.commit_pending = true;
                if (shape & (1 << 16) != 0) {
                    self.put("RESULT_TOKEN_STATUS", self.next_result_token);
                    self.put("RESULT_LOGIT", @bitCast(self.next_result_logit));
                    self.result_pending = true;
                }
                self.put("CYCLES_LO", 1234);
                self.metric_values = @splat(0);
                self.metric_values[@intFromEnum(engine_metrics.MetricId.total_cycles)] = 1234;
                self.metric_values[@intFromEnum(engine_metrics.MetricId.control_cycles)] = 12;
                self.metric_values[@intFromEnum(engine_metrics.MetricId.projection_drain_cycles)] = 34;
                self.put("METRICS_TAG", self.get("CMD_TAG"));
                self.put(
                    "METRICS_STATUS",
                    MetricsStatus.valid | MetricsStatus.enabled |
                        (@as(u32, @intFromEnum(engine_metrics.Outcome.commit)) <<
                            MetricsStatus.outcome_shift),
                );
                self.metrics_pending = true;
            }
        } else if (value & Ctrl.run_clear != 0) {
            self.run_clear_count += 1;
        }
    }

    fn get(self: *const FakeMmio, comptime name: []const u8) u32 {
        return self.regs[engine_regmap.offsetOf(name) / 4];
    }

    fn put(self: *FakeMmio, comptime name: []const u8, value: u32) void {
        self.regs[engine_regmap.offsetOf(name) / 4] = value;
    }
};

const FakeWindow = struct {
    mmio: *FakeMmio,

    fn rd(self: FakeWindow, offset: u32) u32 {
        return self.mmio.rd(offset);
    }

    fn wr(self: FakeWindow, offset: u32, value: u32) void {
        self.mmio.wr(offset, value);
    }
};

fn testModel() !state.ModelRecord {
    const plan = try model_spec.planModelImage(
        .bonsai_1_7b,
        .q1_0,
        0x1000_0000,
    );
    return .{
        .header = .{
            .model_hash = 0x1234_5678_9abc_def0,
            .image_layout_hash = state.image_layout_hash,
            .embedding = plan.embedding,
            .lm_head = plan.lm_head,
            .output_norm = plan.output_norm,
            .rope_table = plan.rope_table,
            .interface_version = state.interface_version,
            .model_slot = 2,
            .spec_id = @intFromEnum(plan.spec_id),
            .layer_count = plan.layer_count,
            .weight_format = @intFromEnum(plan.weight_format),
        },
        .layer_storage = plan.layer_storage,
    };
}

fn testCommandAndLease() struct {
    command: engine_command.ExecuteTile,
    lease: state.ExecutionLease,
} {
    const command = engine_command.ExecuteTile{
        .request_id = 0x1_0000_0001,
        .model_slot = 2,
        .session_slot = 3,
        .session_epoch = 9,
        .first_position = 10,
        .valid_tokens = 4,
        .flags = engine_command.Flags.emit_logits,
        .metrics_level = .summary,
        .token_ids = .{ 1, 2, 3, 4, 0, 0, 0, 0 },
    };
    return .{
        .command = command,
        .lease = .{
            .request_id = command.request_id,
            .model_hash = 0x1234_5678_9abc_def0,
            .kv_base = 0x8000_0000,
            .first_position = command.first_position,
            .pending_end = 14,
            .kv_capacity_tokens = 128,
            .session_epoch = command.session_epoch,
            .model_slot = command.model_slot,
            .session_slot = command.session_slot,
            .spec_id = @intFromEnum(model_spec.SpecId.bonsai_1_7b),
            .contract_version = model_spec.contract_version,
            .weight_format = @intFromEnum(model_spec.WeightFormat.q1_0),
            .valid_tokens = command.valid_tokens,
            .flags = command.flags,
            .committed_tokens = command.first_position,
        },
    };
}

test "inference driver programs every immutable layer word and caches identity" {
    const TestDriver = DriverFor(FakeWindow, 64);
    var mmio = FakeMmio.init();
    var driver = try TestDriver.initWindow(.{ .mmio = &mmio });
    const model = try testModel();
    try driver.programModel(&model);
    try std.testing.expect(mmio.sealed);
    try std.testing.expectEqual(
        @as(u32, model.header.layer_count) *
            @as(u32, model_spec.LayerAddresses.count),
        mmio.layer_writes,
    );
    const writes = mmio.layer_writes;
    try driver.ensureModel(&model);
    try std.testing.expectEqual(writes, mmio.layer_writes);
}

test "inference driver requires validated commit and greedy result" {
    const TestDriver = DriverFor(FakeWindow, 64);
    var mmio = FakeMmio.init();
    var driver = try TestDriver.initWindow(.{ .mmio = &mmio });
    const model = try testModel();
    const request = testCommandAndLease();
    const completion = try driver.execute(
        request.command,
        request.lease,
        &model,
    );
    try std.testing.expectEqual(@as(?u32, 42), completion.token);
    try std.testing.expectEqual(@as(f32, 1.25), completion.logit);
    try std.testing.expectEqual(@as(u32, 14), completion.committed_tokens);
    try std.testing.expectEqual(@as(u64, 1234), completion.cycles);
    const snapshot = completion.metrics_snapshot.?;
    try std.testing.expectEqual(
        @as(u64, 34),
        snapshot.values[@intFromEnum(engine_metrics.MetricId.projection_drain_cycles)],
    );
    try std.testing.expectEqual(
        request.lease.kv_capacity_tokens,
        mmio.get("CMD_KV_CAPACITY"),
    );
    try std.testing.expectEqual(@as(u32, 1), mmio.execute_count);
    try std.testing.expect(!mmio.commit_pending and !mmio.result_pending and
        !mmio.metrics_pending);
    try std.testing.expectEqual(
        @as(u32, engine_metrics.metric_count),
        mmio.metrics_index_writes,
    );
}

test "metrics none validates and ACKs without indexed counter reads" {
    const TestDriver = DriverFor(FakeWindow, 64);
    var mmio = FakeMmio.init();
    var driver = try TestDriver.initWindow(.{ .mmio = &mmio });
    const model = try testModel();
    var request = testCommandAndLease();
    request.command.metrics_level = .none;
    const completion = try driver.execute(request.command, request.lease, &model);
    try std.testing.expect(completion.metrics_snapshot == null);
    try std.testing.expectEqual(@as(u32, 0), mmio.metrics_index_writes);
    try std.testing.expect(!mmio.metrics_pending);
}

test "metrics full remains reserved even when hardware sets its capability bit" {
    const TestDriver = DriverFor(FakeWindow, 64);
    var mmio = FakeMmio.init();
    mmio.put(
        "METRICS_CAPABILITIES",
        engine_metrics.compiled_hardware_capabilities |
            engine_metrics.HardwareCapability.full_bank,
    );
    var driver = try TestDriver.initWindow(.{ .mmio = &mmio });
    try std.testing.expectEqual(
        capabilities.Feature.metrics_summary,
        driver.metricsFeatureMask(),
    );
    const model = try testModel();
    var request = testCommandAndLease();
    request.command.metrics_level = .full;
    try std.testing.expectError(
        error.MetricsUnsupported,
        driver.execute(request.command, request.lease, &model),
    );
    try std.testing.expectEqual(@as(u32, 0), mmio.execute_count);
    try std.testing.expectEqual(@as(u32, 0), mmio.metrics_index_writes);
}

test "inference driver rejects KV capacity outside its lease and spec" {
    const TestDriver = DriverFor(FakeWindow, 64);
    var mmio = FakeMmio.init();
    var driver = try TestDriver.initWindow(.{ .mmio = &mmio });
    const model = try testModel();
    var request = testCommandAndLease();
    request.lease.kv_capacity_tokens = request.lease.pending_end - 1;
    try std.testing.expectError(
        error.InvalidLease,
        driver.execute(request.command, request.lease, &model),
    );
    try std.testing.expectEqual(@as(u32, 0), mmio.execute_count);
    try std.testing.expectEqual(@as(u32, 1), mmio.run_clear_count);

    request = testCommandAndLease();
    request.lease.kv_capacity_tokens = model.spec().context_length + 1;
    try std.testing.expectError(
        error.InvalidLease,
        driver.execute(request.command, request.lease, &model),
    );
    try std.testing.expectEqual(@as(u32, 0), mmio.execute_count);
    try std.testing.expectEqual(@as(u32, 2), mmio.run_clear_count);
}

test "inference driver run-clears and drains mailboxes after mismatch" {
    const TestDriver = DriverFor(FakeWindow, 64);
    var mmio = FakeMmio.init();
    var driver = try TestDriver.initWindow(.{ .mmio = &mmio });
    const model = try testModel();
    const request = testCommandAndLease();
    mmio.corrupt_commit_tag = true;
    try std.testing.expectError(
        error.CommitMismatch,
        driver.execute(request.command, request.lease, &model),
    );
    try std.testing.expectEqual(@as(u32, 1), mmio.run_clear_count);
    try std.testing.expect(!mmio.commit_pending and !mmio.result_pending);
}

test "inference driver accepts commit-only tiles without fabricating a result" {
    const TestDriver = DriverFor(FakeWindow, 64);
    var mmio = FakeMmio.init();
    var driver = try TestDriver.initWindow(.{ .mmio = &mmio });
    const model = try testModel();
    var request = testCommandAndLease();
    request.command.flags = 0;
    request.lease.flags = 0;
    const completion = try driver.execute(
        request.command,
        request.lease,
        &model,
    );
    try std.testing.expectEqual(@as(?u32, null), completion.token);
    try std.testing.expectEqual(@as(f32, 0), completion.logit);
    try std.testing.expect(!mmio.commit_pending and !mmio.result_pending);
}

test "inference driver captures runtime error before run-clear and ACK" {
    const TestDriver = DriverFor(FakeWindow, 64);
    var mmio = FakeMmio.init();
    var driver = try TestDriver.initWindow(.{ .mmio = &mmio });
    const model = try testModel();
    const request = testCommandAndLease();
    mmio.fail_runtime = true;
    try std.testing.expectError(
        error.RuntimeFailure,
        driver.execute(request.command, request.lease, &model),
    );
    try std.testing.expectEqual(@as(u16, 7), driver.last_diagnostic.runtime_code);
    try std.testing.expectEqual(@as(u32, 1), mmio.run_clear_count);
    try std.testing.expect(!mmio.error_pending);
    try std.testing.expect(!driver.isQuarantined());
}

test "inference driver quarantines when run-clear never completes" {
    const TestDriver = DriverFor(FakeWindow, 8);
    var mmio = FakeMmio.init();
    var driver = try TestDriver.initWindow(.{ .mmio = &mmio });
    const model = try testModel();
    const request = testCommandAndLease();
    mmio.fail_runtime = true;
    mmio.clear_never_completes = true;
    try std.testing.expectError(
        error.RecoveryFailure,
        driver.execute(request.command, request.lease, &model),
    );
    try std.testing.expect(driver.isQuarantined());
    try std.testing.expectEqual(@as(u32, 1), mmio.run_clear_count);
    try std.testing.expect(mmio.error_pending);

    try std.testing.expectError(
        error.RecoveryFailure,
        driver.execute(request.command, request.lease, &model),
    );
    try std.testing.expectEqual(@as(u32, 1), mmio.execute_count);
    try std.testing.expectEqual(@as(u32, 1), mmio.run_clear_count);
}
