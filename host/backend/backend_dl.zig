//! Out-of-tree ggml backend entry point — `libggml-penzai`.
//!
//! Stock llama.cpp loads this via `GGML_BACKEND_PATH=/path/to/libggml-penzai.(so|dylib)`
//! plus `--device penzai`: the loader dlsym's `ggml_backend_init`, takes the
//! `ggml_backend_reg` it returns, and registers it. Unlike the in-process `penzai`
//! binary there is no `main` to hand us an allocator / io / transport, so this shim
//! owns process-global state and lazily connects a `TcpLink` to the device daemon
//! (env `PENZAI_HOST` / `PENZAI_PORT`) on the first call. The compute backend is the
//! exact same `backend.Device` the in-process path uses — only the bootstrap differs.
//!
//! The Nix wrappers and in-process launcher supply the required device-selection
//! and offload flags; the shared library only provides the executor.
const std = @import("std");
const c = @import("c_ggml");
const shared = @import("shared");
const backend = @import("backend");
const link_mod = @import("link");

const protocol_transport = shared.protocol_transport;

// Built once on the first `ggml_backend_init`. `g_io` and `g_link` are globals
// (never moved) because the `Io` interface and the `Client` both keep stable
// pointers into them.
var g_io: std.Io.Threaded = undefined;
var g_link: link_mod.TcpLink = undefined;
var g_mock_link: MockLink = .{};
var g_device: ?*backend.Device = null;

const MockLink = struct {
    next_handle: u64 = 1,

    pub fn hello(_: *@This()) link_mod.LinkError!void {}

    pub fn capabilities(_: *@This()) link_mod.LinkError!shared.capabilities.Report {
        return .{
            .wire_abi = shared.wire.version,
            .metrics_schema = shared.engine.metrics.schema_version,
            .feature_mask = shared.capabilities.Feature.inference |
                shared.capabilities.Feature.metrics_summary,
            .format_mask = shared.capabilities.Format.weight_q1_0 |
                shared.capabilities.Format.weight_q2_0_g64 |
                shared.capabilities.Format.activation_q8_0 |
                shared.capabilities.Format.io_f32 |
                shared.capabilities.Format.kv_f16,
            .engine = .{
                .interface_id = backend.inference_engine_id,
                .interface_version = backend.inference_interface_version,
                .clock_hz = 225_000_000,
                .token_tile_max = shared.engine.model_spec.token_tile_max,
                .token_lanes = shared.engine.model_spec.physical_token_lanes,
                .model_spec_count = shared.engine.model_spec.all_specs.len,
                .context_tokens_max = shared.engine.model_spec.bonsai_8b.context_length,
                .address_record_bytes = shared.engine.model_spec.LayerAddresses.encoded_bytes,
            },
        };
    }

    pub fn alloc(self: *@This(), nbytes: u64, alignment: u32) link_mod.LinkError!link_mod.AllocResult {
        _ = alignment;
        const handle = self.next_handle;
        self.next_handle += 1;
        return .{ .range = .{ .handle = handle, .offset = 0, .nbytes = nbytes } };
    }

    pub fn free(_: *@This(), _: shared.wire.TensorRange) link_mod.LinkError!link_mod.OpTiming {
        return .{};
    }

    pub fn upload(_: *@This(), _: shared.wire.TensorRange, _: []const u8) link_mod.LinkError!link_mod.OpTiming {
        return .{};
    }

    pub fn inference(_: *@This(), _: shared.engine.rpc.Payload) link_mod.LinkError!link_mod.InferenceReply {
        return error.RemoteUnsupportedOp;
    }
};

fn envOr(name: [*:0]const u8, fallback: []const u8) []const u8 {
    return if (std.c.getenv(name)) |value| std.mem.span(value) else fallback;
}

fn endpointSpec() protocol_transport.TcpSpec {
    const host = envOr("PENZAI_HOST", "127.0.0.1");
    const port_text = envOr("PENZAI_PORT", "29092");
    const port = std.fmt.parseInt(u16, port_text, 10) catch 29092;
    return .{ .host = host, .port = port };
}

fn metricsLevel() ?shared.engine.metrics.Level {
    return std.meta.stringToEnum(
        shared.engine.metrics.Level,
        envOr("PENZAI_METRICS", "none"),
    );
}

fn bootstrap() ?*backend.Device {
    if (g_device) |device| return device;

    const gpa = std.heap.c_allocator;
    if (std.mem.eql(u8, envOr("PENZAI_EXECUTOR", "tcp"), "mock")) {
        const device = backend.Device.create(gpa, link_mod.Client.init(&g_mock_link)) catch
            return null;
        device.setMetricsLevel(metricsLevel() orelse {
            device.destroy();
            return null;
        }) catch {
            device.destroy();
            return null;
        };
        device.enableMockExecutor();
        g_device = device;
        return device;
    }

    g_io = std.Io.Threaded.init(gpa, .{});
    g_link = link_mod.TcpLink.connect(gpa, g_io.io(), endpointSpec()) catch {
        g_io.deinit();
        return null;
    };
    const client = link_mod.Client.init(&g_link);
    const device = backend.Device.create(gpa, client) catch {
        g_link.deinit();
        g_io.deinit();
        return null;
    };
    device.setMetricsLevel(metricsLevel() orelse {
        device.destroy();
        g_link.deinit();
        g_io.deinit();
        return null;
    }) catch {
        device.destroy();
        g_link.deinit();
        g_io.deinit();
        return null;
    };
    g_device = device;
    return device;
}

/// The symbol llama.cpp's dynamic backend loader resolves. Returns our registry
/// (one device, "penzai"); null on bootstrap failure (e.g. the daemon is down).
export fn ggml_backend_init() callconv(.c) c.ggml_backend_reg_t {
    const device = bootstrap() orelse return null;
    return device.ggmlReg();
}

/// Load priority for the directory-scan path; non-zero so the loader keeps us.
export fn ggml_backend_score() callconv(.c) c_int {
    return 100;
}
