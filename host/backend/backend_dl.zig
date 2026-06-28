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
//! Residency note: stock llama-cli must pass `-ngl 999 --no-op-offload -fa on`; the
//! in-process driver hard-codes those (plan-host-rebuild.md §2.6/§2.7), the .so can't.
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
var g_device: ?*backend.Device = null;

fn envOr(name: [*:0]const u8, fallback: []const u8) []const u8 {
    return if (std.c.getenv(name)) |value| std.mem.span(value) else fallback;
}

fn endpointSpec() protocol_transport.TcpSpec {
    const host = envOr("PENZAI_HOST", "127.0.0.1");
    const port_text = envOr("PENZAI_PORT", "9000");
    const port = std.fmt.parseInt(u16, port_text, 10) catch 9000;
    return .{ .host = host, .port = port };
}

fn bootstrap() ?*backend.Device {
    if (g_device) |device| return device;

    const gpa = std.heap.c_allocator;
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
