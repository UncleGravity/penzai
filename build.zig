const std = @import("std");

const ModuleSet = struct {
    build_options: *std.Build.Module,
    c: ?*std.Build.Module,
    q1a8: *std.Build.Module,
    framing: *std.Build.Module,
    protocol_transport: *std.Build.Module,
    wire: *std.Build.Module,
    heap: *std.Build.Module,
    xrt: *std.Build.Module,
    xrt_bo: *std.Build.Module,
    ps_activations: *std.Build.Module,
    ps_elemwise: *std.Build.Module,
    ps_matmul_q1a8: *std.Build.Module,
    ps_rmsnorm: *std.Build.Module,
    ps_rows: *std.Build.Module,
    ps_rope: *std.Build.Module,
    ps_softmax: *std.Build.Module,
    runtime: *std.Build.Module,
    server: *std.Build.Module,
    host_tcp: *std.Build.Module,
    device_tcp: *std.Build.Module,
    link: *std.Build.Module,
    llama: ?*std.Build.Module,
    lower: ?*std.Build.Module,
    census: ?*std.Build.Module,
    backend: ?*std.Build.Module,
    run: *std.Build.Module,
};

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const llama_src = b.option(
        []const u8,
        "llama-src",
        "Path to the pinned llama.cpp source tree",
    ) orelse "";
    const llama_lib = b.option(
        []const u8,
        "llama-lib",
        "Path to the pinned llama.cpp install/library derivation",
    ) orelse "";
    const model_path = b.option(
        []const u8,
        "model",
        "Default GGUF model path for zig build run",
    ) orelse "";
    if ((llama_src.len == 0) != (llama_lib.len == 0)) {
        @panic("pass both -Dllama-src=/path/to/llama.cpp and -Dllama-lib=/path/to/llama-install, or neither");
    }
    const enable_llama = llama_src.len != 0;

    const options = b.addOptions();
    options.addOption(bool, "enable_llama", enable_llama);
    options.addOption([]const u8, "default_model_path", model_path);
    const options_mod = options.createModule();

    const c_mod = if (enable_llama) createLlamaCModule(b, target, llama_src) else null;

    const modules = createModules(b, target, optimize, options_mod, c_mod, llama_src);

    const test_step = b.step("test", "Run host-only unit and fake full-stack tests");
    addTest(b, test_step, "shared/protocol/framing.zig", target, optimize, modules);
    addTest(b, test_step, "shared/protocol/transport.zig", target, optimize, modules);
    addTest(b, test_step, "shared/protocol/wire.zig", target, optimize, modules);
    addTest(b, test_step, "shared/q1a8.zig", target, optimize, modules);
    addTest(b, test_step, "device/mem/heap.zig", target, optimize, modules);
    addTest(b, test_step, "device/runtime.zig", target, optimize, modules);
    addTest(b, test_step, "device/ps/activations.zig", target, optimize, modules);
    addTest(b, test_step, "device/ps/elemwise.zig", target, optimize, modules);
    addTest(b, test_step, "device/ps/matmul_q1a8.zig", target, optimize, modules);
    addTest(b, test_step, "device/ps/rmsnorm.zig", target, optimize, modules);
    addTest(b, test_step, "device/ps/rows.zig", target, optimize, modules);
    addTest(b, test_step, "device/ps/rope.zig", target, optimize, modules);
    addTest(b, test_step, "device/ps/softmax.zig", target, optimize, modules);
    addTest(b, test_step, "host/run.zig", target, optimize, modules);
    addTest(b, test_step, "test/fullstack_fake.zig", target, optimize, modules);

    const host_mod = b.createModule(.{
        .root_source_file = b.path("host/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    attachCommon(host_mod, modules);
    const host_exe = b.addExecutable(.{
        .name = "penzai",
        .root_module = host_mod,
    });
    if (enable_llama) linkLlama(b, host_mod, llama_lib);
    b.installArtifact(host_exe);

    const kr260_target = b.resolveTargetQuery(.{
        .cpu_arch = .aarch64,
        .os_tag = .linux,
        .abi = .gnu,
        .cpu_model = .{ .explicit = &std.Target.aarch64.cpu.cortex_a53 },
    });
    const kr260_modules = createModules(b, kr260_target, optimize, options_mod, null, "");
    const device_mod = b.createModule(.{
        .root_source_file = b.path("device/main.zig"),
        .target = kr260_target,
        .optimize = optimize,
        .link_libc = true,
    });
    attachCommon(device_mod, kr260_modules);
    const device_exe = b.addExecutable(.{
        .name = "penzaid",
        .root_module = device_mod,
    });
    b.installArtifact(device_exe);
    b.step("device", "Cross-compile the KR260 device daemon skeleton").dependOn(&device_exe.step);

    const native_device_mod = b.createModule(.{
        .root_source_file = b.path("device/main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    attachCommon(native_device_mod, modules);
    const native_device_exe = b.addExecutable(.{
        .name = "penzaid-native",
        .root_module = native_device_mod,
    });
    const install_native_device = b.addInstallArtifact(native_device_exe, .{});
    b.step("device-native", "Build a native penzaid for local TCP smoke tests").dependOn(&install_native_device.step);
}

fn createModules(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    build_options: *std.Build.Module,
    c_mod: ?*std.Build.Module,
    llama_src: []const u8,
) ModuleSet {
    const q1a8 = b.createModule(.{ .root_source_file = b.path("shared/q1a8.zig"), .target = target, .optimize = optimize });
    const framing = b.createModule(.{ .root_source_file = b.path("shared/protocol/framing.zig"), .target = target, .optimize = optimize });
    const protocol_transport = b.createModule(.{ .root_source_file = b.path("shared/protocol/transport.zig"), .target = target, .optimize = optimize });
    const wire = b.createModule(.{ .root_source_file = b.path("shared/protocol/wire.zig"), .target = target, .optimize = optimize });
    const heap = b.createModule(.{ .root_source_file = b.path("device/mem/heap.zig"), .target = target, .optimize = optimize });
    const xrt = b.createModule(.{ .root_source_file = b.path("device/xrt.zig"), .target = target, .optimize = optimize, .link_libc = true });
    const xrt_bo = b.createModule(.{ .root_source_file = b.path("device/mem/xrt_bo.zig"), .target = target, .optimize = optimize, .link_libc = true });
    const ps_activations = b.createModule(.{ .root_source_file = b.path("device/ps/activations.zig"), .target = target, .optimize = optimize });
    const ps_elemwise = b.createModule(.{ .root_source_file = b.path("device/ps/elemwise.zig"), .target = target, .optimize = optimize });
    const ps_matmul_q1a8 = b.createModule(.{ .root_source_file = b.path("device/ps/matmul_q1a8.zig"), .target = target, .optimize = optimize });
    const ps_rmsnorm = b.createModule(.{ .root_source_file = b.path("device/ps/rmsnorm.zig"), .target = target, .optimize = optimize });
    const ps_rows = b.createModule(.{ .root_source_file = b.path("device/ps/rows.zig"), .target = target, .optimize = optimize });
    const ps_rope = b.createModule(.{ .root_source_file = b.path("device/ps/rope.zig"), .target = target, .optimize = optimize });
    const ps_softmax = b.createModule(.{ .root_source_file = b.path("device/ps/softmax.zig"), .target = target, .optimize = optimize });
    const runtime = b.createModule(.{ .root_source_file = b.path("device/runtime.zig"), .target = target, .optimize = optimize });
    const server = b.createModule(.{ .root_source_file = b.path("device/server.zig"), .target = target, .optimize = optimize });
    const host_tcp = b.createModule(.{ .root_source_file = b.path("host/transport/tcp.zig"), .target = target, .optimize = optimize });
    const device_tcp = b.createModule(.{ .root_source_file = b.path("device/transport/tcp.zig"), .target = target, .optimize = optimize });
    const link = b.createModule(.{ .root_source_file = b.path("host/link.zig"), .target = target, .optimize = optimize });
    const llama = if (c_mod != null) b.createModule(.{ .root_source_file = b.path("host/llama.zig"), .target = target, .optimize = optimize, .link_libc = true, .link_libcpp = true }) else null;
    const lower = if (c_mod != null) b.createModule(.{ .root_source_file = b.path("host/lower.zig"), .target = target, .optimize = optimize, .link_libc = true, .link_libcpp = true }) else null;
    const census = if (c_mod != null) b.createModule(.{ .root_source_file = b.path("host/census.zig"), .target = target, .optimize = optimize, .link_libc = true, .link_libcpp = true }) else null;
    const backend = if (c_mod != null) b.createModule(.{ .root_source_file = b.path("host/backend.zig"), .target = target, .optimize = optimize, .link_libc = true, .link_libcpp = true }) else null;
    const run = b.createModule(.{ .root_source_file = b.path("host/run.zig"), .target = target, .optimize = optimize });

    protocol_transport.addImport("framing", framing);
    heap.addImport("wire", wire);
    xrt_bo.addImport("wire", wire);
    xrt_bo.addImport("xrt", xrt);
    ps_matmul_q1a8.addImport("q1a8", q1a8);
    ps_rows.addImport("q1a8", q1a8);
    ps_rows.addImport("wire", wire);
    runtime.addImport("wire", wire);
    runtime.addImport("heap", heap);
    runtime.addImport("ps_activations", ps_activations);
    runtime.addImport("ps_elemwise", ps_elemwise);
    runtime.addImport("ps_matmul_q1a8", ps_matmul_q1a8);
    runtime.addImport("ps_rmsnorm", ps_rmsnorm);
    runtime.addImport("ps_rows", ps_rows);
    runtime.addImport("ps_rope", ps_rope);
    runtime.addImport("ps_softmax", ps_softmax);
    server.addImport("framing", framing);
    server.addImport("wire", wire);
    server.addImport("runtime", runtime);
    host_tcp.addImport("protocol_transport", protocol_transport);
    device_tcp.addImport("protocol_transport", protocol_transport);
    device_tcp.addImport("runtime", runtime);
    device_tcp.addImport("server", server);
    device_tcp.addImport("xrt_bo", xrt_bo);
    link.addImport("framing", framing);
    link.addImport("protocol_transport", protocol_transport);
    link.addImport("wire", wire);
    link.addImport("runtime", runtime);
    link.addImport("server", server);
    link.addImport("host_tcp", host_tcp);
    if (llama) |m| {
        addChatShim(b, m, llama_src);
        m.addImport("build_options", build_options);
        m.addImport("c", c_mod.?);
        m.addImport("backend", backend.?);
        m.addImport("census", census.?);
        m.addImport("link", link);
    }
    if (lower) |m| {
        m.addImport("c", c_mod.?);
        m.addImport("q1a8", q1a8);
        m.addImport("wire", wire);
    }
    if (census) |m| {
        m.addImport("c", c_mod.?);
        m.addImport("lower", lower.?);
    }
    if (backend) |m| {
        m.addImport("c", c_mod.?);
        m.addImport("q1a8", q1a8);
        m.addImport("wire", wire);
        m.addImport("link", link);
        m.addImport("lower", lower.?);
        m.addImport("census", census.?);
    }
    run.addImport("build_options", build_options);
    run.addImport("q1a8", q1a8);
    run.addImport("protocol_transport", protocol_transport);
    run.addImport("wire", wire);
    run.addImport("runtime", runtime);
    run.addImport("link", link);
    if (llama) |m| run.addImport("llama", m);
    if (backend) |m| run.addImport("backend", m);

    return .{
        .build_options = build_options,
        .c = c_mod,
        .q1a8 = q1a8,
        .framing = framing,
        .protocol_transport = protocol_transport,
        .wire = wire,
        .heap = heap,
        .xrt = xrt,
        .xrt_bo = xrt_bo,
        .ps_activations = ps_activations,
        .ps_elemwise = ps_elemwise,
        .ps_matmul_q1a8 = ps_matmul_q1a8,
        .ps_rmsnorm = ps_rmsnorm,
        .ps_rows = ps_rows,
        .ps_rope = ps_rope,
        .ps_softmax = ps_softmax,
        .runtime = runtime,
        .server = server,
        .host_tcp = host_tcp,
        .device_tcp = device_tcp,
        .link = link,
        .llama = llama,
        .lower = lower,
        .census = census,
        .backend = backend,
        .run = run,
    };
}

fn attachCommon(mod: *std.Build.Module, modules: ModuleSet) void {
    mod.addImport("build_options", modules.build_options);
    mod.addImport("q1a8", modules.q1a8);
    mod.addImport("framing", modules.framing);
    mod.addImport("protocol_transport", modules.protocol_transport);
    mod.addImport("wire", modules.wire);
    mod.addImport("heap", modules.heap);
    mod.addImport("xrt", modules.xrt);
    mod.addImport("xrt_bo", modules.xrt_bo);
    mod.addImport("ps_activations", modules.ps_activations);
    mod.addImport("ps_elemwise", modules.ps_elemwise);
    mod.addImport("ps_matmul_q1a8", modules.ps_matmul_q1a8);
    mod.addImport("ps_rmsnorm", modules.ps_rmsnorm);
    mod.addImport("ps_rows", modules.ps_rows);
    mod.addImport("ps_rope", modules.ps_rope);
    mod.addImport("ps_softmax", modules.ps_softmax);
    mod.addImport("runtime", modules.runtime);
    mod.addImport("server", modules.server);
    mod.addImport("host_tcp", modules.host_tcp);
    mod.addImport("device_tcp", modules.device_tcp);
    mod.addImport("link", modules.link);
    if (modules.llama) |m| mod.addImport("llama", m);
    if (modules.lower) |m| mod.addImport("lower", m);
    if (modules.census) |m| mod.addImport("census", m);
    if (modules.backend) |m| mod.addImport("backend", m);
    mod.addImport("run", modules.run);
}

fn addTest(
    b: *std.Build,
    step: *std.Build.Step,
    path: []const u8,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    modules: ModuleSet,
) void {
    const mod = b.createModule(.{
        .root_source_file = b.path(path),
        .target = target,
        .optimize = optimize,
    });
    attachCommon(mod, modules);
    const test_exe = b.addTest(.{ .root_module = mod });
    step.dependOn(&b.addRunArtifact(test_exe).step);
}

fn createLlamaCModule(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    llama_src: []const u8,
) *std.Build.Module {
    const translate_c = b.addTranslateC(.{
        .root_source_file = b.path("host/c_api.h"),
        .target = target,
        .optimize = .Debug,
        .link_libc = true,
    });
    translate_c.addIncludePath(b.path("host"));
    translate_c.addIncludePath(.{ .cwd_relative = b.pathJoin(&.{ llama_src, "include" }) });
    translate_c.addIncludePath(.{ .cwd_relative = b.pathJoin(&.{ llama_src, "ggml", "include" }) });
    translate_c.addIncludePath(.{ .cwd_relative = b.pathJoin(&.{ llama_src, "ggml", "src" }) });
    return translate_c.createModule();
}

fn addChatShim(b: *std.Build, mod: *std.Build.Module, llama_src: []const u8) void {
    mod.addCSourceFile(.{
        .file = b.path("host/chat.cpp"),
        .flags = &.{"-std=c++17"},
    });
    mod.addIncludePath(b.path("host"));
    mod.addIncludePath(.{ .cwd_relative = llama_src });
    mod.addIncludePath(.{ .cwd_relative = b.pathJoin(&.{ llama_src, "include" }) });
    mod.addIncludePath(.{ .cwd_relative = b.pathJoin(&.{ llama_src, "common" }) });
    // common/chat.h reaches into llama.cpp's vendored nlohmann headers.
    mod.addIncludePath(.{ .cwd_relative = b.pathJoin(&.{ llama_src, "vendor" }) });
    mod.addIncludePath(.{ .cwd_relative = b.pathJoin(&.{ llama_src, "ggml", "include" }) });
    mod.addIncludePath(.{ .cwd_relative = b.pathJoin(&.{ llama_src, "ggml", "src" }) });
}

fn linkLlama(b: *std.Build, mod: *std.Build.Module, llama_lib: []const u8) void {
    const lib_path = b.pathJoin(&.{ llama_lib, "lib" });
    mod.addLibraryPath(.{ .cwd_relative = lib_path });
    mod.addRPath(.{ .cwd_relative = lib_path });
    mod.linkSystemLibrary("llama-common", .{});
    mod.linkSystemLibrary("llama", .{});
    mod.linkSystemLibrary("ggml", .{});
    mod.linkSystemLibrary("ggml-base", .{});
    mod.linkSystemLibrary("ggml-cpu", .{});
}
