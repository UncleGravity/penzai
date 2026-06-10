const std = @import("std");

const LlamaConfig = struct {
    enabled: bool,
    src: []const u8,
    lib: []const u8,
};

const SharedModules = struct {
    q1a8: *std.Build.Module,
    framing: *std.Build.Module,
    protocol_transport: *std.Build.Module,
    wire: *std.Build.Module,
    profiling: *std.Build.Module,
};

const DeviceModules = struct {
    shared: SharedModules,
    build_options: *std.Build.Module,
    profile: *std.Build.Module,
    heap: *std.Build.Module,
    xrt: *std.Build.Module,
    xrt_bo: *std.Build.Module,
    ps_activations: *std.Build.Module,
    ps_elemwise: *std.Build.Module,
    ps_matmul_q1a8: *std.Build.Module,
    ps_flash_attn: *std.Build.Module,
    ps_rmsnorm: *std.Build.Module,
    ps_rows: *std.Build.Module,
    ps_rope: *std.Build.Module,
    ps_softmax: *std.Build.Module,
    runtime: *std.Build.Module,
    server: *std.Build.Module,
    device_tcp: *std.Build.Module,
};

const HostModules = struct {
    shared: SharedModules,
    device_for_fake: DeviceModules,
    build_options: *std.Build.Module,
    c: ?*std.Build.Module,
    prof_report: *std.Build.Module,
    trace: *std.Build.Module,
    host_tcp: *std.Build.Module,
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
        "Default GGUF model path for penzai run",
    ) orelse "";
    if ((llama_src.len == 0) != (llama_lib.len == 0)) {
        @panic("pass both -Dllama-src=/path/to/llama.cpp and -Dllama-lib=/path/to/llama-install, or neither");
    }
    const llama_config = LlamaConfig{
        .enabled = llama_src.len != 0,
        .src = llama_src,
        .lib = llama_lib,
    };

    const enable_profiling = b.option(bool, "profiling", "Compile device-side profiling collection (default true)") orelse true;

    const host_options = createBuildOptions(b, llama_config.enabled, enable_profiling, model_path);
    const host_shared = createSharedModules(b, target, optimize);
    const host_device_for_fake = createDeviceModules(b, target, optimize, host_options, host_shared);
    const host_modules = createHostModules(b, target, optimize, host_options, host_shared, host_device_for_fake, llama_config);

    const host_exe = addPenzai(b, target, optimize, host_modules, llama_config);
    b.installArtifact(host_exe);
    const install_penzai = b.addInstallArtifact(host_exe, .{});
    b.step("install-penzai", "Install the host penzai CLI").dependOn(&install_penzai.step);

    const kr260_target = kr260Target(b);
    const device_options = createBuildOptions(b, false, enable_profiling, "");
    const kr260_shared = createSharedModules(b, kr260_target, optimize);
    const kr260_modules = createDeviceModules(b, kr260_target, optimize, device_options, kr260_shared);
    const device_exe = addPenzaid(b, kr260_target, optimize, kr260_modules, "penzaid");
    b.installArtifact(device_exe);
    b.step("device", "Cross-compile the KR260 device daemon").dependOn(&device_exe.step);
    const install_penzaid = b.addInstallArtifact(device_exe, .{});
    b.step("install-penzaid", "Install the KR260 penzaid daemon").dependOn(&install_penzaid.step);

    const native_device_options = createBuildOptions(b, false, enable_profiling, "");
    const native_device_modules = createDeviceModules(b, target, optimize, native_device_options, createSharedModules(b, target, optimize));
    const native_device_exe = addPenzaid(b, target, optimize, native_device_modules, "penzaid-native");
    const install_native_device = b.addInstallArtifact(native_device_exe, .{});
    b.step("device-native", "Build and install a native penzaid for local TCP smoke tests").dependOn(&install_native_device.step);
    b.step("install-penzaid-native", "Install a native penzaid for local TCP smoke tests").dependOn(&install_native_device.step);

    addTests(b, target, optimize, host_modules, host_device_for_fake);
}

fn createBuildOptions(
    b: *std.Build,
    enable_llama: bool,
    enable_profiling: bool,
    model_path: []const u8,
) *std.Build.Module {
    const options = b.addOptions();
    options.addOption(bool, "enable_llama", enable_llama);
    options.addOption(bool, "enable_profiling", enable_profiling);
    options.addOption([]const u8, "default_model_path", model_path);
    return options.createModule();
}

fn kr260Target(b: *std.Build) std.Build.ResolvedTarget {
    return b.resolveTargetQuery(.{
        .cpu_arch = .aarch64,
        .os_tag = .linux,
        .abi = .gnu,
        .cpu_model = .{ .explicit = &std.Target.aarch64.cpu.cortex_a53 },
    });
}

fn addPenzai(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    modules: HostModules,
    llama_config: LlamaConfig,
) *std.Build.Step.Compile {
    const host_mod = b.createModule(.{
        .root_source_file = b.path("host/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    attachHostImports(host_mod, modules);

    const host_exe = b.addExecutable(.{
        .name = "penzai",
        .root_module = host_mod,
    });
    if (llama_config.enabled) linkLlama(b, host_mod, llama_config.lib);
    return host_exe;
}

fn addPenzaid(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    modules: DeviceModules,
    name: []const u8,
) *std.Build.Step.Compile {
    const device_mod = b.createModule(.{
        .root_source_file = b.path("device/main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    attachDeviceImports(device_mod, modules);

    return b.addExecutable(.{
        .name = name,
        .root_module = device_mod,
    });
}

fn createSharedModules(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
) SharedModules {
    const q1a8 = b.createModule(.{ .root_source_file = b.path("shared/q1a8.zig"), .target = target, .optimize = optimize });
    const framing = b.createModule(.{ .root_source_file = b.path("shared/protocol/framing.zig"), .target = target, .optimize = optimize });
    const protocol_transport = b.createModule(.{ .root_source_file = b.path("shared/protocol/transport.zig"), .target = target, .optimize = optimize });
    const wire = b.createModule(.{ .root_source_file = b.path("shared/protocol/wire.zig"), .target = target, .optimize = optimize });
    const profiling = b.createModule(.{ .root_source_file = b.path("shared/profiling.zig"), .target = target, .optimize = optimize });

    protocol_transport.addImport("framing", framing);

    return .{
        .q1a8 = q1a8,
        .framing = framing,
        .protocol_transport = protocol_transport,
        .wire = wire,
        .profiling = profiling,
    };
}

fn createDeviceModules(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    build_options: *std.Build.Module,
    shared: SharedModules,
) DeviceModules {
    const profile = b.createModule(.{ .root_source_file = b.path("device/profile.zig"), .target = target, .optimize = optimize });
    const heap = b.createModule(.{ .root_source_file = b.path("device/mem/heap.zig"), .target = target, .optimize = optimize });
    const xrt = b.createModule(.{ .root_source_file = b.path("device/xrt.zig"), .target = target, .optimize = optimize, .link_libc = true });
    const xrt_bo = b.createModule(.{ .root_source_file = b.path("device/mem/xrt_bo.zig"), .target = target, .optimize = optimize, .link_libc = true });
    const ps_activations = b.createModule(.{ .root_source_file = b.path("device/ps/activations.zig"), .target = target, .optimize = optimize });
    const ps_elemwise = b.createModule(.{ .root_source_file = b.path("device/ps/elemwise.zig"), .target = target, .optimize = optimize });
    const ps_flash_attn = b.createModule(.{ .root_source_file = b.path("device/ps/flash_attn.zig"), .target = target, .optimize = optimize });
    const ps_matmul_q1a8 = b.createModule(.{ .root_source_file = b.path("device/ps/matmul_q1a8.zig"), .target = target, .optimize = optimize });
    const ps_rmsnorm = b.createModule(.{ .root_source_file = b.path("device/ps/rmsnorm.zig"), .target = target, .optimize = optimize });
    const ps_rows = b.createModule(.{ .root_source_file = b.path("device/ps/rows.zig"), .target = target, .optimize = optimize });
    const ps_rope = b.createModule(.{ .root_source_file = b.path("device/ps/rope.zig"), .target = target, .optimize = optimize });
    const ps_softmax = b.createModule(.{ .root_source_file = b.path("device/ps/softmax.zig"), .target = target, .optimize = optimize });
    const runtime = b.createModule(.{ .root_source_file = b.path("device/runtime.zig"), .target = target, .optimize = optimize });
    const server = b.createModule(.{ .root_source_file = b.path("device/server.zig"), .target = target, .optimize = optimize });
    const device_tcp = b.createModule(.{ .root_source_file = b.path("device/transport/tcp.zig"), .target = target, .optimize = optimize });

    heap.addImport("wire", shared.wire);
    xrt_bo.addImport("wire", shared.wire);
    xrt_bo.addImport("xrt", xrt);
    ps_matmul_q1a8.addImport("q1a8", shared.q1a8);
    ps_rows.addImport("q1a8", shared.q1a8);
    ps_rows.addImport("wire", shared.wire);
    profile.addImport("wire", shared.wire);
    profile.addImport("profiling", shared.profiling);
    profile.addImport("q1a8", shared.q1a8);
    runtime.addImport("build_options", build_options);
    runtime.addImport("wire", shared.wire);
    runtime.addImport("profiling", shared.profiling);
    runtime.addImport("profile", profile);
    runtime.addImport("heap", heap);
    runtime.addImport("ps_activations", ps_activations);
    runtime.addImport("ps_elemwise", ps_elemwise);
    runtime.addImport("ps_flash_attn", ps_flash_attn);
    runtime.addImport("ps_matmul_q1a8", ps_matmul_q1a8);
    runtime.addImport("ps_rmsnorm", ps_rmsnorm);
    runtime.addImport("ps_rows", ps_rows);
    runtime.addImport("ps_rope", ps_rope);
    runtime.addImport("ps_softmax", ps_softmax);
    server.addImport("framing", shared.framing);
    server.addImport("wire", shared.wire);
    server.addImport("runtime", runtime);
    device_tcp.addImport("protocol_transport", shared.protocol_transport);
    device_tcp.addImport("runtime", runtime);
    device_tcp.addImport("server", server);
    device_tcp.addImport("xrt_bo", xrt_bo);

    return .{
        .shared = shared,
        .build_options = build_options,
        .profile = profile,
        .heap = heap,
        .xrt = xrt,
        .xrt_bo = xrt_bo,
        .ps_activations = ps_activations,
        .ps_elemwise = ps_elemwise,
        .ps_matmul_q1a8 = ps_matmul_q1a8,
        .ps_flash_attn = ps_flash_attn,
        .ps_rmsnorm = ps_rmsnorm,
        .ps_rows = ps_rows,
        .ps_rope = ps_rope,
        .ps_softmax = ps_softmax,
        .runtime = runtime,
        .server = server,
        .device_tcp = device_tcp,
    };
}

fn createHostModules(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    build_options: *std.Build.Module,
    shared: SharedModules,
    device_for_fake: DeviceModules,
    llama_config: LlamaConfig,
) HostModules {
    const c_mod = if (llama_config.enabled) createLlamaCModule(b, target, llama_config.src) else null;
    const prof_report = b.createModule(.{ .root_source_file = b.path("host/prof_report.zig"), .target = target, .optimize = optimize });
    const trace = b.createModule(.{ .root_source_file = b.path("host/trace.zig"), .target = target, .optimize = optimize });
    const host_tcp = b.createModule(.{ .root_source_file = b.path("host/transport/tcp.zig"), .target = target, .optimize = optimize });
    const link = b.createModule(.{ .root_source_file = b.path("host/link.zig"), .target = target, .optimize = optimize });
    const llama = if (c_mod != null) b.createModule(.{ .root_source_file = b.path("host/llama.zig"), .target = target, .optimize = optimize, .link_libc = true, .link_libcpp = true }) else null;
    const lower = if (c_mod != null) b.createModule(.{ .root_source_file = b.path("host/lower.zig"), .target = target, .optimize = optimize, .link_libc = true, .link_libcpp = true }) else null;
    const census = if (c_mod != null) b.createModule(.{ .root_source_file = b.path("host/census.zig"), .target = target, .optimize = optimize, .link_libc = true, .link_libcpp = true }) else null;
    const backend = if (c_mod != null) b.createModule(.{ .root_source_file = b.path("host/backend.zig"), .target = target, .optimize = optimize, .link_libc = true, .link_libcpp = true }) else null;
    const run = b.createModule(.{ .root_source_file = b.path("host/run.zig"), .target = target, .optimize = optimize });

    host_tcp.addImport("protocol_transport", shared.protocol_transport);
    link.addImport("framing", shared.framing);
    link.addImport("protocol_transport", shared.protocol_transport);
    link.addImport("wire", shared.wire);
    link.addImport("profiling", shared.profiling);
    link.addImport("runtime", device_for_fake.runtime);
    link.addImport("server", device_for_fake.server);
    link.addImport("host_tcp", host_tcp);
    prof_report.addImport("wire", shared.wire);
    prof_report.addImport("profiling", shared.profiling);
    prof_report.addImport("link", link);
    trace.addImport("wire", shared.wire);
    trace.addImport("profiling", shared.profiling);
    trace.addImport("prof_report", prof_report);
    if (llama) |m| {
        addChatShim(b, m, llama_config.src);
        m.addImport("build_options", build_options);
        m.addImport("c", c_mod.?);
        m.addImport("backend", backend.?);
        m.addImport("census", census.?);
        m.addImport("link", link);
        m.addImport("trace", trace);
    }
    if (lower) |m| {
        m.addImport("c", c_mod.?);
        m.addImport("q1a8", shared.q1a8);
        m.addImport("wire", shared.wire);
    }
    if (census) |m| {
        m.addImport("c", c_mod.?);
        m.addImport("lower", lower.?);
    }
    if (backend) |m| {
        m.addImport("c", c_mod.?);
        m.addImport("q1a8", shared.q1a8);
        m.addImport("wire", shared.wire);
        m.addImport("profiling", shared.profiling);
        m.addImport("prof_report", prof_report);
        m.addImport("trace", trace);
        m.addImport("link", link);
        m.addImport("lower", lower.?);
        m.addImport("census", census.?);
    }
    run.addImport("build_options", build_options);
    run.addImport("q1a8", shared.q1a8);
    run.addImport("protocol_transport", shared.protocol_transport);
    run.addImport("profiling", shared.profiling);
    run.addImport("prof_report", prof_report);
    run.addImport("trace", trace);
    run.addImport("wire", shared.wire);
    run.addImport("runtime", device_for_fake.runtime);
    run.addImport("link", link);
    if (llama) |m| run.addImport("llama", m);
    if (backend) |m| run.addImport("backend", m);

    return .{
        .shared = shared,
        .device_for_fake = device_for_fake,
        .build_options = build_options,
        .c = c_mod,
        .prof_report = prof_report,
        .trace = trace,
        .host_tcp = host_tcp,
        .link = link,
        .llama = llama,
        .lower = lower,
        .census = census,
        .backend = backend,
        .run = run,
    };
}

fn attachSharedImports(mod: *std.Build.Module, modules: SharedModules) void {
    mod.addImport("q1a8", modules.q1a8);
    mod.addImport("framing", modules.framing);
    mod.addImport("protocol_transport", modules.protocol_transport);
    mod.addImport("wire", modules.wire);
    mod.addImport("profiling", modules.profiling);
}

fn attachDeviceImports(mod: *std.Build.Module, modules: DeviceModules) void {
    attachSharedImports(mod, modules.shared);
    mod.addImport("build_options", modules.build_options);
    mod.addImport("profile", modules.profile);
    mod.addImport("heap", modules.heap);
    mod.addImport("xrt", modules.xrt);
    mod.addImport("xrt_bo", modules.xrt_bo);
    mod.addImport("ps_activations", modules.ps_activations);
    mod.addImport("ps_elemwise", modules.ps_elemwise);
    mod.addImport("ps_flash_attn", modules.ps_flash_attn);
    mod.addImport("ps_matmul_q1a8", modules.ps_matmul_q1a8);
    mod.addImport("ps_rmsnorm", modules.ps_rmsnorm);
    mod.addImport("ps_rows", modules.ps_rows);
    mod.addImport("ps_rope", modules.ps_rope);
    mod.addImport("ps_softmax", modules.ps_softmax);
    mod.addImport("runtime", modules.runtime);
    mod.addImport("server", modules.server);
    mod.addImport("device_tcp", modules.device_tcp);
}

fn attachHostImports(mod: *std.Build.Module, modules: HostModules) void {
    attachSharedImports(mod, modules.shared);
    mod.addImport("build_options", modules.build_options);
    mod.addImport("profile", modules.device_for_fake.profile);
    mod.addImport("heap", modules.device_for_fake.heap);
    mod.addImport("runtime", modules.device_for_fake.runtime);
    mod.addImport("server", modules.device_for_fake.server);
    mod.addImport("prof_report", modules.prof_report);
    mod.addImport("trace", modules.trace);
    mod.addImport("host_tcp", modules.host_tcp);
    mod.addImport("link", modules.link);
    mod.addImport("run", modules.run);
    if (modules.llama) |m| mod.addImport("llama", m);
    if (modules.lower) |m| mod.addImport("lower", m);
    if (modules.census) |m| mod.addImport("census", m);
    if (modules.backend) |m| mod.addImport("backend", m);
}

fn addTests(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    host_modules: HostModules,
    device_modules: DeviceModules,
) void {
    const test_step = b.step("test", "Run host-only unit and fake full-stack tests");

    addSharedTest(b, test_step, "shared/protocol/framing.zig", target, optimize, host_modules.shared);
    addSharedTest(b, test_step, "shared/protocol/transport.zig", target, optimize, host_modules.shared);
    addSharedTest(b, test_step, "shared/protocol/wire.zig", target, optimize, host_modules.shared);
    addSharedTest(b, test_step, "shared/profiling.zig", target, optimize, host_modules.shared);
    addSharedTest(b, test_step, "shared/q1a8.zig", target, optimize, host_modules.shared);

    addDeviceTest(b, test_step, "device/profile.zig", target, optimize, device_modules);
    addDeviceTest(b, test_step, "device/mem/heap.zig", target, optimize, device_modules);
    addDeviceTest(b, test_step, "device/runtime.zig", target, optimize, device_modules);
    addDeviceTest(b, test_step, "device/ps/activations.zig", target, optimize, device_modules);
    addDeviceTest(b, test_step, "device/ps/elemwise.zig", target, optimize, device_modules);
    addDeviceTest(b, test_step, "device/ps/flash_attn.zig", target, optimize, device_modules);
    addDeviceTest(b, test_step, "device/ps/matmul_q1a8.zig", target, optimize, device_modules);
    addDeviceTest(b, test_step, "device/ps/rmsnorm.zig", target, optimize, device_modules);
    addDeviceTest(b, test_step, "device/ps/rows.zig", target, optimize, device_modules);
    addDeviceTest(b, test_step, "device/ps/rope.zig", target, optimize, device_modules);
    addDeviceTest(b, test_step, "device/ps/softmax.zig", target, optimize, device_modules);

    addHostTest(b, test_step, "host/trace.zig", target, optimize, host_modules);
    addHostTest(b, test_step, "host/run.zig", target, optimize, host_modules);
    addHostTest(b, test_step, "test/fullstack_fake.zig", target, optimize, host_modules);
}

fn addSharedTest(
    b: *std.Build,
    step: *std.Build.Step,
    path: []const u8,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    modules: SharedModules,
) void {
    const mod = b.createModule(.{
        .root_source_file = b.path(path),
        .target = target,
        .optimize = optimize,
    });
    attachSharedImports(mod, modules);
    const test_exe = b.addTest(.{ .root_module = mod });
    step.dependOn(&b.addRunArtifact(test_exe).step);
}

fn addDeviceTest(
    b: *std.Build,
    step: *std.Build.Step,
    path: []const u8,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    modules: DeviceModules,
) void {
    const mod = b.createModule(.{
        .root_source_file = b.path(path),
        .target = target,
        .optimize = optimize,
    });
    attachDeviceImports(mod, modules);
    const test_exe = b.addTest(.{ .root_module = mod });
    step.dependOn(&b.addRunArtifact(test_exe).step);
}

fn addHostTest(
    b: *std.Build,
    step: *std.Build.Step,
    path: []const u8,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    modules: HostModules,
) void {
    const mod = b.createModule(.{
        .root_source_file = b.path(path),
        .target = target,
        .optimize = optimize,
    });
    attachHostImports(mod, modules);
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
