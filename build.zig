const std = @import("std");

const ModuleSet = struct {
    q1a8: *std.Build.Module,
    framing: *std.Build.Module,
    protocol_transport: *std.Build.Module,
    wire: *std.Build.Module,
    heap: *std.Build.Module,
    xrt: *std.Build.Module,
    xrt_bo: *std.Build.Module,
    matmul_ref: *std.Build.Module,
    runtime: *std.Build.Module,
    server: *std.Build.Module,
    host_tcp: *std.Build.Module,
    device_tcp: *std.Build.Module,
    link: *std.Build.Module,
    run: *std.Build.Module,
};

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const modules = createModules(b, target, optimize);

    const test_step = b.step("test", "Run host-only unit and fake full-stack tests");
    addTest(b, test_step, "shared/protocol/framing.zig", target, optimize, modules);
    addTest(b, test_step, "shared/protocol/transport.zig", target, optimize, modules);
    addTest(b, test_step, "shared/protocol/wire.zig", target, optimize, modules);
    addTest(b, test_step, "shared/q1a8.zig", target, optimize, modules);
    addTest(b, test_step, "device/mem/heap.zig", target, optimize, modules);
    addTest(b, test_step, "device/pl/matmul_ref.zig", target, optimize, modules);
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
    b.installArtifact(host_exe);

    const kr260_target = b.resolveTargetQuery(.{
        .cpu_arch = .aarch64,
        .os_tag = .linux,
        .abi = .gnu,
        .cpu_model = .{ .explicit = &std.Target.aarch64.cpu.cortex_a53 },
    });
    const kr260_modules = createModules(b, kr260_target, optimize);
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
) ModuleSet {
    const q1a8 = b.createModule(.{ .root_source_file = b.path("shared/q1a8.zig"), .target = target, .optimize = optimize });
    const framing = b.createModule(.{ .root_source_file = b.path("shared/protocol/framing.zig"), .target = target, .optimize = optimize });
    const protocol_transport = b.createModule(.{ .root_source_file = b.path("shared/protocol/transport.zig"), .target = target, .optimize = optimize });
    const wire = b.createModule(.{ .root_source_file = b.path("shared/protocol/wire.zig"), .target = target, .optimize = optimize });
    const heap = b.createModule(.{ .root_source_file = b.path("device/mem/heap.zig"), .target = target, .optimize = optimize });
    const xrt = b.createModule(.{ .root_source_file = b.path("device/xrt.zig"), .target = target, .optimize = optimize, .link_libc = true });
    const xrt_bo = b.createModule(.{ .root_source_file = b.path("device/mem/xrt_bo.zig"), .target = target, .optimize = optimize, .link_libc = true });
    const matmul_ref = b.createModule(.{ .root_source_file = b.path("device/pl/matmul_ref.zig"), .target = target, .optimize = optimize });
    const runtime = b.createModule(.{ .root_source_file = b.path("device/runtime.zig"), .target = target, .optimize = optimize });
    const server = b.createModule(.{ .root_source_file = b.path("device/server.zig"), .target = target, .optimize = optimize });
    const host_tcp = b.createModule(.{ .root_source_file = b.path("host/transport/tcp.zig"), .target = target, .optimize = optimize });
    const device_tcp = b.createModule(.{ .root_source_file = b.path("device/transport/tcp.zig"), .target = target, .optimize = optimize });
    const link = b.createModule(.{ .root_source_file = b.path("host/link.zig"), .target = target, .optimize = optimize });
    const run = b.createModule(.{ .root_source_file = b.path("host/run.zig"), .target = target, .optimize = optimize });

    protocol_transport.addImport("framing", framing);
    heap.addImport("wire", wire);
    xrt_bo.addImport("wire", wire);
    xrt_bo.addImport("xrt", xrt);
    matmul_ref.addImport("q1a8", q1a8);
    runtime.addImport("wire", wire);
    runtime.addImport("heap", heap);
    runtime.addImport("matmul_ref", matmul_ref);
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
    run.addImport("q1a8", q1a8);
    run.addImport("protocol_transport", protocol_transport);
    run.addImport("wire", wire);
    run.addImport("runtime", runtime);
    run.addImport("link", link);

    return .{
        .q1a8 = q1a8,
        .framing = framing,
        .protocol_transport = protocol_transport,
        .wire = wire,
        .heap = heap,
        .xrt = xrt,
        .xrt_bo = xrt_bo,
        .matmul_ref = matmul_ref,
        .runtime = runtime,
        .server = server,
        .host_tcp = host_tcp,
        .device_tcp = device_tcp,
        .link = link,
        .run = run,
    };
}

fn attachCommon(mod: *std.Build.Module, modules: ModuleSet) void {
    mod.addImport("q1a8", modules.q1a8);
    mod.addImport("framing", modules.framing);
    mod.addImport("protocol_transport", modules.protocol_transport);
    mod.addImport("wire", modules.wire);
    mod.addImport("heap", modules.heap);
    mod.addImport("xrt", modules.xrt);
    mod.addImport("xrt_bo", modules.xrt_bo);
    mod.addImport("matmul_ref", modules.matmul_ref);
    mod.addImport("runtime", modules.runtime);
    mod.addImport("server", modules.server);
    mod.addImport("host_tcp", modules.host_tcp);
    mod.addImport("device_tcp", modules.device_tcp);
    mod.addImport("link", modules.link);
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
