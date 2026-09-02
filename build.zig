const std = @import("std");
const fpga_verify = @import("fpga/verify/suites.zig");

const LlamaConfig = struct {
    enabled: bool,
    src: []const u8,
    lib: []const u8,
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

    const host_options = createBuildOptions(b, llama_config.enabled, model_path);
    const host_shared = createSharedModule(b, target, optimize);
    const host_c = if (llama_config.enabled) createLlamaCModule(b, target, llama_config.src) else null;
    const host_engine_regmap = createEngineRegmapModule(b, target, optimize);
    const host_engine = createEngineModule(b, target, optimize, host_shared, host_engine_regmap);
    const host_memory = createMemoryModule(b, target, optimize, host_shared);
    const host_runtime = createRuntimeModule(b, target, optimize, host_shared, host_engine, host_memory, host_engine_regmap);
    const host_server = createServerModule(b, target, optimize, host_shared, host_runtime);
    const host_link = createHostLinkModule(b, target, optimize, host_shared);

    const host_exe = addPenzai(b, target, optimize, host_options, host_shared, host_link, llama_config, host_c);
    b.installArtifact(host_exe);
    const install_penzai = b.addInstallArtifact(host_exe, .{});
    b.step("install-penzai", "Install the host penzai CLI").dependOn(&install_penzai.step);
    addPenzaiBackendLib(b, target, optimize, host_shared, host_link, llama_config);

    const kr260_target = kr260Target(b);
    const device_options = createBuildOptions(b, false, "");
    const kr260_shared = createSharedModule(b, kr260_target, optimize);
    const kr260_engine_regmap = createEngineRegmapModule(b, kr260_target, optimize);
    const kr260_engine = createEngineModule(b, kr260_target, optimize, kr260_shared, kr260_engine_regmap);
    const kr260_memory = createMemoryModule(b, kr260_target, optimize, kr260_shared);
    const kr260_runtime = createRuntimeModule(b, kr260_target, optimize, kr260_shared, kr260_engine, kr260_memory, kr260_engine_regmap);
    const kr260_server = createServerModule(b, kr260_target, optimize, kr260_shared, kr260_runtime);
    const device_exe = addPenzaid(b, kr260_target, optimize, device_options, kr260_shared, kr260_runtime, kr260_server, "penzaid");
    b.step("device", "Cross-compile the KR260 device daemon").dependOn(&device_exe.step);
    const install_penzaid = b.addInstallArtifact(device_exe, .{});
    b.step("install-penzaid", "Install the KR260 penzaid daemon").dependOn(&install_penzaid.step);

    const native_device_options = createBuildOptions(b, false, "");
    const native_runtime = createRuntimeModule(b, target, optimize, host_shared, host_engine, host_memory, host_engine_regmap);
    const native_server = createServerModule(b, target, optimize, host_shared, native_runtime);
    const native_device_exe = addPenzaid(b, target, optimize, native_device_options, host_shared, native_runtime, native_server, "penzaid-native");
    const install_native_device = b.addInstallArtifact(native_device_exe, .{});
    b.step("device-native", "Build a native penzaid for local TCP smoke tests").dependOn(&native_device_exe.step);
    b.step("install-penzaid-native", "Install a native penzaid for local TCP smoke tests").dependOn(&install_native_device.step);

    const all = b.step("all", "Build host, KR260 device, and native device binaries");
    all.dependOn(&host_exe.step);
    all.dependOn(&device_exe.step);
    all.dependOn(&native_device_exe.step);

    addTests(b, target, optimize, host_options, host_shared, host_engine, host_memory, host_runtime, host_server, host_link, llama_config, host_c);
    addRegmapSteps(b);
    addVerificationSteps(b);
}

fn addVerificationSteps(b: *std.Build) void {
    validateVerificationRegistry(b);

    const generated_tables = addGeneratedTableVerification(b);
    const lint = b.step("lint-rtl", "Lint the closed production RTL source set");
    const simulation = b.step("test-rtl", "Run all current-engine RTL simulations");
    const synthesis = b.step("synth-rtl", "Map the production RTL and enforce resource invariants");
    const synthesis_all = b.step("synth-rtl-all", "Run production and extended Yosys checks");
    const formal = b.step("formal", "Run the fast production RTL proofs");
    const formal_all = b.step("formal-all", "Run all production RTL proofs and covers");
    const verify = b.step("verify-rtl", "Run production RTL lint, simulation, synthesis, and fast formal checks");

    for (fpga_verify.suites) |suite| {
        const run = addVerificationCommand(b, suite);
        const focused = b.step(b.fmt("verify-{s}", .{suite.name}), suite.description);
        focused.dependOn(&run.step);

        switch (suite.kind) {
            .lint => if (suite.tier == .default) lint.dependOn(focused),
            .simulation => if (suite.tier == .default) simulation.dependOn(focused),
            .synthesis => {
                synthesis_all.dependOn(focused);
                if (suite.tier == .default) synthesis.dependOn(focused);
            },
            .formal => {
                formal_all.dependOn(focused);
                if (suite.tier == .default) formal.dependOn(focused);
            },
        }
    }

    verify.dependOn(lint);
    verify.dependOn(simulation);
    verify.dependOn(synthesis);
    verify.dependOn(formal);
    verify.dependOn(generated_tables);
}

fn addGeneratedTableVerification(b: *std.Build) *std.Build.Step {
    const flash_generator = b.addExecutable(.{
        .name = "flash-luts",
        .root_module = b.createModule(.{
            .root_source_file = b.path("fpga/build/generators/flash_luts.zig"),
            .target = b.graph.host,
            .optimize = .Debug,
        }),
    });
    const swiglu_generator = b.addExecutable(.{
        .name = "swiglu-lut",
        .root_module = b.createModule(.{
            .root_source_file = b.path("fpga/build/generators/swiglu_lut.zig"),
            .target = b.graph.host,
            .optimize = .Debug,
        }),
    });
    const checker = b.addExecutable(.{
        .name = "generated-table-check",
        .root_module = b.createModule(.{
            .root_source_file = b.path("test/generated_tables.zig"),
            .target = b.graph.host,
            .optimize = .Debug,
        }),
    });

    const generate_flash = b.addRunArtifact(flash_generator);
    const generate_swiglu = b.addRunArtifact(swiglu_generator);
    const check = b.addRunArtifact(checker);
    check.addFileArg(generate_flash.captureStdOut(.{}));
    check.addFileArg(b.path("fpga/rtl/attention/flash_luts.vh"));
    check.addFileArg(generate_swiglu.captureStdOut(.{}));
    check.addFileArg(b.path("fpga/rtl/vector/swiglu.v"));

    const step = b.step("verify-generated-tables", "Regenerate and compare the committed flash and SwiGLU LUTs");
    step.dependOn(&check.step);
    return step;
}

fn addVerificationCommand(b: *std.Build, suite: fpga_verify.Suite) *std.Build.Step.Run {
    return switch (suite.runner) {
        .shell => |shell| blk: {
            const run = b.addSystemCommand(&.{ "bash", shell.path });
            run.addArgs(shell.args);
            break :blk run;
        },
        .sby => |sby| blk: {
            const run = b.addSystemCommand(&.{ "sby", "-f" });
            if (sby.sequential) run.addArg("--sequential");
            run.addArgs(&.{
                "--prefix",
                b.fmt(".zig-cache/fpga-verify/formal/{s}", .{suite.name}),
                sby.path,
            });
            run.addArgs(sby.tasks);
            break :blk run;
        },
    };
}

fn validateVerificationRegistry(b: *std.Build) void {
    const manifest_path = "fpga/build/sources.f";
    const manifest = std.Io.Dir.cwd().readFileAlloc(
        b.graph.io,
        manifest_path,
        b.allocator,
        .limited(1024 * 1024),
    ) catch |err| {
        @panic(b.fmt("cannot read {s}: {s}", .{ manifest_path, @errorName(err) }));
    };

    for (fpga_verify.suites, 0..) |suite, index| {
        for (fpga_verify.suites[0..index]) |prior| {
            if (std.mem.eql(u8, suite.name, prior.name))
                @panic(b.fmt("duplicate FPGA verification suite name: {s}", .{suite.name}));
        }

        const runner_path = switch (suite.runner) {
            .shell => |shell| shell.path,
            .sby => |sby| sby.path,
        };
        std.Io.Dir.cwd().access(b.graph.io, runner_path, .{}) catch |err| {
            @panic(b.fmt("verification suite {s} has missing runner {s}: {s}", .{
                suite.name,
                runner_path,
                @errorName(err),
            }));
        };

        if (suite.source_policy == .production_subset and suite.rtl_sources.len == 0)
            @panic(b.fmt("verification suite {s} declares an empty production subset", .{suite.name}));

        for (suite.rtl_sources) |path| {
            const prefix = "fpga/rtl/";
            if (!std.mem.startsWith(u8, path, prefix))
                @panic(b.fmt("verification suite {s} has non-RTL production dependency: {s}", .{ suite.name, path }));
            if (!productionManifestContains(manifest, path[prefix.len..]))
                @panic(b.fmt("verification suite {s} references RTL outside {s}: {s}", .{
                    suite.name,
                    manifest_path,
                    path,
                }));
        }
    }
}

fn productionManifestContains(manifest: []const u8, wanted: []const u8) bool {
    var lines = std.mem.splitScalar(u8, manifest, '\n');
    while (lines.next()) |raw_line| {
        const comment = std.mem.indexOfScalar(u8, raw_line, '#') orelse raw_line.len;
        const entry = std.mem.trim(u8, raw_line[0..comment], " \t\r");
        if (std.mem.eql(u8, entry, wanted)) return true;
    }
    return false;
}

fn addRegmapSteps(b: *std.Build) void {
    // Generate the software and RTL views of the one production engine map.
    const emit = b.addExecutable(.{
        .name = "regmap-emit",
        .root_module = b.createModule(.{
            .root_source_file = b.path("fpga/regmap/emit.zig"),
            .target = b.graph.host,
            .optimize = .Debug,
        }),
    });
    const update = b.addUpdateSourceFiles();

    const run_engine_vh = b.addRunArtifact(emit);
    run_engine_vh.addArgs(&.{ "engine", "vh" });
    update.addCopyFileToSource(run_engine_vh.captureStdOut(.{}), "fpga/regmap/engine_regs.vh");

    const run_engine_tcl = b.addRunArtifact(emit);
    run_engine_tcl.addArgs(&.{ "engine", "tcl" });
    update.addCopyFileToSource(run_engine_tcl.captureStdOut(.{}), "fpga/regmap/engine_address_map.tcl");

    b.step("regmap", "Generate accelerator register headers and address maps").dependOn(&update.step);
}

fn createBuildOptions(
    b: *std.Build,
    enable_llama: bool,
    model_path: []const u8,
) *std.Build.Module {
    const options = b.addOptions();
    options.addOption(bool, "enable_llama", enable_llama);
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
    build_options: *std.Build.Module,
    shared: *std.Build.Module,
    link: *std.Build.Module,
    llama_config: LlamaConfig,
    c_mod: ?*std.Build.Module,
) *std.Build.Step.Compile {
    const host_mod = b.createModule(.{
        .root_source_file = b.path("host/main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = llama_config.enabled,
        .link_libcpp = llama_config.enabled,
    });
    addHostImports(host_mod, build_options, shared, link);
    if (llama_config.enabled) addLlamaSupport(b, host_mod, llama_config, c_mod.?);
    addHostExtraModules(b, host_mod, target, optimize, shared, link, llama_config);

    return b.addExecutable(.{
        .name = "penzai",
        .root_module = host_mod,
    });
}

fn addPenzaid(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    build_options: *std.Build.Module,
    shared: *std.Build.Module,
    runtime: *std.Build.Module,
    server: *std.Build.Module,
    name: []const u8,
) *std.Build.Step.Compile {
    const device_mod = b.createModule(.{
        .root_source_file = b.path("device/main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    addDeviceImports(device_mod, build_options, shared, runtime, server);

    return b.addExecutable(.{
        .name = name,
        .root_module = device_mod,
    });
}

fn createSharedModule(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
) *std.Build.Module {
    return b.createModule(.{
        .root_source_file = b.path("shared/root.zig"),
        .target = target,
        .optimize = optimize,
    });
}

fn createEngineRegmapModule(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
) *std.Build.Module {
    return b.createModule(.{
        .root_source_file = b.path("fpga/regmap/engine.zig"),
        .target = target,
        .optimize = optimize,
    });
}

fn createRuntimeModule(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    shared: *std.Build.Module,
    engine: *std.Build.Module,
    memory: *std.Build.Module,
    engine_regmap: *std.Build.Module,
) *std.Build.Module {
    const runtime = b.createModule(.{
        .root_source_file = b.path("device/daemon/runtime.zig"),
        .target = target,
        .optimize = optimize,
    });
    runtime.addImport("shared", shared);
    runtime.addImport("engine", engine);
    runtime.addImport("memory", memory);
    runtime.addImport("engine_regmap", engine_regmap);
    return runtime;
}

fn createEngineModule(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    shared: *std.Build.Module,
    engine_regmap: *std.Build.Module,
) *std.Build.Module {
    const engine = b.createModule(.{
        .root_source_file = b.path("device/engine/root.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    engine.addImport("shared", shared);
    engine.addImport("engine_regmap", engine_regmap);
    return engine;
}

fn createMemoryModule(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    shared: *std.Build.Module,
) *std.Build.Module {
    const memory = b.createModule(.{
        .root_source_file = b.path("device/mem/root.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    memory.addImport("shared", shared);
    return memory;
}

fn createServerModule(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    shared: *std.Build.Module,
    runtime: *std.Build.Module,
) *std.Build.Module {
    const server = b.createModule(.{
        .root_source_file = b.path("device/daemon/server.zig"),
        .target = target,
        .optimize = optimize,
    });
    server.addImport("shared", shared);
    server.addImport("runtime", runtime);
    return server;
}

fn createHostLinkModule(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    shared: *std.Build.Module,
) *std.Build.Module {
    const link = b.createModule(.{
        .root_source_file = b.path("host/link/client.zig"),
        .target = target,
        .optimize = optimize,
    });
    link.addImport("shared", shared);
    return link;
}

fn addCommonImports(mod: *std.Build.Module, build_options: *std.Build.Module, shared: *std.Build.Module) void {
    mod.addImport("build_options", build_options);
    mod.addImport("shared", shared);
}

fn addDeviceImports(
    mod: *std.Build.Module,
    build_options: *std.Build.Module,
    shared: *std.Build.Module,
    runtime: *std.Build.Module,
    server: *std.Build.Module,
) void {
    addCommonImports(mod, build_options, shared);
    mod.addImport("runtime", runtime);
    mod.addImport("server", server);
}

fn addHostImports(
    mod: *std.Build.Module,
    build_options: *std.Build.Module,
    shared: *std.Build.Module,
    link: *std.Build.Module,
) void {
    addCommonImports(mod, build_options, shared);
    mod.addImport("link", link);
}

fn addTests(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    build_options: *std.Build.Module,
    shared: *std.Build.Module,
    engine: *std.Build.Module,
    memory: *std.Build.Module,
    runtime: *std.Build.Module,
    server: *std.Build.Module,
    link: *std.Build.Module,
    llama_config: LlamaConfig,
    c_mod: ?*std.Build.Module,
) void {
    const test_step = b.step("test", "Run host and device unit tests");

    addSharedTest(b, test_step, "shared/root.zig", target, optimize, shared);
    addSharedTest(b, test_step, "shared/capabilities.zig", target, optimize, shared);
    addSharedTest(b, test_step, "shared/protocol/framing.zig", target, optimize, shared);
    addSharedTest(b, test_step, "shared/protocol/transport.zig", target, optimize, shared);
    addSharedTest(b, test_step, "shared/protocol/wire.zig", target, optimize, shared);
    addSharedTest(b, test_step, "shared/timing.zig", target, optimize, shared);
    addSharedTest(b, test_step, "host/cli/args.zig", target, optimize, shared);
    addSharedTest(b, test_step, "fpga/regmap/engine.zig", target, optimize, shared);
    addHostTest(b, test_step, null, "host/main.zig", target, optimize, build_options, shared, link, llama_config, c_mod, false);
    addModuleTest(b, test_step, engine);
    addModuleTest(b, test_step, memory);
    addModuleTest(b, test_step, runtime);
    addModuleTest(b, test_step, server);
    addDeviceRuntimeTest(b, test_step, "device/main.zig", target, optimize, build_options, shared, runtime, server);
    const device_tcp_test_step = b.step("test-device-tcp", "Run native TCP daemon lifecycle tests");
    addDeviceServiceTest(b, test_step, device_tcp_test_step, "device/transport/tcp.zig", target, optimize, shared, runtime, server);

    const llama_test_step = b.step("test-host-llama", "Run only the llama driver and backend unit tests");
    addHostTest(b, test_step, llama_test_step, "host/run.zig", target, optimize, build_options, shared, link, llama_config, c_mod, true);
    addBackendTest(b, test_step, llama_test_step, target, optimize, shared, link, llama_config);
}

fn addDeviceServiceTest(
    b: *std.Build,
    step: *std.Build.Step,
    focused_step: *std.Build.Step,
    path: []const u8,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    shared: *std.Build.Module,
    runtime: *std.Build.Module,
    server: *std.Build.Module,
) void {
    const mod = b.createModule(.{
        .root_source_file = b.path(path),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    mod.addImport("shared", shared);
    mod.addImport("runtime", runtime);
    mod.addImport("server", server);
    const test_exe = b.addTest(.{ .root_module = mod });
    const run = b.addRunArtifact(test_exe);
    step.dependOn(&run.step);
    focused_step.dependOn(&run.step);
}

fn addSharedTest(
    b: *std.Build,
    step: *std.Build.Step,
    path: []const u8,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    shared: *std.Build.Module,
) void {
    const mod = b.createModule(.{
        .root_source_file = b.path(path),
        .target = target,
        .optimize = optimize,
        // Keep test linkage consistent with runtime modules that use OS memory APIs.
        .link_libc = true,
    });
    mod.addImport("shared", shared);
    const test_exe = b.addTest(.{ .root_module = mod });
    step.dependOn(&b.addRunArtifact(test_exe).step);
}

fn addModuleTest(
    b: *std.Build,
    step: *std.Build.Step,
    mod: *std.Build.Module,
) void {
    const test_exe = b.addTest(.{ .root_module = mod });
    step.dependOn(&b.addRunArtifact(test_exe).step);
}

fn addDeviceRuntimeTest(
    b: *std.Build,
    step: *std.Build.Step,
    path: []const u8,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    build_options: *std.Build.Module,
    shared: *std.Build.Module,
    runtime: *std.Build.Module,
    server: *std.Build.Module,
) void {
    const mod = b.createModule(.{
        .root_source_file = b.path(path),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    addDeviceImports(mod, build_options, shared, runtime, server);
    const test_exe = b.addTest(.{ .root_module = mod });
    step.dependOn(&b.addRunArtifact(test_exe).step);
}

fn addHostTest(
    b: *std.Build,
    step: *std.Build.Step,
    focused_step: ?*std.Build.Step,
    path: []const u8,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    build_options: *std.Build.Module,
    shared: *std.Build.Module,
    link: *std.Build.Module,
    llama_config: LlamaConfig,
    c_mod: ?*std.Build.Module,
    needs_llama: bool,
) void {
    const with_llama = needs_llama and llama_config.enabled;
    const mod = b.createModule(.{
        .root_source_file = b.path(path),
        .target = target,
        .optimize = optimize,
        .link_libc = with_llama,
        .link_libcpp = with_llama,
    });
    addHostImports(mod, build_options, shared, link);
    if (with_llama) addLlamaSupport(b, mod, llama_config, c_mod.?);
    addHostExtraModules(b, mod, target, optimize, shared, link, llama_config);
    const test_exe = b.addTest(.{ .root_module = mod });
    const run = b.addRunArtifact(test_exe);
    step.dependOn(&run.step);
    if (focused_step) |focused| focused.dependOn(&run.step);
}

fn createLlamaCModule(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    llama_src: []const u8,
) *std.Build.Module {
    const translate_c = b.addTranslateC(.{
        .root_source_file = b.path("host/llama/c_api.h"),
        .target = target,
        .optimize = .Debug,
        .link_libc = true,
    });
    translate_c.addIncludePath(b.path("host/llama"));
    translate_c.addIncludePath(.{ .cwd_relative = b.pathJoin(&.{ llama_src, "include" }) });
    translate_c.addIncludePath(.{ .cwd_relative = b.pathJoin(&.{ llama_src, "ggml", "include" }) });
    translate_c.addIncludePath(.{ .cwd_relative = b.pathJoin(&.{ llama_src, "ggml", "src" }) });
    return translate_c.createModule();
}

fn addLlamaSupport(
    b: *std.Build,
    mod: *std.Build.Module,
    llama_config: LlamaConfig,
    c_mod: *std.Build.Module,
) void {
    mod.addImport("c", c_mod);
    addChatShim(b, mod, llama_config.src);
    linkLlama(b, mod, llama_config.lib);
}

fn addChatShim(b: *std.Build, mod: *std.Build.Module, llama_src: []const u8) void {
    mod.addCSourceFile(.{
        .file = b.path("host/llama/prompt.cpp"),
        .flags = &.{"-std=c++17"},
    });
    mod.addIncludePath(b.path("host/llama"));
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

/// ggml-only translate-c root (module `c_ggml`) for the llama-free backend core —
/// the same ggml/ggml-backend headers as `c`, minus llama.h.
fn createGgmlCModule(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    llama_src: []const u8,
) *std.Build.Module {
    const translate_c = b.addTranslateC(.{
        .root_source_file = b.path("host/backend/c_ggml.h"),
        .target = target,
        .optimize = .Debug,
        .link_libc = true,
    });
    translate_c.addIncludePath(.{ .cwd_relative = b.pathJoin(&.{ llama_src, "include" }) });
    translate_c.addIncludePath(.{ .cwd_relative = b.pathJoin(&.{ llama_src, "ggml", "include" }) });
    translate_c.addIncludePath(.{ .cwd_relative = b.pathJoin(&.{ llama_src, "ggml", "src" }) });
    return translate_c.createModule();
}

/// The ggml backend core as its own module: its import table carries `c_ggml`, not
/// the full llama `c`, so the llama-free invariant is enforced by the build graph
/// rather than by convention.
fn createBackendModule(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    c_ggml: *std.Build.Module,
    shared: *std.Build.Module,
    link: *std.Build.Module,
) *std.Build.Module {
    const backend = b.createModule(.{
        .root_source_file = b.path("host/backend/registry.zig"),
        .target = target,
        .optimize = optimize,
    });
    const inference_engine = b.createModule(.{
        .root_source_file = b.path("host/engine/root.zig"),
        .target = target,
        .optimize = optimize,
    });
    inference_engine.addImport("shared", shared);
    backend.addImport("c_ggml", c_ggml);
    backend.addImport("shared", shared);
    backend.addImport("link", link);
    backend.addImport("inference_engine", inference_engine);
    return backend;
}

fn addBackendTest(
    b: *std.Build,
    test_step: *std.Build.Step,
    focused_step: *std.Build.Step,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    shared: *std.Build.Module,
    link: *std.Build.Module,
    llama_config: LlamaConfig,
) void {
    if (!llama_config.enabled) return;

    const c_ggml = createGgmlCModule(b, target, llama_config.src);
    const mod = b.createModule(.{
        .root_source_file = b.path("host/backend/registry.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    mod.addImport("c_ggml", c_ggml);
    mod.addImport("shared", shared);
    mod.addImport("link", link);
    const inference_engine = b.createModule(.{
        .root_source_file = b.path("host/engine/root.zig"),
        .target = target,
        .optimize = optimize,
    });
    inference_engine.addImport("shared", shared);
    mod.addImport("inference_engine", inference_engine);
    const lib_path = b.pathJoin(&.{ llama_config.lib, "lib" });
    mod.addLibraryPath(.{ .cwd_relative = lib_path });
    mod.addRPath(.{ .cwd_relative = lib_path });
    mod.linkSystemLibrary("ggml-base", .{});

    const test_exe = b.addTest(.{ .root_module = mod });
    const run = b.addRunArtifact(test_exe);
    test_step.dependOn(&run.step);
    focused_step.dependOn(&run.step);
}

// Add the v1 executor registry to host modules when llama support is enabled.
fn addHostExtraModules(
    b: *std.Build,
    mod: *std.Build.Module,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    shared: *std.Build.Module,
    link: *std.Build.Module,
    llama_config: LlamaConfig,
) void {
    if (llama_config.enabled) {
        const c_ggml = createGgmlCModule(b, target, llama_config.src);
        const backend = createBackendModule(b, target, optimize, c_ggml, shared, link);
        mod.addImport("backend", backend);
    }
}

/// `libggml-penzai` — the out-of-tree ggml backend. Stock llama.cpp loads it via
/// `GGML_BACKEND_PATH` + `--device penzai`. Same backend core module as the
/// in-process path (so it stays llama-free), rooted on the `ggml_backend_init`
/// entry shim and linking only libggml-base.
fn addPenzaiBackendLib(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    shared: *std.Build.Module,
    link: *std.Build.Module,
    llama_config: LlamaConfig,
) void {
    if (!llama_config.enabled) return; // needs the ggml headers + libggml-base

    const c_ggml = createGgmlCModule(b, target, llama_config.src);
    const backend = createBackendModule(b, target, optimize, c_ggml, shared, link);

    const lib_mod = b.createModule(.{
        .root_source_file = b.path("host/backend/backend_dl.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    lib_mod.addImport("c_ggml", c_ggml);
    lib_mod.addImport("shared", shared);
    lib_mod.addImport("link", link);
    lib_mod.addImport("backend", backend);

    // The backend core calls into ggml (ggml_nbytes, ggml_backend_buffer_init, …),
    // which resolve against the host process's libggml-base.so at load time; link it
    // so the .so carries the NEEDED reference (and the dlopen smoke can satisfy it).
    const lib_path = b.pathJoin(&.{ llama_config.lib, "lib" });
    lib_mod.addLibraryPath(.{ .cwd_relative = lib_path });
    lib_mod.addRPath(.{ .cwd_relative = lib_path });
    lib_mod.linkSystemLibrary("ggml-base", .{});

    const lib = b.addLibrary(.{
        .name = "ggml-penzai",
        .root_module = lib_mod,
        .linkage = .dynamic,
    });
    const install = b.addInstallArtifact(lib, .{});
    b.step("backend-so", "Build libggml-penzai — the out-of-tree ggml backend").dependOn(&install.step);
}
