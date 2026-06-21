const std = @import("std");

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

    const enable_profiling = b.option(bool, "profiling", "Compile device-side profiling collection (default true)") orelse true;

    const host_options = createBuildOptions(b, llama_config.enabled, enable_profiling, model_path);
    const host_shared = createSharedModule(b, target, optimize);
    const host_c = if (llama_config.enabled) createLlamaCModule(b, target, llama_config.src) else null;
    const host_runtime = createRuntimeModule(b, target, optimize, host_options, host_shared);
    const host_server = createServerModule(b, target, optimize, host_options, host_shared, host_runtime);
    const host_link = createHostLinkModule(b, target, optimize, host_shared, host_runtime, host_server);

    const host_exe = addPenzai(b, target, optimize, host_options, host_shared, host_runtime, host_server, host_link, llama_config, host_c);
    b.installArtifact(host_exe);
    const install_penzai = b.addInstallArtifact(host_exe, .{});
    b.step("install-penzai", "Install the host penzai CLI").dependOn(&install_penzai.step);

    const kr260_target = kr260Target(b);
    const device_options = createBuildOptions(b, false, enable_profiling, "");
    const kr260_shared = createSharedModule(b, kr260_target, optimize);
    const kr260_runtime = createRuntimeModule(b, kr260_target, optimize, device_options, kr260_shared);
    const kr260_server = createServerModule(b, kr260_target, optimize, device_options, kr260_shared, kr260_runtime);
    const device_exe = addPenzaid(b, kr260_target, optimize, device_options, kr260_shared, kr260_runtime, kr260_server, "penzaid");
    b.step("device", "Cross-compile the KR260 device daemon").dependOn(&device_exe.step);
    const install_penzaid = b.addInstallArtifact(device_exe, .{});
    b.step("install-penzaid", "Install the KR260 penzaid daemon").dependOn(&install_penzaid.step);

    const native_device_options = createBuildOptions(b, false, enable_profiling, "");
    const native_runtime = createRuntimeModule(b, target, optimize, native_device_options, host_shared);
    const native_server = createServerModule(b, target, optimize, native_device_options, host_shared, native_runtime);
    const native_device_exe = addPenzaid(b, target, optimize, native_device_options, host_shared, native_runtime, native_server, "penzaid-native");
    const install_native_device = b.addInstallArtifact(native_device_exe, .{});
    b.step("device-native", "Build a native penzaid for local TCP smoke tests").dependOn(&native_device_exe.step);
    b.step("install-penzaid-native", "Install a native penzaid for local TCP smoke tests").dependOn(&install_native_device.step);

    const all = b.step("all", "Build host, KR260 device, and native device binaries");
    all.dependOn(&host_exe.step);
    all.dependOn(&device_exe.step);
    all.dependOn(&native_device_exe.step);

    addTests(b, target, optimize, host_options, host_shared, host_runtime, host_server, host_link, llama_config, host_c);
    addRtlSteps(b, target, optimize);
}

// RTL file sets (paths relative to the repo root, where `zig build` runs). The
// matmul op lives in rtl/matmul/; the reusable fp32 cells in rtl/fp/ (a leaf
// library shared with future ops). One module per file, file named after the module.
const fp_rtl = [_][]const u8{
    "fpga/rtl/fp/fp32_add_pipe.v", "fpga/rtl/fp/fp32_mul_pipe.v",
    "fpga/rtl/fp/fp16_to_fp32.v",  "fpga/rtl/fp/int_to_fp32.v",
};
const matmul_rtl = [_][]const u8{
    "fpga/rtl/matmul/matmul_kernel.v", "fpga/rtl/matmul/matmul_rowblock.v",
    "fpga/rtl/matmul/matmul_reducer.v",
} ++ fp_rtl;
const matmul_top_rtl = [_][]const u8{
    "fpga/rtl/matmul/matmul_top.v",
} ++ matmul_rtl;

const matmul_rtl_args = "fpga/rtl/matmul/matmul_top.v fpga/rtl/matmul/matmul_kernel.v fpga/rtl/matmul/matmul_rowblock.v fpga/rtl/matmul/matmul_reducer.v fpga/rtl/fp/fp32_add_pipe.v fpga/rtl/fp/fp32_mul_pipe.v fpga/rtl/fp/fp16_to_fp32.v fpga/rtl/fp/int_to_fp32.v";

// Flash-attention fp leaves (rebuild 1D-4): exp + reciprocal LUT units and their
// shared interpolation, on the rtl/fp leaf library. The cosim harness top wraps
// both leaves so one Verilator model exercises them against flash_ref.
const flash_fp_rtl = [_][]const u8{
    "fpga/rtl/flash_attn/flash_fp_top.v",
    "fpga/rtl/flash_attn/fp_exp.v",
    "fpga/rtl/flash_attn/fp_recip.v",
    "fpga/rtl/flash_attn/fp_dot.v",
    "fpga/rtl/flash_attn/fp_interp.v",
    "fpga/rtl/fp/fp32_add_pipe.v",
    "fpga/rtl/fp/fp32_mul_pipe.v",
    "fpga/rtl/fp/fp16_to_fp32.v",
    "fpga/rtl/fp/int_to_fp32.v",
};
// The online-softmax transform on the exp leaf (+ its interp/fp deps).
const flash_softmax_rtl = [_][]const u8{
    "fpga/rtl/flash_attn/flash_softmax.v",
    "fpga/rtl/flash_attn/fp_exp.v",
    "fpga/rtl/flash_attn/fp_interp.v",
    "fpga/rtl/fp/fp32_add_pipe.v",
    "fpga/rtl/fp/fp32_mul_pipe.v",
    "fpga/rtl/fp/int_to_fp32.v",
};
// The full kernel composing fp_dot + flash_softmax + fp_recip.
const flash_kernel_rtl = [_][]const u8{
    "fpga/rtl/flash_attn/flash_kernel.v",
    "fpga/rtl/flash_attn/fp_dot.v",
    "fpga/rtl/flash_attn/fp_addtree.v",
    "fpga/rtl/flash_attn/fp_axpy8.v",
    "fpga/rtl/flash_attn/flash_softmax.v",
    "fpga/rtl/flash_attn/fp_exp.v",
    "fpga/rtl/flash_attn/fp_recip.v",
    "fpga/rtl/flash_attn/fp_interp.v",
    "fpga/rtl/fp/fp32_add_pipe.v",
    "fpga/rtl/fp/fp32_mul_pipe.v",
    "fpga/rtl/fp/fp16_to_fp32.v",
    "fpga/rtl/fp/int_to_fp32.v",
};
// The AXI-Lite/DMA wrapper (includes the generated flash_regs.vh) + the kernel.
const flash_top_rtl = [_][]const u8{
    "fpga/rtl/flash_attn/flash_top.v",
} ++ flash_kernel_rtl;

fn addRtlSteps(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) void {
    // regmap -> the generated bitstream-contract files (single source: the
    // matmul regmap module). One emitter, two artifacts: the Verilog register
    // header (offsets/resets + MATMUL_COLS_MAX) and the Vivado address-map TCL.
    const emit = b.addExecutable(.{
        .name = "regmap-emit",
        .root_module = b.createModule(.{
            .root_source_file = b.path("fpga/regmap/emit.zig"),
            .target = b.graph.host,
            .optimize = .Debug,
        }),
    });
    const update = b.addUpdateSourceFiles();

    const run_vh = b.addRunArtifact(emit);
    run_vh.addArgs(&.{ "matmul", "vh" });
    update.addCopyFileToSource(run_vh.captureStdOut(.{}), "fpga/rtl/matmul/matmul_regs.vh");

    const run_tcl = b.addRunArtifact(emit);
    run_tcl.addArgs(&.{ "matmul", "tcl" });
    update.addCopyFileToSource(run_tcl.captureStdOut(.{}), "fpga/bitstreams/q1a8-w256-mc/tcl/address_map.tcl");

    const run_flash_vh = b.addRunArtifact(emit);
    run_flash_vh.addArgs(&.{ "flash", "vh" });
    update.addCopyFileToSource(run_flash_vh.captureStdOut(.{}), "fpga/rtl/flash_attn/flash_regs.vh");

    const run_flash_tcl = b.addRunArtifact(emit);
    run_flash_tcl.addArgs(&.{ "flash", "tcl" });
    update.addCopyFileToSource(run_flash_tcl.captureStdOut(.{}), "fpga/bitstreams/flash-v1/tcl/address_map.tcl");

    b.step("regmap", "Generate matmul/flash register headers + address maps").dependOn(&update.step);

    // Verilator lint of the deployable matmul RTL, including matmul_top + the
    // counter bank + the generated header. This is the structural gate for the
    // gateware that the cosim (which drives the core) does not exercise.
    const lint = b.addSystemCommand(&.{
        "sh",                                                                                                                                                                     "-c",
        "verilator --lint-only -Wall -Wno-UNUSEDSIGNAL -Wno-UNUSEDPARAM -Wno-PINMISSING +incdir+fpga/rtl/matmul --top-module matmul_top " ++ matmul_rtl_args,
    });
    b.step("lint-rtl", "Verilator lint the matmul RTL").dependOn(&lint.step);

    addCosim(b, target, optimize, "test-rtl-matmul", "Verilator cosim: matmul kernel vs matmul_ref", "matmul_kernel", "fpga/sim/matmul_kernel", &matmul_rtl);
    addCosim(b, target, optimize, "test-rtl-matmul-top", "Verilator cosim: matmul AXI-Lite top (four-port zip) vs matmul_ref", "matmul_top", "fpga/sim/matmul_top", &matmul_top_rtl);
    addFlashCosim(b, target, optimize, "test-rtl-flash-fp", "Verilator cosim: flash fp_exp/fp_recip/fp_dot vs flash_ref", "flash_fp_top", "fpga/sim/flash_fp", &flash_fp_rtl);
    addFlashCosim(b, target, optimize, "test-rtl-flash-softmax", "Verilator cosim: flash online-softmax step vs flash_ref", "flash_softmax", "fpga/sim/flash_softmax", &flash_softmax_rtl);
    addFlashCosim(b, target, optimize, "test-rtl-flash-kernel", "Verilator cosim: full flash kernel vs flash_ref.attendHead", "flash_kernel", "fpga/sim/flash_kernel", &flash_kernel_rtl);
    addFlashCosim(b, target, optimize, "test-rtl-flash-top", "Verilator cosim: flash_top AXI-Lite/DMA wrapper vs flash_ref", "flash_top", "fpga/sim/flash_top", &flash_top_rtl);
}

// Verilator cosim of a flash RTL top, driven from Zig, checked vs flash_ref.
// Separate from addCosim because the flash tb imports the standalone flash_ref
// model (no layout/pack/matmul_ref) and includes the flash LUT header.
fn addFlashCosim(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    step_name: []const u8,
    desc: []const u8,
    top: []const u8,
    dir: []const u8,
    rtl: []const []const u8,
) void {
    const step = b.step(step_name, desc);
    const verilator = b.findProgram(&.{"verilator"}, &.{}) catch {
        const msg = b.addSystemCommand(&.{ "sh", "-c", "echo 'verilator not found — run inside: nix develop' >&2; exit 1" });
        step.dependOn(&msg.step);
        return;
    };
    const vroot = std.mem.trim(u8, b.run(&.{ verilator, "--getenv", "VERILATOR_ROOT" }), " \n\r");
    const gen = b.fmt("{s}/obj_dir", .{dir});

    const vcmd = b.addSystemCommand(&.{ verilator, "--cc", "--build", "--lib-create" });
    vcmd.addArg(top);
    vcmd.addArgs(&.{ "-Wno-fatal", "-Wno-WIDTHEXPAND", "-Wno-UNUSEDSIGNAL", "--top-module" });
    vcmd.addArg(top);
    vcmd.addArgs(&.{ "--Mdir", gen });
    vcmd.addArg("+incdir+fpga/rtl/flash_attn");
    vcmd.addArgs(rtl);

    const tb_mod = b.createModule(.{
        .root_source_file = b.path(b.fmt("{s}/tb.zig", .{dir})),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    tb_mod.addImport("flash_ref", b.createModule(.{
        .root_source_file = b.path("fpga/sim/support/flash_ref.zig"),
        .target = target,
        .optimize = optimize,
    }));
    // The flash regmap (offsets/resets), so the flash_top tb drives AXI-Lite from the
    // single source. Harmless to the other flash tbs, which simply don't import it.
    tb_mod.addImport("regmap", b.createModule(.{
        .root_source_file = b.path("fpga/regmap/flash_attn.zig"),
        .target = target,
        .optimize = optimize,
    }));
    tb_mod.addIncludePath(b.path(dir));
    tb_mod.addIncludePath(.{ .cwd_relative = b.fmt("{s}/include", .{vroot}) });
    tb_mod.addIncludePath(.{ .cwd_relative = b.fmt("{s}/include/vltstd", .{vroot}) });
    tb_mod.addIncludePath(b.path(gen));
    tb_mod.addCSourceFile(.{
        .file = b.path(b.fmt("{s}/shim.cpp", .{dir})),
        .flags = &.{ "-std=c++17", b.fmt("-I{s}", .{gen}), b.fmt("-I{s}/include", .{vroot}), b.fmt("-I{s}/include/vltstd", .{vroot}) },
    });
    tb_mod.addObjectFile(b.path(b.fmt("{s}/lib{s}.a", .{ gen, top })));
    tb_mod.linkSystemLibrary("pthread", .{});
    tb_mod.link_libcpp = true;

    const tb = b.addExecutable(.{ .name = b.fmt("{s}-cosim", .{top}), .root_module = tb_mod });
    tb.step.dependOn(&vcmd.step);
    step.dependOn(&b.addRunArtifact(tb).step);
}

fn attachCosimSupport(b: *std.Build, mod: *std.Build.Module, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) void {
    // The one production layout source; the sim layout shim derives its block
    // constants from it so cosim and device never drift (rebuild 1D-2c).
    const shared_layout = b.createModule(.{ .root_source_file = b.path("shared/layout.zig"), .target = target, .optimize = optimize });
    const layout = b.createModule(.{ .root_source_file = b.path("fpga/sim/support/layout.zig"), .target = target, .optimize = optimize });
    layout.addImport("shared_layout", shared_layout);
    const pack = b.createModule(.{ .root_source_file = b.path("fpga/sim/support/pack.zig"), .target = target, .optimize = optimize });
    const ref = b.createModule(.{ .root_source_file = b.path("fpga/sim/support/matmul_ref.zig"), .target = target, .optimize = optimize });
    pack.addImport("layout", layout);
    ref.addImport("layout", layout);
    mod.addImport("layout", layout);
    mod.addImport("pack", pack);
    mod.addImport("matmul_ref", ref);
}

// Verilator cosim of a kernel top, driven from Zig, checked vs matmul_ref.
fn addCosim(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    step_name: []const u8,
    desc: []const u8,
    top: []const u8,
    dir: []const u8,
    rtl: []const []const u8,
) void {
    const step = b.step(step_name, desc);
    const verilator = b.findProgram(&.{"verilator"}, &.{}) catch {
        const msg = b.addSystemCommand(&.{ "sh", "-c", "echo 'verilator not found — run inside: nix develop' >&2; exit 1" });
        step.dependOn(&msg.step);
        return;
    };
    const vroot = std.mem.trim(u8, b.run(&.{ verilator, "--getenv", "VERILATOR_ROOT" }), " \n\r");
    const gen = b.fmt("{s}/obj_dir", .{dir});

    const vcmd = b.addSystemCommand(&.{ verilator, "--cc", "--build", "--lib-create" });
    vcmd.addArg(top);
    vcmd.addArgs(&.{ "-Wno-fatal", "-Wno-WIDTHEXPAND", "-Wno-UNUSEDSIGNAL", "--top-module" });
    vcmd.addArg(top);
    vcmd.addArgs(&.{ "--Mdir", gen });
    vcmd.addArg("+incdir+fpga/rtl/matmul");
    vcmd.addArgs(rtl);

    const tb_mod = b.createModule(.{
        .root_source_file = b.path(b.fmt("{s}/tb.zig", .{dir})),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    attachCosimSupport(b, tb_mod, target, optimize);
    tb_mod.addIncludePath(b.path(dir));
    tb_mod.addIncludePath(.{ .cwd_relative = b.fmt("{s}/include", .{vroot}) });
    tb_mod.addIncludePath(.{ .cwd_relative = b.fmt("{s}/include/vltstd", .{vroot}) });
    tb_mod.addIncludePath(b.path(gen));
    tb_mod.addCSourceFile(.{
        .file = b.path(b.fmt("{s}/shim.cpp", .{dir})),
        .flags = &.{ "-std=c++17", b.fmt("-I{s}", .{gen}), b.fmt("-I{s}/include", .{vroot}), b.fmt("-I{s}/include/vltstd", .{vroot}) },
    });
    tb_mod.addObjectFile(b.path(b.fmt("{s}/lib{s}.a", .{ gen, top })));
    tb_mod.linkSystemLibrary("pthread", .{});
    tb_mod.link_libcpp = true;

    const tb = b.addExecutable(.{ .name = b.fmt("{s}-cosim", .{top}), .root_module = tb_mod });
    tb.step.dependOn(&vcmd.step);
    step.dependOn(&b.addRunArtifact(tb).step);
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
    build_options: *std.Build.Module,
    shared: *std.Build.Module,
    runtime: *std.Build.Module,
    server: *std.Build.Module,
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
    addHostImports(host_mod, build_options, shared, runtime, server, link);
    if (llama_config.enabled) addLlamaSupport(b, host_mod, llama_config, c_mod.?);

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

/// The generated register map (offsets/resets parsed from fpga/regmap/matmul.regmap),
/// consumed by the device PL driver. A cross-boundary contract module, like shared.
fn createRegmapModule(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
) *std.Build.Module {
    return b.createModule(.{
        .root_source_file = b.path("fpga/regmap/matmul.zig"),
        .target = target,
        .optimize = optimize,
    });
}

/// The flash_attn register map, consumed by the PL flash tenant (device/pl/flash_attn.zig).
fn createFlashRegmapModule(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
) *std.Build.Module {
    return b.createModule(.{
        .root_source_file = b.path("fpga/regmap/flash_attn.zig"),
        .target = target,
        .optimize = optimize,
    });
}

fn createRuntimeModule(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    build_options: *std.Build.Module,
    shared: *std.Build.Module,
) *std.Build.Module {
    const runtime = b.createModule(.{
        .root_source_file = b.path("device/runtime.zig"),
        .target = target,
        .optimize = optimize,
    });
    addCommonImports(runtime, build_options, shared);
    runtime.addImport("regmap", createRegmapModule(b, target, optimize));
    runtime.addImport("flash_regmap", createFlashRegmapModule(b, target, optimize));
    return runtime;
}

fn createServerModule(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    build_options: *std.Build.Module,
    shared: *std.Build.Module,
    runtime: *std.Build.Module,
) *std.Build.Module {
    const server = b.createModule(.{
        .root_source_file = b.path("device/server.zig"),
        .target = target,
        .optimize = optimize,
    });
    server.addImport("build_options", build_options);
    server.addImport("shared", shared);
    server.addImport("runtime", runtime);
    return server;
}

fn createHostLinkModule(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    shared: *std.Build.Module,
    runtime: *std.Build.Module,
    server: *std.Build.Module,
) *std.Build.Module {
    const link = b.createModule(.{
        .root_source_file = b.path("host/link.zig"),
        .target = target,
        .optimize = optimize,
    });
    link.addImport("shared", shared);
    link.addImport("runtime", runtime);
    link.addImport("server", server);
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
    runtime: *std.Build.Module,
    server: *std.Build.Module,
    link: *std.Build.Module,
) void {
    addDeviceImports(mod, build_options, shared, runtime, server);
    mod.addImport("link", link);
}

fn addTests(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    build_options: *std.Build.Module,
    shared: *std.Build.Module,
    runtime: *std.Build.Module,
    server: *std.Build.Module,
    link: *std.Build.Module,
    llama_config: LlamaConfig,
    c_mod: ?*std.Build.Module,
) void {
    const test_step = b.step("test", "Run host-only unit and fake full-stack tests");

    addSharedTest(b, test_step, "shared/root.zig", target, optimize, shared);
    addSharedTest(b, test_step, "shared/protocol/framing.zig", target, optimize, shared);
    addSharedTest(b, test_step, "shared/protocol/transport.zig", target, optimize, shared);
    addSharedTest(b, test_step, "shared/protocol/wire.zig", target, optimize, shared);
    addSharedTest(b, test_step, "shared/profiling.zig", target, optimize, shared);
    addSharedTest(b, test_step, "shared/layout.zig", target, optimize, shared);
    addSharedTest(b, test_step, "fpga/regmap/matmul.zig", target, optimize, shared);
    addSharedTest(b, test_step, "fpga/regmap/flash_attn.zig", target, optimize, shared);

    addDeviceTest(b, test_step, "device/profile.zig", target, optimize, build_options, shared);
    addDeviceTest(b, test_step, "device/mem/heap.zig", target, optimize, build_options, shared);
    addDeviceTest(b, test_step, "device/runtime.zig", target, optimize, build_options, shared);
    addDeviceTest(b, test_step, "device/ps/activations.zig", target, optimize, build_options, shared);
    addDeviceTest(b, test_step, "device/ps/elemwise.zig", target, optimize, build_options, shared);
    addDeviceTest(b, test_step, "device/ps/flash_attn.zig", target, optimize, build_options, shared);
    addDeviceTest(b, test_step, "device/ps/matmul_q1a8.zig", target, optimize, build_options, shared);
    addDeviceTest(b, test_step, "device/ps/rmsnorm.zig", target, optimize, build_options, shared);
    addDeviceTest(b, test_step, "device/ps/rows.zig", target, optimize, build_options, shared);
    addDeviceTest(b, test_step, "device/ps/rope.zig", target, optimize, build_options, shared);
    addDeviceTest(b, test_step, "device/ps/select.zig", target, optimize, build_options, shared);
    addDeviceTest(b, test_step, "device/ps/pad.zig", target, optimize, build_options, shared);
    addDeviceTest(b, test_step, "device/ps/softmax.zig", target, optimize, build_options, shared);
    addSharedTest(b, test_step, "device/pl/gather.zig", target, optimize, shared);
    addSharedTest(b, test_step, "device/pl/flash_feed.zig", target, optimize, shared);
    addDeviceRuntimeTest(b, test_step, "device/main.zig", target, optimize, build_options, shared, runtime, server);

    addHostTest(b, test_step, "host/run.zig", target, optimize, build_options, shared, runtime, server, link, llama_config, c_mod, true);
    addHostTest(b, test_step, "test/fullstack_fake.zig", target, optimize, build_options, shared, runtime, server, link, llama_config, c_mod, false);
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
    });
    mod.addImport("shared", shared);
    const test_exe = b.addTest(.{ .root_module = mod });
    step.dependOn(&b.addRunArtifact(test_exe).step);
}

fn addDeviceTest(
    b: *std.Build,
    step: *std.Build.Step,
    path: []const u8,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    build_options: *std.Build.Module,
    shared: *std.Build.Module,
) void {
    const mod = b.createModule(.{
        .root_source_file = b.path(path),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    addCommonImports(mod, build_options, shared);
    mod.addImport("regmap", createRegmapModule(b, target, optimize));
    mod.addImport("flash_regmap", createFlashRegmapModule(b, target, optimize));
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
    path: []const u8,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    build_options: *std.Build.Module,
    shared: *std.Build.Module,
    runtime: *std.Build.Module,
    server: *std.Build.Module,
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
    addHostImports(mod, build_options, shared, runtime, server, link);
    if (with_llama) addLlamaSupport(b, mod, llama_config, c_mod.?);
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
