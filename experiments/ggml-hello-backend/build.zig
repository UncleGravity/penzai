const std = @import("std");

// Builds the experiment by compiling ggml-base's C/C++ sources with Zig's own
// bundled clang — no CMake, no system toolchain. This is the "one build"
// panzai is aiming for: ggml linked straight into the Zig artifact.
const ggml = "vendor/llama.cpp/ggml";

const base_c = [_][]const u8{
    "ggml.c",
    "ggml-alloc.c",
    "ggml-quants.c",
};

const base_cpp = [_][]const u8{
    "ggml.cpp",
    "ggml-backend.cpp",
    "ggml-backend-meta.cpp",
    "ggml-opt.cpp",
    "ggml-threading.cpp",
    "gguf.cpp",
};

const defines = [_][]const u8{
    "-D_DARWIN_C_SOURCE",
    "-DGGML_VERSION=\"0.13.1-panzai-hello\"",
    "-DGGML_COMMIT=\"hello\"",
    // ggml computes graph sizes via pointer arithmetic from a NULL base
    // (incr_ptr_aligned), which is UB in C and trips Zig's UB sanitizer when
    // these TUs are built in Debug. Disable it for ggml's own sources only;
    // our Zig keeps full safety. panzai's build will need the same.
    "-fno-sanitize=undefined",
};

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Translate the ggml C API into a Zig module via the build system
    // (@cImport was removed from the language in 0.15+).
    const translate = b.addTranslateC(.{
        .root_source_file = b.path("src/ggml_all.h"),
        .target = target,
        .optimize = optimize,
    });
    translate.addIncludePath(b.path(ggml ++ "/include"));
    translate.addIncludePath(b.path(ggml ++ "/src"));
    const c_mod = translate.createModule();

    const root = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    root.addImport("c", c_mod);

    const exe = b.addExecutable(.{
        .name = "hello-backend",
        .root_module = root,
    });

    const m = exe.root_module;
    m.addIncludePath(b.path(ggml ++ "/include"));
    m.addIncludePath(b.path(ggml ++ "/src"));

    const c_flags = defines ++ [_][]const u8{"-std=c11"};
    const cpp_flags = defines ++ [_][]const u8{"-std=c++17"};

    m.addCSourceFiles(.{ .root = b.path(ggml ++ "/src"), .files = &base_c, .flags = &c_flags });
    m.addCSourceFiles(.{ .root = b.path(ggml ++ "/src"), .files = &base_cpp, .flags = &cpp_flags });

    m.link_libc = true;
    m.link_libcpp = true;

    b.installArtifact(exe);

    const run = b.addRunArtifact(exe);
    const run_step = b.step("run", "Build and run the ggml hello-backend experiment");
    run_step.dependOn(&run.step);
}
