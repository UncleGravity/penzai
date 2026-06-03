const std = @import("std");

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

    if (llama_src.len == 0 or llama_lib.len == 0) {
        @panic("pass -Dllama-src=/path/to/llama.cpp -Dllama-lib=/path/to/llama-install");
    }

    const root_mod = b.createModule(.{
        .root_source_file = b.path("src/ggml_smoke.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .link_libcpp = true,
    });

    const translate_c = b.addTranslateC(.{
        .root_source_file = b.path("src/c_api.h"),
        .target = target,
        .optimize = .Debug,
        .link_libc = true,
    });
    translate_c.addIncludePath(.{ .cwd_relative = b.pathJoin(&.{ llama_src, "include" }) });
    translate_c.addIncludePath(.{ .cwd_relative = b.pathJoin(&.{ llama_src, "ggml", "include" }) });
    translate_c.addIncludePath(.{ .cwd_relative = b.pathJoin(&.{ llama_src, "ggml", "src" }) });
    root_mod.addImport("c", translate_c.createModule());

    const exe = b.addExecutable(.{
        .name = "penzai-ggml-smoke",
        .root_module = root_mod,
    });

    root_mod.addIncludePath(.{ .cwd_relative = b.pathJoin(&.{ llama_src, "include" }) });
    root_mod.addIncludePath(.{ .cwd_relative = b.pathJoin(&.{ llama_src, "ggml", "include" }) });
    root_mod.addIncludePath(.{ .cwd_relative = b.pathJoin(&.{ llama_src, "ggml", "src" }) });

    root_mod.addLibraryPath(.{ .cwd_relative = b.pathJoin(&.{ llama_lib, "lib" }) });
    root_mod.addRPath(.{ .cwd_relative = b.pathJoin(&.{ llama_lib, "lib" }) });

    root_mod.linkSystemLibrary("llama", .{});
    root_mod.linkSystemLibrary("ggml", .{});
    root_mod.linkSystemLibrary("ggml-base", .{});
    root_mod.linkSystemLibrary("ggml-cpu", .{});

    b.installArtifact(exe);

    const run = b.addRunArtifact(exe);

    const run_step = b.step("run", "Run the Zig/llama.cpp/ggml smoke test");
    run_step.dependOn(&run.step);
}
