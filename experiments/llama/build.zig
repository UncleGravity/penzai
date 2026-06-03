const std = @import("std");

const Experiment = struct {
    name: []const u8,
    source: []const u8,
    run_step: []const u8,
    run_description: []const u8,
};

const experiments = [_]Experiment{
    .{
        .name = "llama-backend-e2e",
        .source = "src/backend_e2e.zig",
        .run_step = "run-backend-e2e",
        .run_description = "Run llama.cpp decode through the Zig ggml backend",
    },
    .{
        .name = "llama-remote-buffer-e2e",
        .source = "src/remote_buffer_e2e.zig",
        .run_step = "run-remote-buffer-e2e",
        .run_description = "Run llama.cpp decode through the Zig fake remote buffer backend",
    },
    .{
        .name = "llama-partial-offload",
        .source = "src/partial_offload.zig",
        .run_step = "run-partial-offload",
        .run_description = "Run llama.cpp decode with only MUL_MAT offloaded",
    },
    .{
        .name = "llama-op-census",
        .source = "src/op_census.zig",
        .run_step = "run-op-census",
        .run_description = "Run llama.cpp decode and print a ggml op census",
    },
    .{
        .name = "llama-binding-lower-dryrun",
        .source = "src/binding_lower_dryrun.zig",
        .run_step = "run-binding-lower-dryrun",
        .run_description = "Run llama.cpp decode through remote bindings plus dry-run lowering",
    },
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

    if (llama_src.len == 0 or llama_lib.len == 0) {
        @panic("pass -Dllama-src=/path/to/llama.cpp -Dllama-lib=/path/to/llama-install");
    }

    const options = b.addOptions();
    options.addOption([]const u8, "default_model_path", model_path);

    const translate_c = b.addTranslateC(.{
        .root_source_file = b.path("src/c_api.h"),
        .target = target,
        // Zig 0.17-dev currently emits `translate-c -Osafe` for ReleaseSafe,
        // but this translate-c binary only accepts the older mode names.
        .optimize = .Debug,
        .link_libc = true,
    });
    translate_c.addIncludePath(.{ .cwd_relative = b.pathJoin(&.{ llama_src, "include" }) });
    translate_c.addIncludePath(.{ .cwd_relative = b.pathJoin(&.{ llama_src, "ggml", "include" }) });
    translate_c.addIncludePath(.{ .cwd_relative = b.pathJoin(&.{ llama_src, "ggml", "src" }) });

    for (experiments, 0..) |experiment, index| {
        const root_mod = b.createModule(.{
            .root_source_file = b.path(experiment.source),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .link_libcpp = true,
        });

        root_mod.addOptions("build_options", options);
        root_mod.addImport("c", translate_c.createModule());

        root_mod.addLibraryPath(.{ .cwd_relative = b.pathJoin(&.{ llama_lib, "lib" }) });
        root_mod.addRPath(.{ .cwd_relative = b.pathJoin(&.{ llama_lib, "lib" }) });

        root_mod.linkSystemLibrary("llama", .{});
        root_mod.linkSystemLibrary("ggml", .{});
        root_mod.linkSystemLibrary("ggml-base", .{});
        root_mod.linkSystemLibrary("ggml-cpu", .{});

        const exe = b.addExecutable(.{
            .name = experiment.name,
            .root_module = root_mod,
        });

        b.installArtifact(exe);

        const run = b.addRunArtifact(exe);
        const run_step = b.step(experiment.run_step, experiment.run_description);
        run_step.dependOn(&run.step);

        if (index == 0) {
            const default_run = b.step("run", experiment.run_description);
            default_run.dependOn(&run.step);
        }
    }
}
