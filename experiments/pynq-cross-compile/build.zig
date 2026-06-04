const std = @import("std");

const cross_smoke_board_binary = "/tmp/penzai-pynq-cross-smoke";
const stdlib_smoke_board_binary = "/tmp/penzai-pynq-stdlib-smoke";

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{
        .default_target = .{
            .cpu_arch = .arm,
            .os_tag = .linux,
            .abi = .gnueabihf,
            .cpu_model = .{ .explicit = &std.Target.arm.cpu.cortex_a9 },
        },
    });
    const optimize = b.standardOptimizeOption(.{});
    const board = b.option(
        []const u8,
        "board",
        "SSH destination for run-board, for example xilinx@pynq",
    ) orelse "xilinx@pynq";

    const cross_mod = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });

    const cross_exe = b.addExecutable(.{
        .name = "penzai-pynq-cross-smoke",
        .root_module = cross_mod,
    });
    b.installArtifact(cross_exe);

    const stdlib_mod = b.createModule(.{
        .root_source_file = b.path("src/stdlib_smoke.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });

    const stdlib_exe = b.addExecutable(.{
        .name = "penzai-pynq-stdlib-smoke",
        .root_module = stdlib_mod,
    });
    b.installArtifact(stdlib_exe);

    const copy = b.addSystemCommand(&.{ "scp" });
    copy.addFileArg(cross_exe.getEmittedBin());
    copy.addFileArg(stdlib_exe.getEmittedBin());
    copy.addArg(b.fmt("{s}:/tmp/", .{board}));

    const run_remote = b.addSystemCommand(&.{
        "ssh",
        board,
        b.fmt(
            "chmod +x {s} {s} && {s} && {s}",
            .{
                cross_smoke_board_binary,
                stdlib_smoke_board_binary,
                cross_smoke_board_binary,
                stdlib_smoke_board_binary,
            },
        ),
    });
    run_remote.step.dependOn(&copy.step);

    const run_board = b.step("run-board", "Copy the ARM binary to the PYNQ-Z1 and run it over SSH");
    run_board.dependOn(&run_remote.step);
}
