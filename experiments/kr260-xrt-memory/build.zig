const std = @import("std");

const config_prelude =
    \\set -euo pipefail
    \\[[ -f config.env ]] || { echo "ERROR: missing config.env; copy config.env.example first" >&2; exit 1; }
    \\set -a
    \\source config.env
    \\set +a
;

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{
        .default_target = .{
            .cpu_arch = .aarch64,
            .os_tag = .linux,
            .abi = .gnu,
            .cpu_model = .{ .explicit = &std.Target.aarch64.cpu.cortex_a53 },
        },
    });
    const optimize = b.standardOptimizeOption(.{});
    const board_args = b.option(
        []const u8,
        "board-args",
        "Arguments passed to kr260-xrt-memory by zig build run-board",
    ) orelse "all";

    const mod = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });

    const exe = b.addExecutable(.{
        .name = "kr260-xrt-memory",
        .root_module = mod,
    });
    b.installArtifact(exe);

    const run_board = b.step("run-board", "Copy and run the KR260 memory probe over SSH");
    run_board.dependOn(&addRunBoardCommand(b, exe, board_args).step);
}

fn addRunBoardCommand(
    b: *std.Build,
    exe: *std.Build.Step.Compile,
    board_args: []const u8,
) *std.Build.Step.Run {
    const run = b.addSystemCommand(&.{
        "bash",
        "-lc",
        config_prelude ++
            \\
            \\: "${BOARD:?config.env must set BOARD}"
            \\: "${BOARD_TMP:?config.env must set BOARD_TMP}"
            \\BIN="${1:?missing probe binary}"
            \\ARGS="${2:-all}"
            \\
            \\case "$BOARD_TMP" in
            \\  /tmp/*) ;;
            \\  *) echo "ERROR: BOARD_TMP must be under /tmp, got: $BOARD_TMP" >&2; exit 1 ;;
            \\esac
            \\
            \\echo "== copy kr260-xrt-memory -> $BOARD:$BOARD_TMP =="
            \\ssh "$BOARD" "mkdir -p '$BOARD_TMP'"
            \\scp "$BIN" "$BOARD:$BOARD_TMP/kr260-xrt-memory"
            \\
            \\echo "== run kr260-xrt-memory $ARGS =="
            \\ssh "$BOARD" "BOARD_TMP='$BOARD_TMP' ARGS='$ARGS' bash -s" <<'REMOTE'
            \\set -euo pipefail
            \\chmod +x "$BOARD_TMP/kr260-xrt-memory"
            \\"$BOARD_TMP/kr260-xrt-memory" $ARGS
            \\REMOTE
        ,
        "zig-build",
    });
    run.addFileArg(exe.getEmittedBin());
    run.addArg(board_args);
    return run;
}
