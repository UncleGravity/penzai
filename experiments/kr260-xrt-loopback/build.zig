const std = @import("std");

const config_prelude =
    \\set -euo pipefail
    \\[[ -f config.env ]] || { echo "ERROR: missing config.env" >&2; exit 1; }
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

    const mod = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });

    const exe = b.addExecutable(.{
        .name = "kr260-xrt-loopback",
        .root_module = mod,
    });
    b.installArtifact(exe);

    const bitstream = b.step("bitstream", "Build loopback.bit.bin on the Vivado VM");
    bitstream.dependOn(&addBitstreamCommand(b).step);

    const deploy = b.step("deploy", "Install and load the XRT loopback app on the KR260");
    deploy.dependOn(&addDeployCommand(b).step);

    const verify = b.step("verify", "Copy and run the Zig verifier on the KR260");
    verify.dependOn(&addVerifyCommand(b, exe).step);

    const all_bitstream = addBitstreamCommand(b);
    const all_deploy = addDeployCommand(b);
    all_deploy.step.dependOn(&all_bitstream.step);
    const all_verify = addVerifyCommand(b, exe);
    all_verify.step.dependOn(&all_deploy.step);

    const all = b.step("all", "Build bitstream, deploy app, and run the Zig verifier");
    all.dependOn(&all_verify.step);
}

fn addBitstreamCommand(b: *std.Build) *std.Build.Step.Run {
    return b.addSystemCommand(&.{
        "bash",
        "-lc",
        config_prelude ++
            \\
            \\: "${VM:?config.env must set VM}"
            \\: "${VM_DIR:?config.env must set VM_DIR}"
            \\
            \\echo "== sync FPGA build inputs -> $VM:$VM_DIR =="
            \\ssh "$VM" "if not exist $VM_DIR mkdir $VM_DIR"
            \\scp fpga/build.tcl fpga/build.bat "$VM:$VM_DIR/"
            \\
            \\echo "== Vivado build on $VM =="
            \\ssh "$VM" "cd $VM_DIR && build.bat"
            \\
            \\echo "== fetch bitstream outputs -> fpga/out/ =="
            \\mkdir -p fpga/out
            \\scp "$VM:$VM_DIR/out/loopback.bit.bin" "$VM:$VM_DIR/out/loopback.bit" fpga/out/
            \\ls -la fpga/out/loopback.bit.bin
        ,
    });
}

fn addDeployCommand(b: *std.Build) *std.Build.Step.Run {
    return b.addSystemCommand(&.{
        "bash",
        "-lc",
        config_prelude ++
            \\
            \\: "${BOARD:?config.env must set BOARD}"
            \\: "${APP:?config.env must set APP}"
            \\: "${BOARD_TMP:?config.env must set BOARD_TMP}"
            \\
            \\case "$BOARD_TMP" in
            \\  /tmp/*) ;;
            \\  *) echo "ERROR: BOARD_TMP must be under /tmp, got: $BOARD_TMP" >&2; exit 1 ;;
            \\esac
            \\
            \\BIT="fpga/out/loopback.bit.bin"
            \\DTS="overlay/$APP.dts"
            \\[[ -f "$BIT" ]] || { echo "ERROR: missing $BIT. Run 'zig build bitstream' first." >&2; exit 1; }
            \\[[ -f "$DTS" ]] || { echo "ERROR: missing $DTS." >&2; exit 1; }
            \\
            \\echo "== copy app inputs -> $BOARD:$BOARD_TMP =="
            \\ssh "$BOARD" "rm -rf '$BOARD_TMP' && mkdir -p '$BOARD_TMP'"
            \\scp "$BIT" "$DTS" "$BOARD:$BOARD_TMP/"
            \\
            \\echo "== package firmware app =="
            \\ssh "$BOARD" "APP='$APP' BOARD_TMP='$BOARD_TMP' bash -s" <<'REMOTE'
            \\set -euo pipefail
            \\FW="/lib/firmware/xilinx/$APP"
            \\cd "$BOARD_TMP"
            \\
            \\dtc -@ -O dtb -o "$APP.dtbo" "$APP.dts"
            \\sudo mkdir -p "$FW"
            \\sudo rm -f "$FW"/*.bit.bin "$FW"/*.dtbo "$FW"/shell.json
            \\sudo cp "$APP.dtbo" "$FW/$APP.dtbo"
            \\sudo cp loopback.bit.bin "$FW/$APP.bit.bin"
            \\printf '{"shell_type":"XRT_FLAT","num_slots":"1"}\n' | sudo tee "$FW/shell.json" >/dev/null
            \\sudo ls -l "$FW"
            \\REMOTE
            \\
            \\echo "== load $APP =="
            \\ssh "$BOARD" "APP='$APP' bash -s" <<'REMOTE'
            \\set -euo pipefail
            \\sudo xmutil unloadapp 2>/dev/null || true
            \\sudo xmutil loadapp "$APP"
            \\sudo xmutil listapps
            \\REMOTE
        ,
    });
}

fn addVerifyCommand(b: *std.Build, exe: *std.Build.Step.Compile) *std.Build.Step.Run {
    const run = b.addSystemCommand(&.{
        "bash",
        "-lc",
        config_prelude ++
            \\
            \\: "${BOARD:?config.env must set BOARD}"
            \\: "${BOARD_TMP:?config.env must set BOARD_TMP}"
            \\BIN="${1:?missing verifier binary}"
            \\
            \\echo "== copy Zig verifier -> $BOARD:$BOARD_TMP =="
            \\ssh "$BOARD" "mkdir -p '$BOARD_TMP'"
            \\scp "$BIN" "$BOARD:$BOARD_TMP/kr260-xrt-loopback"
            \\
            \\echo "== run Zig verifier =="
            \\ssh "$BOARD" "BOARD_TMP='$BOARD_TMP' bash -s" <<'REMOTE'
            \\set -euo pipefail
            \\chmod +x "$BOARD_TMP/kr260-xrt-loopback"
            \\sudo "$BOARD_TMP/kr260-xrt-loopback"
            \\REMOTE
        ,
        "zig-build",
    });
    run.addFileArg(exe.getEmittedBin());
    return run;
}
