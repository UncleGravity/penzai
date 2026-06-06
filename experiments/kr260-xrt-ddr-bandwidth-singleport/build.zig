const std = @import("std");

const config_prelude =
    \\set -euo pipefail
    \\[[ -f config.env ]] || { echo "ERROR: missing config.env; copy config.env.example first" >&2; exit 1; }
    \\set -a
    \\source config.env
    \\set +a
;

const binary_name = "kr260-xrt-ddr-bandwidth-singleport";
const app_name = "penzai-ddr-bandwidth-singleport";
const bit_prefix = "penzai-ddr-bandwidth-singleport";

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
    const variant = b.option(
        []const u8,
        "variant",
        "Hardware variant: w128-f100, w128-f200, w128-f250, or w128-f300",
    ) orelse "w128-f100";
    const board_args = b.option(
        []const u8,
        "board-args",
        "Arguments passed to the board binary by zig build run",
    ) orelse "run";

    const options = b.addOptions();
    options.addOption([]const u8, "variant", variant);

    const mod = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    mod.addOptions("build_options", options);

    const exe = b.addExecutable(.{
        .name = binary_name,
        .root_module = mod,
    });
    b.installArtifact(exe);

    const bitstream = b.step("bitstream", "Build the selected bitstream on the Vivado VM");
    bitstream.dependOn(&addBitstreamCommand(b, variant).step);

    const deploy = b.step("deploy", "Install and load the selected XRT app on the KR260");
    deploy.dependOn(&addDeployCommand(b, variant).step);

    const run = b.step("run", "Copy and run the benchmark on the KR260");
    run.dependOn(&addRunCommand(b, exe, board_args).step);

    const all_bitstream = addBitstreamCommand(b, variant);
    const all_deploy = addDeployCommand(b, variant);
    all_deploy.step.dependOn(&all_bitstream.step);
    const all_run = addRunCommand(b, exe, board_args);
    all_run.step.dependOn(&all_deploy.step);

    const all = b.step("all", "Build bitstream, deploy app, and run the benchmark");
    all.dependOn(&all_run.step);
}

fn addBitstreamCommand(b: *std.Build, variant: []const u8) *std.Build.Step.Run {
    const run = b.addSystemCommand(&.{
        "bash",
        "-lc",
        config_prelude ++
            \\
            \\: "${VM:?config.env must set VM}"
            \\: "${VM_DIR:?config.env must set VM_DIR}"
            \\VARIANT="${1:?missing variant}"
            \\BIT_PREFIX="${2:?missing bit prefix}"
            \\
            \\echo "== sync FPGA build inputs -> $VM:$VM_DIR =="
            \\ssh "$VM" "if not exist $VM_DIR mkdir $VM_DIR"
            \\ssh "$VM" "if not exist $VM_DIR\\rtl mkdir $VM_DIR\\rtl"
            \\scp fpga/build.tcl fpga/build.bat "$VM:$VM_DIR/"
            \\scp fpga/rtl/*.v "$VM:$VM_DIR/rtl/"
            \\
            \\echo "== Vivado build variant=$VARIANT on $VM =="
            \\ssh "$VM" "cd $VM_DIR && build.bat $VARIANT"
            \\
            \\echo "== fetch bitstream outputs -> fpga/out/ =="
            \\mkdir -p fpga/out
            \\scp "$VM:$VM_DIR/out/$BIT_PREFIX-$VARIANT.bit.bin" "$VM:$VM_DIR/out/$BIT_PREFIX-$VARIANT.bit" fpga/out/
            \\ls -la "fpga/out/$BIT_PREFIX-$VARIANT.bit.bin"
        ,
        "zig-build",
    });
    run.addArg(variant);
    run.addArg(bit_prefix);
    return run;
}

fn addDeployCommand(b: *std.Build, variant: []const u8) *std.Build.Step.Run {
    const run = b.addSystemCommand(&.{
        "bash",
        "-lc",
        config_prelude ++
            \\
            \\: "${BOARD:?config.env must set BOARD}"
            \\: "${BOARD_TMP:?config.env must set BOARD_TMP}"
            \\APP="${APP:-penzai-ddr-bandwidth-singleport}"
            \\VARIANT="${1:?missing variant}"
            \\BIT_PREFIX="${2:?missing bit prefix}"
            \\
            \\case "$BOARD_TMP" in
            \\  /tmp/*) ;;
            \\  *) echo "ERROR: BOARD_TMP must be under /tmp, got: $BOARD_TMP" >&2; exit 1 ;;
            \\esac
            \\
            \\BIT_NAME="$BIT_PREFIX-$VARIANT.bit.bin"
            \\BIT="fpga/out/$BIT_NAME"
            \\DTS_TEMPLATE="overlay/penzai-ddr-bandwidth-singleport.dts"
            \\[[ -f "$BIT" ]] || { echo "ERROR: missing $BIT. Run 'zig build bitstream -Dvariant=$VARIANT' first." >&2; exit 1; }
            \\[[ -f "$DTS_TEMPLATE" ]] || { echo "ERROR: missing $DTS_TEMPLATE." >&2; exit 1; }
            \\
            \\echo "== copy app inputs -> $BOARD:$BOARD_TMP =="
            \\ssh "$BOARD" "rm -rf '$BOARD_TMP' && mkdir -p '$BOARD_TMP'"
            \\scp "$BIT" "$DTS_TEMPLATE" "$BOARD:$BOARD_TMP/"
            \\
            \\echo "== package firmware app variant=$VARIANT =="
            \\ssh "$BOARD" "APP='$APP' BOARD_TMP='$BOARD_TMP' BIT_NAME='$BIT_NAME' bash -s" <<'REMOTE'
            \\set -euo pipefail
            \\FW="/lib/firmware/xilinx/$APP"
            \\cd "$BOARD_TMP"
            \\
            \\GENERATED_DTS="$APP.generated.dts"
            \\sed "s/@FIRMWARE_NAME@/$BIT_NAME/g" penzai-ddr-bandwidth-singleport.dts > "$GENERATED_DTS"
            \\dtc -@ -O dtb -o "$APP.dtbo" "$GENERATED_DTS"
            \\sudo mkdir -p "$FW"
            \\sudo rm -f "$FW"/*.bit.bin "$FW"/*.dtbo "$FW"/shell.json
            \\sudo cp "$APP.dtbo" "$FW/$APP.dtbo"
            \\sudo cp "$BIT_NAME" "$FW/$BIT_NAME"
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
        "zig-build",
    });
    run.addArg(variant);
    run.addArg(bit_prefix);
    return run;
}

fn addRunCommand(
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
            \\BIN="${1:?missing benchmark binary}"
            \\ARGS="${2:-run}"
            \\
            \\echo "== copy Zig benchmark -> $BOARD:$BOARD_TMP =="
            \\ssh "$BOARD" "mkdir -p '$BOARD_TMP'"
            \\scp "$BIN" "$BOARD:$BOARD_TMP/kr260-xrt-ddr-bandwidth-singleport"
            \\
            \\echo "== run kr260-xrt-ddr-bandwidth-singleport $ARGS =="
            \\ssh "$BOARD" "BOARD_TMP='$BOARD_TMP' ARGS='$ARGS' bash -s" <<'REMOTE'
            \\set -euo pipefail
            \\chmod +x "$BOARD_TMP/kr260-xrt-ddr-bandwidth-singleport"
            \\sudo "$BOARD_TMP/kr260-xrt-ddr-bandwidth-singleport" $ARGS
            \\REMOTE
        ,
        "zig-build",
    });
    run.addFileArg(exe.getEmittedBin());
    run.addArg(board_args);
    return run;
}
