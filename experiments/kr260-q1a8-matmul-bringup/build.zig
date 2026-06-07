const std = @import("std");

const app_name = "penzai-q1a8-matmul";
const bit_prefix = "penzai-q1a8-matmul";
const board_bin = "kr260-q1a8-board";

const config_prelude =
    \\set -euo pipefail
    \\[[ -f config.env ]] || { echo "ERROR: missing config.env; copy config.env.example first" >&2; exit 1; }
    \\set -a
    \\source config.env
    \\set +a
;

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const variant = b.option([]const u8, "variant", "Bitstream variant w64-f<MHz> (e.g. w64-f100; ~137 MHz fmax unpipelined)") orelse "w64-f100";
    const board_args = b.option([]const u8, "board-args", "Args passed to the board binary by zig build run") orelse "run";

    // ---- Host self-test (native) ----
    const exe_mod = b.createModule(.{ .root_source_file = b.path("src/main.zig"), .target = target, .optimize = optimize });
    attachSrc(b, exe_mod, target, optimize);
    const exe = b.addExecutable(.{ .name = "kr260-q1a8-selftest", .root_module = exe_mod });
    b.installArtifact(exe);
    b.step("run-selftest", "Run the laptop self-test").dependOn(&b.addRunArtifact(exe).step);

    // ---- Host unit tests (M0 oracle, M1 packer) ----
    const test_step = b.step("test", "Run host unit tests");
    for ([_][]const u8{ "src/matmul_ref.zig", "src/pack.zig" }) |path| {
        const tmod = b.createModule(.{ .root_source_file = b.path(path), .target = target, .optimize = optimize });
        tmod.addImport("q1a8", b.createModule(.{ .root_source_file = b.path("src/q1a8.zig"), .target = target, .optimize = optimize }));
        test_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = tmod })).step);
    }

    addCosim(b, target, optimize); // M2: zig build test-rtl

    // ---- Board binary (M4): cross-compiled aarch64 ----
    const board_target = b.resolveTargetQuery(.{
        .cpu_arch = .aarch64,
        .os_tag = .linux,
        .abi = .gnu,
        .cpu_model = .{ .explicit = &std.Target.aarch64.cpu.cortex_a53 },
    });
    const board_mod = b.createModule(.{
        .root_source_file = b.path("src/board.zig"),
        .target = board_target,
        .optimize = optimize,
        .link_libc = true,
    });
    attachBoard(b, board_mod, board_target, optimize);
    const board_exe = b.addExecutable(.{ .name = board_bin, .root_module = board_mod });
    b.installArtifact(board_exe);
    b.step("board", "Cross-compile the board binary (aarch64)").dependOn(&board_exe.step);

    // ---- Hardware flow (M3/M4): bitstream -> deploy -> run on the board ----
    b.step("bitstream", "Build the bitstream on the Vivado VM").dependOn(&bitstreamCmd(b, variant).step);
    b.step("deploy", "Install and load the XRT app on the KR260").dependOn(&deployCmd(b, variant).step);
    b.step("run", "Copy and run the board binary on the KR260").dependOn(&runCmd(b, board_exe, board_args).step);

    const all_bit = bitstreamCmd(b, variant);
    const all_dep = deployCmd(b, variant);
    all_dep.step.dependOn(&all_bit.step);
    const all_run = runCmd(b, board_exe, board_args);
    all_run.step.dependOn(&all_dep.step);
    b.step("all", "bitstream + deploy + run").dependOn(&all_run.step);
}

/// q1a8 <- pack, matmul_ref.
fn attachSrc(b: *std.Build, mod: *std.Build.Module, t: std.Build.ResolvedTarget, o: std.builtin.OptimizeMode) void {
    const q1a8 = b.createModule(.{ .root_source_file = b.path("src/q1a8.zig"), .target = t, .optimize = o });
    const pack = b.createModule(.{ .root_source_file = b.path("src/pack.zig"), .target = t, .optimize = o });
    const ref = b.createModule(.{ .root_source_file = b.path("src/matmul_ref.zig"), .target = t, .optimize = o });
    pack.addImport("q1a8", q1a8);
    ref.addImport("q1a8", q1a8);
    mod.addImport("q1a8", q1a8);
    mod.addImport("pack", pack);
    mod.addImport("matmul_ref", ref);
}

/// Board module graph: config, xrt, dma, mmio + the src oracle/packer.
fn attachBoard(b: *std.Build, mod: *std.Build.Module, t: std.Build.ResolvedTarget, o: std.builtin.OptimizeMode) void {
    const config = b.createModule(.{ .root_source_file = b.path("src/config.zig"), .target = t, .optimize = o });
    const xrt = b.createModule(.{ .root_source_file = b.path("src/xrt.zig"), .target = t, .optimize = o });
    const dma = b.createModule(.{ .root_source_file = b.path("src/dma.zig"), .target = t, .optimize = o });
    const mmio = b.createModule(.{ .root_source_file = b.path("src/mmio.zig"), .target = t, .optimize = o });
    dma.addImport("config", config);
    mmio.addImport("config", config);
    mod.addImport("config", config);
    mod.addImport("xrt", xrt);
    mod.addImport("dma", dma);
    mod.addImport("mmio", mmio);
    attachSrc(b, mod, t, o);
}

const datapath_rtl = [_][]const u8{
    "fpga/rtl/q1a8/q1a8_kernel.v",     "fpga/rtl/q1a8/q1a8_rowblock.v",
    "fpga/rtl/q1a8/q1a8_reducer.v",    "fpga/rtl/q1a8/fp32_add.v",
    "fpga/rtl/q1a8/fp32_mul.v",        "fpga/rtl/q1a8/fp16_to_fp32.v",
    "fpga/rtl/q1a8/int_to_fp32.v",
};

// M2: Verilator cosim of q1a8_kernel, driven from Zig, checked vs matmul_ref.
fn addCosim(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) void {
    const step = b.step("test-rtl", "M2: Verilator cosim of q1a8_kernel vs matmul_ref");
    const verilator = b.findProgram(&.{"verilator"}, &.{}) catch {
        const msg = b.addSystemCommand(&.{ "sh", "-c", "echo 'verilator not found — run inside: nix develop' >&2; exit 1" });
        step.dependOn(&msg.step);
        return;
    };
    const vroot = std.mem.trim(u8, b.run(&.{ verilator, "--getenv", "VERILATOR_ROOT" }), " \n\r");
    const gen = "fpga/sim/q1a8_kernel/obj_dir";

    const vcmd = b.addSystemCommand(&.{
        verilator,      "--cc",             "--build",          "--lib-create", "vq1a8",
        "-Wno-fatal",   "-Wno-WIDTHEXPAND", "-Wno-UNUSEDSIGNAL",
        "--top-module", "q1a8_kernel",      "--Mdir",           gen,
    });
    vcmd.addArgs(&datapath_rtl);

    const tb_mod = b.createModule(.{
        .root_source_file = b.path("fpga/sim/q1a8_kernel/tb.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    attachSrc(b, tb_mod, target, optimize);
    tb_mod.addIncludePath(b.path("fpga/sim/q1a8_kernel"));
    tb_mod.addIncludePath(.{ .cwd_relative = b.fmt("{s}/include", .{vroot}) });
    tb_mod.addIncludePath(.{ .cwd_relative = b.fmt("{s}/include/vltstd", .{vroot}) });
    tb_mod.addIncludePath(b.path(gen));
    tb_mod.addCSourceFile(.{
        .file = b.path("fpga/sim/q1a8_kernel/shim.cpp"),
        .flags = &.{ "-std=c++17", b.fmt("-I{s}", .{gen}), b.fmt("-I{s}/include", .{vroot}), b.fmt("-I{s}/include/vltstd", .{vroot}) },
    });
    tb_mod.addObjectFile(b.path(b.fmt("{s}/libvq1a8.a", .{gen})));
    tb_mod.linkSystemLibrary("pthread", .{});
    tb_mod.link_libcpp = true;

    const tb = b.addExecutable(.{ .name = "q1a8-cosim", .root_module = tb_mod });
    tb.step.dependOn(&vcmd.step);
    step.dependOn(&b.addRunArtifact(tb).step);
}

fn bitstreamCmd(b: *std.Build, variant: []const u8) *std.Build.Step.Run {
    const run = b.addSystemCommand(&.{ "bash", "-lc", config_prelude ++
        \\
        \\: "${VM:?config.env must set VM}"
        \\: "${VM_DIR:?config.env must set VM_DIR}"
        \\VARIANT="${1:?missing variant}"
        \\BIT_PREFIX="${2:?missing bit prefix}"
        \\echo "== sync FPGA inputs -> $VM:$VM_DIR =="
        \\ssh "$VM" "if not exist $VM_DIR mkdir $VM_DIR"
        \\ssh "$VM" "if not exist $VM_DIR\\rtl mkdir $VM_DIR\\rtl"
        \\scp fpga/build.tcl fpga/build.bat "$VM:$VM_DIR/"
        \\scp fpga/rtl/q1a8/*.v "$VM:$VM_DIR/rtl/"
        \\echo "== Vivado build variant=$VARIANT on $VM =="
        \\ssh "$VM" "cd $VM_DIR && build.bat $VARIANT"
        \\echo "== fetch outputs -> fpga/out/ =="
        \\mkdir -p fpga/out
        \\scp "$VM:$VM_DIR/out/$BIT_PREFIX-$VARIANT.bit.bin" "$VM:$VM_DIR/out/$BIT_PREFIX-$VARIANT.bit" fpga/out/
        \\ls -la "fpga/out/$BIT_PREFIX-$VARIANT.bit.bin"
    , "zig-build" });
    run.addArg(variant);
    run.addArg(bit_prefix);
    return run;
}

fn deployCmd(b: *std.Build, variant: []const u8) *std.Build.Step.Run {
    const run = b.addSystemCommand(&.{ "bash", "-lc", config_prelude ++
        \\
        \\: "${BOARD:?config.env must set BOARD}"
        \\: "${BOARD_TMP:?config.env must set BOARD_TMP}"
        \\APP="${APP:-penzai-q1a8-matmul}"
        \\VARIANT="${1:?missing variant}"
        \\BIT_PREFIX="${2:?missing bit prefix}"
        \\case "$BOARD_TMP" in /tmp/*) ;; *) echo "ERROR: BOARD_TMP must be under /tmp" >&2; exit 1 ;; esac
        \\BIT_NAME="$BIT_PREFIX-$VARIANT.bit.bin"
        \\BIT="fpga/out/$BIT_NAME"
        \\DTS_TEMPLATE="overlay/penzai-q1a8-matmul.dts"
        \\[[ -f "$BIT" ]] || { echo "ERROR: missing $BIT. Run 'zig build bitstream -Dvariant=$VARIANT' first." >&2; exit 1; }
        \\[[ -f "$DTS_TEMPLATE" ]] || { echo "ERROR: missing $DTS_TEMPLATE." >&2; exit 1; }
        \\echo "== copy app inputs -> $BOARD:$BOARD_TMP =="
        \\ssh "$BOARD" "rm -rf '$BOARD_TMP' && mkdir -p '$BOARD_TMP'"
        \\scp "$BIT" "$DTS_TEMPLATE" "$BOARD:$BOARD_TMP/"
        \\echo "== package + load app variant=$VARIANT =="
        \\ssh "$BOARD" "APP='$APP' BOARD_TMP='$BOARD_TMP' BIT_NAME='$BIT_NAME' bash -s" <<'REMOTE'
        \\set -euo pipefail
        \\FW="/lib/firmware/xilinx/$APP"
        \\cd "$BOARD_TMP"
        \\GENERATED_DTS="$APP.generated.dts"
        \\sed "s/@FIRMWARE_NAME@/$BIT_NAME/g" penzai-q1a8-matmul.dts > "$GENERATED_DTS"
        \\dtc -@ -O dtb -o "$APP.dtbo" "$GENERATED_DTS"
        \\sudo mkdir -p "$FW"
        \\sudo rm -f "$FW"/*.bit.bin "$FW"/*.dtbo "$FW"/shell.json
        \\sudo cp "$APP.dtbo" "$FW/$APP.dtbo"
        \\sudo cp "$BIT_NAME" "$FW/$BIT_NAME"
        \\printf '{"shell_type":"XRT_FLAT","num_slots":"1"}\n' | sudo tee "$FW/shell.json" >/dev/null
        \\sudo xmutil unloadapp 2>/dev/null || true
        \\sudo xmutil loadapp "$APP"
        \\sudo xmutil listapps
        \\REMOTE
    , "zig-build" });
    run.addArg(variant);
    run.addArg(bit_prefix);
    return run;
}

fn runCmd(b: *std.Build, exe: *std.Build.Step.Compile, board_args: []const u8) *std.Build.Step.Run {
    const run = b.addSystemCommand(&.{ "bash", "-lc", config_prelude ++
        \\
        \\: "${BOARD:?config.env must set BOARD}"
        \\: "${BOARD_TMP:?config.env must set BOARD_TMP}"
        \\BIN="${1:?missing board binary}"
        \\ARGS="${2:-run}"
        \\echo "== copy board binary -> $BOARD:$BOARD_TMP =="
        \\ssh "$BOARD" "mkdir -p '$BOARD_TMP'"
        \\scp "$BIN" "$BOARD:$BOARD_TMP/kr260-q1a8-board"
        \\echo "== run kr260-q1a8-board $ARGS =="
        \\ssh "$BOARD" "BOARD_TMP='$BOARD_TMP' bash -s" <<'REMOTE'
        \\set -euo pipefail
        \\chmod +x "$BOARD_TMP/kr260-q1a8-board"
        \\sudo "$BOARD_TMP/kr260-q1a8-board"
        \\REMOTE
    , "zig-build" });
    run.addFileArg(exe.getEmittedBin());
    run.addArg(board_args);
    return run;
}
