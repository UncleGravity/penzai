pub const Kind = enum {
    lint,
    simulation,
    synthesis,
    formal,
};

pub const Tier = enum {
    default,
    extended,
};

pub const SourcePolicy = enum {
    production_closure,
    production_subset,
    script_checked,
};

pub const ShellRunner = struct {
    path: []const u8,
    args: []const []const u8 = &.{},
};

pub const SbyRunner = struct {
    path: []const u8,
    tasks: []const []const u8,
    sequential: bool = false,
};

pub const Runner = union(enum) {
    shell: ShellRunner,
    sby: SbyRunner,
};

pub const Suite = struct {
    name: []const u8,
    description: []const u8,
    kind: Kind,
    tier: Tier = .default,
    source_policy: SourcePolicy,
    rtl_sources: []const []const u8 = &.{},
    runner: Runner,
};

pub const suites = [_]Suite{
    .{
        .name = "source-closure",
        .description = "Validate the closed production RTL source set",
        .kind = .lint,
        .source_policy = .production_closure,
        .runner = .{ .shell = .{ .path = "fpga/verify/check_sources.sh" } },
    },
    .{
        .name = "production-top",
        .description = "Lint the complete production RTL top",
        .kind = .lint,
        .source_policy = .production_closure,
        .runner = .{ .shell = .{ .path = "fpga/verify/lint/run.sh" } },
    },

    .{
        .name = "axi-read",
        .description = "Simulate the AXI read mover",
        .kind = .simulation,
        .source_policy = .script_checked,
        .runner = .{ .shell = .{ .path = "fpga/verify/sim/unit/io/axi_read/run.sh" } },
    },
    .{
        .name = "axi-write",
        .description = "Simulate the AXI write mover",
        .kind = .simulation,
        .source_policy = .script_checked,
        .runner = .{ .shell = .{ .path = "fpga/verify/sim/unit/io/axi_write/run.sh" } },
    },
    .{
        .name = "quad-read-arbiter",
        .description = "Simulate the four-port read arbiter",
        .kind = .simulation,
        .source_policy = .script_checked,
        .runner = .{ .shell = .{ .path = "fpga/verify/sim/unit/io/quad_read_arbiter/run.sh" } },
    },
    .{
        .name = "small-read-mux",
        .description = "Simulate the shared small-read mux",
        .kind = .simulation,
        .source_policy = .script_checked,
        .runner = .{ .shell = .{ .path = "fpga/verify/sim/unit/io/small_read_mux/run.sh" } },
    },
    .{
        .name = "rope-fetch",
        .description = "Simulate RoPE table fetch",
        .kind = .simulation,
        .source_policy = .script_checked,
        .runner = .{ .shell = .{ .path = "fpga/verify/sim/unit/io/rope_fetch/run.sh" } },
    },
    .{
        .name = "embedding-decode",
        .description = "Simulate embedding decode",
        .kind = .simulation,
        .source_policy = .script_checked,
        .runner = .{ .shell = .{ .path = "fpga/verify/sim/unit/vector/embedding_decode/run.sh" } },
    },
    .{
        .name = "embedding-service",
        .description = "Simulate the embedding service",
        .kind = .simulation,
        .source_policy = .script_checked,
        .runner = .{ .shell = .{ .path = "fpga/verify/sim/unit/vector/embedding_service/run.sh" } },
    },
    .{
        .name = "projection-engine",
        .description = "Simulate Q1/Q2 projection arithmetic and DSP mapping",
        .kind = .simulation,
        .source_policy = .script_checked,
        .runner = .{ .shell = .{ .path = "fpga/verify/sim/unit/projection/engine/run.sh" } },
    },
    .{
        .name = "projection-emit",
        .description = "Simulate projection output framing",
        .kind = .simulation,
        .source_policy = .script_checked,
        .runner = .{ .shell = .{ .path = "fpga/verify/sim/unit/projection/emit/run.sh" } },
    },
    .{
        .name = "projection-service",
        .description = "Simulate projection scheduling",
        .kind = .simulation,
        .source_policy = .script_checked,
        .runner = .{ .shell = .{ .path = "fpga/verify/sim/unit/projection/service/run.sh" } },
    },
    .{
        .name = "projection-sink",
        .description = "Simulate projection result routing",
        .kind = .simulation,
        .source_policy = .script_checked,
        .runner = .{ .shell = .{ .path = "fpga/verify/sim/unit/projection/sink/run.sh" } },
    },
    .{
        .name = "logits",
        .description = "Simulate logits reduction",
        .kind = .simulation,
        .source_policy = .script_checked,
        .runner = .{ .shell = .{ .path = "fpga/verify/sim/unit/projection/logits/run.sh" } },
    },
    .{
        .name = "shared-q8",
        .description = "Simulate shared Q8 packing",
        .kind = .simulation,
        .source_policy = .script_checked,
        .runner = .{ .shell = .{ .path = "fpga/verify/sim/unit/vector/shared_q8/run.sh" } },
    },
    .{
        .name = "rms-reduce",
        .description = "Simulate RMS reduction",
        .kind = .simulation,
        .source_policy = .script_checked,
        .runner = .{ .shell = .{ .path = "fpga/verify/sim/unit/vector/rms_reduce/run.sh" } },
    },
    .{
        .name = "vector-service",
        .description = "Simulate vector operations",
        .kind = .simulation,
        .source_policy = .script_checked,
        .runner = .{ .shell = .{ .path = "fpga/verify/sim/unit/vector/service/run.sh" } },
    },
    .{
        .name = "rope",
        .description = "Simulate RoPE arithmetic",
        .kind = .simulation,
        .source_policy = .script_checked,
        .runner = .{ .shell = .{ .path = "fpga/verify/sim/unit/vector/rope/run.sh" } },
    },
    .{
        .name = "kv-append",
        .description = "Simulate KV append",
        .kind = .simulation,
        .source_policy = .script_checked,
        .runner = .{ .shell = .{ .path = "fpga/verify/sim/unit/memory/kv_append/run.sh" } },
    },
    .{
        .name = "kv-join",
        .description = "Simulate KV joining",
        .kind = .simulation,
        .source_policy = .script_checked,
        .runner = .{ .shell = .{ .path = "fpga/verify/sim/unit/memory/kv_join/run.sh" } },
    },
    .{
        .name = "attention-query-gather",
        .description = "Simulate attention query gathering",
        .kind = .simulation,
        .source_policy = .script_checked,
        .runner = .{ .shell = .{ .path = "fpga/verify/sim/unit/attention/query_gather/run.sh" } },
    },
    .{
        .name = "attention-kernel",
        .description = "Simulate the eight-head attention kernel",
        .kind = .simulation,
        .source_policy = .script_checked,
        .runner = .{ .shell = .{ .path = "fpga/verify/sim/unit/attention/kernel/run.sh" } },
    },
    .{
        .name = "attention-groups",
        .description = "Simulate grouped attention scheduling",
        .kind = .simulation,
        .source_policy = .script_checked,
        .runner = .{ .shell = .{ .path = "fpga/verify/sim/unit/attention/groups/run.sh" } },
    },
    .{
        .name = "attention-output-q8",
        .description = "Simulate attention output quantization",
        .kind = .simulation,
        .source_policy = .script_checked,
        .runner = .{ .shell = .{ .path = "fpga/verify/sim/unit/attention/output_q8/run.sh" } },
    },
    .{
        .name = "attention-service",
        .description = "Simulate the attention service",
        .kind = .simulation,
        .source_policy = .script_checked,
        .runner = .{ .shell = .{ .path = "fpga/verify/sim/unit/attention/service/run.sh" } },
    },
    .{
        .name = "engine-control",
        .description = "Simulate controller, model_spec, arena, clear, and restart behavior",
        .kind = .simulation,
        .source_policy = .script_checked,
        .runner = .{ .shell = .{ .path = "fpga/verify/sim/integration/engine/run.sh" } },
    },
    .{
        .name = "engine-metrics",
        .description = "Simulate metrics lifecycle, accounting, saturation, and snapshots",
        .kind = .simulation,
        .source_policy = .script_checked,
        .runner = .{ .shell = .{ .path = "fpga/verify/sim/unit/engine/metrics/run.sh" } },
    },
    .{
        .name = "datapath",
        .description = "Simulate the complete production datapath",
        .kind = .simulation,
        .source_policy = .production_closure,
        .runner = .{ .shell = .{ .path = "fpga/verify/sim/integration/datapath/run.sh" } },
    },
    .{
        .name = "register-interface",
        .description = "Simulate the production AXI-Lite wrapper",
        .kind = .simulation,
        .source_policy = .script_checked,
        .runner = .{ .shell = .{ .path = "fpga/verify/sim/integration/top/run.sh" } },
    },

    .{
        .name = "production-map",
        .description = "Map the production top and enforce structural resource invariants",
        .kind = .synthesis,
        .source_policy = .production_closure,
        .runner = .{ .shell = .{ .path = "fpga/verify/qor/yosys/production/run.sh" } },
    },
    .{
        .name = "projection-map",
        .description = "Enforce projection DSP resource invariants",
        .kind = .synthesis,
        .tier = .extended,
        .source_policy = .script_checked,
        .runner = .{ .shell = .{ .path = "fpga/verify/qor/yosys/projection/run.sh" } },
    },
    .{
        .name = "attention-service-map",
        .description = "Map the attention service",
        .kind = .synthesis,
        .tier = .extended,
        .source_policy = .script_checked,
        .runner = .{ .shell = .{ .path = "fpga/verify/qor/yosys/attention_service/run.sh" } },
    },
    .{
        .name = "vector-cluster-map",
        .description = "Map the vector and projection sink cluster",
        .kind = .synthesis,
        .tier = .extended,
        .source_policy = .script_checked,
        .runner = .{ .shell = .{ .path = "fpga/verify/qor/yosys/vector_cluster/run.sh" } },
    },

    .{
        .name = "formal-attention-kernel",
        .description = "Prove bounded attention control, slot mapping, and pipeline behavior",
        .kind = .formal,
        .source_policy = .production_subset,
        .rtl_sources = &.{
            "fpga/rtl/attention/flash_kernel.v",
            "fpga/rtl/lib/fmt.vh",
        },
        .runner = .{ .sby = .{
            .path = "fpga/verify/formal/attention/flash_kernel/flash_kernel.sby",
            .tasks = &.{ "bmc", "slots_prove", "pipeline_prove" },
        } },
    },
    .{
        .name = "formal-attention-kernel-deep",
        .description = "Run the deep attention proofs, liveness checks, and covers",
        .kind = .formal,
        .tier = .extended,
        .source_policy = .production_subset,
        .rtl_sources = &.{
            "fpga/rtl/attention/flash_kernel.v",
            "fpga/rtl/lib/fmt.vh",
        },
        .runner = .{ .sby = .{
            .path = "fpga/verify/formal/attention/flash_kernel/flash_kernel.sby",
            .tasks = &.{ "prove", "completion", "liveness", "cover", "slots_bmc", "slots_cover" },
            .sequential = true,
        } },
    },
    .{
        .name = "formal-ternary-select",
        .description = "Prove ternary selector arithmetic",
        .kind = .formal,
        .source_policy = .production_subset,
        .rtl_sources = &.{"fpga/rtl/projection/ternary_select.v"},
        .runner = .{ .sby = .{
            .path = "fpga/verify/formal/math/ternary_select/ternary_select.sby",
            .tasks = &.{"prove"},
        } },
    },
    .{
        .name = "formal-engine-metrics",
        .description = "Prove metrics lifecycle, reconciliation, saturation, and snapshot stability",
        .kind = .formal,
        .source_policy = .production_subset,
        .rtl_sources = &.{"fpga/rtl/engine/metrics.v"},
        .runner = .{ .sby = .{
            .path = "fpga/verify/formal/engine/metrics/metrics.sby",
            .tasks = &.{"prove"},
        } },
    },
    .{
        .name = "formal-engine-metrics-cover",
        .description = "Cover metrics commit, cancel, and saturation behavior",
        .kind = .formal,
        .tier = .extended,
        .source_policy = .production_subset,
        .rtl_sources = &.{"fpga/rtl/engine/metrics.v"},
        .runner = .{ .sby = .{
            .path = "fpga/verify/formal/engine/metrics/metrics.sby",
            .tasks = &.{"cover"},
        } },
    },
    .{
        .name = "formal-ternary-select-cover",
        .description = "Cover ternary selector behavior",
        .kind = .formal,
        .tier = .extended,
        .source_policy = .production_subset,
        .rtl_sources = &.{"fpga/rtl/projection/ternary_select.v"},
        .runner = .{ .sby = .{
            .path = "fpga/verify/formal/math/ternary_select/ternary_select.sby",
            .tasks = &.{"cover"},
        } },
    },
    .{
        .name = "formal-rms-inverse",
        .description = "Prove RMS inverse ordering, faults, abort, and restart",
        .kind = .formal,
        .source_policy = .production_subset,
        .rtl_sources = &.{"fpga/rtl/vector/rms_inverse.v"},
        .runner = .{ .sby = .{
            .path = "fpga/verify/formal/math/rms_inverse/rms_inverse.sby",
            .tasks = &.{ "bmc", "faults" },
        } },
    },
    .{
        .name = "formal-rms-inverse-cover",
        .description = "Cover RMS inverse behavior",
        .kind = .formal,
        .tier = .extended,
        .source_policy = .production_subset,
        .rtl_sources = &.{"fpga/rtl/vector/rms_inverse.v"},
        .runner = .{ .sby = .{
            .path = "fpga/verify/formal/math/rms_inverse/rms_inverse.sby",
            .tasks = &.{"cover"},
        } },
    },
    .{
        .name = "formal-swiglu",
        .description = "Prove SwiGLU stream control and backpressure",
        .kind = .formal,
        .source_policy = .production_subset,
        .rtl_sources = &.{
            "fpga/rtl/vector/swiglu.v",
            "fpga/rtl/lib/fmt.vh",
        },
        .runner = .{ .sby = .{
            .path = "fpga/verify/formal/math/swiglu/swiglu.sby",
            .tasks = &.{ "prove", "bmc" },
        } },
    },
    .{
        .name = "formal-swiglu-cover",
        .description = "Cover SwiGLU stream behavior",
        .kind = .formal,
        .tier = .extended,
        .source_policy = .production_subset,
        .rtl_sources = &.{
            "fpga/rtl/vector/swiglu.v",
            "fpga/rtl/lib/fmt.vh",
        },
        .runner = .{ .sby = .{
            .path = "fpga/verify/formal/math/swiglu/swiglu.sby",
            .tasks = &.{"cover"},
        } },
    },
};
