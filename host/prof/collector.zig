const std = @import("std");
const shared = @import("shared");
const model = @import("model.zig");
const link_mod = @import("link");

const profiling = shared.profiling;

/// Host-side profiling collector: the phase cursor, the per-phase budget
/// accumulators, and the decode throughput split. `backend.zig` feeds it through
/// the ggml callbacks (the `timedX` wrappers + graph_compute); `render.zig` reads
/// it to produce the `--prof` report. Pure data + hooks — no rendering, and no I/O
/// beyond the monotonic clock.
pub const Collector = struct {
    io: std.Io,
    phases: [model.phase_count]model.PhaseAccum =
        [_]model.PhaseAccum{.{}} ** model.phase_count,
    current: model.Phase = .model_load,
    // Decode throughput split (a subset of the decode phase wall): time to the
    // first generated token vs the steady-state tokens after it.
    first_decode_ns: u64 = 0,
    steady_decode_ns: u64 = 0,
    steady_decode_count: u64 = 0,
    generated_tokens: u64 = 0,
    // Residency diagnostic: which tensors get uploaded, in which phase.
    upload_census: model.UploadCensus = .{},

    pub fn init(io: std.Io) Collector {
        return .{ .io = io };
    }

    pub fn now(self: *const Collector) u64 {
        return profiling.nowNs(self.io);
    }

    pub fn elapsedSince(self: *const Collector, start_ns: u64) u64 {
        return profiling.elapsed(start_ns, self.now());
    }

    /// Direct subsequent link ops/run_graphs to phase `p`. Set this around each
    /// phase region so the ggml callbacks firing inside it attribute correctly.
    pub fn setPhase(self: *Collector, p: model.Phase) void {
        self.current = p;
    }

    /// Add host wall time to a phase (the timer around a phase region).
    pub fn addWall(self: *Collector, p: model.Phase, ns: u64) void {
        self.phases[@intFromEnum(p)].wall_ns += ns;
    }

    fn cur(self: *Collector) *model.PhaseAccum {
        return &self.phases[@intFromEnum(self.current)];
    }

    pub fn recordAlloc(self: *Collector, nbytes: u64, host_ns: u64, device_ns: u64) void {
        self.cur().alloc.record(nbytes, host_ns, device_ns);
    }

    pub fn recordUpload(self: *Collector, nbytes: u64, host_ns: u64, device_ns: u64) void {
        self.cur().upload.record(nbytes, host_ns, device_ns);
    }

    pub fn recordFill(self: *Collector, nbytes: u64, host_ns: u64, device_ns: u64) void {
        self.cur().fill.record(nbytes, host_ns, device_ns);
    }

    pub fn recordDownload(self: *Collector, nbytes: u64, host_ns: u64, device_ns: u64) void {
        self.cur().download.record(nbytes, host_ns, device_ns);
    }

    pub fn recordFree(self: *Collector, host_ns: u64, device_ns: u64) void {
        self.cur().free.record(0, host_ns, device_ns);
    }

    pub fn recordRunGraph(self: *Collector, profiled: link_mod.ProfiledRunGraph) void {
        self.cur().rg.record(profiled);
    }

    pub fn recordUploadTensor(self: *Collector, name: []const u8, nbytes: u64) void {
        self.upload_census.record(@intFromEnum(self.current), name, nbytes);
    }

    /// One decode token: extends the decode phase wall and the TTFT/steady split.
    pub fn recordDecodeToken(self: *Collector, is_first: bool, ns: u64) void {
        self.phases[@intFromEnum(model.Phase.decode)].wall_ns += ns;
        if (is_first) {
            self.first_decode_ns += ns;
        } else {
            self.steady_decode_ns += ns;
            self.steady_decode_count += 1;
        }
        self.generated_tokens += 1;
    }

    /// Total wall = Σ phase walls. Nothing is excluded; every phase reconciles.
    pub fn wallNs(self: *const Collector) u64 {
        var total: u64 = 0;
        for (self.phases) |phase| total += phase.wall_ns;
        return total;
    }
};
