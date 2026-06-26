//! seq_ctrl.zig - the host-side driver for seq_top's control AXI-Lite slave (the thing the PS pokes
//! ONCE per descriptor run, instead of ~34 register pokes per op). Point seq.v at a DRAM descriptor
//! buffer + entry count, strobe go, then poll STATUS.done once. MMIO only — this is the control
//! around the descriptor, not part of it. Paired with seq.zig's Recorder (builds the descriptor)
//! and matmul.recordRun (fills it for a matmul run). Used on silicon once the BD instantiates
//! seq_top (increment 3.4); the descriptor itself is host-tested (matmul recordRun, dma/kernel record).

const regwin = @import("regwin.zig");
const bus_mod = @import("bus.zig");
const seq = @import("seq.zig");

pub const Error = bus_mod.Error || error{ SeqTimeout, SeqError };

pub const SeqCtrl = struct {
    bus: bus_mod.Bus,

    pub fn open(base: i64) bus_mod.Error!SeqCtrl {
        return .{ .bus = try bus_mod.Bus.mmio(@intCast(base)) };
    }

    pub fn deinit(self: *SeqCtrl) void {
        self.bus.deinit();
    }

    /// Kick a run: point seq.v at the descriptor buffer (physical addr) + entry count, strobe go.
    /// `count` must match the Recorder's count(); descriptor bytes must be flushed to DRAM first.
    pub fn run(self: *SeqCtrl, desc_phys: u64, count: u32) void {
        self.bus.wr(seq.ctrl.DESC_BASE_LO, @truncate(desc_phys & 0xffff_ffff));
        self.bus.wr(seq.ctrl.DESC_BASE_HI, @truncate(desc_phys >> 32));
        self.bus.wr(seq.ctrl.DESC_COUNT, count);
        self.bus.wr(seq.ctrl.CTRL, seq.ctrl.CTRL_GO);
    }

    /// Poll STATUS once-per-run until done; SeqError if a WAIT in the run timed out.
    pub fn waitDone(self: *SeqCtrl) Error!void {
        var i: usize = 0;
        while (i < regwin.wait_limit) : (i += 1) {
            const s = self.bus.rd(seq.ctrl.STATUS);
            if (s & seq.ctrl.STATUS_ERR != 0) return error.SeqError;
            if (s & seq.ctrl.STATUS_DONE != 0) return;
        }
        return error.SeqTimeout;
    }

    /// The entry index a failed run's WAIT timed out on (debug).
    pub fn errIndex(self: *SeqCtrl) u32 {
        return self.bus.rd(seq.ctrl.ERR_INDEX);
    }
};
