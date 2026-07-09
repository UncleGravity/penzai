//! seq_ctrl.zig - host driver for seq_top's control slave (fpga/rtl/seq/seq_top.v). The PS
//! loads a run's entries into the command BRAM through the CMD window (posted 32-bit writes,
//! once per run — cheap; the profile showed polls, not writes, dominate dispatch), kicks it
//! with RUN_START/RUN_COUNT/GO, and polls STATUS once per run. ABORT is the escape hatch: it
//! soft-resets the executor to IDLE so a wedged run is reclaimable over SSH, never a power
//! cycle. MMIO only — the run itself is built by seq.Builder and gated by seq.validate.

const regwin = @import("regwin.zig");
const seq = @import("seq.zig");

pub const Error = regwin.Error || error{ SeqTimeout, SeqWatchdog, SeqPollTimeout, SeqOverflow };

pub const SeqCtrl = struct {
    win: regwin.RegWindow,

    pub fn open(base: i64) regwin.Error!SeqCtrl {
        return .{ .win = try regwin.RegWindow.mapWindow(base) };
    }

    pub fn deinit(self: *SeqCtrl) void {
        self.win.deinit();
    }

    /// Load entries into the command BRAM starting at entry index `at`. Returns SeqOverflow
    /// if the slice would run past the BRAM (the AXI window would alias, not wrap cleanly).
    pub fn load(self: *SeqCtrl, at: u32, items: []const seq.Entry) Error!void {
        if (at + items.len > seq.ctrl.cmd_capacity) return error.SeqOverflow;
        var off: u32 = seq.ctrl.CMD_OFF + at * @sizeOf(seq.Entry);
        for (items) |e| {
            self.win.wr(off + 0x0, e.tag);
            self.win.wr(off + 0x4, e.addr);
            self.win.wr(off + 0x8, e.a);
            self.win.wr(off + 0xC, e.b);
            off += @sizeOf(seq.Entry);
        }
    }

    /// Kick a run of `count` entries starting at entry index `start`. Bounds-checked here
    /// because the RTL's index math wraps modulo the BRAM depth rather than faulting.
    pub fn run(self: *SeqCtrl, start: u32, count: u32) Error!void {
        if (start + count > seq.ctrl.cmd_capacity) return error.SeqOverflow;
        self.win.wr(seq.ctrl.RUN_START, start);
        self.win.wr(seq.ctrl.RUN_COUNT, count);
        self.win.wr(seq.ctrl.CTRL, seq.ctrl.CTRL_GO);
    }

    /// Poll STATUS until done — the once-per-run wait that replaces the per-op register
    /// dance. Distinguishes the executor's two error exits (a WAIT that never matched vs a
    /// bus transaction that never completed) from the PS-side budget running out.
    pub fn waitDone(self: *SeqCtrl) Error!void {
        var i: usize = 0;
        while (i < regwin.wait_limit) : (i += 1) {
            const s = self.win.rd(seq.ctrl.STATUS);
            if (s & seq.ctrl.STATUS_ERR_WATCHDOG != 0) return error.SeqWatchdog;
            if (s & seq.ctrl.STATUS_ERR_TIMEOUT != 0) return error.SeqPollTimeout;
            if (s & seq.ctrl.STATUS_DONE != 0) return;
        }
        return error.SeqTimeout;
    }

    /// Force the executor back to IDLE (any state, mid-run). The recovery path after
    /// waitDone errors — always available to the PS, independent of the executor's state.
    pub fn abort(self: *SeqCtrl) void {
        self.win.wr(seq.ctrl.CTRL, seq.ctrl.CTRL_ABORT);
    }

    /// The entry index a failed run faulted on — RUN-RELATIVE (the executor counts within the
    /// run; add your RUN_START for the absolute BRAM index).
    pub fn errIndex(self: *SeqCtrl) u32 {
        return self.win.rd(seq.ctrl.ERR_INDEX);
    }

    /// Raw STATUS bits (see seq.ctrl.STATUS_*).
    pub fn status(self: *SeqCtrl) u32 {
        return self.win.rd(seq.ctrl.STATUS);
    }
};
