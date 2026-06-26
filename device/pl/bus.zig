//! bus.zig - the control-register backend the dispatch drivers (dma.zig, the op kernels)
//! talk to. Either the real MMIO window (regwin.RegWindow) or a record-mode Recorder that
//! appends seq.v descriptor entries instead of poking hardware. The MMIO path is byte-for-byte
//! the prior RegWindow behavior (this just wraps it), so the silicon dispatch is unchanged;
//! the record path is what increment 2 adds. `base` is the absolute control-bus byte address
//! of this target's window, so recorded entries carry sc_ctrl-space addresses (what seq.v issues).

const std = @import("std");
const regwin = @import("regwin.zig");
const seq = @import("seq.zig");

pub const Error = regwin.Error;

pub const Bus = struct {
    base: u32,
    impl: union(enum) {
        mmio: regwin.RegWindow,
        rec: *seq.Recorder,
    },

    pub fn mmio(base: u32) Error!Bus {
        return .{ .base = base, .impl = .{ .mmio = try regwin.RegWindow.mapWindow(@as(i64, base)) } };
    }
    pub fn record(base: u32, rec: *seq.Recorder) Bus {
        return .{ .base = base, .impl = .{ .rec = rec } };
    }

    pub fn deinit(self: *Bus) void {
        switch (self.impl) {
            .mmio => |*w| w.deinit(),
            .rec => {},
        }
    }

    pub fn isRecording(self: *const Bus) bool {
        return self.impl == .rec;
    }

    /// Write a register at window offset `off`. MMIO pokes; record appends a WRITE entry
    /// at the absolute address (base + off).
    pub fn wr(self: *Bus, off: u32, val: u32) void {
        switch (self.impl) {
            .mmio => |w| w.wr(off, val),
            .rec => |r| r.write(self.base + off, val),
        }
    }

    /// Read a register (MMIO only — used by the MMIO-mode poll loops). Returns 0 in record
    /// mode, where the driver emits a WAIT entry via `recordWait` instead of polling.
    pub fn rd(self: *const Bus, off: u32) u32 {
        return switch (self.impl) {
            .mmio => |w| w.rd(off),
            .rec => 0,
        };
    }

    /// Record-mode only: append a WAIT for (rd(off) & mask) == expected. Callers use this in
    /// the `.rec` arm of their wait loops (the `.mmio` arm keeps its own poll + error checks).
    pub fn recordWait(self: *Bus, off: u32, mask: u32, expected: u32) void {
        switch (self.impl) {
            .rec => |r| r.wait(self.base + off, mask, expected),
            .mmio => unreachable,
        }
    }
};
