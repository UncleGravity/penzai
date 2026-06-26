//! Aggregates device/pl/ unit tests. Several pl/ files (matmul.zig) @import "../profile.zig",
//! so they can't be test roots themselves (a root's module path can't be escaped with `..`).
//! This root lives in device/, where ../profile resolves, and pulls their tests in via
//! `_ = @import` (a normal import does NOT run an imported file's tests).

test {
    _ = @import("pl/matmul.zig");
    _ = @import("pl/dma.zig");
    _ = @import("pl/seq.zig");
    _ = @import("pl/seq_ctrl.zig");
}
