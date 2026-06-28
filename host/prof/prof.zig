//! Profiling — the host-side report/scoreboard, as one module rather than three
//! loose files. Promoted to a module so the backend core and the llama driver
//! share the *same* `Collector` type across the module boundary: `Device.profile`
//! (backend module) and the `Collector` llama.zig constructs must match, which
//! only holds if both import one `prof`. Re-exports the three internal files.
pub const model = @import("model.zig");
pub const collector = @import("collector.zig");
pub const render = @import("render.zig");

pub const Collector = collector.Collector;
