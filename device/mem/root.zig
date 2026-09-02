pub const fake = @import("heap.zig");
pub const xrt = @import("xrt_heap.zig");
pub const xrt_api = @import("xrt_api.zig");

test {
    _ = fake;
    _ = xrt;
    _ = xrt_api;
}
