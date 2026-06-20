//! Op-agnostic PL substrate: a /dev/mem AXI-Lite register window. The DMA driver
//! (dma.zig) and each op's kernel driver (device/pl/matmul.zig; flash later) build on
//! this. Extracted from the former mmio.zig so "PL fabric" is no longer welded to
//! matmul — flash_attn is a second tenant on the same substrate. This is the only
//! place volatile MMIO and /dev/mem effects live (plan-long §9).

const std = @import("std");

pub const Error = error{ OpenDevMem, MapFailed };

/// Bytes mapped per AXI-Lite block. All our blocks fit in one 64 KiB page span.
pub const mmio_span: usize = 0x10000;
/// Poll budget for DMA/kernel completion. Matches the bring-up's value; a single op
/// completes far under this, so hitting it means a real hang. Shared by dma.zig and
/// the tenant kernel drivers.
pub const wait_limit: usize = 500_000_000;

const O_RDWR: c_int = 0x2;
const O_SYNC: c_int = 0o4010000;
const PROT_READ: c_int = 0x1;
const PROT_WRITE: c_int = 0x2;
const MAP_SHARED: c_int = 0x1;
const MAP_FAILED = std.math.maxInt(usize);

extern "c" fn open(path: [*:0]const u8, flags: c_int, mode: c_uint) c_int;
extern "c" fn close(fd: c_int) c_int;
extern "c" fn mmap(addr: ?*anyopaque, len: usize, prot: c_int, flags: c_int, fd: c_int, offset: i64) ?*anyopaque;
extern "c" fn munmap(addr: ?*anyopaque, len: usize) c_int;

/// A mapped AXI-Lite register window. `byte_off` arguments are register byte
/// offsets (as in the regmap); indexing converts to the u32 word index.
pub const RegWindow = struct {
    fd: c_int,
    map: *anyopaque,

    pub fn mapWindow(base: i64) Error!RegWindow {
        const fd = open(@as([*:0]const u8, "/dev/mem"), O_RDWR | O_SYNC, 0);
        if (fd < 0) return error.OpenDevMem;
        errdefer _ = close(fd);
        const m = mmap(null, mmio_span, PROT_READ | PROT_WRITE, MAP_SHARED, fd, base);
        if (m == null or @intFromPtr(m.?) == MAP_FAILED) return error.MapFailed;
        return .{ .fd = fd, .map = m.? };
    }

    pub fn deinit(self: *RegWindow) void {
        _ = munmap(self.map, mmio_span);
        _ = close(self.fd);
        self.* = undefined;
    }

    inline fn regs(self: RegWindow) [*]volatile u32 {
        return @ptrCast(@alignCast(self.map));
    }

    pub inline fn rd(self: RegWindow, byte_off: u32) u32 {
        return self.regs()[byte_off / 4];
    }

    pub inline fn wr(self: RegWindow, byte_off: u32, value: u32) void {
        self.regs()[byte_off / 4] = value;
    }
};
