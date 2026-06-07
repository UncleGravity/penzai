//! MMIO driver for the q1a8_kernel_top AXI-Lite register block.
//! Register offsets mirror fpga/regmap/q1a8.regmap and q1a8_kernel_top.v.

const std = @import("std");
const config = @import("config");

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

const REG_ID = 0x00 / 4;
const REG_VERSION = 0x04 / 4;
const REG_CTRL = 0x08 / 4;
const REG_STATUS = 0x0C / 4;
const REG_NUM_Q1 = 0x10 / 4;
const REG_NUM_RB = 0x14 / 4;
const REG_CYCLES = 0x18 / 4;
const REG_ROWS = 0x1C / 4;

const CTRL_START: u32 = 1 << 0;
const STATUS_BUSY: u32 = 1 << 0;
const STATUS_DONE: u32 = 1 << 1;

pub const ID_VALUE: u32 = 0xB05A_2000;
pub const VERSION_VALUE: u32 = 0x0000_0004;

pub const Kernel = struct {
    fd: c_int,
    map: *anyopaque,
    regs: [*]volatile u32,

    pub fn openDevice() !Kernel {
        const fd = open("/dev/mem", O_RDWR | O_SYNC, 0);
        if (fd < 0) return error.OpenDevMem;
        errdefer _ = close(fd);
        const m = mmap(null, config.mmio_span, PROT_READ | PROT_WRITE, MAP_SHARED, fd, config.kernel_base);
        if (m == null or @intFromPtr(m.?) == MAP_FAILED) return error.MapKernel;
        return .{ .fd = fd, .map = m.?, .regs = @ptrCast(@alignCast(m.?)) };
    }

    pub fn closeDevice(self: *Kernel) void {
        _ = munmap(self.map, config.mmio_span);
        _ = close(self.fd);
    }

    pub fn id(self: *Kernel) u32 {
        return self.regs[REG_ID];
    }
    pub fn version(self: *Kernel) u32 {
        return self.regs[REG_VERSION];
    }
    pub fn rows(self: *Kernel) u32 {
        return self.regs[REG_ROWS];
    }
    pub fn cycles(self: *Kernel) u32 {
        return self.regs[REG_CYCLES];
    }
    pub fn busy(self: *Kernel) bool {
        return self.regs[REG_STATUS] & STATUS_BUSY != 0;
    }
    pub fn done(self: *Kernel) bool {
        return self.regs[REG_STATUS] & STATUS_DONE != 0;
    }

    /// Set dims then strobe start. done_latched clears on the strobe.
    pub fn run(self: *Kernel, num_q1_blocks: u16, num_rowblocks: u16) void {
        self.regs[REG_NUM_Q1] = num_q1_blocks;
        self.regs[REG_NUM_RB] = num_rowblocks;
        self.regs[REG_CTRL] = CTRL_START;
    }

    pub fn waitDone(self: *Kernel) !void {
        var i: usize = 0;
        while (i < config.wait_limit) : (i += 1) {
            if (self.done()) return;
        }
        return error.KernelTimeout;
    }
};
