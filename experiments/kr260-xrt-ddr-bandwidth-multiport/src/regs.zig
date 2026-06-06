//! MMIO driver for the custom generator/checker register block.

const std = @import("std");
const config = @import("config.zig");

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

const REG_CONTROL = 0x00 / 4;
const REG_STATUS = 0x04 / 4;
const REG_LENGTH_LO = 0x08 / 4;
const REG_LENGTH_HI = 0x0C / 4;
const REG_SEED = 0x10 / 4;
const REG_GEN_CYC_LO = 0x14 / 4;
const REG_GEN_CYC_HI = 0x18 / 4;
const REG_CHK_CYC_LO = 0x1C / 4;
const REG_CHK_CYC_HI = 0x20 / 4;
const REG_BYTES_LO = 0x24 / 4;
const REG_BYTES_HI = 0x28 / 4;
const REG_FIRST_ERR_LO = 0x2C / 4;
const REG_FIRST_ERR_HI = 0x30 / 4;
const REG_EXPECTED = 0x34 / 4;
const REG_ACTUAL = 0x38 / 4;
const REG_BASE_LO = 0x3C / 4;

const CONTROL_START_WRITE: u32 = 1 << 0;
const CONTROL_START_READ: u32 = 1 << 1;

const STATUS_GEN_BUSY: u32 = 1 << 0;
const STATUS_GEN_DONE: u32 = 1 << 1;
const STATUS_CHECK_BUSY: u32 = 1 << 2;
const STATUS_CHECK_DONE: u32 = 1 << 3;
const STATUS_CHECK_ERROR: u32 = 1 << 4;

const CONFIG_ATTEMPTS: usize = 64;

pub const Status = struct {
    raw: u32,
    gen_busy: bool,
    gen_done: bool,
    check_busy: bool,
    check_done: bool,
    check_error: bool,
};

pub const Snapshot = struct {
    status: Status,
    gen_cycles: u64,
    check_cycles: u64,
    bytes_checked: u64,
    first_error_index: u64,
    expected: u8,
    actual: u8,
};

pub const ConfigSnapshot = struct {
    length_lo: u32,
    length_hi: u32,
    base_lo: u32,
    seed: u32,
};

pub const Engine = struct {
    fd: c_int,
    map: *anyopaque,
    regs: [*]volatile u32,
    base: i64,

    pub fn openDevice(base: i64) !Engine {
        const fd = open("/dev/mem", O_RDWR | O_SYNC, 0);
        if (fd < 0) return error.OpenDevMem;
        errdefer _ = close(fd);

        const maybe_map = mmap(null, config.mmio_span, PROT_READ | PROT_WRITE, MAP_SHARED, fd, base);
        if (maybe_map == null or @intFromPtr(maybe_map.?) == MAP_FAILED) return error.MapRegs;

        const map = maybe_map.?;
        return .{
            .fd = fd,
            .map = map,
            .regs = @ptrCast(@alignCast(map)),
            .base = base,
        };
    }

    pub fn closeDevice(self: *Engine) void {
        _ = munmap(self.map, config.mmio_span);
        _ = close(self.fd);
    }

    pub fn startWrite(self: *Engine, length: usize, base_index: usize, seed: u8) !void {
        try self.configure(length, base_index, seed);
        self.regs[REG_CONTROL] = CONTROL_START_WRITE;
    }

    pub fn startRead(self: *Engine, length: usize, base_index: usize, seed: u8) !void {
        try self.configure(length, base_index, seed);
        self.regs[REG_CONTROL] = CONTROL_START_READ;
    }

    pub fn waitWriteDone(self: *Engine) !void {
        var i: usize = 0;
        while (i < config.wait_limit) : (i += 1) {
            const st = self.status();
            if (st.gen_done and !st.gen_busy) return;
        }
        const snap = self.snapshot();
        std.debug.print("case=engine_wait test=write ok=0 status=0x{x:0>8} gen_cycles={d}\n", .{
            snap.status.raw,
            snap.gen_cycles,
        });
        return error.EngineTimeout;
    }

    pub fn waitReadDone(self: *Engine) !void {
        var i: usize = 0;
        while (i < config.wait_limit) : (i += 1) {
            const st = self.status();
            if (st.check_done and !st.check_busy) {
                if (st.check_error) return error.CheckerMismatch;
                return;
            }
        }
        const snap = self.snapshot();
        std.debug.print("case=engine_wait test=read ok=0 status=0x{x:0>8} check_cycles={d} bytes_checked={d}\n", .{
            snap.status.raw,
            snap.check_cycles,
            snap.bytes_checked,
        });
        return error.EngineTimeout;
    }

    pub fn status(self: *Engine) Status {
        const raw = self.regs[REG_STATUS];
        return .{
            .raw = raw,
            .gen_busy = raw & STATUS_GEN_BUSY != 0,
            .gen_done = raw & STATUS_GEN_DONE != 0,
            .check_busy = raw & STATUS_CHECK_BUSY != 0,
            .check_done = raw & STATUS_CHECK_DONE != 0,
            .check_error = raw & STATUS_CHECK_ERROR != 0,
        };
    }

    pub fn snapshot(self: *Engine) Snapshot {
        return .{
            .status = self.status(),
            .gen_cycles = self.read64(REG_GEN_CYC_LO, REG_GEN_CYC_HI),
            .check_cycles = self.read64(REG_CHK_CYC_LO, REG_CHK_CYC_HI),
            .bytes_checked = self.read64(REG_BYTES_LO, REG_BYTES_HI),
            .first_error_index = self.read64(REG_FIRST_ERR_LO, REG_FIRST_ERR_HI),
            .expected = @truncate(self.regs[REG_EXPECTED]),
            .actual = @truncate(self.regs[REG_ACTUAL]),
        };
    }

    pub fn configSnapshot(self: *Engine) ConfigSnapshot {
        return self.readConfigSnapshot();
    }

    fn configure(self: *Engine, length: usize, base_index: usize, seed: u8) !void {
        if (length == 0) return error.InvalidTransfer;
        if (length % config.data_width_bytes != 0) return error.UnalignedTransfer;
        if (base_index > std.math.maxInt(u32)) return error.BaseIndexTooLarge;

        const len64: u64 = @intCast(length);
        // The RTL pattern is periodic over 256 bytes; only base_index[7:0]
        // affects generated/checked data.
        const expected_base_low: u32 = @intCast(base_index & 0xff);
        const expected: ConfigSnapshot = .{
            .length_lo = @truncate(len64),
            .length_hi = @truncate(len64 >> 32),
            .base_lo = @intCast(base_index),
            .seed = seed,
        };

        var actual: ConfigSnapshot = .{
            .length_lo = 0,
            .length_hi = 0,
            .base_lo = 0,
            .seed = 0,
        };

        var attempt: usize = 0;
        while (attempt < CONFIG_ATTEMPTS) : (attempt += 1) {
            self.regs[REG_LENGTH_LO] = expected.length_lo;
            self.regs[REG_LENGTH_HI] = expected.length_hi;
            self.regs[REG_BASE_LO] = expected.base_lo;
            self.regs[REG_SEED] = expected.seed;

            actual = self.readConfigSnapshot();
            if (actual.length_lo == expected.length_lo and
                actual.length_hi == expected.length_hi and
                (actual.base_lo & 0xff) == expected_base_low and
                actual.seed == expected.seed)
            {
                return;
            }
        }

        std.debug.print(
            "case=engine_config ok=0 base=0x{x} expected_length_lo=0x{x:0>8} actual_length_lo=0x{x:0>8} expected_length_hi=0x{x:0>8} actual_length_hi=0x{x:0>8} expected_base_lo=0x{x:0>8} actual_base_lo=0x{x:0>8} expected_seed=0x{x:0>2} actual_seed=0x{x:0>2}\n",
            .{
                @as(u64, @intCast(self.base)),
                expected.length_lo,
                actual.length_lo,
                expected.length_hi,
                actual.length_hi,
                expected.base_lo,
                actual.base_lo,
                expected.seed,
                actual.seed,
            },
        );
        return error.EngineConfigMismatch;
    }

    fn readConfigSnapshot(self: *Engine) ConfigSnapshot {
        return .{
            .length_lo = self.regs[REG_LENGTH_LO],
            .length_hi = self.regs[REG_LENGTH_HI],
            .base_lo = self.regs[REG_BASE_LO],
            .seed = self.regs[REG_SEED] & 0xff,
        };
    }

    fn read64(self: *Engine, lo: usize, hi: usize) u64 {
        const low: u64 = self.regs[lo];
        const high: u64 = self.regs[hi];
        return low | (high << 32);
    }
};
