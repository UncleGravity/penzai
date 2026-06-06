pub const mmio_span: usize = 0x10000;
pub const lane_count: usize = 4;

pub const data_width_bytes: usize = 16;
pub const max_dma_transfer: usize = (1 << 26) - data_width_bytes;
pub const smoke_transfer_size: usize = 4 * 1024;
pub const smoke_bo_size: usize = 8 * 1024 * 1024;
pub const default_bo_size: usize = 768 * 1024 * 1024;
pub const default_transfer_size: usize = 192 * 1024 * 1024;
pub const default_chunk_size: usize = 32 * 1024 * 1024;
pub const wait_limit: usize = 500_000_000;

pub const LaneConfig = struct {
    name: []const u8,
    dma_base: i64,
    regs_base: i64,
    bo_offset: usize,
    seed: u8,
};

pub const lanes = [_]LaneConfig{
    .{
        .name = "hp0",
        .dma_base = 0xA000_0000,
        .regs_base = 0xA001_0000,
        .bo_offset = 0 * default_transfer_size,
        .seed = 1,
    },
    .{
        .name = "hp1",
        .dma_base = 0xA002_0000,
        .regs_base = 0xA003_0000,
        .bo_offset = 1 * default_transfer_size,
        .seed = 17,
    },
    .{
        .name = "hp2",
        .dma_base = 0xA004_0000,
        .regs_base = 0xA005_0000,
        .bo_offset = 2 * default_transfer_size,
        .seed = 33,
    },
    .{
        .name = "hp3",
        .dma_base = 0xA006_0000,
        .regs_base = 0xA007_0000,
        .bo_offset = 3 * default_transfer_size,
        .seed = 49,
    },
};
