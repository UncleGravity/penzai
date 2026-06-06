pub const dma_base: i64 = 0xA000_0000;
pub const regs_base: i64 = 0xA001_0000;
pub const mmio_span: usize = 0x10000;

pub const data_width_bytes: usize = 16;
pub const max_dma_transfer: usize = (1 << 26) - data_width_bytes;
pub const smoke_transfer_size: usize = 4 * 1024;
pub const smoke_bo_size: usize = 8 * 1024 * 1024;
pub const default_bo_size: usize = 768 * 1024 * 1024;
pub const default_transfer_size: usize = 384 * 1024 * 1024;
pub const default_chunk_size: usize = 32 * 1024 * 1024;
pub const default_seed: u8 = 1;
pub const wait_limit: usize = 500_000_000;
