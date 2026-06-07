//! Board addresses and BO layout for the Q1A8 matmul bring-up. The AXI-Lite
//! offsets must match fpga/build.tcl's address map exactly.

pub const mmio_span: usize = 0x10000;
pub const wait_limit: usize = 500_000_000;

pub const dma_w_base: i64 = 0xA000_0000; // weights MM2S + results S2MM
pub const dma_a_base: i64 = 0xA001_0000; // acts MM2S
pub const kernel_base: i64 = 0xA002_0000; // kernel AXI-Lite

// One XRT BO holds all three regions. Offsets are generous (bring-up shapes are
// tiny); keep them aligned and non-overlapping.
pub const default_bo_size: usize = 64 * 1024 * 1024;
pub const weights_offset: usize = 0;
pub const acts_offset: usize = 32 * 1024 * 1024;
pub const results_offset: usize = 48 * 1024 * 1024;
