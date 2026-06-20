# Generated from fpga/regmap/flash_attn.zig — do not edit.
# AXI-Lite address map for the flash bitstream; {cell intf offset} per block.
set flash_address_map {
    {dma_q S_AXI_LITE 0xA0100000}
    {dma_k S_AXI_LITE 0xA0110000}
    {dma_v S_AXI_LITE 0xA0120000}
    {dma_mask S_AXI_LITE 0xA0130000}
    {dma_o S_AXI_LITE 0xA0140000}
    {kernel S_AXI 0xA0150000}
}
