# Generated from fpga/regmap/matmul.zig — do not edit.
# AXI-Lite address map for the matmul bitstream; {cell intf offset} per block.
set matmul_address_map {
    {dma_w0 S_AXI_LITE 0xA0000000}
    {dma_w1 S_AXI_LITE 0xA0010000}
    {dma_w2 S_AXI_LITE 0xA0020000}
    {dma_w3 S_AXI_LITE 0xA0030000}
    {dma_a S_AXI_LITE 0xA0040000}
    {kernel S_AXI 0xA0050000}
}
