# pblock.xdc — floorplan the two PL kernels into separate clock-region bands to de-interleave
# them (the f300 congestion fix; see docs/plan-f300-pblock.md). The xck26 grid is 3x4
# (X0-X2 columns, Y0-Y3 rows). The kernels currently overlap in row Y1, which smears the
# matmul's fanout-1503 `pc_col` net across the die (the f300 worst path). Split by rows:
# matmul -> top 2 rows, flash -> bottom 2 rows (where it already sits, by HPC0/1).
#
# Kernels only — DMAs and SmartConnects float (small, and they gravitate to the bottom HP
# ports on their own). Soft (CONTAIN_ROUTING 0) so routing may spill across the boundary
# rather than fail to route. kernel_mm / kernel_fa are OOC BD IPs, so these hierarchical
# cells survive impl. Applied impl-only by build.tcl when present (USE_PBLOCK in build.sh).

create_pblock pblock_matmul
add_cells_to_pblock pblock_matmul [get_cells design_1_i/kernel_mm]
resize_pblock pblock_matmul -add {CLOCKREGION_X0Y2:CLOCKREGION_X2Y3}

create_pblock pblock_flash
add_cells_to_pblock pblock_flash [get_cells design_1_i/kernel_fa]
resize_pblock pblock_flash -add {CLOCKREGION_X0Y0:CLOCKREGION_X2Y1}

set_property CONTAIN_ROUTING 0 [get_pblocks pblock_matmul]
set_property CONTAIN_ROUTING 0 [get_pblocks pblock_flash]
