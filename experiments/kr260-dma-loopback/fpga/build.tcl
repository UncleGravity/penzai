# build.tcl - KR260 minimal AXI-DMA loopback overlay (no custom RTL)
#
#   vivado -mode batch -source build.tcl
#
# Design:
#   zynq_ultra_ps_e -> axi_dma.S_AXI_LITE       (control, MMIO poke)
#   axi_dma.M_AXIS_MM2S -> axis_data_fifo -> axi_dma.S_AXIS_S2MM
#   axi_dma.{M_AXI_MM2S,M_AXI_S2MM} -> ps.S_AXI_HP0_FPD       (DDR via HP port)
#
# Validates the whole KR260 path: ZynqMP bitstream build, PL bring-up via
# dtbo, an XRT device appearing, XRT BO memory, AXI-DMA MMIO control, and
# src==dst data integrity. Pure Xilinx IP - nothing to stage but this file.
#
# CONFIRM THESE before a build run (they are the usual first-try failures):
#   * board_part string  -> in Vivado tcl console: `get_board_parts *kr260*`
#   * KR260 board files installed (Tools > Xilinx > Store, or already present)
#   * Vivado version matches the board image (2024.1) for the dtbo branch later

set proj     loopback
set bd       design_1
set part     xck26-sfvc784-2LV-c
set board    xilinx.com:kr260_som:part0:1.1   ;# confirmed on VM (Vivado 2025.2)
set fclk_mhz 100
set outdir   [file normalize ./out]

file delete -force $outdir [file normalize ./$proj]
file mkdir $outdir

create_project $proj [file normalize ./$proj] -part $part -force
# board_part drives the K26 SOM PS preset (DDR/MIO). Soft-fail if the board
# files are missing so the error is explicit rather than a cryptic preset gap.
if {[catch {set_property board_part $board [current_project]} err]} {
    puts "WARNING: could not set board_part '$board': $err"
    puts "         install KR260 board files or fix the string (get_board_parts *kr260*)"
}
set_property target_language Verilog [current_project]

create_bd_design $bd

# ---- Processing system (K26 SOM preset) -----------------------------------
create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e:* ps
apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e \
    -config {apply_board_preset 1} [get_bd_cells ps]

# Enable exactly the PL-facing ports we need on top of the SOM preset:
#   M_AXI_HPM0_FPD (GP0)  -> AXI-Lite control to the DMA
#   S_AXI_HP0_FPD  (GP2)  -> DMA mastering into DDR
#   PL clock 0 @ fclk_mhz -> the single PL clock domain
#   TTC0 waveout[2]        -> fan_en_b (KR260 fan gate, Linux pwm-fan)
set_property -dict [list \
    CONFIG.PSU__USE__M_AXI_GP0 {1} \
    CONFIG.PSU__USE__M_AXI_GP1 {0} \
    CONFIG.PSU__USE__S_AXI_GP2 {1} \
    CONFIG.PSU__FPGA_PL0_ENABLE {1} \
    CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ $fclk_mhz \
    CONFIG.PSU__TTC0__PERIPHERAL__ENABLE {1} \
    CONFIG.PSU__TTC0__WAVEOUT__ENABLE {1} \
    CONFIG.PSU__TTC0__WAVEOUT__IO {EMIO} \
] [get_bd_cells ps]

# KR260's Linux base DT controls /pwm-fan with TTC0 channel 2:
#   pwms = <&ttc0 2 40000 PWM_POLARITY_INVERTED>
# Leave PSU__TTC0__CLOCK__ENABLE off: that is the optional external TTC clock
# input and it defaults to MIO 6, conflicting with the KR260 SPI1 preset.
# The fan gate is a PL HDIO pin, so a full replacement bitstream must route
# that TTC wave output to the board pin or the fan falls back to full speed.
create_bd_cell -type ip -vlnv xilinx.com:ip:xlslice:* fan_ttc0_ch2
set_property -dict [list \
    CONFIG.DIN_WIDTH {3} \
    CONFIG.DIN_FROM {2} \
    CONFIG.DIN_TO {2} \
] [get_bd_cells fan_ttc0_ch2]
create_bd_port -dir O fan_en_b
connect_bd_net [get_bd_pins ps/emio_ttc0_wave_o] [get_bd_pins fan_ttc0_ch2/Din]
connect_bd_net [get_bd_pins fan_ttc0_ch2/Dout]   [get_bd_ports fan_en_b]

# ---- AXI DMA (direct register mode, MM2S + S2MM) --------------------------
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:* dma
# c_addr_width 40 so the DMA can reach all of the KR260's 4 GB DDR (incl. the
# high bank > 2 GB); 32-bit would exclude HP0_DDR_HIGH and could leave XRT
# buffers unreachable. Adds the SA/DA MSB registers (0x1C/0x4C).
set_property -dict [list \
    CONFIG.c_include_sg {0} \
    CONFIG.c_include_mm2s {1} \
    CONFIG.c_include_s2mm {1} \
    CONFIG.c_sg_length_width {26} \
    CONFIG.c_addr_width {40} \
] [get_bd_cells dma]

# ---- AXIS loopback: MM2S -> fifo -> S2MM ----------------------------------
create_bd_cell -type ip -vlnv xilinx.com:ip:axis_data_fifo:* fifo
connect_bd_intf_net [get_bd_intf_pins dma/M_AXIS_MM2S] [get_bd_intf_pins fifo/S_AXIS]
connect_bd_intf_net [get_bd_intf_pins fifo/M_AXIS]     [get_bd_intf_pins dma/S_AXIS_S2MM]

# ---- AXI plumbing, wired by hand ------------------------------------------
# 2025.2's axi4 connection-automation rule fails to extract options for the
# axi_dma master -> PS-slave direction, so we wire it explicitly. Everything
# runs in the single PL clock domain (pl_clk0); one proc_sys_reset, two
# SmartConnects (PS->DMA control, DMA->PS memory).

# Reset block for the pl_clk0 domain. pl_resetn0 is active-low, so tell
# proc_sys_reset its external reset is active-low (C_EXT_RESET_HIGH=0).
create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:* rst
set_property -dict [list CONFIG.C_EXT_RESET_HIGH {0}] [get_bd_cells rst]
connect_bd_net [get_bd_pins ps/pl_clk0]    [get_bd_pins rst/slowest_sync_clk]
connect_bd_net [get_bd_pins ps/pl_resetn0] [get_bd_pins rst/ext_reset_in]

# All PL-facing clocks on pl_clk0.
foreach clkpin {
    ps/maxihpm0_fpd_aclk ps/saxihp0_fpd_aclk
    dma/s_axi_lite_aclk dma/m_axi_mm2s_aclk dma/m_axi_s2mm_aclk
    fifo/s_axis_aclk
} { connect_bd_net [get_bd_pins ps/pl_clk0] [get_bd_pins $clkpin] }

# Active-low peripheral reset to the DMA + FIFO.
foreach rstpin { dma/axi_resetn fifo/s_axis_aresetn } {
    connect_bd_net [get_bd_pins rst/peripheral_aresetn] [get_bd_pins $rstpin]
}

# Control path: PS M_AXI_HPM0_FPD -> sc_ctrl -> dma/S_AXI_LITE
create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:* sc_ctrl
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {1}] [get_bd_cells sc_ctrl]
connect_bd_net [get_bd_pins ps/pl_clk0]             [get_bd_pins sc_ctrl/aclk]
connect_bd_net [get_bd_pins rst/peripheral_aresetn] [get_bd_pins sc_ctrl/aresetn]
connect_bd_intf_net [get_bd_intf_pins ps/M_AXI_HPM0_FPD] [get_bd_intf_pins sc_ctrl/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins sc_ctrl/M00_AXI]   [get_bd_intf_pins dma/S_AXI_LITE]

# Memory path: dma/{M_AXI_MM2S,M_AXI_S2MM} -> sc_mem -> PS S_AXI_HP0_FPD
create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:* sc_mem
set_property -dict [list CONFIG.NUM_SI {2} CONFIG.NUM_MI {1}] [get_bd_cells sc_mem]
connect_bd_net [get_bd_pins ps/pl_clk0]             [get_bd_pins sc_mem/aclk]
connect_bd_net [get_bd_pins rst/peripheral_aresetn] [get_bd_pins sc_mem/aresetn]
connect_bd_intf_net [get_bd_intf_pins dma/M_AXI_MM2S] [get_bd_intf_pins sc_mem/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins dma/M_AXI_S2MM] [get_bd_intf_pins sc_mem/S01_AXI]
connect_bd_intf_net [get_bd_intf_pins sc_mem/M00_AXI] [get_bd_intf_pins ps/S_AXI_HP0_FPD]

assign_bd_address
validate_bd_design
save_bd_design

# Print the DMA control base address - needed for the MMIO-poke driver.
puts "==============================================================="
puts "DIAG: assigned addresses"
foreach seg [get_bd_addr_segs] { puts "  $seg" }
puts "==============================================================="

# ---- Wrap, synth, impl, bitstream -----------------------------------------
make_wrapper -files [get_files [file normalize ./$proj/$proj.srcs/sources_1/bd/$bd/$bd.bd]] -top
set wrap [lindex [glob -nocomplain \
    [file normalize ./$proj/$proj.gen/sources_1/bd/$bd/hdl/${bd}_wrapper.v] \
    [file normalize ./$proj/$proj.srcs/sources_1/bd/$bd/hdl/${bd}_wrapper.v]] 0]
add_files -norecurse $wrap
update_compile_order -fileset sources_1
set_property top ${bd}_wrapper [current_fileset]

# KR260 fan gate pin. This is intentionally constrained in the project script
# instead of being left to board automation: the fan route is part of the
# board-support contract for any full PL image loaded on this carrier.
set fan_xdc $outdir/fan_en_b.xdc
set fh [open $fan_xdc w]
puts $fh {set_property PACKAGE_PIN A12 [get_ports fan_en_b]}
puts $fh {set_property IOSTANDARD LVCMOS33 [get_ports fan_en_b]}
puts $fh {set_property SLEW SLOW [get_ports fan_en_b]}
puts $fh {set_property DRIVE 4 [get_ports fan_en_b]}
close $fh
add_files -fileset constrs_1 -norecurse $fan_xdc

launch_runs synth_1 -jobs 4
wait_on_run synth_1
if {[get_property PROGRESS [get_runs synth_1]] != "100%"} { error "synthesis failed" }

launch_runs impl_1 -to_step write_bitstream -jobs 4
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] != "100%"} { error "implementation failed" }

# ---- Exports: .bit and .xsa -----------------------------------------------
# build.bat converts loopback.bit to loopback.bit.bin with bootgen. deploy.sh
# packages that .bit.bin with overlay/penzai-loopback.dts for xmutil/dfx-mgr.
# The .xsa is retained for hardware handoff and inspection.
set bit [lindex [glob [file normalize ./$proj/$proj.runs/impl_1/${bd}_wrapper.bit]] 0]
file copy -force $bit $outdir/loopback.bit
open_run impl_1
write_hw_platform -fixed -include_bit -force $outdir/loopback.xsa

puts "==> Built: $outdir/loopback.bit and loopback.xsa"
puts "==> Use the DMA control base printed under 'DIAG: assigned addresses'"
puts "    above as DMA_BASE for the /dev/mem register check on the board."
