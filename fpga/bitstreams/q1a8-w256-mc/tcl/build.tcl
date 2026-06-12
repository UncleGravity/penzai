# build.tcl - KR260 Q1A8 matmul bitstream (v6: multi-column kernel).
#
# Design: one q1a8_kernel_top fed by two AXI DMAs.
#   PS pl_clk0 @ ~100 MHz -> clk_wiz -> fabric clock (fclk_mhz from variant)
#   PS M_AXI_HPM0_FPD -> AXI-Lite -> dma_w + dma_a + kernel
#   dma_w MM2S: DDR -> 128->256 upsizer -> kernel.S_AXIS (wide weights)
#   kernel.M_AXIS -> dma_w S2MM: DDR        (results)
#   dma_a MM2S: DDR -> kernel.S_AXIS_ACTS   (activations)
#   dma_w mem -> S_AXI_HP0_FPD, dma_a mem -> S_AXI_HP1_FPD
#
# Datapath is 64-bit to match the kernel AXIS width. This is the same proven
# block design as the bring-up's v4; the only RTL change is the v5 kernel_top's
# performance counter bank (AXI-Lite registers, no new ports). The wide-array /
# multi-HP work is a later, larger build.

set variant [expr {$argc >= 1 ? [lindex $argv 0] : "w256-f125"}]

# variant = w64-f<MHz>; the fabric clock is parsed from the name. The unpipelined
# fp32 reducer closes around ~137 MHz, so 100 MHz is the safe correctness target.
if {![regexp {^w256-f([0-9]+)$} $variant -> fclk_mhz]} {
    error "unknown variant '$variant'; expected w256-f<MHz>, e.g. w256-f125"
}

set bit_prefix penzai-q1a8-mc
set bit_name   "$bit_prefix-$variant"
set proj       "q1a8_matmul_[string map {- _} $variant]"
set bd         design_1
set part       xck26-sfvc784-2LV-c
set board      xilinx.com:kr260_som:part0:1.1
set outdir     [file normalize ./out]
set ps_fclk_mhz 99.999001

file delete -force $outdir [file normalize ./$proj]
file mkdir $outdir

proc assert_config {cell_name prop expected} {
    set cell [get_bd_cells $cell_name]
    set actual [get_property "CONFIG.$prop" $cell]
    if {$actual != $expected} { error "$cell_name CONFIG.$prop expected $expected, got $actual" }
}

proc first_addr_seg {patterns} {
    foreach pattern $patterns {
        set segs [get_bd_addr_segs -quiet $pattern]
        if {[llength $segs] > 0} { return [lindex $segs 0] }
    }
    error "none of these address segments exist: $patterns"
}

proc mi_pin {index} { return [format "M%02d_AXI" $index] }

create_project $proj [file normalize ./$proj] -part $part -force
if {[catch {set_property board_part $board [current_project]} err]} {
    puts "WARNING: could not set board_part '$board': $err"
}
set_property target_language Verilog [current_project]

set rtl_files [glob -nocomplain [file normalize ./rtl/*.v]]
if {[llength $rtl_files] == 0} { error "missing RTL files under ./rtl" }
add_files -norecurse $rtl_files
# kernel_top `include`s the generated q1a8_regs.vh. For a BD `module` reference
# to elaborate, Vivado needs the header added to the project (as a Verilog header,
# not a compilable module) *and* its directory on the include path.
set vh_file [file normalize ./rtl/q1a8_regs.vh]
add_files -norecurse $vh_file
set_property file_type "Verilog Header" [get_files $vh_file]
set_property include_dirs [list [file normalize ./rtl]] [get_filesets sources_1]
update_compile_order -fileset sources_1

create_bd_design $bd

# ---- Processing system ------------------------------------------------------
create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e:* ps
apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e \
    -config {apply_board_preset 1} [get_bd_cells ps]

set_property -dict [list \
    CONFIG.PSU__USE__M_AXI_GP0 {1} \
    CONFIG.PSU__USE__M_AXI_GP1 {0} \
    CONFIG.PSU__USE__S_AXI_GP2 {1} \
    CONFIG.PSU__USE__S_AXI_GP3 {1} \
    CONFIG.PSU__SAXIGP2__DATA_WIDTH {128} \
    CONFIG.PSU__SAXIGP3__DATA_WIDTH {128} \
    CONFIG.PSU__FPGA_PL0_ENABLE {1} \
    CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ {100} \
    CONFIG.PSU__TTC0__PERIPHERAL__ENABLE {1} \
    CONFIG.PSU__TTC0__WAVEOUT__ENABLE {1} \
    CONFIG.PSU__TTC0__WAVEOUT__IO {EMIO} \
] [get_bd_cells ps]

# KR260 fan gate on TTC0 channel 2 (carried over from the bandwidth fixtures).
create_bd_cell -type ip -vlnv xilinx.com:ip:xlslice:* fan_ttc0_ch2
set_property -dict [list CONFIG.DIN_WIDTH {3} CONFIG.DIN_FROM {2} CONFIG.DIN_TO {2}] \
    [get_bd_cells fan_ttc0_ch2]
create_bd_port -dir O fan_en_b
connect_bd_net [get_bd_pins ps/emio_ttc0_wave_o] [get_bd_pins fan_ttc0_ch2/Din]
connect_bd_net [get_bd_pins fan_ttc0_ch2/Dout]   [get_bd_ports fan_en_b]

# ---- Fabric clock -----------------------------------------------------------
create_bd_cell -type ip -vlnv xilinx.com:ip:clk_wiz:* clk_wiz
set_property -dict [list \
    CONFIG.PRIM_IN_FREQ $ps_fclk_mhz \
    CONFIG.CLKOUT1_REQUESTED_OUT_FREQ $fclk_mhz \
    CONFIG.USE_LOCKED {true} \
    CONFIG.USE_RESET {false} \
] [get_bd_cells clk_wiz]
connect_bd_net [get_bd_pins ps/pl_clk0] [get_bd_pins clk_wiz/clk_in1]

# ---- DMAs, width converters, and kernel -------------------------------------
# The AXI DMA forces a 128-bit memory-map/stream width for this addr/burst
# config (a 64-bit DMA silently became mem=128/stream=64 and corrupted data).
# So run the DMAs at the proven 128-bit width and bridge to the 64-bit kernel
# with AXIS data-width converters, which pack/unpack byte-exactly.
proc make_dma {name include_s2mm} {
    create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:* $name
    set_property -dict [list \
        CONFIG.c_include_sg {0} \
        CONFIG.c_include_mm2s {1} \
        CONFIG.c_include_s2mm $include_s2mm \
        CONFIG.c_sg_length_width {26} \
        CONFIG.c_addr_width {40} \
        CONFIG.c_m_axi_mm2s_data_width {128} \
        CONFIG.c_m_axis_mm2s_tdata_width {128} \
        CONFIG.c_mm2s_burst_size {16} \
    ] [get_bd_cells $name]
    assert_config $name c_m_axis_mm2s_tdata_width 128
    if {$include_s2mm} {
        set_property -dict [list \
            CONFIG.c_m_axi_s2mm_data_width {128} \
            CONFIG.c_s_axis_s2mm_tdata_width {128} \
            CONFIG.c_s2mm_burst_size {16} \
        ] [get_bd_cells $name]
        assert_config $name c_s_axis_s2mm_tdata_width 128
    }
}
make_dma dma_w 1
make_dma dma_a 0

# AXIS data-width converter (single clock). s/m widths in bytes.
proc make_dwc {name s_bytes m_bytes} {
    create_bd_cell -type ip -vlnv xilinx.com:ip:axis_dwidth_converter:* $name
    set_property -dict [list \
        CONFIG.S_TDATA_NUM_BYTES $s_bytes \
        CONFIG.M_TDATA_NUM_BYTES $m_bytes \
        CONFIG.HAS_TLAST {1} \
        CONFIG.HAS_TKEEP {1} \
    ] [get_bd_cells $name]
}
make_dwc dwc_w 16 32 ;# weights 128 -> 256
make_dwc dwc_a 16 8 ;# acts     128 -> 64
make_dwc dwc_r 8 16 ;# results   64 -> 128

create_bd_cell -type module -reference q1a8_kernel_mc_top kernel

connect_bd_intf_net [get_bd_intf_pins dma_w/M_AXIS_MM2S] [get_bd_intf_pins dwc_w/S_AXIS]
connect_bd_intf_net [get_bd_intf_pins dwc_w/M_AXIS]      [get_bd_intf_pins kernel/S_AXIS]
connect_bd_intf_net [get_bd_intf_pins dma_a/M_AXIS_MM2S] [get_bd_intf_pins dwc_a/S_AXIS]
connect_bd_intf_net [get_bd_intf_pins dwc_a/M_AXIS]      [get_bd_intf_pins kernel/S_AXIS_ACTS]
connect_bd_intf_net [get_bd_intf_pins kernel/M_AXIS]     [get_bd_intf_pins dwc_r/S_AXIS]
connect_bd_intf_net [get_bd_intf_pins dwc_r/M_AXIS]      [get_bd_intf_pins dma_w/S_AXIS_S2MM]

# ---- Reset ------------------------------------------------------------------
create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:* rst
set_property -dict [list CONFIG.C_EXT_RESET_HIGH {0}] [get_bd_cells rst]
connect_bd_net [get_bd_pins clk_wiz/clk_out1] [get_bd_pins rst/slowest_sync_clk]
connect_bd_net [get_bd_pins ps/pl_resetn0]    [get_bd_pins rst/ext_reset_in]
connect_bd_net [get_bd_pins clk_wiz/locked]   [get_bd_pins rst/dcm_locked]

# ---- Clock fan-out ----------------------------------------------------------
foreach clkpin {
    ps/maxihpm0_fpd_aclk ps/saxihp0_fpd_aclk ps/saxihp1_fpd_aclk
    dma_w/s_axi_lite_aclk dma_w/m_axi_mm2s_aclk dma_w/m_axi_s2mm_aclk
    dma_a/s_axi_lite_aclk dma_a/m_axi_mm2s_aclk
    dwc_w/aclk dwc_a/aclk dwc_r/aclk
    kernel/s_axi_aclk
} { connect_bd_net [get_bd_pins clk_wiz/clk_out1] [get_bd_pins $clkpin] }

foreach rstpin {
    dma_w/axi_resetn dma_a/axi_resetn
    dwc_w/aresetn dwc_a/aresetn dwc_r/aresetn
    kernel/s_axi_aresetn
} { connect_bd_net [get_bd_pins rst/peripheral_aresetn] [get_bd_pins $rstpin] }

# ---- Control path: PS -> dma_w, dma_a, kernel AXI-Lite ----------------------
create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:* sc_ctrl
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {3}] [get_bd_cells sc_ctrl]
connect_bd_net [get_bd_pins clk_wiz/clk_out1]       [get_bd_pins sc_ctrl/aclk]
connect_bd_net [get_bd_pins rst/peripheral_aresetn] [get_bd_pins sc_ctrl/aresetn]
connect_bd_intf_net [get_bd_intf_pins ps/M_AXI_HPM0_FPD] [get_bd_intf_pins sc_ctrl/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins sc_ctrl/[mi_pin 0]] [get_bd_intf_pins dma_w/S_AXI_LITE]
connect_bd_intf_net [get_bd_intf_pins sc_ctrl/[mi_pin 1]] [get_bd_intf_pins dma_a/S_AXI_LITE]
connect_bd_intf_net [get_bd_intf_pins sc_ctrl/[mi_pin 2]] [get_bd_intf_pins kernel/S_AXI]

# ---- Memory paths -----------------------------------------------------------
# dma_w MM2S + S2MM share HP0; dma_a MM2S uses HP1.
create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:* sc_mem_w
set_property -dict [list CONFIG.NUM_SI {2} CONFIG.NUM_MI {1}] [get_bd_cells sc_mem_w]
connect_bd_net [get_bd_pins clk_wiz/clk_out1]       [get_bd_pins sc_mem_w/aclk]
connect_bd_net [get_bd_pins rst/peripheral_aresetn] [get_bd_pins sc_mem_w/aresetn]
connect_bd_intf_net [get_bd_intf_pins dma_w/M_AXI_MM2S] [get_bd_intf_pins sc_mem_w/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins dma_w/M_AXI_S2MM] [get_bd_intf_pins sc_mem_w/S01_AXI]
connect_bd_intf_net [get_bd_intf_pins sc_mem_w/M00_AXI] [get_bd_intf_pins ps/S_AXI_HP0_FPD]

create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:* sc_mem_a
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {1}] [get_bd_cells sc_mem_a]
connect_bd_net [get_bd_pins clk_wiz/clk_out1]       [get_bd_pins sc_mem_a/aclk]
connect_bd_net [get_bd_pins rst/peripheral_aresetn] [get_bd_pins sc_mem_a/aresetn]
connect_bd_intf_net [get_bd_intf_pins dma_a/M_AXI_MM2S] [get_bd_intf_pins sc_mem_a/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins sc_mem_a/M00_AXI] [get_bd_intf_pins ps/S_AXI_HP1_FPD]

# ---- Address map (must match device/pl/matmul.zig) --------------------------
foreach entry {
    {dma_w  S_AXI_LITE 0xA0000000}
    {dma_a  S_AXI_LITE 0xA0010000}
    {kernel S_AXI      0xA0020000}
} {
    lassign $entry cell intf offset
    assign_bd_address -offset $offset -range 64K \
        [first_addr_seg [list "$cell/$intf/Reg" "$cell/$intf/*"]]
}
# Route the DMA mem masters to PS DDR through their HP ports.
assign_bd_address

validate_bd_design
save_bd_design

# ---- Wrap, synth, impl, bitstream ------------------------------------------
make_wrapper -files [get_files [file normalize ./$proj/$proj.srcs/sources_1/bd/$bd/$bd.bd]] -top
set wrap [lindex [glob -nocomplain \
    [file normalize ./$proj/$proj.gen/sources_1/bd/$bd/hdl/${bd}_wrapper.v] \
    [file normalize ./$proj/$proj.srcs/sources_1/bd/$bd/hdl/${bd}_wrapper.v]] 0]
add_files -norecurse $wrap
update_compile_order -fileset sources_1
set_property top ${bd}_wrapper [current_fileset]

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

launch_runs impl_1 -to_step route_design -jobs 4
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] != "100%"} { error "implementation failed" }

open_run impl_1
report_timing_summary -max_paths 10 -routable_nets -report_unconstrained \
    -file $outdir/${bit_name}_timing_summary_routed.rpt
set failing_paths [get_timing_paths -quiet -max_paths 1 -slack_lesser_than 0]
if {[llength $failing_paths] > 0} {
    set wns [get_property SLACK [lindex $failing_paths 0]]
    error "timing failed after route_design; WNS=${wns}ns. Refusing to write an invalid bitstream."
}

write_bitstream -force $outdir/$bit_name.bit
# Emit the .hwh (hardware handoff) alongside the bitstream for XRT/PYNQ tooling.
catch {write_hw_platform -fixed -force -file $outdir/$bit_name.xsa}
puts "==> Built timing-clean bitstream: $outdir/$bit_name.bit"
