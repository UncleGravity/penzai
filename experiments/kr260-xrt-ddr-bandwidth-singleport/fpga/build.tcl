# build.tcl - KR260 HP0 DDR bandwidth single-port fixture.
#
# Design:
#   PS pl_clk0 @ runtime 100 MHz -> clk_wiz -> variant fabric clock
#   PS M_AXI_HPM0_FPD -> AXI-Lite -> AXI DMA + bandwidth_regs
#   bandwidth_regs.M_AXIS -> AXI DMA S2MM -> S_AXI_HP0_FPD -> DDR
#   DDR -> S_AXI_HP0_FPD -> AXI DMA MM2S -> bandwidth_regs.S_AXIS
#
# The measurement question is whether one HP0 path, deliberately configured as
# 128-bit wide, reaches the expected per-port bandwidth at a few PL clocks.

set variant [expr {$argc >= 1 ? [lindex $argv 0] : "w128-f100"}]

switch -- $variant {
    w128-f100 { set fclk_mhz 100 }
    w128-f200 { set fclk_mhz 200 }
    w128-f250 { set fclk_mhz 250 }
    w128-f300 { set fclk_mhz 300 }
    default { error "unknown variant '$variant'; expected w128-f100, w128-f200, w128-f250, or w128-f300" }
}

set bit_prefix penzai-ddr-bandwidth-singleport
set bit_name   "$bit_prefix-$variant"
set proj       "ddr_singleport_[string map {- _} $variant]"
set bd         design_1
set part       xck26-sfvc784-2LV-c
set board      xilinx.com:kr260_som:part0:1.1
set outdir     [file normalize ./out]
set ps_fclk_mhz 99.999001

file delete -force $outdir [file normalize ./$proj]
file mkdir $outdir

proc assert_config {cell_name prop expected} {
    set cell [get_bd_cells $cell_name]
    set key "CONFIG.$prop"
    set actual [get_property $key $cell]
    if {$actual != $expected} {
        error "$cell_name $key expected $expected, got $actual"
    }
}

proc first_addr_seg {patterns} {
    foreach pattern $patterns {
        set segs [get_bd_addr_segs -quiet $pattern]
        if {[llength $segs] > 0} {
            return [lindex $segs 0]
        }
    }
    error "none of these address segments exist: $patterns"
}

proc print_width_props {label obj} {
    puts "DIAG: width-related properties for $label"
    foreach prop [list_property $obj] {
        if {[regexp {WIDTH|DATA_WIDTH|TDATA|FREQ|BURST} $prop]} {
            if {![catch {set value [get_property $prop $obj]}]} {
                puts "  $prop = $value"
            }
        }
    }
}

create_project $proj [file normalize ./$proj] -part $part -force
if {[catch {set_property board_part $board [current_project]} err]} {
    puts "WARNING: could not set board_part '$board': $err"
    puts "         install KR260 board files or fix the string (get_board_parts *kr260*)"
}
set_property target_language Verilog [current_project]

set rtl_files [glob -nocomplain [file normalize ./rtl/*.v]]
if {[llength $rtl_files] == 0} { error "missing RTL files under ./rtl" }
add_files -norecurse $rtl_files
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
    CONFIG.PSU__SAXIGP2__DATA_WIDTH {128} \
    CONFIG.PSU__FPGA_PL0_ENABLE {1} \
    CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ {100} \
    CONFIG.PSU__TTC0__PERIPHERAL__ENABLE {1} \
    CONFIG.PSU__TTC0__WAVEOUT__ENABLE {1} \
    CONFIG.PSU__TTC0__WAVEOUT__IO {EMIO} \
] [get_bd_cells ps]
assert_config ps PSU__SAXIGP2__DATA_WIDTH 128

# KR260 fan gate on TTC0 channel 2, carried over from the loopback fixture.
create_bd_cell -type ip -vlnv xilinx.com:ip:xlslice:* fan_ttc0_ch2
set_property -dict [list \
    CONFIG.DIN_WIDTH {3} \
    CONFIG.DIN_FROM {2} \
    CONFIG.DIN_TO {2} \
] [get_bd_cells fan_ttc0_ch2]
create_bd_port -dir O fan_en_b
connect_bd_net [get_bd_pins ps/emio_ttc0_wave_o] [get_bd_pins fan_ttc0_ch2/Din]
connect_bd_net [get_bd_pins fan_ttc0_ch2/Dout]   [get_bd_ports fan_en_b]

# ---- Variant fabric clock ---------------------------------------------------
# Loading a flat bitstream through xmutil does not reprogram the PS FCLK rate on
# this board, so treat pl_clk0 as the stable runtime ~100 MHz input and
# synthesize the measured fabric clock in PL. The K26 preset annotates pl_clk0 as
# 99,999,001 Hz, so clk_wiz must use the matching input metadata.
create_bd_cell -type ip -vlnv xilinx.com:ip:clk_wiz:* clk_wiz
set_property -dict [list \
    CONFIG.PRIM_IN_FREQ $ps_fclk_mhz \
    CONFIG.CLKOUT1_REQUESTED_OUT_FREQ $fclk_mhz \
    CONFIG.USE_LOCKED {true} \
    CONFIG.USE_RESET {false} \
] [get_bd_cells clk_wiz]
connect_bd_net [get_bd_pins ps/pl_clk0] [get_bd_pins clk_wiz/clk_in1]

# ---- AXI DMA, forced to 128-bit memory and stream paths ---------------------
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:* dma
set_property -dict [list \
    CONFIG.c_include_sg {0} \
    CONFIG.c_include_mm2s {1} \
    CONFIG.c_include_s2mm {1} \
    CONFIG.c_sg_length_width {26} \
    CONFIG.c_addr_width {40} \
    CONFIG.c_m_axi_mm2s_data_width {128} \
    CONFIG.c_m_axi_s2mm_data_width {128} \
    CONFIG.c_m_axis_mm2s_tdata_width {128} \
    CONFIG.c_s_axis_s2mm_tdata_width {128} \
    CONFIG.c_mm2s_burst_size {256} \
    CONFIG.c_s2mm_burst_size {256} \
] [get_bd_cells dma]

assert_config dma c_m_axi_mm2s_data_width 128
assert_config dma c_m_axi_s2mm_data_width 128
assert_config dma c_m_axis_mm2s_tdata_width 128
assert_config dma c_s_axis_s2mm_tdata_width 128

# ---- Custom generator/checker/register block --------------------------------
create_bd_cell -type module -reference bandwidth_regs engine

connect_bd_intf_net [get_bd_intf_pins engine/M_AXIS]      [get_bd_intf_pins dma/S_AXIS_S2MM]
connect_bd_intf_net [get_bd_intf_pins dma/M_AXIS_MM2S]    [get_bd_intf_pins engine/S_AXIS]

# ---- Clocks and reset -------------------------------------------------------
create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:* rst
set_property -dict [list CONFIG.C_EXT_RESET_HIGH {0}] [get_bd_cells rst]
connect_bd_net [get_bd_pins clk_wiz/clk_out1] [get_bd_pins rst/slowest_sync_clk]
connect_bd_net [get_bd_pins ps/pl_resetn0] [get_bd_pins rst/ext_reset_in]
connect_bd_net [get_bd_pins clk_wiz/locked] [get_bd_pins rst/dcm_locked]

foreach clkpin {
    ps/maxihpm0_fpd_aclk ps/saxihp0_fpd_aclk
    dma/s_axi_lite_aclk dma/m_axi_mm2s_aclk dma/m_axi_s2mm_aclk
    engine/aclk
} { connect_bd_net [get_bd_pins clk_wiz/clk_out1] [get_bd_pins $clkpin] }

foreach rstpin { dma/axi_resetn engine/aresetn } {
    connect_bd_net [get_bd_pins rst/peripheral_aresetn] [get_bd_pins $rstpin]
}

# ---- Control path: PS -> DMA + engine AXI-Lite ------------------------------
create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:* sc_ctrl
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {2}] [get_bd_cells sc_ctrl]
connect_bd_net [get_bd_pins clk_wiz/clk_out1]       [get_bd_pins sc_ctrl/aclk]
connect_bd_net [get_bd_pins rst/peripheral_aresetn] [get_bd_pins sc_ctrl/aresetn]
connect_bd_intf_net [get_bd_intf_pins ps/M_AXI_HPM0_FPD] [get_bd_intf_pins sc_ctrl/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins sc_ctrl/M00_AXI]   [get_bd_intf_pins dma/S_AXI_LITE]
connect_bd_intf_net [get_bd_intf_pins sc_ctrl/M01_AXI]   [get_bd_intf_pins engine/S_AXI]

# ---- Memory path: DMA -> HP0 DDR -------------------------------------------
create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:* sc_mem
set_property -dict [list CONFIG.NUM_SI {2} CONFIG.NUM_MI {1}] [get_bd_cells sc_mem]
connect_bd_net [get_bd_pins clk_wiz/clk_out1]       [get_bd_pins sc_mem/aclk]
connect_bd_net [get_bd_pins rst/peripheral_aresetn] [get_bd_pins sc_mem/aresetn]
connect_bd_intf_net [get_bd_intf_pins dma/M_AXI_MM2S] [get_bd_intf_pins sc_mem/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins dma/M_AXI_S2MM] [get_bd_intf_pins sc_mem/S01_AXI]
connect_bd_intf_net [get_bd_intf_pins sc_mem/M00_AXI] [get_bd_intf_pins ps/S_AXI_HP0_FPD]

assign_bd_address -offset 0xA0000000 -range 64K \
    [first_addr_seg {"dma/S_AXI_LITE/Reg" "dma/S_AXI_LITE/*"}]
assign_bd_address -offset 0xA0010000 -range 64K \
    [first_addr_seg {"engine/S_AXI/Reg" "engine/S_AXI/*"}]

# Map the remaining reachable segments. This gives the DMA MM2S/S2MM master
# address spaces a route to PS DDR through HP0; without it, S2MM can halt with
# DMADecErr when handed a valid XRT BO physical address.
assign_bd_address

validate_bd_design
save_bd_design

puts "==============================================================="
puts "DIAG: variant=$variant fclk_mhz=$fclk_mhz"
print_width_props "ps/S_AXI_HP0_FPD" [get_bd_intf_pins ps/S_AXI_HP0_FPD]
print_width_props "clk_wiz" [get_bd_cells clk_wiz]
print_width_props "dma" [get_bd_cells dma]
print_width_props "sc_mem" [get_bd_cells sc_mem]
print_width_props "engine/M_AXIS" [get_bd_intf_pins engine/M_AXIS]
print_width_props "engine/S_AXIS" [get_bd_intf_pins engine/S_AXIS]
puts "DIAG: assigned addresses"
foreach seg [get_bd_addr_segs] { puts "  $seg" }
puts "==============================================================="

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

launch_runs impl_1 -to_step write_bitstream -jobs 4
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] != "100%"} { error "implementation failed" }

set bit [lindex [glob [file normalize ./$proj/$proj.runs/impl_1/${bd}_wrapper.bit]] 0]
file copy -force $bit $outdir/$bit_name.bit
open_run impl_1
write_hw_platform -fixed -include_bit -force $outdir/$bit_name.xsa

puts "==> Built: $outdir/$bit_name.bit and $bit_name.xsa"
