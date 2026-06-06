# build.tcl - KR260 HP0-HP3 DDR bandwidth fixture.
#
# Design:
#   PS pl_clk0 @ runtime 100 MHz -> clk_wiz -> 300 MHz fabric clock
#   PS M_AXI_HPM0_FPD -> AXI-Lite -> 4x AXI DMA + 4x bandwidth_regs
#   engineN.M_AXIS -> dmaN S2MM -> S_AXI_HPN_FPD -> DDR
#   DDR -> S_AXI_HPN_FPD -> dmaN MM2S -> engineN.S_AXIS
#
# The measurement question is how close four 128-bit HP ports at 300 MHz can
# get to the KR260 DDR theoretical ceiling of 19.2 GB/s.

set variant [expr {$argc >= 1 ? [lindex $argv 0] : "w128-f300-p4"}]

switch -- $variant {
    w128-f300-p4 {
        set fclk_mhz 300
        set port_count 4
    }
    default { error "unknown variant '$variant'; expected w128-f300-p4" }
}

set bit_prefix penzai-ddr-bandwidth-multiport
set bit_name   "$bit_prefix-$variant"
set proj       "ddr_multiport_[string map {- _} $variant]"
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

proc mi_pin {index} {
    return [format "M%02d_AXI" $index]
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
    CONFIG.PSU__USE__S_AXI_GP3 {1} \
    CONFIG.PSU__USE__S_AXI_GP4 {1} \
    CONFIG.PSU__USE__S_AXI_GP5 {1} \
    CONFIG.PSU__SAXIGP2__DATA_WIDTH {128} \
    CONFIG.PSU__SAXIGP3__DATA_WIDTH {128} \
    CONFIG.PSU__SAXIGP4__DATA_WIDTH {128} \
    CONFIG.PSU__SAXIGP5__DATA_WIDTH {128} \
    CONFIG.PSU__FPGA_PL0_ENABLE {1} \
    CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ {100} \
    CONFIG.PSU__TTC0__PERIPHERAL__ENABLE {1} \
    CONFIG.PSU__TTC0__WAVEOUT__ENABLE {1} \
    CONFIG.PSU__TTC0__WAVEOUT__IO {EMIO} \
] [get_bd_cells ps]

foreach gp {2 3 4 5} {
    assert_config ps "PSU__SAXIGP${gp}__DATA_WIDTH" 128
}

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
create_bd_cell -type ip -vlnv xilinx.com:ip:clk_wiz:* clk_wiz
set_property -dict [list \
    CONFIG.PRIM_IN_FREQ $ps_fclk_mhz \
    CONFIG.CLKOUT1_REQUESTED_OUT_FREQ $fclk_mhz \
    CONFIG.USE_LOCKED {true} \
    CONFIG.USE_RESET {false} \
] [get_bd_cells clk_wiz]
connect_bd_net [get_bd_pins ps/pl_clk0] [get_bd_pins clk_wiz/clk_in1]

# ---- Lanes: AXI DMA + custom generator/checker/register block ---------------
for {set i 0} {$i < $port_count} {incr i} {
    set dma "dma$i"
    set engine "engine$i"

    create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:* $dma
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
    ] [get_bd_cells $dma]

    assert_config $dma c_m_axi_mm2s_data_width 128
    assert_config $dma c_m_axi_s2mm_data_width 128
    assert_config $dma c_m_axis_mm2s_tdata_width 128
    assert_config $dma c_s_axis_s2mm_tdata_width 128

    create_bd_cell -type module -reference bandwidth_regs $engine

    connect_bd_intf_net [get_bd_intf_pins $engine/M_AXIS]   [get_bd_intf_pins $dma/S_AXIS_S2MM]
    connect_bd_intf_net [get_bd_intf_pins $dma/M_AXIS_MM2S] [get_bd_intf_pins $engine/S_AXIS]
}

# ---- Clocks and reset -------------------------------------------------------
create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:* rst
set_property -dict [list CONFIG.C_EXT_RESET_HIGH {0}] [get_bd_cells rst]
connect_bd_net [get_bd_pins clk_wiz/clk_out1] [get_bd_pins rst/slowest_sync_clk]
connect_bd_net [get_bd_pins ps/pl_resetn0] [get_bd_pins rst/ext_reset_in]
connect_bd_net [get_bd_pins clk_wiz/locked] [get_bd_pins rst/dcm_locked]

foreach clkpin {
    ps/maxihpm0_fpd_aclk
    ps/saxihp0_fpd_aclk ps/saxihp1_fpd_aclk ps/saxihp2_fpd_aclk ps/saxihp3_fpd_aclk
} { connect_bd_net [get_bd_pins clk_wiz/clk_out1] [get_bd_pins $clkpin] }

for {set i 0} {$i < $port_count} {incr i} {
    foreach clkpin [list \
        dma${i}/s_axi_lite_aclk \
        dma${i}/m_axi_mm2s_aclk \
        dma${i}/m_axi_s2mm_aclk \
        engine${i}/aclk \
    ] { connect_bd_net [get_bd_pins clk_wiz/clk_out1] [get_bd_pins $clkpin] }

    foreach rstpin [list dma${i}/axi_resetn engine${i}/aresetn] {
        connect_bd_net [get_bd_pins rst/peripheral_aresetn] [get_bd_pins $rstpin]
    }
}

# ---- Control path: PS -> per-lane DMA + engine AXI-Lite ---------------------
create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:* sc_ctrl
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {4}] [get_bd_cells sc_ctrl]
connect_bd_net [get_bd_pins clk_wiz/clk_out1]       [get_bd_pins sc_ctrl/aclk]
connect_bd_net [get_bd_pins rst/peripheral_aresetn] [get_bd_pins sc_ctrl/aresetn]
connect_bd_intf_net [get_bd_intf_pins ps/M_AXI_HPM0_FPD] [get_bd_intf_pins sc_ctrl/S00_AXI]

for {set i 0} {$i < $port_count} {incr i} {
    set lane_ctrl "sc_ctrl_lane$i"
    create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:* $lane_ctrl
    set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {2}] [get_bd_cells $lane_ctrl]
    connect_bd_net [get_bd_pins clk_wiz/clk_out1]       [get_bd_pins $lane_ctrl/aclk]
    connect_bd_net [get_bd_pins rst/peripheral_aresetn] [get_bd_pins $lane_ctrl/aresetn]
    connect_bd_intf_net [get_bd_intf_pins sc_ctrl/[mi_pin $i]] [get_bd_intf_pins $lane_ctrl/S00_AXI]
    connect_bd_intf_net [get_bd_intf_pins $lane_ctrl/M00_AXI] [get_bd_intf_pins dma${i}/S_AXI_LITE]
    connect_bd_intf_net [get_bd_intf_pins $lane_ctrl/M01_AXI] [get_bd_intf_pins engine${i}/S_AXI]
}

# ---- Memory paths: one SmartConnect per HP DDR port -------------------------
for {set i 0} {$i < $port_count} {incr i} {
    set sc "sc_mem$i"
    create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:* $sc
    set_property -dict [list CONFIG.NUM_SI {2} CONFIG.NUM_MI {1}] [get_bd_cells $sc]
    connect_bd_net [get_bd_pins clk_wiz/clk_out1]       [get_bd_pins $sc/aclk]
    connect_bd_net [get_bd_pins rst/peripheral_aresetn] [get_bd_pins $sc/aresetn]
    connect_bd_intf_net [get_bd_intf_pins dma${i}/M_AXI_MM2S] [get_bd_intf_pins $sc/S00_AXI]
    connect_bd_intf_net [get_bd_intf_pins dma${i}/M_AXI_S2MM] [get_bd_intf_pins $sc/S01_AXI]
    connect_bd_intf_net [get_bd_intf_pins $sc/M00_AXI]        [get_bd_intf_pins ps/S_AXI_HP${i}_FPD]
}

foreach entry {
    {dma0    S_AXI_LITE 0xA0000000}
    {engine0 S_AXI      0xA0010000}
    {dma1    S_AXI_LITE 0xA0020000}
    {engine1 S_AXI      0xA0030000}
    {dma2    S_AXI_LITE 0xA0040000}
    {engine2 S_AXI      0xA0050000}
    {dma3    S_AXI_LITE 0xA0060000}
    {engine3 S_AXI      0xA0070000}
} {
    lassign $entry cell intf offset
    assign_bd_address -offset $offset -range 64K \
        [first_addr_seg [list "$cell/$intf/Reg" "$cell/$intf/*"]]
}

# Map the remaining reachable segments. This gives each DMA MM2S/S2MM master
# address space a route to PS DDR through its HP port.
assign_bd_address

validate_bd_design
save_bd_design

puts "==============================================================="
puts "DIAG: variant=$variant fclk_mhz=$fclk_mhz port_count=$port_count"
print_width_props "clk_wiz" [get_bd_cells clk_wiz]
for {set i 0} {$i < $port_count} {incr i} {
    print_width_props "ps/S_AXI_HP${i}_FPD" [get_bd_intf_pins ps/S_AXI_HP${i}_FPD]
    print_width_props "dma$i" [get_bd_cells dma$i]
    print_width_props "sc_mem$i" [get_bd_cells sc_mem$i]
    print_width_props "engine$i/M_AXIS" [get_bd_intf_pins engine$i/M_AXIS]
    print_width_props "engine$i/S_AXIS" [get_bd_intf_pins engine$i/S_AXIS]
}
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

launch_runs impl_1 -to_step route_design -jobs 4
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] != "100%"} { error "implementation failed" }

open_run impl_1
report_timing_summary -max_paths 10 -routable_nets -report_unconstrained \
    -file $outdir/${bit_name}_timing_summary_routed.rpt
set failing_paths [get_timing_paths -quiet -max_paths 1 -slack_lesser_than 0]
if {[llength $failing_paths] > 0} {
    set wns [get_property SLACK [lindex $failing_paths 0]]
    error "timing failed after route_design; WNS=${wns}ns. Refusing to write/package an invalid bitstream."
}

write_bitstream -force $outdir/$bit_name.bit

puts "==> Built timing-clean bitstream: $outdir/$bit_name.bit"
