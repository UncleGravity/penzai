# build.tcl - KR260 flash-attention bitstream (v1 sequential kernel).
#
# Design: one flash_top fed by five AXI DMAs on a single fabric clock. The v1
# kernel is correctness-first / sequential, so unlike the matmul bitstream there is
# no weight-feed clock or CDC — one clock domain throughout.
#
#   PS pl_clk0 -> clk_wiz -> fabric clock (fclk_mhz from variant)
#   PS M_AXI_HPM0_FPD -> AXI-Lite -> dma_q/k/v/mask/o + kernel
#   dma_q  MM2S: DDR -> 128->256 dwc -> kernel.S_AXIS_Q
#   dma_k  MM2S: DDR ->            kernel.S_AXIS_K     (128, direct)
#   dma_v  MM2S: DDR ->            kernel.S_AXIS_V     (128, direct)
#   dma_mask MM2S: DDR -> 128->16 dwc -> kernel.S_AXIS_MASK
#   kernel.M_AXIS_O -> 256->128 dwc -> dma_o S2MM -> DDR
#
# Memory: HP0 carries dma_q + dma_o + dma_mask; HP1 dma_k; HP2 dma_v.

set variant [expr {$argc >= 1 ? [lindex $argv 0] : "f100"}]
if {![regexp {^f([0-9]+)$} $variant -> fclk_mhz]} {
    error "unknown variant '$variant'; expected f<MHz>, e.g. f100"
}

set bit_prefix penzai-flash-v1
set bit_name   "$bit_prefix-$variant"
set proj       "flash_[string map {- _} $variant]"
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
proc clock_hz {pin fallback_mhz} {
    set hz ""
    catch {set hz [get_property CONFIG.FREQ_HZ [get_bd_pins $pin]]}
    if {$hz ne "" && $hz != 0} { return [expr {int(round(double($hz)))}] }
    set actual_mhz ""
    catch {set actual_mhz [get_property CONFIG.CLKOUT1_ACTUAL_OUT_FREQ [get_bd_cells clk_wiz]]}
    if {$actual_mhz ne "" && $actual_mhz != 0} {
        return [expr {int(round(double($actual_mhz) * 1000000.0))}]
    }
    return [expr {int(round(double($fallback_mhz) * 1000000.0))}]
}

create_project $proj [file normalize ./$proj] -part $part -force
if {[catch {set_property board_part $board [current_project]} err]} {
    puts "WARNING: could not set board_part '$board': $err"
}
set_property target_language Verilog [current_project]

set rtl_files [glob -nocomplain [file normalize ./rtl/*.v]]
if {[llength $rtl_files] == 0} { error "missing RTL files under ./rtl" }
add_files -norecurse $rtl_files
# flash_top `include`s flash_regs.vh; fp_exp/fp_recip `include` flash_luts.vh. Both
# must be added as Verilog headers and their dir put on the include path.
foreach vh {flash_regs.vh flash_luts.vh} {
    set f [file normalize ./rtl/$vh]
    if {![file exists $f]} { error "missing ./rtl/$vh (run 'zig build regmap' for flash_regs.vh)" }
    add_files -norecurse $f
    set_property file_type "Verilog Header" [get_files $f]
}
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
    CONFIG.PSU__USE__S_AXI_GP4 {1} \
    CONFIG.PSU__SAXIGP2__DATA_WIDTH {128} \
    CONFIG.PSU__SAXIGP3__DATA_WIDTH {128} \
    CONFIG.PSU__SAXIGP4__DATA_WIDTH {128} \
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

# ---- Fabric clock (single domain) -------------------------------------------
create_bd_cell -type ip -vlnv xilinx.com:ip:clk_wiz:* clk_wiz
set_property -dict [list \
    CONFIG.PRIM_IN_FREQ $ps_fclk_mhz \
    CONFIG.CLKOUT1_REQUESTED_OUT_FREQ $fclk_mhz \
    CONFIG.USE_LOCKED {true} \
    CONFIG.USE_RESET {false} \
] [get_bd_cells clk_wiz]
connect_bd_net [get_bd_pins ps/pl_clk0] [get_bd_pins clk_wiz/clk_in1]
set fclk_pin "clk_wiz/clk_out1"

# ---- DMAs (128-bit mem + stream; converters bridge to the kernel widths) -----
proc make_dma {name mm2s s2mm} {
    create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:* $name
    set_property -dict [list \
        CONFIG.c_include_sg {0} \
        CONFIG.c_include_mm2s $mm2s \
        CONFIG.c_include_s2mm $s2mm \
        CONFIG.c_sg_length_width {26} \
        CONFIG.c_addr_width {40} \
    ] [get_bd_cells $name]
    if {$mm2s} {
        set_property -dict [list \
            CONFIG.c_m_axi_mm2s_data_width {128} \
            CONFIG.c_m_axis_mm2s_tdata_width {128} \
            CONFIG.c_mm2s_burst_size {16} \
        ] [get_bd_cells $name]
        assert_config $name c_m_axis_mm2s_tdata_width 128
    }
    if {$s2mm} {
        set_property -dict [list \
            CONFIG.c_m_axi_s2mm_data_width {128} \
            CONFIG.c_s_axis_s2mm_tdata_width {128} \
            CONFIG.c_s2mm_burst_size {16} \
        ] [get_bd_cells $name]
        assert_config $name c_s_axis_s2mm_tdata_width 128
    }
}
make_dma dma_q 1 0
make_dma dma_k 1 0
make_dma dma_v 1 0
make_dma dma_mask 1 0
make_dma dma_o 0 1

# AXIS data-width converters (single clock; bytes are TDATA widths).
proc make_dwc {name s_bytes m_bytes} {
    create_bd_cell -type ip -vlnv xilinx.com:ip:axis_dwidth_converter:* $name
    set_property -dict [list \
        CONFIG.S_TDATA_NUM_BYTES $s_bytes \
        CONFIG.M_TDATA_NUM_BYTES $m_bytes \
        CONFIG.HAS_TLAST {1} \
        CONFIG.HAS_TKEEP {1} \
    ] [get_bd_cells $name]
}
make_dwc dwc_q 16 32  ;# Q     128 -> 256
make_dwc dwc_mask 16 2 ;# mask  128 -> 16
make_dwc dwc_o 32 16  ;# O     256 -> 128

create_bd_cell -type module -reference flash_top kernel

# ---- AXIS wiring (one clock domain, no clock converters) --------------------
connect_bd_intf_net [get_bd_intf_pins dma_q/M_AXIS_MM2S]    [get_bd_intf_pins dwc_q/S_AXIS]
connect_bd_intf_net [get_bd_intf_pins dwc_q/M_AXIS]         [get_bd_intf_pins kernel/S_AXIS_Q]
connect_bd_intf_net [get_bd_intf_pins dma_k/M_AXIS_MM2S]    [get_bd_intf_pins kernel/S_AXIS_K]
connect_bd_intf_net [get_bd_intf_pins dma_v/M_AXIS_MM2S]    [get_bd_intf_pins kernel/S_AXIS_V]
connect_bd_intf_net [get_bd_intf_pins dma_mask/M_AXIS_MM2S] [get_bd_intf_pins dwc_mask/S_AXIS]
connect_bd_intf_net [get_bd_intf_pins dwc_mask/M_AXIS]      [get_bd_intf_pins kernel/S_AXIS_MASK]
connect_bd_intf_net [get_bd_intf_pins kernel/M_AXIS_O]      [get_bd_intf_pins dwc_o/S_AXIS]
connect_bd_intf_net [get_bd_intf_pins dwc_o/M_AXIS]         [get_bd_intf_pins dma_o/S_AXIS_S2MM]

# ---- Reset ------------------------------------------------------------------
create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:* rst
set_property -dict [list CONFIG.C_EXT_RESET_HIGH {0}] [get_bd_cells rst]
connect_bd_net [get_bd_pins $fclk_pin]     [get_bd_pins rst/slowest_sync_clk]
connect_bd_net [get_bd_pins ps/pl_resetn0] [get_bd_pins rst/ext_reset_in]
connect_bd_net [get_bd_pins clk_wiz/locked] [get_bd_pins rst/dcm_locked]

# ---- Clock fan-out (single clock to everything) -----------------------------
foreach clkpin {
    ps/maxihpm0_fpd_aclk
    ps/saxihp0_fpd_aclk ps/saxihp1_fpd_aclk ps/saxihp2_fpd_aclk
    dma_q/s_axi_lite_aclk dma_k/s_axi_lite_aclk dma_v/s_axi_lite_aclk
    dma_mask/s_axi_lite_aclk dma_o/s_axi_lite_aclk
    dma_q/m_axi_mm2s_aclk dma_k/m_axi_mm2s_aclk dma_v/m_axi_mm2s_aclk dma_mask/m_axi_mm2s_aclk
    dma_o/m_axi_s2mm_aclk
    dwc_q/aclk dwc_mask/aclk dwc_o/aclk
    kernel/s_axi_aclk
} { connect_bd_net [get_bd_pins $fclk_pin] [get_bd_pins $clkpin] }

foreach rstpin {
    dma_q/axi_resetn dma_k/axi_resetn dma_v/axi_resetn dma_mask/axi_resetn dma_o/axi_resetn
    dwc_q/aresetn dwc_mask/aresetn dwc_o/aresetn
    kernel/s_axi_aresetn
} { connect_bd_net [get_bd_pins rst/peripheral_aresetn] [get_bd_pins $rstpin] }

# ---- Control path: PS -> 5 DMAs + kernel AXI-Lite ---------------------------
create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:* sc_ctrl
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {6}] [get_bd_cells sc_ctrl]
connect_bd_net [get_bd_pins $fclk_pin]              [get_bd_pins sc_ctrl/aclk]
connect_bd_net [get_bd_pins rst/peripheral_aresetn] [get_bd_pins sc_ctrl/aresetn]
connect_bd_intf_net [get_bd_intf_pins ps/M_AXI_HPM0_FPD]   [get_bd_intf_pins sc_ctrl/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins sc_ctrl/[mi_pin 0]]  [get_bd_intf_pins dma_q/S_AXI_LITE]
connect_bd_intf_net [get_bd_intf_pins sc_ctrl/[mi_pin 1]]  [get_bd_intf_pins dma_k/S_AXI_LITE]
connect_bd_intf_net [get_bd_intf_pins sc_ctrl/[mi_pin 2]]  [get_bd_intf_pins dma_v/S_AXI_LITE]
connect_bd_intf_net [get_bd_intf_pins sc_ctrl/[mi_pin 3]]  [get_bd_intf_pins dma_mask/S_AXI_LITE]
connect_bd_intf_net [get_bd_intf_pins sc_ctrl/[mi_pin 4]]  [get_bd_intf_pins dma_o/S_AXI_LITE]
connect_bd_intf_net [get_bd_intf_pins sc_ctrl/[mi_pin 5]]  [get_bd_intf_pins kernel/S_AXI]

# ---- Memory paths -----------------------------------------------------------
# HP0: dma_q MM2S + dma_o S2MM + dma_mask MM2S (small streams). HP1: dma_k. HP2: dma_v.
create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:* sc_hp0
set_property -dict [list CONFIG.NUM_SI {3} CONFIG.NUM_MI {1}] [get_bd_cells sc_hp0]
connect_bd_net [get_bd_pins $fclk_pin]              [get_bd_pins sc_hp0/aclk]
connect_bd_net [get_bd_pins rst/peripheral_aresetn] [get_bd_pins sc_hp0/aresetn]
connect_bd_intf_net [get_bd_intf_pins dma_q/M_AXI_MM2S]    [get_bd_intf_pins sc_hp0/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins dma_o/M_AXI_S2MM]    [get_bd_intf_pins sc_hp0/S01_AXI]
connect_bd_intf_net [get_bd_intf_pins dma_mask/M_AXI_MM2S] [get_bd_intf_pins sc_hp0/S02_AXI]
connect_bd_intf_net [get_bd_intf_pins sc_hp0/M00_AXI]      [get_bd_intf_pins ps/S_AXI_HP0_FPD]

create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:* sc_hp1
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {1}] [get_bd_cells sc_hp1]
connect_bd_net [get_bd_pins $fclk_pin]              [get_bd_pins sc_hp1/aclk]
connect_bd_net [get_bd_pins rst/peripheral_aresetn] [get_bd_pins sc_hp1/aresetn]
connect_bd_intf_net [get_bd_intf_pins dma_k/M_AXI_MM2S] [get_bd_intf_pins sc_hp1/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins sc_hp1/M00_AXI]   [get_bd_intf_pins ps/S_AXI_HP1_FPD]

create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:* sc_hp2
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {1}] [get_bd_cells sc_hp2]
connect_bd_net [get_bd_pins $fclk_pin]              [get_bd_pins sc_hp2/aclk]
connect_bd_net [get_bd_pins rst/peripheral_aresetn] [get_bd_pins sc_hp2/aresetn]
connect_bd_intf_net [get_bd_intf_pins dma_v/M_AXI_MM2S] [get_bd_intf_pins sc_hp2/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins sc_hp2/M00_AXI]   [get_bd_intf_pins ps/S_AXI_HP2_FPD]

# ---- Address map (generated; single source: fpga/regmap/flash_attn.zig) -----
source [file normalize ./address_map.tcl]
foreach entry $flash_address_map {
    lassign $entry cell intf offset
    assign_bd_address -offset $offset -range 64K \
        [first_addr_seg [list "$cell/$intf/Reg" "$cell/$intf/*"]]
}
assign_bd_address

validate_bd_design
set kernel_clk_hz [clock_hz $fclk_pin $fclk_mhz]
puts "==> kernel CLK_HZ=$kernel_clk_hz"
set_property CONFIG.CLK_HZ $kernel_clk_hz [get_bd_cells kernel]
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
catch {write_hw_platform -fixed -force -file $outdir/$bit_name.xsa}
puts "==> Built timing-clean bitstream: $outdir/$bit_name.bit"
