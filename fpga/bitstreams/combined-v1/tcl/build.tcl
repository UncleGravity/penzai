# build.tcl - KR260 COMBINED bitstream: matmul (q1a8 v8 four-port) + flash (kv-major v2).
#
# Both PL ops in one design so a single bitstream serves decode end-to-end (matmul +
# attention on PL). They run sequentially in the graph, so their DDR feeds are kept
# independent rather than time-shared:
#
#   matmul  — UNCHANGED from q1a8-w256-mc: kernel_mm fed by dma_w0..w3 + dma_a over
#             HP0..HP3 on the faster weight clock (wclk); kernel + control on fclk.
#   flash   — UNCHANGED from flash-v1 (v2 kernel): kernel_fa fed by dma_q/k/v/mask/o,
#             single clock (fclk), over the coherent HPC0/HPC1 ports (free — matmul
#             owns HP0..3). Q/K/V/mask DMA straight from the resident tensors.
#
#   PS M_AXI_HPM0_FPD -> AXI-Lite -> 10 DMAs + both kernels (sc_ctrl, 12 MI, fclk)
#   matmul mem: HP0 = dma_w0 MM2S+S2MM + dma_a; HP1/2/3 = dma_w1/2/3   (wclk)
#   flash  mem: HPC0 = dma_q + dma_o + dma_mask; HPC1 = dma_k + dma_v  (fclk)
#
# Address map: matmul 0xA00x_0000, flash 0xA01x_0000 (non-colliding by design — the
# flash regmap was allocated in 0xA01x precisely for this). Both maps are generated
# (`zig build regmap`); this script sources both and remaps their "kernel" cell to
# kernel_mm / kernel_fa.

set variant [expr {$argc >= 1 ? [lindex $argv 0] : "w512-p4-f200-wc300"}]

# variant = w512-p4-f<MHz>-wc<MHz>. fclk drives BOTH kernels + all control; wclk is the
# matmul weight-feed clock (HP0..HP3 + their DMAs). flash runs entirely on fclk, so the
# shared fclk must be one the flash kernel closes at (validated f200; matmul closes higher).
if {![regexp {^w512-p4-f([0-9]+)-wc([0-9]+)$} $variant -> fclk_mhz wclk_mhz]} {
    error "unknown variant '$variant'; expected w512-p4-f<MHz>-wc<MHz>, e.g. w512-p4-f200-wc300"
}

set bit_prefix penzai-combined-v1
set bit_name   "$bit_prefix-$variant"
set proj       "combined_[string map {- _} $variant]"
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

# Both RTL sets live under ./rtl (build.sh syncs matmul/* + flash_attn/* + fp/* once).
set rtl_files [glob -nocomplain [file normalize ./rtl/*.v]]
if {[llength $rtl_files] == 0} { error "missing RTL files under ./rtl" }
add_files -norecurse $rtl_files
# Verilog headers: matmul_top includes matmul_regs.vh; flash_top includes flash_regs.vh;
# fp_exp/fp_recip include flash_luts.vh; matmul reducer/rowblock include fmt.vh (the
# numeric format+latency contract). All must be added as headers + on the include path.
foreach vh {matmul_regs.vh flash_regs.vh flash_luts.vh fmt.vh} {
    set f [file normalize ./rtl/$vh]
    if {![file exists $f]} { error "missing ./rtl/$vh (run 'zig build regmap' for the *_regs.vh)" }
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

# GP0/GP1 = coherent HPC0/HPC1 (flash). GP2..GP5 = HP0..HP3 (matmul weights). All 128-bit.
set_property -dict [list \
    CONFIG.PSU__USE__M_AXI_GP0 {1} \
    CONFIG.PSU__USE__M_AXI_GP1 {0} \
    CONFIG.PSU__USE__S_AXI_GP0 {1} \
    CONFIG.PSU__USE__S_AXI_GP1 {1} \
    CONFIG.PSU__USE__S_AXI_GP2 {1} \
    CONFIG.PSU__USE__S_AXI_GP3 {1} \
    CONFIG.PSU__USE__S_AXI_GP4 {1} \
    CONFIG.PSU__USE__S_AXI_GP5 {1} \
    CONFIG.PSU__SAXIGP0__DATA_WIDTH {128} \
    CONFIG.PSU__SAXIGP1__DATA_WIDTH {128} \
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

# KR260 fan gate on TTC0 channel 2.
create_bd_cell -type ip -vlnv xilinx.com:ip:xlslice:* fan_ttc0_ch2
set_property -dict [list CONFIG.DIN_WIDTH {3} CONFIG.DIN_FROM {2} CONFIG.DIN_TO {2}] \
    [get_bd_cells fan_ttc0_ch2]
create_bd_port -dir O fan_en_b
connect_bd_net [get_bd_pins ps/emio_ttc0_wave_o] [get_bd_pins fan_ttc0_ch2/Din]
connect_bd_net [get_bd_pins fan_ttc0_ch2/Dout]   [get_bd_ports fan_en_b]

# ---- Fabric clocks ----------------------------------------------------------
# clk_out1 = fclk (both kernels + all control + flash datapath). clk_out2 = wclk
# (matmul weight feed only). Same VCO; AXIS clock converters handle matmul's CDC.
create_bd_cell -type ip -vlnv xilinx.com:ip:clk_wiz:* clk_wiz
set_property -dict [list \
    CONFIG.PRIM_IN_FREQ $ps_fclk_mhz \
    CONFIG.CLKOUT1_REQUESTED_OUT_FREQ $fclk_mhz \
    CONFIG.USE_LOCKED {true} \
    CONFIG.USE_RESET {false} \
    CONFIG.CLKOUT2_USED {true} \
    CONFIG.CLKOUT2_REQUESTED_OUT_FREQ $wclk_mhz \
] [get_bd_cells clk_wiz]
connect_bd_net [get_bd_pins ps/pl_clk0] [get_bd_pins clk_wiz/clk_in1]
set fclk_pin "clk_wiz/clk_out1"
set wclk_pin "clk_wiz/clk_out2"

# ---- DMA / converter helpers ------------------------------------------------
# 128-bit mem + stream (the proven width; a 64-bit DMA silently corrupted data).
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
proc make_dwc {name s_bytes m_bytes} {
    create_bd_cell -type ip -vlnv xilinx.com:ip:axis_dwidth_converter:* $name
    set_property -dict [list \
        CONFIG.S_TDATA_NUM_BYTES $s_bytes \
        CONFIG.M_TDATA_NUM_BYTES $m_bytes \
        CONFIG.HAS_TLAST {1} \
        CONFIG.HAS_TKEEP {1} \
    ] [get_bd_cells $name]
}
proc make_clk_conv {name bytes} {
    create_bd_cell -type ip -vlnv xilinx.com:ip:axis_clock_converter:* $name
    set_property -dict [list \
        CONFIG.TDATA_NUM_BYTES $bytes \
        CONFIG.HAS_TLAST {1} \
        CONFIG.HAS_TKEEP {1} \
    ] [get_bd_cells $name]
}

# ====================== MATMUL section (HP0..3, wclk) ========================
make_dma dma_w0 1 1
make_dma dma_w1 1 0
make_dma dma_w2 1 0
make_dma dma_w3 1 0
make_dma dma_a  1 0
# Matmul stream DMAs are async: data movers on wclk, AXI-Lite control on fclk.
foreach dma {dma_w0 dma_w1 dma_w2 dma_w3 dma_a} {
    set_property CONFIG.c_prmry_is_aclk_async {1} [get_bd_cells $dma]
}
make_dwc dwc_a 16 8 ;# acts    128 -> 64
make_dwc dwc_r 8 16 ;# results  64 -> 128
create_bd_cell -type module -reference decode_top kernel_mm ;# plan-7 fixed-point gemm (was matmul_top)

# Acts: dma_a(wclk) -> 128->64(wclk) -> clk_conv -> kernel_mm(fclk).
make_clk_conv clk_conv_a 8
connect_bd_intf_net [get_bd_intf_pins dma_a/M_AXIS_MM2S] [get_bd_intf_pins dwc_a/S_AXIS]
connect_bd_intf_net [get_bd_intf_pins dwc_a/M_AXIS]      [get_bd_intf_pins clk_conv_a/S_AXIS]
connect_bd_intf_net [get_bd_intf_pins clk_conv_a/M_AXIS] [get_bd_intf_pins kernel_mm/S_AXIS_ACTS]

# Weight ports: dma_wN(wclk) -> clk_conv -> kernel_mm(fclk).
make_clk_conv clk_conv_w0 16
make_clk_conv clk_conv_w1 16
make_clk_conv clk_conv_w2 16
make_clk_conv clk_conv_w3 16
connect_bd_intf_net [get_bd_intf_pins dma_w0/M_AXIS_MM2S] [get_bd_intf_pins clk_conv_w0/S_AXIS]
connect_bd_intf_net [get_bd_intf_pins clk_conv_w0/M_AXIS] [get_bd_intf_pins kernel_mm/S_AXIS_W0]
connect_bd_intf_net [get_bd_intf_pins dma_w1/M_AXIS_MM2S] [get_bd_intf_pins clk_conv_w1/S_AXIS]
connect_bd_intf_net [get_bd_intf_pins clk_conv_w1/M_AXIS] [get_bd_intf_pins kernel_mm/S_AXIS_W1]
connect_bd_intf_net [get_bd_intf_pins dma_w2/M_AXIS_MM2S] [get_bd_intf_pins clk_conv_w2/S_AXIS]
connect_bd_intf_net [get_bd_intf_pins clk_conv_w2/M_AXIS] [get_bd_intf_pins kernel_mm/S_AXIS_W2]
connect_bd_intf_net [get_bd_intf_pins dma_w3/M_AXIS_MM2S] [get_bd_intf_pins clk_conv_w3/S_AXIS]
connect_bd_intf_net [get_bd_intf_pins clk_conv_w3/M_AXIS] [get_bd_intf_pins kernel_mm/S_AXIS_W3]

# Results: kernel_mm.M_AXIS(fclk) -> dwc_r 64->128(fclk) -> clk_conv -> dma_w0.S2MM(wclk).
make_clk_conv clk_conv_r 16
connect_bd_intf_net [get_bd_intf_pins kernel_mm/M_AXIS]  [get_bd_intf_pins dwc_r/S_AXIS]
connect_bd_intf_net [get_bd_intf_pins dwc_r/M_AXIS]      [get_bd_intf_pins clk_conv_r/S_AXIS]
connect_bd_intf_net [get_bd_intf_pins clk_conv_r/M_AXIS] [get_bd_intf_pins dma_w0/S_AXIS_S2MM]

# ====================== FLASH section (HPC0/1, fclk) =========================
make_dma dma_q    1 0
make_dma dma_k    1 0
make_dma dma_v    1 0
make_dma dma_mask 1 0
make_dma dma_o    0 1
make_dwc dwc_q 16 32  ;# Q     128 -> 256
make_dwc dwc_mask 16 2 ;# mask 128 -> 16
make_dwc dwc_o 32 16  ;# O     256 -> 128
create_bd_cell -type module -reference flash_top kernel_fa

# Single clock (fclk), no clock converters.
connect_bd_intf_net [get_bd_intf_pins dma_q/M_AXIS_MM2S]    [get_bd_intf_pins dwc_q/S_AXIS]
connect_bd_intf_net [get_bd_intf_pins dwc_q/M_AXIS]         [get_bd_intf_pins kernel_fa/S_AXIS_Q]
connect_bd_intf_net [get_bd_intf_pins dma_k/M_AXIS_MM2S]    [get_bd_intf_pins kernel_fa/S_AXIS_K]
connect_bd_intf_net [get_bd_intf_pins dma_v/M_AXIS_MM2S]    [get_bd_intf_pins kernel_fa/S_AXIS_V]
connect_bd_intf_net [get_bd_intf_pins dma_mask/M_AXIS_MM2S] [get_bd_intf_pins dwc_mask/S_AXIS]
connect_bd_intf_net [get_bd_intf_pins dwc_mask/M_AXIS]      [get_bd_intf_pins kernel_fa/S_AXIS_MASK]
connect_bd_intf_net [get_bd_intf_pins kernel_fa/M_AXIS_O]   [get_bd_intf_pins dwc_o/S_AXIS]
connect_bd_intf_net [get_bd_intf_pins dwc_o/M_AXIS]         [get_bd_intf_pins dma_o/S_AXIS_S2MM]

# ---- Resets -----------------------------------------------------------------
# rst on fclk (control + both kernels + flash); rst_fast on wclk (matmul weight feed).
create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:* rst
set_property -dict [list CONFIG.C_EXT_RESET_HIGH {0}] [get_bd_cells rst]
connect_bd_net [get_bd_pins $fclk_pin]      [get_bd_pins rst/slowest_sync_clk]
connect_bd_net [get_bd_pins ps/pl_resetn0]  [get_bd_pins rst/ext_reset_in]
connect_bd_net [get_bd_pins clk_wiz/locked] [get_bd_pins rst/dcm_locked]

create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:* rst_fast
set_property -dict [list CONFIG.C_EXT_RESET_HIGH {0}] [get_bd_cells rst_fast]
connect_bd_net [get_bd_pins $wclk_pin]      [get_bd_pins rst_fast/slowest_sync_clk]
connect_bd_net [get_bd_pins ps/pl_resetn0]  [get_bd_pins rst_fast/ext_reset_in]
connect_bd_net [get_bd_pins clk_wiz/locked] [get_bd_pins rst_fast/dcm_locked]
set wrst_pin "rst_fast/peripheral_aresetn"

# ---- Clock fan-out ----------------------------------------------------------
# fclk: HPM0 control, both kernels, all AXI-Lite, matmul result upsizer, ALL flash.
foreach clkpin {
    ps/maxihpm0_fpd_aclk
    ps/saxihpc0_fpd_aclk ps/saxihpc1_fpd_aclk
    dma_w0/s_axi_lite_aclk dma_w1/s_axi_lite_aclk dma_w2/s_axi_lite_aclk dma_w3/s_axi_lite_aclk
    dma_a/s_axi_lite_aclk
    dwc_r/aclk kernel_mm/s_axi_aclk
    dma_q/s_axi_lite_aclk dma_k/s_axi_lite_aclk dma_v/s_axi_lite_aclk
    dma_mask/s_axi_lite_aclk dma_o/s_axi_lite_aclk
    dma_q/m_axi_mm2s_aclk dma_k/m_axi_mm2s_aclk dma_v/m_axi_mm2s_aclk dma_mask/m_axi_mm2s_aclk
    dma_o/m_axi_s2mm_aclk
    dwc_q/aclk dwc_mask/aclk dwc_o/aclk kernel_fa/s_axi_aclk
} { connect_bd_net [get_bd_pins $fclk_pin] [get_bd_pins $clkpin] }

# wclk: HP0..HP3 + matmul stream-DMA data movers + acts downsizer.
foreach clkpin {
    ps/saxihp0_fpd_aclk ps/saxihp1_fpd_aclk ps/saxihp2_fpd_aclk ps/saxihp3_fpd_aclk
    dma_w0/m_axi_mm2s_aclk dma_w0/m_axi_s2mm_aclk
    dma_w1/m_axi_mm2s_aclk dma_w2/m_axi_mm2s_aclk dma_w3/m_axi_mm2s_aclk
    dma_a/m_axi_mm2s_aclk
    dwc_a/aclk
} { connect_bd_net [get_bd_pins $wclk_pin] [get_bd_pins $clkpin] }

# Reset fan-out. fclk-domain resets:
foreach rstpin {
    dma_w0/axi_resetn dma_w1/axi_resetn dma_w2/axi_resetn dma_w3/axi_resetn dma_a/axi_resetn
    dwc_r/aresetn kernel_mm/s_axi_aresetn
    dma_q/axi_resetn dma_k/axi_resetn dma_v/axi_resetn dma_mask/axi_resetn dma_o/axi_resetn
    dwc_q/aresetn dwc_mask/aresetn dwc_o/aresetn kernel_fa/s_axi_aresetn
} { connect_bd_net [get_bd_pins rst/peripheral_aresetn] [get_bd_pins $rstpin] }
connect_bd_net [get_bd_pins $wrst_pin] [get_bd_pins dwc_a/aresetn]

# Matmul clock converters straddle wclk(slave)->fclk(master); result is fclk->wclk.
foreach conv {clk_conv_w0 clk_conv_w1 clk_conv_w2 clk_conv_w3 clk_conv_a} {
    connect_bd_net [get_bd_pins $wclk_pin]              [get_bd_pins $conv/s_axis_aclk]
    connect_bd_net [get_bd_pins $wrst_pin]              [get_bd_pins $conv/s_axis_aresetn]
    connect_bd_net [get_bd_pins $fclk_pin]              [get_bd_pins $conv/m_axis_aclk]
    connect_bd_net [get_bd_pins rst/peripheral_aresetn] [get_bd_pins $conv/m_axis_aresetn]
}
connect_bd_net [get_bd_pins $fclk_pin]              [get_bd_pins clk_conv_r/s_axis_aclk]
connect_bd_net [get_bd_pins rst/peripheral_aresetn] [get_bd_pins clk_conv_r/s_axis_aresetn]
connect_bd_net [get_bd_pins $wclk_pin]              [get_bd_pins clk_conv_r/m_axis_aclk]
connect_bd_net [get_bd_pins $wrst_pin]              [get_bd_pins clk_conv_r/m_axis_aresetn]

# ---- Control path: PS -> 10 DMAs + 2 kernels AXI-Lite (fclk) ----------------
create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:* sc_ctrl
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {12}] [get_bd_cells sc_ctrl]
connect_bd_net [get_bd_pins $fclk_pin]              [get_bd_pins sc_ctrl/aclk]
connect_bd_net [get_bd_pins rst/peripheral_aresetn] [get_bd_pins sc_ctrl/aresetn]
connect_bd_intf_net [get_bd_intf_pins ps/M_AXI_HPM0_FPD] [get_bd_intf_pins sc_ctrl/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins sc_ctrl/[mi_pin 0]]  [get_bd_intf_pins dma_w0/S_AXI_LITE]
connect_bd_intf_net [get_bd_intf_pins sc_ctrl/[mi_pin 1]]  [get_bd_intf_pins dma_w1/S_AXI_LITE]
connect_bd_intf_net [get_bd_intf_pins sc_ctrl/[mi_pin 2]]  [get_bd_intf_pins dma_w2/S_AXI_LITE]
connect_bd_intf_net [get_bd_intf_pins sc_ctrl/[mi_pin 3]]  [get_bd_intf_pins dma_w3/S_AXI_LITE]
connect_bd_intf_net [get_bd_intf_pins sc_ctrl/[mi_pin 4]]  [get_bd_intf_pins dma_a/S_AXI_LITE]
connect_bd_intf_net [get_bd_intf_pins sc_ctrl/[mi_pin 5]]  [get_bd_intf_pins kernel_mm/S_AXI]
connect_bd_intf_net [get_bd_intf_pins sc_ctrl/[mi_pin 6]]  [get_bd_intf_pins dma_q/S_AXI_LITE]
connect_bd_intf_net [get_bd_intf_pins sc_ctrl/[mi_pin 7]]  [get_bd_intf_pins dma_k/S_AXI_LITE]
connect_bd_intf_net [get_bd_intf_pins sc_ctrl/[mi_pin 8]]  [get_bd_intf_pins dma_v/S_AXI_LITE]
connect_bd_intf_net [get_bd_intf_pins sc_ctrl/[mi_pin 9]]  [get_bd_intf_pins dma_mask/S_AXI_LITE]
connect_bd_intf_net [get_bd_intf_pins sc_ctrl/[mi_pin 10]] [get_bd_intf_pins dma_o/S_AXI_LITE]
connect_bd_intf_net [get_bd_intf_pins sc_ctrl/[mi_pin 11]] [get_bd_intf_pins kernel_fa/S_AXI]

# ---- Memory paths -----------------------------------------------------------
# Matmul: HP0 = dma_w0 MM2S+S2MM + dma_a MM2S; HP1/2/3 = dma_w1/2/3 (all wclk).
create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:* sc_hp0
set_property -dict [list CONFIG.NUM_SI {3} CONFIG.NUM_MI {1}] [get_bd_cells sc_hp0]
connect_bd_net [get_bd_pins $wclk_pin] [get_bd_pins sc_hp0/aclk]
connect_bd_net [get_bd_pins $wrst_pin] [get_bd_pins sc_hp0/aresetn]
connect_bd_intf_net [get_bd_intf_pins dma_w0/M_AXI_MM2S] [get_bd_intf_pins sc_hp0/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins dma_w0/M_AXI_S2MM] [get_bd_intf_pins sc_hp0/S01_AXI]
connect_bd_intf_net [get_bd_intf_pins dma_a/M_AXI_MM2S]  [get_bd_intf_pins sc_hp0/S02_AXI]
connect_bd_intf_net [get_bd_intf_pins sc_hp0/M00_AXI]    [get_bd_intf_pins ps/S_AXI_HP0_FPD]

foreach {sc dma hp} {sc_hp1 dma_w1 S_AXI_HP1_FPD  sc_hp2 dma_w2 S_AXI_HP2_FPD  sc_hp3 dma_w3 S_AXI_HP3_FPD} {
    create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:* $sc
    set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {1}] [get_bd_cells $sc]
    connect_bd_net [get_bd_pins $wclk_pin] [get_bd_pins $sc/aclk]
    connect_bd_net [get_bd_pins $wrst_pin] [get_bd_pins $sc/aresetn]
    connect_bd_intf_net [get_bd_intf_pins $dma/M_AXI_MM2S] [get_bd_intf_pins $sc/S00_AXI]
    connect_bd_intf_net [get_bd_intf_pins $sc/M00_AXI]     [get_bd_intf_pins ps/$hp]
}

# Flash: HPC0 = dma_q MM2S + dma_o S2MM + dma_mask MM2S; HPC1 = dma_k + dma_v (all fclk).
create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:* sc_hpc0
set_property -dict [list CONFIG.NUM_SI {3} CONFIG.NUM_MI {1}] [get_bd_cells sc_hpc0]
connect_bd_net [get_bd_pins $fclk_pin]              [get_bd_pins sc_hpc0/aclk]
connect_bd_net [get_bd_pins rst/peripheral_aresetn] [get_bd_pins sc_hpc0/aresetn]
connect_bd_intf_net [get_bd_intf_pins dma_q/M_AXI_MM2S]    [get_bd_intf_pins sc_hpc0/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins dma_o/M_AXI_S2MM]    [get_bd_intf_pins sc_hpc0/S01_AXI]
connect_bd_intf_net [get_bd_intf_pins dma_mask/M_AXI_MM2S] [get_bd_intf_pins sc_hpc0/S02_AXI]
connect_bd_intf_net [get_bd_intf_pins sc_hpc0/M00_AXI]     [get_bd_intf_pins ps/S_AXI_HPC0_FPD]

create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:* sc_hpc1
set_property -dict [list CONFIG.NUM_SI {2} CONFIG.NUM_MI {1}] [get_bd_cells sc_hpc1]
connect_bd_net [get_bd_pins $fclk_pin]              [get_bd_pins sc_hpc1/aclk]
connect_bd_net [get_bd_pins rst/peripheral_aresetn] [get_bd_pins sc_hpc1/aresetn]
connect_bd_intf_net [get_bd_intf_pins dma_k/M_AXI_MM2S] [get_bd_intf_pins sc_hpc1/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins dma_v/M_AXI_MM2S] [get_bd_intf_pins sc_hpc1/S01_AXI]
connect_bd_intf_net [get_bd_intf_pins sc_hpc1/M00_AXI]  [get_bd_intf_pins ps/S_AXI_HPC1_FPD]

# ---- Address map (generated; single source: fpga/regmap/{matmul,flash_attn}.zig) ----
# build.sh copies the two generated maps here (renamed). Each defines its own list;
# the "kernel" cell is remapped to this design's kernel_mm / kernel_fa.
source [file normalize ./matmul_address_map.tcl] ;# -> $matmul_address_map
source [file normalize ./flash_address_map.tcl]  ;# -> $flash_address_map
foreach {addr_list kcell} [list $matmul_address_map kernel_mm $flash_address_map kernel_fa] {
    foreach entry $addr_list {
        lassign $entry cell intf offset
        if {$cell eq "kernel"} { set cell $kcell }
        assign_bd_address -offset $offset -range 64K \
            [first_addr_seg [list "$cell/$intf/Reg" "$cell/$intf/*"]]
    }
}
# Route the DMA mem masters to PS DDR through their HP/HPC ports.
assign_bd_address

validate_bd_design
set kernel_clk_hz [clock_hz $fclk_pin $fclk_mhz]
puts "==> kernel CLK_HZ=$kernel_clk_hz (both kernels on fclk)"
set_property CONFIG.CLK_HZ $kernel_clk_hz [get_bd_cells kernel_mm]
set_property CONFIG.CLK_HZ $kernel_clk_hz [get_bd_cells kernel_fa]
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

# Post-synth resource utilization (the combined-fit question) — logged before the long
# impl so an over-map is visible early. Best-effort: never let it abort the build.
if {![catch {open_run synth_1 -name synth_1} err]} {
    catch {report_utilization -file $outdir/${bit_name}_utilization_synth.rpt}
    puts "==> wrote $outdir/${bit_name}_utilization_synth.rpt (check DSP/LUT/BRAM fit)"
    catch {close_design}
} else {
    puts "WARNING: could not open synth_1 for utilization report: $err"
}

launch_runs impl_1 -to_step route_design -jobs 4
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] != "100%"} { error "implementation failed" }

open_run impl_1
catch {report_utilization -file $outdir/${bit_name}_utilization_routed.rpt}
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
