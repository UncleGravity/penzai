# build.tcl - KR260 COMBINED bitstream: dual-format matmul (v11 four-port) + flash (kv-major v2).
#
# Both PL ops in one design so a single bitstream serves decode end-to-end (matmul +
# attention on PL). They run sequentially in the graph, so their DDR feeds are kept
# independent rather than time-shared:
#
#   matmul  — kernel_mm fed by dma_w0..w3 + dma_a over HP0..HP3.
#   flash   — kernel_fa fed by dma_q/k/v/mask/o over the coherent HPC0/HPC1 ports (free —
#             matmul owns HP0..3). Q/K/V/mask DMA straight from the resident tensors.
#
#   SINGLE CLOCK (fclk): both kernels + all control + all 10 DMAs + all width/data movers.
#   The dual-clock CDC (a second wclk domain + 6 axis_clock_converters + rst_fast) was
#   deleted because the two domains ran at the same frequency, making the CDC pure
#   congestion/skew tax (plan-fpga-7 Part 6).
#
#   PS M_AXI_HPM0_FPD -> AXI-Lite -> 10 DMAs + both kernels (sc_ctrl, 12 MI, fclk)
#   matmul mem: HP0 = dma_w0 MM2S+S2MM + dma_a; HP1/2/3 = dma_w1/2/3   (fclk)
#   flash  mem: HPC0 = dma_q + dma_o + dma_mask; HPC1 = dma_k + dma_v  (fclk)
#
# Address map: matmul 0xA00x_0000, flash 0xA01x_0000 (non-colliding by design — the
# flash regmap was allocated in 0xA01x precisely for this). Both maps are generated
# (`zig build regmap`); this script sources both and remaps their "kernel" cell to
# kernel_mm / kernel_fa.

set_param general.maxThreads 8
puts "==> Vivado general.maxThreads=[get_param general.maxThreads]"

set variant    [expr {$argc >= 1 ? [lindex $argv 0] : "w512-p4-f300"}]
set build_mode [expr {$argc >= 2 ? [lindex $argv 1] : "clean"}]
set run_id     [expr {$argc >= 3 ? [lindex $argv 2] : "manual-$variant"}]
set git_commit [expr {$argc >= 4 ? [lindex $argv 3] : "unknown"}]
set git_dirty  [expr {$argc >= 5 ? [lindex $argv 4] : "unknown"}]
set source_hash [expr {$argc >= 6 ? [lindex $argv 5] : "unknown"}]

# variant = w512-p4-f<MHz>. fclk drives both kernels, all control, and all data movement.
if {![regexp {^w512-p4-f([0-9]+)$} $variant -> fclk_mhz]} {
    error "unknown variant '$variant'; expected w512-p4-f<MHz>, e.g. w512-p4-f300"
}
if {$build_mode ni {clean incremental}} {
    error "unknown build mode '$build_mode'; expected clean or incremental"
}
if {![regexp {^[A-Za-z0-9._-]+$} $run_id]} {
    error "invalid run ID '$run_id'"
}

set bit_prefix penzai-combined-v1
set bit_name   "$bit_prefix-$variant"
set proj       "combined_[string map {- _} $variant]"
set bd         design_1
set part       xck26-sfvc784-2LV-c
set board      xilinx.com:kr260_som:part0:1.1
set run_root   [file normalize ./out/runs]
set outdir     [file join $run_root $run_id]
set cache_root [file normalize ./cache]
set ip_cache   [file join $cache_root ip]
set checkpoint_dir [file join $cache_root checkpoints]
set reference_dcp [file join $checkpoint_dir ${bit_name}-routed.dcp]
set ps_fclk_mhz 99.999001

file delete -force $outdir [file normalize ./$proj]
file mkdir $outdir $ip_cache $checkpoint_dir

source [file normalize ./metrics.tcl]
::penzai_analysis::init $outdir [dict create \
    run_id $run_id \
    variant $variant \
    build_mode $build_mode \
    requested_fclk_mhz $fclk_mhz \
    git_commit $git_commit \
    git_dirty $git_dirty \
    source_sha256 $source_hash \
    vivado_version [version -short] \
    part $part \
    board $board \
    max_threads [get_param general.maxThreads] \
    place_directive AltSpreadLogic_high \
    route_directive Explore]

set build_started [clock milliseconds]
set phase_times {}
proc record_phase {name started_ms} {
    global phase_times outdir build_started
    set elapsed_s [expr {([clock milliseconds] - $started_ms) / 1000.0}]
    lappend phase_times [list $name $elapsed_s]
    ::penzai_analysis::metric_set build_seconds.$name [format "%.1f" $elapsed_s]
    ::penzai_analysis::metric_set build_seconds.total_so_far \
        [format "%.1f" [expr {([clock milliseconds] - $build_started) / 1000.0}]]
    set report [file join $outdir build_times.rpt]
    set fh [open $report w]
    puts $fh "phase seconds"
    foreach phase $phase_times {
        puts $fh [format "%-28s %.1f" [lindex $phase 0] [lindex $phase 1]]
    }
    puts $fh [format "%-28s %.1f" total_so_far \
        [expr {([clock milliseconds] - $build_started) / 1000.0}]]
    close $fh
    puts [format "==> phase %-22s %.1fs" $name $elapsed_s]
}

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
# The project is intentionally rebuilt from its Tcl source, but generated AMD IP and a
# timing-clean routed checkpoint persist outside the project directory across builds.
config_ip_cache -use_cache_location $ip_cache
puts "==> persistent IP cache: $ip_cache"
puts "==> build mode: $build_mode"
puts "==> run ID: $run_id"
if {[catch {set_property board_part $board [current_project]} err]} {
    puts "WARNING: could not set board_part '$board': $err"
}
set_property target_language Verilog [current_project]

# Both RTL sets and their generated headers are flattened under ./rtl by build.sh.
set rtl_files [glob -nocomplain [file normalize ./rtl/*.v]]
if {[llength $rtl_files] == 0} { error "missing RTL files under ./rtl" }
add_files -norecurse $rtl_files
# Verilog headers: decode_top and flash_top include their generated register contracts;
# numeric exp/recip include flash_luts.vh; numeric leaves include fmt.vh. All must be
# added as headers and placed on the include path.
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

# ---- Fabric clock (SINGLE domain) -------------------------------------------
# ONE clock (fclk) drives both kernels + all control + all data movement. The dual-clock
# CDC apparatus is gone: both domains ran at 300 MHz, so the 6
# axis_clock_converters + second clock tree + rst_fast were pure congestion/skew tax with
# zero functional benefit (plan-fpga-7 Part 6 §"Collapses with the single-clock phase").
create_bd_cell -type ip -vlnv xilinx.com:ip:clk_wiz:* clk_wiz
set_property -dict [list \
    CONFIG.PRIM_IN_FREQ $ps_fclk_mhz \
    CONFIG.CLKOUT1_REQUESTED_OUT_FREQ $fclk_mhz \
    CONFIG.USE_LOCKED {true} \
    CONFIG.USE_RESET {false} \
] [get_bd_cells clk_wiz]
connect_bd_net [get_bd_pins ps/pl_clk0] [get_bd_pins clk_wiz/clk_in1]
set fclk_pin "clk_wiz/clk_out1"

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
# Memory-side SmartConnect: $si_list DMA mem masters -> one PS HP/HPC port. With the single
# clock the HP (matmul-weight) and HPC (flash) feeds are structurally identical, so one helper
# serves both (before the CDC collapse, HP was wclk and HPC fclk — they could not share this).
proc make_mem_sc {name si_list ps_port} {
    global fclk_pin
    create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:* $name
    set_property -dict [list CONFIG.NUM_SI [llength $si_list] CONFIG.NUM_MI {1}] [get_bd_cells $name]
    connect_bd_net [get_bd_pins $fclk_pin]              [get_bd_pins $name/aclk]
    connect_bd_net [get_bd_pins rst/peripheral_aresetn] [get_bd_pins $name/aresetn]
    set i 0
    foreach si $si_list {
        connect_bd_intf_net [get_bd_intf_pins $si] [get_bd_intf_pins [format "$name/S%02d_AXI" $i]]
        incr i
    }
    connect_bd_intf_net [get_bd_intf_pins $name/M00_AXI] [get_bd_intf_pins ps/$ps_port]
}

# ====================== MATMUL section (HP0..3, fclk) ========================
make_dma dma_w0 1 1
make_dma dma_w1 1 0
make_dma dma_w2 1 0
make_dma dma_w3 1 0
make_dma dma_a  1 0
# Single clock: matmul stream DMAs are SYNCHRONOUS (data movers + AXI-Lite both on fclk).
# (was: c_prmry_is_aclk_async + 6 axis_clock_converters straddling wclk->fclk — all deleted)
make_dwc dwc_a 16 8 ;# acts    128 -> 64
make_dwc dwc_r 8 16 ;# results  64 -> 128
create_bd_cell -type module -reference decode_top kernel_mm ;# plan-7 fixed-point gemm (was matmul_top)

# Acts: dma_a -> 128->64 -> kernel_mm  (all fclk, no clock converter).
connect_bd_intf_net [get_bd_intf_pins dma_a/M_AXIS_MM2S] [get_bd_intf_pins dwc_a/S_AXIS]
connect_bd_intf_net [get_bd_intf_pins dwc_a/M_AXIS]      [get_bd_intf_pins kernel_mm/S_AXIS_ACTS]

# Weight ports: dma_wN -> kernel_mm  (all fclk, no clock converter).
connect_bd_intf_net [get_bd_intf_pins dma_w0/M_AXIS_MM2S] [get_bd_intf_pins kernel_mm/S_AXIS_W0]
connect_bd_intf_net [get_bd_intf_pins dma_w1/M_AXIS_MM2S] [get_bd_intf_pins kernel_mm/S_AXIS_W1]
connect_bd_intf_net [get_bd_intf_pins dma_w2/M_AXIS_MM2S] [get_bd_intf_pins kernel_mm/S_AXIS_W2]
connect_bd_intf_net [get_bd_intf_pins dma_w3/M_AXIS_MM2S] [get_bd_intf_pins kernel_mm/S_AXIS_W3]

# Results: kernel_mm.M_AXIS -> dwc_r 64->128 -> dma_w0.S2MM  (all fclk, no clock converter).
connect_bd_intf_net [get_bd_intf_pins kernel_mm/M_AXIS]  [get_bd_intf_pins dwc_r/S_AXIS]
connect_bd_intf_net [get_bd_intf_pins dwc_r/M_AXIS]      [get_bd_intf_pins dma_w0/S_AXIS_S2MM]

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

# ====================== SEQ.V command executor (v2.1) =======================
# seq_top batches the per-op PS->PL register dance: the PS loads a {WRITE|WAIT} command stream
# into seq_top's OWN BRAM through its control slave (S_AXI, a 13th sc_ctrl target), kicks it once,
# and seq.v replays the stream into sc_ctrl (M_AXI_REG, a 2nd sc_ctrl master). No DRAM, no fetch
# master, no cache coherency — v1's brick class is unrepresentable. (docs/plan-seq-impl-v2.md)
create_bd_cell -type module -reference seq_top seq_top

# ---- Reset (SINGLE domain) --------------------------------------------------
# One reset (rst, on fclk) for everything. (was: a second rst_fast on wclk for the matmul
# weight feed — deleted with the CDC.)
create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:* rst
set_property -dict [list CONFIG.C_EXT_RESET_HIGH {0}] [get_bd_cells rst]
connect_bd_net [get_bd_pins $fclk_pin]      [get_bd_pins rst/slowest_sync_clk]
connect_bd_net [get_bd_pins ps/pl_resetn0]  [get_bd_pins rst/ext_reset_in]
connect_bd_net [get_bd_pins clk_wiz/locked] [get_bd_pins rst/dcm_locked]

# ---- Clock fan-out (SINGLE clock) -------------------------------------------
# fclk drives EVERYTHING: HPM0 control, all 10 DMAs (AXI-Lite + data movers), both kernels,
# all width converters, all flash. The HP0..HP3 weight-feed movers + acts downsizer that
# used to be on wclk are now on fclk (synchronous — no CDC).
foreach clkpin {
    ps/maxihpm0_fpd_aclk
    ps/saxihpc0_fpd_aclk ps/saxihpc1_fpd_aclk
    ps/saxihp0_fpd_aclk ps/saxihp1_fpd_aclk ps/saxihp2_fpd_aclk ps/saxihp3_fpd_aclk
    dma_w0/s_axi_lite_aclk dma_w1/s_axi_lite_aclk dma_w2/s_axi_lite_aclk dma_w3/s_axi_lite_aclk
    dma_a/s_axi_lite_aclk
    dma_w0/m_axi_mm2s_aclk dma_w0/m_axi_s2mm_aclk
    dma_w1/m_axi_mm2s_aclk dma_w2/m_axi_mm2s_aclk dma_w3/m_axi_mm2s_aclk
    dma_a/m_axi_mm2s_aclk
    dwc_a/aclk dwc_r/aclk kernel_mm/s_axi_aclk
    dma_q/s_axi_lite_aclk dma_k/s_axi_lite_aclk dma_v/s_axi_lite_aclk
    dma_mask/s_axi_lite_aclk dma_o/s_axi_lite_aclk
    dma_q/m_axi_mm2s_aclk dma_k/m_axi_mm2s_aclk dma_v/m_axi_mm2s_aclk dma_mask/m_axi_mm2s_aclk
    dma_o/m_axi_s2mm_aclk
    dwc_q/aclk dwc_mask/aclk dwc_o/aclk kernel_fa/s_axi_aclk
    seq_top/clk
} { connect_bd_net [get_bd_pins $fclk_pin] [get_bd_pins $clkpin] }

# ---- Reset fan-out (SINGLE reset) -------------------------------------------
foreach rstpin {
    dma_w0/axi_resetn dma_w1/axi_resetn dma_w2/axi_resetn dma_w3/axi_resetn dma_a/axi_resetn
    dwc_a/aresetn dwc_r/aresetn kernel_mm/s_axi_aresetn
    dma_q/axi_resetn dma_k/axi_resetn dma_v/axi_resetn dma_mask/axi_resetn dma_o/axi_resetn
    dwc_q/aresetn dwc_mask/aresetn dwc_o/aresetn kernel_fa/s_axi_aresetn
    seq_top/rst_n
} { connect_bd_net [get_bd_pins rst/peripheral_aresetn] [get_bd_pins $rstpin] }

# ---- Control path: PS -> 10 DMAs + 2 kernels AXI-Lite (fclk) ----------------
# TWO masters now: PS (S00) for setup/teardown + seq.v's replay master (S01). They never run
# concurrently (PS programs -> go -> seq.v runs -> PS polls done), so SmartConnect arbitration is
# free. 13 targets: the 10 DMAs + 2 kernels + seq_top's own control slave (MI 12).
create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:* sc_ctrl
set_property -dict [list CONFIG.NUM_SI {2} CONFIG.NUM_MI {13}] [get_bd_cells sc_ctrl]
connect_bd_net [get_bd_pins $fclk_pin]              [get_bd_pins sc_ctrl/aclk]
connect_bd_net [get_bd_pins rst/peripheral_aresetn] [get_bd_pins sc_ctrl/aresetn]
connect_bd_intf_net [get_bd_intf_pins ps/M_AXI_HPM0_FPD] [get_bd_intf_pins sc_ctrl/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins seq_top/M_AXI_REG] [get_bd_intf_pins sc_ctrl/S01_AXI]
# MI index -> AXI-Lite slave (order is the address-map's; matmul 0..5, flash 6..11, seq 12).
foreach {idx target} {
    0 dma_w0/S_AXI_LITE   1 dma_w1/S_AXI_LITE   2 dma_w2/S_AXI_LITE   3 dma_w3/S_AXI_LITE
    4 dma_a/S_AXI_LITE    5 kernel_mm/S_AXI
    6 dma_q/S_AXI_LITE    7 dma_k/S_AXI_LITE    8 dma_v/S_AXI_LITE    9 dma_mask/S_AXI_LITE
    10 dma_o/S_AXI_LITE  11 kernel_fa/S_AXI
    12 seq_top/S_AXI
} { connect_bd_intf_net [get_bd_intf_pins sc_ctrl/[mi_pin $idx]] [get_bd_intf_pins $target] }

# ---- Memory paths: each DMA mem master -> its PS HP/HPC port (all fclk) ------
# Matmul over HP0..3, flash over HPC0/1 — same shape now, one helper. S-pin order = list order.
make_mem_sc sc_hp0  {dma_w0/M_AXI_MM2S dma_w0/M_AXI_S2MM dma_a/M_AXI_MM2S}    S_AXI_HP0_FPD
make_mem_sc sc_hp1  {dma_w1/M_AXI_MM2S}                                       S_AXI_HP1_FPD
make_mem_sc sc_hp2  {dma_w2/M_AXI_MM2S}                                       S_AXI_HP2_FPD
make_mem_sc sc_hp3  {dma_w3/M_AXI_MM2S}                                       S_AXI_HP3_FPD
make_mem_sc sc_hpc0 {dma_q/M_AXI_MM2S dma_o/M_AXI_S2MM dma_mask/M_AXI_MM2S} S_AXI_HPC0_FPD
make_mem_sc sc_hpc1 {dma_k/M_AXI_MM2S dma_v/M_AXI_MM2S}                       S_AXI_HPC1_FPD

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
# seq_top control slave: a fixed base the host SeqCtrl matches (device/pl/seq.zig ctrl.base).
# Above the matmul (0xA00x) / flash (0xA01x) windows. 64K covers the register block + the CMD
# window at +0x8000 (2048 entries).
assign_bd_address -offset 0xA0200000 -range 64K \
    [first_addr_seg [list "seq_top/S_AXI/Reg" "seq_top/S_AXI/*"]]

# Route the DMA mem masters and seq_top's M_AXI_REG -> sc_ctrl targets.
assign_bd_address

# Pin the replay master's view: the host builds command streams from the PS address map, so
# seq_top/M_AXI_REG MUST see every sc_ctrl target at the same offset the PS does. Fail the
# build on any divergence rather than ship a bitstream that replays onto wrong registers.
foreach {addr_list kcell} [list $matmul_address_map kernel_mm $flash_address_map kernel_fa] {
    foreach entry $addr_list {
        lassign $entry cell intf offset
        if {$cell eq "kernel"} { set cell $kcell }
        set segs [get_bd_addr_segs -of_objects [get_bd_addr_spaces seq_top/M_AXI_REG] \
            -filter "NAME =~ *${cell}*"]
        if {[llength $segs] != 1} { error "seq M_AXI_REG: want 1 addr seg for $cell, got: $segs" }
        set got [format 0x%08X [get_property OFFSET [lindex $segs 0]]]
        set want [format 0x%08X $offset]
        if {$got ne $want} { error "seq M_AXI_REG sees $cell at $got, PS map says $want" }
    }
}
puts "==> seq_top/M_AXI_REG address view matches the PS map (12 targets)"

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

record_phase project_generation $build_started
set phase_started [clock milliseconds]
launch_runs synth_1 -jobs 4
wait_on_run synth_1
if {[get_property PROGRESS [get_runs synth_1]] != "100%"} { error "synthesis failed" }
record_phase synthesis $phase_started

# Post-synth resource utilization is both a human report and machine-readable run metric.
if {![catch {open_run synth_1 -name synth_1} err]} {
    set synth_util [report_utilization -return_string]
    ::penzai_analysis::write_text [file join $outdir utilization_synth.rpt] $synth_util
    ::penzai_analysis::collect_utilization synth $synth_util
    puts "==> wrote $outdir/utilization_synth.rpt (check DSP/LUT/BRAM fit)"
    catch {close_design}
} else {
    puts "WARNING: could not open synth_1 for utilization report: $err"
}

# f300 implementation effort for the remaining routing-sensitive paths:
#   - PLACE  AltSpreadLogic_high  — spreads logic (cleared the old pc_col congestion); keep.
#   - ROUTE  Explore             — extra routing effort for the routing-bound DMA-FIFO worst path.
#   - POST-ROUTE phys_opt AggressiveExplore — run EXPLICITLY in-memory after open_run (the run-step
#     -to_step token for post-route phys_opt is version-inconsistent), to grab the last ~0.04ns.
set_property STEPS.PLACE_DESIGN.ARGS.DIRECTIVE AltSpreadLogic_high [get_runs impl_1]
set_property STEPS.ROUTE_DESIGN.ARGS.DIRECTIVE Explore [get_runs impl_1]

set incremental_active 0
if {$build_mode eq "incremental"} {
    if {[file exists $reference_dcp]} {
        set_property INCREMENTAL_CHECKPOINT $reference_dcp [get_runs impl_1]
        set incremental_active 1
        puts "==> incremental reference: $reference_dcp"
    } else {
        puts "WARNING: no reference checkpoint at $reference_dcp; running a clean implementation"
    }
}
::penzai_analysis::manifest_set incremental_active $incremental_active
::penzai_analysis::manifest_set reference_checkpoint $reference_dcp

set phase_started [clock milliseconds]
launch_runs impl_1 -to_step route_design -jobs 4
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] != "100%"} { error "implementation failed" }
record_phase implementation_through_route $phase_started

open_run impl_1
# AggressiveExplore is expensive and only useful when the routed design misses timing.
# When needed it incrementally reroutes touched nets, leaving the design routed.
set routed_failing [get_timing_paths -quiet -max_paths 1 -slack_lesser_than 0]
if {[llength $routed_failing] == 0} {
    puts "==> routed design is timing-clean; skipping post-route phys_opt_design"
} else {
    set routed_wns [get_property SLACK [lindex $routed_failing 0]]
    puts "==> routed WNS=${routed_wns}ns; running post-route AggressiveExplore"
    set phase_started [clock milliseconds]
    if {[catch {phys_opt_design -directive AggressiveExplore} pe]} {
        puts "WARNING: post-route phys_opt_design failed ($pe) — gating on the routed result as-is"
    } else {
        puts "==> post-route phys_opt_design (AggressiveExplore) complete"
    }
    record_phase post_route_phys_opt $phase_started
}
if {$incremental_active} {
    if {[catch {
        report_incremental_reuse -file $outdir/incremental_reuse.rpt
    } reuse_err]} {
        puts "WARNING: incremental reuse report failed: $reuse_err"
    }
}
set phase_started [clock milliseconds]
::penzai_analysis::collect_routed 1
record_phase routed_reports $phase_started

set phase_started [clock milliseconds]
write_bitstream -force $outdir/$bit_name.bit
record_phase bitstream_generation $phase_started

# Only timing-clean, successfully generated designs become future references. Keeping
# this outside the disposable project directory makes reuse explicit and variant-scoped.
set phase_started [clock milliseconds]
write_checkpoint -force $reference_dcp
record_phase checkpoint_save $phase_started
::penzai_analysis::finalize vivado_pass
puts "==> Built timing-clean bitstream: $outdir/$bit_name.bit"
puts "==> Incremental reference updated: $reference_dcp"
puts "RUN_OUTPUT $outdir"
