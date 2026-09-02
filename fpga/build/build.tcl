# build.tcl - production KR260 Penzai bitstream.
#
# One penzai_top owns the complete token walk. Four packed-weight readers use
# HP0..HP3. Committed K and V history use separate HPC ports; the KV writer
# shares HPC0 with K because append and history attention are phase-disjoint.
# One AXI-Lite control cell is visible to the PS at 0xA0000000.

set_param general.maxThreads 8
puts "==> Vivado general.maxThreads=[get_param general.maxThreads]"

set variant    [expr {$argc >= 1 ? [lindex $argv 0] : "f225"}]
set build_mode [expr {$argc >= 2 ? [lindex $argv 1] : "clean"}]
set run_id     [expr {$argc >= 3 ? [lindex $argv 2] : "manual-$variant"}]
set git_commit [expr {$argc >= 4 ? [lindex $argv 3] : "unknown"}]
set git_dirty  [expr {$argc >= 5 ? [lindex $argv 4] : "unknown"}]
set source_hash [expr {$argc >= 6 ? [lindex $argv 5] : "unknown"}]

# variant = f<MHz>. fclk drives the engine and all AXI fabrics.
if {![regexp {^f([0-9]+)$} $variant -> fclk_mhz]} {
    error "unknown variant '$variant'; expected f<MHz>, e.g. f225"
}
if {$build_mode ni {clean incremental}} {
    error "unknown build mode '$build_mode'; expected clean or incremental"
}
if {![regexp {^[A-Za-z0-9._-]+$} $run_id]} {
    error "invalid run ID '$run_id'"
}

set bit_prefix penzai
set bit_name   "$bit_prefix-$variant"
set proj       "penzai_[string map {- _} $variant]"
set bd         design_1
set part       xck26-sfvc784-2LV-c
set board      xilinx.com:kr260_som:part0:1.1
set run_root   [file normalize ./out/runs]
set outdir     [file join $run_root $run_id]
set cache_root [file normalize ./cache]
set ip_cache   [file join $cache_root ip]
set checkpoint_dir [file join $cache_root checkpoints]
set reference_dcp [file join $checkpoint_dir penzai-${variant}-routed.dcp]
set ps_fclk_mhz 99.999001
set implementation_clock_name clk_out1_design_1_clk_wiz_0
set place_setup_guardband_ns 0.075
set nominal_setup_uncertainty_ns 0.000

file delete -force $outdir [file normalize ./$proj]
file mkdir $outdir $ip_cache $checkpoint_dir

source [file normalize ./metrics.tcl]
set expected_dsp48e2 $::penzai_analysis::expected_dsp48e2
set expected_uram288 $::penzai_analysis::expected_uram288
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
    phys_opt_enabled 1 \
    phys_opt_directive AggressiveExplore \
    route_directive Explore \
    implementation_clock $implementation_clock_name \
    place_setup_guardband_ns $place_setup_guardband_ns \
    nominal_setup_uncertainty_ns $nominal_setup_uncertainty_ns \
    expected_dsp48e2 $expected_dsp48e2 \
    expected_uram288 $expected_uram288]
::penzai_analysis::metric_set timing.place_setup_guardband_ns $place_setup_guardband_ns
::penzai_analysis::metric_set timing.nominal_setup_uncertainty_ns $nominal_setup_uncertainty_ns

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

proc first_addr_seg {patterns} {
    foreach pattern $patterns {
        set segs [get_bd_addr_segs -quiet $pattern]
        if {[llength $segs] > 0} { return [lindex $segs 0] }
    }
    error "none of these address segments exist: $patterns"
}
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

# The closed production source set is flattened under ./rtl by build.sh. Check
# the copied directory against the transferred manifest before adding anything.
set rtl_dir [file normalize ./rtl]
set rtl_manifest [file normalize ./sources.f]
if {![file exists $rtl_manifest]} { error "missing production source manifest $rtl_manifest" }
set expected_rtl_names {}
set fh [open $rtl_manifest r]
while {[gets $fh line] >= 0} {
    regsub {#.*$} $line {} line
    set relative [string trim $line]
    if {$relative eq ""} { continue }
    if {[file pathtype $relative] ne "relative" ||
        [regexp {(^|/)\.\.?(/|$)} $relative] ||
        [regexp {[[:space:]\\]} $relative]} {
        error "invalid production RTL path in sources.f: $relative"
    }
    if {[lsearch -exact {.v .sv .vh} [file extension $relative]] < 0} {
        error "unsupported production RTL extension in sources.f: $relative"
    }
    lappend expected_rtl_names [file tail $relative]
}
close $fh
lappend expected_rtl_names engine_regs.vh
set unique_rtl_names [lsort -dictionary -unique $expected_rtl_names]
if {[llength $unique_rtl_names] != [llength $expected_rtl_names]} {
    error "duplicate basename in production source manifest"
}
set expected_rtl_names [lsort -dictionary $expected_rtl_names]
set actual_rtl_names {}
foreach f [glob -nocomplain -directory $rtl_dir *] {
    if {![file isfile $f]} { error "unexpected non-file entry under $rtl_dir: $f" }
    lappend actual_rtl_names [file tail $f]
}
set actual_rtl_names [lsort -dictionary $actual_rtl_names]
if {$actual_rtl_names ne $expected_rtl_names} {
    error "remote RTL closure differs from sources.f: expected={$expected_rtl_names} actual={$actual_rtl_names}"
}

set rtl_files {}
set rtl_headers {}
foreach name $expected_rtl_names {
    set f [file join $rtl_dir $name]
    switch -- [file extension $name] {
        .v - .sv { lappend rtl_files $f }
        .vh { lappend rtl_headers $f }
        default { error "unsupported production RTL extension: $name" }
    }
}
if {[llength $rtl_files] == 0} { error "production source manifest contains no modules" }
add_files -norecurse $rtl_files
# All included RTL contracts are explicit source headers.
foreach f $rtl_headers {
    add_files -norecurse $f
    set_property file_type "Verilog Header" [get_files $f]
}
set_property include_dirs [list $rtl_dir] [get_filesets sources_1]
# Select the explicit FOUR12 DSP48E2 implementation used by the routed
# GEMM. The portable branch is only for Verilator and vendor-independent Yosys.
set_property verilog_define {PENZAI_XILINX_DSP48E2} [get_filesets sources_1]
update_compile_order -fileset sources_1

create_bd_design $bd

# ---- Processing system ------------------------------------------------------
create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e:* ps
apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e \
    -config {apply_board_preset 1} [get_bd_cells ps]

# HPC0/HPC1 carry KV history/append. HP0..HP3 carry four packed-weight
# lanes. Every native memory interface is 128-bit and 40-bit addressed.
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

# ---- Fabric clock ------------------------------------------------------------
# One clock drives the token core and both control/memory AXI fabrics.
create_bd_cell -type ip -vlnv xilinx.com:ip:clk_wiz:* clk_wiz
set_property -dict [list \
    CONFIG.PRIM_IN_FREQ $ps_fclk_mhz \
    CONFIG.CLKOUT1_REQUESTED_OUT_FREQ $fclk_mhz \
    CONFIG.USE_LOCKED {true} \
    CONFIG.USE_RESET {false} \
] [get_bd_cells clk_wiz]
connect_bd_net [get_bd_pins ps/pl_clk0] [get_bd_pins clk_wiz/clk_in1]
set fclk_pin "clk_wiz/clk_out1"

# A SmartConnect is used only where clients must merge or protocol adaptation is
# required. One-client memory paths connect directly to the matching PS port.
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

# ---- Accelerator root ------------------------------------------------------
create_bd_cell -type module -reference penzai_top engine

# ---- Reset ------------------------------------------------------------------
create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:* rst
set_property -dict [list CONFIG.C_EXT_RESET_HIGH {0}] [get_bd_cells rst]
connect_bd_net [get_bd_pins $fclk_pin]      [get_bd_pins rst/slowest_sync_clk]
connect_bd_net [get_bd_pins ps/pl_resetn0]  [get_bd_pins rst/ext_reset_in]
connect_bd_net [get_bd_pins clk_wiz/locked] [get_bd_pins rst/dcm_locked]

# ---- Clock fan-out -----------------------------------------------------------
foreach clkpin {
    ps/maxihpm0_fpd_aclk
    ps/saxihpc0_fpd_aclk ps/saxihpc1_fpd_aclk
    ps/saxihp0_fpd_aclk ps/saxihp1_fpd_aclk ps/saxihp2_fpd_aclk ps/saxihp3_fpd_aclk
    engine/s_axi_aclk
} { connect_bd_net [get_bd_pins $fclk_pin] [get_bd_pins $clkpin] }

connect_bd_net [get_bd_pins rst/peripheral_aresetn] \
    [get_bd_pins engine/s_axi_aresetn]

# ---- Control path: one PS-visible engine cell -------------------------------
create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:* sc_ctrl
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {1}] [get_bd_cells sc_ctrl]
connect_bd_net [get_bd_pins $fclk_pin]              [get_bd_pins sc_ctrl/aclk]
connect_bd_net [get_bd_pins rst/peripheral_aresetn] [get_bd_pins sc_ctrl/aresetn]
connect_bd_intf_net [get_bd_intf_pins ps/M_AXI_HPM0_FPD] [get_bd_intf_pins sc_ctrl/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins sc_ctrl/M00_AXI] \
    [get_bd_intf_pins engine/S_AXI]

# ---- Native memory paths ----------------------------------------------------
# Preserve one physical DDR ingress per lockstep weight lane. Direct connections
# avoid six unnecessary one-client SmartConnects in the already logic-heavy root.
connect_bd_intf_net [get_bd_intf_pins engine/M_AXI_W0] \
    [get_bd_intf_pins ps/S_AXI_HP0_FPD]
connect_bd_intf_net [get_bd_intf_pins engine/M_AXI_W1] \
    [get_bd_intf_pins ps/S_AXI_HP1_FPD]
connect_bd_intf_net [get_bd_intf_pins engine/M_AXI_W2] \
    [get_bd_intf_pins ps/S_AXI_HP2_FPD]
connect_bd_intf_net [get_bd_intf_pins engine/M_AXI_W3] \
    [get_bd_intf_pins ps/S_AXI_HP3_FPD]
connect_bd_intf_net [get_bd_intf_pins engine/M_AXI_HIST_V] \
    [get_bd_intf_pins ps/S_AXI_HPC1_FPD]
make_mem_sc sc_hpc0 {engine/M_AXI_HIST_K engine/M_AXI_KV} S_AXI_HPC0_FPD

# ---- Address map ------------------------------------------------------------
source [file normalize ./engine_address_map.tcl]
foreach entry $engine_address_map {
    lassign $entry cell intf offset
    assign_bd_address -offset $offset -range 4K \
        [first_addr_seg [list "$cell/$intf/Reg" "$cell/$intf/*"]]
}
assign_bd_address

validate_bd_design
set kernel_clk_hz [clock_hz $fclk_pin $fclk_mhz]
puts "==> engine CLK_HZ=$kernel_clk_hz"
set_property CONFIG.CLK_HZ $kernel_clk_hz [get_bd_cells engine]
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

# Post-synth resource utilization is both a human report and a signoff gate.
if {[catch {open_run synth_1 -name synth_1} err]} {
    error "could not open synth_1 for resource signoff: $err"
}
set synth_util [report_utilization -return_string]
::penzai_analysis::write_text [file join $outdir utilization_synth.rpt] $synth_util
::penzai_analysis::collect_utilization synth $synth_util
::penzai_analysis::require_exact_resources synth
puts "==> wrote $outdir/utilization_synth.rpt (exact DSP/URAM contract passed)"
close_design

# Timing-sensitive implementation effort for the remaining critical paths. Placement
# sees an additional setup-only guardband so it optimizes the near-critical tail;
# the guardband is removed before routing and never weakens the nominal signoff.
set_property STEPS.PLACE_DESIGN.ARGS.DIRECTIVE AltSpreadLogic_high [get_runs impl_1]
set_property STEPS.PHYS_OPT_DESIGN.IS_ENABLED true [get_runs impl_1]
set_property STEPS.PHYS_OPT_DESIGN.ARGS.DIRECTIVE AggressiveExplore [get_runs impl_1]
set_property STEPS.ROUTE_DESIGN.ARGS.DIRECTIVE Explore [get_runs impl_1]

set apply_margin_hook [file join $outdir apply_setup_margin.tcl]
set reset_margin_hook [file join $outdir reset_setup_margin.tcl]
set preroute_margin_report [file join $outdir timing_preroute_margin.tsv]

set fh [open $apply_margin_hook w]
puts $fh "set penzai_clock_name [list $implementation_clock_name]"
puts $fh "set penzai_guardband_ns [list $place_setup_guardband_ns]"
puts $fh "set penzai_nominal_uncertainty_ns [list $nominal_setup_uncertainty_ns]"
puts $fh {
set penzai_clocks [get_clocks -quiet $penzai_clock_name]
if {[llength $penzai_clocks] != 1} {
    error "expected one implementation clock '$penzai_clock_name', got [llength $penzai_clocks]"
}
update_timing
set penzai_paths [get_timing_paths -quiet -setup -from $penzai_clocks -to $penzai_clocks -max_paths 1]
if {[llength $penzai_paths] != 1} { error "no initial pre-place path on $penzai_clock_name" }
set penzai_initial_wns [get_property SLACK [lindex $penzai_paths 0]]
set_clock_uncertainty -setup $penzai_nominal_uncertainty_ns $penzai_clocks
update_timing
set penzai_paths [get_timing_paths -quiet -setup -from $penzai_clocks -to $penzai_clocks -max_paths 1]
set penzai_nominal_wns [get_property SLACK [lindex $penzai_paths 0]]
if {abs($penzai_nominal_wns - $penzai_initial_wns) > 0.002} {
    error "existing setup uncertainty differs from declared nominal ${penzai_nominal_uncertainty_ns}ns"
}
set_clock_uncertainty -setup $penzai_guardband_ns $penzai_clocks
puts "==> placement setup guardband: ${penzai_guardband_ns}ns on $penzai_clock_name"
}
close $fh

set fh [open $reset_margin_hook w]
puts $fh "set penzai_clock_name [list $implementation_clock_name]"
puts $fh "set penzai_guardband_ns [list $place_setup_guardband_ns]"
puts $fh "set penzai_nominal_uncertainty_ns [list $nominal_setup_uncertainty_ns]"
puts $fh "set penzai_margin_report [list $preroute_margin_report]"
puts $fh {
set penzai_clocks [get_clocks -quiet $penzai_clock_name]
if {[llength $penzai_clocks] != 1} {
    error "expected one implementation clock '$penzai_clock_name', got [llength $penzai_clocks]"
}
update_timing
set penzai_paths [get_timing_paths -quiet -setup -from $penzai_clocks -to $penzai_clocks -max_paths 1]
if {[llength $penzai_paths] != 1} { error "no constrained pre-route path on $penzai_clock_name" }
set penzai_constrained_wns [get_property SLACK [lindex $penzai_paths 0]]
set_clock_uncertainty -setup $penzai_nominal_uncertainty_ns $penzai_clocks
update_timing
set penzai_paths [get_timing_paths -quiet -setup -from $penzai_clocks -to $penzai_clocks -max_paths 1]
if {[llength $penzai_paths] != 1} { error "no nominal pre-route path on $penzai_clock_name" }
set penzai_nominal_wns [get_property SLACK [lindex $penzai_paths 0]]
set penzai_delta [expr {$penzai_nominal_wns - $penzai_constrained_wns}]
if {abs($penzai_delta - $penzai_guardband_ns) > 0.002} {
    error "setup guardband reset delta ${penzai_delta}ns, expected ${penzai_guardband_ns}ns"
}
set penzai_fh [open $penzai_margin_report w]
puts $penzai_fh "key\tvalue"
puts $penzai_fh "clock\t$penzai_clock_name"
puts $penzai_fh "guardband_ns\t$penzai_guardband_ns"
puts $penzai_fh "nominal_uncertainty_ns\t$penzai_nominal_uncertainty_ns"
puts $penzai_fh "constrained_wns_ns\t$penzai_constrained_wns"
puts $penzai_fh "nominal_wns_ns\t$penzai_nominal_wns"
puts $penzai_fh "observed_delta_ns\t$penzai_delta"
close $penzai_fh
puts "==> pre-route setup guardband removed: constrained=${penzai_constrained_wns}ns nominal=${penzai_nominal_wns}ns delta=${penzai_delta}ns"
}
close $fh

set_property STEPS.PLACE_DESIGN.TCL.PRE [file normalize $apply_margin_hook] [get_runs impl_1]
set_property STEPS.ROUTE_DESIGN.TCL.PRE [file normalize $reset_margin_hook] [get_runs impl_1]

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

# Prove that the routed checkpoint carries nominal constraints. Reapplying the
# placement guardband must move setup slack by exactly that amount, and removing
# it must restore the original result before reports, bitstream, and checkpoint.
set routed_clocks [get_clocks -quiet $implementation_clock_name]
if {[llength $routed_clocks] != 1} {
    error "expected one routed clock '$implementation_clock_name', got [llength $routed_clocks]"
}
set routed_paths [get_timing_paths -quiet -setup -from $routed_clocks -to $routed_clocks -max_paths 1]
if {[llength $routed_paths] != 1} { error "routed design has no setup path on $implementation_clock_name" }
set routed_nominal_wns [get_property SLACK [lindex $routed_paths 0]]
set_clock_uncertainty -setup $place_setup_guardband_ns $routed_clocks
update_timing
set routed_paths [get_timing_paths -quiet -setup -from $routed_clocks -to $routed_clocks -max_paths 1]
set routed_constrained_wns [get_property SLACK [lindex $routed_paths 0]]
set_clock_uncertainty -setup $nominal_setup_uncertainty_ns $routed_clocks
update_timing
set routed_paths [get_timing_paths -quiet -setup -from $routed_clocks -to $routed_clocks -max_paths 1]
set routed_restored_wns [get_property SLACK [lindex $routed_paths 0]]
set routed_margin_delta [expr {$routed_nominal_wns - $routed_constrained_wns}]
if {abs($routed_margin_delta - $place_setup_guardband_ns) > 0.002} {
    error "routed setup guardband delta ${routed_margin_delta}ns, expected ${place_setup_guardband_ns}ns"
}
if {abs($routed_restored_wns - $routed_nominal_wns) > 0.001} {
    error "routed nominal WNS did not restore: before=${routed_nominal_wns}ns after=${routed_restored_wns}ns"
}
::penzai_analysis::metric_set timing.routed_guardband_delta_ns $routed_margin_delta
::penzai_analysis::metric_set timing.routed_nominal_restore_delta_ns \
    [expr {$routed_restored_wns - $routed_nominal_wns}]

if {![file exists $preroute_margin_report]} { error "missing $preroute_margin_report" }
set fh [open $preroute_margin_report r]
set preroute_values [dict create]
gets $fh
while {[gets $fh line] >= 0} {
    set fields [split $line "\t"]
    if {[llength $fields] == 2} { dict set preroute_values [lindex $fields 0] [lindex $fields 1] }
}
close $fh
foreach key {constrained_wns_ns nominal_wns_ns observed_delta_ns} {
    if {![dict exists $preroute_values $key]} { error "missing '$key' in $preroute_margin_report" }
    ::penzai_analysis::metric_set timing.preroute_$key [dict get $preroute_values $key]
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
