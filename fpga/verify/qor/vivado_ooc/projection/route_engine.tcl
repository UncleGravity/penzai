# Vivado OOC route for the full serialized four-lane, tile-8 projection engine.

set_param general.maxThreads 8
set part xck26-sfvc784-2LV-c

if {$argc != 3} {
    error "usage: route_engine.tcl <period_ns> <out_dir> <repo_root>"
}

set period [lindex $argv 0]
set outdir [file normalize [lindex $argv 1]]
set repo [file normalize [lindex $argv 2]]
file mkdir $outdir

proc source_path {repo relative fallback} {
    set candidate [file join $repo {*}$relative]
    if {[file exists $candidate]} { return $candidate }
    set candidate [file join $repo $fallback]
    if {[file exists $candidate]} { return $candidate }
    error "missing OOC source $fallback under $repo"
}

set sources [list \
    [source_path $repo {fpga rtl projection dot4.v} dot4.v] \
    [source_path $repo {fpga rtl projection digit_accum.v} digit_accum.v] \
    [source_path $repo {fpga rtl projection engine.v} engine.v] \
    [source_path $repo {fpga verify qor vivado_ooc projection projection_ooc.v} projection_ooc.v] \
    [source_path $repo {fpga rtl projection ternary_select.v} ternary_select.v] \
    [source_path $repo {fpga rtl projection gemm.v} gemm.v] \
    [source_path $repo {fpga rtl lib fma.v} fma.v]]

set_property verilog_define {PENZAI_XILINX_DSP48E2} [current_fileset]
read_verilog $sources

synth_design -mode out_of_context -flatten_hierarchy rebuilt \
    -part $part -top  projection_ooc
create_clock -name clk -period $period [get_ports clk]

set data_inputs [get_ports -quiet -filter {DIRECTION == IN && NAME != clk}]
if {[llength $data_inputs] > 0} {
    set_false_path -from $data_inputs
}
set_false_path -to [all_outputs]

opt_design
place_design
phys_opt_design
route_design

report_utilization -hierarchical -file [file join $outdir utilization_hier.rpt]
report_utilization -file [file join $outdir utilization.rpt]
report_timing_summary -delay_type min_max -max_paths 20 \
    -file [file join $outdir timing.rpt]
report_route_status -file [file join $outdir route_status.rpt]
report_design_analysis -congestion -file [file join $outdir congestion.rpt]
write_checkpoint -force [file join $outdir post_route.dcp]

set setup_path [get_timing_paths -quiet -delay_type max -max_paths 1 -nworst 1]
if {[llength $setup_path] == 0} { error "no routed setup path" }
set setup_wns [get_property SLACK [lindex $setup_path 0]]
set hold_path [get_timing_paths -quiet -delay_type min -max_paths 1 -nworst 1]
set hold_wns [expr {[llength $hold_path] == 0 ? "n/a" :
    [get_property SLACK [lindex $hold_path 0]]}]
set util [report_utilization -return_string]

proc used {report label} {
    set pattern [format {\|[[:space:]]*%s[[:space:]]*\|[[:space:]]*([0-9]+)[[:space:]]*\|} $label]
    if {![regexp -- $pattern $report -> value]} { return 0 }
    return $value
}

set metrics [open [file join $outdir metrics.tsv] w]
puts $metrics "key\tvalue"
puts $metrics "top\tprojection_ooc"
puts $metrics "part\t$part"
puts $metrics "period_ns\t$period"
puts $metrics "vivado_version\t[version -short]"
puts $metrics "setup_wns_ns\t$setup_wns"
puts $metrics "hold_wns_ns\t$hold_wns"
puts $metrics "lut\t[used $util {CLB LUTs\*?}]"
puts $metrics "ff\t[used $util {Register as Flip Flop}]"
puts $metrics "bram36\t[used $util {RAMB36/FIFO\*}]"
puts $metrics "bram18\t[used $util {RAMB18}]"
puts $metrics "uram\t[used $util {URAM}]"
puts $metrics "dsp\t[used $util {DSPs}]"
close $metrics

puts "RESULT top=projection_ooc period_ns=$period setup_wns_ns=$setup_wns hold_wns_ns=$hold_wns"
if {$setup_wns < 0.0} {
    error "projection_ooc failed setup timing"
}
