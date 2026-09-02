# Vivado OOC route for the explicit DSP48E2 FOUR12 dot fabric.

set_param general.maxThreads 8
set part xck26-sfvc784-2LV-c

if {$argc != 3} {
    error "usage: route_dot.tcl <period_ns> <out_dir> <repo_root>"
}

set period [lindex $argv 0]
set outdir [file normalize [lindex $argv 1]]
set repo [file normalize [lindex $argv 2]]
file mkdir $outdir

set dot_src [file join $repo fpga rtl projection dot4.v]
set shell_src [file join $repo fpga verify qor vivado_ooc projection dot4_ooc.v]
if {![file exists $dot_src]} { set dot_src [file join $repo dot4.v] }
if {![file exists $shell_src]} { set shell_src [file join $repo dot4_ooc.v] }
if {![file exists $dot_src] || ![file exists $shell_src]} {
    error "missing dot4 OOC source under $repo"
}
set_property verilog_define {PENZAI_XILINX_DSP48E2} [current_fileset]
read_verilog [list $dot_src $shell_src]

synth_design -mode out_of_context -flatten_hierarchy rebuilt \
    -part $part -top dot4_ooc
create_clock -name clk -period $period [get_ports clk]

# Package inputs and outputs are harness boundaries.  The registered shell
# keeps all dot data/control paths timed internally.
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
if {[llength $setup_path] == 0} {
    error "no routed setup path"
}
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
puts $metrics "top\tdot4_ooc"
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

puts "RESULT top=dot4_ooc period_ns=$period setup_wns_ns=$setup_wns"
if {$setup_wns < 0.0} {
    error "dot4_ooc failed setup timing"
}
