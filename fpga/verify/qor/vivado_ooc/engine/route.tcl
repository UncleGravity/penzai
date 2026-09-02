set_param general.maxThreads 8

set part xck26-sfvc784-2LV-c
if {$argc != 4} {
    error "usage: route.tcl <repo> <period_ns> <out_dir> <source_manifest>"
}

set repo [file normalize [lindex $argv 0]]
set period [lindex $argv 1]
set out_dir [file normalize [lindex $argv 2]]
set manifest [file normalize [lindex $argv 3]]
set top engine_ooc
file mkdir $out_dir

set fh [open $manifest r]
set source_text [read $fh]
close $fh
set sources {}
foreach raw [split $source_text "\n"] {
    set line [string trim $raw]
    if {$line eq "" || [string index $line 0] eq "#"} { continue }
    set source [file normalize [file join $repo $line]]
    if {![file exists $source]} { error "manifest source missing: $source" }
    if {[file extension $source] ne ".vh"} { lappend sources $source }
}

puts "==> engine OOC top=$top period=${period}ns"
foreach source $sources {
    read_verilog $source
}

synth_design -mode out_of_context -part $part -top $top -flatten_hierarchy rebuilt
create_clock -name clk -period $period [get_ports clk]
foreach port [get_ports] {
    if {[get_property NAME $port] ne "clk" &&
        [get_property DIRECTION $port] eq "IN"} {
        set_false_path -from $port
    }
}
set_false_path -to [all_outputs]

opt_design
place_design -directive Explore
phys_opt_design -directive AggressiveExplore
route_design -directive Explore

report_utilization -file [file join $out_dir utilization.rpt]
report_utilization -hierarchical -file [file join $out_dir utilization_hier.rpt]
report_timing_summary -max_paths 20 -file [file join $out_dir timing_summary.rpt]
write_checkpoint -force [file join $out_dir routed.dcp]

set setup_paths [get_timing_paths -quiet -setup -max_paths 1]
set hold_paths [get_timing_paths -quiet -hold -max_paths 1]
set setup_wns [expr {[llength $setup_paths] ?
    [get_property SLACK [lindex $setup_paths 0]] : "n/a"}]
set hold_wns [expr {[llength $hold_paths] ?
    [get_property SLACK [lindex $hold_paths 0]] : "n/a"}]
set util [report_utilization -return_string]

proc used {report label} {
    set pattern [format {\|[[:space:]]*%s[[:space:]]*\|[[:space:]]*([0-9]+)[[:space:]]*\|} $label]
    if {![regexp -- $pattern $report -> value]} { return 0 }
    return $value
}

set metrics [open [file join $out_dir metrics.tsv] w]
puts $metrics "key\tvalue"
puts $metrics "top\t$top"
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

puts "RESULT setup_wns_ns=$setup_wns hold_wns_ns=$hold_wns"
if {$setup_wns eq "n/a" || $hold_wns eq "n/a" ||
    $setup_wns < 0.0 || $hold_wns < 0.0} {
    error "engine OOC timing failed"
}
