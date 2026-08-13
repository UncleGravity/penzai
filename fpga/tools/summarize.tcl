# summarize.tcl - apply production routed metrics and gates to an existing checkpoint.
# Usage: vivado -mode batch -source tools/summarize.tcl -tclargs <dcp> <output-dir>

if {$argc != 2} {
    error "usage: summarize.tcl <routed.dcp> <output-dir>"
}

set checkpoint [file normalize [lindex $argv 0]]
set output_dir [file normalize [lindex $argv 1]]
if {![file exists $checkpoint]} { error "checkpoint does not exist: $checkpoint" }

source [file join [file dirname [file normalize [info script]]] metrics.tcl]
open_checkpoint $checkpoint

set checkpoint_name [file tail $checkpoint]
set variant unknown
regexp {penzai-combined-v1-(w512-p4-f[0-9]+)-routed\.dcp$} $checkpoint_name -> variant
::penzai_analysis::init $output_dir [dict create \
    run_id [file tail $output_dir] \
    analysis_mode summary \
    variant $variant \
    checkpoint $checkpoint \
    vivado_version [version -short] \
    part [get_property PART [current_design]]]
# A retained combined-design checkpoint must satisfy the same timing and
# structural gates as a fresh bitstream build.
::penzai_analysis::collect_routed 1
::penzai_analysis::finalize complete
puts "ANALYSIS_DONE $output_dir"
