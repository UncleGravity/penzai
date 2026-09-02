# report.tcl - collect detailed diagnostics from a routed Vivado checkpoint.
#
# Usage:
#   vivado -mode batch -source report.tcl -tclargs \
#     <routed.dcp> <output-directory> ?replay-post-route-phys-opt?
#
# Set replay-post-route-phys-opt to 1 only when the checkpoint predates the build's
# post-route AggressiveExplore pass. Reporting is non-mutating by default.

if {$argc < 2 || $argc > 3} {
    error "usage: report.tcl <routed.dcp> <output-directory> ?replay-post-route-phys-opt?"
}

set checkpoint [file normalize [lindex $argv 0]]
set report_dir [file normalize [lindex $argv 1]]
set replay_phys_opt [expr {$argc == 3 ? [lindex $argv 2] : 0}]

if {![file exists $checkpoint]} { error "checkpoint does not exist: $checkpoint" }
file mkdir $report_dir

proc try_report {label command} {
    puts "==> report: $label"
    if {[catch {uplevel 1 $command} err opts]} {
        puts "WARNING: $label failed: $err"
        set fh [open [file join $::report_dir ${label}_ERROR.txt] w]
        puts $fh $err
        if {[dict exists $opts -errorinfo]} { puts $fh [dict get $opts -errorinfo] }
        close $fh
    }
}

proc safe_property {property object} {
    if {$object eq ""} { return "" }
    if {[catch {get_property $property $object} value]} { return "" }
    return $value
}

proc write_path_table {filename paths} {
    set fh [open $filename w]
    puts $fh "slack_ns\tdatapath_ns\tlogic_ns\troute_ns\tlogic_levels\trequirement_ns\tstartpoint\tendpoint\tpath_group"
    foreach path $paths {
        set datapath [safe_property DATAPATH_DELAY $path]
        set logic [safe_property LOGIC_DELAY $path]
        set route [safe_property ROUTE_DELAY $path]
        if {$route eq "" && $datapath ne "" && $logic ne ""} {
            set route [expr {double($datapath) - double($logic)}]
        }
        set values [list \
            [safe_property SLACK $path] \
            $datapath \
            $logic \
            $route \
            [safe_property LOGIC_LEVELS $path] \
            [safe_property REQUIREMENT $path] \
            [safe_property STARTPOINT_PIN $path] \
            [safe_property ENDPOINT_PIN $path] \
            [safe_property PATH_GROUP $path]]
        puts $fh [join $values "\t"]
    }
    close $fh
}

proc write_clock_region_occupancy {filename} {
    set fh [open $filename w]
    puts $fh "all_clock_regions=[join [lsort [get_clock_regions]] { }]"
    foreach instance {engine sc_ctrl sc_hp0 sc_hp1 sc_hp2 sc_hp3 sc_hpc0 sc_hpc1} {
        set cells [get_cells -quiet -hier -filter "NAME =~ */$instance/*"]
        set regions [lsort -unique [get_clock_regions -quiet -of_objects $cells]]
        puts $fh "$instance cells=[llength $cells] regions=[join $regions { }]"
    }
    close $fh
}

open_checkpoint $checkpoint

set phys_opt_status "not requested"
if {$replay_phys_opt} {
    puts "==> replaying post-route phys_opt_design -directive AggressiveExplore"
    if {[catch {phys_opt_design -directive AggressiveExplore} err]} {
        set phys_opt_status "failed: $err"
        puts "WARNING: $phys_opt_status"
    } else {
        set phys_opt_status "completed"
        write_checkpoint -force [file join $report_dir design_postroute_physopt.dcp]
    }
}

set failing [get_timing_paths -quiet -max_paths 10000 -slack_lesser_than 0.0]
set nearcritical [get_timing_paths -quiet -max_paths 10000 -slack_lesser_than 0.200 -unique_pins]
set worst [get_timing_paths -quiet -max_paths 1]
set wns [expr {[llength $worst] ? [safe_property SLACK [lindex $worst 0]] : "n/a"}]

set manifest [open [file join $report_dir manifest.txt] w]
puts $manifest "generated=[clock format [clock seconds] -format {%Y-%m-%d %H:%M:%S %Z}]"
puts $manifest "vivado=[version -short]"
puts $manifest "checkpoint=$checkpoint"
puts $manifest "part=[get_property PART [current_design]]"
puts $manifest "phys_opt=$phys_opt_status"
puts $manifest "wns_ns=$wns"
puts $manifest "failing_paths=[llength $failing]"
puts $manifest "nearcritical_unique_paths_lt_0.200ns=[llength $nearcritical]"
close $manifest

write_path_table [file join $report_dir timing_paths_failing.tsv] $failing
write_path_table [file join $report_dir timing_paths_nearcritical.tsv] $nearcritical

try_report timing_summary [list report_timing_summary \
    -max_paths 100 -routable_nets -report_unconstrained \
    -file [file join $report_dir timing_summary.rpt]]
try_report timing_setup_500_summary [list report_timing \
    -delay_type max -max_paths 500 -unique_pins -path_type summary \
    -file [file join $report_dir timing_setup_500_summary.rpt]]
try_report timing_setup_100_full [list report_timing \
    -delay_type max -max_paths 100 -unique_pins -path_type full_clock_expanded \
    -input_pins -file [file join $report_dir timing_setup_100_full.rpt]]
try_report timing_hold_100 [list report_timing \
    -delay_type min -max_paths 100 -unique_pins -path_type full_clock_expanded \
    -file [file join $report_dir timing_hold_100.rpt]]

try_report utilization_flat [list report_utilization \
    -file [file join $report_dir utilization_flat.rpt]]
try_report utilization_hierarchical [list report_utilization \
    -hierarchical -hierarchical_depth 12 \
    -file [file join $report_dir utilization_hierarchical.rpt]]
try_report high_fanout_nets [list report_high_fanout_nets \
    -fanout_greater_than 100 -max_nets 1000 \
    -file [file join $report_dir high_fanout_nets.rpt]]
try_report congestion [list report_design_analysis \
    -congestion -min_congestion_level 3 \
    -file [file join $report_dir congestion.rpt]]
try_report complexity [list report_design_analysis \
    -complexity -hierarchical_depth 6 \
    -file [file join $report_dir complexity.rpt]]
try_report logic_levels [list report_design_analysis \
    -logic_level_distribution \
    -file [file join $report_dir logic_levels.rpt]]
try_report timing_analysis [list report_design_analysis \
    -timing -setup -max_paths 100 \
    -file [file join $report_dir timing_analysis.rpt]]

try_report qor_assessment [list report_qor_assessment \
    -file [file join $report_dir qor_assessment.rpt]]
try_report qor_suggestions [list report_qor_suggestions \
    -file [file join $report_dir qor_suggestions.rpt]]
try_report methodology [list report_methodology \
    -file [file join $report_dir methodology.rpt]]
try_report route_status [list report_route_status \
    -file [file join $report_dir route_status.rpt]]
write_clock_region_occupancy [file join $report_dir clock_region_occupancy.txt]
try_report clock_utilization [list report_clock_utilization \
    -file [file join $report_dir clock_utilization.rpt]]
try_report clock_interaction [list report_clock_interaction \
    -file [file join $report_dir clock_interaction.rpt]]
try_report control_sets [list report_control_sets \
    -verbose -file [file join $report_dir control_sets.rpt]]
try_report exceptions [list report_exceptions \
    -summary -file [file join $report_dir exceptions.rpt]]
try_report cdc [list report_cdc \
    -details -file [file join $report_dir cdc.rpt]]
try_report power [list report_power \
    -file [file join $report_dir power.rpt]]

puts "ANALYSIS_DONE $report_dir"
