# metrics.tcl - shared Vivado build/checkpoint metrics and routed build gates.

namespace eval ::penzai_analysis {
    variable output_dir ""
    variable manifest [dict create]
    variable metrics [dict create]
    variable expected_dsp48e2 695
    variable expected_uram288 8
}

proc ::penzai_analysis::clean_value {value} {
    return [string map [list "\t" " " "\r" " " "\n" " "] $value]
}

proc ::penzai_analysis::write_dict {filename values} {
    set fh [open $filename w]
    puts $fh "key\tvalue"
    foreach key [lsort [dict keys $values]] {
        puts $fh "$key\t[clean_value [dict get $values $key]]"
    }
    close $fh
}

proc ::penzai_analysis::write_text {filename value} {
    set fh [open $filename w]
    puts -nonewline $fh $value
    close $fh
}

proc ::penzai_analysis::init {dir metadata} {
    variable output_dir
    variable manifest
    variable metrics
    set output_dir [file normalize $dir]
    file mkdir $output_dir
    set manifest $metadata
    dict set manifest schema_version 1
    dict set manifest generated_utc [clock format [clock seconds] -gmt 1 -format {%Y-%m-%dT%H:%M:%SZ}]
    dict set manifest status running
    set metrics [dict create]
    flush
}

proc ::penzai_analysis::flush {} {
    variable output_dir
    variable manifest
    variable metrics
    write_dict [file join $output_dir manifest.tsv] $manifest
    write_dict [file join $output_dir metrics.tsv] $metrics
}

proc ::penzai_analysis::manifest_set {key value} {
    variable manifest
    dict set manifest $key $value
    flush
}

proc ::penzai_analysis::metric_set {key value} {
    variable metrics
    dict set metrics $key $value
    flush
}

proc ::penzai_analysis::dict_get_or {values key fallback} {
    if {[dict exists $values $key]} { return [dict get $values $key] }
    return $fallback
}

proc ::penzai_analysis::utilization_used {report label} {
    set pattern [format {\|[[:space:]]*%s[[:space:]]*\|[[:space:]]*([0-9.]+)[[:space:]]*\|} $label]
    if {![regexp -- $pattern $report -> used]} {
        error "could not find utilization row '$label'"
    }
    return $used
}

proc ::penzai_analysis::collect_utilization {stage report} {
    metric_set ${stage}.clb_luts [utilization_used $report {CLB LUTs\*?}]
    metric_set ${stage}.registers [utilization_used $report {Register as Flip Flop}]
    metric_set ${stage}.carry8 [utilization_used $report {CARRY8}]
    metric_set ${stage}.bram_tiles [utilization_used $report {Block RAM Tile}]
    metric_set ${stage}.uram [utilization_used $report {URAM}]
    metric_set ${stage}.dsps [utilization_used $report {DSPs}]
}

proc ::penzai_analysis::check_timing_count {report check_name} {
    set pattern [format {checking[[:space:]]+%s[[:space:]]+\(([0-9]+)\)} $check_name]
    if {![regexp -- $pattern $report -> count]} {
        error "could not find check_timing result '$check_name'"
    }
    return $count
}

proc ::penzai_analysis::methodology_severity_counts {} {
    set critical 0
    set warning 0
    foreach violation [get_methodology_violations -quiet] {
        set severity [string tolower [get_property SEVERITY $violation]]
        if {$severity eq "error" || $severity eq "critical warning"} {
            incr critical
        } elseif {$severity eq "warning"} {
            incr warning
        }
    }
    return [list $critical $warning]
}

proc ::penzai_analysis::collect_exact_resources {stage} {
    variable expected_dsp48e2
    variable expected_uram288
    set dsp48e2 [llength [get_cells -hierarchical -quiet -filter \
        {IS_PRIMITIVE == 1 && REF_NAME == DSP48E2}]]
    set uram288 [llength [get_cells -hierarchical -quiet -filter \
        {IS_PRIMITIVE == 1 && REF_NAME == URAM288}]]
    metric_set utilization.${stage}_dsp48e2 $dsp48e2
    metric_set utilization.${stage}_uram288 $uram288
    return [dict create \
        valid [expr {$dsp48e2 == $expected_dsp48e2 && $uram288 == $expected_uram288}] \
        dsp48e2 $dsp48e2 uram288 $uram288 \
        expected_dsp48e2 $expected_dsp48e2 expected_uram288 $expected_uram288]
}

proc ::penzai_analysis::require_exact_resources {stage} {
    set audit [collect_exact_resources $stage]
    if {![dict get $audit valid]} {
        error "$stage resource contract failed: DSP48E2=[dict get $audit dsp48e2] (expected [dict get $audit expected_dsp48e2]), URAM288=[dict get $audit uram288] (expected [dict get $audit expected_uram288])"
    }
    return $audit
}

proc ::penzai_analysis::audit_methodology_whitelist {} {
    variable output_dir
    set allowed_critical_rules {TIMING-2 TIMING-4}
    set recognized_severities {Error {Critical Warning} Warning Info Advisory}
    set timing_2_count 0
    set timing_4_count 0
    set timing_28_warning_count 0
    set error_count 0
    set unexpected_critical_count 0
    set malformed_count 0
    set unknown_severity_count 0
    array set seen_names {}

    set fh [open [file join $output_dir methodology_violations.tsv] w]
    puts $fh "name\trule\tseverity\tallowed"
    foreach violation [get_methodology_violations -quiet] {
        set name [get_property -quiet NAME $violation]
        set severity [get_property -quiet SEVERITY $violation]
        set rule [lindex [split $name #] 0]
        set allowed 1

        if {$name eq "" || $severity eq "" || $rule eq ""} {
            incr malformed_count
            set allowed 0
        } elseif {[lsearch -exact $recognized_severities $severity] < 0} {
            incr unknown_severity_count
            set allowed 0
        }
        if {$name ne ""} {
            if {[info exists seen_names($name)]} {
                incr malformed_count
                set allowed 0
            }
            set seen_names($name) 1
        }

        if {$severity eq "Error"} {
            incr error_count
            set allowed 0
        } elseif {$severity eq "Critical Warning"} {
            if {[lsearch -exact $allowed_critical_rules $rule] < 0 ||
                ($name ne $rule && $name ne "${rule}#1")} {
                incr unexpected_critical_count
                set allowed 0
            } elseif {$rule eq "TIMING-2"} {
                incr timing_2_count
                if {$timing_2_count > 1} {
                    incr unexpected_critical_count
                    set allowed 0
                }
            } else {
                incr timing_4_count
                if {$timing_4_count > 1} {
                    incr unexpected_critical_count
                    set allowed 0
                }
            }
        } elseif {$severity eq "Warning" && $rule eq "TIMING-28"} {
            incr timing_28_warning_count
        }

        puts $fh "[clean_value $name]\t[clean_value $rule]\t[clean_value $severity]\t$allowed"
    }
    close $fh

    metric_set methodology.allowed_timing_2 $timing_2_count
    metric_set methodology.allowed_timing_4 $timing_4_count
    metric_set methodology.recorded_timing_28_warnings $timing_28_warning_count
    metric_set methodology.error $error_count
    metric_set methodology.unexpected_critical $unexpected_critical_count
    metric_set methodology.malformed $malformed_count
    metric_set methodology.unknown_severity $unknown_severity_count
    return [dict create \
        valid [expr {$error_count == 0 && $unexpected_critical_count == 0 && \
            $malformed_count == 0 && $unknown_severity_count == 0}] \
        error $error_count unexpected_critical $unexpected_critical_count \
        malformed $malformed_count unknown_severity $unknown_severity_count]
}

proc ::penzai_analysis::write_summary {status failures} {
    variable output_dir
    variable manifest
    variable metrics
    set mode [dict_get_or $manifest build_mode [dict_get_or $manifest analysis_mode unknown]]
    set fh [open [file join $output_dir summary.txt] w]
    puts $fh "FPGA_BUILD $status"
    puts $fh "run=[dict_get_or $manifest run_id unknown] variant=[dict_get_or $manifest variant unknown] mode=$mode"
    puts $fh "setup_wns_ns=[dict_get_or $metrics timing.setup_wns_ns n/a] hold_whs_ns=[dict_get_or $metrics timing.hold_whs_ns n/a] near_lt_0.050ns=[dict_get_or $metrics timing.setup_paths_lt_0_050ns n/a]"
    puts $fh "routed_lut=[dict_get_or $metrics routed.clb_luts n/a] routed_ff=[dict_get_or $metrics routed.registers n/a] bram_tiles=[dict_get_or $metrics routed.bram_tiles n/a] uram=[dict_get_or $metrics routed.uram n/a] dsp=[dict_get_or $metrics routed.dsps n/a]"
    puts $fh "route_fully=[dict_get_or $metrics route.fully_routed n/a] no_clock=[dict_get_or $metrics constraints.no_clock n/a] unconstrained_endpoints=[dict_get_or $metrics constraints.unconstrained_internal_endpoints n/a] methodology_critical=[dict_get_or $metrics methodology.critical n/a]"
    if {[llength $failures] > 0} {
        puts $fh "failures=[join $failures {; }]"
    }
    close $fh
}

proc ::penzai_analysis::collect_routed {strict} {
    variable output_dir
    variable manifest

    set util [report_utilization -return_string]
    write_text [file join $output_dir utilization_routed.rpt] $util
    collect_utilization routed $util
    set resource_audit [collect_exact_resources routed]
    report_utilization -hierarchical -hierarchical_depth 4 \
        -file [file join $output_dir utilization_hierarchical.rpt]

    report_methodology -file [file join $output_dir methodology.rpt]
    lassign [methodology_severity_counts] methodology_critical methodology_warning
    metric_set methodology.critical $methodology_critical
    metric_set methodology.warning $methodology_warning
    set methodology_audit [audit_methodology_whitelist]

    set timing_checks [check_timing -return_string]
    write_text [file join $output_dir check_timing.rpt] $timing_checks
    foreach check_name {no_clock unconstrained_internal_endpoints pulse_width_clock multiple_clock generated_clocks loops latch_loops} {
        metric_set constraints.$check_name [check_timing_count $timing_checks $check_name]
    }

    report_route_status -file [file join $output_dir route_status.rpt]
    set fully_routed [report_route_status -boolean_check ROUTED_FULLY]
    metric_set route.fully_routed $fully_routed

    report_timing_summary -max_paths 10 -routable_nets -report_unconstrained \
        -file [file join $output_dir timing_summary.rpt]

    set setup_paths [get_timing_paths -quiet -setup -max_paths 1]
    set hold_paths [get_timing_paths -quiet -hold -max_paths 1]
    set setup_wns [expr {[llength $setup_paths] ? [get_property SLACK [lindex $setup_paths 0]] : "n/a"}]
    set hold_whs [expr {[llength $hold_paths] ? [get_property SLACK [lindex $hold_paths 0]] : "n/a"}]
    metric_set timing.setup_wns_ns $setup_wns
    metric_set timing.hold_whs_ns $hold_whs
    metric_set timing.setup_met [expr {$setup_wns ne "n/a" && $setup_wns >= 0.0 ? 1 : 0}]
    metric_set timing.hold_met [expr {$hold_whs ne "n/a" && $hold_whs >= 0.0 ? 1 : 0}]

    set path_query_limit 10000
    metric_set timing.path_count_limit $path_query_limit
    foreach threshold {0.000 0.050 0.100 0.200} {
        set key [string map {. _} $threshold]
        set setup_near [get_timing_paths -quiet -setup -max_paths $path_query_limit -unique_pins -slack_lesser_than $threshold]
        set hold_near [get_timing_paths -quiet -hold -max_paths $path_query_limit -unique_pins -slack_lesser_than $threshold]
        metric_set timing.setup_paths_lt_${key}ns [llength $setup_near]
        metric_set timing.hold_paths_lt_${key}ns [llength $hold_near]
    }

    set failures {}
    if {$setup_wns eq "n/a"} {
        lappend failures "no setup timing path"
    } elseif {$setup_wns < 0.0} {
        lappend failures "setup WNS ${setup_wns}ns"
    }
    if {$hold_whs eq "n/a"} {
        lappend failures "no hold timing path"
    } elseif {$hold_whs < 0.0} {
        lappend failures "hold WHS ${hold_whs}ns"
    }
    if {!$fully_routed} { lappend failures "design is not fully routed" }
    if {![dict get $resource_audit valid]} {
        lappend failures "routed resource contract: DSP48E2=[dict get $resource_audit dsp48e2]/[dict get $resource_audit expected_dsp48e2] URAM288=[dict get $resource_audit uram288]/[dict get $resource_audit expected_uram288]"
    }
    if {![dict get $methodology_audit valid]} {
        lappend failures "methodology whitelist: errors=[dict get $methodology_audit error] unexpected_critical=[dict get $methodology_audit unexpected_critical] malformed=[dict get $methodology_audit malformed] unknown_severity=[dict get $methodology_audit unknown_severity]"
    }
    foreach check_name {no_clock unconstrained_internal_endpoints} {
        set count [check_timing_count $timing_checks $check_name]
        if {$count != 0} { lappend failures "$check_name=$count" }
    }

    set status [expr {[llength $failures] ? "FAIL" : "PASS"}]
    dict set manifest status [expr {[llength $failures] ? "routed_gate_failed" : "routed_gate_passed"}]
    flush
    write_summary $status $failures
    if {$strict && [llength $failures] > 0} {
        error "routed build failed: [join $failures {; }]"
    }
    return [llength $failures]
}

proc ::penzai_analysis::finalize {status} {
    variable manifest
    dict set manifest status $status
    flush
    write_summary [expr {$status eq "complete" || $status eq "vivado_pass" ? "PASS" : [string toupper $status]}] {}
}
