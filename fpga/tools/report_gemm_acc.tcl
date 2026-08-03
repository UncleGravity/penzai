# report_gemm_acc.tcl - inspect accumulator clock-enable fanout and locality in a routed checkpoint.
# Usage: vivado -mode batch -source report_gemm_acc.tcl -tclargs <dcp> <output-dir>

if {$argc != 2} {
    error "usage: report_gemm_acc.tcl <dcp> <output-dir>"
}

set checkpoint [file normalize [lindex $argv 0]]
set output_dir [file normalize [lindex $argv 1]]
file mkdir $output_dir
open_checkpoint $checkpoint

set acc_regs [get_cells -quiet -hier -regexp {.*u_rowblock/gen_lane\[[0-9]+\]\.gen_acc\[[0-9]+\]\.acc[SC]_reg.*}]
set leaf_regs [get_cells -quiet -hier -regexp {.*u_rowblock/gen_lane\[[0-9]+\]\.acc[SC]_write_enable_reg\[[0-9]+\].*}]

set ce_pins {}
foreach cell $acc_regs {
    set pin [get_pins -quiet [get_property NAME $cell]/CE]
    if {[llength $pin]} { lappend ce_pins $pin }
}
set ce_nets [lsort -unique [get_nets -quiet -of_objects $ce_pins]]

set min_loads 1000000000
set max_loads 0
set total_loads 0
set cross_lane_loads 0
set cross_bank_loads 0
set max_load_lanes 0
set table [open [file join $output_dir accumulator_ce_locality.tsv] w]
puts $table "loads\tdriver\tdriver_lane\tdriver_bank\tload_lane_count\tload_lanes\tload_banks\tlocal_lane_loads\tlocal_bank_loads\tnet"

foreach net $ce_nets {
    set loads [get_pins -quiet -of_objects $net -filter {DIRECTION == IN}]
    set drivers [get_pins -quiet -of_objects $net -filter {DIRECTION == OUT}]
    set load_count [llength $loads]
    set driver_name [join $drivers ,]
    set driver_lane -1
    set driver_bank ?
    regexp {gen_lane\[([0-9]+)\]\.acc([SC])_write_enable} \
        $driver_name -> driver_lane driver_bank

    set load_lanes {}
    set load_banks {}
    set local_lane_loads 0
    set local_bank_loads 0
    foreach load $loads {
        set load_name [get_property NAME $load]
        set load_lane -1
        set load_bank ?
        if {[regexp {gen_lane\[([0-9]+)\]\.gen_acc\[[0-9]+\]\.acc([SC])_reg} \
                $load_name -> load_lane load_bank]} {
            lappend load_lanes $load_lane
            lappend load_banks $load_bank
            if {$load_lane == $driver_lane} { incr local_lane_loads }
            if {$load_bank eq $driver_bank} { incr local_bank_loads }
        }
    }
    set load_lanes [lsort -integer -unique $load_lanes]
    set load_banks [lsort -unique $load_banks]
    set load_lane_count [llength $load_lanes]

    if {$load_count < $min_loads} { set min_loads $load_count }
    if {$load_count > $max_loads} { set max_loads $load_count }
    if {$load_lane_count > $max_load_lanes} { set max_load_lanes $load_lane_count }
    incr total_loads $load_count
    incr cross_lane_loads [expr {$load_count - $local_lane_loads}]
    incr cross_bank_loads [expr {$load_count - $local_bank_loads}]
    puts $table "$load_count\t$driver_name\t$driver_lane\t$driver_bank\t$load_lane_count\t[join $load_lanes ,]\t[join $load_banks ,]\t$local_lane_loads\t$local_bank_loads\t$net"
}
close $table

set failures {}
if {[llength $leaf_regs] == 0} { lappend failures "no accumulator enable leaf registers matched" }
if {[llength $ce_pins] == 0} { lappend failures "no accumulator CE pins matched" }
if {[llength $ce_nets] == 0} {
    lappend failures "no accumulator CE nets matched"
    set min_loads 0
}
set status [expr {[llength $failures] ? "FAIL" : "PASS"}]

set summary [open [file join $output_dir accumulator_ce_locality_summary.txt] w]
puts $summary "status=$status"
puts $summary "leaf_enable_registers=[llength $leaf_regs]"
puts $summary "accumulator_registers_with_ce=[llength $ce_pins]"
puts $summary "accumulator_ce_nets=[llength $ce_nets]"
puts $summary "accumulator_ce_loads_total=$total_loads"
puts $summary "accumulator_ce_loads_min=$min_loads"
puts $summary "accumulator_ce_loads_max=$max_loads"
puts $summary "max_load_lanes_per_ce_net=$max_load_lanes"
puts $summary "cross_lane_loads=$cross_lane_loads"
puts $summary "cross_bank_loads=$cross_bank_loads"
if {[llength $failures]} { puts $summary "failures=[join $failures {; }]" }
close $summary

puts "ROUTED_ACC_FANOUT leaf_regs=[llength $leaf_regs] ce_nets=[llength $ce_nets] loads=$min_loads..$max_loads cross_lane=$cross_lane_loads cross_bank=$cross_bank_loads"
if {[llength $failures]} { error "GEMM accumulator check failed: [join $failures {; }]" }
