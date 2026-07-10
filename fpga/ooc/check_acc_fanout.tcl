# Synthesize gemm_rowblock out of context and verify the accumulator CE topology.
# Usage: vivado -mode batch -source check_gemm_acc_fanout.tcl -tclargs <rtl-dir> <report-dir>

if {$argc != 2} {
    error "usage: check_gemm_acc_fanout.tcl <rtl-dir> <report-dir>"
}

set rtl_dir [file normalize [lindex $argv 0]]
set report_dir [file normalize [lindex $argv 1]]
file mkdir $report_dir

read_verilog [file join $rtl_dir fma.v]
read_verilog [file join $rtl_dir gemm.v]
synth_design -mode out_of_context -top gemm_rowblock -part xck26-sfvc784-2LV-c

set leaf_regs [get_cells -quiet -hier -filter {
    NAME =~ *accS_write_enable_reg* || NAME =~ *accC_write_enable_reg*
}]
set acc_regs [get_cells -quiet -hier -filter {
    NAME =~ *gen_acc*accS_reg* || NAME =~ *gen_acc*accC_reg*
}]

set ce_pins {}
foreach cell $acc_regs {
    set pin [get_pins -quiet [get_property NAME $cell]/CE]
    if {[llength $pin]} { lappend ce_pins $pin }
}
set ce_nets [lsort -unique [get_nets -quiet -of_objects $ce_pins]]

set min_loads 1000000000
set max_loads 0
set total_loads 0
set table [open [file join $report_dir accumulator_ce_nets.tsv] w]
puts $table "loads\tdriver\tnet"
foreach net $ce_nets {
    set loads [get_pins -quiet -of_objects $net -filter {DIRECTION == IN}]
    set load_count [llength $loads]
    set drivers [get_pins -quiet -of_objects $net -filter {DIRECTION == OUT}]
    if {$load_count < $min_loads} { set min_loads $load_count }
    if {$load_count > $max_loads} { set max_loads $load_count }
    incr total_loads $load_count
    puts $table "$load_count\t[join $drivers ,]\t$net"
}
close $table

set summary [open [file join $report_dir summary.txt] w]
puts $summary "leaf_enable_registers=[llength $leaf_regs] expected=256"
puts $summary "accumulator_registers_with_ce=[llength $ce_pins] rtl_upper_bound=26624"
puts $summary "accumulator_ce_nets=[llength $ce_nets] expected=256"
puts $summary "accumulator_ce_loads_total=$total_loads rtl_upper_bound=26624"
puts $summary "accumulator_ce_loads_min=$min_loads"
puts $summary "accumulator_ce_loads_max=$max_loads"
close $summary

report_utilization -file [file join $report_dir utilization.rpt]
report_high_fanout_nets -fanout_greater_than 100 -max_nets 1000 \
    -file [file join $report_dir high_fanout_nets.rpt]

puts "ACC_FANOUT_CHECK leaf_regs=[llength $leaf_regs] acc_regs_with_ce=[llength $ce_pins] ce_nets=[llength $ce_nets] loads=$min_loads..$max_loads"
