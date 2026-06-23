# ooc_synth.tcl - out-of-context synthesis probe (no place/route, ~1-2 min).
#
# Quantifies DSP / LUT / CARRY8 / FF and a synth-level Fmax for one module, to
# derisk the plan-7 phase-2 fixed-point-accumulate bet before a 30-min build.
#
#   vivado -mode batch -source ooc_synth.tcl -tclargs <top> <period_ns> <out_prefix> <rtl...>
#
# I/O is false-pathed so the reported worst path is internal reg->reg logic only
# (the accumulate recurrence / barrel shift / DSP path) -- fair across tops with
# different port shapes and not polluted by unconstrained I/O.

set part xck26-sfvc784-2LV-c

set top     [lindex $argv 0]
set period  [lindex $argv 1]
set outpfx  [lindex $argv 2]
set rtls    [lrange $argv 3 end]

puts "==> OOC synth: top=$top period=${period}ns part=$part"
puts "==> rtl: $rtls"

foreach f $rtls { read_verilog $f }

synth_design -mode out_of_context -part $part -top $top

create_clock -name clk -period $period [get_ports clk]
set_false_path -from [all_inputs]
set_false_path -to   [all_outputs]

report_utilization      -file ${outpfx}_util.rpt
report_timing_summary   -max_paths 5 -file ${outpfx}_timing.rpt

# ---- machine-readable one-liner to stdout ----
set luts   [llength [get_cells -hier -quiet -filter {PRIMITIVE_GROUP == LUT}]]
set carry8 [llength [get_cells -hier -quiet -filter {REF_NAME == CARRY8}]]
set dsps   [llength [get_cells -hier -quiet -filter {PRIMITIVE_GROUP == DSP || REF_NAME =~ DSP*}]]
set ffs    [llength [get_cells -hier -quiet -filter {PRIMITIVE_GROUP == REGISTER || PRIMITIVE_SUBGROUP == SDR || REF_NAME =~ FD*}]]

set wns "n/a"
set fmax "n/a"
set paths [get_timing_paths -quiet -max_paths 1 -nworst 1 -setup]
if {[llength $paths] > 0} {
    set wns [get_property SLACK [lindex $paths 0]]
    if {$wns != ""} {
        set achieved [expr {$period - $wns}]
        if {$achieved > 0} { set fmax [format "%.1f" [expr {1000.0 / $achieved}]] }
    }
}

puts "RESULT top=$top dsp=$dsps lut=$luts carry8=$carry8 ff=$ffs wns_ns=$wns fmax_mhz=$fmax (period=${period}ns)"
