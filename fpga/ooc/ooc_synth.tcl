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

set_param general.maxThreads 8
set part xck26-sfvc784-2LV-c

if {$argc < 4} {
    error "usage: ooc_synth.tcl <top> <period_ns> <out_prefix> <rtl...>"
}

set top     [lindex $argv 0]
set period  [lindex $argv 1]
set outpfx  [lindex $argv 2]
set rtls    [lrange $argv 3 end]

puts "==> OOC synth: top=$top period=${period}ns part=$part"
puts "==> rtl: $rtls"

foreach f $rtls {
    if {[file extension $f] ne ".vh"} { read_verilog $f }
}

synth_design -mode out_of_context -part $part -top $top

create_clock -name clk -period $period [get_ports clk]
set_false_path -from [all_inputs]
set_false_path -to   [all_outputs]

set util_report [report_utilization -return_string]
set util_fh [open ${outpfx}_util.rpt w]
puts -nonewline $util_fh $util_report
close $util_fh
report_timing_summary   -max_paths 5 -file ${outpfx}_timing.rpt

# ---- machine-readable one-liner to stdout ----
proc utilization_used {report label} {
    set pattern [format {\|[[:space:]]*%s[[:space:]]*\|[[:space:]]*([0-9]+)[[:space:]]*\|} $label]
    if {![regexp -- $pattern $report -> used]} {
        error "could not find utilization row '$label'"
    }
    return $used
}
proc utilization_used_or_zero {report label} {
    set pattern [format {\|[[:space:]]*%s[[:space:]]*\|[[:space:]]*([0-9]+)[[:space:]]*\|} $label]
    if {![regexp -- $pattern $report -> used]} {
        return 0
    }
    return $used
}
set luts   [utilization_used $util_report {CLB LUTs\*?}]
set carry8 [utilization_used $util_report {CARRY8}]
set dsps   [utilization_used $util_report {DSPs}]
set ffs    [utilization_used $util_report {Register as Flip Flop}]
set bram36 [utilization_used $util_report {RAMB36/FIFO\*}]
set bram18 [utilization_used $util_report {RAMB18}]
set uram   [utilization_used $util_report {URAM}]
# Vivado omits this hierarchy row entirely when no distributed RAM is present.
set lutram [utilization_used_or_zero $util_report {LUT as Distributed RAM}]

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

puts "RESULT top=$top dsp=$dsps lut=$luts carry8=$carry8 ff=$ffs bram36=$bram36 bram18=$bram18 uram=$uram lutram=$lutram wns_ns=$wns fmax_mhz=$fmax (period=${period}ns)"

set metrics [open ${outpfx}_metrics.tsv w]
puts $metrics "key\tvalue"
foreach {key value} [list \
        top $top period_ns $period part $part vivado_version [version -short] \
        dsp $dsps lut $luts carry8 $carry8 ff $ffs bram36 $bram36 bram18 $bram18 \
        uram $uram lutram $lutram wns_ns $wns fmax_mhz $fmax] {
    puts $metrics "$key\t$value"
}
close $metrics

set summary [open ${outpfx}_summary.txt w]
set status [expr {$wns ne "n/a" && $wns >= 0.0 ? "PASS" : "FAIL"}]
if {$top eq "section_f32_scratch_ooc" &&
    ($uram != 16 || $bram36 != 0 || $bram18 != 0 || $lutram != 0)} {
    set status "FAIL"
}
puts $summary "OOC $status top=$top period_ns=$period wns_ns=$wns fmax_mhz=$fmax"
puts $summary "dsp=$dsps lut=$luts carry8=$carry8 ff=$ffs"
puts $summary "bram36=$bram36 bram18=$bram18 uram=$uram lutram=$lutram"
close $summary

if {$status ne "PASS"} {
    error "OOC gate failed: top=$top period=${period}ns wns=${wns}ns bram36=$bram36 bram18=$bram18 uram=$uram lutram=$lutram"
}
