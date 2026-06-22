# Open the fully-linked post-opt_design checkpoint and dump hierarchical + flat util.
open_checkpoint combined_w512_p4_f200_wc300/combined_w512_p4_f200_wc300.runs/impl_1/design_1_wrapper_opt.dcp
report_utilization -hierarchical -hierarchical_depth 3 -file util_hier.rpt
report_utilization -file util_flat.rpt
puts "UTIL_REPORTS_DONE"
exit
