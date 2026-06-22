# Open the routed f200 checkpoint and list DISTINCT near-critical paths (slack < 1 ns),
# one per endpoint, as a compact summary — to scope how much pipelining f250 needs.
open_checkpoint combined_w512_p4_f200_wc300/combined_w512_p4_f200_wc300.runs/impl_1/design_1_wrapper_routed.dcp
report_timing -slack_lesser_than 1.0 -max_paths 60 -unique_pins -path_type summary -file nearcrit.rpt
puts "NEARCRIT_DONE"
exit
