# Count logic-LUT primitives (REF_NAME LUT*) under the matmul emit/acc/reducer subtrees,
# so we know exactly what each ACCUM_DEPTH step (or the shared-reducer refactor) frees.
open_checkpoint combined_w512_p4_f200_wc300/combined_w512_p4_f200_wc300.runs/impl_1/design_1_wrapper_opt.dcp
proc nlut {pat} { return [llength [get_cells -hier -filter "REF_NAME =~ LUT* && NAME =~ $pat"]] }
puts "COUNT emit_sum   [nlut *u_rowblock*gen_emit_sum*]"
puts "COUNT acc_add    [nlut *u_rowblock*u_acc_add*]"
puts "COUNT reducer    [nlut *u_rowblock*u_reducer*]"
puts "COUNT rowblock   [nlut *u_rowblock*]"
puts "COUNT kernel_mm  [nlut *kernel_mm*]"
puts "COUNT kernel_fa  [nlut *kernel_fa*]"
exit
