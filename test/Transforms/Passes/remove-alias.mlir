// RUN: p4mlir-opt --p4hir-remove-alias %s | FileCheck %s

!b32i = !p4hir.bit<32>
!b9i = !p4hir.bit<9>
// CHECK-NOT: !Narrow
!Narrow = !p4hir.alias<"Narrow", !b9i>
!Wide = !p4hir.alias<"Wide", !b32i>
#int10_b9i = #p4hir.int<10> : !b9i
#int3_b32i = #p4hir.int<3> : !b32i
#int192_Narrow = #p4hir.int<192> : !Narrow
// CHECK-LABEL: module
module {
  p4hir.func @process_narrow(%arg0: !Narrow) {
    p4hir.return
  }

  // CHECK: %[[PSA_CPU_PORT:.*]] = p4hir.const ["PSA_CPU_PORT"] #int192_b9i
  %PSA_CPU_PORT = p4hir.const ["PSA_CPU_PORT"] #int192_Narrow


}
