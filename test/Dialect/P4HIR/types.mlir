// RUN: p4mlir-opt %s | FileCheck %s

!unknown = !p4hir.unknown
!error = !p4hir.error
!dontcare = !p4hir.dontcare

// No need to check stuff. If it parses, it's fine.
// CHECK: module
module {
}
