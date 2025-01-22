// RUN: p4mlir-opt %s | FileCheck %s

#true = #p4hir.bool<true> : !p4hir.bool

// No need to check stuff. If it parses, it's fine.
// CHECK: module
module {
  %0 = p4hir.const #true
  %1 = p4hir.const #p4hir.bool<false> : !p4hir.bool
}
