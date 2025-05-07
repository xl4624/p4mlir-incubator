#ifndef P4MLIR_DIALECT_P4HIR_P4HIR_INTERFACES_TD
#define P4MLIR_DIALECT_P4HIR_P4HIR_INTERFACES_TD

include "mlir/IR/OpBase.td"

def AnnotatedInterface : OpInterface<"AnnotatedInterface"> {
  let description = [{
    Interface for operations that can have annotations.
  }];

  let methods = [
    InterfaceMethod<[{Returns true if the operation has annotations attached}], "bool", "hasAnnotations", (ins), [{
      auto ann = $_op.getAnnotations();
      return ann && !ann->empty();
    }]>,
  ];
}

#endif // P4MLIR_DIALECT_P4HIR_P4HIR_INTERFACES_TD
