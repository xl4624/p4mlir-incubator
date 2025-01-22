#ifndef P4MLIR_DIALECT_P4HIR_P4HIR_ATTRS_H
#define P4MLIR_DIALECT_P4HIR_P4HIR_ATTRS_H

// We explicitly do not use push / pop for diagnostic in
// order to propagate pragma further on
#pragma GCC diagnostic ignored "-Wunused-parameter"

#include "mlir/IR/BuiltinAttributes.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Types.h"

#define GET_ATTRDEF_CLASSES
#include "p4mlir/Dialect/P4HIR/P4HIR_Attrs.h.inc"

#endif  // P4MLIR_DIALECT_P4HIR_P4HIR_ATTRS_H
