#ifndef P4MLIR_DIALECT_P4HIR_P4HIR_INTERFACES_H
#define P4MLIR_DIALECT_P4HIR_P4HIR_INTERFACES_H

// We explicitly do not use push / pop for diagnostic in
// order to propagate pragma further on
#pragma GCC diagnostic ignored "-Wunused-parameter"

namespace P4::P4MLIR::P4HIR {
#include "p4mlir/Dialect/P4HIR/P4HIR_Interfaces.h.inc"
}  // namespace P4::P4MLIR::P4HIR

#include "mlir/IR/OpDefinition.h"

#endif  // P4MLIR_DIALECT_P4HIR_P4HIR_INTERFACES_H
