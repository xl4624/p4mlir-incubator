#ifndef P4MLIR_DIALECT_P4HIR_P4HIR_OPS_TD
#define P4MLIR_DIALECT_P4HIR_P4HIR_OPS_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "p4mlir/Dialect/P4HIR/P4HIR_Dialect.td"
include "p4mlir/Dialect/P4HIR/P4HIR_Types.td"
include "p4mlir/Dialect/P4HIR/P4HIR_Attrs.td"

//===----------------------------------------------------------------------===//
// Base P4HIR operation definition.
//===----------------------------------------------------------------------===//

class P4HIR_Op<string mnemonic, list<Trait> traits = []> :
        Op<P4HIR_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// P4HIR operation definitions.
//===----------------------------------------------------------------------===//

def P4HIR_ConstOp : P4HIR_Op<"const",
    [ConstantLike, Pure, AllTypesMatch<["value", "res"]>]> {
    let summary = "Defines a P4 constant";
    let description = [{
        The `p4hir.const` operation turns a literal into an SSA value.
        The data is attached to the operation as an attribute. It
        represents a constant declaration in P4.

        Example:

        ```mlir
        %0 = p4hir.const p4hir.int<-128> : !p4hir.int<8>
        ```
    }];

    // The constant operation takes an attribute as the only input.
    let arguments = (ins TypedAttrInterface:$value);

    // The constant operation returns a single value of AnyP4Type.
    let results = (outs AnyP4Type:$res);

    let assemblyFormat = "attr-dict $value";

    let hasVerifier = 1;
}

#endif // P4MLIR_DIALECT_P4HIR_P4HIR_OPS_TD
