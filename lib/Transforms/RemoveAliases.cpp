#include "llvm/ADT/STLExtras.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Ops.h"
#include "p4mlir/Transforms/DialectConversion.h"
#include "p4mlir/Transforms/Passes.h"

#define DEBUG_TYPE "p4hir-remove-alias"

using namespace mlir;

namespace P4::P4MLIR {
#define GEN_PASS_DEF_REMOVEALIASES
#include "p4mlir/Transforms/Passes.cpp.inc"
}  // namespace P4::P4MLIR

using namespace P4::P4MLIR;

namespace {
struct RemoveAliasesPass : public P4::P4MLIR::impl::RemoveAliasesBase<RemoveAliasesPass> {
    RemoveAliasesPass() = default;
    void runOnOperation() override;
};

struct ConstOpConversion : public OpConversionPattern<P4HIR::ConstOp> {
    using OpConversionPattern<P4HIR::ConstOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(P4HIR::ConstOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        Type newType = getTypeConverter()->convertType(op.getValue().getType());

        TypedAttr newAttr;
        if (auto intAttr = dyn_cast<P4HIR::IntAttr>(op.getValue())) {
            newAttr = P4HIR::IntAttr::get(newType, intAttr.getValue());
        } else if (auto boolAttr = dyn_cast<P4HIR::BoolAttr>(op.getValue())) {
            newAttr = P4HIR::BoolAttr::get(getContext(), newType, boolAttr.getValue());
        } else if (auto errorCodeAttr = dyn_cast<P4HIR::ErrorCodeAttr>(op.getValue())) {
            newAttr = P4HIR::ErrorCodeAttr::get(newType, errorCodeAttr.getField());
        } else if (auto aggAttr = dyn_cast<P4HIR::AggAttr>(op.getValue())) {
            newAttr = P4HIR::AggAttr::get(newType, aggAttr.getFields());
        } else {
            return rewriter.notifyMatchFailure(op, "unhandled attribute kind for conversion");
        }

        rewriter.replaceOpWithNewOp<P4HIR::ConstOp>(op, newAttr, op.getNameAttr(),
                                                    op.getAnnotationsAttr());

        return success();
    }
};

}  // namespace

void RemoveAliasesPass::runOnOperation() {
    mlir::ModuleOp module = getOperation();
    MLIRContext &context = getContext();

    ConversionTarget target(context);

    TypeConverter typeConverter;
    typeConverter.addConversion([&](Type type) -> Type {
        if (auto aliasType = mlir::dyn_cast<P4HIR::AliasType>(type)) {
            return typeConverter.convertType(aliasType.getAliasedType());
        }
        return type;
    });

    target.addDynamicallyLegalOp<P4HIR::FuncOp>([&](P4HIR::FuncOp func) {
        auto fnType = func.getFunctionType();
        return typeConverter.isLegal(fnType.getInputs()) &&
               typeConverter.isLegal(fnType.getReturnTypes());
    });

    // target.addDynamicallyLegalOp<P4HIR::CaseOp>([&](P4HIR::CaseOp caseOp) {
    //     return llvm::all_of(caseOp.getValue(), [&](Attribute val) {
    //         if (auto typedAttr = mlir::dyn_cast<mlir::TypedAttr>(val)) {
    //             return typeConverter.isLegal(typedAttr.getType());
    //         }
    //         return true;
    //     });
    // });

    target.markUnknownOpDynamicallyLegal([&](Operation *op) {
        return typeConverter.isLegal(op->getOperandTypes()) &&
               typeConverter.isLegal(op->getResultTypes());
    });

    RewritePatternSet patterns(&context);
    patterns.add<ConstOpConversion>(typeConverter, &context);
    P4::P4MLIR::populateFunctionOpInterfaceTypeConversionPattern<P4HIR::FuncOp>(patterns,
                                                                                typeConverter);
    populateGenericOpTypeConversionPattern<P4HIR::CallOp, P4HIR::InstantiateOp, P4HIR::ApplyOp,
                                           P4HIR::VariableOp, P4HIR::AssignOp, P4HIR::ReadOp,
                                           P4HIR::CmpOp>(patterns, typeConverter);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) signalPassFailure();
}

std::unique_ptr<Pass> P4::P4MLIR::createRemoveAliasesPass() {
    return std::make_unique<RemoveAliasesPass>();
}
