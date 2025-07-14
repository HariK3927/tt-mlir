// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Iterators.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIROPTIMIZETENSORLAYOUT
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

static ttcore::GridAttr getOptimalGrid(PatternRewriter &rewriter,
                                       ArrayRef<int64_t> memrefShape,
                                       ArrayRef<int64_t> deviceGridShape) {
  assert(memrefShape.size() == deviceGridShape.size());
  std::vector<int64_t> gridShape;
  for (size_t i = 0; i < memrefShape.size(); i++) {
    int64_t dim = memrefShape[i];
    for (size_t grid = deviceGridShape[i]; grid > 0; grid--) {
      if (dim % grid == 0) {
        gridShape.push_back(grid);
        break;
      }
    }
  }
  return rewriter.getAttr<ttcore::GridAttr>(gridShape);
}

static RankedTensorType applyGridShape(RankedTensorType tensorType,
                                       ArrayRef<int64_t> gridShape) {
  auto tensorEncoding =
      mlir::cast_if_present<ttcore::MetalLayoutAttr>(tensorType.getEncoding());
  assert(tensorEncoding && "Tensor type must have a MetalLayoutAttr encoding");

  auto logicalShape = tensorEncoding.getLogicalShape();

  auto newTensorEncoding = ttcore::MetalLayoutAttr::get(
      tensorType.getContext(), logicalShape, gridShape.size(),
      tensorEncoding.getOobVal(), tensorEncoding.getMemorySpace(),
      tensorEncoding.getCollapsedIntervals(),
      tensorEncoding.getDimAlignments());

  auto newPhysicalShape = ttcore::MetalLayoutAttr::derivePhysicalShape(
      logicalShape, gridShape, ttcore::getTensorTileShapeOrEmpty(tensorType),
      newTensorEncoding.getCollapsedIntervals(),
      newTensorEncoding.getDimAlignments());
  return RankedTensorType::get(newPhysicalShape, tensorType.getElementType(),
                               newTensorEncoding);
}

static RankedTensorType calculateOptimalLayoutForTensorType(
    PatternRewriter &rewriter, Value tensor,
    const SmallVector<int64_t> &workerGridShape) {
  RankedTensorType tensorType = mlir::cast<RankedTensorType>(tensor.getType());
  auto tensorEncoding =
      mlir::cast_if_present<ttcore::MetalLayoutAttr>(tensorType.getEncoding());
  assert(tensorEncoding && "Tensor type must have a MetalLayoutAttr encoding");
  ttcore::GridAttr optimalOutputGrid =
      getOptimalGrid(rewriter,
                     tensorEncoding.getUnshardedShape(
                         tensorEncoding.getGridShape(tensorType),
                         tensorEncoding.getShardShape(tensorType)),
                     workerGridShape);
  return applyGridShape(tensorType, optimalOutputGrid.getShape());
}

namespace {
struct TTIRGenericTensorLayoutRewriter : public OpRewritePattern<GenericOp> {
  TTIRGenericTensorLayoutRewriter(MLIRContext *context,
                                  SmallVector<int64_t> workerGridShape)
      : OpRewritePattern<GenericOp>(context), workerGridShape(workerGridShape) {
  }

  LogicalResult matchAndRewrite(GenericOp op,
                                PatternRewriter &rewriter) const final {

    // Update output tensor type
    assert(op->getResults().size() == 1 &&
           "Only one result tensor is supported for now");
    Type originalType = op->getResult(0).getType();
    auto newTensorType = calculateOptimalLayoutForTensorType(
        rewriter, op->getResult(0), workerGridShape);
    ttcore::MetalLayoutAttr metalLayout =
        mlir::cast<ttcore::MetalLayoutAttr>(newTensorType.getEncoding());
    SmallVector<int64_t> blockFactors(op.getIndexingMaps().size(), 1);
    // TODO(jdesousa): We have moved the subblock logic to the
    // GenericTileComputeLoops pass. Because of this, block factors will always
    // be 1. We need to add capability here to size block factors based on L1
    // availability.
    bool blockFactorsChanged = blockFactors != op.getBlockFactorsValue();
    if (op.getGrid().getShape() == metalLayout.getGridShape(newTensorType) &&
        !blockFactorsChanged) {
      return failure();
    }

    auto layout =
        mlir::cast<ttcore::MetalLayoutAttr>(newTensorType.getEncoding());
    rewriter.modifyOpInPlace(op, [&]() {
      // Update generic grid (match worker cores to output grid)
      op.setGridAttr(rewriter.getAttr<ttcore::GridAttr>(
          layout.getGridShape(newTensorType)));
      op.setBlockFactorsAttr(rewriter.getI64ArrayAttr(blockFactors));
    });

    auto dpsOp = mlir::cast<DestinationStyleOpInterface>(op.getOperation());
    assert(dpsOp.getNumDpsInits() == 1 &&
           "Only one result tensor is supported for now");
    for (OpOperand &operand : op->getOpOperands()) {
      auto newOperandType = calculateOptimalLayoutForTensorType(
          rewriter, operand.get(), workerGridShape);
      if (operand.get().getType() != newOperandType || blockFactorsChanged) {
        Value view =
            blockedView(rewriter, op->getLoc(), operand.get(), newOperandType,
                        op.getIndexingMapsValue()[operand.getOperandNumber()],
                        blockFactors);
        rewriter.modifyOpInPlace(op, [&]() { operand.set(view); });

        if (dpsOp.isDpsInit(&operand)) {
          assert(newOperandType == newTensorType &&
                 "DPS init tensor must have the same type as the result");
          rewriter.modifyOpInPlace(
              op, [&]() { op->getResult(0).setType(view.getType()); });
        }

        for (auto &region : op->getRegions()) {
          assert(region.getBlocks().size() == 1 &&
                 "Only one block per region is supported.");
          Block &genericBlock = region.front();
          auto arg = genericBlock.getArgument(operand.getOperandNumber());
          rewriter.modifyOpInPlace(op, [&]() {
            arg.setType(ttcore::MetalLayoutAttr::getMemRefType(
                mlir::cast<RankedTensorType>(view.getType())));
          });
        }
      }
    }

    rewriter.setInsertionPointAfter(op);
    auto emptyOp = rewriter.create<EmptyOp>(op->getLoc(), originalType);
    auto toLayoutOp = rewriter.create<ToLayoutOp>(
        op->getLoc(), op->getResult(0), emptyOp.getResult());
    rewriter.replaceAllUsesExcept(op->getResult(0), toLayoutOp.getResult(0),
                                  toLayoutOp);

    return success();
  }

  static Value blockedView(PatternRewriter &rewriter, Location loc,
                           Value tensor, RankedTensorType newOperandType,
                           AffineMap indexingMap,
                           ArrayRef<int64_t> blockFactors) {
    auto emptyOp = rewriter.create<EmptyOp>(loc, newOperandType);
    auto toLayoutOp =
        rewriter.create<ToLayoutOp>(loc, tensor, emptyOp.getResult());
    ttcore::MetalLayoutAttr metalLayout =
        mlir::cast<ttcore::MetalLayoutAttr>(newOperandType.getEncoding());
    SmallVector<int64_t> blockShape = indexingMap.compose(blockFactors);
    for (auto [i, dim] :
         llvm::enumerate(metalLayout.getGridShape(newOperandType))) {
      // Handle the edge case where a 0 constant appears in the affine map, i.e.
      // some kind of reduction or broadcast:
      //   (d0, d1) -> (d0, 0)
      if (blockShape[i] == 0) {
        blockShape[i] = 1;
      }
      blockShape[i] *= dim;
    }
    auto viewOperandType = applyGridShape(newOperandType, blockShape);
    return rewriter
        .create<ViewLayoutOp>(loc, viewOperandType, toLayoutOp.getResult(0))
        .getResult();
  }

  SmallVector<int64_t> workerGridShape;
};
} // namespace

// This pass rewrites ToLayoutOps that are host transactions (host tensor ->
// device / device tensor -> host) with the largest possible grid. This enables
// us to load much larger tensors to device, by reading/writing them directly
// from/to multiple cores, instead of forcing the default <1x1> grid.
namespace {
struct TTIRHostTxsRewriter : public OpRewritePattern<ToLayoutOp> {
  TTIRHostTxsRewriter(MLIRContext *context,
                      SmallVector<int64_t> workerGridShape)
      : OpRewritePattern<ToLayoutOp>(context),
        workerGridShape(workerGridShape) {}

public:
  LogicalResult matchAndRewrite(ToLayoutOp op,
                                PatternRewriter &rewriter) const override {

    auto inputTy = mlir::cast<RankedTensorType>(op.getInput().getType());
    auto outputTy = mlir::cast<RankedTensorType>(op.getOutput().getType());
    ttcore::MetalLayoutAttr inputMemoryLayout =
        mlir::dyn_cast_if_present<ttcore::MetalLayoutAttr>(
            inputTy.getEncoding());
    ttcore::MetalLayoutAttr outputMemoryLayout =
        mlir::dyn_cast_if_present<ttcore::MetalLayoutAttr>(
            outputTy.getEncoding());
    if (inputMemoryLayout && outputMemoryLayout) {
      // Not a host tx
      return failure();
    }

    auto deviceTensor = inputMemoryLayout ? op.getInput() : op.getOutput();
    auto optimalDeviceLayout = calculateOptimalLayoutForTensorType(
        rewriter, deviceTensor, workerGridShape);
    if (deviceTensor.getType() == optimalDeviceLayout) {
      return failure();
    }

    // Update device tensor type
    rewriter.modifyOpInPlace(op, [&]() {
      deviceTensor.setType(optimalDeviceLayout);
      if (outputMemoryLayout) {
        op->getResult(0).setType(optimalDeviceLayout);
      }
    });
    return success();
  }

  SmallVector<int64_t> workerGridShape;
};
} // namespace

namespace {
class TTIROptimizeTensorLayout
    : public impl::TTIROptimizeTensorLayoutBase<TTIROptimizeTensorLayout> {

  using impl::TTIROptimizeTensorLayoutBase<
      TTIROptimizeTensorLayout>::TTIROptimizeTensorLayoutBase;

  void runOnOperation() final {
    auto device = ttcore::lookupDevice(getOperation());
    assert(device && "Device not found");

    SmallVector<int64_t> workerGridShape = llvm::to_vector(overrideDeviceShape);
    if (workerGridShape.empty()) {
      workerGridShape = llvm::to_vector(device.getWorkerGrid().getShape());
    }

    RewritePatternSet patterns(&getContext());
    patterns.add<TTIRGenericTensorLayoutRewriter>(&getContext(),
                                                  workerGridShape);
    patterns.add<TTIRHostTxsRewriter>(&getContext(), workerGridShape);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::tt::ttir::TTIRDialect>();
    registry.insert<mlir::tt::ttcore::TTCoreDialect>();
  }
};
} // namespace

} // namespace mlir::tt::ttir
