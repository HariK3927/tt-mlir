// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Utils/Utils.h"

#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Types/Types.h"
#include "ttmlir/Utils.h"

#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Value.h"
#include "llvm/Support/Casting.h"

#include <optional>

namespace mlir::tt::ttnn::utils {

bool isTensorOnDevice(::mlir::RankedTensorType tensorType) {
  auto ttnnLayoutAttr =
      ::mlir::cast<ttnn::TTNNLayoutAttr>(tensorType.getEncoding());
  bool isOnDevice =
      ttnnLayoutAttr.getBufferType() != ttnn::BufferType::SystemMemory;
  return isOnDevice;
}

// Map ttcore::MemorySpace to TTNN::BufferType
//
mlir::tt::ttnn::BufferType
toTTNNBufferType(const mlir::tt::ttcore::MemorySpace memorySpace) {
  switch (memorySpace) {
  case mlir::tt::ttcore::MemorySpace::System:
  case mlir::tt::ttcore::MemorySpace::SystemMMIO:
    return BufferType::SystemMemory;
  case mlir::tt::ttcore::MemorySpace::DeviceDRAM:
    return BufferType::DRAM;
  case mlir::tt::ttcore::MemorySpace::DeviceL1:
    return BufferType::L1;
  case mlir::tt::ttcore::MemorySpace::RegisterDst:
    llvm_unreachable("MemorySpace::RegisterDst not supported");
  }

  llvm_unreachable("Unknown MemorySpace");
}

mlir::tt::ttcore::MemorySpace
toTTMemorySpace(const mlir::tt::ttnn::BufferType bufferType) {
  switch (bufferType) {
  case ttnn::BufferType::SystemMemory:
    return mlir::tt::ttcore::MemorySpace::System;
  case ttnn::BufferType::DRAM:
    return mlir::tt::ttcore::MemorySpace::DeviceDRAM;
  case ttnn::BufferType::L1:
    return mlir::tt::ttcore::MemorySpace::DeviceL1;
  case ttnn::BufferType::L1Small:
    assert(false && "BufferType::L1Small not supported");
  case ttnn::BufferType::Trace:
    assert(false && "BufferType::Trace not supported");
  }

  llvm_unreachable("Unknown MemorySpace");
}

RankedTensorType
RankedTensorTypeFactory::create(RankedTensorType tensorType,
                                ttnn::TTNNLayoutAttr encoding) {
  return RankedTensorType::get(tensorType.getShape(),
                               tensorType.getElementType(), encoding);
}

RankedTensorType RankedTensorTypeFactory::create(RankedTensorType tensorType,
                                                 Type memrefElementType) {
  TTNNLayoutAttr oldEncoding = getLayoutAttrFromTensor(tensorType);
  TTNNLayoutAttr newEncoding =
      oldEncoding.withElementType(memrefElementType, tensorType.getShape());
  Type newElementType = memrefElementType;
  if (ttcore::TileType tileType = dyn_cast<ttcore::TileType>(newElementType)) {
    newElementType = tileType.getElementType();
  }
  return RankedTensorType::get(tensorType.getShape(), newElementType,
                               newEncoding);
}

RankedTensorType RankedTensorTypeFactory::create(RankedTensorType tensorType,
                                                 ttnn::BufferType bufferType) {
  TTNNLayoutAttr oldEncoding = getLayoutAttrFromTensor(tensorType);
  TTNNLayoutAttr newEncoding = oldEncoding.withBufferType(bufferType);
  return create(tensorType, newEncoding);
}

RankedTensorType
RankedTensorTypeFactory::create(RankedTensorType tensorType,
                                ttnn::TensorMemoryLayout memoryLayout) {
  TTNNLayoutAttr oldEncoding = getLayoutAttrFromTensor(tensorType);
  TTNNLayoutAttr newEncoding = oldEncoding.withMemoryLayout(memoryLayout);
  return create(tensorType, newEncoding);
}

RankedTensorType RankedTensorTypeFactory::create(RankedTensorType tensorType,
                                                 Layout layout) {
  // If the tensor is quantized, only update the layout in the encoding.
  if (auto quantType =
          dyn_cast<mlir::quant::QuantizedType>(tensorType.getElementType())) {
    ttcore::DataType dataType =
        mlir::tt::ttcore::elementTypeToDataType(quantType);
    Type memrefElementType =
        utils::getElementType(tensorType.getContext(), layout, dataType);
    TTNNLayoutAttr oldEncoding = getLayoutAttrFromTensor(tensorType);
    TTNNLayoutAttr newEncoding =
        oldEncoding.withElementType(memrefElementType, tensorType.getShape());
    return RankedTensorType::get(tensorType.getShape(), quantType, newEncoding);
  }
  ttcore::DataType dataType =
      mlir::tt::ttcore::elementTypeToDataType(tensorType.getElementType());
  Type memrefElementType =
      utils::getElementType(tensorType.getContext(), layout, dataType);
  return create(tensorType, memrefElementType);
}

RankedTensorType RankedTensorTypeFactory::create(RankedTensorType tensorType,
                                                 ttcore::GridAttr grid) {
  TTNNLayoutAttr oldEncoding = getLayoutAttrFromTensor(tensorType);
  TTNNLayoutAttr newEncoding =
      oldEncoding.withGrid(tensorType.getShape(), grid);
  return create(tensorType, newEncoding);
}

RankedTensorType RankedTensorTypeFactory::create(RankedTensorType tensorType,
                                                 ttcore::DataType dataType) {
  TTNNLayoutAttr oldEncoding = getLayoutAttrFromTensor(tensorType);
  Type nmemrefElementType = utils::getElementType(
      tensorType.getContext(), oldEncoding.getLayout(), dataType);
  return create(tensorType, nmemrefElementType);
}

RankedTensorType
RankedTensorTypeFactory::create(RankedTensorType tensorType,
                                ArrayRef<int64_t> tensorShape) {
  TTNNLayoutAttr oldEncoding = getLayoutAttrFromTensor(tensorType);
  TTNNLayoutAttr newEncoding = oldEncoding.withTensorShape(tensorShape);
  return RankedTensorType::get(tensorShape, tensorType.getElementType(),
                               newEncoding);
}

// Return the L1 memory usage of the output tensor of the given op.
// Used within L1 interleaved policies.
//
uint64_t getOpOutputL1Usage(TTNNLayoutAttr opLayout) {
  // In case the opLayout is not in L1 memory space, L1 memory usage is 0.
  //
  if (opLayout.hasDRAMBufferType()) {
    return 0;
  }

  return opLayout.getShardSizeInBytes();
}

// Helper method to get the tensor layout attribute from the value.
TTNNLayoutAttr getLayoutAttrFromTensor(RankedTensorType tensorType) {
  return mlir::cast<TTNNLayoutAttr>(tensorType.getEncoding());
}

// Helper method to get the element type for the given tensor layout and data.
Type getElementType(MLIRContext *context, Layout tensorLayout,
                    ttcore::DataType dataType) {
  return tensorLayout == Layout::Tile
             ? ttcore::TileType::get(
                   context, {ttnn::TILE_HEIGHT, ttnn::TILE_WIDTH}, dataType)
             : mlir::tt::ttcore::dataTypeToElementType(context, dataType);
}

// Save the IR to a file for debugging.
void irToFile(mlir::Operation *op, std::string filename) {
  OpPrintingFlags printFlags;
  printFlags = printFlags.enableDebugInfo();

  std::error_code ec;
  llvm::raw_fd_ostream file(filename, ec);
  if (ec) {
    llvm::errs() << "Error opening file: " << ec.message() << "\n";
    return;
  }
  op->print(file, printFlags);
}

std::string getOpLocName(Operation *op) {
  if (NameLoc loc = llvm::dyn_cast<NameLoc>(op->getLoc())) {
    return loc.getName().str();
  }
  return "";
}

llvm::SmallVector<int64_t> getTilePaddedShape(llvm::ArrayRef<int64_t> shape) {
  llvm::SmallVector<int64_t, 4> tiledShape(shape);
  const size_t rank = shape.size();
  if (rank > 0) {
    tiledShape[shape.size() - 1] =
        ttmlir::utils::alignUp<int64_t>(shape[shape.size() - 1], TILE_WIDTH);
  }
  if (rank > 1) {
    tiledShape[shape.size() - 2] =
        ttmlir::utils::alignUp<int64_t>(shape[shape.size() - 2], TILE_HEIGHT);
  }
  return tiledShape;
}

// Helper method to create a ShardSpecAttr if needed.
std::optional<ShardSpecAttr>
createShardSpecIfNeeded(TTNNLayoutAttr layoutAttr,
                        ttcore::GridAttr deviceGridAttr) {
  std::optional<ShardSpecAttr> shardSpecAttr = std::nullopt;
  TensorMemoryLayoutAttr tensorMemoryLayout = layoutAttr.getMemLayout();
  if (tensorMemoryLayout &&
      isShardedMemoryLayout(tensorMemoryLayout.getValue())) {
    shardSpecAttr =
        ShardSpecAttr::get(layoutAttr.getContext(), layoutAttr, deviceGridAttr);
  }
  return shardSpecAttr;
}

// Helper method to create a ShardSpecAttr if needed.
std::optional<ShardSpecAttr> createShardSpecIfNeeded(
    TensorMemoryLayoutAttr tensorMemoryLayoutAttr, ShapeAttr shardShapeAttr,
    ttcore::GridAttr shardGridAttr, ttcore::GridAttr deviceGridAttr) {
  std::optional<ShardSpecAttr> shardSpecAttr = std::nullopt;
  if (tensorMemoryLayoutAttr &&
      isShardedMemoryLayout(tensorMemoryLayoutAttr.getValue())) {
    shardSpecAttr =
        ShardSpecAttr::get(tensorMemoryLayoutAttr.getContext(), shardShapeAttr,
                           shardGridAttr, deviceGridAttr);
  }
  return shardSpecAttr;
}

bool isTTNNTraceFunc(func::FuncOp funcOp) {
  return funcOp->hasAttr(g_TTNNTraceAttrName);
}

// Converts TTNNLayoutAttr to RowMajor layout and returns new layout.
TTNNLayoutAttr convertTTNNLayoutToRowMajor(MLIRContext *context,
                                           TTNNLayoutAttr layout,
                                           llvm::ArrayRef<int64_t> shape) {
  Type elementType =
      utils::getElementType(context, Layout::RowMajor, layout.getDataType());
  return layout.withElementType(elementType, shape);
}

std::set<mlir::StringRef> getAllTTNNDialectOps(MLIRContext *context) {
  std::set<mlir::StringRef> opNames;
  TTNNDialect *dialect = context->getLoadedDialect<TTNNDialect>();

  // We should use getRegisteredOperationsByDialect but it has a bug in MLIR.
  // See https://github.com/llvm/llvm-project/issues/146940.
  for (mlir::RegisteredOperationName opName :
       context->getRegisteredOperations()) {
    if (opName.getDialectNamespace() != dialect->getNamespace()) {
      continue;
    }
    opNames.insert(opName.getStringRef());
  }
  return opNames;
}

} // namespace mlir::tt::ttnn::utils
