// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_alchemist.hpp"

#include "tt-alchemist/tt_alchemist_c_api.hpp"

// MLIR includes
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"
#include "mlir/IR/MLIRContext.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"
#include "ttmlir/Dialect/TTNN/IR/TTNN.h"

namespace tt::alchemist {

// Singleton implementation
TTAlchemist &TTAlchemist::getInstance() {
  static TTAlchemist instance;
  return instance;
}

TTAlchemist::TTAlchemist() {
  mlir::DialectRegistry registry;
  mlir::func::registerInlinerExtension(registry);
  mlir::LLVM::registerInlinerInterface(registry);

  registry.insert<mlir::tt::ttcore::TTCoreDialect, mlir::tt::ttir::TTIRDialect,
                  mlir::tt::ttnn::TTNNDialect, mlir::func::FuncDialect,
                  mlir::emitc::EmitCDialect, mlir::LLVM::LLVMDialect>();
  context.appendDialectRegistry(registry);

  context.loadDialect<mlir::tt::ttcore::TTCoreDialect>();
  context.loadDialect<mlir::tt::ttir::TTIRDialect>();
  context.loadDialect<mlir::tt::ttnn::TTNNDialect>();
  context.loadDialect<mlir::func::FuncDialect>();
  context.loadDialect<mlir::emitc::EmitCDialect>();
  context.loadDialect<mlir::LLVM::LLVMDialect>();
}

} // namespace tt::alchemist

// C-compatible API implementations
extern "C" {

// Get the singleton instance
void *tt_alchemist_TTAlchemist_getInstance() {
  return static_cast<void *>(&tt::alchemist::TTAlchemist::getInstance());
}

} // extern "C"
