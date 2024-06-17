// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_TTNNTOCPP_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_TTNNTOCPP_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

namespace mlir::tt::ttnn {
LogicalResult emitTTNNAsCpp(ModuleOp origOp, llvm::raw_ostream &os);
} // namespace mlir::tt::ttnn
#endif
