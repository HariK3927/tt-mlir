// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/optimization_barrier/optimization_barrier.h"

namespace tt::runtime::ttnn::operations::optimization_barrier {
void run(const ::tt::target::ttnn::OptimizationBarrierOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  ::ttnn::Tensor &input = tensorPool.getTTNNTensorAndValidate(op->in());
  tensorPool.insertTTNNTensorAndValidate(op->out(), input);  
}
} // namespace tt::runtime::ttnn::operations::optimization_barrier
