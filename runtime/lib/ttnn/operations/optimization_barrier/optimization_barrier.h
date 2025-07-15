// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef RUNTIME_LIB_TTNN_OPERATIONS_OPTIMIZATION_BARRIER_H
#define RUNTIME_LIB_TTNN_OPERATIONS_OPTIMIZATION_BARRIER_H

#include "tt/runtime/detail/ttnn/types.h"
#include "ttmlir/Target/TTNN/program_generated.h"

namespace tt::runtime::ttnn::operations::optimization_barrier {
void run(const ::tt::target::ttnn::OptimizationBarrierOp *op, ProgramContext &context);
} // namespace tt::runtime::ttnn::operations::optimization_barrier

#endif