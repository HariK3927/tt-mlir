// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_TTNN_OPERATIONS_TRACE_BEGIN_TRACE_CAPTURE_H
#define TT_RUNTIME_TTNN_OPERATIONS_TRACE_BEGIN_TRACE_CAPTURE_H

#include "tt/runtime/detail/ttnn/types/types.h"
#include "ttmlir/Target/TTNN/Target.h"

namespace tt::runtime::ttnn::operations::trace {

void run(const ::tt::target::ttnn::BeginTraceCaptureOp *op,
         ProgramContext &context);

} // namespace tt::runtime::ttnn::operations::trace

#endif // TT_RUNTIME_TTNN_OPERATIONS_TRACE_BEGIN_TRACE_CAPTURE_H
