// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/pool/upsample.h"

#include "tt/runtime/detail/common/logger.h"

#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"

namespace tt::runtime::ttnn::operations::pool {
void run(const ::tt::target::ttnn::UpsampleOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  ::ttnn::Tensor &input = tensorPool.getTTNNTensorAndValidate(op->in());

  std::variant<int32_t, std::array<uint32_t, 2>> scaleFactor;
  if (op->scale_factor_type() == ::tt::target::ttnn::Scale2D::UniformScale2D) {
    scaleFactor = op->scale_factor_as_UniformScale2D()->scale();
  } else if (op->scale_factor_type() ==
             ::tt::target::ttnn::Scale2D::NonUniformScale2D) {
    std::array<uint32_t, 2> scaleHW;
    const ::flatbuffers::Vector<int32_t> *fbScaleFactor =
        op->scale_factor_as_NonUniformScale2D()->scale();
    std::copy(fbScaleFactor->begin(), fbScaleFactor->end(), scaleHW.begin());
    scaleFactor = scaleHW;
  } else {
    LOG_FATAL("Invalid scale factor type");
  }

  std::string mode = op->mode()->str();

  std::optional<::ttnn::MemoryConfig> memoryConfig =
      op->memory_config()
          ? ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
                op->memory_config())
          : ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
                ::tt::runtime::ttnn::utils::getTensorRefMemoryConfig(
                    op->out()));

  ::ttnn::Tensor output =
      ::ttnn::upsample(input, scaleFactor, mode, memoryConfig);

  tensorPool.insertTTNNTensorAndValidate(op->out(), output);
}
} // namespace tt::runtime::ttnn::operations::pool
