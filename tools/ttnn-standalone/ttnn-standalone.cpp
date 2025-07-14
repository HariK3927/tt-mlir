// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn-precompiled.hpp"
using namespace tt::runtime::ttnn::operations::ccl::mesh_shard;
using namespace tt::runtime::ttnn::operations::ccl::point_to_point;
::ttnn::Tensor point_to_point_provide_output_tensor(::ttnn::Tensor v1) {
  ttnn::distributed::MeshDevice* v2 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v3 = mesh_shard(v1, *v2, MeshShardDirection::FullToShardShape, MeshShardType::Devices, {1, 1, 1, 8}, {-1, 3});
  ttnn::deallocate(v1, false);
  ::ttnn::Tensor v4 = ttnn::to_layout(v3, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ttnn::deallocate(v3, false);
  ::ttnn::Tensor v5 = ttnn::to_device(v3, v2, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::empty(::ttnn::Shape({1, 1, 256, 64}), ::ttnn::DataType::FLOAT32, ::ttnn::Layout::TILE, v2, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v7 = point_to_point(v5, ::ttnn::MeshCoordinate(ttsl::Span<const uint32_t>{0, 0}), ::ttnn::MeshCoordinate(ttsl::Span<const uint32_t>{0, 1}), v6);
  ttnn::deallocate(v6, false);
  ::ttnn::Tensor v8 = point_to_point(v5, ::ttnn::MeshCoordinate(ttsl::Span<const uint32_t>{0, 1}), ::ttnn::MeshCoordinate(ttsl::Span<const uint32_t>{0, 0}), v7);
  ttnn::deallocate(v7, false);
  ::ttnn::Tensor v9 = ttnn::from_device(v8);
  ttnn::deallocate(v8, false);
  ::ttnn::Tensor v10 = ttnn::to_layout(v9, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ttnn::deallocate(v9, false);
  ::ttnn::Tensor v11 = mesh_shard(v10, *v2, MeshShardDirection::ShardToFullShape, MeshShardType::Devices, {1, 1, 1, 8}, {-1, 3});
  ttnn::deallocate(v10, false);
  return v11;
}

::ttnn::Tensor point_to_point_case_generate_output_tensor(::ttnn::Tensor v1) {
  ttnn::distributed::MeshDevice* v2 = ttnn::DeviceGetter::getInstance();
  assert(0 && "Mesh shard operation is not supported in emitc yet."); // ::mesh_shard
  ::ttnn::Tensor v3 = mesh_shard(v1, *v2, MeshShardDirection::FullToShardShape, MeshShardType::Devices, {1, 1, 1, 8}, {-1, 3});
  ttnn::deallocate(v1, false);
  ::ttnn::Tensor v4 = ttnn::to_layout(v3, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ttnn::deallocate(v3, false);
  ::ttnn::Tensor v5 = ttnn::to_device(v3, v2, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = point_to_point(v5, ::ttnn::MeshCoordinate(ttsl::Span<const uint32_t>{0, 0}), ::ttnn::MeshCoordinate(ttsl::Span<const uint32_t>{0, 1}), ::std::nullopt);
  ::ttnn::Tensor v7 = point_to_point(v5, ::ttnn::MeshCoordinate(ttsl::Span<const uint32_t>{0, 1}), ::ttnn::MeshCoordinate(ttsl::Span<const uint32_t>{0, 0}), v6);
  ttnn::deallocate(v6, false);
  ::ttnn::Tensor v8 = ttnn::from_device(v7);
  ttnn::deallocate(v7, false);
  ::ttnn::Tensor v9 = ttnn::to_layout(v8, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ttnn::deallocate(v8, false);
  ::ttnn::Tensor v11 = mesh_shard(v9, *v2, MeshShardDirection::ShardToFullShape, MeshShardType::Devices, {1, 1, 1, 8}, {-1, 3});
  ttnn::deallocate(v9, false);
  return v9;
}


std::tuple<::ttnn::Tensor, ::ttnn::Tensor> create_inputs_for_add() {
  ttnn::distributed::MeshDevice *v1 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v2 =
      ttnn::ones(::ttnn::Shape({1, 1, 256, 512}), ::ttnn::DataType::FLOAT32,
                 ::ttnn::Layout::TILE, ::std::nullopt,
                 ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                      ::ttnn::BufferType::DRAM});
  // ::ttnn::Tensor v3 = ttnn::to_device(v2, v1,
  // ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
  // ::ttnn::BufferType::DRAM});
  ::ttnn::Tensor v4 =
      ttnn::ones(::ttnn::Shape({1, 1, 256, 512}), ::ttnn::DataType::FLOAT32,
                 ::ttnn::Layout::TILE, ::std::nullopt,
                 ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                      ::ttnn::BufferType::DRAM});
  // ::ttnn::Tensor v5 = ttnn::to_device(v4, v1,
  // ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
  // ::ttnn::BufferType::DRAM});
  return std::make_tuple(v2, v4);
}

int32_t main() {
  ::ttnn::Tensor v1;
  ::ttnn::Tensor v2;
  std::tie(v1, v2) = create_inputs_for_add();
  ::ttnn::Tensor v3 = point_to_point_provide_output_tensor(v1);
  ::ttnn::Tensor v3_2 = point_to_point_provide_output_tensor(v2);
  int32_t v4 = 0;
  return v4;
}
