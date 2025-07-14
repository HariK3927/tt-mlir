// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn-precompiled.hpp"
::ttnn::Tensor point_to_point_provide_output_tensor(::ttnn::Tensor v1) {
  ttnn::distributed::MeshDevice* v2 = ttnn::DeviceGetter::getInstance();
  assert(0 && "Mesh shard operation is not supported in emitc yet."); // ::ttnn::mesh_shard
  ::ttnn::Tensor v3 = ttnn::mesh_shard(v1);
  ttnn::deallocate(v1, false);
  ::ttnn::Tensor v4 = ttnn::to_layout(v3, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ttnn::deallocate(v3, false);
  ::ttnn::Tensor v5 = ttnn::to_device(v4, v2, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::empty(::ttnn::Shape({1, 1, 256, 64}), ::ttnn::DataType::FLOAT32, ::ttnn::Layout::TILE, v2, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  assert(0 && "PointToPoint  operation is not supported in emitc yet."); // ::ttnn::PointToPoint
  ::ttnn::Tensor v7 = ttnn::point_to_point(v5, ::ttnn::MeshCoordinate(tt::stl::Span<const uint32_t>{0, 0}), ::ttnn::MeshCoordinate(tt::stl::Span<const uint32_t>{0, 1}), v6);
  ttnn::deallocate(v6, false);
  assert(0 && "PointToPoint  operation is not supported in emitc yet."); // ::ttnn::PointToPoint
  ::ttnn::Tensor v8 = ttnn::point_to_point(v5, ::ttnn::MeshCoordinate(tt::stl::Span<const uint32_t>{0, 1}), ::ttnn::MeshCoordinate(tt::stl::Span<const uint32_t>{0, 0}), v7);
  ttnn::deallocate(v7, false);
  ::ttnn::Tensor v9 = ttnn::from_device(v8);
  ttnn::deallocate(v8, false);
  ::ttnn::Tensor v10 = ttnn::to_layout(v9, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ttnn::deallocate(v9, false);
  assert(0 && "Mesh shard operation is not supported in emitc yet."); // ::ttnn::mesh_shard
  ::ttnn::Tensor v11 = ttnn::mesh_shard(v10);
  ttnn::deallocate(v10, false);
  return v11;
}

::ttnn::Tensor point_to_point_case_generate_output_tensor(::ttnn::Tensor v1) {
  ttnn::distributed::MeshDevice* v2 = ttnn::DeviceGetter::getInstance();
  assert(0 && "Mesh shard operation is not supported in emitc yet."); // ::ttnn::mesh_shard
  ::ttnn::Tensor v3 = ttnn::mesh_shard(v1);
  ttnn::deallocate(v1, false);
  ::ttnn::Tensor v4 = ttnn::to_layout(v3, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ttnn::deallocate(v3, false);
  ::ttnn::Tensor v5 = ttnn::to_device(v4, v2, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  assert(0 && "PointToPoint  operation is not supported in emitc yet."); // ::ttnn::PointToPoint
  ::ttnn::Tensor v6 = ttnn::point_to_point(v5, ::ttnn::MeshCoordinate(tt::stl::Span<const uint32_t>{0, 0}), ::ttnn::MeshCoordinate(tt::stl::Span<const uint32_t>{0, 1}), ::std::nullopt);
  assert(0 && "PointToPoint  operation is not supported in emitc yet."); // ::ttnn::PointToPoint
  ::ttnn::Tensor v7 = ttnn::point_to_point(v5, ::ttnn::MeshCoordinate(tt::stl::Span<const uint32_t>{0, 1}), ::ttnn::MeshCoordinate(tt::stl::Span<const uint32_t>{0, 0}), v6);
  ttnn::deallocate(v6, false);
  ::ttnn::Tensor v8 = ttnn::from_device(v7);
  ttnn::deallocate(v7, false);
  ::ttnn::Tensor v9 = ttnn::to_layout(v8, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ttnn::deallocate(v8, false);
  assert(0 && "Mesh shard operation is not supported in emitc yet."); // ::ttnn::mesh_shard
  ::ttnn::Tensor v10 = ttnn::mesh_shard(v9);
  ttnn::deallocate(v9, false);
  return v10;
}
