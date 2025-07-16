// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %basename_t.ttnn
// RUN: ttmlir-opt --ttnn-tuplify-tensors --convert-ttnn-to-emitc %t.mlir > %t2.mlir
// RUN: ttmlir-translate --mlir-to-cpp %t2.mlir > %basename_t.cpp

module {
  func.func public @optimization_barrier(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
    %0 = "ttir.optimization_barrier"(%arg0) : (tensor<64x128xf32>) -> tensor<64x128xf32>
    %1 = "ttir.optimization_barrier"(%arg1) : (tensor<64x128xf32>) -> tensor<64x128xf32>
    %2 = ttir.empty() : tensor<64x128xf32>
    %3 = "ttir.add"(%0, %1, %2) : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    return %3 : tensor<64x128xf32>
  }
}
