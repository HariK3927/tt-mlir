// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %basename_t.ttnn %t.mlir
// RUN: ttmlir-opt --ttnn-tuplify-tensors --convert-ttnn-to-emitc -o %t2.mlir %t.mlir
// RUN: ttmlir-translate --mlir-to-cpp -o %basename_t.cpp %t2.mlir

func.func @bitwise_not(%arg0: tensor<64x128xi32>) -> tensor<64x128xi32> {
    %0 = ttir.empty() : tensor<64x128xi32>
    %1 = "ttir.bitwise_not"(%arg0, %0) : (tensor<64x128xi32>, tensor<64x128xi32>) -> tensor<64x128xi32>
    return %1 : tensor<64x128xi32>
}
