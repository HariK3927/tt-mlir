// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

module attributes {} {
  func.func @bitwise_and(%arg0: tensor<64x128xi32>, %arg1: tensor<64x128xi32>) -> tensor<64x128xi32> {
    %0 = ttir.empty() : tensor<64x128xi32>
    %1 = "ttir.bitwise_and"(%arg0, %arg1, %0) : (tensor<64x128xi32>, tensor<64x128xi32>, tensor<64x128xi32>) -> tensor<64x128xi32>
    // CHECK: "ttnn.bitwise_and"
    // CHECK-SAME: tensor<64x128xsi32
    // CHECK-SAME: tensor<64x128xsi32
    // CHECK-SAME: -> tensor<64x128xsi32
    return %1 : tensor<64x128xi32>
  }

  func.func @bitwise_or(%arg0: tensor<64x128xi32>, %arg1: tensor<64x128xi32>) -> tensor<64x128xi32> {
    %0 = ttir.empty() : tensor<64x128xi32>
    %1 = "ttir.bitwise_or"(%arg0, %arg1, %0) : (tensor<64x128xi32>, tensor<64x128xi32>, tensor<64x128xi32>) -> tensor<64x128xi32>
    // CHECK: "ttnn.bitwise_or"
    // CHECK-SAME: tensor<64x128xsi32
    // CHECK-SAME: tensor<64x128xsi32
    // CHECK-SAME: -> tensor<64x128xsi32
    return %1 : tensor<64x128xi32>
  }

  func.func @bitwise_xor(%arg0: tensor<64x128xi32>, %arg1: tensor<64x128xi32>) -> tensor<64x128xi32> {
    %0 = ttir.empty() : tensor<64x128xi32>
    %1 = "ttir.bitwise_xor"(%arg0, %arg1, %0) : (tensor<64x128xi32>, tensor<64x128xi32>, tensor<64x128xi32>) -> tensor<64x128xi32>
    // CHECK: "ttnn.bitwise_xor"
    // CHECK-SAME: tensor<64x128xsi32
    // CHECK-SAME: tensor<64x128xsi32
    // CHECK-SAME: -> tensor<64x128xsi32
    return %1 : tensor<64x128xi32>
  }
}
