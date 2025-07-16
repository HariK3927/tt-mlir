// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s

module @jit_eltwise_optimization_barrier {
  func.func public @test_optimization_barrier(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
    %0 = "ttir.optimization_barrier"(%arg0) : (tensor<64x128xf32>) -> tensor<64x128xf32>
    %1 = "ttir.optimization_barrier"(%arg1) : (tensor<64x128xf32>) -> tensor<64x128xf32>
    // CHECK: %0 = "ttnn.optimization_barrier"(%arg0) : (tensor<64x128xf32, #ttnn_layout>) -> tensor<64x128xf32, #ttnn_layout>
    // CHECK: %1 = "ttnn.optimization_barrier"(%arg1) : (tensor<64x128xf32, #ttnn_layout>) -> tensor<64x128xf32, #ttnn_layout>
    // CHECK-NOT: "ttnn.deallocate"(%arg0)
    // CHECK-NOT: "ttnn.deallocate"(%arg0)
    %2 = ttir.empty() : tensor<64x128xf32>
    %3 = "ttir.add"(%0, %1, %2) : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    // CHECK: "ttnn.deallocate"(%1) <{force = false}> : (tensor<64x128xf32, #ttnn_layout>) -> ()
    // CHECK: "ttnn.deallocate"(%0) <{force = false}> : (tensor<64x128xf32, #ttnn_layout>) -> ()
    return %3 : tensor<64x128xf32>
  }
}
