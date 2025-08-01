// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t
module @jit_eltwise_logical attributes {} {
  func.func public @logical_not(%arg0: tensor<32x32xi1>) -> tensor<32x32xi1> {
    // CHECK: %[[E:.*]] = ttir.empty() : [[TENSOR:tensor<32x32xi1>]]
    // CHECK: = "ttir.logical_not"(%arg0, %[[E]])
    // CHECK-SAME: ([[TENSOR]], [[TENSOR]]) -> [[TENSOR]]
    %0 = stablehlo.not  %arg0 : tensor<32x32xi1>
    // CHECK: return %1 : [[TENSOR]]
    return %0 : tensor<32x32xi1>
  }

  func.func public @logical_not_scalar(%arg0: tensor<i1>) -> tensor<i1> {
    // CHECK: %[[E:.*]] = ttir.empty() : [[TENSOR:tensor<i1>]]
    // CHECK: = "ttir.logical_not"(%arg0, %[[E]])
    // CHECK-SAME: ([[TENSOR]], [[TENSOR]]) -> [[TENSOR]]
    %0 = stablehlo.not  %arg0 : tensor<i1>
    // CHECK: return %1 : [[TENSOR]]
    return %0 : tensor<i1>
  }
}
