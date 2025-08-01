// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t
module @jit_eltwise_isfinite attributes {} {
  func.func public @test_isfinite(%arg0: tensor<32x32x3xf32>) -> tensor<32x32x3xi1> {
    // CHECK: %[[E:.*]] = ttir.empty() : tensor<32x32x3xi1>
    // CHECK: %[[C:.*]] = "ttir.isfinite"(%arg0, %[[E]])
    // CHECK-SAME: (tensor<32x32x3xf32>, tensor<32x32x3xi1>) -> tensor<32x32x3xi1>
    %0 = stablehlo.is_finite %arg0 : (tensor<32x32x3xf32>) -> tensor<32x32x3xi1>
    // CHECK: return %[[C]] : tensor<32x32x3xi1>
    return %0 : tensor<32x32x3xi1>
  }
}
