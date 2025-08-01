// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t
module @jit_eltwise_floor attributes {} {
  func.func public @test_floor(%arg0: tensor<32x32x3xf32>) -> tensor<32x32x3xf32> {
    %0 = stablehlo.floor %arg0 : tensor<32x32x3xf32>
    // CHECK: = ttir.empty
    // CHECK: = "ttir.floor"
    return %0 : tensor<32x32x3xf32>
  }
}
