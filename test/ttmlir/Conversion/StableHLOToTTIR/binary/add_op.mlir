// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t
module @jit_eltwise_add attributes {} {
  func.func public @test_add(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<13x21x3xf32>
    // CHECK: = ttir.empty
    // CHECK: = "ttir.add"
    return %0 : tensor<13x21x3xf32>
  }
}
