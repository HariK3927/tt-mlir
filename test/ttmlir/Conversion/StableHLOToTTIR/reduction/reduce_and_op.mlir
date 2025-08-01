// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t
module @jit_reduce_and attributes {} {
  func.func public @test_reduce_and_4to3dim(%arg0: tensor<128x10x32x4xi1>, %cst_0: tensor<i1>) -> tensor<128x10x32xi1> {
    // CHECK-LABEL: func.func public @test_reduce_and_4to3dim
    // CHECK: ttir.empty
    // CHECK: "ttir.reduce_and"
    // CHECK-SAME: dim_arg = [3 : i32]
    // CHECK-SAME: keep_dim = false
    // CHECK-SAME: tensor<128x10x32x4xi1>
    // CHECK-SAME: -> tensor<128x10x32xi1>
    %0 = stablehlo.reduce(%arg0 init: %cst_0) applies stablehlo.and across dimensions = [3] : (tensor<128x10x32x4xi1>, tensor<i1>) -> tensor<128x10x32xi1>
    return %0 : tensor<128x10x32xi1>
  }

  func.func public @test_reduce_and_3to2dim(%arg0: tensor<128x10x4xi1>, %cst_0: tensor<i1>) -> tensor<128x4xi1> {
    // CHECK-LABEL: func.func public @test_reduce_and_3to2dim
    // CHECK: ttir.empty
    // CHECK: "ttir.reduce_and"
    // CHECK-SAME: dim_arg = [1 : i32]
    // CHECK-SAME: keep_dim = false
    // CHECK-SAME: tensor<128x10x4xi1>
    // CHECK-SAME: -> tensor<128x4xi1>
    %0 = stablehlo.reduce(%arg0 init: %cst_0) applies stablehlo.and across dimensions = [1] : (tensor<128x10x4xi1>, tensor<i1>) -> tensor<128x4xi1>
    return %0 : tensor<128x4xi1>
  }

  func.func public @test_reduce_and_2to1dim(%arg0: tensor<128x10xi1>, %cst_0: tensor<i1>) -> tensor<10xi1> {
    // CHECK-LABEL: func.func public @test_reduce_and_2to1dim
    // CHECK: ttir.empty
    // CHECK: "ttir.reduce_and"
    // CHECK-SAME: dim_arg = [0 : i32]
    // CHECK-SAME: keep_dim = false
    // CHECK-SAME: tensor<128x10xi1>
    // CHECK-SAME: -> tensor<10xi1>
    %0 = stablehlo.reduce(%arg0 init: %cst_0) applies stablehlo.and across dimensions = [0] : (tensor<128x10xi1>, tensor<i1>) -> tensor<10xi1>
    return %0 : tensor<10xi1>
  }
}
