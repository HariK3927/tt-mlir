// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t
module @jit_reduce_minimum attributes {} {
  func.func public @test_reduce_minimum_4to3dim(%arg0: tensor<128x10x32x4xf32>, %cst_0: tensor<f32>) -> tensor<128x32x4xf32> {
    // CHECK: ttir.empty
    // CHECK: "ttir.min"
    // CHECK-SAME: dim_arg = [1 : i32]
    // CHECK-SAME: keep_dim = false
    // CHECK-SAME: tensor<128x10x32x4xf32>
    // CHECK-SAME: -> tensor<128x32x4xf32>
    %0 = stablehlo.reduce(%arg0 init: %cst_0) applies stablehlo.minimum across dimensions = [1] : (tensor<128x10x32x4xf32>, tensor<f32>) -> tensor<128x32x4xf32>
    return %0 : tensor<128x32x4xf32>
  }

  func.func public @test_reduce_minimum_4to2dim(%arg0: tensor<128x10x32x4xf32>, %cst_0: tensor<f32>) -> tensor<128x32xf32> {
    // CHECK: ttir.empty
    // CHECK: "ttir.min"
    // CHECK-SAME: dim_arg = [1 : i32, 3 : i32]
    // CHECK-SAME: keep_dim = false
    // CHECK-SAME: tensor<128x10x32x4xf32>
    // CHECK-SAME: -> tensor<128x32xf32>
    %0 = stablehlo.reduce(%arg0 init: %cst_0) applies stablehlo.minimum across dimensions = [1, 3] : (tensor<128x10x32x4xf32>, tensor<f32>) -> tensor<128x32xf32>
    return %0 : tensor<128x32xf32>
  }

  func.func public @test_reduce_minimum_4to1dim(%arg0: tensor<128x10x32x4xf32>, %cst_0: tensor<f32>) -> tensor<128xf32> {
    // CHECK: ttir.empty
    // CHECK: "ttir.min"
    // CHECK-SAME: dim_arg = [1 : i32, 2 : i32, 3 : i32]
    // CHECK-SAME: keep_dim = false
    // CHECK-SAME: tensor<128x10x32x4xf32>
    // CHECK-SAME: -> tensor<128xf32>
    %0 = stablehlo.reduce(%arg0 init: %cst_0) applies stablehlo.minimum across dimensions = [1, 2, 3] : (tensor<128x10x32x4xf32>, tensor<f32>) -> tensor<128xf32>
    return %0 : tensor<128xf32>
  }

  func.func public @test_reduce_minimum_4to0dim(%arg0: tensor<128x10x32x4xf32>, %cst_0: tensor<f32>) -> tensor<f32> {
    // CHECK: ttir.empty
    // CHECK: "ttir.min"
    // CHECK-SAME: dim_arg = [0 : i32, 1 : i32, 2 : i32, 3 : i32]
    // CHECK-SAME: keep_dim = false
    // CHECK-SAME: tensor<128x10x32x4xf32>
    // CHECK-SAME: -> tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst_0) applies stablehlo.minimum across dimensions = [0, 1, 2, 3] : (tensor<128x10x32x4xf32>, tensor<f32>) -> tensor<f32>
    return %0 : tensor<f32>
  }

  func.func public @test_reduce_minimum_3to2dim(%arg0: tensor<128x10x4xf32>, %cst_0: tensor<f32>) -> tensor<128x4xf32> {
    // CHECK: ttir.empty
    // CHECK: "ttir.min"
    // CHECK-SAME: dim_arg = [1 : i32]
    // CHECK-SAME: keep_dim = false
    // CHECK-SAME: tensor<128x10x4xf32>
    // CHECK-SAME: -> tensor<128x4xf32>
    %0 = stablehlo.reduce(%arg0 init: %cst_0) applies stablehlo.minimum across dimensions = [1] : (tensor<128x10x4xf32>, tensor<f32>) -> tensor<128x4xf32>
    return %0 : tensor<128x4xf32>
  }

  func.func public @test_reduce_minimum_3to1dim(%arg0: tensor<128x10x4xf32>, %cst_0: tensor<f32>) -> tensor<128xf32> {
    // CHECK: ttir.empty
    // CHECK: "ttir.min"
    // CHECK-SAME: dim_arg = [1 : i32, 2 : i32]
    // CHECK-SAME: keep_dim = false
    // CHECK-SAME: tensor<128x10x4xf32>
    // CHECK-SAME: -> tensor<128xf32>
    %0 = stablehlo.reduce(%arg0 init: %cst_0) applies stablehlo.minimum across dimensions = [1, 2] : (tensor<128x10x4xf32>, tensor<f32>) -> tensor<128xf32>
    return %0 : tensor<128xf32>
  }

  func.func public @test_reduce_minimum_3to0dim(%arg0: tensor<128x10x4xf32>, %cst_0: tensor<f32>) -> tensor<f32> {
    // CHECK: ttir.empty
    // CHECK: "ttir.min"
    // CHECK-SAME: dim_arg = [0 : i32, 1 : i32, 2 : i32]
    // CHECK-SAME: keep_dim = false
    // CHECK-SAME: tensor<128x10x4xf32>
    // CHECK-SAME: -> tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst_0) applies stablehlo.minimum across dimensions = [0, 1, 2] : (tensor<128x10x4xf32>, tensor<f32>) -> tensor<f32>
    return %0 : tensor<f32>
  }

  func.func public @test_reduce_minimum_2to1dim(%arg0: tensor<128x10xf32>, %cst_0: tensor<f32>) -> tensor<128xf32> {
    // CHECK: ttir.empty
    // CHECK: "ttir.min"
    // CHECK-SAME: dim_arg = [1 : i32]
    // CHECK-SAME: keep_dim = false
    // CHECK-SAME: tensor<128x10xf32>
    // CHECK-SAME: -> tensor<128xf32>
    %0 = stablehlo.reduce(%arg0 init: %cst_0) applies stablehlo.minimum across dimensions = [1] : (tensor<128x10xf32>, tensor<f32>) -> tensor<128xf32>
    return %0 : tensor<128xf32>
  }

  func.func public @test_reduce_minimum_2to0dim(%arg0: tensor<128x10xf32>, %cst_0: tensor<f32>) -> tensor<f32> {
    // CHECK: ttir.empty
    // CHECK: "ttir.min"
    // CHECK-SAME: dim_arg = [0 : i32, 1 : i32]
    // CHECK-SAME: keep_dim = false
    // CHECK-SAME: tensor<128x10xf32>
    // CHECK-SAME: -> tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst_0) applies stablehlo.minimum across dimensions = [0, 1] : (tensor<128x10xf32>, tensor<f32>) -> tensor<f32>
    return %0 : tensor<f32>
  }

  func.func public @test_reduce_minimum_1to0dim(%arg0: tensor<128xf32>, %cst_0: tensor<f32>) -> tensor<f32> {
    // CHECK: ttir.empty
    // CHECK: "ttir.min"
    // CHECK-SAME: dim_arg = [0 : i32]
    // CHECK-SAME: keep_dim = false
    // CHECK-SAME: tensor<128xf32>
    // CHECK-SAME: -> tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst_0) applies stablehlo.minimum across dimensions = [0] : (tensor<128xf32>, tensor<f32>) -> tensor<f32>
    return %0 : tensor<f32>
  }
}
