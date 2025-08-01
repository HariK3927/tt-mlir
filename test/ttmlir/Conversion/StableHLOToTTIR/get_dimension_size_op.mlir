// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t
module @jit_get_dimension_size attributes {} {
  func.func public @test_get_dimension_size(%arg0: tensor<13x21x3xf32>) -> tensor<i32> {
    %0 = stablehlo.get_dimension_size %arg0, dim = 1 : (tensor<13x21x3xf32>) -> tensor<i32>
    // CHECK: [[VAL:%[0-9]+]] = "ttir.get_dimension_size"(%arg0) <{dimension = 1 : i32}> : (tensor<13x21x3xf32>) -> tensor<i32>
    return %0 : tensor<i32>
    // CHECK: return [[VAL]] : tensor<i32>
  }

  func.func public @test_get_dimension_size_f64(%arg0: tensor<64x64xf64>) -> tensor<i32> {
    // CHECK: [[VAL:%[0-9]+]] = "ttir.get_dimension_size"(%arg0) <{dimension = 1 : i32}> : (tensor<64x64xf64>) -> tensor<i32>
    %0 = stablehlo.get_dimension_size %arg0, dim = 1 : (tensor<64x64xf64>) -> tensor<i32>
    // CHECK: return [[VAL]] : tensor<i32>
    return %0 : tensor<i32>
  }
}
