// REQUIRES: stablehlo
// RUN: rm -rf %t.ttnn
// RUN: rm -rf %t.mlir
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% enable-const-eval=false" -o %t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir
// RUN: FileCheck --input-file=%t.mlir %s

module @jit_get_dimension_size attributes {} {
  func.func public @test_get_dimension_size(%arg0: tensor<64x128xf32>) -> tensor<i32> {
    // CHECK-LABEL: func.func public @test_get_dimension_size
    // CHECK: ttnn.full
    // CHECK-SAME: fill_value = 128 : i32
    // CHECK-SAME: -> tensor<f32
    %0 = stablehlo.get_dimension_size %arg0, dim = 1 : (tensor<64x128xf32>) -> tensor<i32>
    return %0 : tensor<i32>
  }

  func.func public @test_get_dimension_size_f64(%arg0: tensor<64x128xf64>) -> tensor<i32> {
    // CHECK-LABEL: func.func public @test_get_dimension_size_f64
    // CHECK: ttnn.full
    // CHECK-SAME: fill_value = 128 : i32
    // CHECK-SAME: -> tensor<f32
    %0 = stablehlo.get_dimension_size %arg0, dim = 1 : (tensor<64x128xf64>) -> tensor<i32>
    return %0 : tensor<i32>
  }
}
