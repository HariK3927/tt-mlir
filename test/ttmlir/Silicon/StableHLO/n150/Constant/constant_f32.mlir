// REQUIRES: stablehlo
// RUN: rm -rf %t.ttnn
// RUN: rm -rf %t.mlir
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% enable-const-eval=false" -o %t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir
// RUN: FileCheck --input-file=%t.mlir %s

module @jit_constant attributes {} {
  func.func public @test_float_scalar() -> tensor<f32> {
    // CHECK-LABEL: func.func public @test_float_scalar
    // CHECK: ttnn.full
    // CHECK-SAME: fill_value = 3.000000e+00 : f32
    // CHECK-SAME: -> tensor<f32
    %0 = stablehlo.constant dense<3.0> : tensor<f32>
    return %0 : tensor<f32>
  }

  func.func public @test_float_scalar_empty() -> tensor<f32> {
    // CHECK-LABEL: func.func public @test_float_scalar_empty
    // CHECK: ttnn.full
    // CHECK-SAME: fill_value = 0.000000e+00 : f32
    // CHECK-SAME: -> tensor<f32
    %0 = stablehlo.constant dense<0.0> : tensor<f32>
    return %0 : tensor<f32>
  }

  func.func public @test_float_empty() -> tensor<64x128xf32> {
    // CHECK-LABEL: func.func public @test_float_empty
    // CHECK: ttnn.full
    // CHECK-SAME: fill_value = 0.000000e+00 : f32
    // CHECK-SAME: -> tensor<64x128xf32
    %0 = stablehlo.constant dense<0.0> : tensor<64x128xf32>
    return %0 : tensor<64x128xf32>
  }

  func.func public @test_float_splat() -> tensor<64x128xf32> {
    // CHECK-LABEL: func.func public @test_float_splat
    // CHECK: ttnn.full
    // CHECK-SAME: fill_value = 3.000000e+00 : f32
    // CHECK-SAME: -> tensor<64x128xf32
    %0 = stablehlo.constant dense<3.0> : tensor<64x128xf32>
    return %0 : tensor<64x128xf32>
  }
}
