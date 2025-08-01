// REQUIRES: stablehlo
// RUN: rm -rf %t.ttnn
// RUN: rm -rf %t.mlir
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir
// RUN: FileCheck --input-file=%t.mlir %s

module @jit_reduce_add attributes {} {
  func.func public @test_reduce_and_4to3dim(%arg0: tensor<128x10x32x4xi1>, %cst_0: tensor<i1>) -> tensor<128x10x32xi1> {
    // CHECK-LABEL: func.func public @test_reduce_and_4to3dim
    // CHECK: "ttnn.prod"
    // CHECK-SAME: dim_arg = 3
    // CHECK-SAME: keep_dim = false
    // CHECK-SAME: -> tensor<128x10x32xbf16,
    %0 = stablehlo.reduce(%arg0 init: %cst_0) applies stablehlo.and across dimensions = [3] : (tensor<128x10x32x4xi1>, tensor<i1>) -> tensor<128x10x32xi1>
    return %0 : tensor<128x10x32xi1>
  }
}
