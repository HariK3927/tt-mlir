// REQUIRES: stablehlo
// RUN: rm -rf %t.ttnn
// RUN: rm -rf %t.mlir
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir
// RUN: FileCheck --input-file=%t.mlir %s

module @jit_eltwise_compare attributes {} {
  func.func public @logical_not(%arg0: tensor<64x128xi1>) -> tensor<64x128xi1> {
    // CHECK-LABEL: func.func public @logical_not
    // CHECK: ttnn.logical_not
    // CHECK-SAME: tensor<64x128xbf16,
    // CHECK-SAME: -> tensor<64x128xbf16,
    %0 = stablehlo.not  %arg0 : tensor<64x128xi1>
    return %0 : tensor<64x128xi1>
  }
}
