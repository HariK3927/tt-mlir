// REQUIRES: stablehlo
// RUN: rm -rf %t.ttnn
// RUN: rm -rf %t.mlir
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir
// RUN: FileCheck --input-file=%t.mlir %s
module attributes {} {
  func.func @forward(%arg0: tensor<1x1x32x128xbf16>) -> tensor<1x1x32x128xbf16> {
    // CHECK: ttnn.arange
    %0 = "stablehlo.iota"() {iota_dimension = 2: i64} : () -> tensor<1x1x32x128xbf16>
    %2 = "stablehlo.multiply"(%arg0, %0) : (tensor<1x1x32x128xbf16>, tensor<1x1x32x128xbf16>) -> tensor<1x1x32x128xbf16>
    return %2 : tensor<1x1x32x128xbf16>
  }
}
