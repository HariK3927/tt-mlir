// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

module attributes {} {
  func.func @forward(%arg0: tensor<3x8x8x256xbf16>, %arg1: tensor<256x256x3x3xbf16>, %arg2: tensor<1x1x1x256xbf16>) -> tensor<3x10x10x256xbf16> {
    %0 = ttir.empty() : tensor<3x10x10x256xbf16>
    // CHECK: = "ttnn.conv_transpose2d"
    %1 = "ttir.conv_transpose2d"(%arg0, %arg1, %arg2, %0)
            <{
              stride = 1: i32,
              padding = 0: i32,
              output_padding = 0: i32,
              dilation = 1: i32,
              groups = 1: i32}
            > : (tensor<3x8x8x256xbf16>, tensor<256x256x3x3xbf16>, tensor<1x1x1x256xbf16>, tensor<3x10x10x256xbf16>) -> tensor<3x10x10x256xbf16>
    return %1 : tensor<3x10x10x256xbf16>
  }
}
