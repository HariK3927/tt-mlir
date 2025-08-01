// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t
module attributes {} {
  func.func @forward(%arg0: tensor<4x2x32x32xbf16>) -> tensor<2x4x32x32xbf16> {
    // CHECK-NOT: "ttnn.empty"
    %0 = ttir.empty() : tensor<2x4x32x32xbf16>
    // CHECK: = "ttnn.reshape"
    %1 = "ttir.reshape"(%arg0, %0) <{shape = [2: i32, 4: i32, 32: i32, 32: i32]}> : (tensor<4x2x32x32xbf16>, tensor<2x4x32x32xbf16>) -> tensor<2x4x32x32xbf16>
    return %1 : tensor<2x4x32x32xbf16>
  }
}
