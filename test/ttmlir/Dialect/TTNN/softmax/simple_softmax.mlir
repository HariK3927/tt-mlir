// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t
module attributes {} {
  func.func @forward(%arg0: tensor<512x1024xbf16>) -> tensor<512x1024xbf16> {
    %0 = ttir.empty() : tensor<512x1024xbf16>
    // CHECK: = "ttnn.softmax"
    // Check for positive dimension attribute
    %1 = "ttir.softmax"(%arg0, %0) <{dimension = 1 : si32}> : (tensor<512x1024xbf16>, tensor<512x1024xbf16>) -> tensor<512x1024xbf16>
    %2 = ttir.empty() : tensor<512x1024xbf16>
    // CHECK: = "ttnn.softmax"
    // Check for negative dimension attribute
    %3 = "ttir.softmax"(%1, %2) <{dimension = -1 : si32}> : (tensor<512x1024xbf16>, tensor<512x1024xbf16>) -> tensor<512x1024xbf16>
    return %3 : tensor<512x1024xbf16>
  }
}
