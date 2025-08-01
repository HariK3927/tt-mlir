// RUN: ttmlir-opt --ttir-to-ttir-decomposition -o %t %s
// RUN: FileCheck %s --input-file=%t

module attributes {} {
  func.func @select_identity(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %0 = ttir.empty() : tensor<4x4xf32>
    // CHECK: %{{[0-9]+}} = "ttir.slice"
    %1 = "ttir.index_select"(%arg0, %0) <{dim = 1: si32, begin = 0: si32, length = 4: si32, stride = 4: si32}>  :
        (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    return %1 : tensor<4x4xf32>
  }

  func.func @select_multi_slice(%arg0: tensor<4x2x64x128xf32>) -> tensor<4x2x64x32xf32> {
    %0 = ttir.empty() : tensor<4x2x64x32xf32>

    // CHECK: %{{[0-9]+}} = "ttir.slice"
    // CHECK: %{{[0-9]+}} = "ttir.slice"
    // CHECK: %{{[0-9]+}} = "ttir.slice"
    // CHECK: %{{[0-9]+}} = "ttir.slice"
    // CHECK: %{{[0-9]+}} = "ttir.concat"
    %1 = "ttir.index_select"(%arg0, %0) <{dim = -1: si32, begin = 0: si32, length = 4: si32, stride = 16: si32}>  :
        (tensor<4x2x64x128xf32>, tensor<4x2x64x32xf32>) -> tensor<4x2x64x32xf32>

    return %1 : tensor<4x2x64x32xf32>
  }
}
