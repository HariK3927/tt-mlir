// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t
module attributes {} {
  func.func @forward(%arg0: tensor<512x32xbf16>) -> tensor<512xbf16> {
    %0 = ttir.empty() : tensor<512xbf16>
    // CHECK: = "ttnn.max"
    %1 = "ttir.max"(%arg0, %0) <{dim_arg = [1: i32], keep_dim = false}> : (tensor<512x32xbf16>, tensor<512xbf16>) -> tensor<512xbf16>
    return %1 : tensor<512xbf16>
  }

  func.func @test_reduce_max_multi_dim(%arg0: tensor<128x32x64x32xbf16>) -> tensor<128x1x1x1xbf16> {
    // CHECK-LABEL: @test_reduce_max_multi_dim(
    %0 = ttir.empty() : tensor<128x1x1x1xbf16>
    // CHECK: "ttnn.max"(%arg0)
    // CHECK-SAME: dim_arg = [1 : i32, 2 : i32, 3 : i32]
    // CHECK-SAME: keep_dim = true
    // CHECK-SAME: tensor<128x32x64x32xbf16
    // CHECK-SAME: -> tensor<128x1x1x1xbf16
    %1 = "ttir.max"(%arg0, %0) <{dim_arg = [1: i32, 2: i32, 3: i32], keep_dim = true}> : (tensor<128x32x64x32xbf16>, tensor<128x1x1x1xbf16>) -> tensor<128x1x1x1xbf16>
    return %1 : tensor<128x1x1x1xbf16>
  }

  func.func @test_reduce_max_requires_padding_workaround(%arg0: tensor<128x32x60x90xbf16>) -> tensor<128x1x1x90xbf16> {
    // CHECK-LABEL: @test_reduce_max_requires_padding_workaround(
    %0 = ttir.empty() : tensor<128x1x1x90xbf16>
    // CHECK: "ttnn.pad"
    // CHECK-SAME: 0, 0, 0, 0, 0, 4, 0, 0
    // CHECK: "ttnn.max"
    // CHECK-SAME: dim_arg = [1 : i32, 2 : i32]
    // CHECK-SAME: keep_dim = true
    // CHECK-SAME: tensor<128x32x64x90xbf16
    // CHECK-SAME: -> tensor<128x1x1x90xbf16
    %1 = "ttir.max"(%arg0, %0) <{dim_arg = [1: i32, 2: i32], keep_dim = true}> : (tensor<128x32x60x90xbf16>, tensor<128x1x1x90xbf16>) -> tensor<128x1x1x90xbf16>
    return %1 : tensor<128x1x1x90xbf16>
  }
}
