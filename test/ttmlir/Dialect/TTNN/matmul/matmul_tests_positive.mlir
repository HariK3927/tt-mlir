// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t
module {
  func.func @matmul_1d_1d(%arg0: tensor<128xbf16>, %arg1: tensor<128xbf16>) -> tensor<1xbf16> {
    %0 = ttir.empty() : tensor<1xbf16>
    // CHECK: "ttnn.matmul"
    %1 = "ttir.matmul"(%arg0, %arg1, %0) : (tensor<128xbf16>, tensor<128xbf16>, tensor<1xbf16>) -> tensor<1xbf16>
    return %1 : tensor<1xbf16>
  }

  func.func @matmul_1d_2d(%arg0: tensor<128xbf16>, %arg1: tensor<128x64xbf16>) -> tensor<64xbf16> {
    %0 = ttir.empty() : tensor<64xbf16>
    // CHECK: "ttnn.matmul"
    %1 = "ttir.matmul"(%arg0, %arg1, %0) : (tensor<128xbf16>, tensor<128x64xbf16>, tensor<64xbf16>) -> tensor<64xbf16>
    return %1 : tensor<64xbf16>
  }

  func.func @matmul_2d_1d(%arg0: tensor<64x128xbf16>, %arg1: tensor<128xbf16>) -> tensor<64xbf16> {
    %0 = ttir.empty() : tensor<64xbf16>
    // CHECK: "ttnn.matmul"
    %1 = "ttir.matmul"(%arg0, %arg1, %0) : (tensor<64x128xbf16>, tensor<128xbf16>, tensor<64xbf16>) -> tensor<64xbf16>
    return %1 : tensor<64xbf16>
  }

  func.func @matmul_2d_2d(%arg0: tensor<64x128xbf16>, %arg1: tensor<128x64xbf16>) -> tensor<64x64xbf16> {
    %0 = ttir.empty() : tensor<64x64xbf16>
    // CHECK: "ttnn.matmul"
    %1 = "ttir.matmul"(%arg0, %arg1, %0) : (tensor<64x128xbf16>, tensor<128x64xbf16>, tensor<64x64xbf16>) -> tensor<64x64xbf16>
    return %1 : tensor<64x64xbf16>
  }

  func.func @matmul_1d_nd(%arg0: tensor<128xbf16>, %arg1: tensor<12x7x128x64xbf16>) -> tensor<12x7x64xbf16> {
    %0 = ttir.empty() : tensor<12x7x64xbf16>
    // CHECK: "ttnn.matmul"
    %1 = "ttir.matmul"(%arg0, %arg1, %0) : (tensor<128xbf16>, tensor<12x7x128x64xbf16>, tensor<12x7x64xbf16>) -> tensor<12x7x64xbf16>
    return %1 : tensor<12x7x64xbf16>
  }

  func.func @matmul_nd_1d(%arg0: tensor<12x7x128x64xbf16>, %arg1: tensor<64xbf16>) -> tensor<12x7x128xbf16> {
    %0 = ttir.empty() : tensor<12x7x128xbf16>
    // CHECK: "ttnn.matmul"
    %1 = "ttir.matmul"(%arg0, %arg1, %0) : (tensor<12x7x128x64xbf16>, tensor<64xbf16>, tensor<12x7x128xbf16>) -> tensor<12x7x128xbf16>
    return %1 : tensor<12x7x128xbf16>
  }

  func.func @matmul_2d_nd(%arg0: tensor<64x128xbf16>, %arg1: tensor<12x7x128x64xbf16>) -> tensor<12x7x64x64xbf16> {
    %0 = ttir.empty() : tensor<12x7x64x64xbf16>
    // CHECK: "ttnn.matmul"
    %1 = "ttir.matmul"(%arg0, %arg1, %0) : (tensor<64x128xbf16>, tensor<12x7x128x64xbf16>, tensor<12x7x64x64xbf16>) -> tensor<12x7x64x64xbf16>
    return %1 : tensor<12x7x64x64xbf16>
  }

  func.func @matmul_nd_2d(%arg0: tensor<12x7x128x64xbf16>, %arg1: tensor<64x128xbf16>) -> tensor<12x7x128x128xbf16> {
    %0 = ttir.empty() : tensor<12x7x128x128xbf16>
    // CHECK: "ttnn.matmul"
    %1 = "ttir.matmul"(%arg0, %arg1, %0) : (tensor<12x7x128x64xbf16>, tensor<64x128xbf16>, tensor<12x7x128x128xbf16>) -> tensor<12x7x128x128xbf16>
    return %1 : tensor<12x7x128x128xbf16>
  }

  // Matmul nd - nd tests.
  func.func @matmul_nd_nd_same_rank_same_dims(%arg0: tensor<7x64x128xbf16>, %arg1: tensor<7x128x64xbf16>) -> tensor<7x64x64xbf16> {
    %0 = ttir.empty() : tensor<7x64x64xbf16>
    // CHECK: "ttnn.matmul"
    %1 = "ttir.matmul"(%arg0, %arg1, %0) : (tensor<7x64x128xbf16>, tensor<7x128x64xbf16>, tensor<7x64x64xbf16>) -> tensor<7x64x64xbf16>
    return %1 : tensor<7x64x64xbf16>
  }

  func.func @matmul_nd_nd_same_rank_broadcastable_dims_1(%arg0: tensor<7x64x128xbf16>, %arg1: tensor<1x128x64xbf16>) -> tensor<7x64x64xbf16> {
    %0 = ttir.empty() : tensor<7x64x64xbf16>
    // CHECK: "ttnn.matmul"
    %1 = "ttir.matmul"(%arg0, %arg1, %0) : (tensor<7x64x128xbf16>, tensor<1x128x64xbf16>, tensor<7x64x64xbf16>) -> tensor<7x64x64xbf16>
    return %1 : tensor<7x64x64xbf16>
  }

  func.func @matmul_nd_nd_same_rank_broadcastable_dims_2(%arg0: tensor<1x7x64x128xbf16>, %arg1: tensor<7x1x128x64xbf16>) -> tensor<7x7x64x64xbf16> {
    %0 = ttir.empty() : tensor<7x7x64x64xbf16>
    // CHECK: "ttnn.matmul"
    %1 = "ttir.matmul"(%arg0, %arg1, %0) : (tensor<1x7x64x128xbf16>, tensor<7x1x128x64xbf16>, tensor<7x7x64x64xbf16>) -> tensor<7x7x64x64xbf16>
    return %1 : tensor<7x7x64x64xbf16>
  }

  func.func @matmul_nd_nd_different_rank_broadcastable_dims_2(%arg0: tensor<12x1x7x64x128xbf16>, %arg1: tensor<7x1x128x64xbf16>) -> tensor<12x7x7x64x64xbf16> {
    %0 = ttir.empty() : tensor<12x7x7x64x64xbf16>
    // CHECK: "ttnn.matmul"
    %1 = "ttir.matmul"(%arg0, %arg1, %0) : (tensor<12x1x7x64x128xbf16>, tensor<7x1x128x64xbf16>, tensor<12x7x7x64x64xbf16>) -> tensor<12x7x7x64x64xbf16>
    return %1 : tensor<12x7x7x64x64xbf16>
  }

  // Matmul with transposed inputs tests.
  func.func @matmul_2d_tranpose_2d(%arg0: tensor<64x128xbf16>, %arg1: tensor<64x128xbf16>) -> tensor<128x128xbf16> {
    %0 = ttir.empty() : tensor<128x128xbf16>
    // CHECK: "ttnn.matmul"
    %1 = "ttir.matmul"(%arg0, %arg1, %0) <{transpose_a = true}> : (tensor<64x128xbf16>, tensor<64x128xbf16>, tensor<128x128xbf16>) -> tensor<128x128xbf16>
    return %1 : tensor<128x128xbf16>
  }

  func.func @matmul_2d_2d_transpose(%arg0: tensor<64x128xbf16>, %arg1: tensor<64x128xbf16>) -> tensor<64x64xbf16> {
    %0 = ttir.empty() : tensor<64x64xbf16>
    // CHECK: "ttnn.matmul"
    %1 = "ttir.matmul"(%arg0, %arg1, %0) <{transpose_b = true}> : (tensor<64x128xbf16>, tensor<64x128xbf16>, tensor<64x64xbf16>) -> tensor<64x64xbf16>
    return %1 : tensor<64x64xbf16>
  }

  func.func @matmul_2d_tranpose_2d_transpose(%arg0: tensor<64x128xbf16>, %arg1: tensor<128x64xbf16>) -> tensor<128x128xbf16> {
    %0 = ttir.empty() : tensor<128x128xbf16>
    // CHECK: "ttnn.matmul"
    %1 = "ttir.matmul"(%arg0, %arg1, %0) <{transpose_a = true, transpose_b = true}> : (tensor<64x128xbf16>, tensor<128x64xbf16>, tensor<128x128xbf16>) -> tensor<128x128xbf16>
    return %1 : tensor<128x128xbf16>
  }
}
