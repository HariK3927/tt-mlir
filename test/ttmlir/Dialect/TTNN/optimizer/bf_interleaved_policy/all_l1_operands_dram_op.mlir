// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-optimizer=true memory-layout-analysis-enabled=true memory-layout-analysis-policy=BFInterleaved" -o %t %s
// RUN: FileCheck %s --input-file=%t
module attributes {} {
  func.func @forward(%arg0: tensor<6144x1024xbf16>, %arg1: tensor<1024x6144xbf16>) -> tensor<6144x6144xbf16> {
    // CHECK: #[[L1_:.*]] = #ttnn.buffer_type<l1>
    // CHECK-DAG: #[[LAYOUT_5:.*]] = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <8x8, (d0, d1) -> (0, d0, d1)>, memref<1x96x!ttcore.tile<32x32, bf16>, #l1>, <interleaved>>
    // CHECK-DAG: #[[LAYOUT_6:.*]] = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<192x192x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
    %0 = ttir.empty() : tensor<6144x1024xbf16>
    // CHECK-DAG: %{{.*}} = "ttnn.relu"{{.*}} -> tensor<6144x1024xbf16, #[[LAYOUT_5]]>
    %1 = "ttir.relu"(%arg0, %0) : (tensor<6144x1024xbf16>, tensor<6144x1024xbf16>) -> tensor<6144x1024xbf16>
    %2 = ttir.empty() : tensor<1024x6144xbf16>
    // CHECK-DAG: %{{.*}} = "ttnn.relu"{{.*}} -> tensor<1024x6144xbf16, #[[LAYOUT_5]]>
    %3 = "ttir.relu"(%arg1, %2) : (tensor<1024x6144xbf16>, tensor<1024x6144xbf16>) -> tensor<1024x6144xbf16>
    %4 = ttir.empty() : tensor<6144x6144xbf16>
    // CHECK: %{{.*}} = "ttnn.matmul"{{.*}} -> tensor<6144x6144xbf16, #[[LAYOUT_6]]>
    %5 = "ttir.matmul"(%1, %3, %4) : (tensor<6144x1024xbf16>, tensor<1024x6144xbf16>, tensor<6144x6144xbf16>) -> tensor<6144x6144xbf16>
    return %5 : tensor<6144x6144xbf16>
  }
}
