// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-optimizer=true memory-layout-analysis-enabled=true memory-layout-analysis-policy=GreedyL1Interleaved" -o %t %s
// RUN: FileCheck %s --input-file=%t
//
//       A     B
//        \   /
//          C
//          |
//          D
//
//  (A > L1) AND (B > L1) AND (C > L1)
//      =>
//  DRAM: ABC; L1: None
//
module attributes {} {
  func.func @forward(%arg0: tensor<8192x8192xbf16>, %arg1: tensor<8192x8192xbf16>, %arg2: tensor<8192x8192xbf16>, %arg3: tensor<8192x8192xbf16>) -> tensor<8192x8192xbf16> {
    // CHECK-DAG: #[[LAYOUT_2:.*]] = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<{{.*}}>, #dram>, <interleaved>>
    %0 = ttir.empty() : tensor<8192x8192xbf16>
    // CHECK-DAG: %{{.*}} = "ttnn.add"{{.*}} -> tensor<8192x8192xbf16, #[[LAYOUT_2]]>
    %1 = "ttir.add"(%arg0, %arg1, %0) : (tensor<8192x8192xbf16>, tensor<8192x8192xbf16>, tensor<8192x8192xbf16>) -> tensor<8192x8192xbf16>
    %2 = ttir.empty() : tensor<8192x8192xbf16>
    // CHECK-DAG: %{{.*}} = "ttnn.add"{{.*}} -> tensor<8192x8192xbf16, #[[LAYOUT_2]]>
    %3 = "ttir.add"(%arg2, %arg3, %2) : (tensor<8192x8192xbf16>, tensor<8192x8192xbf16>, tensor<8192x8192xbf16>) -> tensor<8192x8192xbf16>
    %4 = ttir.empty() : tensor<8192x8192xbf16>
    // CHECK-DAG: %{{.*}} = "ttnn.matmul"{{.*}} -> tensor<8192x8192xbf16, #[[LAYOUT_2]]>
    %5 = "ttir.matmul"(%1, %3, %4) : (tensor<8192x8192xbf16>, tensor<8192x8192xbf16>, tensor<8192x8192xbf16>) -> tensor<8192x8192xbf16>
    return %5 : tensor<8192x8192xbf16>
  }
}
