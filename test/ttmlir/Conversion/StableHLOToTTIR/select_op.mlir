// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t
module @jit_eltwise_select attributes {} {
  func.func public @test_select(%arg0: tensor<13x37xf32>, %arg1: tensor<13x37xf32>) -> tensor<13x37xf32> {
    %0 = stablehlo.compare EQ, %arg0, %arg1 : (tensor<13x37xf32>, tensor<13x37xf32>) -> tensor<13x37xi1>
    %1 = stablehlo.select %0, %arg0, %arg1 : (tensor<13x37xi1>, tensor<13x37xf32>, tensor<13x37xf32>) -> tensor<13x37xf32>
    // CHECK: %[[EMPTY:[0-9]+]] = ttir.empty()
    // CHECK: %[[VAL1:[0-9]+]] = "ttir.eq"
    // CHECK: %[[SELECT:[0-9]+]] = "ttir.where"(%[[VAL1:[0-9]+]], %arg0, %arg1, %[[EMPTY:[0-9]+]]) : (tensor<13x37xi1>, tensor<13x37xf32>, tensor<13x37xf32>, tensor<13x37xf32>) -> tensor<13x37xf32>
    return %1 : tensor<13x37xf32>
  }
}
