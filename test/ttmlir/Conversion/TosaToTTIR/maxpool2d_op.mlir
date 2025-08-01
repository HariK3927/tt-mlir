// RUN: ttmlir-opt --convert-tosa-to-ttir -o %t %s
// RUN: FileCheck %s --input-file=%t
module attributes {} {
  func.func @test_maxpool(%arg0: tensor<32x800x600x6xf32>) -> tensor<32x400x300x6xf32> {
    // CHECK: func.func {{.+}} [[IN_SIZE:tensor<[0-9]+x[0-9]+x[0-9]+x[0-9]+xf32>]]{{.*}} ->
    %1 = tosa.max_pool2d %arg0 {kernel = array<i64: 2, 2>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>} : (tensor<32x800x600x6xf32>) -> tensor<32x400x300x6xf32>
    // CHECK: %[[OP_OUT:[0-9]+]] = ttir.empty() : [[OUT_SIZE:tensor<[0-9]+x[0-9]+x[0-9]+x[0-9]+xf32>]]
    // CHECK: %[[VAL:[0-9]+]] = "ttir.max_pool2d"(%arg{{[0-9]+}}, %[[OP_OUT]]){{.+}} ([[IN_SIZE]], [[OUT_SIZE]]) -> [[OUT_SIZE]]
    // CHECK: return %[[VAL]] : [[OUT_SIZE]]
    return %1 : tensor<32x400x300x6xf32>
  }

  func.func @test_maxpool_with_different_padding(%arg0: tensor<32x800x600x6xf32>) -> tensor<32x398x302x6xf32> {
    // CHECK: func.func {{.+}} [[IN_SIZE:tensor<[0-9]+x[0-9]+x[0-9]+x[0-9]+xf32>]]{{.*}} ->
    %1 = tosa.max_pool2d %arg0 {kernel = array<i64: 8, 8>, pad = array<i64: 0, 2, 4, 6>, stride = array<i64: 2, 2>} : (tensor<32x800x600x6xf32>) -> tensor<32x398x302x6xf32>
    // CHECK: %[[OP_OUT:[0-9]+]] = ttir.empty() : [[OUT_SIZE:tensor<[0-9]+x[0-9]+x[0-9]+x[0-9]+xf32>]]
    // CHECK: %[[VAL:[0-9]+]] = "ttir.max_pool2d"(%arg{{[0-9]+}}, %[[OP_OUT]])
    // CHECK-SAME: padding = array<i32: 0, 4, 2, 6>
    // CHECK-SAME: ([[IN_SIZE]], [[OUT_SIZE]]) -> [[OUT_SIZE]]
    // CHECK: return %[[VAL]] : [[OUT_SIZE]]
    return %1 : tensor<32x398x302x6xf32>
  }
}
