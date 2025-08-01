// RUN: ttmlir-opt --convert-tosa-to-ttir -o %t %s
// RUN: FileCheck %s --input-file=%t
module attributes {} {
  func.func @test_cast(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xbf16> {
    %0 = tosa.cast %arg0 : (tensor<13x21x3xf32>) -> tensor<13x21x3xbf16>
    // CHECK: [[VAL0:%[0-9]+]] = ttir.empty() : [[TENSOR_SIZE:tensor<13x21x3xbf16>]]
    // CHECK: [[VAL1:%[0-9]+]] = "ttir.typecast"(%arg{{[0-9]+}}, [[VAL0]]) <{conservative_folding = false}> : (tensor<13x21x3xf32>, [[TENSOR_SIZE]]) -> [[TENSOR_SIZE]]
    return %0 : tensor<13x21x3xbf16>
    // CHECK: return [[VAL1]] : [[TENSOR_SIZE]]
  }
}
