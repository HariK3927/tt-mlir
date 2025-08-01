// RUN: ttmlir-opt --convert-tosa-to-ttir -o %t %s
// RUN: FileCheck %s --input-file=%t
module attributes {} {
  func.func @test_matmul(%arg0: tensor<13x21x16xf32>, %arg1: tensor<13x16x31xf32>) -> tensor<13x21x31xf32> {
    // CHECK: func.func {{.+}}%arg{{[0-9]+}}: tensor<[[B:[0-9]+]]x[[I:[0-9]+]]x[[J:[0-9]+]]xf32>, %arg{{[0-9]+}}: tensor<[[B:[0-9]+]]x[[J:[0-9]+]]x[[K:[0-9]+]]xf32>
    %0 = "tosa.const"() { values = dense<0.0> : tensor<1xf32> } : () -> tensor<1xf32>
    %1 = tosa.matmul %arg0, %arg1, %0, %0 : (tensor<13x21x16xf32>, tensor<13x16x31xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<13x21x31xf32>
    // CHECK: %[[CST:[0-9]+]] = "ttir.constant"() <{value = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
    // CHECK: %[[OP_OUT:[0-9]+]] = ttir.empty() : tensor<[[B]]x[[I]]x[[K]]xf32>
    // CHECK: %[[VAL:[0-9]+]] = "ttir.matmul"(%arg{{[0-9]+}}, %arg{{[0-9]+}}, %[[OP_OUT]]){{.+}} (tensor<[[B]]x[[I]]x[[J]]xf32>, tensor<[[B]]x[[J]]x[[K]]xf32>, tensor<[[B]]x[[I]]x[[K]]xf32>) -> tensor<[[B]]x[[I]]x[[K]]xf32>
    // CHECK: return %[[VAL]] : tensor<[[B]]x[[I]]x[[K]]xf32>
    return %1 : tensor<13x21x31xf32>
  }
}
