// RUN: ttmlir-opt --ttir-to-ttir-decomposition -o %t %s
// RUN: FileCheck %s --input-file=%t
module @jit_loss {
  func.func public @main(%arg0: tensor<2x1xf32>, %arg1: tensor<1xf32>, %arg2: tensor<127x2xf32>, %arg3: tensor<127x1xf32>) -> (tensor<2x1xf32>, tensor<f32>) {
    %0 = "ttir.dot_general"(%arg2, %arg0) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<127x2xf32>, tensor<2x1xf32>) -> tensor<127x1xf32>
    %1 = ttir.empty() : tensor<1xf32>
    %2 = "ttir.typecast"(%arg1, %1) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
    %3 = ttir.empty() : tensor<1x1xf32>
    %4 = "ttir.reshape"(%2, %3) <{shape = [1 : i32, 1 : i32]}> : (tensor<1xf32>, tensor<1x1xf32>) -> tensor<1x1xf32>
    %5 = ttir.empty() : tensor<127x1xf32>
    %6 = "ttir.broadcast"(%4, %5) <{broadcast_dimensions = array<i64: 127, 1>}> : (tensor<1x1xf32>, tensor<127x1xf32>) -> tensor<127x1xf32>
    %7 = ttir.empty() : tensor<127x1xf32>
    %8 = "ttir.add"(%0, %6, %7) : (tensor<127x1xf32>, tensor<127x1xf32>, tensor<127x1xf32>) -> tensor<127x1xf32>
    %9 = ttir.empty() : tensor<127x1xf32>
    %10 = "ttir.subtract"(%8, %arg3, %9) : (tensor<127x1xf32>, tensor<127x1xf32>, tensor<127x1xf32>) -> tensor<127x1xf32>
    %11 = call @integer_pow(%10) : (tensor<127x1xf32>) -> tensor<127x1xf32>
    %12 = "ttir.constant"() <{value = dense<2.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
    %13 = ttir.empty() : tensor<1x1xf32>
    %14 = "ttir.reshape"(%12, %13) <{shape = [1 : i32, 1 : i32]}> : (tensor<1xf32>, tensor<1x1xf32>) -> tensor<1x1xf32>
    %15 = ttir.empty() : tensor<127x1xf32>
    %16 = "ttir.broadcast"(%14, %15) <{broadcast_dimensions = array<i64: 127, 1>}> : (tensor<1x1xf32>, tensor<127x1xf32>) -> tensor<127x1xf32>
    %17 = ttir.empty() : tensor<127x1xf32>
    %18 = "ttir.multiply"(%16, %11, %17) : (tensor<127x1xf32>, tensor<127x1xf32>, tensor<127x1xf32>) -> tensor<127x1xf32>
    %19 = "ttir.constant"() <{value = dense<1.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
    %20 = "ttir.constant"() <{value = dense<1.270000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
    %21 = ttir.empty() : tensor<1xf32>
    %22 = "ttir.div"(%19, %20, %21) : (tensor<1xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
    %23 = ttir.empty() : tensor<1x1xf32>
    %24 = "ttir.reshape"(%22, %23) <{shape = [1 : i32, 1 : i32]}> : (tensor<1xf32>, tensor<1x1xf32>) -> tensor<1x1xf32>
    %25 = ttir.empty() : tensor<127x1xf32>
    %26 = "ttir.broadcast"(%24, %25) <{broadcast_dimensions = array<i64: 127, 1>}> : (tensor<1x1xf32>, tensor<127x1xf32>) -> tensor<127x1xf32>
    %27 = ttir.empty() : tensor<127x1xf32>
    %28 = "ttir.multiply"(%26, %18, %27) : (tensor<127x1xf32>, tensor<127x1xf32>, tensor<127x1xf32>) -> tensor<127x1xf32>
    %29 = ttir.empty() : tensor<f32>
    %30 = "ttir.sum"(%28, %29) <{dim_arg = [0 : i32, 1 : i32], keep_dim = false}> : (tensor<127x1xf32>, tensor<f32>) -> tensor<f32>
    %31 = ttir.empty() : tensor<f32>
    %32 = "ttir.typecast"(%30, %31) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %33 = "ttir.dot_general"(%28, %arg2) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 0>, contract_dims_rhs = array<i64: 0>}> : (tensor<127x1xf32>, tensor<127x2xf32>) -> tensor<1x2xf32>
    // CHECK: "ttir.permute"
    // CHECK: {permutation = array<i64: 1, 0>}> : (tensor<127x1xf32>, tensor<1x127xf32>) -> tensor<1x127xf32>
    // CHECK: "ttir.matmul"
    // CHECK: (tensor<1x127xf32>, tensor<127x2xf32>, tensor<1x2xf32>) -> tensor<1x2xf32>
    %34 = ttir.empty() : tensor<2x1xf32>
    %35 = "ttir.permute"(%33, %34) <{permutation = array<i64: 1, 0>}> : (tensor<1x2xf32>, tensor<2x1xf32>) -> tensor<2x1xf32>
    return %35, %32 : tensor<2x1xf32>, tensor<f32>
  }
  func.func private @integer_pow(%arg0: tensor<127x1xf32>) -> tensor<127x1xf32> {
    return %arg0 : tensor<127x1xf32>
  }
}
