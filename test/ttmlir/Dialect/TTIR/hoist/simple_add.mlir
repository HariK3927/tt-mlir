// RUN: ttmlir-opt --ttcore-wrap-device-module --ttir-cpu-hoist-transform --canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t

// CHECK: ttcore.device_module {
// CHECK: builtin.module {

// CHECK: func.func @add1
func.func @add1(%arg0: tensor<32x32xbf16>, %arg1: tensor<32x32xbf16>) -> tensor<32x32xbf16> {
  %0 = ttir.empty() : tensor<32x32xbf16>
  // CHECK: %{{.*}} = ttir.empty() : tensor<{{.*}}xf32>
  // CHECK: %{{.*}} = ttir.to_layout %{{.*}}, %{{.*}}
  // CHECK: %{{.*}} = ttir.empty() : tensor<{{.*}}xf32>
  // CHECK: %{{.*}} = ttir.to_layout %{{.*}}, %{{.*}}
  // CHECK: %{{.*}} = call @hoisted_ttir_add_32x32_32x32_32x32_func_decl
  %1 = "ttir.add"(%arg0, %arg1, %0) {ttir.should_hoist} : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
  // CHECK: %{{.*}} = ttir.empty() : tensor<{{.*}}xbf16>
  // CHECK: %{{.*}} = ttir.to_layout %{{.*}}, %{{.*}}
  return %1 : tensor<32x32xbf16>
}

// CHECK: func.func @add2
func.func @add2(%arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %0 = ttir.empty() : tensor<32x32xf32>
  // CHECK: %{{.*}} = call @hoisted_ttir_add_32x32_32x32_32x32_func_decl
  %1 = "ttir.add"(%arg0, %arg1, %0) {ttir.should_hoist} : (tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  return %1 : tensor<32x32xf32>
}

// CHECK: func.func @add3
func.func @add3(%arg0: tensor<32x3xf32>, %arg1: tensor<32x3xf32>) -> tensor<32x3xf32> {
  %0 = ttir.empty() : tensor<32x3xf32>
  // CHECK: %{{.*}} = call @hoisted_ttir_add_32x3_32x3_32x3_func_decl
  %1 = "ttir.add"(%arg0, %arg1, %0) {ttir.should_hoist} : (tensor<32x3xf32>, tensor<32x3xf32>, tensor<32x3xf32>) -> tensor<32x3xf32>
  return %1 : tensor<32x3xf32>
}

// CHECK: func.func @add4
func.func @add4(%arg0: tensor<32x32xbf16>, %arg1: tensor<32x32xbf16>) -> tensor<32x32xbf16> {
  %0 = ttir.empty() : tensor<32x32xbf16>
  // CHECK: %{{.*}} = ttir.empty() : tensor<{{.*}}xf32>
  // CHECK: %{{.*}} = ttir.to_layout %{{.*}}, %{{.*}}
  // CHECK: %{{.*}} = ttir.empty() : tensor<{{.*}}xf32>
  // CHECK: %{{.*}} = ttir.to_layout %{{.*}}, %{{.*}}
  // CHECK: %{{.*}} = call @hoisted_ttir_add_32x32_32x32_32x32_func_decl
  %1 = "ttir.add"(%arg0, %arg1, %0) {ttir.should_hoist} : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
  // CHECK: %{{.*}} = ttir.empty() : tensor<{{.*}}xbf16>
  // CHECK: %{{.*}} = ttir.to_layout %{{.*}}, %{{.*}}
  return %1 : tensor<32x32xbf16>
}
// CHECK: func.func private @hoisted_ttir_add_32x32_32x32_32x32_func_decl
// CHECK: func.func private @hoisted_ttir_add_32x3_32x3_32x3_func_decl

// CHECK: ttcore.cpu_module {
// CHECK: builtin.module {
// CHECK: func.func @hoisted_ttir_add_32x32_32x32_32x32_func
// CHECK: func.func @hoisted_ttir_add_32x3_32x3_32x3_func
