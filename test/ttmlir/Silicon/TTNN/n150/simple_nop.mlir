// RUN: ttmlir-opt --ttcore-register-device="system-desc-path=%system_desc_path%" --ttir-to-ttnn-backend-pipeline -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir
module @jit_convert_element_type attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<2x2xf32> {mhlo.layout_mode = "default"}) -> (tensor<2x2xf32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    // CHECK: return %arg0 : tensor<2x2xf32, #ttnn_layout>
    return %arg0 : tensor<2x2xf32>
  }
}
