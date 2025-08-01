// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-const-eval=false enable-trace=true" -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  // CHECK-LABEL: func.func @trace_0_matmul_with_bias
  // CHECK: "ttnn.matmul"
  // CHECK: "ttnn.add"(%0, %arg2)

  // CHECK-LABEL: func.func @run_and_capture_trace_0_matmul_with_bias
  // CHECK: "ttnn.write_tensor"
  // CHECK: "ttnn.begin_trace_capture"
  // CHECK: "ttnn.end_trace_capture"

  // CHECK-LABEL: func.func @execute_trace_0_matmul_with_bias
  // CHECK: "ttnn.execute_trace"

  // CHECK-LABEL: func.func @matmul_with_bias(
  func.func @matmul_with_bias(%arg0: tensor<64x32xbf16>, %arg1: tensor<32x64xbf16>, %arg2: tensor<64x64xbf16>) -> tensor<64x64xbf16> {
    // CHECK: %[[GET_DEVICE:.+]] = "ttnn.get_device"()
    // CHECK-NEXT: %[[TRACE_RESULT:.+]] = "ttnn.capture_or_execute_trace"(%[[GET_DEVICE]], %arg0, %arg1, %arg2) <{capture_callee = @run_and_capture_trace_0_matmul_with_bias, execute_callee = @execute_trace_0_matmul_with_bias}>
    // CHECK-NOT: "ttnn.add"
    // CHECK-NOT: "ttnn.matmul"
    // CHECK: return %[[TRACE_RESULT]]
    %0 = ttir.empty() : tensor<64x64xbf16>
    %1 = "ttir.matmul"(%arg0, %arg1, %0) : (tensor<64x32xbf16>, tensor<32x64xbf16>, tensor<64x64xbf16>) -> tensor<64x64xbf16>
    %2 = ttir.empty() : tensor<64x64xbf16>
    %3 = "ttir.add"(%1, %arg2, %2) : (tensor<64x64xbf16>, tensor<64x64xbf16>, tensor<64x64xbf16>) -> tensor<64x64xbf16>
    return %3 : tensor<64x64xbf16>
  }
}
