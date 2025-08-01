// RUN: ttmlir-opt --tt-populate-argument-types="argument-types=forward=input,parameter,parameter,constant" -o %t %s
// RUN: FileCheck %s --input-file=%t
module attributes {} {
  // CHECK: ttcore.argument_type = #ttcore.argument_type<input>
  // CHECK: ttcore.argument_type = #ttcore.argument_type<parameter>
  // CHECK: ttcore.argument_type = #ttcore.argument_type<parameter>
  // CHECK: ttcore.argument_type = #ttcore.argument_type<constant>
  func.func @forward(%arg0: tensor<1x32x32x64xbf16>, %arg1: tensor<64x64x3x3xbf16>, %arg2: tensor<1x1x1x64xbf16>, %arg3: tensor<1x32x32x64xbf16>) -> (tensor<1x32x32x64xbf16>, tensor<64x64x3x3xbf16>, tensor<1x1x1x64xbf16>, tensor<1x32x32x64xbf16>) {
    return %arg0, %arg1, %arg2, %arg3 : tensor<1x32x32x64xbf16>, tensor<64x64x3x3xbf16>, tensor<1x1x1x64xbf16>, tensor<1x32x32x64xbf16>
  }
}
