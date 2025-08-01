// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% enable-optimizer=true memory-layout-analysis-enabled=true max-legal-layouts=32 insert-memreconfig=max_pool2d_17.dc.max_pool2d.2=0 override-output-layout=max_pool2d_17.dc.max_pool2d.2=dram:interleaved:row_major:1x1:bf16" -o resnet50_first_module_ttnn.mlir %s --mlir-print-debuginfo
// RUN: FileCheck %s --input-file=resnet50_first_module_ttnn.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn resnet50_first_module_ttnn.mlir
#loc = loc("ResNetForImageClassification":0:0)
module @ResNetForImageClassification attributes {} {
  func.func @forward(%arg0: tensor<8x3x224x224xbf16> {ttir.name = "pixel_values"} loc("ResNetForImageClassification":0:0), %arg1: tensor<1x1x1x64xbf16> {ttir.name = "input_1_add_2"} loc("ResNetForImageClassification":0:0), %arg2: tensor<1x1x1x64xbf16> {ttir.name = "input_1_add_2_fork_clone1229"} loc("ResNetForImageClassification":0:0), %arg3: tensor<1x1x1x64xbf16> {ttir.name = "input_1_add_19"} loc("ResNetForImageClassification":0:0), %arg4: tensor<1x1x1x64xbf16> {ttir.name = "input_1_add_19_fork_clone1271"} loc("ResNetForImageClassification":0:0), %arg5: tensor<1x1x1x64xbf16> {ttir.name = "input_1_add_35"} loc("ResNetForImageClassification":0:0), %arg6: tensor<1x1x1x64xbf16> {ttir.name = "input_1_add_35_fork_clone1204"} loc("ResNetForImageClassification":0:0), %arg7: tensor<1x1x1x256xbf16> {ttir.name = "input_1_add_51"} loc("ResNetForImageClassification":0:0), %arg8: tensor<1x1x1x256xbf16> {ttir.name = "input_1_add_51_fork_clone1108"} loc("ResNetForImageClassification":0:0), %arg9: tensor<1x1x1x256xbf16> {ttir.name = "input_1_add_66"} loc("ResNetForImageClassification":0:0), %arg10: tensor<1x1x1x256xbf16> {ttir.name = "input_1_add_66_fork_clone1112"} loc("ResNetForImageClassification":0:0), %arg107: tensor<64x3x7x7xbf16> {ttir.name = "resnet.embedder.embedder.convolution.weight"} loc("ResNetForImageClassification":0:0), %arg108: tensor<64x64x1x1xbf16> {ttir.name = "resnet.encoder.stages.0.layers.0.layer.0.convolution.weight"} loc("ResNetForImageClassification":0:0), %arg109: tensor<64x64x3x3xbf16> {ttir.name = "resnet.encoder.stages.0.layers.0.layer.1.convolution.weight"} loc("ResNetForImageClassification":0:0), %arg110: tensor<256x64x1x1xbf16> {ttir.name = "resnet.encoder.stages.0.layers.0.layer.2.convolution.weight"} loc("ResNetForImageClassification":0:0), %arg111: tensor<256x64x1x1xbf16> {ttir.name = "resnet.encoder.stages.0.layers.0.shortcut.convolution.weight"} loc("ResNetForImageClassification":0:0)) -> (tensor<8x56x56x256xbf16> {ttir.name = "ResNetForImageClassification.output_add_868"}) {
    // CHECK-DAG: #[[SHARDED_LAYOUT:.*]] = #ttnn.ttnn_layout<{{.*}}_sharded
    // CHECK: %{{.*}} = "ttnn.{{.*}}#[[SHARDED_LAYOUT]]>
    %0 = ttir.empty() : tensor<8x224x3x224xbf16> loc(#loc249)
    %1 = "ttir.transpose"(%arg0, %0) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<8x3x224x224xbf16>, tensor<8x224x3x224xbf16>) -> tensor<8x224x3x224xbf16> loc(#loc249)
    %2 = ttir.empty() : tensor<8x224x224x3xbf16> loc(#loc250)
    %3 = "ttir.transpose"(%1, %2) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<8x224x3x224xbf16>, tensor<8x224x224x3xbf16>) -> tensor<8x224x224x3xbf16> loc(#loc250)
    %4 = ttir.empty() : tensor<8x112x112x64xbf16> loc(#loc251)
    %5 = "ttir.conv2d"(%3, %arg107, %4) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 3, 3, 3, 3>, stride = array<i32: 2, 2>}> {channel_last = 1 : si32} : (tensor<8x224x224x3xbf16>, tensor<64x3x7x7xbf16>, tensor<8x112x112x64xbf16>) -> tensor<8x112x112x64xbf16> loc(#loc251)
    %6 = ttir.empty() : tensor<8x112x112x64xbf16> loc(#loc252)
    %7 = "ttir.multiply"(%5, %arg1, %6) : (tensor<8x112x112x64xbf16>, tensor<1x1x1x64xbf16>, tensor<8x112x112x64xbf16>) -> tensor<8x112x112x64xbf16> loc(#loc252)
    %8 = ttir.empty() : tensor<8x112x112x64xbf16> loc(#loc253)
    %9 = "ttir.add"(%7, %arg2, %8) : (tensor<8x112x112x64xbf16>, tensor<1x1x1x64xbf16>, tensor<8x112x112x64xbf16>) -> tensor<8x112x112x64xbf16> loc(#loc253)
    %10 = ttir.empty() : tensor<8x112x112x64xbf16> loc(#loc254)
    %11 = "ttir.relu"(%9, %10) : (tensor<8x112x112x64xbf16>, tensor<8x112x112x64xbf16>) -> tensor<8x112x112x64xbf16> loc(#loc254)
    %12 = ttir.empty() : tensor<8x56x56x64xbf16> loc(#loc255)
    %13 = "ttir.max_pool2d"(%11, %12) <{ceil_mode = false, dilation = array<i32: 1, 1>, kernel = array<i32: 3, 3>, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 2, 2>}> {channel_last = true} : (tensor<8x112x112x64xbf16>, tensor<8x56x56x64xbf16>) -> tensor<8x56x56x64xbf16> loc(#loc255)
    %14 = ttir.empty() : tensor<8x56x56x64xbf16> loc(#loc256)
    %15 = "ttir.conv2d"(%13, %arg108, %14) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<8x56x56x64xbf16>, tensor<64x64x1x1xbf16>, tensor<8x56x56x64xbf16>) -> tensor<8x56x56x64xbf16> loc(#loc256)
    %16 = ttir.empty() : tensor<8x56x56x64xbf16> loc(#loc257)
    %17 = "ttir.multiply"(%15, %arg3, %16) : (tensor<8x56x56x64xbf16>, tensor<1x1x1x64xbf16>, tensor<8x56x56x64xbf16>) -> tensor<8x56x56x64xbf16> loc(#loc257)
    %18 = ttir.empty() : tensor<8x56x56x64xbf16> loc(#loc258)
    %19 = "ttir.add"(%17, %arg4, %18) : (tensor<8x56x56x64xbf16>, tensor<1x1x1x64xbf16>, tensor<8x56x56x64xbf16>) -> tensor<8x56x56x64xbf16> loc(#loc258)
    %20 = ttir.empty() : tensor<8x56x56x64xbf16> loc(#loc259)
    %21 = "ttir.relu"(%19, %20) : (tensor<8x56x56x64xbf16>, tensor<8x56x56x64xbf16>) -> tensor<8x56x56x64xbf16> loc(#loc259)
    %22 = ttir.empty() : tensor<8x56x56x64xbf16> loc(#loc260)
    %23 = "ttir.conv2d"(%21, %arg109, %22) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<8x56x56x64xbf16>, tensor<64x64x3x3xbf16>, tensor<8x56x56x64xbf16>) -> tensor<8x56x56x64xbf16> loc(#loc260)
    %24 = ttir.empty() : tensor<8x56x56x64xbf16> loc(#loc261)
    %25 = "ttir.multiply"(%23, %arg5, %24) : (tensor<8x56x56x64xbf16>, tensor<1x1x1x64xbf16>, tensor<8x56x56x64xbf16>) -> tensor<8x56x56x64xbf16> loc(#loc261)
    %26 = ttir.empty() : tensor<8x56x56x64xbf16> loc(#loc262)
    %27 = "ttir.add"(%25, %arg6, %26) : (tensor<8x56x56x64xbf16>, tensor<1x1x1x64xbf16>, tensor<8x56x56x64xbf16>) -> tensor<8x56x56x64xbf16> loc(#loc262)
    %28 = ttir.empty() : tensor<8x56x56x64xbf16> loc(#loc263)
    %29 = "ttir.relu"(%27, %28) : (tensor<8x56x56x64xbf16>, tensor<8x56x56x64xbf16>) -> tensor<8x56x56x64xbf16> loc(#loc263)
    %30 = ttir.empty() : tensor<8x56x56x256xbf16> loc(#loc264)
    %31 = "ttir.conv2d"(%29, %arg110, %30) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<8x56x56x64xbf16>, tensor<256x64x1x1xbf16>, tensor<8x56x56x256xbf16>) -> tensor<8x56x56x256xbf16> loc(#loc264)
    %32 = ttir.empty() : tensor<8x56x56x256xbf16> loc(#loc265)
    %33 = "ttir.multiply"(%31, %arg7, %32) : (tensor<8x56x56x256xbf16>, tensor<1x1x1x256xbf16>, tensor<8x56x56x256xbf16>) -> tensor<8x56x56x256xbf16> loc(#loc265)
    %34 = ttir.empty() : tensor<8x56x56x256xbf16> loc(#loc266)
    %35 = "ttir.add"(%33, %arg8, %34) : (tensor<8x56x56x256xbf16>, tensor<1x1x1x256xbf16>, tensor<8x56x56x256xbf16>) -> tensor<8x56x56x256xbf16> loc(#loc266)
    %36 = ttir.empty() : tensor<8x56x56x256xbf16> loc(#loc267)
    %37 = "ttir.conv2d"(%13, %arg111, %36) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<8x56x56x64xbf16>, tensor<256x64x1x1xbf16>, tensor<8x56x56x256xbf16>) -> tensor<8x56x56x256xbf16> loc(#loc267)
    %38 = ttir.empty() : tensor<8x56x56x256xbf16> loc(#loc268)
    %39 = "ttir.multiply"(%37, %arg9, %38) : (tensor<8x56x56x256xbf16>, tensor<1x1x1x256xbf16>, tensor<8x56x56x256xbf16>) -> tensor<8x56x56x256xbf16> loc(#loc268)
    %40 = ttir.empty() : tensor<8x56x56x256xbf16> loc(#loc269)
    %41 = "ttir.add"(%39, %arg10, %40) : (tensor<8x56x56x256xbf16>, tensor<1x1x1x256xbf16>, tensor<8x56x56x256xbf16>) -> tensor<8x56x56x256xbf16> loc(#loc269)
    %42 = ttir.empty() : tensor<8x56x56x256xbf16> loc(#loc270)
    %43 = "ttir.add"(%35, %41, %42) : (tensor<8x56x56x256xbf16>, tensor<8x56x56x256xbf16>, tensor<8x56x56x256xbf16>) -> tensor<8x56x56x256xbf16> loc(#loc270)
    return %43 : tensor<8x56x56x256xbf16> loc(#loc248)
  } loc(#loc)
} loc(#loc)
#loc1 = loc("forward":4294967295:3880)
#loc2 = loc("forward":4294967295:3881)
#loc3 = loc("forward":4294967295:3883)
#loc4 = loc("forward":4294967295:3885)
#loc5 = loc("forward":4294967295:3887)
#loc6 = loc("forward":4294967295:3888)
#loc7 = loc("forward":4294967295:3889)
#loc8 = loc("forward":4294967295:3891)
#loc9 = loc("forward":4294967295:3893)
#loc10 = loc("forward":4294967295:3895)
#loc11 = loc("forward":4294967295:3896)
#loc12 = loc("forward":4294967295:3898)
#loc13 = loc("forward":4294967295:3900)
#loc14 = loc("forward":4294967295:3902)
#loc15 = loc("forward":4294967295:3903)
#loc16 = loc("forward":4294967295:3905)
#loc17 = loc("forward":4294967295:3907)
#loc18 = loc("forward":4294967295:3909)
#loc19 = loc("forward":4294967295:3911)
#loc20 = loc("forward":4294967295:3913)
#loc21 = loc("forward":4294967295:3915)
#loc22 = loc("forward":4294967295:3916)
#loc248 = loc(unknown)
#loc249 = loc("conv2d_1.dc.transpose.0"(#loc1))
#loc250 = loc("conv2d_1.dc.transpose.1"(#loc2))
#loc251 = loc("conv2d_1.dc.conv2d.2"(#loc3))
#loc252 = loc("multiply_9"(#loc4))
#loc253 = loc("add_15"(#loc5))
#loc254 = loc("relu_16"(#loc6))
#loc255 = loc("max_pool2d_17.dc.max_pool2d.2"(#loc7))
#loc256 = loc("conv2d_18.dc.conv2d.2"(#loc8))
#loc257 = loc("multiply_26"(#loc9))
#loc258 = loc("add_32"(#loc10))
#loc259 = loc("relu_33"(#loc11))
#loc260 = loc("conv2d_34.dc.conv2d.2"(#loc12))
#loc261 = loc("multiply_42"(#loc13))
#loc262 = loc("add_48"(#loc14))
#loc263 = loc("relu_49"(#loc15))
#loc264 = loc("conv2d_50.dc.conv2d.2"(#loc16))
#loc265 = loc("multiply_58"(#loc17))
#loc266 = loc("add_64"(#loc18))
#loc267 = loc("conv2d_65.dc.conv2d.2"(#loc19))
#loc268 = loc("multiply_73"(#loc20))
#loc269 = loc("add_79"(#loc21))
#loc270 = loc("add_80"(#loc22))
