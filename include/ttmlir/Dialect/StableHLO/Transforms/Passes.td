// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_StableHLO_DIALECT_PASSES_TD
#define TTMLIR_StableHLO_DIALECT_PASSES_TD

include "mlir/Pass/PassBase.td"
include "shardy/dialect/sdy/ir/dialect.td"
include "shardy/dialect/sdy/ir/op_interface.td"

def ApplyArgumentShardStatusPass : Pass<"apply-argument-shard-status", "::mlir::ModuleOp">
{
  let summary = "Annotate arguments with their shard status.";
  let description = [{
    This pass will analyze the module arguments and annotate arguments with whether they are pre-sharded or not. It will determine this based on analyzing shardy or mhlo annotations.
    If shardy or mhlo annotations exist, the argument/results is presharded. If not, it is unsharded.

    Example, this pass will convert the following code:
    ```mlir
    module {
      sdy.mesh @mesh = <["x"=1, "batch"=8]>
      func.func @main(%arg0: tensor<1024x2x32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, \[{"batch"}, {}, {}, {}\]>}) -> (tensor<2048x1024xf32>) {
        %0 = stablehlo.reshape %arg0 : (tensor<1024x2x32x32xf32>) -> tensor<2048x1024xf32>
        return %0 : tensor<2048x1024xf32>
      }
    }
    ```

    Into:
    ```mlir
    module {
      sdy.mesh @mesh = <["x"=1, "batch"=8]>
      func.func @main(%arg0: tensor<1024x2x32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, \[{"batch"}, {}, {}, {}\]>, ttcore.shard_status = #ttcore.shard_status<presharded>}) -> (tensor<2048x1024xf32> {ttcore.shard_status = #ttcore.shard_status<unsharded>}) {
        %0 = stablehlo.reshape %arg0 : (tensor<1024x2x32x32xf32>) -> tensor<2048x1024xf32>
        return %0 : tensor<2048x1024xf32>
      }
    }
    ```
  }];

  let dependentDialects = [
    "::mlir::sdy::SdyDialect",
    "::mlir::tt::ttir::TTIRDialect"
  ];
}

def AnalyzeMeshPass : Pass<"analyze-mesh", "::mlir::ModuleOp">
{
  let summary = "Analyze the mesh compilation target.";
  let description = [{
    This pass will analyze the mesh of the graph and update shardings or annotations to match the target device. It will attempt to automatically parallelize if user provided option and mesh shape.
    For GSPMD graphs, it will analyze the mhlo annotations and attempt to deduce the mesh shape from them.
    For shardy based graphs, it will check for the existence of a shardy mesh, from which it will query the mesh target.
    If no mesh is present, it will assume a single chip graph and insert a 1x1 mesh.

    Example, this pass will convert the following code:
    ```mlir
    func.func @main(%arg0: tensor<1x128xf32>, %arg1: tensor<128xf32>) -> tensor<1x128xf32> {
        %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1] : (tensor<1x128xf32>) -> tensor<1x128xf32>
        %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
        %2 = stablehlo.add %0, %1 : tensor<1x128xf32>
        return %2 : tensor<1x128xf32>
    }
    ```

    Into:
    ```mlir
    module {
      sdy.mesh @mesh = <["x"=1, "y"=1]>
      func.func @main(%arg0: tensor<1x128xf32>, %arg1: tensor<128xf32>) -> tensor<1x128xf32> {
        %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1] : (tensor<1x128xf32>) -> tensor<1x128xf32>
        %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
        %2 = stablehlo.add %0, %1 : tensor<1x128xf32>
        return %2 : tensor<1x128xf32>
      }
    }
    ```
  }];

  let options = [
    ListOption<"meshShape", "mesh-shape", "int64_t", "Set the mesh shape">,
    Option<"automaticArgAnalysis", "automatic-arg-analysis", "bool", /*default=*/"false", "Automatically determine argument shardings">,
  ];

  let dependentDialects = [
    "::mlir::sdy::SdyDialect",
    "::mlir::tt::ttir::TTIRDialect"
  ];
}

def WrapUnderManualComputationPass : Pass<"wrap-under-manual-computation", "::mlir::ModuleOp">
{
  let summary = "Wrap all operations within a sdy manual computation op.";
  let description = [{
    This pass will wrap all the operations within a module under a manual computation op that defines per device tensor shapes.

    Example, this pass will convert the following code:
    ```mlir
    module {
      sdy.mesh @mesh = <["x"=1, "y"=1]>
      func.func @add(%arg0: tensor<32x48x24x32xf32>, %arg1: tensor<32x48x24x32xf32>) -> tensor<32x48x24x32xf32> {
        %0 = stablehlo.add %arg0, %arg1 : tensor<32x48x24x32xf32>
        return %0 : tensor<32x48x24x32xf32>
      }
    }
    ```

    Into:
    ```mlir
    module {
      sdy.mesh @mesh = <["x"=1, "y"=1]>
      func.func @add(%arg0: tensor<32x48x24x32xf32> {sdy.sharding = #sdy.sharding<@mesh, \[{?}, {?}, {?}, {?}\]>}, %arg1: tensor<32x48x24x32xf32> {sdy.sharding = #sdy.sharding<@mesh, \[{?}, {?}, {?}, {?}\]>}) -> tensor<32x48x24x32xf32> {
        %0 = sdy.manual_computation(%arg0, %arg1) in_shardings=\[<@mesh, \[{?}, {?}, {?}, {?}\]>, <@mesh, \[{?}, {?}, {?}, {?}\]>\] out_shardings=\[<@mesh, \[{?}, {?}, {?}, {?}\]>\] manual_axes={} (%arg2: tensor<32x48x24x32xf32>, %arg3: tensor<32x48x24x32xf32>) {
          %1 = stablehlo.add %arg2, %arg3 : tensor<32x48x24x32xf32>
          sdy.return %1 : tensor<32x48x24x32xf32>
        } : (tensor<32x48x24x32xf32>, tensor<32x48x24x32xf32>) -> tensor<32x48x24x32xf32>
        return %0 : tensor<32x48x24x32xf32>
      }
    }
    ```
  }];

  let dependentDialects = [
    "::mlir::sdy::SdyDialect"
  ];
}

def UpdateGlobalToLocalShapesPass : Pass<"update-global-to-local-shapes", "::mlir::ModuleOp">
{
  let summary = "Update all the tensor dimensions based on their tensor sharding annotation from global shapes to local per device shapes.";
  let description = [{
    This pass will update all tensor dimensions from global to local shapes.

    Example, this pass will convert the following code:
    ```mlir
    module {
      sdy.mesh @mesh = <["model"=1, "batch"=2]>
      func.func @full_arg_annotation(%arg0: tensor<64x128xf32> {sdy.sharding = #sdy.sharding<@mesh, \[{"batch"}, {"model"}\]>, ttcore.shard_status = #ttcore.shard_status<presharded>}) -> (tensor<64x128xf32> {sdy.sharding = #sdy.sharding<@mesh, \[{"batch", ?}, {"model", ?}\]>, ttcore.shard_status = #ttcore.shard_status<unsharded>}) {
        %0 = sdy.manual_computation(%arg0) in_shardings=\[<@mesh, \[{"batch"}, {"model"}\]>\] out_shardings=\[<@mesh, \[{"batch", ?}, {"model", ?}\]>\] manual_axes={} (%arg1: tensor<64x128xf32>) {
          %1 = stablehlo.cbrt %arg1 {sdy.sharding = #sdy.sharding_per_value<\[<@mesh, \[{"batch", ?}, {"model", ?}\]>\]>} : tensor<64x128xf32>
          sdy.return %1 : tensor<64x128xf32>
        } : (tensor<64x128xf32>) -> tensor<64x128xf32>
        return %0 : tensor<64x128xf32>
      }
    }
    ```

    Into:
    ```mlir
    module {
      sdy.mesh @mesh = <["model"=1, "batch"=2]>
      func.func @full_arg_annotation(%arg0: tensor<64x128xf32> {ttcore.shard_status = #ttcore.shard_status<presharded>}) -> (tensor<64x128xf32> {ttcore.shard_status = #ttcore.shard_status<unsharded>}) {
        %0 = sdy.manual_computation(%arg0) in_shardings=\[<@mesh, \[{"batch"}, {"model"}\]>\] out_shardings=\[<@mesh, \[{"batch", ?}, {"model", ?}\]>\] manual_axes={"model", "batch"} (%arg1: tensor<32x128xf32>) {
          %1 = stablehlo.cbrt %arg1 : tensor<32x128xf32>
          sdy.return %1 : tensor<32x128xf32>
        } : (tensor<64x128xf32>) -> tensor<64x128xf32>
        return %0 : tensor<64x128xf32>
      }
    }
    ```
  }];

  let dependentDialects = [
    "::mlir::sdy::SdyDialect"
  ];
}

#endif
