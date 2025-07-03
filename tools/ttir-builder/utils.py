# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import inspect
import torch
import pytest
from typing import Callable, List, Optional, Tuple, Union, Literal, Dict

from ttmlir import optimizer_overrides
from ttmlir.dialects import func
from ttmlir.ir import *
from ttmlir.passmanager import PassManager
from ttmlir.passes import (
    tt_populate_argument_types,
    ttir_to_ttnn_backend_pipeline,
    ttnn_to_flatbuffer_file,
    ttir_to_ttmetal_backend_pipeline,
    ttmetal_to_flatbuffer_file,
    MLIRModuleLogger,
)

from .builder import Shape, TTIRBuilder, TypeInfo

TT_MLIR_HOME = os.environ.get("TT_MLIR_HOME", "")

# Default output to the current directory from where this module is being invoked
OUTPUT_PATH = ""

OVERRIDE_PARAMETER_DISABLED_STR = "None"

OPTIMIZER_DISABLED_POLICY = "Optimizer Disabled"

OPTIMIZATION_POLICIES = {
    OPTIMIZER_DISABLED_POLICY: False,
    "DF Sharding": optimizer_overrides.MemoryLayoutAnalysisPolicyType.DFSharding,
    "Greedy L1 Interleaved": optimizer_overrides.MemoryLayoutAnalysisPolicyType.GreedyL1Interleaved,
    "BF Interleaved": optimizer_overrides.MemoryLayoutAnalysisPolicyType.BFInterleaved,
}

# Convenience class for adding pytest marks
class Marks:
    def __init__(self, *marks):
        self.marks = marks

    def __ror__(self, lhs):
        return pytest.param(lhs, marks=self.marks)


#  ----- General Purpose Helpers - Could Be Used In Other Files -----


def shape_str(shape):
    return "x".join(map(str, shape))


def set_output_path(path):
    global OUTPUT_PATH
    if not os.path.exists(path):
        raise ValueError(f"The provided path '{path}' is not a valid path.")
    OUTPUT_PATH = path


def get_target_path(output_path, filename, target):
    target_dir = os.path.join(output_path, target)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    return os.path.join(target_dir, filename)


def create_custom_pipeline_fn(
    pipeline: str, verify: bool = True, print_ir: Union[bool, str] = False
) -> Callable:
    def wrapper(module, device_register_options):
        register_device = "ttcore-register-device"
        if device_register_options:
            register_device = f"{register_device}{{{device_register_options}}}"

        pipeline_str = f"builtin.module({','.join([register_device, pipeline])})"
        with module.context:
            pm = PassManager.parse(pipeline_str)
            pm.enable_verifier(verify)
            print("Running custom pipeline:", pm)
            if print_ir:
                print_ir_path = print_ir if isinstance(print_ir, str) else None
                pm.enable_ir_printing(tree_printing_dir_path=print_ir_path)
            pm.run(module.operation)

    return wrapper


def settings_to_overrides(settings, system_desc_path):
    override_handler = optimizer_overrides.OptimizerOverridesHandler()
    override_handler.set_system_desc_path(system_desc_path)
    print(override_handler.to_string())
    # Parse optimization policy from settings.
    optimization_policy = settings.get("optimizationPolicy")
    if optimization_policy not in OPTIMIZATION_POLICIES:
        raise ValueError(f"Invalid optimization policy selected: {optimization_policy}")
    print("222")
    if optimization_policy == OPTIMIZER_DISABLED_POLICY:
        override_handler.set_enable_optimizer(False)
    else:
        override_handler.set_enable_optimizer(True)
        print(override_handler.to_string())
        override_handler.set_enable_memory_layout_analysis(False)
        print(override_handler.to_string())
        override_handler.set_memory_layout_analysis_policy(
            OPTIMIZATION_POLICIES[optimization_policy]
        )
    print(override_handler.to_string())
    # Convert settings to output layout overrides.
    if settings.get("overrides"):
        print(override_handler.to_string())
        for op_id, overrides in settings["overrides"].items():
            op_name_loc = overrides["named_location"]
            output_layout_override = optimizer_overrides.OutputLayoutOverrideParams()
            conv2d_config_override = optimizer_overrides.Conv2dConfigOverrideParams()
            for attr in overrides["attributes"]:
                match attr["key"]:
                    case "data_type":
                        output_layout_override.set_data_type_from_str(attr["value"])
                    case "memory_layout":
                        output_layout_override.set_memory_layout_from_str(attr["value"])
                    case "buffer_type":
                        output_layout_override.set_buffer_type_from_str(attr["value"])
                    case "tensor_memory_layout":
                        output_layout_override.set_tensor_memory_layout_from_str(
                            attr["value"]
                        )
                    case "grid_shape":
                        output_layout_override.grid = [
                            int(x) for x in attr["value"].strip("[]").split("x")
                        ]
                    case "dtype":
                        conv2d_config_override.set_dtype_from_str(attr["value"])
                    case "weights_dtype":
                        conv2d_config_override.set_weights_dtype_from_str(attr["value"])
                    case "activation":
                        conv2d_config_override.set_activation_from_str(
                            "none"
                            if attr["value"] == OVERRIDE_PARAMETER_DISABLED_STR
                            else attr["value"]
                        )
                    case "deallocate_activation":
                        conv2d_config_override.set_deallocate_activation_from_str(
                            attr["value"]
                        )
                    case "reallocate_halo_output":
                        conv2d_config_override.set_reallocate_halo_output_from_str(
                            attr["value"]
                        )
                    case "act_block_h_override":
                        conv2d_config_override.set_act_block_h_override_from_str(
                            attr["value"].strip("[]")
                        )
                    case "act_block_w_div":
                        conv2d_config_override.set_act_block_w_div_from_str(
                            attr["value"].strip("[]")
                        )
                    case "reshard_if_not_optimal":
                        conv2d_config_override.set_reshard_if_not_optimal_from_str(
                            attr["value"]
                        )
                    case "override_sharding_config":
                        conv2d_config_override.set_override_sharding_config_from_str(
                            attr["value"]
                        )
                    case "shard_layout":
                        if attr["value"] != OVERRIDE_PARAMETER_DISABLED_STR:
                            conv2d_config_override.set_shard_layout_from_str(
                                attr["value"]
                            )
                    case "core_grid":
                        conv2d_config_override.set_core_grid_from_str(attr["value"])
                    case "transpose_shards":
                        conv2d_config_override.set_transpose_shards_from_str(
                            attr["value"]
                        )
                    case "output_layout":
                        conv2d_config_override.set_output_layout_from_str(attr["value"])
                    case "preprocess_weights_on_device":
                        conv2d_config_override.set_preprocess_weights_on_device_from_str(
                            attr["value"]
                        )
                    case "always_preprocess_weights":
                        conv2d_config_override.set_always_preprocess_weights_from_str(
                            attr["value"]
                        )
                    case "enable_act_double_buffer":
                        conv2d_config_override.set_enable_act_double_buffer_from_str(
                            attr["value"]
                        )
                    case "enable_weights_double_buffer":
                        conv2d_config_override.set_enable_weights_double_buffer_from_str(
                            attr["value"]
                        )
                    case "enable_split_reader":
                        conv2d_config_override.set_enable_split_reader_from_str(
                            attr["value"]
                        )
                    case "enable_subblock_padding":
                        conv2d_config_override.set_enable_subblock_padding_from_str(
                            attr["value"]
                        )
                    case _:
                        raise ValueError(f"Invalid override attribute: {attr['key']}")
            if not output_layout_override.empty():
                override_handler.add_output_layout_override(
                    op_name_loc, output_layout_override
                )
            if not conv2d_config_override.empty():
                override_handler.add_conv2d_config_override(
                    op_name_loc, conv2d_config_override
                )
    print(override_handler.to_string())
    return override_handler.to_string()
    # return "enable-optimizer=true memory-layout-analysis-policy=DFSharding memreconfig-enabled=true override-output-layout=matmul_1=8x8:dram:interleaved:tile:f32 system-desc-path=ttrt-artifacts/system_desc.ttsys"


def build_mlir_module(
    fn: Callable,
    inputs_shapes: List[Shape],
    inputs_types: Optional[List[Union[torch.dtype, TypeInfo]]] = None,
    mesh_shape: Optional[Tuple[int, int]] = None,
    module_dump: bool = False,
    base: Optional[str] = None,
    output_root: str = ".",
):
    """
    Define a MLIR module specified as a python function.

    It will wrap `fn` in a MLIR FuncOp and then wrap that in a MLIR
    module, and finally tie arguments of that FuncOp to test function inputs. It will
    also pass a `TTIRBuilder` object as the last argument of test function.

    Arguments
    ---------
    fn : Callable
        Python function to be converted to MLIR

    inputs_shapes: List[Shape]
        Shapes of the respective ranked tensor inputs of the test function.

    module_dump: bool
        Set to True to print out generated MLIR module.

    golden_dump: bool
        Set to True to dump golden info to flatbuffer file.


    Returns
    -------
    MLIR module containing MLIR op graph defined by `fn`

    Example
    -------

    ```python
        def test_add(in0: Operand, in1: Operand, builder: TTIRBuilder):
            return builder.add(in0, in1)

        build_mlir_module(test_add, ((32, 32), (32, 32)))
    ```

    which returns

    ```
        #any = #ttcore.operand_constraint<...>
        module {
            func.func @test_add(
                %arg0: tensor<32x32xf32>,
                %arg1: tensor<32x32xf32>
            ) -> tensor<32x32xf32> {
                %0 = ttir.empty() : tensor<32x32xf32>
                %1 = "ttir.add"(%arg0, %arg1, %0) ...
                return %1 : tensor<32x32xf32>
            }
        }
    ```

    Check out:
    https://github.com/llvm/llvm-project/blob/main/mlir/test/python/dialects/tensor.py
    """

    ctx = Context()

    # Grab the location of the test function in python for later debugging
    try:
        fname = inspect.getfile(fn)
        line_no = inspect.getsourcelines(fn)[1]
        loc = Location.file(fname, line_no, 0, ctx)
    except (OSError, TypeError):
        loc = Location.unknown(ctx)

    # Instantiate builder which is passed as the last argument to
    # `fn` so the user can use it to build ops.
    builder = TTIRBuilder(ctx, loc)

    # deliver mesh_shape to TTIRBuilder
    # TTIR itself does not require mesh_shape information; however, it is needed to generate the golden tensor.
    if mesh_shape is not None:
        builder.set_mesh_shape(mesh_shape)

    # Default to all f32s
    if inputs_types is None:
        inputs_types = [torch.float32] * len(inputs_shapes)

    assert inputs_types is not None and len(inputs_shapes) == len(inputs_types)
    with ctx, loc:
        fn_input_types = [
            builder.ranked_tensor_type(
                shape,
                builder.get_type_from_torch_dtype(
                    dtype if isinstance(dtype, torch.dtype) else dtype
                ),
            )
            for (shape, dtype) in zip(inputs_shapes, inputs_types)
        ]

        # Wrap everything in a mlir module.
        module = Module.create()
        with InsertionPoint(module.body):
            # Wrap everything in a mlir function.
            @func.func(*fn_input_types, name=fn.__name__)
            def decorated_func(*inputs):
                # Randomly generate golden tensors for function inputs.
                input_goldens = []
                for index, (operand, dtype) in enumerate(zip(inputs, inputs_types)):
                    input_goldens.append(
                        builder.generate_input_golden(operand, dtype, index).tensor
                    )
                result = fn(*inputs, builder=builder)
                output_ops = result if hasattr(result, "__iter__") else (result,)
                output_goldens = [builder._get_golden_tensor(op) for op in output_ops]
                builder.set_graph_input_output(input_goldens, output_goldens)
                return result

        print(f"`{fn.__name__}` sucessfully transformed into a MLIR module.")

        base = fn.__name__ if base is None else base

        filename = get_target_path(output_root, base + "_ttir.mlir", "ttir")

        if module_dump:
            with open(filename, "w") as f:
                f.write(str(module))
                print(module)

        return module, builder


def run_pipeline(
    module,
    pipeline_fn: Callable = ttir_to_ttnn_backend_pipeline,
    pipeline_options: List[str] = None,
    dump_to_file: bool = True,
    output_file_name: str = "test.mlir",
    system_desc_path: Optional[str] = None,
    mesh_shape: Optional[Tuple[int, int]] = None,
    argument_types_string: Optional[str] = None,
    settings: Optional[Dict[str, str]] = None,
):
    """
    Runs a pipeline over a module and optionally dumps to file.

    Arguments
    ---------
    pipeline_fn: Callable
        Pipeline function to run. pipeline_fn(module, options)

    dump_to_file: bool
        Flag which indicates that generated TTNN module will be dumped to file.

    output_file_name: str
        Name of the output file.

    Returns
    -------
    MLIR module containing MLIR op graph defined by `module` and pipeline_fn.
    """

    if pipeline_options is None:
        pipeline_options = []

    if argument_types_string:
        tt_populate_argument_types(module, argument_types_string)

    # Default to the `SYSTEM_DESC_PATH` envvar
    if system_desc_path is None:
        system_desc_path = os.getenv("SYSTEM_DESC_PATH", "")
    print(settings)
    print(pipeline_options)
    # Generate option string
    # if system_desc_path:
    #    pipeline_options.append(f"system-desc-path={system_desc_path}")
    if mesh_shape and len(mesh_shape) == 2:
        pipeline_options.append(f"mesh-shape={mesh_shape[0]},{mesh_shape[1]}")
    if argument_types_string:
        pipeline_options.append("enable-const-eval=true")
    if settings:
        overrides = settings_to_overrides(settings, os.path.dirname(output_file_name))
        pipeline_options.append(overrides)
    print(f"Pipeline options: {pipeline_options}")
    print(pipeline_fn)
    print(pipeline_fn.__name__)
    print(" ".join(pipeline_options))
    # assert False, "ttt"
    # // RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% enable-optimizer=true memory-layout-analysis-enabled=true memreconfig-enabled=true insert-memreconfig=relu=0 override-output-layout=relu=tile row-major-enabled=true" -o shard_transpose.mlir %s
    # "enable-optimizer=true memory-layout-analysis-enabled=true memreconfig-enabled=true insert-memreconfig=relu=0 override-output-layout=relu=tile row-major-enabled=true system-desc-path=ttrt-artifacts/system_desc.ttsys")
    # Now, pass it through the pipeline. Module gets modified in place.
    pipeline_fn(
        module,
        # "memreconfig-enabled=true system-desc-path=/home/jgrim/wh-01-src/tt-mlir/ttrt-artifacts/system_desc.ttsys",
        # " ".join(pipeline_options),
        # "enable-optimizer=true"
        "enable-optimizer=true memory-layout-analysis-policy=DFSharding memreconfig-enabled=true system-desc-path=/home/jgrim/wh-01-src/tt-mlir/ttrt-artifacts/system_desc.ttsys",
    )

    # Optionally dump to file.
    if dump_to_file:
        with open(output_file_name, "w") as f:
            f.write(str(module))

    return module


def compile_to_flatbuffer(
    fn: Callable,
    inputs_shapes: List[Shape],
    inputs_types: Optional[List[Union[torch.dtype, TypeInfo]]] = None,
    system_desc_path: str = "ttrt-artifacts/system_desc.ttsys",
    test_base: str = "test",
    output_root: str = ".",
    target: Literal["ttnn", "ttmetal"] = "ttnn",
    mesh_shape: Optional[Tuple[int, int]] = None,
    module_dump: bool = True,
    argument_types_string: Optional[str] = None,
    custom_pipeline: Optional[Union[Callable, str]] = None,
    pipeline_options: Optional[List[str]] = None,
    settings: Optional[Dict[str, str]] = None,
    print_ir: Union[bool, str] = False,
):
    """
    Compiles a TTIRBuilder function `fn` to TTIR MLIR -> TT{Metal,NN} MLIR -> Flatbuffer

    This decorator is mainly a wrapper around the following functions, with
    each next function called on the output of the last:

    1. `build_mlir_module`
    2. `run_pipeline`
    3. `to_flatbuffer`

    The choice of TTNN vs. TTMetal is controlled by the `target` parameter

    Arguments
    ---------

    fn: Callable
        The TTIRBuilder function to compile. Must take `builder : TTIRBuilder` as a kwarg

    inputs_shapes: List[Shape]
        Shapes of the respective ranked tensor inputs of the test function.

    inputs_types: Optional[List[torch.dtype]]
        The dtypes to use for the inputs to `fn`. Note that if supplied,
        `len(inputs_shapes) == len(inputs_types)` must be true. Defaults to
        `None`

    test_base: str
        The string to be used as the base name for dumped files throughout the
        process. If `None` is provided, then the `__name__` of `fn` will be used.

    output_root: str
        The path to dump all generated arguments under. If this path doesn't
        exist, it will be created

    target: str
        Either `"ttnn"` or `"ttmetal"`. This controls which backend to use

    custom_pipeline: Union[Callable, str]
        Pipeline function to run.
        Either a Callable:
            custom_pipeline(module, options)
        Or a str:
            "ttir-lower-to-layout,ttir-bufferization-pipeline"

    mesh_shape: Optional[Tuple[int, int]]
        A list that contains shape of the mesh to be applied on ttir to ttnn
        conversion path. Defaults to `None`

    module_dump: bool
        Set to `True` to print out generated TTIR MLIR module.

    print_ir: Union[bool, str]
        Set to `True` to print IR to stdout.  Set to dir path to print IR after
        each pass to its own file under _this_ directory.
    """
    print("5")
    if inputs_types is not None:
        assert len(inputs_shapes) == len(inputs_types)

    if type(custom_pipeline) is str:
        custom_pipeline = create_custom_pipeline_fn(custom_pipeline, print_ir=print_ir)

    if pipeline_options is None:
        pipeline_options = []

    if settings is None:
        settings = {}
    print("6")
    pipeline_fn: Callable
    to_flatbuffer: Callable
    mlir_suffix: str
    target_extension: str

    if target == "ttnn":
        pipeline_fn = (
            custom_pipeline if custom_pipeline else ttir_to_ttnn_backend_pipeline
        )
        to_flatbuffer = ttnn_to_flatbuffer_file
        mlir_suffix = "_ttnn.mlir"
        target_extension = "ttnn"
    elif target == "ttmetal":
        pipeline_fn = (
            custom_pipeline if custom_pipeline else ttir_to_ttmetal_backend_pipeline
        )
        to_flatbuffer = ttmetal_to_flatbuffer_file
        mlir_suffix = "_ttm.mlir"
        target_extension = "ttm"
    else:
        raise ValueError("Unsupported target: " + target)
    print("7")
    # Compile model to TTIR MLIR
    module, builder = build_mlir_module(
        fn,
        inputs_shapes,
        inputs_types,
        mesh_shape=mesh_shape,
        module_dump=module_dump,
        output_root=output_root,
    )
    print("8")
    output_file_mlir = get_target_path(output_root, test_base + mlir_suffix, target)
    output_file_fbb = ".".join([output_file_mlir, target_extension])

    # Compile TTIR MLIR -> TT{Metal,NN} MLIR
    module = run_pipeline(
        module,
        pipeline_fn,
        pipeline_options=pipeline_options,
        dump_to_file=module_dump,
        output_file_name=output_file_mlir,
        system_desc_path=system_desc_path,
        mesh_shape=mesh_shape,
        argument_types_string=argument_types_string,
        settings=settings,
    )
    print(f"{target} pipeline ran successfully.")
    print("9")
    module_logger = MLIRModuleLogger()
    module_logger.attach_context(module.context)

    # Compile TT{Metal,NN} MLIR -> flatbuffer
    to_flatbuffer(
        module,
        output_file_fbb,
        builder.get_golden_map(),
        module_logger.module_log if module_logger.module_log else [],
    )
    print(f"{target} flatbuffer created successfully.")
