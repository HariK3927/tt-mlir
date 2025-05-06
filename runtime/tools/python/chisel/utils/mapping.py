# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# DISCLAIMER: this file will be removed very soon and is a temporary solution

import ttmlir
import torch


ttir_dtype_maps = {
    "i32": torch.int32,
    "i64": torch.int64,
    "f32": torch.float32,
    "f64": torch.float64,
    "i1": torch.bool,
    "bf16": torch.bfloat16,
    "f16": torch.float16,
}


def resolve_dense_attr(dense_attr):
    if dense_attr.is_splat:
        value = dense_attr.get_splat_value()
        return value
    try:
        values = [dense_attr[i] for i in range(len(dense_attr))]
        return values
    except Exception as e:
        print(f"Indexing error: {e}")

    try:
        shape = dense_attr.type.shape
        if len(shape) == 2:
            values = [
                [dense_attr.get(i, j) for j in range(shape[1])] for i in range(shape[0])
            ]
        elif len(shape) == 1:
            values = [dense_attr.get(i) for i in range(shape[0])]
        return values
    except Exception as e:
        print(f"Get method error: {e}")


class OpMapping:
    def __init__(self, torch_op, arg_map=None, unpack_inputs=True):
        self.torch_op = torch_op
        self.arg_map = arg_map or {}
        self.unpack_inputs = unpack_inputs

    def __call__(self, ir_op, inputs):
        if isinstance(inputs, list) and len(inputs) > 1:
            result_inputs = inputs
        else:
            result_inputs = (
                inputs[0] if isinstance(inputs, list) and len(inputs) == 1 else inputs
            )

        torch_args = {}
        for k, v in self.arg_map.items():
            if k in ir_op.attributes:
                if isinstance(ir_op.attributes[k], ttmlir.ir.DenseElementsAttr):
                    torch_args[v] = resolve_dense_attr(ir_op.attributes[k]).value
                elif isinstance(ir_op.attributes[k], ttmlir.ir.DenseI64ArrayAttr):
                    torch_args[v] = [x for x in ir_op.attributes[k]]
                elif isinstance(ir_op.attributes[k], ttmlir.ir.ArrayAttr):
                    torch_args[v] = [x.value for x in ir_op.attributes[k]]
                else:
                    torch_args[v] = ir_op.attributes[k].value
            if v == "dim":
                torch_args[v] = torch_args[v][0]

        if ir_op.name == "ttir.constant":
            torch_args["dtype"] = ttir_dtype_maps[
                str(ir_op.attributes[0].attr.type.element_type)
            ]

        if not self.unpack_inputs:
            print("torch op", self.torch_op, result_inputs, torch_args)
            result = self.torch_op(result_inputs, **torch_args)
            if result is not None:
                print("torch op result", result.shape, result)
            return result

        result = self.torch_op(*result_inputs, **torch_args)
        print("torch op unpack", self.torch_op, result_inputs, torch_args)
        if result is not None:
            print("torch op result", result.shape, result)
        return result


def custom_broadcast(x, size=None):
    for i in range(len(size)):
        try:
            if size[i] < x.shape[i]:
                size[i] = x.shape[i]
        except Exception as e:
            import pdb

            pdb.set_trace()
            print(f"Broadcasting error: {e}")
    return x.expand(size)


def custom_where(a, b, c):
    a = a.to(torch.bool)
    return torch.where(a, b, c)


def custom_typecast(x, dtype=None):
    if dtype is None:
        return x
    return x.to(dtype)


def custom_constant(*args, **kwargs):
    data = kwargs["data"]
    kwargs.pop("data")
    return torch.tensor([data], *args, **kwargs)


ttir_to_torch_mapping = {
    # do nothing
    "ttir.empty": OpMapping(lambda x=None, *args, **kwargs: None),
    "func.return": OpMapping(lambda x=None, *args, **kwargs: x),
    "ttir.add": OpMapping(torch.add),
    "ttir.arange": OpMapping(
        torch.arange, {"start": "start", "end": "end", "step": "step"}
    ),
    "ttir.broadcast": OpMapping(
        custom_broadcast, {"broadcast_dimensions": "size"}, unpack_inputs=False
    ),
    "ttir.constant": OpMapping(custom_constant, {"value": "data"}),
    "ttir.div": OpMapping(torch.div),
    "ttir.exp": OpMapping(torch.exp, unpack_inputs=False),
    "ttir.ge": OpMapping(torch.ge),
    "ttir.gt": OpMapping(torch.gt),
    "ttir.le": OpMapping(torch.le),
    "ttir.logical_and": OpMapping(torch.logical_and),
    "ttir.lt": OpMapping(torch.lt),
    "ttir.matmul": OpMapping(torch.matmul),
    "ttir.max": OpMapping(torch.max),
    "ttir.maximum": OpMapping(torch.maximum),
    "ttir.multiply": OpMapping(torch.multiply),
    "ttir.permute": OpMapping(
        torch.permute, {"permutation": "dims"}, unpack_inputs=False
    ),
    "ttir.prod": OpMapping(
        torch.prod, {"dim_arg": "dim", "keep_dim": "keepdim"}, unpack_inputs=False
    ),
    "ttir.reshape": OpMapping(torch.reshape, {"shape": "shape"}, unpack_inputs=False),
    "ttir.rsqrt": OpMapping(torch.rsqrt, unpack_inputs=False),
    "ttir.sqrt": OpMapping(torch.sqrt, unpack_inputs=False),
    "ttir.subtract": OpMapping(torch.subtract),
    "ttir.sum": OpMapping(
        torch.sum, {"dim_arg": "dim", "keep_dim": "keepdim"}, unpack_inputs=False
    ),
    "ttir.tanh": OpMapping(torch.tanh, unpack_inputs=False),
    "ttir.typecast": OpMapping(custom_typecast, {"dtype": "dtype"}),
    "ttir.where": OpMapping(custom_where),
    "ttir.concat": OpMapping(torch.concat, {"dim": "dim"}, unpack_inputs=False),
    "ttir.embedding": OpMapping(torch.nn.functional.embedding),
}
