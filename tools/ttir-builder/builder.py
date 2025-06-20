# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import List, Optional, Union, Tuple, Callable, Dict, Any
from ttmlir.ir import *
from ttmlir.dialects import ttir, ttcore, tensor, quant
from ttmlir.passes import GoldenTensor, DataType
import torch
from enum import Enum, auto
import re
from .ccl_golden import *
from .ops import *
from .apis import *
from sphinx.ext.autodoc import FunctionDocumenter

# Alias for operands of ops which can be either BlockArguments, Values, or other
# ops wrapped in OpView or Operation.
Operand = Union[Value, OpView, Operation]

# Convenience alias for shape
Shape = Union[List[int], Tuple[int, ...]]


class TTIRBuilder(TTIRBuilderOps, TTIRBuilderAPIs):
    def __init__(self, ctx: Context, location: Location):
        self._ctx = ctx
        self._loc = location

        self._seed = 0
        # Dictionary to store Golden for each Operand we encounter in MLIR
        # graph.
        self._goldens: Dict[Operand, Golden] = {}

        # global ID of operations
        self._global_id = -1

        # id to golden map
        self.id_golden_map = {}

        # mesh_shape for multi-device
        self.mesh_shape = ()

        # golden check level
        self._golden_check_level = GoldenCheckLevel.OP_LEVEL
